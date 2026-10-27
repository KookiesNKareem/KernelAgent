```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernels
# ---------------------------------------------------------------------------

@triton.jit
def _wy_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    w_ptr,
    u_ptr,
    a_ptr,
    T,
    K,
    V,
    NT,
    CHUNK: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BVF: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NT
    chunk = pid % NT

    offs_c = tl.arange(0, CHUNK)
    offs_s = tl.arange(0, BS)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BVF)
    k_mask = offs_k < K
    v_mask = offs_v < V

    offs_t_all = chunk * CHUNK + offs_c
    t_mask_all = offs_t_all < T

    g_full = tl.load(
        g_ptr + bh * T + offs_t_all, mask=t_mask_all, other=0.0
    ).to(tl.float32)
    cumg_full = tl.cumsum(g_full, axis=0)

    # Blocked forward solve of (I + L) X = RHS for both right-hand sides
    # (W from beta*exp(cumg)*k, Uhat from beta*v), in BS-row segments.
    k_prev = tl.zeros([BS, BK], dtype=tl.bfloat16)
    xw_prev = tl.zeros([BS, BK], dtype=tl.float32)
    xu_prev = tl.zeros([BS, BVF], dtype=tl.float32)
    cumg_prev = tl.zeros([BS], dtype=tl.float32)

    for seg in range(CHUNK // BS):
        seg_idx = seg * BS + offs_s
        seg_t = chunk * CHUNK + seg_idx
        seg_mask = seg_t < T

        k_i = tl.load(
            k_ptr + bh * T * K + seg_t[:, None] * K + offs_k[None, :],
            mask=seg_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        v_i = tl.load(
            v_ptr + bh * T * V + seg_t[:, None] * V + offs_v[None, :],
            mask=seg_mask[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        beta_i = tl.load(
            beta_ptr + bh * T + seg_t, mask=seg_mask, other=0.0
        ).to(tl.float32)
        cumg_i = tl.sum(
            tl.where(
                offs_c[:, None] == seg_idx[None, :],
                cumg_full[:, None],
                0.0,
            ),
            axis=0,
        )

        rhs_w = beta_i[:, None] * tl.exp(cumg_i)[:, None] * k_i.to(tl.float32)
        rhs_u = beta_i[:, None] * v_i.to(tl.float32)

        if seg > 0:
            l_cross = (
                tl.dot(k_i, tl.trans(k_prev)).to(tl.float32)
                * tl.exp(cumg_i[:, None] - cumg_prev[None, :])
                * beta_i[:, None]
            ).to(tl.bfloat16)
            rhs_w = rhs_w - tl.dot(l_cross, xw_prev.to(tl.bfloat16)).to(
                tl.float32
            )
            rhs_u = rhs_u - tl.dot(l_cross, xu_prev.to(tl.bfloat16)).to(
                tl.float32
            )

        gram = tl.dot(k_i, tl.trans(k_i)).to(tl.float32)
        diff = cumg_i[:, None] - cumg_i[None, :]
        lower = offs_s[:, None] > offs_s[None, :]
        exp_diff = tl.where(lower, tl.exp(tl.where(lower, diff, 0.0)), 0.0)
        l_ii = tl.where(lower, gram * exp_diff * beta_i[:, None], 0.0)

        inv_b = (offs_s[:, None] == offs_s[None, :]).to(tl.float32)
        for r in range(BS):
            row_r = tl.sum(
                tl.where(offs_s[:, None] == r, inv_b, 0.0), axis=0
            )
            col_r = tl.sum(
                tl.where(offs_s[None, :] == r, l_ii, 0.0), axis=1
            )
            inv_b = inv_b - col_r[:, None] * row_r[None, :]

        inv_bf = inv_b.to(tl.bfloat16)
        x_w = tl.dot(inv_bf, rhs_w.to(tl.bfloat16)).to(tl.float32)
        x_u = tl.dot(inv_bf, rhs_u.to(tl.bfloat16)).to(tl.float32)

        tl.store(
            w_ptr + bh * T * K + seg_t[:, None] * K + offs_k[None, :],
            x_w.to(tl.bfloat16),
            mask=seg_mask[:, None] & k_mask[None, :],
        )
        tl.store(
            u_ptr + bh * T * V + seg_t[:, None] * V + offs_v[None, :],
            x_u.to(tl.bfloat16),
            mask=seg_mask[:, None] & v_mask[None, :],
        )

        k_prev = k_i
        xw_prev = x_w
        xu_prev = x_u
        cumg_prev = cumg_i

    # Decay-weighted intra-chunk q-k matrix for the output kernel.
    q_full = tl.load(
        q_ptr + bh * T * K + offs_t_all[:, None] * K + offs_k[None, :],
        mask=t_mask_all[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    k_full = tl.load(
        k_ptr + bh * T * K + offs_t_all[:, None] * K + offs_k[None, :],
        mask=t_mask_all[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    qk = tl.dot(q_full, tl.trans(k_full)).to(tl.float32)
    diff_all = cumg_full[:, None] - cumg_full[None, :]
    causal = offs_c[None, :] <= offs_c[:, None]
    exp_causal = tl.where(
        causal, tl.exp(tl.where(causal, diff_all, 0.0)), 0.0
    )
    a_mat = tl.where(causal, qk * exp_causal, 0.0)

    a_ptrs = (
        a_ptr
        + pid * CHUNK * CHUNK
        + offs_c[:, None] * CHUNK
        + offs_c[None, :]
    )
    tl.store(a_ptrs, a_mat.to(tl.bfloat16))


@triton.jit
def _state_kernel(
    k_ptr,
    g_ptr,
    w_ptr,
    u_ptr,
    h_ptr,
    T,
    K,
    V,
    NT,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    v_cols = pid_v * BV + offs_v
    v_mask = v_cols < V
    k_mask = offs_k < K

    state = tl.zeros([BK, BV], dtype=tl.float32)

    for chunk in range(NT):
        offs_t = chunk * CHUNK + offs_c
        t_mask = offs_t < T

        g_vals = tl.load(
            g_ptr + bh * T + offs_t, mask=t_mask, other=0.0
        ).to(tl.float32)
        cumg = tl.cumsum(g_vals, axis=0)

        w_vals = tl.load(
            w_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        u_hat = tl.load(
            u_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
            mask=t_mask[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        state_bf = state.to(tl.bfloat16)
        u_eff = u_hat - tl.dot(w_vals, state_bf).to(tl.float32)

        h_ptrs = (
            h_ptr
            + ((bh * NT + chunk) * K + offs_k[:, None]) * V
            + v_cols[None, :]
        )
        tl.store(
            h_ptrs, state_bf, mask=k_mask[:, None] & v_mask[None, :]
        )
        tl.store(
            u_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
            u_eff.to(tl.bfloat16),
            mask=t_mask[:, None] & v_mask[None, :],
        )

        k_vals = tl.load(
            k_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        last_cum = tl.sum(tl.where(offs_c == CHUNK - 1, cumg, 0.0))
        k_scaled = (
            k_vals.to(tl.float32) * tl.exp(last_cum - cumg)[:, None]
        ).to(tl.bfloat16)
        state = state * tl.exp(last_cum) + tl.dot(
            tl.trans(k_scaled), u_eff.to(tl.bfloat16)
        ).to(tl.float32)


@triton.jit
def _out_kernel(
    q_ptr,
    g_ptr,
    a_ptr,
    u_ptr,
    h_ptr,
    out_ptr,
    T,
    K,
    V,
    NT,
    scale,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_v = tl.program_id(1)
    bh = pid // NT
    chunk = pid % NT

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    v_cols = pid_v * BV + offs_v
    v_mask = v_cols < V
    k_mask = offs_k < K

    offs_t = chunk * CHUNK + offs_c
    t_mask = offs_t < T

    g_vals = tl.load(
        g_ptr + bh * T + offs_t, mask=t_mask, other=0.0
    ).to(tl.float32)
    exp_cum = tl.exp(tl.cumsum(g_vals, axis=0))

    q_vals = tl.load(
        q_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
        mask=t_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    h_vals = tl.load(
        h_ptr + (pid * K + offs_k[:, None]) * V + v_cols[None, :],
        mask=k_mask[:, None] & v_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    a_vals = tl.load(
        a_ptr + pid * CHUNK * CHUNK + offs_c[:, None] * CHUNK + offs_c[None, :]
    ).to(tl.bfloat16)
    u_vals = tl.load(
        u_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
        mask=t_mask[:, None] & v_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    q_state = tl.dot(q_vals, h_vals).to(tl.float32)
    intra = tl.dot(a_vals, u_vals).to(tl.float32)

    out_vals = scale * (exp_cum[:, None] * q_state + intra)
    tl.store(
        out_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
        out_vals,
        mask=t_mask[:, None] & v_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Backward kernels
#
# The previous _bwd_chunk_kernel (45 ms) combined, in one program, a
# 64-step serial triangular inversion, a runtime-length loop over V, and
# four live [CHUNK, BK] fp32 accumulators plus [K, BV] state tiles: it
# spilled registers catastrophically.  The backward is now split so no
# kernel holds more than a couple of large tiles at once:
#   1. _bwd_state_kernel2: reverse state-adjoint sweep (unchanged, 1 ms).
#   2. _bwd_inv_kernel:    blocked (2x2) inverse of I + L per chunk.
#   3. _bwd_vblock_kernel: per (chunk, V-block): transposed solve, dv,
#      and per-V-block partials of every V-reduction, stored immediately.
#   4. _bwd_final_kernel:  per chunk: sums the few partials and finishes
#      dq / dk / dg / dbeta with only [CHUNK, CHUNK]/[CHUNK, BK] tiles.
# ---------------------------------------------------------------------------

@triton.jit
def _bwd_state_kernel2(
    q_ptr,
    k_ptr,
    g_ptr,
    w_ptr,
    h_ptr,
    a_ptr,
    go_ptr,
    du_ptr,
    ds_ptr,
    T,
    K,
    V,
    NT,
    scale,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    v_cols = pid_v * BV + offs_v
    v_mask = v_cols < V
    k_mask = offs_k < K

    dstate = tl.zeros([BK, BV], dtype=tl.float32)

    for rev in range(NT):
        chunk = NT - 1 - rev
        pid_c = bh * NT + chunk
        offs_t = chunk * CHUNK + offs_c
        t_mask = offs_t < T

        ds_ptrs = (
            ds_ptr + (pid_c * K + offs_k[:, None]) * V + v_cols[None, :]
        )
        tl.store(
            ds_ptrs, dstate, mask=k_mask[:, None] & v_mask[None, :]
        )

        g_vals = tl.load(
            g_ptr + bh * T + offs_t, mask=t_mask, other=0.0
        ).to(tl.float32)
        cumg = tl.cumsum(g_vals, axis=0)
        exp_cum = tl.exp(cumg)
        last_cum = tl.sum(tl.where(offs_c == CHUNK - 1, cumg, 0.0))

        go_s = (
            tl.load(
                go_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
                mask=t_mask[:, None] & v_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            * scale
        )
        a_vals = tl.load(
            a_ptr
            + pid_c * CHUNK * CHUNK
            + offs_c[:, None] * CHUNK
            + offs_c[None, :]
        ).to(tl.bfloat16)
        k_vals = tl.load(
            k_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        w_vals = tl.load(
            w_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        q_vals = tl.load(
            q_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        k_scaled = (
            k_vals.to(tl.float32) * tl.exp(last_cum - cumg)[:, None]
        ).to(tl.bfloat16)
        du_vals = tl.dot(
            tl.trans(a_vals), go_s.to(tl.bfloat16)
        ).to(tl.float32) + tl.dot(
            k_scaled, dstate.to(tl.bfloat16)
        ).to(tl.float32)

        tl.store(
            du_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
            du_vals,
            mask=t_mask[:, None] & v_mask[None, :],
        )

        go_e = (go_s * exp_cum[:, None]).to(tl.bfloat16)
        dstate = (
            tl.dot(tl.trans(q_vals), go_e).to(tl.float32)
            + dstate * tl.exp(last_cum)
            - tl.dot(tl.trans(w_vals), du_vals.to(tl.bfloat16)).to(
                tl.float32
            )
        )


@triton.jit
def _bwd_inv_kernel(
    k_ptr,
    g_ptr,
    beta_ptr,
    tinv_ptr,
    T,
    K,
    NT,
    CHUNK: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
):
    # Blocked inverse of Tmat = I + L via its 2x2 (BS = CHUNK/2) split:
    # inv = [[A, 0], [-C L21 A, C]], A = T11^-1, C = T22^-1.  Only small
    # BS x BS tiles go through forward substitution; the rest is dots.
    pid = tl.program_id(0)
    bh = pid // NT
    chunk = pid % NT

    offs_s = tl.arange(0, BS)
    offs_k = tl.arange(0, BK)
    k_mask = offs_k < K
    lower = offs_s[:, None] > offs_s[None, :]

    t0 = chunk * CHUNK + offs_s
    t1 = t0 + BS
    m0 = t0 < T
    m1 = t1 < T

    k0 = tl.load(
        k_ptr + bh * T * K + t0[:, None] * K + offs_k[None, :],
        mask=m0[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    k1 = tl.load(
        k_ptr + bh * T * K + t1[:, None] * K + offs_k[None, :],
        mask=m1[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    g0 = tl.load(g_ptr + bh * T + t0, mask=m0, other=0.0).to(tl.float32)
    g1 = tl.load(g_ptr + bh * T + t1, mask=m1, other=0.0).to(tl.float32)
    b0 = tl.load(beta_ptr + bh * T + t0, mask=m0, other=0.0).to(tl.float32)
    b1 = tl.load(beta_ptr + bh * T + t1, mask=m1, other=0.0).to(tl.float32)

    cum0 = tl.cumsum(g0, axis=0)
    cum1 = tl.sum(g0) + tl.cumsum(g1, axis=0)

    d00 = cum0[:, None] - cum0[None, :]
    l00 = tl.where(
        lower,
        tl.dot(k0, tl.trans(k0)).to(tl.float32)
        * tl.where(lower, tl.exp(tl.where(lower, d00, 0.0)), 0.0)
        * b0[:, None],
        0.0,
    )
    d11 = cum1[:, None] - cum1[None, :]
    l11 = tl.where(
        lower,
        tl.dot(k1, tl.trans(k1)).to(tl.float32)
        * tl.where(lower, tl.exp(tl.where(lower, d11, 0.0)), 0.0)
        * b1[:, None],
        0.0,
    )
    l10 = (
        tl.dot(k1, tl.trans(k0)).to(tl.float32)
        * tl.exp(cum1[:, None] - cum0[None, :])
        * b1[:, None]
    )

    inv0 = (offs_s[:, None] == offs_s[None, :]).to(tl.float32)
    for r in range(BS):
        row_r = tl.sum(tl.where(offs_s[:, None] == r, inv0, 0.0), axis=0)
        col_r = tl.sum(tl.where(offs_s[None, :] == r, l00, 0.0), axis=1)
        inv0 = inv0 - col_r[:, None] * row_r[None, :]
    inv1 = (offs_s[:, None] == offs_s[None, :]).to(tl.float32)
    for r in range(BS):
        row_r = tl.sum(tl.where(offs_s[:, None] == r, inv1, 0.0), axis=0)
        col_r = tl.sum(tl.where(offs_s[None, :] == r, l11, 0.0), axis=1)
        inv1 = inv1 - col_r[:, None] * row_r[None, :]

    inv10 = -tl.dot(
        tl.dot(inv1.to(tl.bfloat16), l10.to(tl.bfloat16)).to(tl.bfloat16),
        inv0.to(tl.bfloat16),
    ).to(tl.float32)

    base = tinv_ptr + pid * CHUNK * CHUNK
    tl.store(base + offs_s[:, None] * CHUNK + offs_s[None, :], inv0)
    tl.store(
        base + offs_s[:, None] * CHUNK + (BS + offs_s)[None, :],
        tl.zeros([BS, BS], dtype=tl.float32),
    )
    tl.store(
        base + (BS + offs_s)[:, None] * CHUNK + offs_s[None, :], inv10
    )
    tl.store(
        base + (BS + offs_s)[:, None] * CHUNK + (BS + offs_s)[None, :],
        inv1,
    )


@triton.jit
def _bwd_vblock_kernel(
    q_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    w_ptr,
    u_ptr,
    h_ptr,
    go_ptr,
    du_ptr,
    ds_ptr,
    tinv_ptr,
    dv_ptr,
    p_da_ptr,
    p_dt_ptr,
    p_dw_ptr,
    p_dks_ptr,
    p_dq_ptr,
    p_de_ptr,
    p_dbeta_ptr,
    p_ss_ptr,
    T,
    K,
    V,
    NT,
    scale,
    NVB: tl.constexpr,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_v = tl.program_id(1)
    bh = pid // NT
    chunk = pid % NT
    part = pid * NVB + pid_v

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    v_cols = pid_v * BV + offs_v
    v_mask = v_cols < V
    k_mask = offs_k < K
    offs_t = chunk * CHUNK + offs_c
    t_mask = offs_t < T

    g_vals = tl.load(
        g_ptr + bh * T + offs_t, mask=t_mask, other=0.0
    ).to(tl.float32)
    beta_vals = tl.load(
        beta_ptr + bh * T + offs_t, mask=t_mask, other=0.0
    ).to(tl.float32)
    exp_cum = tl.exp(tl.cumsum(g_vals, axis=0))

    go_s = (
        tl.load(
            go_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
            mask=t_mask[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        * scale
    )
    du_vals = tl.load(
        du_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
        mask=t_mask[:, None] & v_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    u_vals = tl.load(
        u_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
        mask=t_mask[:, None] & v_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    s0_blk = tl.load(
        h_ptr + (pid * K + offs_k[:, None]) * V + v_cols[None, :],
        mask=k_mask[:, None] & v_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    ds_blk = tl.load(
        ds_ptr + (pid * K + offs_k[:, None]) * V + v_cols[None, :],
        mask=k_mask[:, None] & v_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Transposed triangular solve: Y_u = Tmat^-T Du; dv = beta * Y_u.
    tinv = tl.load(
        tinv_ptr
        + pid * CHUNK * CHUNK
        + offs_c[:, None] * CHUNK
        + offs_c[None, :]
    ).to(tl.bfloat16)
    y_u = tl.dot(tl.trans(tinv), du_vals.to(tl.bfloat16)).to(tl.float32)
    tl.store(
        dv_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
        y_u * beta_vals[:, None],
        mask=t_mask[:, None] & v_mask[None, :],
    )

    v_vals = tl.load(
        v_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
        mask=t_mask[:, None] & v_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        p_dbeta_ptr + part * CHUNK + offs_c,
        tl.sum(y_u * v_vals, axis=1),
    )

    # Uhat = u + W S0, then the [CHUNK, CHUNK] partial Y_u @ Uhat^T.
    w_vals = tl.load(
        w_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
        mask=t_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    uhat = u_vals.to(tl.float32) + tl.dot(w_vals, s0_blk).to(tl.float32)
    tl.store(
        p_dt_ptr
        + part * CHUNK * CHUNK
        + offs_c[:, None] * CHUNK
        + offs_c[None, :],
        tl.dot(y_u.to(tl.bfloat16), tl.trans(uhat.to(tl.bfloat16))).to(
            tl.float32
        ),
    )

    # Remaining V-reduction partials; each is stored as soon as it is
    # computed so large tiles do not stay live simultaneously.
    tl.store(
        p_da_ptr
        + part * CHUNK * CHUNK
        + offs_c[:, None] * CHUNK
        + offs_c[None, :],
        tl.dot(go_s.to(tl.bfloat16), tl.trans(u_vals)).to(tl.float32),
    )
    tl.store(
        p_dw_ptr + part * CHUNK * K + offs_c[:, None] * K + offs_k[None, :],
        tl.dot(du_vals.to(tl.bfloat16), tl.trans(s0_blk)).to(tl.float32),
        mask=k_mask[None, :],
    )
    tl.store(
        p_dks_ptr + part * CHUNK * K + offs_c[:, None] * K + offs_k[None, :],
        tl.dot(u_vals, tl.trans(ds_blk.to(tl.bfloat16))).to(tl.float32),
        mask=k_mask[None, :],
    )
    tl.store(
        p_dq_ptr + part * CHUNK * K + offs_c[:, None] * K + offs_k[None, :],
        tl.dot(
            (go_s * exp_cum[:, None]).to(tl.bfloat16), tl.trans(s0_blk)
        ).to(tl.float32),
        mask=k_mask[None, :],
    )

    q_vals = tl.load(
        q_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
        mask=t_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    q_s0 = tl.dot(q_vals, s0_blk).to(tl.float32)
    tl.store(
        p_de_ptr + part * CHUNK + offs_c, tl.sum(go_s * q_s0, axis=1)
    )
    tl.store(
        p_ss_ptr + part, tl.sum(ds_blk * s0_blk.to(tl.float32))
    )


@triton.jit
def _bwd_final_kernel(
    q_ptr,
    k_ptr,
    g_ptr,
    beta_ptr,
    w_ptr,
    tinv_ptr,
    p_da_ptr,
    p_dt_ptr,
    p_dw_ptr,
    p_dks_ptr,
    p_dq_ptr,
    p_de_ptr,
    p_dbeta_ptr,
    p_ss_ptr,
    dq_ptr,
    dk_ptr,
    dg_ptr,
    dbeta_ptr,
    T,
    K,
    V,
    NT,
    NVB: tl.constexpr,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NT
    chunk = pid % NT

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    k_mask = offs_k < K
    offs_t = chunk * CHUNK + offs_c
    t_mask = offs_t < T

    g_vals = tl.load(
        g_ptr + bh * T + offs_t, mask=t_mask, other=0.0
    ).to(tl.float32)
    beta_vals = tl.load(
        beta_ptr + bh * T + offs_t, mask=t_mask, other=0.0
    ).to(tl.float32)
    cumg = tl.cumsum(g_vals, axis=0)
    exp_cum = tl.exp(cumg)
    last_cum = tl.sum(tl.where(offs_c == CHUNK - 1, cumg, 0.0))
    exp_last = tl.exp(last_cum)

    diff = cumg[:, None] - cumg[None, :]
    lower = offs_c[:, None] > offs_c[None, :]
    causal = offs_c[None, :] <= offs_c[:, None]
    exp_lower = tl.where(lower, tl.exp(tl.where(lower, diff, 0.0)), 0.0)
    exp_causal = tl.where(
        causal, tl.exp(tl.where(causal, diff, 0.0)), 0.0
    )

    q_vals = tl.load(
        q_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
        mask=t_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    k_vals = tl.load(
        k_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
        mask=t_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    k_fp = k_vals.to(tl.float32)

    # dq starts from the aggregated state contribution.
    dq_acc = tl.zeros([CHUNK, BK], dtype=tl.float32)
    for vb in range(NVB):
        dq_acc += tl.load(
            p_dq_ptr
            + (pid * NVB + vb) * CHUNK * K
            + offs_c[:, None] * K
            + offs_k[None, :],
            mask=k_mask[None, :],
            other=0.0,
        )
    dk_acc = tl.zeros([CHUNK, BK], dtype=tl.float32)

    dbeta_acc = tl.zeros([CHUNK], dtype=tl.float32)
    for vb in range(NVB):
        dbeta_acc += tl.load(p_dbeta_ptr + (pid * NVB + vb) * CHUNK + offs_c)

    # Through A: dq / dk and the cumg adjoint.
    agg_cc = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)
    for vb in range(NVB):
        agg_cc += tl.load(
            p_da_ptr
            + (pid * NVB + vb) * CHUNK * CHUNK
            + offs_c[:, None] * CHUNK
            + offs_c[None, :]
        )
    qk = tl.dot(q_vals, tl.trans(k_vals)).to(tl.float32)
    d_qk = agg_cc * exp_causal
    dq_acc += tl.dot(d_qk.to(tl.bfloat16), k_vals).to(tl.float32)
    dk_acc += tl.dot(tl.trans(d_qk).to(tl.bfloat16), q_vals).to(tl.float32)
    prod_a = agg_cc * qk * exp_causal
    dcumg = tl.sum(prod_a, axis=1) - tl.sum(prod_a, axis=0)

    # Through the outgoing state Ks^T u.
    agg_ck = tl.zeros([CHUNK, BK], dtype=tl.float32)
    for vb in range(NVB):
        agg_ck += tl.load(
            p_dks_ptr
            + (pid * NVB + vb) * CHUNK * K
            + offs_c[:, None] * K
            + offs_k[None, :],
            mask=k_mask[None, :],
            other=0.0,
        )
    f_vals = tl.exp(last_cum - cumg)
    dk_acc += agg_ck * f_vals[:, None]
    df_vals = tl.sum(agg_ck * k_fp, axis=1)
    dcumg -= df_vals * f_vals
    ss_acc = 0.0
    for vb in range(NVB):
        ss_acc += tl.load(p_ss_ptr + pid * NVB + vb)
    dcumg += tl.where(
        offs_c == CHUNK - 1,
        tl.sum(df_vals * f_vals) + ss_acc * exp_last,
        0.0,
    )

    # Adjoint of the triangular solves.
    agg_cc = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)
    for vb in range(NVB):
        agg_cc += tl.load(
            p_dt_ptr
            + (pid * NVB + vb) * CHUNK * CHUNK
            + offs_c[:, None] * CHUNK
            + offs_c[None, :]
        )
    agg_ck = tl.zeros([CHUNK, BK], dtype=tl.float32)
    for vb in range(NVB):
        agg_ck += tl.load(
            p_dw_ptr
            + (pid * NVB + vb) * CHUNK * K
            + offs_c[:, None] * K
            + offs_k[None, :],
            mask=k_mask[None, :],
            other=0.0,
        )
    w_vals = tl.load(
        w_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
        mask=t_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    tinv = tl.load(
        tinv_ptr
        + pid * CHUNK * CHUNK
        + offs_c[:, None] * CHUNK
        + offs_c[None, :]
    ).to(tl.bfloat16)
    y_w = tl.dot(tl.trans(tinv), (-agg_ck).to(tl.bfloat16)).to(tl.float32)
    d_tmat = -agg_cc - tl.dot(y_w.to(tl.bfloat16), tl.trans(w_vals)).to(
        tl.float32
    )
    d_l = tl.where(lower, d_tmat, 0.0)
    gram = tl.dot(k_vals, tl.trans(k_vals)).to(tl.float32)
    coeff = d_l * beta_vals[:, None] * exp_lower
    dk_acc += tl.dot(coeff.to(tl.bfloat16), k_vals).to(tl.float32)
    dk_acc += tl.dot(tl.trans(coeff).to(tl.bfloat16), k_vals).to(tl.float32)
    prod_l = coeff * gram
    dcumg += tl.sum(prod_l, axis=1) - tl.sum(prod_l, axis=0)
    dbeta_acc += tl.sum(d_l * exp_lower * gram, axis=1)

    # Through M = beta * exp(cumg) * k (y_w is dM here).
    dk_acc += y_w * (beta_vals * exp_cum)[:, None]
    dbeta_acc += tl.sum(y_w * (exp_cum[:, None] * k_fp), axis=1)
    de_m = tl.sum(y_w * (beta_vals[:, None] * k_fp), axis=1)
    de_o = tl.zeros([CHUNK], dtype=tl.float32)
    for vb in range(NVB):
        de_o += tl.load(p_de_ptr + (pid * NVB + vb) * CHUNK + offs_c)
    dcumg += (de_o + de_m) * exp_cum

    # dg is a reverse cumsum of the cumg adjoint within the chunk.
    dg_vals = tl.sum(
        tl.where(offs_c[None, :] >= offs_c[:, None], dcumg[None, :], 0.0),
        axis=1,
    )

    tl.store(
        dq_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
        dq_acc,
        mask=t_mask[:, None] & k_mask[None, :],
    )
    tl.store(
        dk_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
        dk_acc,
        mask=t_mask[:, None] & k_mask[None, :],
    )
    tl.store(dg_ptr + bh * T + offs_t, dg_vals, mask=t_mask)
    tl.store(dbeta_ptr + bh * T + offs_t, dbeta_acc, mask=t_mask)


# ---------------------------------------------------------------------------
# Wrapper / autograd
# ---------------------------------------------------------------------------

def _pow2_at_least(x, lo=16):
    p = lo
    while p < x:
        p *= 2
    return p


def _config(K, V):
    bk = _pow2_at_least(K)
    bvf = _pow2_at_least(V)
    if bk <= 128 and bvf <= 128:
        return bk, bvf, 64, 32, 64, 8
    return bk, bvf, 32, 16, 32, 4


def _forward_impl(q_f, k_f, v_f, g_f, beta_f, scale_val, cfg):
    BK, BVF, CHUNK, BS, BV, num_warps = cfg
    BH, T, K = q_f.shape
    V = v_f.shape[-1]
    NT = (T + CHUNK - 1) // CHUNK
    num_v_blocks = (V + BV - 1) // BV
    dev = q_f.device

    w = torch.empty((BH, T, K), device=dev, dtype=torch.bfloat16)
    u = torch.empty((BH, T, V), device=dev, dtype=torch.bfloat16)
    a_qk = torch.empty(
        (BH * NT, CHUNK, CHUNK), device=dev, dtype=torch.bfloat16
    )
    h_snap = torch.empty((BH * NT, K, V), device=dev, dtype=torch.bfloat16)
    out = torch.empty((BH, T, V), device=dev, dtype=torch.float32)

    _wy_kernel[(BH * NT,)](
        q_f, k_f, v_f, g_f, beta_f, w, u, a_qk,
        T, K, V, NT, CHUNK, BS, BK, BVF,
        num_warps=num_warps, num_stages=1,
    )
    _state_kernel[(BH, num_v_blocks)](
        k_f, g_f, w, u, h_snap,
        T, K, V, NT, CHUNK, BK, BV,
        num_warps=num_warps, num_stages=1,
    )
    _out_kernel[(BH * NT, num_v_blocks)](
        q_f, g_f, a_qk, u, h_snap, out,
        T, K, V, NT, scale_val, CHUNK, BK, BV,
        num_warps=num_warps, num_stages=1,
    )
    return out, w, u, a_qk, h_snap, NT


class _GatedDeltaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        ctx.shape = (B, H, T, K, V)
        ctx.scale = float(scale)
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype)
        ctx.special = T == 0 or K == 0 or V == 0

        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        g_c = g.contiguous()
        beta_c = beta.contiguous()

        if ctx.special:
            ctx.save_for_backward(q_c, k_c, v_c, g_c, beta_c)
            if T == 0 or V == 0:
                return torch.empty(
                    (B, H, T, V), device=q.device, dtype=torch.float32
                )
            return torch.zeros(
                (B, H, T, V), device=q.device, dtype=torch.float32
            )

        cfg = _config(K, V)
        ctx.cfg = cfg
        BH = B * H
        q_f = q_c.reshape(BH, T, K)
        k_f = k_c.reshape(BH, T, K)
        v_f = v_c.reshape(BH, T, V)
        g_f = g_c.reshape(BH, T)
        beta_f = beta_c.reshape(BH, T)

        out, w, u, a_qk, h_snap, NT = _forward_impl(
            q_f, k_f, v_f, g_f, beta_f, ctx.scale, cfg
        )
        ctx.NT = NT
        ctx.save_for_backward(
            q_f, k_f, v_f, g_f, beta_f, w, u, a_qk, h_snap
        )
        return out.reshape(B, H, T, V)

    @staticmethod
    def backward(ctx, grad_out):
        B, H, T, K, V = ctx.shape
        dq_t, dk_t, dv_t, dg_t, dbeta_t = ctx.dtypes
        needs = ctx.needs_input_grad

        if ctx.special:
            dev = grad_out.device
            zeros = (
                torch.zeros((B, H, T, K), device=dev, dtype=dq_t),
                torch.zeros((B, H, T, K), device=dev, dtype=dk_t),
                torch.zeros((B, H, T, V), device=dev, dtype=dv_t),
                torch.zeros((B, H, T), device=dev, dtype=dg_t),
                torch.zeros((B, H, T), device=dev, dtype=dbeta_t),
            )
            return (
                zeros[0] if needs[0] else None,
                zeros[1] if needs[1] else None,
                zeros[2] if needs[2] else None,
                zeros[3] if needs[3] else None,
                zeros[4] if needs[4] else None,
                None,
            )

        (q_f, k_f, v_f, g_f, beta_f, w, u, a_qk, h_snap) = (
            ctx.saved_tensors
        )
        BK, BVF, CHUNK, BS, BV, num_warps = ctx.cfg
        NT = ctx.NT
        BH = B * H
        dev = grad_out.device

        go_f = grad_out.contiguous().reshape(BH, T, V)
        num_v_blocks = (V + BV - 1) // BV
        n_chunks = BH * NT
        n_part = n_chunks * num_v_blocks

        du = torch.empty((BH, T, V), device=dev, dtype=torch.float32)
        ds_snap = torch.empty(
            (n_chunks, K, V), device=dev, dtype=torch.float32
        )
        tinv = torch.empty(
            (n_chunks, CHUNK, CHUNK), device=dev, dtype=torch.float32
        )
        p_da = torch.empty(
            (n_part, CHUNK, CHUNK), device=dev, dtype=torch.float32
        )
        p_dt = torch.empty(
            (n_part, CHUNK, CHUNK), device=dev, dtype=torch.float32
        )
        p_dw = torch.empty(
            (n_part, CHUNK, K), device=dev, dtype=torch.float32
        )
        p_dks = torch.empty(
            (n_part, CHUNK, K), device=dev, dtype=torch.float32
        )
        p_dq = torch.empty(
            (n_part, CHUNK, K), device=dev, dtype=torch.float32
        )
        p_de = torch.empty((n_part, CHUNK), device=dev, dtype=torch.float32)
        p_dbeta = torch.empty(
            (n_part, CHUNK), device=dev, dtype=torch.float32
        )
        p_ss = torch.empty((n_part,), device=dev, dtype=torch.float32)

        # Gradients are stored directly in each input's dtype by the
        # kernels, removing the separate cast kernels from the profile.
        dq_f = torch.empty((BH, T, K), device=dev, dtype=dq_t)
        dk_f = torch.empty((BH, T, K), device=dev, dtype=dk_t)
        dv_f = torch.empty((BH, T, V), device=dev, dtype=dv_t)
        dg_f = torch.empty((BH, T), device=dev, dtype=dg_t)
        dbeta_f = torch.empty((BH, T), device=dev, dtype=dbeta_t)

        _bwd_state_kernel2[(BH, num_v_blocks)](
            q_f, k_f, g_f, w, h_snap, a_qk, go_f, du, ds_snap,
            T, K, V, NT, ctx.scale, CHUNK, BK, BV,
            num_warps=num_warps, num_stages=1,
        )
        _bwd_inv_kernel[(n_chunks,)](
            k_f, g_f, beta_f, tinv,
            T, K, NT, CHUNK, BS, BK,
            num_warps=4, num_stages=1,
        )
        _bwd_vblock_kernel[(n_chunks, num_v_blocks)](
            q_f, v_f, g_f, beta_f, w, u, h_snap, go_f, du, ds_snap,
            tinv, dv_f, p_da, p_dt, p_dw, p_dks, p_dq, p_de, p_dbeta,
            p_ss,
            T, K, V, NT, ctx.scale, num_v_blocks, CHUNK, BK, BV,
            num_warps=num_warps, num_stages=1,
        )
        _bwd_final_kernel[(n_chunks,)](
            q_f, k_f, g_f, beta_f, w, tinv,
            p_da, p_dt, p_dw, p_dks, p_dq, p_de, p_dbeta, p_ss,
            dq_f, dk_f, dg_f, dbeta_f,
            T, K, V, NT, num_v_blocks, CHUNK, BK,
            num_warps=num_warps, num_stages=1,
        )

        grads = (
            dq_f.reshape(B, H, T, K),
            dk_f.reshape(B, H, T, K),
            dv_f.reshape(B, H, T, V),
            dg_f.reshape(B, H, T),
            dbeta_f.reshape(B, H, T),
        )
        return (
            grads[0] if needs[0] else None,
            grads[1] if needs[1] else None,
            grads[2] if needs[2] else None,
            grads[3] if needs[3] else None,
            grads[4] if needs[4] else None,
            None,
        )


def kernel_function(q, k, v, g, beta, scale):
    if isinstance(scale, torch.Tensor):
        scale_val = float(scale.detach().item())
    else:
        scale_val = float(scale)
    return _GatedDeltaFunction.apply(q, k, v, g, beta, scale_val)
```

The 45 ms backward kernel was diagnosed as a register-spill problem: it fused a 64-step serial triangular inversion, a runtime-length loop over V, and many simultaneously live `[chunk, K]` fp32 accumulators and state tiles into one program. The backward is therefore split into a blocked 2×2 inverse kernel, a per-(chunk, V-block) kernel that performs the transposed solve and stores each V-reduction partial immediately, and a lean per-chunk final kernel that sums the partials and finishes dq/dk/dg/dbeta, while gradients are now stored directly in the input dtypes to remove cast kernels. The forward and reverse state sweep are unchanged from the previously passing version. The main unverified risks are Triton register allocation in the new V-block kernel for the wide-tile fallback shapes and the extra partial-buffer traffic, both of which should still be far cheaper than the original spilled kernel.