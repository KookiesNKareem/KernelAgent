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

        # Snapshot the incoming state for the output kernel and for bwd.
        h_ptrs = (
            h_ptr
            + ((bh * NT + chunk) * K + offs_k[:, None]) * V
            + v_cols[None, :]
        )
        tl.store(
            h_ptrs, state_bf, mask=k_mask[:, None] & v_mask[None, :]
        )

        # Overwrite Uhat with the effective update.
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
# Chunk closed form (S0 = incoming state, e_i = exp(cumg_i), Tmat = I + L,
# W = Tmat^-1 (beta*e*k), Uhat = Tmat^-1 (beta*v), u = Uhat - W S0,
# S_out = e_last S0 + Ks^T u, o/scale = e * (q S0) + A u):
#   _bwd_state_kernel sweeps chunks in reverse carrying the state adjoint
#   DS, emits the per-chunk state-adjoint snapshot and Du (grad wrt u).
#   _bwd_chunk_kernel then differentiates the closed form per chunk,
#   including the adjoint of the triangular solve, in parallel.
# ---------------------------------------------------------------------------

@triton.jit
def _bwd_state_kernel(
    k_ptr,
    g_ptr,
    w_ptr,
    u_ptr,
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

    # Adjoint of the state block [BK, BV] leaving the current chunk.
    dstate = tl.zeros([BK, BV], dtype=tl.float32)

    for rev in range(NT):
        chunk = NT - 1 - rev
        pid_c = bh * NT + chunk
        offs_t = chunk * CHUNK + offs_c
        t_mask = offs_t < T

        # Snapshot DS (adjoint wrt this chunk's outgoing state).
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

        go_vals = tl.load(
            go_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
            mask=t_mask[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        go_s = go_vals * scale

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
            go_ptr * 0 + k_ptr * 0 + w_ptr * 0 + u_ptr * 0 + h_ptr * 0
            + a_ptr * 0 + ds_ptr * 0 + go_ptr * 0,  # placeholder, unused
            mask=False,
            other=0.0,
        ) if False else tl.zeros([CHUNK, BK], dtype=tl.bfloat16)

        # Du = A^T Go_s + Ks DS, Ks_j = k_j * exp(last_cum - cumg_j).
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

        # dS0 = q^T (Go_s * e) + e_last * DS - W^T Du.
        # q is loaded through the go-independent path below.
        q_load = tl.load(
            ds_ptr * 0 + h_ptr * 0 + a_ptr * 0 + du_ptr * 0 + go_ptr * 0
            + k_ptr * 0 + g_ptr * 0 + w_ptr * 0 + u_ptr * 0,
            mask=False,
            other=0.0,
        ) if False else q_vals
        _ = q_load

        dstate = dstate  # updated by _bwd_state_kernel_impl semantics below
        # NOTE: q-dependent term and -W^T Du are folded in via the
        # auxiliary buffers in the wrapper (see _bwd_state_kernel2).
        _ = exp_cum
        _ = w_vals
        _ = du_vals


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
def _bwd_chunk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    w_ptr,
    u_ptr,
    h_ptr,
    go_ptr,
    du_ptr,
    ds_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dg_ptr,
    dbeta_ptr,
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
    bh = pid // NT
    chunk = pid % NT

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
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
    w_vals = tl.load(
        w_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
        mask=t_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    k_fp = k_vals.to(tl.float32)

    # Rebuild L and invert Tmat = I + L by forward substitution.
    gram = tl.dot(k_vals, tl.trans(k_vals)).to(tl.float32)
    diff = cumg[:, None] - cumg[None, :]
    lower = offs_c[:, None] > offs_c[None, :]
    causal = offs_c[None, :] <= offs_c[:, None]
    exp_lower = tl.where(lower, tl.exp(tl.where(lower, diff, 0.0)), 0.0)
    exp_causal = tl.where(
        causal, tl.exp(tl.where(causal, diff, 0.0)), 0.0
    )
    mat_l = tl.where(lower, gram * exp_lower * beta_vals[:, None], 0.0)

    tinv = (offs_c[:, None] == offs_c[None, :]).to(tl.float32)
    for r in range(CHUNK):
        row_r = tl.sum(
            tl.where(offs_c[:, None] == r, tinv, 0.0), axis=0
        )
        col_r = tl.sum(
            tl.where(offs_c[None, :] == r, mat_l, 0.0), axis=1
        )
        tinv = tinv - col_r[:, None] * row_r[None, :]
    tinv_t = tl.trans(tinv).to(tl.bfloat16)

    qk = tl.dot(q_vals, tl.trans(k_vals)).to(tl.float32)

    dq_acc = tl.zeros([CHUNK, BK], dtype=tl.float32)
    dk_acc = tl.zeros([CHUNK, BK], dtype=tl.float32)
    dw_acc = tl.zeros([CHUNK, BK], dtype=tl.float32)
    dks_acc = tl.zeros([CHUNK, BK], dtype=tl.float32)
    da_mat = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)
    dt_u = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)
    dbeta_acc = tl.zeros([CHUNK], dtype=tl.float32)
    de_o = tl.zeros([CHUNK], dtype=tl.float32)
    ss_acc = 0.0

    # Accumulate every sum over V in BV-wide blocks.
    for vb in range(0, V, BV):
        cols = vb + offs_v
        v_mask = cols < V

        go_s = (
            tl.load(
                go_ptr + bh * T * V + offs_t[:, None] * V + cols[None, :],
                mask=t_mask[:, None] & v_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            * scale
        )
        u_vals = tl.load(
            u_ptr + bh * T * V + offs_t[:, None] * V + cols[None, :],
            mask=t_mask[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        du_vals = tl.load(
            du_ptr + bh * T * V + offs_t[:, None] * V + cols[None, :],
            mask=t_mask[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v_vals = tl.load(
            v_ptr + bh * T * V + offs_t[:, None] * V + cols[None, :],
            mask=t_mask[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        s0_blk = tl.load(
            h_ptr + (pid * K + offs_k[:, None]) * V + cols[None, :],
            mask=k_mask[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        ds_blk = tl.load(
            ds_ptr + (pid * K + offs_k[:, None]) * V + cols[None, :],
            mask=k_mask[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        q_s0 = tl.dot(q_vals, s0_blk).to(tl.float32)
        de_o = de_o + tl.sum(go_s * q_s0, axis=1)
        dq_acc = dq_acc + tl.dot(
            (go_s * exp_cum[:, None]).to(tl.bfloat16), tl.trans(s0_blk)
        ).to(tl.float32)
        da_mat = da_mat + tl.dot(
            go_s.to(tl.bfloat16), tl.trans(u_vals)
        ).to(tl.float32)
        dw_acc = dw_acc + tl.dot(
            du_vals.to(tl.bfloat16), tl.trans(s0_blk)
        ).to(tl.float32)
        dks_acc = dks_acc + tl.dot(
            u_vals, tl.trans(ds_blk.to(tl.bfloat16))
        ).to(tl.float32)
        ss_acc = ss_acc + tl.sum(ds_blk * s0_blk.to(tl.float32))

        uhat_blk = u_vals.to(tl.float32) + tl.dot(w_vals, s0_blk).to(
            tl.float32
        )
        y_u = tl.dot(tinv_t, du_vals.to(tl.bfloat16)).to(tl.float32)

        tl.store(
            dv_ptr + bh * T * V + offs_t[:, None] * V + cols[None, :],
            y_u * beta_vals[:, None],
            mask=t_mask[:, None] & v_mask[None, :],
        )
        dt_u = dt_u + tl.dot(
            y_u.to(tl.bfloat16), tl.trans(uhat_blk.to(tl.bfloat16))
        ).to(tl.float32)
        dbeta_acc = dbeta_acc + tl.sum(y_u * v_vals, axis=1)

    # Gradients through A (intra-chunk output matrix).
    d_qk = da_mat * exp_causal
    dq_acc = dq_acc + tl.dot(d_qk.to(tl.bfloat16), k_vals).to(tl.float32)
    dk_acc = dk_acc + tl.dot(
        tl.trans(d_qk).to(tl.bfloat16), q_vals
    ).to(tl.float32)
    prod_a = da_mat * qk * exp_causal
    dcumg = tl.sum(prod_a, axis=1) - tl.sum(prod_a, axis=0)

    # Gradients through the outgoing state Ks^T u term.
    f_vals = tl.exp(last_cum - cumg)
    dk_acc = dk_acc + dks_acc * f_vals[:, None]
    df_vals = tl.sum(dks_acc * k_fp, axis=1)
    dcumg = dcumg - df_vals * f_vals
    dcumg = dcumg + tl.where(
        offs_c == CHUNK - 1,
        tl.sum(df_vals * f_vals) + ss_acc * exp_last,
        0.0,
    )

    # Adjoint of the triangular solves.
    y_w = tl.dot(tinv_t, (-dw_acc).to(tl.bfloat16)).to(tl.float32)
    d_mat = y_w
    dt_mat = -dt_u - tl.dot(
        y_w.to(tl.bfloat16), tl.trans(w_vals)
    ).to(tl.float32)
    d_l = tl.where(lower, dt_mat, 0.0)

    coeff_kk = d_l * beta_vals[:, None] * exp_lower
    dk_acc = dk_acc + tl.dot(coeff_kk.to(tl.bfloat16), k_vals).to(
        tl.float32
    ) + tl.dot(tl.trans(coeff_kk).to(tl.bfloat16), k_vals).to(tl.float32)
    prod_l = coeff_kk * gram
    dcumg = dcumg + tl.sum(prod_l, axis=1) - tl.sum(prod_l, axis=0)
    dbeta_acc = dbeta_acc + tl.sum(d_l * exp_lower * gram, axis=1)

    # Gradients through M = beta * exp(cumg) * k.
    dk_acc = dk_acc + d_mat * (beta_vals * exp_cum)[:, None]
    dbeta_acc = dbeta_acc + tl.sum(
        d_mat * (exp_cum[:, None] * k_fp), axis=1
    )
    de_m = tl.sum(d_mat * (beta_vals[:, None] * k_fp), axis=1)
    dcumg = dcumg + (de_o + de_m) * exp_cum

    # cumg_j = sum_{i <= j} g_i (local), so dg is a reverse cumsum.
    dg_vals = tl.sum(
        tl.where(
            offs_c[None, :] >= offs_c[:, None], dcumg[None, :], 0.0
        ),
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

        du = torch.empty((BH, T, V), device=dev, dtype=torch.float32)
        ds_snap = torch.empty(
            (BH * NT, K, V), device=dev, dtype=torch.float32
        )
        dq_f = torch.empty((BH, T, K), device=dev, dtype=torch.float32)
        dk_f = torch.empty((BH, T, K), device=dev, dtype=torch.float32)
        dv_f = torch.empty((BH, T, V), device=dev, dtype=torch.float32)
        dg_f = torch.empty((BH, T), device=dev, dtype=torch.float32)
        dbeta_f = torch.empty((BH, T), device=dev, dtype=torch.float32)

        _bwd_state_kernel2[(BH, num_v_blocks)](
            q_f, k_f, g_f, w, h_snap, a_qk, go_f, du, ds_snap,
            T, K, V, NT, ctx.scale, CHUNK, BK, BV,
            num_warps=num_warps, num_stages=1,
        )
        _bwd_chunk_kernel[(BH * NT,)](
            q_f, k_f, v_f, g_f, beta_f, w, u, h_snap, go_f, du,
            ds_snap, dq_f, dk_f, dv_f, dg_f, dbeta_f,
            T, K, V, NT, ctx.scale, CHUNK, BK, BV,
            num_warps=num_warps, num_stages=1,
        )

        grads = (
            dq_f.reshape(B, H, T, K).to(dq_t),
            dk_f.reshape(B, H, T, K).to(dk_t),
            dv_f.reshape(B, H, T, V).to(dv_t),
            dg_f.reshape(B, H, T).to(dg_t),
            dbeta_f.reshape(B, H, T).to(dbeta_t),
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

The forward reuses the chunkwise WY/state/output kernels and saves the W and effective-update factors, the decay-weighted intra-chunk qk matrix, and per-chunk incoming-state snapshots. The backward is derived from the chunk closed form: a reverse sequential kernel carries the state adjoint across chunks and produces the gradient with respect to the effective updates, then a fully parallel per-chunk kernel differentiates the output/state terms and applies the adjoint of the triangular WY solve (via the transposed inverse) to obtain all of dq, dk, dv, dg, and dbeta, with dg recovered by a reverse cumsum of the cumulative-decay adjoints. Gradients are accumulated in fp32 and cast to each input's dtype in the autograd wrapper. The main risks I cannot verify without running are Triton corner cases in the reverse-sweep kernels (scalar accumulation and the blocked V loop in the chunk backward) and accuracy loss from the bf16 forward intermediates feeding the analytic backward.