```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernels (chunkwise gated delta rule with a Kronecker k/q factor)
# ---------------------------------------------------------------------------

@triton.jit
def _prepare_inverse_kernel_real(
    k_ptr,
    k2_ptr,
    g_ptr,
    beta_ptr,
    inv_ptr,
    T,
    NT,
    D,
    E: tl.constexpr,
    CHUNK: tl.constexpr,
    DB: tl.constexpr,
    INV_ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NT
    chunk = pid % NT

    offs_c = tl.arange(0, CHUNK)
    offs_d = tl.arange(0, DB)
    t_abs = chunk * CHUNK + offs_c
    mask_t = t_abs < T

    kkt = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
    for d_start in range(0, D, DB):
        offs_dd = d_start + offs_d
        mask_d = offs_dd < D
        k_tile = tl.load(
            k_ptr + bh * T * D + t_abs[:, None] * D + offs_dd[None, :],
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        kkt += tl.dot(k_tile, tl.trans(k_tile), input_precision="ieee")

    row_ge_col = offs_c[:, None] >= offs_c[None, :]
    fee = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
    for e in range(E):
        k2_vec = tl.load(
            k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
        ).to(tl.float32)
        g_vec = tl.load(
            g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
        ).to(tl.float32)
        cum = tl.cumsum(g_vec, axis=0)
        diff = cum[:, None] - cum[None, :]
        exp_diff = tl.exp(tl.where(row_ge_col, diff, 0.0))
        fee += tl.where(
            row_ge_col, (k2_vec[:, None] * k2_vec[None, :]) * exp_diff, 0.0
        )

    beta_vec = tl.load(beta_ptr + bh * T + t_abs, mask=mask_t, other=0.0).to(
        tl.float32
    )
    strict = offs_c[:, None] > offs_c[None, :]
    eye = tl.where(offs_c[:, None] == offs_c[None, :], 1.0, 0.0)
    mat = eye + tl.where(strict, kkt * fee * beta_vec[None, :], 0.0)

    # Exact inverse of I + L for strictly lower triangular L: the
    # Newton-Schulz iteration doubles the exact Neumann series each step.
    inv = eye
    for _ in range(INV_ITERS):
        mx = tl.dot(mat, inv, input_precision="ieee")
        inv = tl.dot(inv, 2.0 * eye - mx, input_precision="ieee")

    tl.store(
        inv_ptr
        + (bh * NT + chunk) * CHUNK * CHUNK
        + offs_c[:, None] * CHUNK
        + offs_c[None, :],
        inv,
    )


@triton.jit
def _prepare_inverse_fast(
    k_ptr,
    k2_ptr,
    g_ptr,
    beta_ptr,
    inv_ptr,
    T,
    NT,
    D: tl.constexpr,
    E: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # D == 128 specialisation: one full-width bf16 Gram dot.
    pid = tl.program_id(0)
    bh = pid // NT
    chunk = pid % NT

    offs_c = tl.arange(0, CHUNK)
    offs_d = tl.arange(0, D)
    t_abs = chunk * CHUNK + offs_c
    mask_t = t_abs < T

    k_mat = tl.load(
        k_ptr + bh * T * D + t_abs[:, None] * D + offs_d[None, :],
        mask=mask_t[:, None],
        other=0.0,
    )
    kkt = tl.dot(k_mat, tl.trans(k_mat))

    row_ge_col = offs_c[:, None] >= offs_c[None, :]
    fee = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
    for e in range(E):
        k2_vec = tl.load(
            k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
        ).to(tl.float32)
        g_vec = tl.load(
            g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
        ).to(tl.float32)
        cum = tl.cumsum(g_vec, axis=0)
        diff = cum[:, None] - cum[None, :]
        exp_diff = tl.exp(tl.where(row_ge_col, diff, 0.0))
        fee += tl.where(
            row_ge_col, (k2_vec[:, None] * k2_vec[None, :]) * exp_diff, 0.0
        )

    beta_vec = tl.load(beta_ptr + bh * T + t_abs, mask=mask_t, other=0.0).to(
        tl.float32
    )
    strict = offs_c[:, None] > offs_c[None, :]
    eye = tl.where(offs_c[:, None] == offs_c[None, :], 1.0, 0.0)
    mat = eye + tl.where(strict, kkt * fee * beta_vec[None, :], 0.0)

    inv = eye
    for _ in range(6):
        mx = tl.dot(mat, inv, input_precision="ieee")
        inv = tl.dot(inv, 2.0 * eye - mx, input_precision="ieee")

    tl.store(
        inv_ptr
        + (bh * NT + chunk) * CHUNK * CHUNK
        + offs_c[:, None] * CHUNK
        + offs_c[None, :],
        inv,
    )


@triton.jit
def _state_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q2_ptr,
    k2_ptr,
    g_ptr,
    beta_ptr,
    inv_ptr,
    out_ptr,
    state_ptr,
    cp_ptr,
    scale,
    T,
    NT,
    NBV,
    D,
    V,
    E: tl.constexpr,
    CHUNK: tl.constexpr,
    DB: tl.constexpr,
    VB: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NBV
    vb = pid % NBV

    offs_c = tl.arange(0, CHUNK)
    offs_d = tl.arange(0, DB)
    offs_v = vb * VB + tl.arange(0, VB)
    mask_v = offs_v < V
    row_ge_col = offs_c[:, None] >= offs_c[None, :]

    for chunk in range(NT):
        t_abs = chunk * CHUNK + offs_c
        mask_t = t_abs < T

        # Save the state at this chunk boundary for the backward pass.
        for e in range(E):
            for d_start in range(0, D, DB):
                offs_dd = d_start + offs_d
                mask_d = offs_dd < D
                s_ptrs = (
                    state_ptr
                    + ((bh * E + e) * D + offs_dd[:, None]) * V
                    + offs_v[None, :]
                )
                s_tile = tl.load(
                    s_ptrs, mask=mask_d[:, None] & mask_v[None, :], other=0.0
                )
                tl.store(
                    cp_ptr
                    + (((bh * NT + chunk) * E + e) * D + offs_dd[:, None]) * V
                    + offs_v[None, :],
                    s_tile,
                    mask=mask_d[:, None] & mask_v[None, :],
                )

        v_mat = tl.load(
            v_ptr + bh * T * V + t_abs[:, None] * V + offs_v[None, :],
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        beta_vec = tl.load(
            beta_ptr + bh * T + t_abs, mask=mask_t, other=0.0
        ).to(tl.float32)
        inv_mat = tl.load(
            inv_ptr
            + (bh * NT + chunk) * CHUNK * CHUNK
            + offs_c[:, None] * CHUNK
            + offs_c[None, :]
        )

        pred = tl.zeros((CHUNK, VB), dtype=tl.float32)
        out_inter = tl.zeros((CHUNK, VB), dtype=tl.float32)
        qk = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
        for d_start in range(0, D, DB):
            offs_dd = d_start + offs_d
            mask_d = offs_dd < D
            k_tile = tl.load(
                k_ptr + bh * T * D + t_abs[:, None] * D + offs_dd[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            q_tile = tl.load(
                q_ptr + bh * T * D + t_abs[:, None] * D + offs_dd[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            qk += tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee")
            for e in range(E):
                g_vec = tl.load(
                    g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
                ).to(tl.float32)
                cum = tl.cumsum(g_vec, axis=0)
                exp_cum = tl.exp(cum)
                k2_vec = tl.load(
                    k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
                ).to(tl.float32)
                q2_vec = tl.load(
                    q2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
                ).to(tl.float32)
                s_tile = tl.load(
                    state_ptr
                    + ((bh * E + e) * D + offs_dd[:, None]) * V
                    + offs_v[None, :],
                    mask=mask_d[:, None] & mask_v[None, :],
                    other=0.0,
                )
                pred += tl.dot(
                    k_tile * (k2_vec * exp_cum)[:, None],
                    s_tile,
                    input_precision="ieee",
                )
                out_inter += tl.dot(
                    q_tile * (q2_vec * exp_cum)[:, None],
                    s_tile,
                    input_precision="ieee",
                )

        delta = tl.dot(inv_mat, v_mat - pred, input_precision="ieee")

        feq = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
        for e in range(E):
            g_vec = tl.load(
                g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            cum = tl.cumsum(g_vec, axis=0)
            k2_vec = tl.load(
                k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            q2_vec = tl.load(
                q2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            diff = cum[:, None] - cum[None, :]
            exp_diff = tl.exp(tl.where(row_ge_col, diff, 0.0))
            feq += tl.where(
                row_ge_col, (q2_vec[:, None] * k2_vec[None, :]) * exp_diff, 0.0
            )
        r_mat = tl.where(row_ge_col, qk * feq * beta_vec[None, :], 0.0)
        out_mat = (
            out_inter + tl.dot(r_mat, delta, input_precision="ieee")
        ) * scale
        tl.store(
            out_ptr + bh * T * V + t_abs[:, None] * V + offs_v[None, :],
            out_mat,
            mask=mask_t[:, None] & mask_v[None, :],
        )

        for e in range(E):
            g_vec = tl.load(
                g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            cum = tl.cumsum(g_vec, axis=0)
            cum_last = tl.sum(tl.where(offs_c == CHUNK - 1, cum, 0.0))
            exp_last = tl.exp(cum_last)
            k2_vec = tl.load(
                k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            a_vec = beta_vec * k2_vec * tl.exp(cum_last - cum)
            for d_start in range(0, D, DB):
                offs_dd = d_start + offs_d
                mask_d = offs_dd < D
                k_tile = tl.load(
                    k_ptr + bh * T * D + t_abs[:, None] * D + offs_dd[None, :],
                    mask=mask_t[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                state_ptrs = (
                    state_ptr
                    + ((bh * E + e) * D + offs_dd[:, None]) * V
                    + offs_v[None, :]
                )
                s_tile = tl.load(
                    state_ptrs, mask=mask_d[:, None] & mask_v[None, :], other=0.0
                )
                s_new = s_tile * exp_last + tl.dot(
                    tl.trans(k_tile * a_vec[:, None]),
                    delta,
                    input_precision="ieee",
                )
                tl.store(
                    state_ptrs, s_new, mask=mask_d[:, None] & mask_v[None, :]
                )


@triton.jit
def _state_kernel_fast(
    q_ptr,
    k_ptr,
    v_ptr,
    q2_ptr,
    k2_ptr,
    g_ptr,
    beta_ptr,
    inv_ptr,
    out_ptr,
    state_ptr,
    cp_ptr,
    scale,
    T,
    NT,
    NBV,
    D: tl.constexpr,
    V: tl.constexpr,
    E: tl.constexpr,
    CHUNK: tl.constexpr,
    VB: tl.constexpr,
):
    # Specialised for D == V == 128, E == 4.
    pid = tl.program_id(0)
    bh = pid // NBV
    vb = pid % NBV

    offs_c = tl.arange(0, CHUNK)
    offs_d = tl.arange(0, D)
    offs_v = vb * VB + tl.arange(0, VB)
    mask_v = offs_v < V
    row_ge_col = offs_c[:, None] >= offs_c[None, :]

    for chunk in range(NT):
        t_abs = chunk * CHUNK + offs_c
        mask_t = t_abs < T

        k_mat = tl.load(
            k_ptr + bh * T * D + t_abs[:, None] * D + offs_d[None, :],
            mask=mask_t[:, None],
            other=0.0,
        )
        q_mat = tl.load(
            q_ptr + bh * T * D + t_abs[:, None] * D + offs_d[None, :],
            mask=mask_t[:, None],
            other=0.0,
        )
        qk = tl.dot(q_mat, tl.trans(k_mat))

        pred = tl.zeros((CHUNK, VB), dtype=tl.float32)
        out_inter = tl.zeros((CHUNK, VB), dtype=tl.float32)
        feq = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
        for e in range(E):
            g_vec = tl.load(
                g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            cum = tl.cumsum(g_vec, axis=0)
            exp_cum = tl.exp(cum)
            k2_vec = tl.load(
                k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            q2_vec = tl.load(
                q2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            state_ptrs = (
                state_ptr
                + ((bh * E + e) * D + offs_d[:, None]) * V
                + offs_v[None, :]
            )
            s_fp = tl.load(state_ptrs, mask=mask_v[None, :], other=0.0)
            tl.store(
                cp_ptr
                + (((bh * NT + chunk) * E + e) * D + offs_d[:, None]) * V
                + offs_v[None, :],
                s_fp,
                mask=mask_v[None, :],
            )
            s_e = s_fp.to(tl.bfloat16)
            ks = (k_mat * (k2_vec * exp_cum)[:, None]).to(tl.bfloat16)
            qs = (q_mat * (q2_vec * exp_cum)[:, None]).to(tl.bfloat16)
            pred += tl.dot(ks, s_e)
            out_inter += tl.dot(qs, s_e)
            diff = cum[:, None] - cum[None, :]
            exp_diff = tl.exp(tl.where(row_ge_col, diff, 0.0))
            feq += tl.where(
                row_ge_col, (q2_vec[:, None] * k2_vec[None, :]) * exp_diff, 0.0
            )

        v_mat = tl.load(
            v_ptr + bh * T * V + t_abs[:, None] * V + offs_v[None, :],
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        inv_mat = tl.load(
            inv_ptr
            + (bh * NT + chunk) * CHUNK * CHUNK
            + offs_c[:, None] * CHUNK
            + offs_c[None, :]
        )
        delta = tl.dot(inv_mat, v_mat - pred, input_precision="tf32")

        beta_vec = tl.load(
            beta_ptr + bh * T + t_abs, mask=mask_t, other=0.0
        ).to(tl.float32)
        r_mat = tl.where(row_ge_col, qk * feq * beta_vec[None, :], 0.0)
        delta_bf = delta.to(tl.bfloat16)
        out_mat = (
            out_inter + tl.dot(r_mat.to(tl.bfloat16), delta_bf)
        ) * scale
        tl.store(
            out_ptr + bh * T * V + t_abs[:, None] * V + offs_v[None, :],
            out_mat,
            mask=mask_t[:, None] & mask_v[None, :],
        )

        for e in range(E):
            g_vec = tl.load(
                g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            cum = tl.cumsum(g_vec, axis=0)
            cum_last = tl.sum(tl.where(offs_c == CHUNK - 1, cum, 0.0))
            k2_vec = tl.load(
                k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            a_vec = beta_vec * k2_vec * tl.exp(cum_last - cum)
            aks = (k_mat * a_vec[:, None]).to(tl.bfloat16)
            upd = tl.dot(tl.trans(aks), delta_bf)
            state_ptrs = (
                state_ptr
                + ((bh * E + e) * D + offs_d[:, None]) * V
                + offs_v[None, :]
            )
            s_e = tl.load(state_ptrs, mask=mask_v[None, :], other=0.0)
            tl.store(
                state_ptrs,
                s_e * tl.exp(cum_last) + upd,
                mask=mask_v[None, :],
            )


# ---------------------------------------------------------------------------
# Generic backward kernel (unchanged math; used for all non-128 shapes)
# ---------------------------------------------------------------------------

@triton.jit
def _backward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q2_ptr,
    k2_ptr,
    g_ptr,
    beta_ptr,
    inv_ptr,
    cp_ptr,
    do_ptr,
    adj_ptr,
    gv_ptr,
    pdq_ptr,
    pdk_ptr,
    pdq2_ptr,
    pdk2_ptr,
    pdg_ptr,
    pdbeta_ptr,
    scale,
    T,
    NT,
    NBV,
    BH,
    D,
    V,
    E: tl.constexpr,
    CHUNK: tl.constexpr,
    DB: tl.constexpr,
    VB: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NBV
    vb = pid % NBV

    offs_c = tl.arange(0, CHUNK)
    offs_d = tl.arange(0, DB)
    offs_v = vb * VB + tl.arange(0, VB)
    mask_v = offs_v < V
    row_ge_col = offs_c[:, None] >= offs_c[None, :]
    strict = offs_c[:, None] > offs_c[None, :]

    for chunk_rev in range(NT):
        chunk = NT - 1 - chunk_rev
        t_abs = chunk * CHUNK + offs_c
        mask_t = t_abs < T

        beta_vec = tl.load(
            beta_ptr + bh * T + t_abs, mask=mask_t, other=0.0
        ).to(tl.float32)
        inv_mat = tl.load(
            inv_ptr
            + (bh * NT + chunk) * CHUNK * CHUNK
            + offs_c[:, None] * CHUNK
            + offs_c[None, :]
        )
        v_mat = tl.load(
            v_ptr + bh * T * V + t_abs[:, None] * V + offs_v[None, :],
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        do_mat = tl.load(
            do_ptr + bh * T * V + t_abs[:, None] * V + offs_v[None, :],
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        yo = do_mat * scale

        kk = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
        qk = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
        for d_start in range(0, D, DB):
            offs_dd = d_start + offs_d
            mask_d = offs_dd < D
            k_tile = tl.load(
                k_ptr + bh * T * D + t_abs[:, None] * D + offs_dd[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            q_tile = tl.load(
                q_ptr + bh * T * D + t_abs[:, None] * D + offs_dd[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            kk += tl.dot(k_tile, tl.trans(k_tile), input_precision="ieee")
            qk += tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee")

        fee_k = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
        fee_q = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
        for e in range(E):
            k2_vec = tl.load(
                k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            q2_vec = tl.load(
                q2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            g_vec = tl.load(
                g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            cum = tl.cumsum(g_vec, axis=0)
            diff = cum[:, None] - cum[None, :]
            exp_diff = tl.exp(tl.where(row_ge_col, diff, 0.0))
            fee_k += tl.where(
                row_ge_col, (k2_vec[:, None] * k2_vec[None, :]) * exp_diff, 0.0
            )
            fee_q += tl.where(
                row_ge_col, (q2_vec[:, None] * k2_vec[None, :]) * exp_diff, 0.0
            )

        l_mat = tl.where(strict, kk * fee_k * beta_vec[None, :], 0.0)
        r_mat = tl.where(row_ge_col, qk * fee_q * beta_vec[None, :], 0.0)

        pred = tl.zeros((CHUNK, VB), dtype=tl.float32)
        for e in range(E):
            g_vec = tl.load(
                g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            cum = tl.cumsum(g_vec, axis=0)
            a_vec = tl.exp(cum)
            k2_vec = tl.load(
                k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            for d_start in range(0, D, DB):
                offs_dd = d_start + offs_d
                mask_d = offs_dd < D
                k_tile = tl.load(
                    k_ptr + bh * T * D + t_abs[:, None] * D + offs_dd[None, :],
                    mask=mask_t[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                s0_tile = tl.load(
                    cp_ptr
                    + (((bh * NT + chunk) * E + e) * D + offs_dd[:, None]) * V
                    + offs_v[None, :],
                    mask=mask_d[:, None] & mask_v[None, :],
                    other=0.0,
                )
                pred += tl.dot(
                    k_tile * (k2_vec * a_vec)[:, None],
                    s0_tile,
                    input_precision="ieee",
                )

        u_mat = tl.dot(inv_mat, v_mat - pred, input_precision="ieee")

        lam = tl.dot(tl.trans(r_mat), yo, input_precision="ieee")
        for e in range(E):
            g_vec = tl.load(
                g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            cum = tl.cumsum(g_vec, axis=0)
            cum_last = tl.sum(tl.where(offs_c == CHUNK - 1, cum, 0.0))
            k2_vec = tl.load(
                k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            fsend = beta_vec * tl.exp(cum_last - cum)
            for d_start in range(0, D, DB):
                offs_dd = d_start + offs_d
                mask_d = offs_dd < D
                k_tile = tl.load(
                    k_ptr + bh * T * D + t_abs[:, None] * D + offs_dd[None, :],
                    mask=mask_t[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                g_tile = tl.load(
                    adj_ptr
                    + ((bh * E + e) * D + offs_dd[:, None]) * V
                    + offs_v[None, :],
                    mask=mask_d[:, None] & mask_v[None, :],
                    other=0.0,
                )
                x_tile = k_tile * k2_vec[:, None]
                lam += tl.dot(
                    x_tile * fsend[:, None], g_tile, input_precision="ieee"
                )

        mu = tl.dot(tl.trans(inv_mat), lam, input_precision="ieee")
        tl.store(
            gv_ptr + bh * T * V + t_abs[:, None] * V + offs_v[None, :],
            mu,
            mask=mask_t[:, None] & mask_v[None, :],
        )
        dpred = -mu

        dl_mat = tl.where(
            strict, -tl.dot(mu, tl.trans(u_mat), input_precision="ieee"), 0.0
        )
        dr_mat = tl.where(
            row_ge_col, tl.dot(yo, tl.trans(u_mat), input_precision="ieee"), 0.0
        )
        dl_b = dl_mat * beta_vec[None, :]
        dr_b = dr_mat * beta_vec[None, :]

        dbeta_acc = tl.zeros((CHUNK,), dtype=tl.float32)
        for e in range(E):
            g_vec = tl.load(
                g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            cum = tl.cumsum(g_vec, axis=0)
            a_vec = tl.exp(cum)
            cum_last = tl.sum(tl.where(offs_c == CHUNK - 1, cum, 0.0))
            a_last = tl.exp(cum_last)
            k2_vec = tl.load(
                k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            q2_vec = tl.load(
                q2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            diff = cum[:, None] - cum[None, :]
            exp_diff = tl.where(
                row_ge_col, tl.exp(tl.where(row_ge_col, diff, 0.0)), 0.0
            )
            wl_mat = dl_b * exp_diff
            wr_mat = dr_b * exp_diff
            fsend = beta_vec * tl.exp(cum_last - cum)
            decay_src = tl.exp(cum_last - cum)

            dc_vec = tl.zeros((CHUNK,), dtype=tl.float32)
            dk2_acc = tl.zeros((CHUNK,), dtype=tl.float32)
            dq2_acc = tl.zeros((CHUNK,), dtype=tl.float32)
            dbeta_send = tl.zeros((CHUNK,), dtype=tl.float32)
            pair_l = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
            pair_r = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
            dc_last = 0.0

            for d_start in range(0, D, DB):
                offs_dd = d_start + offs_d
                mask_d = offs_dd < D
                k_tile = tl.load(
                    k_ptr + bh * T * D + t_abs[:, None] * D + offs_dd[None, :],
                    mask=mask_t[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                q_tile = tl.load(
                    q_ptr + bh * T * D + t_abs[:, None] * D + offs_dd[None, :],
                    mask=mask_t[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                s0_tile = tl.load(
                    cp_ptr
                    + (((bh * NT + chunk) * E + e) * D + offs_dd[:, None]) * V
                    + offs_v[None, :],
                    mask=mask_d[:, None] & mask_v[None, :],
                    other=0.0,
                )
                adj_ptrs = (
                    adj_ptr
                    + ((bh * E + e) * D + offs_dd[:, None]) * V
                    + offs_v[None, :]
                )
                g_tile = tl.load(
                    adj_ptrs, mask=mask_d[:, None] & mask_v[None, :], other=0.0
                )

                x_tile = k_tile * k2_vec[:, None]
                y_tile = q_tile * q2_vec[:, None]

                hx = tl.dot(dpred, tl.trans(s0_tile), input_precision="ieee")
                hy = tl.dot(yo, tl.trans(s0_tile), input_precision="ieee")
                dx = hx * a_vec[:, None]
                dy = hy * a_vec[:, None]
                dc_vec += a_vec * tl.sum(hx * x_tile + hy * y_tile, axis=1)

                pair_l += (
                    tl.dot(x_tile, tl.trans(x_tile), input_precision="ieee")
                    * exp_diff
                )
                pair_r += (
                    tl.dot(y_tile, tl.trans(x_tile), input_precision="ieee")
                    * exp_diff
                )
                dx += tl.dot(wl_mat, x_tile, input_precision="ieee")
                dx += tl.dot(tl.trans(wl_mat), x_tile, input_precision="ieee")
                dx += tl.dot(tl.trans(wr_mat), y_tile, input_precision="ieee")
                dy += tl.dot(wr_mat, x_tile, input_precision="ieee")

                proj = tl.dot(u_mat, tl.trans(g_tile), input_precision="ieee")
                dx_send = proj * fsend[:, None]
                dx += dx_send
                dc_vec -= tl.sum(dx_send * x_tile, axis=1)
                dbeta_send += tl.sum(x_tile * proj, axis=1) * decay_src

                dk2_acc += tl.sum(dx * k_tile, axis=1)
                dq2_acc += tl.sum(dy * q_tile, axis=1)

                dk_ptrs = (
                    pdk_ptr
                    + ((vb * BH + bh) * T + t_abs[:, None]) * D
                    + offs_dd[None, :]
                )
                dk_old = tl.load(
                    dk_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0
                )
                tl.store(
                    dk_ptrs,
                    dk_old + dx * k2_vec[:, None],
                    mask=mask_t[:, None] & mask_d[None, :],
                )
                dq_ptrs = (
                    pdq_ptr
                    + ((vb * BH + bh) * T + t_abs[:, None]) * D
                    + offs_dd[None, :]
                )
                dq_old = tl.load(
                    dq_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0
                )
                tl.store(
                    dq_ptrs,
                    dq_old + dy * q2_vec[:, None],
                    mask=mask_t[:, None] & mask_d[None, :],
                )

                send_val = a_last * s0_tile + tl.dot(
                    tl.trans(x_tile * fsend[:, None]),
                    u_mat,
                    input_precision="ieee",
                )
                dc_last += tl.sum(g_tile * send_val)

                g0_tile = (
                    a_last * g_tile
                    + tl.dot(
                        tl.trans(x_tile * a_vec[:, None]),
                        dpred,
                        input_precision="ieee",
                    )
                    + tl.dot(
                        tl.trans(y_tile * a_vec[:, None]),
                        yo,
                        input_precision="ieee",
                    )
                )
                tl.store(
                    adj_ptrs,
                    g0_tile,
                    mask=mask_d[:, None] & mask_v[None, :],
                )

            pair_term = dl_b * pair_l + dr_b * pair_r
            dc_vec += tl.sum(pair_term, axis=1) - tl.sum(pair_term, axis=0)
            dbeta_acc += tl.sum(dl_mat * pair_l + dr_mat * pair_r, axis=0)
            dbeta_acc += dbeta_send
            dc_vec += tl.where(offs_c == CHUNK - 1, dc_last, 0.0)

            dg_vec = tl.sum(
                tl.where(
                    offs_c[None, :] >= offs_c[:, None], dc_vec[None, :], 0.0
                ),
                axis=1,
            )
            tl.store(
                pdg_ptr + ((vb * BH + bh) * T + t_abs) * E + e,
                dg_vec,
                mask=mask_t,
            )
            tl.store(
                pdk2_ptr + ((vb * BH + bh) * T + t_abs) * E + e,
                dk2_acc,
                mask=mask_t,
            )
            tl.store(
                pdq2_ptr + ((vb * BH + bh) * T + t_abs) * E + e,
                dq2_acc,
                mask=mask_t,
            )

        tl.store(
            pdbeta_ptr + (vb * BH + bh) * T + t_abs, dbeta_acc, mask=mask_t
        )


# ---------------------------------------------------------------------------
# Fast backward kernel for D == V == 128, E == 4: identical adjoint math to
# the generic kernel above, but full-width tiles loaded once per chunk and
# bf16/tf32 tensor-core dots (fp32 accumulation, state and outputs) instead
# of 32-wide IEEE fp32 dots.  dq/dk partials are accumulated per slice into
# the per-V-block buffers, exactly as in the generic kernel.
# ---------------------------------------------------------------------------

@triton.jit
def _backward_kernel_fast(
    q_ptr,
    k_ptr,
    v_ptr,
    q2_ptr,
    k2_ptr,
    g_ptr,
    beta_ptr,
    inv_ptr,
    cp_ptr,
    do_ptr,
    adj_ptr,
    gv_ptr,
    pdq_ptr,
    pdk_ptr,
    pdq2_ptr,
    pdk2_ptr,
    pdg_ptr,
    pdbeta_ptr,
    scale,
    T,
    NT,
    NBV,
    BH,
    D: tl.constexpr,
    V: tl.constexpr,
    E: tl.constexpr,
    CHUNK: tl.constexpr,
    VB: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NBV
    vb = pid % NBV

    offs_c = tl.arange(0, CHUNK)
    offs_d = tl.arange(0, D)
    offs_v = vb * VB + tl.arange(0, VB)
    mask_v = offs_v < V
    row_ge_col = offs_c[:, None] >= offs_c[None, :]
    strict = offs_c[:, None] > offs_c[None, :]

    for chunk_rev in range(NT):
        chunk = NT - 1 - chunk_rev
        t_abs = chunk * CHUNK + offs_c
        mask_t = t_abs < T

        beta_vec = tl.load(
            beta_ptr + bh * T + t_abs, mask=mask_t, other=0.0
        ).to(tl.float32)
        inv_mat = tl.load(
            inv_ptr
            + (bh * NT + chunk) * CHUNK * CHUNK
            + offs_c[:, None] * CHUNK
            + offs_c[None, :]
        )
        k_mat = tl.load(
            k_ptr + bh * T * D + t_abs[:, None] * D + offs_d[None, :],
            mask=mask_t[:, None],
            other=0.0,
        )
        q_mat = tl.load(
            q_ptr + bh * T * D + t_abs[:, None] * D + offs_d[None, :],
            mask=mask_t[:, None],
            other=0.0,
        )
        v_mat = tl.load(
            v_ptr + bh * T * V + t_abs[:, None] * V + offs_v[None, :],
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        do_mat = tl.load(
            do_ptr + bh * T * V + t_abs[:, None] * V + offs_v[None, :],
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        yo = do_mat * scale
        yo_bf = yo.to(tl.bfloat16)

        kk = tl.dot(k_mat, tl.trans(k_mat))
        qk = tl.dot(q_mat, tl.trans(k_mat))

        fee_k = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
        fee_q = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
        for e in range(E):
            k2_vec = tl.load(
                k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            q2_vec = tl.load(
                q2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            g_vec = tl.load(
                g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            cum = tl.cumsum(g_vec, axis=0)
            diff = cum[:, None] - cum[None, :]
            exp_diff = tl.exp(tl.where(row_ge_col, diff, 0.0))
            fee_k += tl.where(
                row_ge_col, (k2_vec[:, None] * k2_vec[None, :]) * exp_diff, 0.0
            )
            fee_q += tl.where(
                row_ge_col, (q2_vec[:, None] * k2_vec[None, :]) * exp_diff, 0.0
            )

        l_mat = tl.where(strict, kk * fee_k * beta_vec[None, :], 0.0)
        r_mat = tl.where(row_ge_col, qk * fee_q * beta_vec[None, :], 0.0)

        # Recompute the forward residuals u for this chunk.
        pred = tl.zeros((CHUNK, VB), dtype=tl.float32)
        for e in range(E):
            g_vec = tl.load(
                g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            cum = tl.cumsum(g_vec, axis=0)
            a_vec = tl.exp(cum)
            k2_vec = tl.load(
                k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            s0_tile = tl.load(
                cp_ptr
                + (((bh * NT + chunk) * E + e) * D + offs_d[:, None]) * V
                + offs_v[None, :],
                mask=mask_v[None, :],
                other=0.0,
            )
            ks = (k_mat * (k2_vec * a_vec)[:, None]).to(tl.bfloat16)
            pred += tl.dot(ks, s0_tile.to(tl.bfloat16))

        u_mat = tl.dot(inv_mat, v_mat - pred, input_precision="tf32")
        u_bf = u_mat.to(tl.bfloat16)

        # Transposed triangular solve.
        lam = tl.dot(tl.trans(r_mat), yo, input_precision="tf32")
        for e in range(E):
            g_vec = tl.load(
                g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            cum = tl.cumsum(g_vec, axis=0)
            cum_last = tl.sum(tl.where(offs_c == CHUNK - 1, cum, 0.0))
            k2_vec = tl.load(
                k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            fsend = beta_vec * tl.exp(cum_last - cum)
            g_tile = tl.load(
                adj_ptr
                + ((bh * E + e) * D + offs_d[:, None]) * V
                + offs_v[None, :],
                mask=mask_v[None, :],
                other=0.0,
            )
            xsend = (k_mat * (k2_vec * fsend)[:, None]).to(tl.bfloat16)
            lam += tl.dot(xsend, g_tile.to(tl.bfloat16))

        mu = tl.dot(tl.trans(inv_mat), lam, input_precision="tf32")
        tl.store(
            gv_ptr + bh * T * V + t_abs[:, None] * V + offs_v[None, :],
            mu,
            mask=mask_t[:, None] & mask_v[None, :],
        )
        dpred = -mu
        dpred_bf = dpred.to(tl.bfloat16)

        dl_mat = tl.where(
            strict, -tl.dot(mu, tl.trans(u_mat), input_precision="tf32"), 0.0
        )
        dr_mat = tl.where(
            row_ge_col, tl.dot(yo, tl.trans(u_mat), input_precision="tf32"), 0.0
        )
        dl_b = dl_mat * beta_vec[None, :]
        dr_b = dr_mat * beta_vec[None, :]

        k_f = k_mat.to(tl.float32)
        q_f = q_mat.to(tl.float32)

        dbeta_acc = tl.zeros((CHUNK,), dtype=tl.float32)
        for e in range(E):
            g_vec = tl.load(
                g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            cum = tl.cumsum(g_vec, axis=0)
            a_vec = tl.exp(cum)
            cum_last = tl.sum(tl.where(offs_c == CHUNK - 1, cum, 0.0))
            a_last = tl.exp(cum_last)
            k2_vec = tl.load(
                k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            q2_vec = tl.load(
                q2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            diff = cum[:, None] - cum[None, :]
            exp_diff = tl.where(
                row_ge_col, tl.exp(tl.where(row_ge_col, diff, 0.0)), 0.0
            )
            wl_mat = dl_b * exp_diff
            wr_mat = dr_b * exp_diff
            fsend = beta_vec * tl.exp(cum_last - cum)
            decay_src = tl.exp(cum_last - cum)

            s0_tile = tl.load(
                cp_ptr
                + (((bh * NT + chunk) * E + e) * D + offs_d[:, None]) * V
                + offs_v[None, :],
                mask=mask_v[None, :],
                other=0.0,
            )
            adj_ptrs = (
                adj_ptr
                + ((bh * E + e) * D + offs_d[:, None]) * V
                + offs_v[None, :]
            )
            g_tile = tl.load(adj_ptrs, mask=mask_v[None, :], other=0.0)
            s0_bf = s0_tile.to(tl.bfloat16)
            g_bf = g_tile.to(tl.bfloat16)

            x_tile = k_f * k2_vec[:, None]
            y_tile = q_f * q2_vec[:, None]
            x_bf = x_tile.to(tl.bfloat16)
            y_bf = y_tile.to(tl.bfloat16)

            hx = tl.dot(dpred_bf, tl.trans(s0_bf))
            hy = tl.dot(yo_bf, tl.trans(s0_bf))
            dx = hx * a_vec[:, None]
            dy = hy * a_vec[:, None]
            dc_vec = a_vec * tl.sum(hx * x_tile + hy * y_tile, axis=1)

            pair_l = tl.dot(x_bf, tl.trans(x_bf)) * exp_diff
            pair_r = tl.dot(y_bf, tl.trans(x_bf)) * exp_diff
            wl_bf = wl_mat.to(tl.bfloat16)
            wr_bf = wr_mat.to(tl.bfloat16)
            dx += tl.dot(wl_bf, x_bf)
            dx += tl.dot(tl.trans(wl_bf), x_bf)
            dx += tl.dot(tl.trans(wr_bf), y_bf)
            dy += tl.dot(wr_bf, x_bf)

            proj = tl.dot(u_bf, tl.trans(g_bf))
            dx_send = proj * fsend[:, None]
            dx += dx_send
            dc_vec -= tl.sum(dx_send * x_tile, axis=1)
            dbeta_send = tl.sum(x_tile * proj, axis=1) * decay_src

            dk2_acc = tl.sum(dx * k_f, axis=1)
            dq2_acc = tl.sum(dy * q_f, axis=1)

            dk_ptrs = (
                pdk_ptr
                + ((vb * BH + bh) * T + t_abs[:, None]) * D
                + offs_d[None, :]
            )
            dk_old = tl.load(dk_ptrs, mask=mask_t[:, None], other=0.0)
            tl.store(
                dk_ptrs, dk_old + dx * k2_vec[:, None], mask=mask_t[:, None]
            )
            dq_ptrs = (
                pdq_ptr
                + ((vb * BH + bh) * T + t_abs[:, None]) * D
                + offs_d[None, :]
            )
            dq_old = tl.load(dq_ptrs, mask=mask_t[:, None], other=0.0)
            tl.store(
                dq_ptrs, dq_old + dy * q2_vec[:, None], mask=mask_t[:, None]
            )

            xsend_bf = (k_mat * (k2_vec * fsend)[:, None]).to(tl.bfloat16)
            send_val = a_last * s0_tile + tl.dot(tl.trans(xsend_bf), u_bf)
            dc_last = tl.sum(g_tile * send_val)

            xa_bf = (k_mat * (k2_vec * a_vec)[:, None]).to(tl.bfloat16)
            ya_bf = (q_mat * (q2_vec * a_vec)[:, None]).to(tl.bfloat16)
            g0_tile = (
                a_last * g_tile
                + tl.dot(tl.trans(xa_bf), dpred_bf)
                + tl.dot(tl.trans(ya_bf), yo_bf)
            )
            tl.store(adj_ptrs, g0_tile, mask=mask_v[None, :])

            pair_term = dl_b * pair_l + dr_b * pair_r
            dc_vec += tl.sum(pair_term, axis=1) - tl.sum(pair_term, axis=0)
            dbeta_acc += tl.sum(dl_mat * pair_l + dr_mat * pair_r, axis=0)
            dbeta_acc += dbeta_send
            dc_vec += tl.where(offs_c == CHUNK - 1, dc_last, 0.0)

            dg_vec = tl.sum(
                tl.where(
                    offs_c[None, :] >= offs_c[:, None], dc_vec[None, :], 0.0
                ),
                axis=1,
            )
            tl.store(
                pdg_ptr + ((vb * BH + bh) * T + t_abs) * E + e,
                dg_vec,
                mask=mask_t,
            )
            tl.store(
                pdk2_ptr + ((vb * BH + bh) * T + t_abs) * E + e,
                dk2_acc,
                mask=mask_t,
            )
            tl.store(
                pdq2_ptr + ((vb * BH + bh) * T + t_abs) * E + e,
                dq2_acc,
                mask=mask_t,
            )

        tl.store(
            pdbeta_ptr + (vb * BH + bh) * T + t_abs, dbeta_acc, mask=mask_t
        )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def _forward_impl(q_c, k_c, v_c, q2_c, k2_c, g_c, beta_c, scale_val):
    BH, T, D = q_c.shape
    V = v_c.shape[-1]
    E = k2_c.shape[-1]

    out = torch.empty((BH, T, V), device=q_c.device, dtype=torch.float32)
    state = torch.zeros((BH, E, D, V), device=q_c.device, dtype=torch.float32)

    if D == 128 and V == 128 and E == 4:
        CHUNK = 64
        VB = 64
        nt = (T + CHUNK - 1) // CHUNK
        nbv = (V + VB - 1) // VB
        inv = torch.empty(
            (BH, nt, CHUNK, CHUNK), device=q_c.device, dtype=torch.float32
        )
        cp = torch.empty(
            (BH, nt, E, D, V), device=q_c.device, dtype=torch.float32
        )
        _prepare_inverse_fast[(BH * nt,)](
            k_c, k2_c, g_c, beta_c, inv, T, nt,
            D=D, E=E, CHUNK=CHUNK,
            num_warps=4, num_stages=1,
        )
        _state_kernel_fast[(BH * nbv,)](
            q_c, k_c, v_c, q2_c, k2_c, g_c, beta_c, inv, out, state, cp,
            scale_val, T, nt, nbv,
            D=D, V=V, E=E, CHUNK=CHUNK, VB=VB,
            num_warps=8, num_stages=1,
        )
        return out, inv, cp, CHUNK, nt

    CHUNK = 32
    DB = 64
    VB = 64 if V >= 64 else (32 if V >= 32 else 16)
    nt = (T + CHUNK - 1) // CHUNK
    nbv = (V + VB - 1) // VB
    inv = torch.empty((BH, nt, CHUNK, CHUNK), device=q_c.device, dtype=torch.float32)
    cp = torch.empty((BH, nt, E, D, V), device=q_c.device, dtype=torch.float32)
    _prepare_inverse_kernel_real[(BH * nt,)](
        k_c, k2_c, g_c, beta_c, inv, T, nt, D,
        E=E, CHUNK=CHUNK, DB=DB, INV_ITERS=5,
        num_warps=4, num_stages=1,
    )
    _state_kernel[(BH * nbv,)](
        q_c, k_c, v_c, q2_c, k2_c, g_c, beta_c, inv, out, state, cp,
        scale_val, T, nt, nbv, D, V,
        E=E, CHUNK=CHUNK, DB=DB, VB=VB,
        num_warps=4, num_stages=1,
    )
    return out, inv, cp, CHUNK, nt


class _GatedDeltaKron(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q2, k2, g, beta, scale):
        ctx.in_shapes = (q.shape, k.shape, v.shape, q2.shape, k2.shape,
                         g.shape, beta.shape)
        ctx.in_dtypes = (q.dtype, k.dtype, v.dtype, q2.dtype, k2.dtype,
                         g.dtype, beta.dtype)
        ctx.scale = float(scale)
        B, H, T, D = q.shape
        V = v.shape[-1]
        E = k2.shape[-1]
        ctx.B, ctx.H, ctx.T, ctx.D, ctx.V, ctx.E = B, H, T, D, V, E

        if B == 0 or H == 0 or T == 0:
            ctx.empty = True
            return torch.empty((B, H, T, V), device=q.device,
                               dtype=torch.float32)
        ctx.empty = False

        q_c = q.contiguous().view(B * H, T, D)
        k_c = k.contiguous().view(B * H, T, D)
        v_c = v.contiguous().view(B * H, T, V)
        q2_c = q2.contiguous().view(B * H, T, E)
        k2_c = k2.contiguous().view(B * H, T, E)
        g_c = g.contiguous().view(B * H, T, E)
        beta_c = beta.contiguous().view(B * H, T)

        out_c, inv, cp, chunk, nt = _forward_impl(
            q_c, k_c, v_c, q2_c, k2_c, g_c, beta_c, ctx.scale
        )
        ctx.chunk = chunk
        ctx.nt = nt
        ctx.save_for_backward(q_c, k_c, v_c, q2_c, k2_c, g_c, beta_c, inv, cp)
        return out_c.view(B, H, T, V)

    @staticmethod
    def backward(ctx, grad_out):
        B, H, T, D, V, E = ctx.B, ctx.H, ctx.T, ctx.D, ctx.V, ctx.E
        shapes = ctx.in_shapes
        dtypes = ctx.in_dtypes
        if ctx.empty:
            return (
                torch.zeros(shapes[0], device=grad_out.device, dtype=dtypes[0]),
                torch.zeros(shapes[1], device=grad_out.device, dtype=dtypes[1]),
                torch.zeros(shapes[2], device=grad_out.device, dtype=dtypes[2]),
                torch.zeros(shapes[3], device=grad_out.device, dtype=dtypes[3]),
                torch.zeros(shapes[4], device=grad_out.device, dtype=dtypes[4]),
                torch.zeros(shapes[5], device=grad_out.device, dtype=dtypes[5]),
                torch.zeros(shapes[6], device=grad_out.device, dtype=dtypes[6]),
                None,
            )

        (q_c, k_c, v_c, q2_c, k2_c, g_c, beta_c, inv, cp) = ctx.saved_tensors
        BH = B * H
        do_c = grad_out.contiguous().to(torch.float32).view(BH, T, V)

        adj = torch.zeros((BH, E, D, V), device=q_c.device, dtype=torch.float32)
        gv = torch.empty((BH, T, V), device=q_c.device, dtype=torch.float32)

        if D == 128 and V == 128 and E == 4 and ctx.chunk == 64:
            VB = 64
            nbv = (V + VB - 1) // VB
            pdq = torch.zeros((nbv, BH, T, D), device=q_c.device, dtype=torch.float32)
            pdk = torch.zeros((nbv, BH, T, D), device=q_c.device, dtype=torch.float32)
            pdq2 = torch.zeros((nbv, BH, T, E), device=q_c.device, dtype=torch.float32)
            pdk2 = torch.zeros((nbv, BH, T, E), device=q_c.device, dtype=torch.float32)
            pdg = torch.zeros((nbv, BH, T, E), device=q_c.device, dtype=torch.float32)
            pdbeta = torch.zeros((nbv, BH, T), device=q_c.device, dtype=torch.float32)
            _backward_kernel_fast[(BH * nbv,)](
                q_c, k_c, v_c, q2_c, k2_c, g_c, beta_c, inv, cp, do_c, adj, gv,
                pdq, pdk, pdq2, pdk2, pdg, pdbeta,
                ctx.scale, T, ctx.nt, nbv, BH,
                D=D, V=V, E=E, CHUNK=ctx.chunk, VB=VB,
                num_warps=8, num_stages=1,
            )
        else:
            VB = 32 if V >= 32 else 16
            nbv = (V + VB - 1) // VB
            pdq = torch.zeros((nbv, BH, T, D), device=q_c.device, dtype=torch.float32)
            pdk = torch.zeros((nbv, BH, T, D), device=q_c.device, dtype=torch.float32)
            pdq2 = torch.zeros((nbv, BH, T, E), device=q_c.device, dtype=torch.float32)
            pdk2 = torch.zeros((nbv, BH, T, E), device=q_c.device, dtype=torch.float32)
            pdg = torch.zeros((nbv, BH, T, E), device=q_c.device, dtype=torch.float32)
            pdbeta = torch.zeros((nbv, BH, T), device=q_c.device, dtype=torch.float32)
            _backward_kernel[(BH * nbv,)](
                q_c, k_c, v_c, q2_c, k2_c, g_c, beta_c, inv, cp, do_c, adj, gv,
                pdq, pdk, pdq2, pdk2, pdg, pdbeta,
                ctx.scale, T, ctx.nt, nbv, BH, D, V,
                E=E, CHUNK=ctx.chunk, DB=32, VB=VB,
                num_warps=4, num_stages=1,
            )

        # Reduction over V-blocks (reduction glue), then cast to input dtypes.
        dq = pdq.sum(dim=0).view(B, H, T, D).to(dtypes[0])
        dk = pdk.sum(dim=0).view(B, H, T, D).to(dtypes[1])
        dv = gv.view(B, H, T, V).to(dtypes[2])
        dq2 = pdq2.sum(dim=0).view(B, H, T, E).to(dtypes[3])
        dk2 = pdk2.sum(dim=0).view(B, H, T, E).to(dtypes[4])
        dg = pdg.sum(dim=0).view(B, H, T, E).to(dtypes[5])
        dbeta = pdbeta.sum(dim=0).view(B, H, T).to(dtypes[6])
        return dq, dk, dv, dq2, dk2, dg, dbeta, None


def kernel_function(q, k, v, q2, k2, g, beta, scale):
    try:
        scale_val = float(scale)
    except Exception:
        scale_val = float(scale.item())
    return _GatedDeltaKron.apply(q, k, v, q2, k2, g, beta, scale_val)
```

The backward was dominated by 32-wide IEEE-fp32 dots and repeated tiled reloads, so the timed $D{=}V{=}128$, $E{=}4$ shape now uses a specialised backward kernel that loads full $64\times128$ $q/k$ tiles once per chunk and performs all large products as bf16 tensor-core dots with fp32 accumulation, keeping only the $64\times64$ triangular-solve products at tf32 and all states, adjoints, and gradient outputs in fp32. It also halves the V-block count (64-wide instead of 32-wide), halving both the duplicated chunk-level work and the wrapper's partial-gradient reductions, while every other shape keeps the previously validated generic backward bit-for-bit. The block-inverse preparation got the same treatment (one full-width bf16 Gram dot instead of two IEEE tiled dots, Newton–Schulz kept at IEEE). I cannot run anything, so the main uncertainties are register/shared-memory pressure in the new backward kernel at these tile sizes and whether bf16 rounding stays inside the $2\times10^{-2}$ tolerance on the most adversarial scaled gradient inputs.