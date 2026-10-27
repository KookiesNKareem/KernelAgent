import torch
import triton
import triton.language as tl


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

    # (k_t . k_s), accumulated over fixed-size D tiles.  Fixed tiles keep
    # shared memory bounded independently of D (e.g. D = 129).
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

    # fee[t, s] = sum_e k2[t,e] k2[s,e] exp(cum_g[t,e] - cum_g[s,e]), t >= s.
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

    # Exact inverse of I + L for strictly lower triangular L: L is
    # nilpotent, and each Newton-Schulz step doubles the number of exact
    # Neumann-series terms.
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
    # Specialised for D == V == 128, E == 4: full-D tiles, 64-token chunks,
    # and bf16 tensor-core dots (fp32 accumulation and fp32 state/output).
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

        # Inter-chunk terms and the intra-chunk gate factor share one
        # pass over the slices (previously recomputed three times).
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
            s_e = tl.load(
                state_ptr
                + ((bh * E + e) * D + offs_d[:, None]) * V
                + offs_v[None, :],
                mask=mask_v[None, :],
                other=0.0,
            ).to(tl.bfloat16)
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
        # The triangular solve stays in tf32; the bulk state dots are bf16.
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


def kernel_function(q, k, v, q2, k2, g, beta, scale):
    B, H, T, D = q.shape
    V = v.shape[-1]
    E = k2.shape[-1]
    BH = B * H

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if T == 0 or B == 0 or H == 0:
        return out

    # Layout prep only; all arithmetic is in the Triton kernels above.
    q_c = q.contiguous().view(BH, T, D)
    k_c = k.contiguous().view(BH, T, D)
    v_c = v.contiguous().view(BH, T, V)
    q2_c = q2.contiguous().view(BH, T, E)
    k2_c = k2.contiguous().view(BH, T, E)
    g_c = g.contiguous().view(BH, T, E)
    beta_c = beta.contiguous().view(BH, T)
    out_c = out.view(BH, T, V)

    try:
        scale_val = float(scale)
    except Exception:
        scale_val = float(scale.item())

    # Slice-major state [BH, E, D, V]: slice e is one gated [D, V] block.
    state = torch.zeros((BH, E, D, V), device=q.device, dtype=torch.float32)

    if D == 128 and V == 128 and E == 4:
        CHUNK = 64
        VB = 64
        nt = (T + CHUNK - 1) // CHUNK
        nbv = (V + VB - 1) // VB
        inv = torch.empty(
            (BH, nt, CHUNK, CHUNK), device=q.device, dtype=torch.float32
        )
        _prepare_inverse_kernel_real[(BH * nt,)](
            k_c,
            k2_c,
            g_c,
            beta_c,
            inv,
            T,
            nt,
            D,
            E=E,
            CHUNK=CHUNK,
            DB=64,
            INV_ITERS=6,
            num_warps=4,
            num_stages=1,
        )
        _state_kernel_fast[(BH * nbv,)](
            q_c,
            k_c,
            v_c,
            q2_c,
            k2_c,
            g_c,
            beta_c,
            inv,
            out_c,
            state,
            scale_val,
            T,
            nt,
            nbv,
            D=D,
            V=V,
            E=E,
            CHUNK=CHUNK,
            VB=VB,
            num_warps=8,
            num_stages=1,
        )
        return out

    # General fallback (any D, V, E): the previously validated tiling.
    CHUNK = 32
    DB = 64
    VB = 64 if V >= 64 else (32 if V >= 32 else 16)
    nt = (T + CHUNK - 1) // CHUNK
    nbv = (V + VB - 1) // VB

    inv = torch.empty((BH, nt, CHUNK, CHUNK), device=q.device, dtype=torch.float32)
    _prepare_inverse_kernel_real[(BH * nt,)](
        k_c,
        k2_c,
        g_c,
        beta_c,
        inv,
        T,
        nt,
        D,
        E=E,
        CHUNK=CHUNK,
        DB=DB,
        INV_ITERS=5,
        num_warps=4,
        num_stages=1,
    )
    _state_kernel[(BH * nbv,)](
        q_c,
        k_c,
        v_c,
        q2_c,
        k2_c,
        g_c,
        beta_c,
        inv,
        out_c,
        state,
        scale_val,
        T,
        nt,
        nbv,
        D,
        V,
        E=E,
        CHUNK=CHUNK,
        DB=DB,
        VB=VB,
        num_warps=4,
        num_stages=1,
    )
    return out
