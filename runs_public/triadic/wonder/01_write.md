```python
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_inverse_kernel(
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
    BLOCK_D: tl.constexpr,
    INV_ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NT
    chunk = pid % NT

    offs_c = tl.arange(0, CHUNK)
    offs_d = tl.arange(0, BLOCK_D)
    t_abs = chunk * CHUNK + offs_c
    mask_t = t_abs < T
    mask_d = offs_d < D

    k_ptrs = k_ptr + bh * T * D + t_abs[:, None] * D + offs_d[None, :]
    k_mat = tl.load(k_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(
        tl.float32
    )

    # Pairwise k_t . k_s.
    kkt = tl.dot(k_mat, tl.trans(k_mat), input_precision="ieee")

    # sum_e k2[t,e] * k2[s,e] * exp(cum_g[t,e] - cum_g[s,e]),
    # needed only for t >= s; masking before exp avoids positive exponents.
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

    # L[t,s] = beta[s] * (k_t.k_s) * fee[t,s], strictly lower triangular.
    strict = offs_c[:, None] > offs_c[None, :]
    lower = tl.where(strict, kkt * fee * beta_vec[None, :], 0.0)
    eye = tl.where(offs_c[:, None] == offs_c[None, :], 1.0, 0.0)
    mat = lower + eye

    # Newton-Schulz inversion of I+L.  L is nilpotent (strictly lower
    # triangular), so this doubles the exact Neumann-series order each step
    # and is exact for a CHUNK x CHUNK block after log2(CHUNK) iterations.
    inv = eye
    for _ in range(INV_ITERS):
        mx = tl.dot(mat, inv, input_precision="ieee")
        inv = tl.dot(inv, 2.0 * eye - mx, input_precision="ieee")

    inv_ptrs = (
        inv_ptr
        + (bh * NT + chunk) * CHUNK * CHUNK
        + offs_c[:, None] * CHUNK
        + offs_c[None, :]
    )
    tl.store(inv_ptrs, inv)


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
    D: tl.constexpr,
    E: tl.constexpr,
    V: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    VB: tl.constexpr,
    NBV: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NBV
    vb = pid % NBV

    offs_c = tl.arange(0, CHUNK)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = vb * VB + tl.arange(0, VB)
    mask_d = offs_d < D
    mask_v = offs_v < V
    row_ge_col = offs_c[:, None] >= offs_c[None, :]

    for chunk in range(NT):
        t_abs = chunk * CHUNK + offs_c
        mask_t = t_abs < T

        k_mat = tl.load(
            k_ptr + bh * T * D + t_abs[:, None] * D + offs_d[None, :],
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        q_mat = tl.load(
            q_ptr + bh * T * D + t_abs[:, None] * D + offs_d[None, :],
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
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

        # Inter-chunk terms.  The state slice for slice e is [d, VB];
        # kappa/qq for that slice are k/q scaled by k2/q2 * exp(cum_g).
        pred = tl.zeros((CHUNK, VB), dtype=tl.float32)
        out_inter = tl.zeros((CHUNK, VB), dtype=tl.float32)
        for e in range(E):
            state_ptrs = (
                state_ptr
                + ((bh * E + e) * D + offs_d[:, None]) * V
                + offs_v[None, :]
            )
            s_e = tl.load(
                state_ptrs, mask=mask_d[:, None] & mask_v[None, :], other=0.0
            )
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

            k_scaled = k_mat * (k2_vec * exp_cum)[:, None]
            pred += tl.dot(k_scaled, s_e, input_precision="ieee")
            q_scaled = q_mat * (q2_vec * exp_cum)[:, None]
            out_inter += tl.dot(q_scaled, s_e, input_precision="ieee")

        # Intra-chunk delta-rule residuals: delta = (I+L)^-1 (v - pred).
        rhs = v_mat - pred
        delta = tl.dot(inv_mat, rhs, input_precision="ieee")

        # R[t,s] = beta[s] * (q_t.k_s) *
        #          sum_e q2[t,e] k2[s,e] exp(cum[t,e]-cum[s,e]), s <= t.
        qk = tl.dot(q_mat, tl.trans(k_mat), input_precision="ieee")
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
        out_mat = (out_inter + tl.dot(r_mat, delta, input_precision="ieee")) * scale
        tl.store(
            out_ptr + bh * T * V + t_abs[:, None] * V + offs_v[None, :],
            out_mat,
            mask=mask_t[:, None] & mask_v[None, :],
        )

        # Carry the state to the next chunk, in the original (decayed) scale:
        # S_e <- exp(cum_last[e]) * S_e
        #        + sum_s beta[s] k_s k2[s,e] exp(cum_last[e]-cum[s,e]) delta_s.
        for e in range(E):
            state_ptrs = (
                state_ptr
                + ((bh * E + e) * D + offs_d[:, None]) * V
                + offs_v[None, :]
            )
            s_e = tl.load(
                state_ptrs, mask=mask_d[:, None] & mask_v[None, :], other=0.0
            )
            g_vec = tl.load(
                g_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            cum = tl.cumsum(g_vec, axis=0)
            cum_last = tl.sum(tl.where(offs_c == CHUNK - 1, cum, 0.0))
            k2_vec = tl.load(
                k2_ptr + bh * T * E + t_abs * E + e, mask=mask_t, other=0.0
            ).to(tl.float32)
            a_vec = beta_vec * k2_vec * tl.exp(cum_last - cum)
            a_mat = k_mat * a_vec[:, None]
            s_new = s_e * tl.exp(cum_last) + tl.dot(
                tl.trans(a_mat), delta, input_precision="ieee"
            )
            tl.store(
                state_ptrs, s_new, mask=mask_d[:, None] & mask_v[None, :]
            )


def _next_pow2(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def kernel_function(q, k, v, q2, k2, g, beta, scale):
    B, H, T, D = q.shape
    V = v.shape[-1]
    E = k2.shape[-1]
    BH = B * H

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if T == 0 or B == 0 or H == 0:
        return out

    # Cheap layout prep only; all real computation is in the Triton kernels.
    q_c = q.contiguous().view(BH, T, D)
    k_c = k.contiguous().view(BH, T, D)
    v_c = v.contiguous().view(BH, T, V)
    q2_c = q2.contiguous().view(BH, T, E)
    k2_c = k2.contiguous().view(BH, T, E)
    g_c = g.contiguous().view(BH, T, E)
    beta_c = beta.contiguous().view(BH, T)
    out_c = out.view(BH, T, V)

    if T >= 64:
        chunk = 64
    else:
        chunk = max(16, _next_pow2(T))
    nt = (T + chunk - 1) // chunk
    block_d = max(16, _next_pow2(D))
    inv_iters = chunk.bit_length() - 1

    if V >= 64:
        vb = 64
    elif V >= 32:
        vb = 32
    else:
        vb = max(16, _next_pow2(V))
    nbv = (V + vb - 1) // vb

    inv = torch.empty((BH, nt, chunk, chunk), device=q.device, dtype=torch.float32)
    # Internal state layout is [BH, E, D, V] (slice-major) so each gate
    # slice is a contiguous [D, V] block for tensor-core dots.
    state = torch.zeros((BH, E, D, V), device=q.device, dtype=torch.float32)

    _prepare_inverse_kernel[(BH * nt,)](
        k_c,
        k2_c,
        g_c,
        beta_c,
        inv,
        T,
        nt,
        D=D,
        E=E,
        CHUNK=chunk,
        BLOCK_D=block_d,
        INV_ITERS=inv_iters,
        num_warps=4,
        num_stages=2,
    )

    try:
        scale_val = float(scale)
    except TypeError:
        scale_val = float(scale.item())

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
        D=D,
        E=E,
        V=V,
        CHUNK=chunk,
        BLOCK_D=block_d,
        VB=vb,
        NBV=nbv,
        num_warps=8,
        num_stages=2,
    )
    return out
```

My approach is the chunkwise form of the gated delta rule: per 64-token chunk I build the strictly-lower interaction matrix using only non-positive gate-difference exponentials (factorised into a $k_t\!\cdot\! k_s$ dot times a 4-slice $k_2$ factor), invert $I+L$ exactly via a short Newton–Schulz series exploiting nilpotency, then a second kernel carries the slice-major $[E,d,V]$ state across chunks with tensor-core dots for inter-chunk prediction/output and the state update. Since I cannot run anything, I am unsure about Triton 3.5 register pressure/spilling in the state kernel at these block sizes and about whether the evaluator's tolerance prefers the IEEE-precision dots I chose over faster TF32.