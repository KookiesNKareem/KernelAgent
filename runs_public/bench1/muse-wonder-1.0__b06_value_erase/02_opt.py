import torch
import triton
import triton.language as tl


@triton.jit
def _recurrent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    out_ptr,
    T,
    scale,
    DK,
    DV,
    DE,
    DG,
    BK: tl.constexpr,
    BV: tl.constexpr,
    EXACT: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)

    mask_k = offs_k < DK
    mask_v = offs_v < DV
    mask_e = offs_v < DE

    q_p = q_ptr + bh * T * DK
    k_p = k_ptr + bh * T * DK
    v_p = v_ptr + bh * T * DV
    e_p = e_ptr + bh * T * DE
    g_p = g_ptr + bh * T * DG
    beta_p = beta_ptr + bh * T
    gamma_p = gamma_ptr + bh * T
    out_p = out_ptr + bh * T * DV

    # S in [K, V] orientation (for S @ e) and, redundantly, in [V, K]
    # orientation (for S^T @ k and S^T @ q).  Keeping both orientations
    # makes every large reduction run along axis 1 of its block, avoiding
    # the expensive transposed/layout-changing reductions in the naive
    # single-state version.  Both copies are updated with the same
    # rank-1 terms (transposed), so they represent the same matrix.
    S = tl.zeros([BK, BV], dtype=tl.float32)
    ST = tl.zeros([BV, BK], dtype=tl.float32)

    for _ in range(T):
        if EXACT:
            k_vec = tl.load(k_p + offs_k).to(tl.float32)
            v_vec = tl.load(v_p + offs_v).to(tl.float32)
            e_vec = tl.load(e_p + offs_v).to(tl.float32)
            g_vec = tl.load(g_p + offs_k).to(tl.float32)
            q_vec = tl.load(q_p + offs_k).to(tl.float32)
        else:
            k_vec = tl.load(k_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            v_vec = tl.load(v_p + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            e_vec = tl.load(e_p + offs_v, mask=mask_e, other=0.0).to(tl.float32)
            g_vec = tl.load(g_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            q_vec = tl.load(q_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        beta_s = tl.load(beta_p)
        gamma_s = tl.load(gamma_p)
        qq = q_vec * scale
        decay = tl.exp(g_vec)

        # Decay + value-side erase, reference order: u uses decayed S.
        S = S * decay[:, None]
        u = tl.sum(S * e_vec[None, :], axis=1)          # [K]
        S = S - (gamma_s * u)[:, None] * e_vec[None, :]

        ST = ST * decay[None, :]
        ST = ST - (gamma_s * e_vec)[:, None] * u[None, :]

        # Prediction and M^T q from the erased state M (ST == M^T).
        pred = tl.sum(ST * k_vec[None, :], axis=1)      # [V]
        mq = tl.sum(ST * qq[None, :], axis=1)           # [V]
        resid = beta_s * (v_vec - pred)                 # [V]

        # Delta update of both copies.
        S = S + k_vec[:, None] * resid[None, :]
        ST = ST + resid[:, None] * k_vec[None, :]

        # o = S_new^T qq = M^T qq + resid * (k . qq); no second
        # pass over the updated state is needed.
        out_vec = mq + resid * tl.sum(k_vec * qq)

        if EXACT:
            tl.store(out_p + offs_v, out_vec)
        else:
            tl.store(out_p + offs_v, out_vec, mask=mask_v)

        q_p += DK
        k_p += DK
        v_p += DV
        e_p += DE
        g_p += DG
        beta_p += 1
        gamma_p += 1
        out_p += DV


def _pow2_at_least_16(n):
    x = 16
    while x < n:
        x *= 2
    return x


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    DE = e.shape[-1]
    DG = g.shape[-1]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    e = e.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    gamma = gamma.contiguous()

    out = torch.empty((B, H, T, DV), device=q.device, dtype=torch.float32)

    BH = B * H
    if BH == 0 or T == 0:
        return out

    BK = _pow2_at_least_16(DK)
    BV = _pow2_at_least_16(DV)
    exact = (BK == DK) and (BV == DV) and (BV == DE)

    elems = BK * BV
    if elems >= 16384:
        num_warps = 32
    elif elems >= 4096:
        num_warps = 16
    elif elems >= 1024:
        num_warps = 8
    else:
        num_warps = 4

    _recurrent_kernel[(BH,)](
        q,
        k,
        v,
        e,
        g,
        beta,
        gamma,
        out,
        T,
        float(scale),
        DK,
        DV,
        DE,
        DG,
        BK,
        BV,
        exact,
        num_warps=num_warps,
    )
    return out
