```python
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

    # The whole [K, V] state lives in registers for the entire sequence.
    S = tl.zeros([BK, BV], dtype=tl.float32)

    for _ in range(T):
        if EXACT:
            k_vec = tl.load(k_p + offs_k).to(tl.float32)
            v_vec = tl.load(v_p + offs_v).to(tl.float32)
            e_vec = tl.load(e_p + offs_v).to(tl.float32)
            g_vec = tl.load(g_p + offs_k).to(tl.float32)
        else:
            k_vec = tl.load(k_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            v_vec = tl.load(v_p + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            e_vec = tl.load(e_p + offs_v, mask=mask_e, other=0.0).to(tl.float32)
            g_vec = tl.load(g_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        beta_s = tl.load(beta_p)
        gamma_s = tl.load(gamma_p)

        # diag(exp(g_t)) S
        S = S * tl.exp(g_vec)[:, None]

        # S (I - gamma_t e_t e_t^T)
        u = tl.sum(S * e_vec[None, :], axis=1)
        S = S - gamma_s * u[:, None] * e_vec[None, :]

        # (I - beta_t k_t k_t^T) S + beta_t k_t v_t^T,
        # written in the reference's equivalent innovation form.
        pred = tl.sum(S * k_vec[:, None], axis=0)
        resid = v_vec - pred
        S = S + k_vec[:, None] * (beta_s * resid)[None, :]

        if EXACT:
            q_vec = tl.load(q_p + offs_k).to(tl.float32)
        else:
            q_vec = tl.load(q_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        out_vec = tl.sum(S * (q_vec * scale)[:, None], axis=0)

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
    if elems >= 65536:
        num_warps = 32
    elif elems >= 16384:
        num_warps = 16
    elif elems >= 4096:
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
```

The implementation uses one Triton program per `(batch, head)`, keeps the full fp32 state in registers across the whole time loop, and performs the decay, value-side erase, delta update, and output projection in the reference's exact order. This eliminates the reference's per-timestep kernel-launch overhead while reading each input only once. I could not test it, so the main uncertainty is whether `num_warps=16` is the optimal setting for the 128×128 benchmark tile on the L4, and very large non-power-of-two states could become register-constrained.