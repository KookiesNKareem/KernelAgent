```python
import torch
import triton
import triton.language as tl


@triton.jit
def _scan_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    T, K, V,
    scale,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    vb = tl.program_id(0)
    bh = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    v_offs = vb * BLOCK_V + offs_v
    mask_v = v_offs < V

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    g_base = g_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    o_base = out_ptr + bh * T * V
    beta_base = beta_ptr + bh * T
    gamma_base = gamma_ptr + bh * T

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in range(T):
        k_t = tl.load(k_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        e_t = tl.load(e_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        g_t = tl.load(g_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        q_t = tl.load(q_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_t = tl.load(v_base + t * V + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        b = tl.load(beta_base + t).to(tl.float32)
        gm = tl.load(gamma_base + t).to(tl.float32)
        c = gm * (1.0 - b)

        d_t = tl.exp(g_t)

        # S = diag(d) S
        d_col = tl.expand_dims(d_t, 1)
        S = S * d_col

        # S = (I - c e e^T) S
        e_col = tl.expand_dims(e_t, 1)
        s_e = tl.sum(S * e_col, axis=0)
        S = S - c * e_col * tl.expand_dims(s_e, 0)

        # pred = k^T S ; S += b k (v - pred)^T
        k_col = tl.expand_dims(k_t, 1)
        pred = tl.sum(S * k_col, axis=0)
        delta = b * (v_t - pred)
        S = S + k_col * tl.expand_dims(delta, 0)

        # o = scale * S^T q
        q_col = tl.expand_dims(q_t * scale, 1)
        o_t = tl.sum(S * q_col, axis=0)
        tl.store(o_base + t * V + v_offs, o_t, mask=mask_v)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    Vdim = v.shape[-1]
    if T == 0 or K == 0 or Vdim == 0:
        return torch.empty(B, H, T, Vdim, device=q.device, dtype=torch.float32)
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.detach().to(torch.float32).item())
    else:
        scale_f = float(scale)

    BH = B * H
    q_ = q.contiguous().view(BH, T, K)
    k_ = k.contiguous().view(BH, T, K)
    e_ = e.contiguous().view(BH, T, K)
    g_ = g.contiguous().view(BH, T, K)
    v_ = v.contiguous().view(BH, T, Vdim)
    beta_ = beta.contiguous().view(BH, T)
    gamma_ = gamma.contiguous().view(BH, T)

    out = torch.empty(BH, T, Vdim, device=q.device, dtype=torch.float32)

    BK = triton.next_power_of_2(K)
    if BK < 16:
        BK = 16
    if Vdim >= 32:
        BV = 32
    else:
        BV = triton.next_power_of_2(Vdim)
        if BV < 16:
            BV = 16
    while BK * BV > 4096 and BV > 8:
        BV //= 2

    n_vb = (Vdim + BV - 1) // BV
    grid = (n_vb, BH)
    num_warps = 8 if BK * BV > 2048 else 4

    _scan_kernel[grid](
        q_, k_, v_, e_, g_, beta_, gamma_, out,
        T, K, Vdim,
        scale_f,
        BLOCK_K=BK, BLOCK_V=BV,
        num_warps=num_warps,
    )
    return out.view(B, H, T, Vdim)
```
My approach fuses the whole time recurrence into one Triton kernel that keeps the KxV state in registers, loops sequentially over T, and parallelizes over flattened batch-heads and V-shards. All math is done in fp32 after casting bf16 inputs to match the reference promotion and ordering. I am unsure about ideal BLOCK_V and num_warps occupancy on sm_89 and whether tl.sum reduction order and tl.exp bit-exactness will meet a tight tolerance, plus handling of very large K beyond 256.
