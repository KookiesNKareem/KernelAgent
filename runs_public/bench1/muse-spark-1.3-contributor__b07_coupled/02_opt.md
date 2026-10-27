```python
import torch
import triton
import triton.language as tl

@triton.jit
def _scan_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    T, K, Vd,
    scale,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_V: tl.constexpr,
):
    bh = tl.program_id(1)
    vb = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_offs = vb * BLOCK_V + offs_v

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    g_base = g_ptr + bh * T * K
    v_base = v_ptr + bh * T * Vd
    o_base = out_ptr + bh * T * Vd
    beta_base = beta_ptr + bh * T
    gamma_base = gamma_ptr + bh * T

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in range(T):
        if EVEN_K:
            k_t = tl.load(k_base + t * K + offs_k, eviction_policy='evict_first').to(tl.float32)
            e_t = tl.load(e_base + t * K + offs_k, eviction_policy='evict_first').to(tl.float32)
            g_t = tl.load(g_base + t * K + offs_k, eviction_policy='evict_first').to(tl.float32)
            q_t = tl.load(q_base + t * K + offs_k, eviction_policy='evict_first').to(tl.float32)
        else:
            m_k = offs_k < K
            k_t = tl.load(k_base + t * K + offs_k, mask=m_k, other=0.0).to(tl.float32)
            e_t = tl.load(e_base + t * K + offs_k, mask=m_k, other=0.0).to(tl.float32)
            g_t = tl.load(g_base + t * K + offs_k, mask=m_k, other=0.0).to(tl.float32)
            q_t = tl.load(q_base + t * K + offs_k, mask=m_k, other=0.0).to(tl.float32)
        if EVEN_V:
            v_t = tl.load(v_base + t * Vd + v_offs, eviction_policy='evict_first').to(tl.float32)
        else:
            m_v = v_offs < Vd
            v_t = tl.load(v_base + t * Vd + v_offs, mask=m_v, other=0.0).to(tl.float32)

        b = tl.load(beta_base + t).to(tl.float32)
        gm = tl.load(gamma_base + t).to(tl.float32)
        c = gm * (1.0 - b)

        d_t = tl.exp(g_t)
        qs = q_t * scale
        e1 = d_t * e_t
        k1 = d_t * k_t
        qd = d_t * qs

        ke = tl.sum(k_t * e_t, axis=0)
        eq = tl.sum(e_t * qs, axis=0)
        kq = tl.sum(k_t * qs, axis=0)

        s_e = tl.sum(S * tl.expand_dims(e1, 1), axis=0)
        p1 = tl.sum(S * tl.expand_dims(k1, 1), axis=0)
        r1 = tl.sum(S * tl.expand_dims(qd, 1), axis=0)

        pred = p1 - c * ke * s_e
        delta = b * (v_t - pred)
        o_t = r1 - c * eq * s_e + kq * delta

        S = S * tl.expand_dims(d_t, 1) - c * tl.expand_dims(e_t, 1) * tl.expand_dims(s_e, 0) + tl.expand_dims(k_t, 1) * tl.expand_dims(delta, 0)

        if EVEN_V:
            tl.store(o_base + t * Vd + v_offs, o_t)
        else:
            tl.store(o_base + t * Vd + v_offs, o_t, mask=m_v)

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    if T == 0 or K == 0 or Vd == 0:
        return torch.empty(B, H, T, Vd, device=q.device, dtype=torch.float32)
    scale_f = float(scale.detach().to(torch.float32).item()) if isinstance(scale, torch.Tensor) else float(scale)
    BH = B * H
    q_ = q.contiguous().view(BH, T, K)
    k_ = k.contiguous().view(BH, T, K)
    e_ = e.contiguous().view(BH, T, K)
    g_ = g.contiguous().view(BH, T, K)
    v_ = v.contiguous().view(BH, T, Vd)
    beta_ = beta.contiguous().view(BH, T)
    gamma_ = gamma.contiguous().view(BH, T)
    out = torch.empty(BH, T, Vd, device=q.device, dtype=torch.float32)

    BK = triton.next_power_of_2(K)
    if BK < 16:
        BK = 16
    if Vd <= 32:
        BV = triton.next_power_of_2(Vd)
        if BV < 16:
            BV = 16
    else:
        BV = 64
    while BK * BV > 8192 and BV > 16:
        BV //= 2
    EVEN_K = (K == BK)
    EVEN_V = (Vd % BV == 0)
    n_vb = (Vd + BV - 1) // BV
    grid = (n_vb, BH)
    if BK * BV <= 4096:
        nw = 4
    else:
        nw = 8
    _scan_kernel[grid](
        q_, k_, v_, e_, g_, beta_, gamma_, out,
        T, K, Vd,
        scale_f,
        BLOCK_K=BK, BLOCK_V=BV,
        EVEN_K=EVEN_K, EVEN_V=EVEN_V,
        num_warps=nw, num_stages=2,
    )
    return out.view(B, H, T, Vd)
```