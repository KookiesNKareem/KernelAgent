```python
import torch
import triton
import triton.language as tl


@triton.jit
def _scan_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, o_ptr,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, scale,
    BLOCK: tl.constexpr, EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    k_mask = idx < K
    v_mask = idx < V

    S = tl.zeros((BLOCK, BLOCK), tl.float32)

    q_base = pid * T * K
    k_base = pid * T * K
    e_base = pid * T * K
    d_base = pid * T * K
    v_base = pid * T * V
    o_base = pid * T * V
    b_base = pid * T

    for t in tl.range(T):
        bt = tl.load(beta_ptr + b_base + t)
        ct = tl.load(gamma_ptr + b_base + t)
        if EVEN:
            kt = tl.load(k_ptr + k_base + t * K + idx, eviction_policy='evict_first').to(tl.float32)
            et = tl.load(e_ptr + e_base + t * K + idx, eviction_policy='evict_first').to(tl.float32)
            dt = tl.load(d_ptr + d_base + t * K + idx, eviction_policy='evict_first')
            qt = tl.load(q_ptr + q_base + t * K + idx, eviction_policy='evict_first').to(tl.float32)
            vt = tl.load(v_ptr + v_base + t * V + idx, eviction_policy='evict_first').to(tl.float32)
        else:
            kt = tl.load(k_ptr + k_base + t * K + idx, mask=k_mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
            et = tl.load(e_ptr + e_base + t * K + idx, mask=k_mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
            dt = tl.load(d_ptr + d_base + t * K + idx, mask=k_mask, other=1.0, eviction_policy='evict_first')
            qt = tl.load(q_ptr + q_base + t * K + idx, mask=k_mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
            vt = tl.load(v_ptr + v_base + t * V + idx, mask=v_mask, other=0.0, eviction_policy='evict_first').to(tl.float32)

        S = S * dt[:, None]

        a = tl.sum(S * et[:, None], axis=0)
        S = S - ct * et[:, None] * a[None, :]

        b = tl.sum(S * et[None, :], axis=1)
        S = S - ct * b[:, None] * et[None, :]

        pred = tl.sum(S * kt[:, None], axis=0)
        diff = (vt - pred) * bt
        S = S + kt[:, None] * diff[None, :]

        out = tl.sum(S * (qt * scale)[:, None], axis=0)
        if EVEN:
            tl.store(o_ptr + o_base + t * V + idx, out)
        else:
            tl.store(o_ptr + o_base + t * V + idx, out, mask=v_mask)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    scale_f = float(scale.item()) if torch.is_tensor(scale) else float(scale)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    e = e.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    gamma = gamma.contiguous()

    d = torch.exp(g)

    m = K if K > V else V
    BLOCK = 1 << (m - 1).bit_length()
    if BLOCK < 16:
        BLOCK = 16
    EVEN = (K == BLOCK and V == BLOCK)

    o = torch.empty((B, H, T, V), dtype=torch.float32, device=q.device)

    BH = B * H
    _scan_kernel[(BH,)](
        q, k, v, e, d, beta, gamma, o,
        T, K, V, scale_f, BLOCK, EVEN,
        num_warps=8, num_stages=3,
    )
    return o
```
Same fused scan math as before but looped with tl.range to avoid unrolling 2048 copies, precomputed exp(g) with torch.exp, mask-free fast path and evict-first streaming loads plus 3-stage software pipelining. I am unsure about exact I-cache vs barrier share of the 18ms and whether 8 warps plus 3 stages is optimal for 128x128 fp32 state without spilling.
