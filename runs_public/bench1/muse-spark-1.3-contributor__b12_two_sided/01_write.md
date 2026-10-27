```python
import torch
import triton
import triton.language as tl


@triton.jit
def _scan_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, scale, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    k_mask = idx < K
    v_mask = idx < V

    S = tl.zeros((BLOCK, BLOCK), tl.float32)

    q_base = pid * T * K
    k_base = pid * T * K
    e_base = pid * T * K
    g_base = pid * T * K
    v_base = pid * T * V
    o_base = pid * T * V
    b_base = pid * T

    for t in range(T):
        kt = tl.load(k_ptr + k_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        et = tl.load(e_ptr + e_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        gt = tl.load(g_ptr + g_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        qt = tl.load(q_ptr + q_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        vt = tl.load(v_ptr + v_base + t * V + idx, mask=v_mask, other=0.0).to(tl.float32)
        bt = tl.load(beta_ptr + b_base + t).to(tl.float32)
        ct = tl.load(gamma_ptr + b_base + t).to(tl.float32)

        dg = tl.exp(gt)
        S = S * dg[:, None]

        # left erase: S = (I - c e e^T) S
        a = tl.sum(S * et[:, None], axis=0)
        S = S - ct * et[:, None] * a[None, :]

        # right erase: S = S (I - c e e^T), needs K==V like reference
        b = tl.sum(S * et[None, :], axis=1)
        S = S - ct * b[:, None] * et[None, :]

        # delta: S = (I - b k k^T) S + b k v^T
        pred = tl.sum(S * kt[:, None], axis=0)
        diff = (vt - pred) * bt
        S = S + kt[:, None] * diff[None, :]

        out = tl.sum(S * (qt * scale)[:, None], axis=0)
        tl.store(o_ptr + o_base + t * V + idx, out, mask=v_mask)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    if torch.is_tensor(scale):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    e = e.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    gamma = gamma.contiguous()

    m = K if K > V else V
    BLOCK = 1 << (m - 1).bit_length()
    if BLOCK < 16:
        BLOCK = 16

    o = torch.empty((B, H, T, V), dtype=torch.float32, device=q.device)

    BH = B * H
    num_warps = 8 if BLOCK >= 64 else 4
    _scan_kernel[(BH,)](q, k, v, e, g, beta, gamma, o, T, K, V, scale_f, BLOCK, num_warps=num_warps, num_stages=1)
    return o
```
Single fused scan keeps the KxV state in registers on chip and streams q,k,v,e,g,beta,gamma once, eliminating the Python loop overhead. Decay, left/right erase and delta are done as row scaling plus rank-1 updates with reductions, matching the reference order. I am unsure about register pressure / compile success of a 128x128 fp32 tile with 8 warps and whether Triton keeps the 2048 trip loop as scf.for instead of unrolling.
