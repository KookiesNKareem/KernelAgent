```python
import torch
import triton
import triton.language as tl


@triton.jit
def _scan_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, out_ptr,
    T, K, V,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    mask_k = off_k < K
    mask_v = off_v < V
    mask_ev = off_v < K

    q_base = q_ptr + pid * T * K
    k_base = k_ptr + pid * T * K
    e_base = e_ptr + pid * T * K
    d_base = d_ptr + pid * T * K
    v_base = v_ptr + pid * T * V
    b_base = beta_ptr + pid * T
    c_base = gamma_ptr + pid * T
    o_base = out_ptr + pid * T * V

    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    for t in range(T):
        q_t = tl.load(q_base + t * K + off_k, mask=mask_k, other=0.0)
        k_t = tl.load(k_base + t * K + off_k, mask=mask_k, other=0.0)
        d_t = tl.load(d_base + t * K + off_k, mask=mask_k, other=1.0)
        e_t = tl.load(e_base + t * K + off_v, mask=mask_ev, other=0.0)
        v_t = tl.load(v_base + t * V + off_v, mask=mask_v, other=0.0)
        b_t = tl.load(b_base + t)
        c_t = tl.load(c_base + t)

        # diag(exp g) on the left : row scale
        S = S * d_t[:, None]

        # right erase S <- S (I - gamma e e^T), e used as V-dim
        # u = S e  (sum over V)
        u_t = tl.sum(S * e_t[None, :], axis=1)
        S = S - c_t * u_t[:, None] * e_t[None, :]

        # left delta: pred = k^T S
        pred_t = tl.sum(S * k_t[:, None], axis=0)
        diff_t = v_t - pred_t
        S = S + b_t * k_t[:, None] * diff_t[None, :]

        o_t = tl.sum(S * q_t[:, None], axis=0)
        tl.store(o_base + t * V + off_v, o_t, mask=mask_v)


def _npo2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    if torch.is_tensor(scale):
        scale_f = float(scale.reshape(-1)[0].to(torch.float32).item())
    else:
        scale_f = float(scale)

    # cheap elementwise prep only: fp32 cast, scale, exp
    q32 = (q.to(torch.float32) * scale_f).contiguous()
    k32 = k.to(torch.float32).contiguous()
    v32 = v.to(torch.float32).contiguous()
    e32 = e.to(torch.float32).contiguous()
    d32 = torch.exp(g.to(torch.float32)).contiguous()
    b32 = beta.to(torch.float32).contiguous()
    c32 = gamma.to(torch.float32).contiguous()

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    BH = B * H
    if BH == 0 or T == 0 or K == 0 or V == 0:
        return out

    BLOCK_K = max(16, _npo2(K))
    BLOCK_V = max(16, _npo2(V))

    _scan_kernel[(BH,)](
        q32, k32, v32, e32, d32, b32, c32, out,
        T, K, V,
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        num_warps=8,
    )
    return out
```
My approach fuses the whole time loop into one Triton kernel per BH lane keeping the KxV state in registers so no Python per-step overhead and no HBM traffic for S, with exp/scale/casts done once in the wrapper. I assume the reference is only well-defined for square K==V and load e as a V-vector with zero padding otherwise. I am unsure whether a 128x128 fp32 tile plus temporaries will fit registers without spilling on sm_89 and whether tl.sum reduction order will stay within the grader's tolerance.