```python
import torch
import triton
import triton.language as tl


@triton.jit
def _recurrent_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    sq_b, sq_h, sq_t, sq_k,
    sk_b, sk_h, sk_t, sk_k,
    sv_b, sv_h, sv_t, sv_v,
    se_b, se_h, se_t, se_k,
    sg_b, sg_h, sg_t, sg_k,
    sb_b, sb_h, sb_t,
    sgm_b, sgm_h, sgm_t,
    so_b, so_h, so_t, so_v,
    H, T, K, V,
    scale,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    # The true-K mask is the fix: BK is only the padded block size.
    mask_k = offs_k < K
    mask_v = offs_v < V

    q_ptrs = q_ptr + b * sq_b + h * sq_h + offs_k * sq_k
    k_ptrs = k_ptr + b * sk_b + h * sk_h + offs_k * sk_k
    e_ptrs = e_ptr + b * se_b + h * se_h + offs_k * se_k
    g_ptrs = g_ptr + b * sg_b + h * sg_h + offs_k * sg_k
    v_ptrs = v_ptr + b * sv_b + h * sv_h + offs_v * sv_v
    out_ptrs = out_ptr + b * so_b + h * so_h + offs_v * so_v
    beta_pt = beta_ptr + b * sb_b + h * sb_h
    gamma_pt = gamma_ptr + b * sgm_b + h * sgm_h

    # State held transposed, [BV, BK], so every reduction is over K.
    S = tl.zeros((BV, BK), dtype=tl.float32)

    for _t in range(T):
        g_vec = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        S = S * tl.exp(g_vec)[None, :]

        k_vec = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_pt).to(tl.float32)

        pred = tl.sum(S * k_vec[None, :], axis=1)
        S = S + (beta_t * (v_vec - pred))[:, None] * k_vec[None, :]

        e_vec = tl.load(e_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        gamma_t = tl.load(gamma_pt).to(tl.float32)
        proj = tl.sum(S * e_vec[None, :], axis=1)
        S = S - (gamma_t * proj)[:, None] * e_vec[None, :]

        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        out_vec = tl.sum(S * (q_vec * scale)[None, :], axis=1)
        tl.store(out_ptrs, out_vec, mask=mask_v)

        q_ptrs += sq_t
        k_ptrs += sk_t
        v_ptrs += sv_t
        e_ptrs += se_t
        g_ptrs += sg_t
        out_ptrs += so_t
        beta_pt += sb_t
        gamma_pt += sgm_t


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0 or V == 0:
        return out

    BK = max(1, triton.next_power_of_2(max(1, K)))
    if BK <= 128:
        BV = 32
    elif BK <= 256:
        BV = 16
    elif BK <= 512:
        BV = 8
    else:
        BV = 4
    BV = min(BV, max(1, triton.next_power_of_2(V)))
    bh = B * H
    while BV > 4 and bh * ((V + BV - 1) // BV) < 64:
        BV //= 2
    num_warps = 8 if BV * BK >= 4096 else (4 if BV * BK >= 1024 else 2)

    grid = (bh, (V + BV - 1) // BV)
    _recurrent_kernel[grid](
        q, k, v, e, g, beta, gamma, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        e.stride(0), e.stride(1), e.stride(2), e.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H, T, K, V,
        float(scale),
        BK=BK,
        BV=BV,
        num_warps=num_warps,
    )
    return out
```

The failing cases violate the previous kernel's implicit assumption that `K` is a power of two: it set `mask_k = offs_k < BK`, where `BK` is the padded power-of-two size, so for `K=65` or `129` the padded lanes loaded elements from neighbouring rows and corrupted the state. The fix passes the true `K` into the kernel and masks all `q/k/e/g` loads with `offs_k < K`, using zero padding, which leaves the padded state lanes permanently zero and does not affect any reduction.