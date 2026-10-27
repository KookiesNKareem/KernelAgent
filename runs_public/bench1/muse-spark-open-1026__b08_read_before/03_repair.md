```python
import torch
import triton
import triton.language as tl

# fast resident S = K x BV in registers, 1 launch, no HBM spill per step
@triton.jit
def fast_resident_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr,
    BH, T, K, V,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, SCALE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_vb = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_offs = pid_vb * BLOCK_V + offs_v
    mask_v = v_offs < V
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    for t in range(0, T):
        base_qk = (pid_bh * T + t) * K
        base_v = (pid_bh * T + t) * V
        q = tl.load(q_ptr + base_qk + offs_k).to(tl.float32) * SCALE
        k = tl.load(k_ptr + base_qk + offs_k).to(tl.float32)
        g = tl.load(g_ptr + base_qk + offs_k).to(tl.float32)
        d = tl.exp(g)
        v = tl.load(v_ptr + base_v + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + pid_bh * T + t).to(tl.float32)
        S = S * d[:, None]
        out = tl.sum(S * q[:, None], axis=0)
        pred = tl.sum(S * k[:, None], axis=0)
        tl.store(out_ptr + base_v + v_offs, out, mask=mask_v)
        delta = (v - pred) * beta
        S = S + k[:, None] * delta[None, :]

# generic correct fallback: two-pass per step, S in HBM, handles arbitrary K/V and strides
@triton.jit
def generic_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr, state_ptr,
    BH, T, K, V, NK,
    BK: tl.constexpr, BV: tl.constexpr, SCALE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_vb = tl.program_id(1)
    offs_v = tl.arange(0, BV)
    v_offs = pid_vb * BV + offs_v
    mask_v = v_offs < V
    for t in range(0, T):
        beta = tl.load(beta_ptr + pid_bh * T + t).to(tl.float32)
        v = tl.load(v_ptr + (pid_bh * T + t) * V + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        # phase A : decay and accumulate pred/out
        pred = tl.zeros((BV,), dtype=tl.float32)
        outv = tl.zeros((BV,), dtype=tl.float32)
        for ki in range(NK):
            k_offs = ki * BK + tl.arange(0, BK)
            mask_k = k_offs < K
            # 2D state pointers
            k2d = k_offs[:, None]
            v2d = v_offs[None, :]
            s_ptr = state_ptr + pid_bh * K * V + k2d * V + v2d
            mask2d = (k2d < K) & (v2d < V)
            S = tl.load(s_ptr, mask=mask2d, other=0.0)
            qk = tl.load(q_ptr + (pid_bh * T + t) * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
            kk = tl.load(k_ptr + (pid_bh * T + t) * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
            gg = tl.load(g_ptr + (pid_bh * T + t) * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
            d = tl.exp(gg)
            S = S * d[:, None]
            tl.store(s_ptr, S, mask=mask2d)
            pred += tl.sum(S * kk[:, None], axis=0)
            outv += tl.sum(S * (qk * SCALE)[:, None], axis=0)
        tl.store(out_ptr + (pid_bh * T + t) * V + v_offs, outv, mask=mask_v)
        delta = (v - pred) * beta
        for ki in range(NK):
            k_offs = ki * BK + tl.arange(0, BK)
            k2d = k_offs[:, None]
            v2d = v_offs[None, :]
            s_ptr = state_ptr + pid_bh * K * V + k2d * V + v2d
            mask2d = (k2d < K) & (v2d < V)
            S = tl.load(s_ptr, mask=mask2d, other=0.0)
            kk = tl.load(k_ptr + (pid_bh * T + t) * K + k_offs, mask=k_offs < K, other=0.0).to(tl.float32)
            S = S + kk[:, None] * delta[None, :]
            tl.store(s_ptr, S, mask=mask2d)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e: [B,H,T,K] bf16, v: [B,H,T,V] bf16, g: [B,H,T,K] fp32, beta,gamma: [B,H,T] fp32
    # e,gamma unused
    B, H, T, K = q.shape
    V = v.shape[-1]
    BH = B * H
    device = q.device
    scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)

    # make logical contiguous so simple flattened offsets work for all strided/transposed views
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = g.contiguous()
    beta_c = beta.contiguous()

    out = torch.empty((B, H, T, V), dtype=torch.float32, device=device)

    # fast path: exact K=128 where resident 128 x BV fits in registers, covers timed B=4 H=8 T=2048
    if K == 128 and V % 32 == 0:
        if V % 64 == 0:
            BV = 64
        else:
            BV = 32
        BK = 128
        grid = (BH, V // BV)
        fast_resident_kernel[grid](q_c, k_c, v_c, g_c, beta_c, out,
                                   BH, T, K, V, BK, BV, scale_f,
                                   num_warps=4, num_stages=2)
        return out

    # generic fallback for s2=65/129 etc.
    BK = 32
    BV = 32
    NK = (K + BK - 1) // BK
    NV = (V + BV - 1) // BV
    state = torch.zeros((BH, K, V), dtype=torch.float32, device=device)
    grid = (BH, NV)
    generic_kernel[grid](q_c, k_c, v_c, g_c, beta_c, out, state,
                         BH, T, K, V, NK, BK, BV, scale_f,
                         num_warps=2, num_stages=1)
    return out
```