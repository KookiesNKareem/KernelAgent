```python
import torch
import triton
import triton.language as tl


@triton.jit
def _delta_gated_kernel(
    Qptr, Kptr, Vptr, Optr, Gptr, Bptr, Sptr,
    scale,
    BH, T, Kdim, Vdim,
    chunk_start,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < Kdim
    v_cols = pid_v * BLOCK_V + offs_v
    mask_v = v_cols < Vdim

    # state shard [BLOCK_K, BLOCK_V]
    s_ptrs = Sptr + pid_bh * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
    s_mask = mask_k[:, None] & mask_v[None, :]
    S = tl.load(s_ptrs, mask=s_mask, other=0.0)

    q_bh_off = pid_bh * T * Kdim
    k_bh_off = pid_bh * T * Kdim
    v_bh_off = pid_bh * T * Vdim
    o_bh_off = pid_bh * T * Vdim
    g_bh_off = pid_bh * T
    b_bh_off = pid_bh * T

    for t_off in range(BLOCK_T):
        t = chunk_start + t_off
        q_ptrs = Qptr + q_bh_off + t * Kdim + offs_k
        k_ptrs = Kptr + k_bh_off + t * Kdim + offs_k
        v_ptrs = Vptr + v_bh_off + t * Vdim + v_cols
        o_ptrs = Optr + o_bh_off + t * Vdim + v_cols
        g_ptr = Gptr + g_bh_off + t
        b_ptr = Bptr + b_bh_off + t

        k_vec = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
        g_val = tl.load(g_ptr).to(tl.float32)
        beta_val = tl.load(b_ptr).to(tl.float32)

        decay = tl.exp(g_val)
        S = S * decay

        k_exp = k_vec[:, None]
        pred = tl.sum(S * k_exp, axis=0)
        delta = (v_vec - pred) * beta_val
        S = S + k_exp * delta[None, :]

        q_exp = (q_vec * scale)[:, None]
        out_vec = tl.sum(S * q_exp, axis=0)
        tl.store(o_ptrs, out_vec, mask=mask_v)

    tl.store(s_ptrs, S, mask=s_mask)


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # reference order/semantics, e/gamma unused in math
    B, H, T, K = q.shape
    Vdim = v.shape[-1]
    if T == 0:
        return torch.empty(B, H, 0, Vdim, dtype=torch.float32, device=q.device)
    if isinstance(scale, torch.Tensor):
        scale = float(scale.detach().to(torch.float32).item())
    else:
        scale = float(scale)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    out = torch.empty(B, H, T, Vdim, dtype=torch.float32, device=q.device)
    BH = B * H
    q_ = q.reshape(BH, T, K)
    k_ = k.reshape(BH, T, K)
    v_ = v.reshape(BH, T, Vdim)
    out_ = out.reshape(BH, T, Vdim)
    g_ = g.reshape(BH, T)
    beta_ = beta.reshape(BH, T)
    state = torch.zeros(BH, K, Vdim, dtype=torch.float32, device=q.device)

    BK = _next_pow2(K)
    BV_full = _next_pow2(Vdim)
    if BK * BV_full <= 4096:
        BV = BV_full
    else:
        tgt = 4096 // BK
        if tgt < 1:
            tgt = 1
        BV = 1 << (tgt.bit_length() - 1)
    NV = (Vdim + BV - 1) // BV
    grid = (BH, NV)

    CHUNK = 64
    full = T // CHUNK
    rem = T % CHUNK
    for ci in range(full):
        cs = ci * CHUNK
        _delta_gated_kernel[grid](
            q_, k_, v_, out_, g_, beta_, state,
            scale, BH, T, K, Vdim, cs,
            BLOCK_K=BK, BLOCK_V=BV, BLOCK_T=CHUNK,
            num_warps=4,
        )
    if rem != 0:
        cs = full * CHUNK
        _delta_gated_kernel[grid](
            q_, k_, v_, out_, g_, beta_, state,
            scale, BH, T, K, Vdim, cs,
            BLOCK_K=BK, BLOCK_V=BV, BLOCK_T=rem,
            num_warps=4,
        )
    if T <= CHUNK and full == 0 and rem == 0:
        pass
    return out
```

The recurrence is column-parallel in V, so I shard V over Triton programs and keep the full K×shard state in registers. Each program streams T sequentially, which needs no triangular solve and only HBM traffic for Q/K/V/O. To avoid a 2048-iteration dynamic loop I process 64-step static-unrolled chunks with the state carried in global memory across Python-loop launches.

I am unsure about Triton 3.5 handling of scalar `scale` dtype promotion, unrolled 64×128×32 IR size/compile time, and K>1024 or very small/odd K,V where the BK×BV≤4096 heuristic and masking may be suboptimal but should remain correct.