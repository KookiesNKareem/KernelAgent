import torch
import triton
import triton.language as tl


@triton.jit
def _seq_kernel(
    q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, o_ptr,
    T, K, V, scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, NUM_VB: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NUM_VB
    vb = pid % NUM_VB
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = k_offs < K
    mask_v = v_offs < V

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in range(T):
        base_qk = (bh * T + t) * K
        base_v = (bh * T + t) * V
        base_b = bh * T + t

        k_t = tl.load(k_ptr + base_qk + k_offs, mask=mask_k, other=0.0)
        q_t = tl.load(q_ptr + base_qk + k_offs, mask=mask_k, other=0.0)
        d_t = tl.load(d_ptr + base_qk + k_offs, mask=mask_k, other=0.0)
        beta_t = tl.load(beta_ptr + base_b)
        v_t = tl.load(v_ptr + base_v + v_offs, mask=mask_v, other=0.0)

        k_f = k_t.to(tl.float32)
        q_f = q_t.to(tl.float32)
        v_f = v_t.to(tl.float32)

        S = S * tl.expand_dims(d_t, 1)

        k_col = tl.expand_dims(k_f, 1)
        pred = tl.sum(S * k_col, axis=0)

        err = (v_f - pred) * beta_t
        S = S + k_col * tl.expand_dims(err, 0)

        q_scaled = q_f * scale
        out_t = tl.sum(S * tl.expand_dims(q_scaled, 1), axis=0)
        tl.store(o_ptr + base_v + v_offs, out_t, mask=mask_v)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    _ = (e, gamma)
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    BH = B * H

    if isinstance(scale, torch.Tensor):
        scale_val = float(scale.item())
    else:
        scale_val = float(scale)

    if T == 0 or K == 0 or Vd == 0 or BH == 0:
        return torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)

    q_ = q.reshape(BH, T, K).contiguous()
    k_ = k.reshape(BH, T, K).contiguous()
    v_ = v.reshape(BH, T, Vd).contiguous()
    g_ = g.reshape(BH, T, K).contiguous()
    beta_ = beta.reshape(BH, T).contiguous()

    # cheap elementwise prep: exp once, not once per V-shard
    d_ = torch.exp(g_)

    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_V = triton.next_power_of_2(Vd)
    if BLOCK_K < 16:
        BLOCK_K = 16
    if BLOCK_V < 16:
        BLOCK_V = 16
    if BLOCK_V > 128:
        BLOCK_V = 128
    while BLOCK_K * BLOCK_V > 16384 and BLOCK_V > 16:
        BLOCK_V //= 2

    NUM_VB = (Vd + BLOCK_V - 1) // BLOCK_V

    out = torch.empty((BH, T, Vd), device=q.device, dtype=torch.float32)
    grid = (BH * NUM_VB,)
    num_warps = 8 if BLOCK_K * BLOCK_V > 4096 else 4
    _seq_kernel[grid](
        q_, k_, v_, d_, beta_, out,
        T, K, Vd, scale_val,
        BLOCK_K, BLOCK_V, NUM_VB,
        num_warps=num_warps,
    )
    return out.view(B, H, T, Vd)
