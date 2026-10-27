import torch
import triton
import triton.language as tl


@triton.jit
def _seq_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
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
        g_t = tl.load(g_ptr + base_qk + k_offs, mask=mask_k, other=0.0)
        beta_t = tl.load(beta_ptr + base_b)
        v_t = tl.load(v_ptr + base_v + v_offs, mask=mask_v, other=0.0)

        k_f = k_t.to(tl.float32)
        q_f = q_t.to(tl.float32)
        v_f = v_t.to(tl.float32)
        g_f = g_t.to(tl.float32)
        exp_g = tl.exp(g_f)

        # S = diag(exp(g)) S
        exp_g_col = tl.expand_dims(exp_g, 1)
        S = S * exp_g_col

        # pred = S^T k
        k_col = tl.expand_dims(k_f, 1)
        pred = tl.sum(S * k_col, axis=0)

        # S += k * (beta * (v - pred))
        err = (v_f - pred) * beta_t
        err_row = tl.expand_dims(err, 0)
        S = S + k_col * err_row

        # o = scale * S^T q  (reference scales q first)
        q_scaled = q_f * scale
        q_col = tl.expand_dims(q_scaled, 1)
        out_t = tl.sum(S * q_col, axis=0)
        tl.store(o_ptr + base_v + v_offs, out_t.to(tl.float32), mask=mask_v)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e: [B,H,T,K], v: [B,H,T,V], g: [B,H,T,K], beta,gamma: [B,H,T]
    # e, gamma unused but accepted
    _ = (e, gamma)
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    BH = B * H

    q_ = q.reshape(BH, T, K).contiguous()
    k_ = k.reshape(BH, T, K).contiguous()
    v_ = v.reshape(BH, T, Vd).contiguous()
    g_ = g.reshape(BH, T, K).contiguous()
    beta_ = beta.reshape(BH, T).contiguous()

    if isinstance(scale, torch.Tensor):
        scale_val = float(scale.item())
    else:
        scale_val = float(scale)

    BLOCK_K = triton.next_power_of_2(K)
    if BLOCK_K < 16:
        BLOCK_K = 16
    BLOCK_V = 32
    NUM_VB = (Vd + BLOCK_V - 1) // BLOCK_V

    out = torch.empty((BH, T, Vd), device=q.device, dtype=torch.float32)
    if T > 0 and K > 0 and Vd > 0 and BH > 0:
        grid = (BH * NUM_VB,)
        _seq_kernel[grid](
            q_, k_, v_, g_, beta_, out,
            T, K, Vd, scale_val,
            BLOCK_K, BLOCK_V, NUM_VB,
        )
    return out.view(B, H, T, Vd)
