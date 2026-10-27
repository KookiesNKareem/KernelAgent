import torch
import triton
import triton.language as tl

@triton.jit
def _fast_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr,
                 scale,
                 BH, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BLOCK_V: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    k_offs = tl.arange(0, K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offs < V

    # S[K, BLOCK_V] stays in registers distributed over warps, fp32
    S = tl.zeros((K, BLOCK_V), dtype=tl.float32)

    # base pointers for this BH
    q_base = q_ptr + pid_bh * T * K
    k_base = k_ptr + pid_bh * T * K
    g_base = g_ptr + pid_bh * T * K
    v_base = v_ptr + pid_bh * T * V
    b_base = beta_ptr + pid_bh * T
    o_base = out_ptr + pid_bh * T * V

    for t in range(T):
        off_k = t * K
        off_v = t * V
        # promoted to fp32 like reference: torch.promote_types(bf16,fp32)=fp32
        q_vec = tl.load(q_base + off_k + k_offs).to(tl.float32)
        k_vec = tl.load(k_base + off_k + k_offs).to(tl.float32)
        g_vec = tl.load(g_base + off_k + k_offs).to(tl.float32)
        v_vec = tl.load(v_base + off_v + v_offs, mask=v_mask, other=0.0).to(tl.float32)
        beta_val = tl.load(b_base + t).to(tl.float32)

        # S = diag(exp(g)) * S
        decay = tl.exp(g_vec)
        S = S * tl.expand_dims(decay, 1)

        # pred = S^T k  (sum over K)
        pred = tl.sum(S * tl.expand_dims(k_vec, 1), axis=0)

        # S += k * (beta*(v-pred))^T
        delta = beta_val * (v_vec - pred)
        S = S + tl.expand_dims(k_vec, 1) * tl.expand_dims(delta, 0)

        # o = scale * S^T q
        out_vec = tl.sum(S * tl.expand_dims(q_vec * scale, 1), axis=0)
        tl.store(o_base + off_v + v_offs, out_vec, mask=v_mask)

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e: [B,H,T,K] bf16  v: [B,H,T,V] bf16  g: [B,H,T,K] fp32  beta,gamma: [B,H,T] fp32
    # e,gamma unused for API compatibility
    B, H, T, K = q.shape
    V = v.shape[-1]
    BH = B * H

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = g.contiguous()
    beta_c = beta.contiguous()

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)

    # 64-wide V-tile halves redundant q/k/g/beta loads vs 32
    BLOCK_V = 64
    if V % BLOCK_V != 0:
        BLOCK_V = 32
    n_vblocks = (V + BLOCK_V - 1) // BLOCK_V
    grid = (BH, n_vblocks)

    _fast_kernel[grid](q_c, k_c, v_c, g_c, beta_c, out, scale_f,
                       BH, T, K, V, BLOCK_V,
                       num_warps=8, num_stages=3)
    return out
