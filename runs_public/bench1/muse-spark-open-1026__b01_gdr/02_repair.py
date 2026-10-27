import torch
import triton
import triton.language as tl

@triton.jit
def _seq_chunk_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr, state_ptr,
                      scale,
                      T, K_orig, V_orig,
                      start,
                      Kb: tl.constexpr, Vb: tl.constexpr,
                      BLOCK_C: tl.constexpr, BLOCK_V: tl.constexpr):
    pid = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    offs_kb = tl.arange(0, Kb)
    offs_vb = tl.arange(0, BLOCK_V)

    cur_t = start + offs_c
    mask_c = cur_t < T
    cur_t_clamped = tl.where(mask_c, cur_t, 0)

    g_vals = tl.load(g_ptr + pid * T + cur_t_clamped, mask=mask_c, other=0.0).to(tl.float32)
    d_vec = tl.exp(g_vals)
    d_vec = tl.where(mask_c, d_vec, 1.0)

    beta_vec = tl.load(beta_ptr + pid * T + cur_t_clamped, mask=mask_c, other=0.0).to(tl.float32)
    beta_vec = tl.where(mask_c, beta_vec, 0.0)

    mask_kb = offs_kb < K_orig

    q_base = q_ptr + pid * T * Kb
    k_base = k_ptr + pid * T * Kb

    q_chunk = tl.load(q_base + cur_t_clamped[:, None] * Kb + offs_kb[None, :],
                      mask=mask_c[:, None] & mask_kb[None, :], other=0.0).to(tl.float32)
    k_chunk = tl.load(k_base + cur_t_clamped[:, None] * Kb + offs_kb[None, :],
                      mask=mask_c[:, None] & mask_kb[None, :], other=0.0).to(tl.float32)

    # iterate over V tiles - range bounds are constexpr so unrolled
    for v_off in range(0, Vb, BLOCK_V):
        cols = v_off + offs_vb
        mask_vb = cols < V_orig

        v_base = v_ptr + pid * T * Vb
        v_chunk = tl.load(v_base + cur_t_clamped[:, None] * Vb + cols[None, :],
                          mask=mask_c[:, None] & mask_vb[None, :], other=0.0).to(tl.float32)

        state_base = state_ptr + pid * Kb * Vb
        s_tile = tl.load(state_base + offs_kb[:, None] * Vb + cols[None, :],
                         mask=mask_kb[:, None] & mask_vb[None, :], other=0.0).to(tl.float32)

        out_tile = tl.zeros((BLOCK_C, BLOCK_V), dtype=tl.float32)

        for i in range(0, BLOCK_C):
            is_i = offs_c == i
            d_i = tl.sum(d_vec * is_i)
            b_i = tl.sum(beta_vec * is_i)
            k_i = tl.sum(k_chunk * is_i[:, None], axis=0)
            q_i = tl.sum(q_chunk * is_i[:, None], axis=0)
            v_i = tl.sum(v_chunk * is_i[:, None], axis=0)

            # p = k_i^T S
            p = tl.sum(k_i[:, None] * s_tile, axis=0)

            # S = d*S - d*b*k p^T + b*k v^T  ; for invalid rows d=1,b=0 => no-op
            s_tile = s_tile * d_i
            outer_kp = k_i[:, None] * p[None, :]
            s_tile = s_tile - d_i * b_i * outer_kp
            outer_kv = k_i[:, None] * v_i[None, :]
            s_tile = s_tile + b_i * outer_kv

            o_i = tl.sum(q_i[:, None] * s_tile, axis=0) * scale
            out_tile = out_tile + is_i[:, None] * o_i[None, :]

        tl.store(state_base + offs_kb[:, None] * Vb + cols[None, :],
                 s_tile,
                 mask=mask_kb[:, None] & mask_vb[None, :])

        out_base = out_ptr + pid * T * Vb
        tl.store(out_base + cur_t_clamped[:, None] * Vb + cols[None, :],
                 out_tile,
                 mask=mask_c[:, None] & mask_vb[None, :])

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e:[B,H,T,K] bf16, v:[B,H,T,V] bf16, g,beta,gamma:[B,H,T] fp32
    B, H, T, K = q.shape
    V = v.shape[-1]
    BH = B * H

    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)

    q_flat = q.contiguous().reshape(BH, T, K)
    k_flat = k.contiguous().reshape(BH, T, K)
    v_flat = v.contiguous().reshape(BH, T, V)
    g_flat = g.contiguous().reshape(BH, T).to(torch.float32)
    beta_flat = beta.contiguous().reshape(BH, T).to(torch.float32)

    Kb = ((K + 31) // 32) * 32
    Vb = ((V + 31) // 32) * 32
    if Kb == 0:
        Kb = 32
    if Vb == 0:
        Vb = 32

    q_pad = torch.zeros((BH, T, Kb), device=q.device, dtype=q_flat.dtype)
    k_pad = torch.zeros((BH, T, Kb), device=q.device, dtype=k_flat.dtype)
    v_pad = torch.zeros((BH, T, Vb), device=q.device, dtype=v_flat.dtype)
    q_pad[:, :, :K] = q_flat
    k_pad[:, :, :K] = k_flat
    v_pad[:, :, :V] = v_flat

    state = torch.zeros((BH, Kb, Vb), device=q.device, dtype=torch.float32)
    out_pad = torch.zeros((BH, T, Vb), device=q.device, dtype=torch.float32)

    BLOCK_C = 32
    BLOCK_V = 32
    grid = (BH,)

    for start in range(0, T, BLOCK_C):
        _seq_chunk_kernel[grid](q_pad, k_pad, v_pad, g_flat, beta_flat, out_pad, state,
                                scale_f, T, K, V, start, Kb, Vb, BLOCK_C, BLOCK_V)

    out = out_pad[:, :, :V].reshape(B, H, T, V)
    # return in q's dtype or float32 both allowed
    if q.dtype != torch.float32:
        out = out.to(q.dtype)
    return out
