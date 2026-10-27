import torch
import triton
import triton.language as tl

@triton.jit
def fwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr, scratch_ptr,
    BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    stride_qbh, stride_qt, stride_qk,
    stride_kbh, stride_kt, stride_kk,
    stride_vbh, stride_vt, stride_vk,
    stride_gbh, stride_gt,
    stride_bbh, stride_bt,
    stride_obh, stride_ot, stride_ov,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    NUM_K: tl.constexpr, NUM_V: tl.constexpr,
):
    pid = tl.program_id(0)
    # BH programs, each loops sequentially over T, state S[K,V] lives in global scratch
    # tiled into BLOCK_K x BLOCK_V blocks. All math promoted to fp32.
    for t in range(T):
        g_t = tl.load(g_ptr + pid * stride_gbh + t * stride_gt)
        d = tl.exp(g_t)  # exp(g_t) <=1
        b_t = tl.load(beta_ptr + pid * stride_bbh + t * stride_bt)
        coeff = b_t * 2.0

        for bv in range(NUM_V):
            # pred = d * sum_k S_old * k  for this V-slice
            pred = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for bk in range(NUM_K):
                s_ptr = tl.make_block_ptr(
                    base=scratch_ptr + pid * K * V,
                    shape=(K, V), strides=(V, 1),
                    offsets=(bk * BLOCK_K, bv * BLOCK_V),
                    block_shape=(BLOCK_K, BLOCK_V), order=(1, 0)
                )
                S_blk = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                offs_k = tl.arange(0, BLOCK_K) + bk * BLOCK_K
                k_ptrs = k_ptr + pid * stride_kbh + t * stride_kt + offs_k * stride_kk
                k_blk = tl.load(k_ptrs, mask=offs_k < K).to(tl.float32)
                k_col = k_blk[:, None]
                pred += tl.sum(S_blk * k_col, axis=0)
            pred = pred * d

            offs_v = tl.arange(0, BLOCK_V) + bv * BLOCK_V
            v_ptrs = v_ptr + pid * stride_vbh + t * stride_vt + offs_v * stride_vk
            v_blk = tl.load(v_ptrs, mask=offs_v < V).to(tl.float32)
            delta = (v_blk - pred) * coeff          # 2*beta*(v-pred)
            delta_row = delta[None, :]

            out_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for bk in range(NUM_K):
                s_ptr = tl.make_block_ptr(
                    base=scratch_ptr + pid * K * V,
                    shape=(K, V), strides=(V, 1),
                    offsets=(bk * BLOCK_K, bv * BLOCK_V),
                    block_shape=(BLOCK_K, BLOCK_V), order=(1, 0)
                )
                S_blk = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                S_scaled = S_blk * d
                offs_k = tl.arange(0, BLOCK_K) + bk * BLOCK_K
                k_ptrs = k_ptr + pid * stride_kbh + t * stride_kt + offs_k * stride_kk
                k_blk = tl.load(k_ptrs, mask=offs_k < K).to(tl.float32)
                q_ptrs = q_ptr + pid * stride_qbh + t * stride_qt + offs_k * stride_qk
                q_blk = tl.load(q_ptrs, mask=offs_k < K).to(tl.float32)
                k_col = k_blk[:, None]
                q_col = q_blk[:, None]
                S_new = S_scaled + k_col * delta_row
                tl.store(s_ptr, S_new, boundary_check=(0, 1))
                out_acc += tl.sum(S_new * q_col, axis=0)

            out_ptrs = out_ptr + pid * stride_obh + t * stride_ot + offs_v * stride_ov
            tl.store(out_ptrs, out_acc * scale, mask=offs_v < V)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e: [B,H,T,K] bf16 (k,e unit-norm) v: [B,H,T,V] bf16
    # g: [B,H,T] fp32 <=0 ; beta,gamma: [B,H,T] fp32 (0,1)
    # scale: scalar, o_t = scale * S_t^T q_t ; reference promotes to fp32
    # e, gamma unused but kept for API compatibility
    B, H, T, K = q.shape
    Vdim = v.shape[-1]
    BH = B * H

    # flatten B*H for one program per head
    q_f = q.reshape(BH, T, K).contiguous()
    k_f = k.reshape(BH, T, K).contiguous()
    v_f = v.reshape(BH, T, Vdim).contiguous()
    g_f = g.reshape(BH, T).contiguous()
    beta_f = beta.reshape(BH, T).contiguous()

    out_f = torch.empty(BH, T, Vdim, device=q.device, dtype=torch.float32)
    scratch = torch.zeros(BH, K, Vdim, device=q.device, dtype=torch.float32)

    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        scale = float(scale)

    BLOCK_K = 32
    BLOCK_V = 32
    if K < BLOCK_K:
        BLOCK_K = K
    if Vdim < BLOCK_V:
        BLOCK_V = Vdim
    NUM_K = (K + BLOCK_K - 1) // BLOCK_K
    NUM_V = (Vdim + BLOCK_V - 1) // BLOCK_V

    grid = (BH,)
    fwd_kernel[grid](
        q_f, k_f, v_f, g_f, beta_f, out_f, scratch,
        BH, T, K, Vdim,
        q_f.stride(0), q_f.stride(1), q_f.stride(2),
        k_f.stride(0), k_f.stride(1), k_f.stride(2),
        v_f.stride(0), v_f.stride(1), v_f.stride(2),
        g_f.stride(0), g_f.stride(1),
        beta_f.stride(0), beta_f.stride(1),
        out_f.stride(0), out_f.stride(1), out_f.stride(2),
        scale,
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, NUM_K=NUM_K, NUM_V=NUM_V,
    )
    out = out_f.reshape(B, H, T, Vdim)
    return out
