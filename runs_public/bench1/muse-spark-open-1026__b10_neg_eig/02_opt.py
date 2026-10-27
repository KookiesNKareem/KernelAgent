import torch
import triton
import triton.language as tl

@triton.jit
def fast_fwd_kernel_128(
    q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, out_ptr,
    BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    stride_qbh, stride_qt, stride_qk,
    stride_kbh, stride_kt, stride_kk,
    stride_vbh, stride_vt, stride_vk,
    stride_dbh, stride_dt,
    stride_bbh, stride_bt,
    stride_obh, stride_ot, stride_ov,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, NUM_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NUM_V
    bv = pid % NUM_V

    # state S[K,V] split column-wise: each program holds K x BLOCK_V = 128x32
    # stored as 4 x (32x32) to stay within Triton 1024 element limit
    S0 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S1 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S2 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S3 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    offs_k0 = tl.arange(0, BLOCK_K)
    offs_k1 = tl.arange(0, BLOCK_K) + BLOCK_K
    offs_k2 = tl.arange(0, BLOCK_K) + 2 * BLOCK_K
    offs_k3 = tl.arange(0, BLOCK_K) + 3 * BLOCK_K
    offs_v = bv * BLOCK_V + tl.arange(0, BLOCK_V)

    for t in range(T):
        d_t = tl.load(d_ptr + bh * stride_dbh + t * stride_dt)
        # beta_ptr already contains 2*beta (pre-scaled in wrapper)
        coeff = tl.load(beta_ptr + bh * stride_bbh + t * stride_bt)

        # loads of k and q - 128 elements as 4x32
        k0 = tl.load(k_ptr + bh * stride_kbh + t * stride_kt + offs_k0 * stride_kk).to(tl.float32)
        k1 = tl.load(k_ptr + bh * stride_kbh + t * stride_kt + offs_k1 * stride_kk).to(tl.float32)
        k2 = tl.load(k_ptr + bh * stride_kbh + t * stride_kt + offs_k2 * stride_kk).to(tl.float32)
        k3 = tl.load(k_ptr + bh * stride_kbh + t * stride_kt + offs_k3 * stride_kk).to(tl.float32)

        q0 = tl.load(q_ptr + bh * stride_qbh + t * stride_qt + offs_k0 * stride_qk).to(tl.float32)
        q1 = tl.load(q_ptr + bh * stride_qbh + t * stride_qt + offs_k1 * stride_qk).to(tl.float32)
        q2 = tl.load(q_ptr + bh * stride_qbh + t * stride_qt + offs_k2 * stride_qk).to(tl.float32)
        q3 = tl.load(q_ptr + bh * stride_qbh + t * stride_qt + offs_k3 * stride_qk).to(tl.float32)

        # pred = d * S_old^T k
        pred = tl.zeros((BLOCK_V,), dtype=tl.float32)
        pred += tl.sum(S0 * tl.expand_dims(k0, 1), axis=0)
        pred += tl.sum(S1 * tl.expand_dims(k1, 1), axis=0)
        pred += tl.sum(S2 * tl.expand_dims(k2, 1), axis=0)
        pred += tl.sum(S3 * tl.expand_dims(k3, 1), axis=0)
        pred = pred * d_t

        v_blk = tl.load(v_ptr + bh * stride_vbh + t * stride_vt + offs_v * stride_vk).to(tl.float32)
        delta = (v_blk - pred) * coeff  # 2*beta*(v-pred)
        delta_r = tl.expand_dims(delta, 0)

        # S_new = d*S_old + k * delta^T  and  o = S_new^T q
        out_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)

        S0 = S0 * d_t + tl.expand_dims(k0, 1) * delta_r
        out_acc += tl.sum(S0 * tl.expand_dims(q0, 1), axis=0)

        S1 = S1 * d_t + tl.expand_dims(k1, 1) * delta_r
        out_acc += tl.sum(S1 * tl.expand_dims(q1, 1), axis=0)

        S2 = S2 * d_t + tl.expand_dims(k2, 1) * delta_r
        out_acc += tl.sum(S2 * tl.expand_dims(q2, 1), axis=0)

        S3 = S3 * d_t + tl.expand_dims(k3, 1) * delta_r
        out_acc += tl.sum(S3 * tl.expand_dims(q3, 1), axis=0)

        out_acc = out_acc * scale
        tl.store(out_ptr + bh * stride_obh + t * stride_ot + offs_v * stride_ov, out_acc)


@triton.jit
def generic_fwd_kernel(
    q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, out_ptr, scratch_ptr,
    BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    stride_qbh, stride_qt, stride_qk,
    stride_kbh, stride_kt, stride_kk,
    stride_vbh, stride_vt, stride_vk,
    stride_dbh, stride_dt,
    stride_bbh, stride_bt,
    stride_obh, stride_ot, stride_ov,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    NUM_K: tl.constexpr, NUM_V: tl.constexpr,
):
    pid = tl.program_id(0)
    for t in range(T):
        d_t = tl.load(d_ptr + pid * stride_dbh + t * stride_dt)
        coeff = tl.load(beta_ptr + pid * stride_bbh + t * stride_bt)
        for bv in range(NUM_V):
            pred = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for bk in range(NUM_K):
                s_ptr = tl.make_block_ptr(base=scratch_ptr + pid * K * V, shape=(K, V), strides=(V, 1), offsets=(bk * BLOCK_K, bv * BLOCK_V), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                S_blk = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                offs_k = tl.arange(0, BLOCK_K) + bk * BLOCK_K
                k_blk = tl.load(k_ptr + pid * stride_kbh + t * stride_kt + offs_k * stride_kk, mask=offs_k < K, other=0.0).to(tl.float32)
                pred += tl.sum(S_blk * tl.expand_dims(k_blk, 1), axis=0)
            pred = pred * d_t
            offs_v = tl.arange(0, BLOCK_V) + bv * BLOCK_V
            v_blk = tl.load(v_ptr + pid * stride_vbh + t * stride_vt + offs_v * stride_vk, mask=offs_v < V, other=0.0).to(tl.float32)
            delta = (v_blk - pred) * coeff
            delta_r = tl.expand_dims(delta, 0)
            out_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for bk in range(NUM_K):
                s_ptr = tl.make_block_ptr(base=scratch_ptr + pid * K * V, shape=(K, V), strides=(V, 1), offsets=(bk * BLOCK_K, bv * BLOCK_V), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                S_blk = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                S_blk = S_blk * d_t
                offs_k = tl.arange(0, BLOCK_K) + bk * BLOCK_K
                k_blk = tl.load(k_ptr + pid * stride_kbh + t * stride_kt + offs_k * stride_kk, mask=offs_k < K, other=0.0).to(tl.float32)
                q_blk = tl.load(q_ptr + pid * stride_qbh + t * stride_qt + offs_k * stride_qk, mask=offs_k < K, other=0.0).to(tl.float32)
                S_new = S_blk + tl.expand_dims(k_blk, 1) * delta_r
                tl.store(s_ptr, S_new, boundary_check=(0, 1))
                out_acc += tl.sum(S_new * tl.expand_dims(q_blk, 1), axis=0)
            tl.store(out_ptr + pid * stride_obh + t * stride_ot + offs_v * stride_ov, out_acc * scale, mask=offs_v < V)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e:[B,H,T,K] bf16, v:[B,H,T,V] bf16, g,beta,gamma:[B,H,T] fp32
    # reference promotes to fp32, S_t = exp(g_t)(I-2 beta k k^T) S_{t-1}+2 beta k v^T
    B, H, T, K = q.shape
    Vdim = v.shape[-1]
    BH = B * H

    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)

    q_f = q.reshape(BH, T, K).contiguous()
    k_f = k.reshape(BH, T, K).contiguous()
    v_f = v.reshape(BH, T, Vdim).contiguous()
    # cheap elementwise prep in Python (allowed): exp(g) and 2*beta
    d_f = torch.exp(g.reshape(BH, T).contiguous().float()).contiguous()
    beta_f = (beta.reshape(BH, T).contiguous().float() * 2.0).contiguous()
    out_f = torch.empty(BH, T, Vdim, device=q.device, dtype=torch.float32)

    # fast path for benchmark shape K=V=128 (most common)
    if K == 128 and Vdim == 128 and T >= 1:
        BLOCK_K = 32
        BLOCK_V = 32
        NUM_V = 4
        grid = (BH * NUM_V,)
        fast_fwd_kernel_128[grid](
            q_f, k_f, v_f, d_f, beta_f, out_f,
            BH, T, K, Vdim,
            q_f.stride(0), q_f.stride(1), q_f.stride(2),
            k_f.stride(0), k_f.stride(1), k_f.stride(2),
            v_f.stride(0), v_f.stride(1), v_f.stride(2),
            d_f.stride(0), d_f.stride(1),
            beta_f.stride(0), beta_f.stride(1),
            out_f.stride(0), out_f.stride(1), out_f.stride(2),
            scale_f,
            BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, NUM_V=NUM_V,
            num_warps=4, num_stages=1,
        )
    else:
        BLOCK_K = 32
        BLOCK_V = 32
        NUM_K = (K + BLOCK_K - 1) // BLOCK_K
        NUM_V = (Vdim + BLOCK_V - 1) // BLOCK_V
        scratch = torch.zeros(BH * K * Vdim, device=q.device, dtype=torch.float32)
        grid = (BH,)
        generic_fwd_kernel[grid](
            q_f, k_f, v_f, d_f, beta_f, out_f, scratch,
            BH, T, K, Vdim,
            q_f.stride(0), q_f.stride(1), q_f.stride(2),
            k_f.stride(0), k_f.stride(1), k_f.stride(2),
            v_f.stride(0), v_f.stride(1), v_f.stride(2),
            d_f.stride(0), d_f.stride(1),
            beta_f.stride(0), beta_f.stride(1),
            out_f.stride(0), out_f.stride(1), out_f.stride(2),
            scale_f,
            BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, NUM_K=NUM_K, NUM_V=NUM_V,
        )

    return out_f.reshape(B, H, T, Vdim)
