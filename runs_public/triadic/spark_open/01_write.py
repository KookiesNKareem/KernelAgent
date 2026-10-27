import torch
import triton
import triton.language as tl

@triton.jit
def gated_delta_kernel(
    q_ptr, k_ptr, v_ptr, q2_ptr, k2_ptr, g_ptr, beta_ptr, out_ptr,
    stride_q_bh, stride_q_t,
    stride_k_bh, stride_k_t,
    stride_v_bh, stride_v_t,
    stride_q2_bh, stride_q2_t,
    stride_k2_bh, stride_k2_t,
    stride_g_bh, stride_g_t,
    stride_beta_bh, stride_beta_t,
    stride_out_bh, stride_out_t,
    BH, SEQ_T,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, NK: tl.constexpr, N_SHARDS: tl.constexpr, E: tl.constexpr, SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // N_SHARDS
    v_off = (pid % N_SHARDS) * BLOCK_V

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    offs_k_mod = offs_k % E
    offs_k_div = offs_k // E
    K_PER_BLOCK = BLOCK_K // E  # 16 for d=128,E=4,BLOCK_K=64

    # state sharded over V, kept in registers: 8 blocks of 64x16 = 512x16 per program
    S0 = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)
    S1 = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)
    S2 = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)
    S3 = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)
    S4 = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)
    S5 = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)
    S6 = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)
    S7 = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)

    # e pattern is identical for every K block (BLOCK_K % E ==0)
    # decay / k2 / q2 reused across blocks
    for t in range(SEQ_T):
        beta_t = tl.load(beta_ptr + bh * stride_beta_bh + t * stride_beta_t)
        v_tile = tl.load(v_ptr + bh * stride_v_bh + t * stride_v_t + v_off + offs_v).to(tl.float32)

        q_base = q_ptr + bh * stride_q_bh + t * stride_q_t
        k_base = k_ptr + bh * stride_k_bh + t * stride_k_t
        q2_base = q2_ptr + bh * stride_q2_bh + t * stride_q2_t
        k2_base = k2_ptr + bh * stride_k2_bh + t * stride_k2_t
        g_base = g_ptr + bh * stride_g_bh + t * stride_g_t

        e_idx = offs_k_mod
        k2_vals = tl.load(k2_base + e_idx).to(tl.float32)
        q2_vals = tl.load(q2_base + e_idx).to(tl.float32)
        g_vals = tl.load(g_base + e_idx)
        decay = tl.exp(g_vals)
        decay_col = tl.expand_dims(decay, 1)

        # decay S
        S0 = S0 * decay_col
        S1 = S1 * decay_col
        S2 = S2 * decay_col
        S3 = S3 * decay_col
        S4 = S4 * decay_col
        S5 = S5 * decay_col
        S6 = S6 * decay_col
        S7 = S7 * decay_col

        # kappa per K-block
        k_vals0 = tl.load(k_base + offs_k_div + 0 * K_PER_BLOCK).to(tl.float32)
        k_vals1 = tl.load(k_base + offs_k_div + 1 * K_PER_BLOCK).to(tl.float32)
        k_vals2 = tl.load(k_base + offs_k_div + 2 * K_PER_BLOCK).to(tl.float32)
        k_vals3 = tl.load(k_base + offs_k_div + 3 * K_PER_BLOCK).to(tl.float32)
        k_vals4 = tl.load(k_base + offs_k_div + 4 * K_PER_BLOCK).to(tl.float32)
        k_vals5 = tl.load(k_base + offs_k_div + 5 * K_PER_BLOCK).to(tl.float32)
        k_vals6 = tl.load(k_base + offs_k_div + 6 * K_PER_BLOCK).to(tl.float32)
        k_vals7 = tl.load(k_base + offs_k_div + 7 * K_PER_BLOCK).to(tl.float32)

        kappa0 = k_vals0 * k2_vals
        kappa1 = k_vals1 * k2_vals
        kappa2 = k_vals2 * k2_vals
        kappa3 = k_vals3 * k2_vals
        kappa4 = k_vals4 * k2_vals
        kappa5 = k_vals5 * k2_vals
        kappa6 = k_vals6 * k2_vals
        kappa7 = k_vals7 * k2_vals

        pred = tl.zeros((BLOCK_V,), tl.float32)
        pred = pred + tl.sum(S0 * tl.expand_dims(kappa0, 1), axis=0)
        pred = pred + tl.sum(S1 * tl.expand_dims(kappa1, 1), axis=0)
        pred = pred + tl.sum(S2 * tl.expand_dims(kappa2, 1), axis=0)
        pred = pred + tl.sum(S3 * tl.expand_dims(kappa3, 1), axis=0)
        pred = pred + tl.sum(S4 * tl.expand_dims(kappa4, 1), axis=0)
        pred = pred + tl.sum(S5 * tl.expand_dims(kappa5, 1), axis=0)
        pred = pred + tl.sum(S6 * tl.expand_dims(kappa6, 1), axis=0)
        pred = pred + tl.sum(S7 * tl.expand_dims(kappa7, 1), axis=0)

        err = beta_t * (v_tile - pred)
        err_row = tl.expand_dims(err, 0)

        # update + output, reuse kappa
        q_vals0 = tl.load(q_base + offs_k_div + 0 * K_PER_BLOCK).to(tl.float32)
        q_vals1 = tl.load(q_base + offs_k_div + 1 * K_PER_BLOCK).to(tl.float32)
        q_vals2 = tl.load(q_base + offs_k_div + 2 * K_PER_BLOCK).to(tl.float32)
        q_vals3 = tl.load(q_base + offs_k_div + 3 * K_PER_BLOCK).to(tl.float32)
        q_vals4 = tl.load(q_base + offs_k_div + 4 * K_PER_BLOCK).to(tl.float32)
        q_vals5 = tl.load(q_base + offs_k_div + 5 * K_PER_BLOCK).to(tl.float32)
        q_vals6 = tl.load(q_base + offs_k_div + 6 * K_PER_BLOCK).to(tl.float32)
        q_vals7 = tl.load(q_base + offs_k_div + 7 * K_PER_BLOCK).to(tl.float32)

        qq0 = q_vals0 * q2_vals * SCALE
        qq1 = q_vals1 * q2_vals * SCALE
        qq2 = q_vals2 * q2_vals * SCALE
        qq3 = q_vals3 * q2_vals * SCALE
        qq4 = q_vals4 * q2_vals * SCALE
        qq5 = q_vals5 * q2_vals * SCALE
        qq6 = q_vals6 * q2_vals * SCALE
        qq7 = q_vals7 * q2_vals * SCALE

        S0 = S0 + tl.expand_dims(kappa0, 1) * err_row
        S1 = S1 + tl.expand_dims(kappa1, 1) * err_row
        S2 = S2 + tl.expand_dims(kappa2, 1) * err_row
        S3 = S3 + tl.expand_dims(kappa3, 1) * err_row
        S4 = S4 + tl.expand_dims(kappa4, 1) * err_row
        S5 = S5 + tl.expand_dims(kappa5, 1) * err_row
        S6 = S6 + tl.expand_dims(kappa6, 1) * err_row
        S7 = S7 + tl.expand_dims(kappa7, 1) * err_row

        out_tile = tl.zeros((BLOCK_V,), tl.float32)
        out_tile = out_tile + tl.sum(S0 * tl.expand_dims(qq0, 1), axis=0)
        out_tile = out_tile + tl.sum(S1 * tl.expand_dims(qq1, 1), axis=0)
        out_tile = out_tile + tl.sum(S2 * tl.expand_dims(qq2, 1), axis=0)
        out_tile = out_tile + tl.sum(S3 * tl.expand_dims(qq3, 1), axis=0)
        out_tile = out_tile + tl.sum(S4 * tl.expand_dims(qq4, 1), axis=0)
        out_tile = out_tile + tl.sum(S5 * tl.expand_dims(qq5, 1), axis=0)
        out_tile = out_tile + tl.sum(S6 * tl.expand_dims(qq6, 1), axis=0)
        out_tile = out_tile + tl.sum(S7 * tl.expand_dims(qq7, 1), axis=0)

        tl.store(out_ptr + bh * stride_out_bh + t * stride_out_t + v_off + offs_v, out_tile)


def _reference_fallback(q, k, v, q2, k2, g, beta, scale):
    B, H, T, d = q.shape
    V = v.shape[-1]
    nE = k2.shape[-1]
    q, k, v, q2, k2, g, beta = (x.to(torch.promote_types(x.dtype, torch.float32)) for x in (q, k, v, q2, k2, g, beta))
    S = q.new_zeros(B, H, d * nE, V)
    outs = []
    for t in range(T):
        kap = (k[:, :, t, :, None] * k2[:, :, t, None, :]).flatten(-2)
        qq = (q[:, :, t, :, None] * q2[:, :, t, None, :]).flatten(-2)
        decay = g[:, :, t, None, :].exp().expand(B, H, d, nE).flatten(-2)
        S = S * decay[..., None]
        pred = (S * kap[..., None]).sum(-2)
        S = S + (beta[:, :, t][..., None] * (v[:, :, t] - pred))[..., None, :] * kap[..., None]
        outs.append((S * (qq * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)


def kernel_function(q, k, v, q2, k2, g, beta, scale):
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    # CPU fallback for testing without CUDA / Triton
    if not q.is_cuda:
        return _reference_fallback(q, k, v, q2, k2, g, beta, scale)

    B, H, T, d = q.shape
    V = v.shape[-1]
    E = q2.shape[-1]
    K = d * E
    BH = B * H

    # collapse B*H for simpler strides, keep contiguous
    q_ = q.reshape(BH, T, d).contiguous()
    k_ = k.reshape(BH, T, d).contiguous()
    v_ = v.reshape(BH, T, V).contiguous()
    q2_ = q2.reshape(BH, T, E).contiguous()
    k2_ = k2.reshape(BH, T, E).contiguous()
    g_ = g.reshape(BH, T, E).contiguous()
    beta_ = beta.reshape(BH, T).contiguous()

    out_ = torch.empty(BH, T, V, device=q.device, dtype=torch.float32)

    # for spec d=128,E=4,K=512,V=128 this gives 64x16
    BLOCK_K = 64 if K % 64 == 0 else 32
    BLOCK_V = 16 if V % 16 == 0 else 32
    NK = K // BLOCK_K
    N_SHARDS = V // BLOCK_V

    stride_q_bh, stride_q_t = q_.stride(0), q_.stride(1)
    stride_k_bh, stride_k_t = k_.stride(0), k_.stride(1)
    stride_v_bh, stride_v_t = v_.stride(0), v_.stride(1)
    stride_q2_bh, stride_q2_t = q2_.stride(0), q2_.stride(1)
    stride_k2_bh, stride_k2_t = k2_.stride(0), k2_.stride(1)
    stride_g_bh, stride_g_t = g_.stride(0), g_.stride(1)
    stride_beta_bh, stride_beta_t = beta_.stride(0), beta_.stride(1)
    stride_out_bh, stride_out_t = out_.stride(0), out_.stride(1)

    grid = (BH * N_SHARDS,)

    gated_delta_kernel[grid](
        q_, k_, v_, q2_, k2_, g_, beta_, out_,
        stride_q_bh, stride_q_t,
        stride_k_bh, stride_k_t,
        stride_v_bh, stride_v_t,
        stride_q2_bh, stride_q2_t,
        stride_k2_bh, stride_k2_t,
        stride_g_bh, stride_g_t,
        stride_beta_bh, stride_beta_t,
        stride_out_bh, stride_out_t,
        BH, T,
        BLOCK_K, BLOCK_V, NK, N_SHARDS, E, scale,
    )
    return out_.view(B, H, T, V)
