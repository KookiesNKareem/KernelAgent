import torch
import triton
import triton.language as tl

@triton.jit
def fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    s0_q, s1_q, s2_q, s3_q,
    s0_k, s1_k, s2_k, s3_k,
    s0_v, s1_v, s2_v, s3_v,
    s0_e, s1_e, s2_e, s3_e,
    s0_g, s1_g, s2_g, s3_g,
    s0_b, s1_b, s2_b,
    s0_c, s1_c, s2_c,
    s0_o, s1_o, s2_o, s3_o,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)

    NK = (K + BLOCK_K - 1) // BLOCK_K
    NV = (V + BLOCK_V - 1) // BLOCK_V

    # S[K,V] tiled in SRAM as NK x NV blocks of BLOCK_K x BLOCK_V, init 0
    # use python lists of tl.tensor - number of blocks is constexpr after recompilation
    S = [[tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32) for _ in range(NV)] for _ in range(NK)]

    q_base = q_ptr + b * s0_q + h * s1_q
    k_base = k_ptr + b * s0_k + h * s1_k
    v_base = v_ptr + b * s0_v + h * s1_v
    e_base = e_ptr + b * s0_e + h * s1_e
    g_base = g_ptr + b * s0_g + h * s1_g
    b_base = beta_ptr + b * s0_b + h * s1_b
    c_base = gamma_ptr + b * s0_c + h * s1_c
    o_base = out_ptr + b * s0_o + h * s1_o

    for t in range(T):
        q_t = q_base + t * s2_q
        k_t = k_base + t * s2_k
        v_t = v_base + t * s2_v
        e_t = e_base + t * s2_e
        g_t = g_base + t * s2_g
        bt = b_base + t * s2_b
        gt = c_base + t * s2_c
        ot = o_base + t * s2_o

        b_s = tl.load(bt).to(tl.float32)
        g_s = tl.load(gt).to(tl.float32)
        ct = g_s * (1.0 - b_s)

        # cache vectors per block
        e_blks = []
        for ik in range(NK):
            thresh = K - ik * BLOCK_K
            ptr = e_t + ik * BLOCK_K * s3_e + offs_k * s3_e
            if thresh >= BLOCK_K:
                v_ = tl.load(ptr).to(tl.float32)
            else:
                mask = offs_k < thresh
                v_ = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
            e_blks.append(v_)

        k_blks = []
        for ik in range(NK):
            thresh = K - ik * BLOCK_K
            ptr = k_t + ik * BLOCK_K * s3_k + offs_k * s3_k
            if thresh >= BLOCK_K:
                v_ = tl.load(ptr).to(tl.float32)
            else:
                mask = offs_k < thresh
                v_ = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
            k_blks.append(v_)

        q_blks = []
        for ik in range(NK):
            thresh = K - ik * BLOCK_K
            ptr = q_t + ik * BLOCK_K * s3_q + offs_k * s3_q
            if thresh >= BLOCK_K:
                v_ = tl.load(ptr).to(tl.float32)
            else:
                mask = offs_k < thresh
                v_ = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
            q_blks.append(v_)

        d_blks = []
        for ik in range(NK):
            thresh = K - ik * BLOCK_K
            ptr = g_t + ik * BLOCK_K * s3_g + offs_k * s3_g
            if thresh >= BLOCK_K:
                gv = tl.load(ptr).to(tl.float32)
            else:
                mask = offs_k < thresh
                gv = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
            d_blks.append(tl.exp(gv))

        v_blks = []
        for iv in range(NV):
            thresh = V - iv * BLOCK_V
            ptr = v_t + iv * BLOCK_V * s3_v + offs_v * s3_v
            if thresh >= BLOCK_V:
                v_ = tl.load(ptr).to(tl.float32)
            else:
                mask = offs_v < thresh
                v_ = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
            v_blks.append(v_)

        # 1) decay rows: S *= exp(g)
        for ik in range(NK):
            d = d_blks[ik]
            dc = d[:, None]
            for iv in range(NV):
                S[ik][iv] = S[ik][iv] * dc

        # 2) e-projection: S -= ct * e * (e^T S)
        # e^T S -> [V]
        eTS = []
        for iv in range(NV):
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for ik in range(NK):
                acc += tl.sum(e_blks[ik][:, None] * S[ik][iv], axis=0)
            eTS.append(acc)
        for ik in range(NK):
            ec = e_blks[ik][:, None]
            for iv in range(NV):
                S[ik][iv] = S[ik][iv] - ct * ec * eTS[iv][None, :]

        # 3) pred = k^T S  (V)
        pred = []
        for iv in range(NV):
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for ik in range(NK):
                acc += tl.sum(k_blks[ik][:, None] * S[ik][iv], axis=0)
            pred.append(acc)

        # 4) S += b * k * (v-pred)^T
        for ik in range(NK):
            kc = k_blks[ik][:, None]
            for iv in range(NV):
                delta = v_blks[iv] - pred[iv]
                S[ik][iv] = S[ik][iv] + b_s * kc * delta[None, :]

        # 5) out = scale * S^T q
        for iv in range(NV):
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for ik in range(NK):
                acc += tl.sum(q_blks[ik][:, None] * S[ik][iv], axis=0)
            acc = acc * scale
            thresh = V - iv * BLOCK_V
            ptr = ot + iv * BLOCK_V * s3_o + offs_v * s3_o
            if thresh >= BLOCK_V:
                tl.store(ptr, acc)
            else:
                mask = offs_v < thresh
                tl.store(ptr, acc, mask=mask)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e:[B,H,T,K] bf16, v:[B,H,T,V] bf16, g:[B,H,T,K] fp32, beta,gamma:[B,H,T] fp32
    B, H, T, K = q.shape
    V = v.shape[-1]
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        scale = float(scale)
    out = torch.empty(B, H, T, V, device=q.device, dtype=torch.float32)
    BLOCK_K = 32
    BLOCK_V = 32
    grid = (B * H,)
    fwd_kernel[grid](
        q, k, v, e, g, beta, gamma, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        e.stride(0), e.stride(1), e.stride(2), e.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, T, K, V, scale,
        BLOCK_K, BLOCK_V,
        num_warps=4, num_stages=2,
    )
    return out
