import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr,
                 B, H, T, K, V, scale,
                 BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, VBLOCKS: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // VBLOCKS
    vb = pid % VBLOCKS
    b = bh // H
    h = bh % H

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < V

    # S is K x BLOCK_V kept in SRAM, zero init
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    t = 0
    while t < T:
        bh_t = (b * H + h) * T + t  # flat time index for g/beta/gamma
        g_val = tl.load(g_ptr + bh_t)
        beta_val = tl.load(beta_ptr + bh_t)
        gamma_val = tl.load(gamma_ptr + bh_t)
        d = tl.exp(g_val)
        S = S * d

        # e : [B,H,T,K] bf16 -> float32
        e_vec = tl.load(e_ptr + bh_t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        e_exp = e_vec[:, None]  # (BK,1)
        s_e = tl.sum(e_exp * S, axis=0)  # (BV,)
        s_e = tl.where(mask_v, s_e, 0.0)
        S = S - gamma_val * (e_exp * s_e[None, :])

        # k delta
        k_vec = tl.load(k_ptr + bh_t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        k_exp = k_vec[:, None]
        s_k = tl.sum(k_exp * S, axis=0)
        s_k = tl.where(mask_v, s_k, 0.0)

        v_vec = tl.load(v_ptr + bh_t * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        v_minus = tl.where(mask_v, v_vec - s_k, 0.0)
        S = S + beta_val * (k_exp * v_minus[None, :])

        # output
        q_vec = tl.load(q_ptr + bh_t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        q_exp = q_vec[:, None]
        o_vec = tl.sum(q_exp * S, axis=0) * scale
        tl.store(o_ptr + bh_t * V + offs_v, o_vec, mask=mask_v)
        t += 1


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e:[B,H,T,K] bf16, v:[B,H,T,V] bf16, g/beta/gamma:[B,H,T] fp32
    # reference promotes to float32, we compute in float32 and return float32
    B, H, T, K = q.shape
    Vd = v.shape[-1]

    # wrapper may only do cheap prep -> make contiguous (cheap) and allocate
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    e = e.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    gamma = gamma.contiguous()

    out = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)

    # choose tiling: full K in one block, V sharded across programs
    BLOCK_K = K  # K is constexpr for this launch (128 in benchmark)
    BLOCK_V = 32
    if Vd < BLOCK_V:
        BLOCK_V = Vd
    # VBLOCKS must be constexpr for the kernel, compute on host
    VBLOCKS = (Vd + BLOCK_V - 1) // BLOCK_V

    grid = (B * H * VBLOCKS, )
    # scale may be tensor, extract python float
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        scale = float(scale)

    fused_kernel[grid](q, k, v, e, g, beta, gamma, out,
                       B, H, T, K, Vd, scale,
                       BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, VBLOCKS=VBLOCKS)
    return out
