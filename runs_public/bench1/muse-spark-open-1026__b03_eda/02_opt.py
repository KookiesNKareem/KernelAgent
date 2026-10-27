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

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    # K x BLOCK_V state in SRAM, float32
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    # B*H*T flat index helper: bh = b*H+h, bh_t = bh*T + t
    bh_T = bh * T  # base
    for t in range(T):
        bh_t = bh_T + t

        g_val = tl.load(g_ptr + bh_t)
        beta_val = tl.load(beta_ptr + bh_t)
        gamma_val = tl.load(gamma_ptr + bh_t)
        d = tl.exp(g_val)
        S = S * d

        # e delta: S = S - gamma*e*(e^T S)
        e_vec = tl.load(e_ptr + bh_t * K + offs_k).to(tl.float32)  # bf16->fp32, K==BLOCK_K no mask
        s_e = tl.sum(e_vec[:, None] * S, axis=0)  # [BLOCK_V]
        S = S - gamma_val * (e_vec[:, None] * s_e[None, :])

        # k delta + pred: s_k = k^T S
        k_vec = tl.load(k_ptr + bh_t * K + offs_k).to(tl.float32)
        s_k = tl.sum(k_vec[:, None] * S, axis=0)
        v_vec = tl.load(v_ptr + bh_t * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        # S += beta*k*(v - s_k)^T
        S = S + beta_val * (k_vec[:, None] * (v_vec - s_k)[None, :])

        q_vec = tl.load(q_ptr + bh_t * K + offs_k).to(tl.float32)
        o_vec = tl.sum(q_vec[:, None] * S, axis=0) * scale
        tl.store(o_ptr + bh_t * V + offs_v, o_vec, mask=mask_v)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    Vd = v.shape[-1]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    e = e.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    gamma = gamma.contiguous()

    out = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)

    # 64 is sweet spot on L4: 2x wider than 32 halves duplicate Q/K/E HBM
    # traffic (V sharding) and keeps occupancy 64 > 58 SMs, while 128*64
    # state = 32KB fits in SRAM.
    BLOCK_K = K  # 128, full reduction over K, no cross-block sync
    BLOCK_V = 64
    if Vd < BLOCK_V:
        BLOCK_V = Vd
    VBLOCKS = (Vd + BLOCK_V - 1) // BLOCK_V

    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        scale = float(scale)

    grid = (B * H * VBLOCKS,)

    # tuned for Ada sm_89: 4 warps keeps 128x64 tile in registers,
    # 2 stages pipelines the bf16 loads without spilling
    fused_kernel[grid](q, k, v, e, g, beta, gamma, out,
                       B, H, T, K, Vd, scale,
                       BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, VBLOCKS=VBLOCKS,
                       num_warps=4, num_stages=2)
    return out
