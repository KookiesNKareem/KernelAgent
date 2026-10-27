import torch
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr,
    BH, T, K, V, BV,
    SCALE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_vb = tl.program_id(1)

    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, BV)
    v_offs = pid_vb * BV + offs_v
    mask_v = v_offs < V

    # S lives in registers for the whole sequence, no HBM traffic
    S = tl.zeros((128, 64), dtype=tl.float32)
    # generic fallback when K/BV not 128x64 is handled by masking below,
    # but for the benchmark K=128 BV=64 the zero shape matches exactly.
    # Re-init correctly for other shapes:
    if K != 128 or BV != 64:
        S = tl.zeros((K, BV), dtype=tl.float32)

    for t in range(T):
        q_base = (pid_bh * T + t) * K
        v_base = (pid_bh * T + t) * V

        q = tl.load(q_ptr + q_base + offs_k).to(tl.float32) * SCALE
        k = tl.load(k_ptr + q_base + offs_k).to(tl.float32)
        g = tl.load(g_ptr + q_base + offs_k).to(tl.float32)
        d = tl.exp(g)
        v = tl.load(v_ptr + v_base + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + pid_bh * T + t).to(tl.float32)

        # decay in-place
        S = S * d[:, None]

        q_col = q[:, None]
        k_col = k[:, None]

        out = tl.sum(S * q_col, axis=0)
        pred = tl.sum(S * k_col, axis=0)

        tl.store(out_ptr + v_base + v_offs, out, mask=mask_v)

        delta = (v - pred) * beta
        S = S + k_col * delta[None, :]


@triton.jit
def fused_kernel32(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr,
    BH, T, K, V, BV,
    SCALE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_vb = tl.program_id(1)
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, BV)
    v_offs = pid_vb * BV + offs_v
    mask_v = v_offs < V
    S = tl.zeros((K, BV), dtype=tl.float32)
    for t in range(T):
        q_base = (pid_bh * T + t) * K
        v_base = (pid_bh * T + t) * V
        q = tl.load(q_ptr + q_base + offs_k).to(tl.float32) * SCALE
        k = tl.load(k_ptr + q_base + offs_k).to(tl.float32)
        g = tl.load(g_ptr + q_base + offs_k).to(tl.float32)
        d = tl.exp(g)
        v = tl.load(v_ptr + v_base + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + pid_bh * T + t).to(tl.float32)
        S = S * d[:, None]
        out = tl.sum(S * q[:, None], axis=0)
        pred = tl.sum(S * k[:, None], axis=0)
        tl.store(out_ptr + v_base + v_offs, out, mask=mask_v)
        S = S + k[:, None] * ((v - pred) * beta)[None, :]


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e [B,H,T,K] bf16, v [B,H,T,V] bf16, g [B,H,T,K] fp32, beta/gamma [B,H,T] fp32
    # e,gamma unused
    B, H, T, K = q.shape
    V = v.shape[-1]
    BH = B * H
    device = q.device
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)

    q_f = q.reshape(BH, T, K).contiguous()
    k_f = k.reshape(BH, T, K).contiguous()
    v_f = v.reshape(BH, T, V).contiguous()
    g_f = g.reshape(BH, T, K).contiguous()
    beta_f = beta.reshape(BH, T).contiguous()
    out_f = torch.empty((BH, T, V), dtype=torch.float32, device=device)

    # pick BV to halve duplicate q/k/g loads vs 32, while keeping register pressure OK
    if V % 64 == 0:
        BV = 64
        num_vb = V // BV
        # fast path: K=128 BV=64 uses 128x64 register tile (8192 elems)
        grid = (BH, num_vb)
        fused_kernel[grid](q_f, k_f, v_f, g_f, beta_f, out_f, BH, T, K, V, BV, scale_f, num_warps=4, num_stages=2)
    else:
        BV = 32
        num_vb = (V + BV - 1) // BV
        grid = (BH, num_vb)
        fused_kernel32[grid](q_f, k_f, v_f, g_f, beta_f, out_f, BH, T, K, V, BV, scale_f, num_warps=4, num_stages=2)

    return out_f.view(B, H, T, V)
