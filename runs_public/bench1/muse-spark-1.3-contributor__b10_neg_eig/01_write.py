import torch
import triton
import triton.language as tl


@triton.jit
def _delta_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, g_ptr, beta_ptr,
    sQB, sQH, sQT, sQK,
    sKB, sKH, sKT, sKK,
    sVB, sVH, sVT, sVV,
    sOB, sOH, sOT, sOV,
    sGB, sGH, sGT,
    sBB, sBH, sBT,
    H, T, K, V, NVB,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    vb = pid % NVB
    tmp = pid // NVB
    h = tmp % H
    b = tmp // H
    v_start = vb * BLOCK_V

    k_offs = tl.arange(0, BLOCK_K)
    v_offs = tl.arange(0, BLOCK_V)
    mask_k = k_offs < K
    mask_v = (v_start + v_offs) < V

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in range(T):
        k_ptrs = k_ptr + b * sKB + h * sKH + t * sKT + k_offs * sKK
        q_ptrs = q_ptr + b * sQB + h * sQH + t * sQT + k_offs * sQK
        v_ptrs = v_ptr + b * sVB + h * sVH + t * sVT + (v_start + v_offs) * sVV
        o_ptrs = o_ptr + b * sOB + h * sOH + t * sOT + (v_start + v_offs) * sOV

        k_vec = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
        g_t = tl.load(g_ptr + b * sGB + h * sGH + t * sGT).to(tl.float32)
        beta_t = tl.load(beta_ptr + b * sBB + h * sBH + t * sBT).to(tl.float32)

        d = tl.exp(g_t)
        b2 = beta_t * 2.0

        S = S * d
        k_col = k_vec[:, None]
        pred = tl.sum(S * k_col, axis=0)
        diff = v_vec - pred
        S = S + (k_col * b2) * diff[None, :]
        q_col = q_vec[:, None]
        out_vec = tl.sum(S * q_col, axis=0) * scale
        tl.store(o_ptrs, out_vec, mask=mask_v)


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # e, gamma are unused by the reference math, kept for signature compat
    B, H, T, K = q.shape
    V = v.shape[-1]

    if torch.is_tensor(scale):
        scale_f = float(scale.detach().to(torch.float32).reshape(-1)[0].item()) if scale.numel() >= 1 else 1.0
    else:
        scale_f = float(scale)

    q32 = q.to(dtype=torch.float32)
    k32 = k.to(dtype=torch.float32)
    v32 = v.to(dtype=torch.float32)
    g32 = g.to(dtype=torch.float32)
    beta32 = beta.to(dtype=torch.float32)

    out = torch.empty(B, H, T, V, dtype=torch.float32, device=q.device)

    if T == 0 or K == 0 or V == 0 or B == 0 or H == 0:
        return out

    BLOCK_K = _next_pow2(K)
    if BLOCK_K <= 128:
        BLOCK_V = 32 if V >= 32 else _next_pow2(V)
    elif BLOCK_K <= 256:
        BLOCK_V = 16 if V >= 16 else _next_pow2(V)
    else:
        BLOCK_V = 8 if V >= 8 else _next_pow2(V)

    NVB = (V + BLOCK_V - 1) // BLOCK_V
    grid = (B * H * NVB,)

    sQB, sQH, sQT, sQK = q32.stride(0), q32.stride(1), q32.stride(2), q32.stride(3)
    sKB, sKH, sKT, sKK = k32.stride(0), k32.stride(1), k32.stride(2), k32.stride(3)
    sVB, sVH, sVT, sVV = v32.stride(0), v32.stride(1), v32.stride(2), v32.stride(3)
    sOB, sOH, sOT, sOV = out.stride(0), out.stride(1), out.stride(2), out.stride(3)
    sGB, sGH, sGT = g32.stride(0), g32.stride(1), g32.stride(2)
    sBB, sBH, sBT = beta32.stride(0), beta32.stride(1), beta32.stride(2)

    prod = BLOCK_K * BLOCK_V
    num_warps = 4 if prod <= 2048 else 8

    _delta_fwd_kernel[grid](
        q32, k32, v32, out, g32, beta32,
        sQB, sQH, sQT, sQK,
        sKB, sKH, sKT, sKK,
        sVB, sVH, sVT, sVV,
        sOB, sOH, sOT, sOV,
        sGB, sGH, sGT,
        sBB, sBH, sBT,
        H, T, K, V, NVB,
        scale_f,
        BLOCK_K, BLOCK_V,
        num_warps=num_warps, num_stages=1,
    )
    return out
