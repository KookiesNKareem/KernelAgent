import torch
import triton
import triton.language as tl


@triton.jit
def _gated_delta_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    B, H, T, K, V,
    s_qb, s_qh, s_qt, s_qk,
    s_kb, s_kh, s_kt, s_kk,
    s_vb, s_vh, s_vt, s_vn,
    s_gb, s_gh, s_gt,
    s_bb, s_bh, s_bt,
    s_ob, s_oh, s_ot, s_on,
    scale,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H

    q_base = q_ptr + b * s_qb + h * s_qh
    k_base = k_ptr + b * s_kb + h * s_kh
    v_base = v_ptr + b * s_vb + h * s_vh
    g_base = g_ptr + b * s_gb + h * s_gh
    beta_base = beta_ptr + b * s_bb + h * s_bh
    o_base = o_ptr + b * s_ob + h * s_oh

    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    mask_k = off_k < K
    mask_v = off_v < V

    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    for t in tl.range(0, T):
        q_t = tl.load(q_base + t * s_qt + off_k * s_qk, mask=mask_k, other=0.0)
        k_t = tl.load(k_base + t * s_kt + off_k * s_kk, mask=mask_k, other=0.0)
        v_t = tl.load(v_base + t * s_vt + off_v * s_vn, mask=mask_v, other=0.0)
        g_t = tl.load(g_base + t * s_gt).to(tl.float32)
        beta_t = tl.load(beta_base + t * s_bt).to(tl.float32)

        q_f = q_t.to(tl.float32) * scale
        k_f = k_t.to(tl.float32)
        v_f = v_t.to(tl.float32)

        d = tl.exp(g_t)
        S = S * d

        k_col = tl.expand_dims(k_f, 1)  # [BK,1]
        pred = tl.sum(S * k_col, axis=0)  # [BV]

        diff = (v_f - pred) * beta_t  # [BV]
        diff_row = tl.expand_dims(diff, 0)  # [1,BV]

        S = S + k_col * diff_row

        q_col = tl.expand_dims(q_f, 1)  # [BK,1]
        out_t = tl.sum(S * q_col, axis=0)  # [BV]

        tl.store(o_base + t * s_ot + off_v * s_on, out_t, mask=mask_v)


def _next_pow2(x: int):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def kernel_function(q, k, v, g, beta, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    if T == 0:
        return out

    BLOCK_K = _next_pow2(K)
    BLOCK_V = _next_pow2(V)

    s_qb, s_qh, s_qt, s_qk = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
    s_kb, s_kh, s_kt, s_kk = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
    s_vb, s_vh, s_vt, s_vn = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
    s_gb, s_gh, s_gt = g.stride(0), g.stride(1), g.stride(2)
    s_bb, s_bh, s_bt = beta.stride(0), beta.stride(1), beta.stride(2)
    s_ob, s_oh, s_ot, s_on = out.stride(0), out.stride(1), out.stride(2), out.stride(3)

    grid = (B * H,)
    num_warps = 8 if BLOCK_K * BLOCK_V > 4096 else 4

    _gated_delta_kernel[grid](
        q, k, v, g, beta, out,
        B, H, T, K, V,
        s_qb, s_qh, s_qt, s_qk,
        s_kb, s_kh, s_kt, s_kk,
        s_vb, s_vh, s_vt, s_vn,
        s_gb, s_gh, s_gt,
        s_bb, s_bh, s_bt,
        s_ob, s_oh, s_ot, s_on,
        float(scale),
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        num_warps=num_warps, num_stages=1,
    )
    return out
