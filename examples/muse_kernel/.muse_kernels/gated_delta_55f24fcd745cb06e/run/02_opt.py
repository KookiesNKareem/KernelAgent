import torch
import triton
import triton.language as tl


@triton.jit
def _gated_delta_fwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    T, K, V, NV, scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    EVEN_K: tl.constexpr, EVEN_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV

    seq_base = bh * T

    off_k = tl.arange(0, BLOCK_K)
    off_bv = tl.arange(0, BLOCK_V)
    off_v = vid * BLOCK_V + off_bv

    mask_k = off_k < K
    mask_v = off_v < V

    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    for t in tl.range(0, T):
        row = seq_base + t
        q_row = row * K
        v_row = row * V

        if EVEN_K:
            q_t = tl.load(q_ptr + q_row + off_k)
            k_t = tl.load(k_ptr + q_row + off_k)
        else:
            q_t = tl.load(q_ptr + q_row + off_k, mask=mask_k, other=0.0)
            k_t = tl.load(k_ptr + q_row + off_k, mask=mask_k, other=0.0)

        if EVEN_V:
            v_t = tl.load(v_ptr + v_row + off_v)
        else:
            v_t = tl.load(v_ptr + v_row + off_v, mask=mask_v, other=0.0)

        g_t = tl.load(g_ptr + row).to(tl.float32)
        b_t = tl.load(beta_ptr + row).to(tl.float32)

        q_f = q_t.to(tl.float32) * scale
        k_f = k_t.to(tl.float32)
        v_f = v_t.to(tl.float32)

        d = tl.exp(g_t)

        S = S * d

        k_col = tl.expand_dims(k_f, 1)
        pred = tl.sum(S * k_col, axis=0)
        diff = (v_f - pred) * b_t
        S = S + k_col * tl.expand_dims(diff, 0)

        q_col = tl.expand_dims(q_f, 1)
        out_t = tl.sum(S * q_col, axis=0)

        if EVEN_V:
            tl.store(o_ptr + v_row + off_v, out_t)
        else:
            tl.store(o_ptr + v_row + off_v, out_t, mask=mask_v)


def _next_pow2(x: int):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def kernel_function(q, k, v, g, beta, scale):
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not g.is_contiguous():
        g = g.contiguous()
    if not beta.is_contiguous():
        beta = beta.contiguous()

    out = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)
    if T == 0 or K == 0 or Vd == 0 or B == 0 or H == 0:
        return out

    BLOCK_K = _next_pow2(K)
    # keep per-program state ~4k fp32 to avoid spilling
    BLOCK_V = 32
    while BLOCK_K * BLOCK_V > 4096 and BLOCK_V > 8:
        BLOCK_V //= 2
    if Vd <= 16:
        BLOCK_V = _next_pow2(Vd)
    else:
        BLOCK_V = min(BLOCK_V, 32)
        # keep pow2
        if BLOCK_V not in (8, 16, 32):
            BLOCK_V = 32

    NV = (Vd + BLOCK_V - 1) // BLOCK_V
    EVEN_K = (K == BLOCK_K)
    EVEN_V = (Vd % BLOCK_V == 0)

    grid = (B * H * NV,)
    size = BLOCK_K * BLOCK_V
    num_warps = 4 if size <= 4096 else 8

    _gated_delta_fwd_kernel[grid](
        q, k, v, g, beta, out,
        T, K, Vd, NV, float(scale),
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        EVEN_K=EVEN_K, EVEN_V=EVEN_V,
        num_warps=num_warps, num_stages=2,
    )
    return out
