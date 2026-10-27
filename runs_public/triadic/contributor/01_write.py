import torch
import triton
import triton.language as tl


@triton.jit
def _gated_delta_fwd_kernel(
    q_ptr, k_ptr, v_ptr, q2_ptr, k2_ptr, dec_ptr, beta_ptr, out_ptr,
    T, D, E, V, BH, NV,
    scale,
    BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    d_idx = tl.arange(0, BLOCK_D)
    e_idx = tl.arange(0, BLOCK_E)
    v_idx = vid * BLOCK_V + tl.arange(0, BLOCK_V)
    d_mask = d_idx < D
    e_mask = e_idx < E
    v_mask = v_idx < V
    S = tl.zeros((BLOCK_D, BLOCK_E, BLOCK_V), dtype=tl.float32)
    for t in range(T):
        k_d = tl.load(k_ptr + bh * T * D + t * D + d_idx, mask=d_mask, other=0.0).to(tl.float32)
        q_d = tl.load(q_ptr + bh * T * D + t * D + d_idx, mask=d_mask, other=0.0).to(tl.float32)
        k2_e = tl.load(k2_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=0.0).to(tl.float32)
        q2_e = tl.load(q2_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=0.0).to(tl.float32)
        dec_e = tl.load(dec_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=1.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + bh * T + t).to(tl.float32)
        v_v = tl.load(v_ptr + bh * T * V + t * V + v_idx, mask=v_mask, other=0.0).to(tl.float32)

        k_3 = tl.expand_dims(tl.expand_dims(k_d, 1), 2)
        q_3 = tl.expand_dims(tl.expand_dims(q_d, 1), 2)
        k2_3 = tl.expand_dims(tl.expand_dims(k2_e, 0), 2)
        q2_3 = tl.expand_dims(tl.expand_dims(q2_e, 0), 2)
        dec_3 = tl.expand_dims(tl.expand_dims(dec_e, 0), 2)

        S = S * dec_3

        tmp_p = S * k_3 * k2_3
        sum_d = tl.sum(tmp_p, axis=0)
        pred = tl.sum(sum_d, axis=0)
        delta = (v_v - pred) * beta_t

        kappa = k_3 * k2_3
        delta_3 = tl.expand_dims(tl.expand_dims(delta, 0), 0)
        S = S + kappa * delta_3

        tmp_o = S * q_3 * q2_3
        sum_d2 = tl.sum(tmp_o, axis=0)
        out_v = tl.sum(sum_d2, axis=0) * scale
        tl.store(out_ptr + bh * T * V + t * V + v_idx, out_v, mask=v_mask)


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def kernel_function(q, k, v, q2, k2, g, beta, scale):
    B, H, T, D = q.shape
    E = q2.shape[-1]
    Vd = v.shape[-1]
    BH = B * H

    q3 = q.contiguous().view(BH, T, D)
    k3 = k.contiguous().view(BH, T, D)
    v3 = v.contiguous().view(BH, T, Vd)
    q23 = q2.contiguous().view(BH, T, E)
    k23 = k2.contiguous().view(BH, T, E)
    # cheap elementwise prep, keeps torch.exp exact vs tl.exp
    dec = torch.exp(g.to(torch.float32)).contiguous().view(BH, T, E)
    b3 = beta.contiguous().view(BH, T)
    out3 = torch.empty((BH, T, Vd), device=q.device, dtype=torch.float32)

    if torch.is_tensor(scale):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)

    BLOCK_D = _next_pow2(D)
    BLOCK_E = _next_pow2(E)
    BLOCK_V = 16
    NV = (Vd + BLOCK_V - 1) // BLOCK_V
    grid = (BH * NV,)

    _gated_delta_fwd_kernel[grid](
        q3, k3, v3, q23, k23, dec, b3, out3,
        T, D, E, Vd, BH, NV,
        scale_f,
        BLOCK_D=BLOCK_D, BLOCK_E=BLOCK_E, BLOCK_V=BLOCK_V,
        num_warps=4, num_stages=1,
    )
    return out3.view(B, H, T, Vd)
