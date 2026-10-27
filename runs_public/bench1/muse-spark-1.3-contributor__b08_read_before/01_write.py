import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr,
    BH, T, K, V,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    g_base = g_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    o_base = out_ptr + bh * T * V
    b_base = beta_ptr + bh * T

    for t in range(T):
        q_t = tl.load(q_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        k_t = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        g_t = tl.load(g_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        v_t = tl.load(v_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
        b_t = tl.load(b_base + t).to(tl.float32)

        dec = tl.exp(g_t)
        S = S * tl.expand_dims(dec, 1)

        q_scaled = q_t * scale
        pred = tl.sum(S * tl.expand_dims(k_t, 1), 0)
        out_t = tl.sum(S * tl.expand_dims(q_scaled, 1), 0)
        tl.store(o_base + t * V + v_offs, out_t, mask=v_mask)

        diff = v_t - pred
        diff = tl.where(v_mask, diff, 0.0)
        b_diff = b_t * diff
        S = S + tl.expand_dims(k_t, 1) * tl.expand_dims(b_diff, 0)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, Kd = q.shape
    Vd = v.shape[-1]
    if T == 0 or Kd == 0 or Vd == 0 or B == 0 or H == 0:
        return torch.empty(B, H, T, Vd, device=q.device, dtype=torch.float32)
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc = g.contiguous()
    bc = beta.contiguous()

    BH = B * H
    qc3 = qc.reshape(BH, T, Kd)
    kc3 = kc.reshape(BH, T, Kd)
    gc3 = gc.reshape(BH, T, Kd)
    vc3 = vc.reshape(BH, T, Vd)
    bc2 = bc.reshape(BH, T)
    out3 = torch.empty(BH, T, Vd, device=q.device, dtype=torch.float32)

    def _n2(x):
        return 1 << ((x - 1).bit_length()) if x > 1 else 1
    BLOCK_K = _n2(Kd)
    if BLOCK_K < 16:
        BLOCK_K = 16
    BLOCK_V = 64
    while BLOCK_K * BLOCK_V > 8192 and BLOCK_V > 1:
        BLOCK_V //= 2
    nV = (Vd + BLOCK_V - 1) // BLOCK_V
    num_warps = 8 if BLOCK_K * BLOCK_V > 2048 else 4

    grid = (BH, nV)
    _fwd_kernel[grid](
        qc3, kc3, vc3, gc3, bc2, out3,
        BH, T, Kd, Vd,
        scale_f,
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        num_warps=num_warps, num_stages=1,
    )
    return out3.reshape(B, H, T, Vd)
