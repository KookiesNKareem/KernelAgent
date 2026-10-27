```python
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, out_ptr,
    T, K, V,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    EVEN_K: tl.constexpr, EVEN_V: tl.constexpr,
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
    d_base = d_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    o_base = out_ptr + bh * T * V
    b_base = beta_ptr + bh * T

    for t in range(T):
        if EVEN_K:
            q_t = tl.load(q_base + t * K + k_offs).to(tl.float32)
            k_t = tl.load(k_base + t * K + k_offs).to(tl.float32)
            dec = tl.load(d_base + t * K + k_offs).to(tl.float32)
        else:
            q_t = tl.load(q_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            k_t = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            dec = tl.load(d_base + t * K + k_offs, mask=k_mask, other=1.0).to(tl.float32)
        if EVEN_V:
            v_t = tl.load(v_base + t * V + v_offs).to(tl.float32)
        else:
            v_t = tl.load(v_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
        b_t = tl.load(b_base + t).to(tl.float32)

        S = S * tl.expand_dims(dec, 1)

        q_scaled = q_t * scale
        out_t = tl.sum(S * tl.expand_dims(q_scaled, 1), 0)
        if EVEN_V:
            tl.store(o_base + t * V + v_offs, out_t)
        else:
            tl.store(o_base + t * V + v_offs, out_t, mask=v_mask)

        pred = tl.sum(S * tl.expand_dims(k_t, 1), 0)
        if EVEN_V:
            diff = v_t - pred
            b_diff = b_t * diff
        else:
            pred = tl.where(v_mask, pred, 0.0)
            diff = tl.where(v_mask, v_t - pred, 0.0)
            b_diff = b_t * diff
        S = S + tl.expand_dims(k_t, 1) * tl.expand_dims(b_diff, 0)


def _n2(x):
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


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
    bc = beta.contiguous()
    # cheap elementwise prep: exp exactly as reference (torch exp), hoisted from loop
    d = torch.exp(g)

    BH = B * H
    qc3 = qc.reshape(BH, T, Kd)
    kc3 = kc.reshape(BH, T, Kd)
    dc3 = d.reshape(BH, T, Kd)
    vc3 = vc.reshape(BH, T, Vd)
    bc2 = bc.reshape(BH, T)
    out3 = torch.empty(BH, T, Vd, device=q.device, dtype=torch.float32)

    BLOCK_K = _n2(Kd)
    if BLOCK_K < 16:
        BLOCK_K = 16
    if Vd <= 32:
        BLOCK_V = _n2(Vd)
        if BLOCK_V < 16:
            BLOCK_V = 16
    else:
        BLOCK_V = 32

    EVEN_K = (Kd == BLOCK_K)
    EVEN_V = (Vd % BLOCK_V == 0)
    nV = (Vd + BLOCK_V - 1) // BLOCK_V

    if BLOCK_K * BLOCK_V > 2048:
        num_warps = 8
    else:
        num_warps = 4

    grid = (BH, nV)
    _fwd_kernel[grid](
        qc3, kc3, vc3, dc3, bc2, out3,
        T, Kd, Vd,
        scale_f,
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        EVEN_K=EVEN_K, EVEN_V=EVEN_V,
        num_warps=num_warps, num_stages=2,
    )
    return out3.reshape(B, H, T, Vd)
```