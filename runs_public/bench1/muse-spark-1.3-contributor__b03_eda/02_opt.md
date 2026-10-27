```python
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, Kptr, Vptr, Eptr, Gptr, Bptr, Cptr, Optr,
    scale,
    T,
    Kdim,
    Vdim,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    v_off = pid_v * BLOCK_V + off_v

    Qbh = Q + pid_bh * T * Kdim
    Kb = Kptr + pid_bh * T * Kdim
    Eb = Eptr + pid_bh * T * Kdim
    Vb = Vptr + pid_bh * T * Vdim
    Gb = Gptr + pid_bh * T
    Bb = Bptr + pid_bh * T
    Cb = Cptr + pid_bh * T
    Ob = Optr + pid_bh * T * Vdim

    if EVEN_K:
        k_mask = None
    else:
        k_mask = off_k < Kdim
    if EVEN_V:
        v_mask = None
    else:
        v_mask = v_off < Vdim

    q_ptrs = Qbh + off_k
    k_ptrs = Kb + off_k
    e_ptrs = Eb + off_k
    v_ptrs = Vb + v_off
    o_ptrs = Ob + v_off
    g_ptr = Gb
    b_ptr = Bb
    c_ptr = Cb

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in range(T):
        if EVEN_K:
            qk = tl.load(q_ptrs).to(tl.float32)
            kk = tl.load(k_ptrs).to(tl.float32)
            ek = tl.load(e_ptrs).to(tl.float32)
        else:
            qk = tl.load(q_ptrs, mask=k_mask, other=0.0).to(tl.float32)
            kk = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
            ek = tl.load(e_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        if EVEN_V:
            vv = tl.load(v_ptrs).to(tl.float32)
        else:
            vv = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
        d = tl.load(g_ptr)
        bb = tl.load(b_ptr)
        cc = tl.load(c_ptr)

        S = S * d
        se = tl.sum(S * ek[:, None], axis=0)
        S = S - cc * ek[:, None] * se[None, :]
        pred = tl.sum(S * kk[:, None], axis=0)
        diff = (vv - pred) * bb
        S = S + kk[:, None] * diff[None, :]

        qs = qk * scale
        o = tl.sum(S * qs[:, None], axis=0)
        if EVEN_V:
            tl.store(o_ptrs, o)
        else:
            tl.store(o_ptrs, o, mask=v_mask)

        q_ptrs += Kdim
        k_ptrs += Kdim
        e_ptrs += Kdim
        v_ptrs += Vdim
        o_ptrs += Vdim
        g_ptr += 1
        b_ptr += 1
        c_ptr += 1


def _npo2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if isinstance(scale, torch.Tensor):
        if scale.numel() == 0:
            scale_f = 1.0
        else:
            scale_f = float(scale.to(torch.float32).reshape(-1)[0].item())
    else:
        scale_f = float(scale)

    B, H, T, Kdim = q.shape
    Vdim = v.shape[-1]
    BH = B * H

    if T == 0 or Vdim == 0 or BH == 0 or Kdim == 0:
        return torch.empty(B, H, T, Vdim, dtype=torch.float32, device=q.device)

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    e_c = e.contiguous()
    g_c = g.contiguous().to(torch.float32)
    beta_c = beta.contiguous().to(torch.float32)
    gamma_c = gamma.contiguous().to(torch.float32)

    # cheap elementwise prep: exp outside the loop
    g_exp = torch.exp(g_c)

    if Kdim * Vdim <= 16384:
        BLOCK_K = max(16, _npo2(Kdim))
        BLOCK_V = max(16, _npo2(Vdim))
    else:
        BLOCK_K = max(16, _npo2(Kdim))
        target = 8192 // BLOCK_K
        if target < 16:
            target = 16
        BLOCK_V = 1 << (target.bit_length() - 1)
        BLOCK_V = min(BLOCK_V, _npo2(Vdim))
        BLOCK_V = max(16, BLOCK_V)

    NV = (Vdim + BLOCK_V - 1) // BLOCK_V
    EVEN_K = (Kdim == BLOCK_K)
    EVEN_V = (Vdim % BLOCK_V == 0)

    out = torch.empty(B, H, T, Vdim, dtype=torch.float32, device=q.device)

    num_warps = 8 if (BLOCK_K * BLOCK_V > 4096) else 4
    grid = (BH, NV)

    _fwd_kernel[grid](
        q_c, k_c, v_c, e_c, g_exp, beta_c, gamma_c, out,
        scale_f, T, Kdim, Vdim,
        BLOCK_K, BLOCK_V, EVEN_K, EVEN_V,
        num_warps=num_warps,
    )
    return out
```
