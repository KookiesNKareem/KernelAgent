```python
import torch
import triton
import triton.language as tl


@triton.jit
def _delta_gated_kernel(
    Qptr, Kptr, Vptr, Optr, Gptr, Bptr, Sptr,
    scale,
    BH, T, Kdim, Vdim,
    chunk_start,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < Kdim
    v_cols = pid_v * BLOCK_V + offs_v
    mask_v = v_cols < Vdim
    s_ptrs = Sptr + pid_bh * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
    s_mask = mask_k[:, None] & mask_v[None, :]
    S = tl.load(s_ptrs, mask=s_mask, other=0.0)
    q_bh_off = pid_bh * T * Kdim
    k_bh_off = pid_bh * T * Kdim
    v_bh_off = pid_bh * T * Vdim
    o_bh_off = pid_bh * T * Vdim
    g_bh_off = pid_bh * T
    b_bh_off = pid_bh * T
    for t_off in range(BLOCK_T):
        t = chunk_start + t_off
        k_vec = tl.load(Kptr + k_bh_off + t * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        q_vec = tl.load(Qptr + q_bh_off + t * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(Vptr + v_bh_off + t * Vdim + v_cols, mask=mask_v, other=0.0).to(tl.float32)
        g_val = tl.load(Gptr + g_bh_off + t).to(tl.float32)
        beta_val = tl.load(Bptr + b_bh_off + t).to(tl.float32)
        decay = tl.exp(g_val)
        S = S * decay
        pred = tl.sum(S * k_vec[:, None], axis=0)
        delta = (v_vec - pred) * beta_val
        S = S + k_vec[:, None] * delta[None, :]
        out_vec = tl.sum(S * (q_vec * scale)[:, None], axis=0)
        tl.store(Optr + o_bh_off + t * Vdim + v_cols, out_vec, mask=mask_v)
    tl.store(s_ptrs, S, mask=s_mask)


# ---- backward helpers: all use fixed 32x32 tiles, single step, no unrolled time loop ----
@triton.jit
def _recompute_kernel(
    Kptr, Vptr, Gptr, Bptr, SstartPtr, ScratchPtr,
    BH, T, Kdim, Vdim, chunk_start, cur,
    NK: tl.constexpr, NM: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, CMAX: tl.constexpr,
):
    bh = tl.program_id(0)
    nk = tl.program_id(1)
    nm = tl.program_id(2)
    offs_k = nk * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_v = nm * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < Kdim
    mask_v = offs_v < Vdim
    s_mask = mask_k[:, None] & mask_v[None, :]
    s_ptrs0 = SstartPtr + bh * Kdim * Vdim + offs_k[:, None] * Vdim + offs_v[None, :]
    S = tl.load(s_ptrs0, mask=s_mask, other=0.0).to(tl.float32)
    for j in range(CMAX):
        if j < cur:
            t = chunk_start + j
            k_vec = tl.load(Kptr + (bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            v_vec = tl.load(Vptr + (bh * T + t) * Vdim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            g_val = tl.load(Gptr + bh * T + t).to(tl.float32)
            b_val = tl.load(Bptr + bh * T + t).to(tl.float32)
            decay = tl.exp(g_val)
            Smid = S * decay
            pred = tl.sum(Smid * k_vec[:, None], axis=0)
            ev = v_vec - pred
            S = Smid + k_vec[:, None] * (b_val * ev[None, :])
            sc_ptrs = ScratchPtr + (j * BH + bh) * Kdim * Vdim + offs_k[:, None] * Vdim + offs_v[None, :]
            tl.store(sc_ptrs, S, mask=s_mask)


@triton.jit
def _pred_kernel(
    SprevPtr, Kptr, Gptr, PredPtr,
    BH, T, Kdim, Vdim, t, NK,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(0)
    nk = tl.program_id(1)
    nm = tl.program_id(2)
    offs_k = nk * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_v = nm * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < Kdim
    mask_v = offs_v < Vdim
    S = tl.load(SprevPtr + bh * Kdim * Vdim + offs_k[:, None] * Vdim + offs_v[None, :], mask=mask_k[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
    k_vec = tl.load(Kptr + (bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    g_val = tl.load(Gptr + bh * T + t).to(tl.float32)
    Smid = S * tl.exp(g_val)
    part = tl.sum(Smid * k_vec[:, None], axis=0)
    tl.store(PredPtr + (bh * NK + nk) * Vdim + offs_v, part, mask=mask_v)


@triton.jit
def _dtk_kernel(
    Fptr, Qptr, Kptr, DOptr, DtkPtr, scale,
    BH, T, Kdim, Vdim, t, NK,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(0)
    nk = tl.program_id(1)
    nm = tl.program_id(2)
    offs_k = nk * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_v = nm * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < Kdim
    mask_v = offs_v < Vdim
    F = tl.load(Fptr + bh * Kdim * Vdim + offs_k[:, None] * Vdim + offs_v[None, :], mask=mask_k[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
    q_vec = tl.load(Qptr + (bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    k_vec = tl.load(Kptr + (bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    do_vec = tl.load(DOptr + (bh * T + t) * Vdim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    A = F + (q_vec * scale)[:, None] * do_vec[None, :]
    part = tl.sum(A * k_vec[:, None], axis=0)
    tl.store(DtkPtr + (bh * NK + nk) * Vdim + offs_v, part, mask=mask_v)


@triton.jit
def _bwd_comb_kernel(
    Fptr, SprevPtr, StPtr, Qptr, Kptr, DOptr, Eptr, DtkPtr,
    AePtr, SdtkPtr, DqPtr, SumPtr, scale,
    BH, T, Kdim, Vdim, t, NK, NM,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(0)
    nk = tl.program_id(1)
    nm = tl.program_id(2)
    offs_k = nk * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_v = nm * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < Kdim
    mask_v = offs_v < Vdim
    s_mask = mask_k[:, None] & mask_v[None, :]
    f_ptrs = Fptr + bh * Kdim * Vdim + offs_k[:, None] * Vdim + offs_v[None, :]
    F = tl.load(f_ptrs, mask=s_mask, other=0.0).to(tl.float32)
    Sp = tl.load(SprevPtr + bh * Kdim * Vdim + offs_k[:, None] * Vdim + offs_v[None, :], mask=s_mask, other=0.0).to(tl.float32)
    St = tl.load(StPtr + bh * Kdim * Vdim + offs_k[:, None] * Vdim + offs_v[None, :], mask=s_mask, other=0.0).to(tl.float32)
    q_vec = tl.load(Qptr + (bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    k_vec = tl.load(Kptr + (bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    do_vec = tl.load(DOptr + (bh * T + t) * Vdim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    e_vec = tl.load(Eptr + bh * Vdim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    dtk_vec = tl.load(DtkPtr + bh * Vdim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    g_val = tl.load(SprevPtr + 0 * 0, mask=None)  # dummy to keep signature stable
    # real g/b loaded via extra pointers hidden in Eptr path? reload from global below via offsets:
    # NOTE: g/b are passed through Eptr/DtkPtr slots is wrong; load them from Q/K base is wrong.
    # To keep kernel small we recompute decay/beta from dedicated pointers:
    # (these pointers are aliased: Eptr is [BH,V], so we cannot get g; instead use St/Sp ratio is unstable.
    #  Workaround: pass g/b via scale-like scalars is impossible per-bh.
    #  So we load g/b from DOptr-adjacent memory is wrong.
    #  Fix: use dedicated loads via Fptr offsets is wrong.
    #  Correct approach: reload g/b from separate tensors passed as AePtr/SdtkPtr metadata is wrong.
    #  Simplest correct: compute decay from Smid relation is unstable, so we add explicit G/B pointers.)
    A = F + (q_vec * scale)[:, None] * do_vec[None, :]
    ae = tl.sum(A * e_vec[None, :], axis=1)
    tl.store(AePtr + ((bh * NK + nk) * NM + nm) * BLOCK_K + tl.arange(0, BLOCK_K), ae, mask=mask_k)


@triton.jit
def _bwd_comb2_kernel(
    Fptr, SprevPtr, StPtr, Qptr, Kptr, DOptr, Eptr, DtkPtr, Gptr, Bptr,
    AePtr, SdtkPtr, DqPtr, SumPtr, scale,
    BH, T, Kdim, Vdim, t, NK, NM,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(0)
    nk = tl.program_id(1)
    nm = tl.program_id(2)
    offs_k = nk * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_v = nm * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_bk = tl.arange(0, BLOCK_K)
    mask_k = offs_k < Kdim
    mask_v = offs_v < Vdim
    s_mask = mask_k[:, None] & mask_v[None, :]
    f_ptrs = Fptr + bh * Kdim * Vdim + offs_k[:, None] * Vdim + offs_v[None, :]
    F = tl.load(f_ptrs, mask=s_mask, other=0.0).to(tl.float32)
    Sp = tl.load(SprevPtr + bh * Kdim * Vdim + offs_k[:, None] * Vdim + offs_v[None, :], mask=s_mask, other=0.0).to(tl.float32)
    St = tl.load(StPtr + bh * Kdim * Vdim + offs_k[:, None] * Vdim + offs_v[None, :], mask=s_mask, other=0.0).to(tl.float32)
    q_vec = tl.load(Qptr + (bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    k_vec = tl.load(Kptr + (bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    do_vec = tl.load(DOptr + (bh * T + t) * Vdim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    e_vec = tl.load(Eptr + bh * Vdim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    dtk_vec = tl.load(DtkPtr + bh * Vdim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    g_val = tl.load(Gptr + bh * T + t).to(tl.float32)
    b_val = tl.load(Bptr + bh * T + t).to(tl.float32)
    decay = tl.exp(g_val)
    Smid = Sp * decay
    A = F + (q_vec * scale)[:, None] * do_vec[None, :]
    ae = tl.sum(A * e_vec[None, :], axis=1)
    sdtk = tl.sum(Smid * dtk_vec[None, :], axis=1)
    dq = scale * tl.sum(St * do_vec[None, :], axis=1)
    ssum = tl.sum(A * Smid)
    Fnew = (A - b_val * k_vec[:, None] * dtk_vec[None, :]) * decay
    tl.store(f_ptrs, Fnew, mask=s_mask)
    base = ((bh * NK + nk) * NM + nm) * BLOCK_K + offs_bk
    tl.store(AePtr + base, ae, mask=mask_k)
    tl.store(SdtkPtr + base, sdtk, mask=mask_k)
    tl.store(DqPtr + base, dq, mask=mask_k)
    tl.store(SumPtr + (bh * NK + nk) * NM + nm, ssum)


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


class _GatedDeltaFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        Vdim = v.shape[-1]
        dev = q.device
        if isinstance(scale, torch.Tensor):
            scale_f = float(scale.detach().to(torch.float32).item())
        else:
            scale_f = float(scale)
        if T == 0:
            out = torch.empty(B, H, 0, Vdim, dtype=torch.float32, device=dev)
            ctx._empty = True
            ctx._dev = dev
            ctx._dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
            ctx._e_shape = e.shape
            ctx._gamma_shape = gamma.shape
            return out
        ctx._empty = False
        qf = q.contiguous().reshape(B * H, T, K)
        kf = k.contiguous().reshape(B * H, T, K)
        vf = v.contiguous().reshape(B * H, T, Vdim)
        gf = g.contiguous().reshape(B * H, T)
        bf = beta.contiguous().reshape(B * H, T)
        BH = B * H
        out_ = torch.empty(BH, T, Vdim, dtype=torch.float32, device=dev)
        state = torch.zeros(BH, K, Vdim, dtype=torch.float32, device=dev)
        BK = _next_pow2(K)
        BVf = _next_pow2(Vdim)
        if BK * BVf <= 4096:
            BV = BVf
        else:
            tgt = 4096 // BK
            if tgt < 1:
                tgt = 1
            BV = 1 << (tgt.bit_length() - 1)
        NV = (Vdim + BV - 1) // BV
        grid = (BH, NV)
        C = 64
        N = (T + C - 1) // C
        chk = torch.zeros(N + 1, BH, K, Vdim, dtype=torch.float32, device=dev)
        for ci in range(N):
            cs = ci * C
            cur = C if cs + C <= T else T - cs
            _delta_gated_kernel[grid](qf, kf, vf, out_, gf, bf, state, scale_f,
                                      BH, T, K, Vdim, cs,
                                      BLOCK_K=BK, BLOCK_V=BV, BLOCK_T=cur, num_warps=4)
            chk[ci + 1].copy_(state)
        ctx.save_for_backward(qf, kf, vf, gf, bf)
        ctx.chk = chk
        ctx._shapes = (B, H, T, K, Vdim, BH, BK, BV, NV, C, N)
        ctx._dev = dev
        ctx.scale_f = scale_f
        ctx._dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
        ctx._e_shape = e.shape
        ctx._gamma_shape = gamma.shape
        return out_.reshape(B, H, T, Vdim)

    @staticmethod
    def backward(ctx, grad_out):
        dev = ctx._dev
        if ctx._empty:
            return (None, None, None, None, None, None, None, None)
        qf, kf, vf, gf, bf = ctx.saved_tensors
        chk = ctx.chk
        B, H, T, K, Vdim, BH, BK, BV, NV, C, N = ctx._shapes
        scale_f = ctx.scale_f
        go = grad_out.contiguous().to(torch.float32).reshape(BH, T, Vdim)
        BK2 = 32
        BV2 = 32
        NK = (K + 31) // 32
        NM = (Vdim + 31) // 32
        CMAX = 64
        scratch = torch.empty(CMAX, BH, K, Vdim, dtype=torch.float32, device=dev)
        Fbuf = torch.zeros(BH, K, Vdim, dtype=torch.float32, device=dev)
        pred_parts = torch.empty(BH, NK, Vdim, dtype=torch.float32, device=dev)
        dtk_parts = torch.empty(BH, NK, Vdim, dtype=torch.float32, device=dev)
        ae_parts = torch.empty(BH, NK, NM, BK2, dtype=torch.float32, device=dev)
        sdtk_parts = torch.empty(BH, NK, NM, BK2, dtype=torch.float32, device=dev)
        dq_parts = torch.empty(BH, NK, NM, BK2, dtype=torch.float32, device=dev)
        sum_parts = torch.empty(BH, NK, NM, dtype=torch.float32, device=dev)
        dq_full = torch.zeros(BH, T, K, dtype=torch.float32, device=dev)
        dk_full = torch.zeros(BH, T, K, dtype=torch.float32, device=dev)
        dv_full = torch.zeros(BH, T, Vdim, dtype=torch.float32, device=dev)
        dg_full = torch.zeros(BH, T, dtype=torch.float32, device=dev)
        db_full = torch.zeros(BH, T, dtype=torch.float32, device=dev)
        grid2 = (BH, NK, NM)
        for ci in range(N - 1, -1, -1):
            cs = ci * C
            cur = C if cs + C <= T else T - cs
            Sstart = chk[ci]
            _recompute_kernel[grid2](kf, vf, gf, bf, Sstart, scratch,
                                     BH, T, K, Vdim, cs, cur,
                                     NK, NM, BLOCK_K=BK2, BLOCK_V=BV2, CMAX=CMAX, num_warps=4)
            for j in range(cur - 1, -1, -1):
                t = cs + j
                Sprev_t = Sstart if j == 0 else scratch[j - 1]
                St_t = scratch[j]
                _pred_kernel[grid2](Sprev_t, kf, gf, pred_parts,
                                    BH, T, K, Vdim, t, NK,
                                    BLOCK_K=BK2, BLOCK_V=BV2, num_warps=4)
                pred = pred_parts.sum(dim=1)
                vf_t = vf[:, t, :].to(torch.float32)
                e_t = vf_t - pred
                _dtk_kernel[grid2](Fbuf, qf, kf, go, dtk_parts, scale_f,
                                   BH, T, K, Vdim, t, NK,
                                   BLOCK_K=BK2, BLOCK_V=BV2, num_warps=4)
                dtk = dtk_parts.sum(dim=1)
                _bwd_comb2_kernel[grid2](Fbuf, Sprev_t, St_t, qf, kf, go, e_t, dtk, gf, bf,
                                         ae_parts, sdtk_parts, dq_parts, sum_parts, scale_f,
                                         BH, T, K, Vdim, t, NK, NM,
                                         BLOCK_K=BK2, BLOCK_V=BV2, num_warps=4)
                ae = ae_parts.sum(dim=2).reshape(BH, NK * BK2)[:, :K]
                sdtk = sdtk_parts.sum(dim=2).reshape(BH, NK * BK2)[:, :K]
                dq_p = dq_parts.sum(dim=2).reshape(BH, NK * BK2)[:, :K]
                sumA = sum_parts.sum(dim=(1, 2))
                b_t = bf[:, t]
                dq_full[:, t, :] = dq_p
                dk_full[:, t, :] = b_t[:, None] * (ae - sdtk)
                dv_full[:, t, :] = b_t[:, None] * dtk
                dg_full[:, t] = sumA - b_t * (dtk * pred).sum(dim=1)
                db_full[:, t] = (dtk * e_t).sum(dim=1)
        dq = dq_full.reshape(B, H, T, K).to(ctx._dtypes[0])
        dk = dk_full.reshape(B, H, T, K).to(ctx._dtypes[1])
        dv = dv_full.reshape(B, H, T, Vdim).to(ctx._dtypes[2])
        dg = dg_full.reshape(B, H, T).to(ctx._dtypes[4])
        db = db_full.reshape(B, H, T).to(ctx._dtypes[5])
        de = torch.zeros(ctx._e_shape, dtype=ctx._dtypes[3], device=dev)
        dgam = torch.zeros(ctx._gamma_shape, dtype=ctx._dtypes[6], device=dev)
        return (dq, dk, dv, de, dg, db, dgam, None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _GatedDeltaFunc.apply(q, k, v, e, g, beta, gamma, scale)
```