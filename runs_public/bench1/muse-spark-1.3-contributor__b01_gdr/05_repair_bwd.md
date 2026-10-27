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


@triton.jit
def _recompute_kernel(
    Kptr, Vptr, Gptr, Bptr, SstartPtr, ScratchPtr,
    BH, T, Kdim, Vdim, chunk_start,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_C: tl.constexpr, CMAX: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < Kdim
    v_cols = pid_v * BLOCK_V + offs_v
    mask_v = v_cols < Vdim
    s_mask = mask_k[:, None] & mask_v[None, :]
    s0_ptrs = SstartPtr + pid_bh * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
    S = tl.load(s0_ptrs, mask=s_mask, other=0.0).to(tl.float32)
    for t_off in range(BLOCK_C):
        t = chunk_start + t_off
        k_vec = tl.load(Kptr + (pid_bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(Vptr + (pid_bh * T + t) * Vdim + v_cols, mask=mask_v, other=0.0).to(tl.float32)
        g_val = tl.load(Gptr + pid_bh * T + t).to(tl.float32)
        b_val = tl.load(Bptr + pid_bh * T + t).to(tl.float32)
        decay = tl.exp(g_val)
        Smid = S * decay
        pred = tl.sum(Smid * k_vec[:, None], axis=0)
        evec = v_vec - pred
        S = Smid + k_vec[:, None] * (b_val * evec[None, :])
        sc_ptrs = ScratchPtr + pid_bh * CMAX * Kdim * Vdim + t_off * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
        tl.store(sc_ptrs, S, mask=s_mask)


@triton.jit
def _bwd_chunk_kernel(
    Qptr, Kptr, Vptr, DOptr, Gptr, Bptr,
    SstartPtr, ScratchPtr, Fptr, DVptr,
    DQptr, DKptr, DGptr, DBptr,
    scale,
    BH, T, Kdim, Vdim, NV, chunk_start,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_C: tl.constexpr, CMAX: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < Kdim
    v_cols = pid_v * BLOCK_V + offs_v
    mask_v = v_cols < Vdim
    s_mask = mask_k[:, None] & mask_v[None, :]
    s0_ptrs = SstartPtr + pid_bh * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
    S0 = tl.load(s0_ptrs, mask=s_mask, other=0.0).to(tl.float32)
    f_ptrs = Fptr + pid_bh * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
    Fcur = tl.load(f_ptrs, mask=s_mask, other=0.0).to(tl.float32)
    for ii in range(BLOCK_C):
        t_off = BLOCK_C - 1 - ii
        t = chunk_start + t_off
        q_vec = tl.load(Qptr + (pid_bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        k_vec = tl.load(Kptr + (pid_bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(Vptr + (pid_bh * T + t) * Vdim + v_cols, mask=mask_v, other=0.0).to(tl.float32)
        do_vec = tl.load(DOptr + (pid_bh * T + t) * Vdim + v_cols, mask=mask_v, other=0.0).to(tl.float32)
        g_val = tl.load(Gptr + pid_bh * T + t).to(tl.float32)
        b_val = tl.load(Bptr + pid_bh * T + t).to(tl.float32)
        decay = tl.exp(g_val)
        st_ptrs = ScratchPtr + pid_bh * CMAX * Kdim * Vdim + t_off * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
        St = tl.load(st_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        if t_off == 0:
            Sprev = S0
        else:
            sp_ptrs = ScratchPtr + pid_bh * CMAX * Kdim * Vdim + (t_off - 1) * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
            Sprev = tl.load(sp_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        Smid = Sprev * decay
        pred = tl.sum(Smid * k_vec[:, None], axis=0)
        evec = v_vec - pred
        QO = (q_vec * scale)[:, None] * do_vec[None, :]
        Amat = QO + Fcur
        dtk = tl.sum(Amat * k_vec[:, None], axis=0)
        dv = dtk * b_val
        tl.store(DVptr + (pid_bh * T + t) * Vdim + v_cols, dv, mask=mask_v)
        dq_part = tl.sum(St * do_vec[None, :], axis=1) * scale
        tl.store(DQptr + ((pid_bh * T + t) * NV + pid_v) * Kdim + offs_k, dq_part, mask=mask_k)
        de1 = tl.sum(Amat * evec[None, :], axis=1)
        de2 = tl.sum(Smid * dtk[None, :], axis=1)
        dk_part = (de1 - de2) * b_val
        tl.store(DKptr + ((pid_bh * T + t) * NV + pid_v) * Kdim + offs_k, dk_part, mask=mask_k)
        dg_part = tl.sum(Amat * Smid) - b_val * tl.sum(dtk * pred)
        tl.store(DGptr + (pid_bh * T + t) * NV + pid_v, dg_part)
        db_part = tl.sum(dtk * evec)
        tl.store(DBptr + (pid_bh * T + t) * NV + pid_v, db_part)
        Fcur = (Amat - b_val * k_vec[:, None] * dtk[None, :]) * decay
    tl.store(f_ptrs, Fcur, mask=s_mask)


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
        CMAX = 64
        scratch = torch.empty(BH, CMAX, K, Vdim, dtype=torch.float32, device=dev)
        DQp = torch.empty(BH, T, NV, K, dtype=torch.float32, device=dev)
        DKp = torch.empty(BH, T, NV, K, dtype=torch.float32, device=dev)
        DGp = torch.empty(BH, T, NV, dtype=torch.float32, device=dev)
        DBp = torch.empty(BH, T, NV, dtype=torch.float32, device=dev)
        DV = torch.empty(BH, T, Vdim, dtype=torch.float32, device=dev)
        Fbuf = torch.zeros(BH, K, Vdim, dtype=torch.float32, device=dev)
        grid = (BH, NV)
        for ci in range(N - 1, -1, -1):
            cs = ci * C
            cur = C if cs + C <= T else T - cs
            Sstart = chk[ci]
            _recompute_kernel[grid](kf, vf, gf, bf, Sstart, scratch,
                                    BH, T, K, Vdim, cs,
                                    BLOCK_K=BK, BLOCK_V=BV, BLOCK_C=cur, CMAX=CMAX, num_warps=4)
            _bwd_chunk_kernel[grid](qf, kf, vf, go, gf, bf, Sstart, scratch, Fbuf, DV,
                                    DQp, DKp, DGp, DBp,
                                    scale_f, BH, T, K, Vdim, NV, cs,
                                    BLOCK_K=BK, BLOCK_V=BV, BLOCK_C=cur, CMAX=CMAX, num_warps=4)
        dq = DQp.sum(dim=2).reshape(B, H, T, K).to(ctx._dtypes[0])
        dk = DKp.sum(dim=2).reshape(B, H, T, K).to(ctx._dtypes[1])
        dv = DV.reshape(B, H, T, Vdim).to(ctx._dtypes[2])
        dg = DGp.sum(dim=2).reshape(B, H, T).to(ctx._dtypes[4])
        db = DBp.sum(dim=2).reshape(B, H, T).to(ctx._dtypes[5])
        de = torch.zeros(ctx._e_shape, dtype=ctx._dtypes[3], device=dev)
        dgam = torch.zeros(ctx._gamma_shape, dtype=ctx._dtypes[6], device=dev)
        return (dq, dk, dv, de, dg, db, dgam, None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _GatedDeltaFunc.apply(q, k, v, e, g, beta, gamma, scale)
```