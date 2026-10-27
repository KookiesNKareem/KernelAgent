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
def _bwd_chunk_kernel(
    Qptr, Kptr, Vptr, DOptr, Gptr, Bptr,
    DQpartPtr, DKpartPtr, DVptr, DGpartPtr, DBpartPtr,
    SstartPtr, Fptr, ScratchPtr,
    scale,
    BH, T, Kdim, Vdim, chunk_start, Cmax, NV,
    BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < Kdim
    v_cols = pid_v * BLOCK_V + offs_v
    mask_v = v_cols < Vdim
    s_mask = mask_k[:, None] & mask_v[None, :]
    # load S_start shard
    s0_ptrs = SstartPtr + pid_bh * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
    S = tl.load(s0_ptrs, mask=s_mask, other=0.0).to(tl.float32)
    # forward recompute, store S_t to scratch
    for t_off in range(BLOCK_C):
        t = chunk_start + t_off
        k_vec = tl.load(Kptr + (pid_bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(Vptr + (pid_bh * T + t) * Vdim + v_cols, mask=mask_v, other=0.0).to(tl.float32)
        g_val = tl.load(Gptr + pid_bh * T + t).to(tl.float32)
        b_val = tl.load(Bptr + pid_bh * T + t).to(tl.float32)
        decay = tl.exp(g_val)
        S = S * decay
        pred = tl.sum(S * k_vec[:, None], axis=0)
        delta = (v_vec - pred) * b_val
        S = S + k_vec[:, None] * delta[None, :]
        sc_ptrs = ScratchPtr + pid_bh * Cmax * Kdim * Vdim + t_off * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
        tl.store(sc_ptrs, S, mask=s_mask)
    f_ptrs = Fptr + pid_bh * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
    Fcur = tl.load(f_ptrs, mask=s_mask, other=0.0).to(tl.float32)
    for ii in range(BLOCK_C):
        t_off = BLOCK_C - 1 - ii
        t = chunk_start + t_off
        k_vec = tl.load(Kptr + (pid_bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        q_vec = tl.load(Qptr + (pid_bh * T + t) * Kdim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(Vptr + (pid_bh * T + t) * Vdim + v_cols, mask=mask_v, other=0.0).to(tl.float32)
        do_vec = tl.load(DOptr + (pid_bh * T + t) * Vdim + v_cols, mask=mask_v, other=0.0).to(tl.float32)
        g_val = tl.load(Gptr + pid_bh * T + t).to(tl.float32)
        b_val = tl.load(Bptr + pid_bh * T + t).to(tl.float32)
        decay = tl.exp(g_val)
        st_ptrs = ScratchPtr + pid_bh * Cmax * Kdim * Vdim + t_off * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
        St = tl.load(st_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        if t_off == 0:
            Sprev = tl.load(s0_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        else:
            sp_ptrs = ScratchPtr + pid_bh * Cmax * Kdim * Vdim + (t_off - 1) * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
            Sprev = tl.load(sp_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        Smid = Sprev * decay
        pred_s = tl.sum(Smid * k_vec[:, None], axis=0)
        e_vec = v_vec - pred_s
        QO = (q_vec * scale)[:, None] * do_vec[None, :]
        Dt = QO + Fcur
        dtk = tl.sum(Dt * k_vec[:, None], axis=0)
        dv_vec = dtk * b_val
        tl.store(DVptr + (pid_bh * T + t) * Vdim + v_cols, dv_vec, mask=mask_v)
        pdq = tl.sum(St * do_vec[None, :] * scale, axis=1)
        dq_base = (pid_bh * T * NV + t * NV + pid_v) * Kdim
        tl.store(DQpartPtr + dq_base + offs_k, pdq, mask=mask_k)
        de1 = tl.sum(Dt * e_vec[None, :], axis=1)
        de2 = tl.sum(Smid * dtk[None, :], axis=1)
        pdk = (de1 - de2) * b_val
        tl.store(DKpartPtr + dq_base + offs_k, pdk, mask=mask_k)
        tmp_b = Dt * k_vec[:, None] * e_vec[None, :]
        s_b0 = tl.sum(tmp_b, axis=0)
        s_b = tl.sum(s_b0, axis=0)
        tl.store(DBpartPtr + (pid_bh * T + t) * NV + pid_v, s_b)
        tmp_g = Dt * Smid
        s_g0 = tl.sum(tmp_g, axis=0)
        s_g = tl.sum(s_g0, axis=0) - b_val * tl.sum(dtk * pred_s, axis=0)
        tl.store(DGpartPtr + (pid_bh * T + t) * NV + pid_v, s_g)
        Fcur = (Dt - b_val * k_vec[:, None] * dtk[None, :]) * decay
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
            ctx._shapes = (B, H, 0, K, Vdim)
            ctx._dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
            ctx._dev = dev
            return out
        ctx._empty = False
        q_ = q.contiguous()
        k_ = k.contiguous()
        v_ = v.contiguous()
        g_ = g.contiguous()
        b_ = beta.contiguous()
        BH = B * H
        qf = q_.reshape(BH, T, K)
        kf = k_.reshape(BH, T, K)
        vf = v_.reshape(BH, T, Vdim)
        gf = g_.reshape(BH, T)
        bf = b_.reshape(BH, T)
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
        # chk[0] already zero; state starts zero
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
        if ctx._empty:
            B, H, T0, K0, V0 = ctx._shapes
            dq_, dk_, dv_, de_, dg_, db_, dgam_ = ctx._dtypes
            dev = ctx._dev
            def _z(sh, dt):
                return torch.zeros(sh, dtype=dt, device=dev)
            # shapes unknown fully; return Nones (won't be called for empty in practice)
            return (None, None, None, None, None, None, None, None)
        qf, kf, vf, gf, bf = ctx.saved_tensors
        chk = ctx.chk
        B, H, T, K, Vdim, BH, BK, BV, NV, C, N = ctx._shapes
        dev = ctx._dev
        scale_f = ctx.scale_f
        go = grad_out.contiguous().to(torch.float32).reshape(BH, T, Vdim)
        DQp = torch.empty(BH, T, NV, K, dtype=torch.float32, device=dev)
        DKp = torch.empty(BH, T, NV, K, dtype=torch.float32, device=dev)
        DGp = torch.empty(BH, T, NV, dtype=torch.float32, device=dev)
        DBp = torch.empty(BH, T, NV, dtype=torch.float32, device=dev)
        DV = torch.empty(BH, T, Vdim, dtype=torch.float32, device=dev)
        Fbuf = torch.zeros(BH, K, Vdim, dtype=torch.float32, device=dev)
        Cmax = C
        scratch = torch.empty(BH, Cmax, K, Vdim, dtype=torch.float32, device=dev)
        grid = (BH, NV)
        for ci in range(N - 1, -1, -1):
            cs = ci * C
            cur = C if cs + C <= T else T - cs
            Sstart = chk[ci]
            _bwd_chunk_kernel[grid](qf, kf, vf, go, gf, bf,
                                    DQp, DKp, DV, DGp, DBp,
                                    Sstart, Fbuf, scratch,
                                    scale_f, BH, T, K, Vdim, cs, Cmax, NV,
                                    BLOCK_C=cur, BLOCK_K=BK, BLOCK_V=BV, num_warps=4)
        dq_full = DQp.sum(dim=2).reshape(B, H, T, K)
        dk_full = DKp.sum(dim=2).reshape(B, H, T, K)
        dv_full = DV.reshape(B, H, T, Vdim)
        dg_full = DGp.sum(dim=2).reshape(B, H, T)
        db_full = DBp.sum(dim=2).reshape(B, H, T)
        qd, kd, vd, ed, gd, bd, gamd = ctx._dtypes
        dq = dq_full.to(qd)
        dk = dk_full.to(kd)
        dv = dv_full.to(vd)
        dg = dg_full.to(gd)
        db = db_full.to(bd)
        de = torch.zeros(ctx._e_shape, dtype=ed, device=dev)
        dgam = torch.zeros(ctx._gamma_shape, dtype=gamd, device=dev)
        return (dq, dk, dv, de, dg, db, dgam, None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _GatedDeltaFunc.apply(q, k, v, e, g, beta, gamma, scale)
```

My forward keeps the proven V-sharded sequential Triton loop and saves per-64-step state checkpoints for reuse. My backward processes chunks in reverse with a fused Triton kernel that first recomputes the chunk forward into scratch memory, then walks backward maintaining the adjoint state with locally computable transforms, emitting direct stores for dv and per-shard partials for dq, dk, dg and dbeta that Python sums over shards. I am unsure about Triton scalar-store replication versus atomic duplication, large 128 by 32 tensor register pressure with four warps, and numerical sensitivity of long gated recurrences in fp32, but the formulation avoids any unstable division by using only forward decays.
