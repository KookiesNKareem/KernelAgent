import torch
import triton
import triton.language as tl

CHUNK = 64
BV = 32

@triton.jit
def _fwd(
    Q, Kp, Vp, Dp, Bp, Out, Bounds,
    BH, T, Kdim, Vdim, NV, N,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    iv = pid % NV
    offK = tl.arange(0, BLOCK_K)
    offV = tl.arange(0, BLOCK_V)
    maskK = offK < Kdim
    vIdx = iv * BLOCK_V + offV
    maskV = vIdx < Vdim
    mask2d = maskK[:, None] & maskV[None, :]

    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    for t in tl.range(0, T, num_stages=2):
        bq = (bh * T + t) * Kdim
        bv = (bh * T + t) * Vdim
        bb = bh * T + t
        q = tl.load(Q + bq + offK, mask=maskK, other=0.0).to(tl.float32)
        kk = tl.load(Kp + bq + offK, mask=maskK, other=0.0).to(tl.float32)
        dd = tl.load(Dp + bq + offK, mask=maskK, other=1.0)
        vv = tl.load(Vp + bv + vIdx, mask=maskV, other=0.0).to(tl.float32)
        bt = tl.load(Bp + bb)
        pred = tl.sum(S * kk[:, None], axis=0)
        delta = (vv - pred) * bt
        S = (S + kk[:, None] * delta[None, :]) * dd[:, None]
        o = tl.sum(S * (q * scale)[:, None], axis=0)
        tl.store(Out + bv + vIdx, o, mask=maskV)
        t1 = t + 1
        is_b = (t1 % CHUNK) == 0
        c = t1 // CHUNK
        ok = is_b & (c <= N)
        m = mask2d & ok
        bptr = (bh * (N + 1) + c) * Kdim * Vdim
        tl.store(Bounds + bptr + offK[:, None] * Vdim + vIdx[None, :], S, mask=m)


@triton.jit
def _bwd(
    Q, Kp, Vp, Dp, Bp, GOut, Bounds, Scratch, CurDS,
    DqP, DkP, DgP, DbP, Dv,
    BH, T, Kdim, Vdim, NV, N, t0, L, chunk,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    iv = pid % NV
    offK = tl.arange(0, BLOCK_K)
    offV = tl.arange(0, BLOCK_V)
    maskK = offK < Kdim
    vIdx = iv * BLOCK_V + offV
    maskV = vIdx < Vdim
    mask2d = maskK[:, None] & maskV[None, :]

    bstart = (bh * (N + 1) + chunk) * Kdim * Vdim
    has = chunk != 0
    S_start = tl.load(Bounds + bstart + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d & has, other=0.0)

    S = S_start
    for j in tl.range(0, L, num_stages=2):
        t = t0 + j
        bq = (bh * T + t) * Kdim
        bv = (bh * T + t) * Vdim
        bb = bh * T + t
        kk = tl.load(Kp + bq + offK, mask=maskK, other=0.0).to(tl.float32)
        dd = tl.load(Dp + bq + offK, mask=maskK, other=1.0)
        vv = tl.load(Vp + bv + vIdx, mask=maskV, other=0.0).to(tl.float32)
        bt = tl.load(Bp + bb)
        pred = tl.sum(S * kk[:, None], axis=0)
        delta = (vv - pred) * bt
        S = (S + kk[:, None] * delta[None, :]) * dd[:, None]
        sptr = (bh * CHUNK + j) * Kdim * Vdim
        tl.store(Scratch + sptr + offK[:, None] * Vdim + vIdx[None, :], S, mask=mask2d)

    cptr = bh * Kdim * Vdim
    cur = tl.load(CurDS + cptr + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d, other=0.0)

    for jj in tl.range(0, L, num_stages=2):
        j = L - 1 - jj
        t = t0 + j
        bq = (bh * T + t) * Kdim
        bv = (bh * T + t) * Vdim
        bb = bh * T + t
        q = tl.load(Q + bq + offK, mask=maskK, other=0.0).to(tl.float32)
        kk = tl.load(Kp + bq + offK, mask=maskK, other=0.0).to(tl.float32)
        dd = tl.load(Dp + bq + offK, mask=maskK, other=1.0)
        vv = tl.load(Vp + bv + vIdx, mask=maskV, other=0.0).to(tl.float32)
        bt = tl.load(Bp + bb)
        do = tl.load(GOut + bv + vIdx, mask=maskV, other=0.0).to(tl.float32)
        sptr = (bh * CHUNK + j) * Kdim * Vdim
        Sc = tl.load(Scratch + sptr + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d, other=0.0)
        jm1 = j - 1
        jm1c = tl.maximum(jm1, 0)
        sptr2 = (bh * CHUNK + jm1c) * Kdim * Vdim
        not_first = j > 0
        Sp_ld = tl.load(Scratch + sptr2 + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d & not_first, other=0.0)
        is_first = j == 0
        Sp = tl.where(is_first, S_start, Sp_ld)
        qs = q * scale
        total = cur + qs[:, None] * do[None, :]
        dg_p = tl.sum(total * Sc, axis=1)
        dq_p = tl.sum(Sc * do[None, :], axis=1) * scale
        da = total * dd[:, None]
        c = tl.sum(da * kk[:, None], axis=0)
        pred2 = tl.sum(Sp * kk[:, None], axis=0)
        err = vv - pred2
        dv_v = bt * c
        tl.store(Dv + bv + vIdx, dv_v, mask=maskV)
        db_p = tl.sum(c * err, axis=0)
        t1 = tl.sum(da * err[None, :], axis=1)
        t2 = tl.sum(Sp * c[None, :], axis=1)
        dk_p = (t1 - t2) * bt
        # partial buffers: [BH*T*NV + ...] layout
        # DqP base: ((bh*T+t)*NV+iv)*Kdim
        pbase = (bh * T + t) * NV + iv
        tl.store(DqP + pbase * Kdim + offK, dq_p, mask=maskK)
        tl.store(DkP + pbase * Kdim + offK, dk_p, mask=maskK)
        tl.store(DgP + pbase * Kdim + offK, dg_p, mask=maskK)
        tl.store(DbP + pbase, db_p)
        cur = da - bt * kk[:, None] * c[None, :]

    tl.store(CurDS + cptr + offK[:, None] * Vdim + vIdx[None, :], cur, mask=mask2d)


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, Kdim = q.shape
        Vdim = v.shape[-1]
        scale_f = float(scale.item() if torch.is_tensor(scale) else scale)
        dev = q.device
        ctx.e_shape = e.shape
        ctx.e_dtype = e.dtype
        ctx.ga_shape = gamma.shape
        ctx.ga_dtype = gamma.dtype
        ctx.B, ctx.H, ctx.T, ctx.Kdim, ctx.Vdim = B, H, T, Kdim, Vdim
        ctx.scale_f = scale_f
        if T == 0:
            out = torch.empty(B, H, 0, Vdim, device=dev, dtype=torch.float32)
            ctx.BH = B * H
            ctx.NV = 1
            ctx.N = 0
            ctx.BK = 16
            ctx.save_for_backward(q, k, v, torch.empty(0, device=dev), beta, torch.empty(0, device=dev))
            return out
        BK = triton.next_power_of_2(Kdim)
        if BK < 16:
            BK = 16
        if BK > 256:
            BK = 256
        NV = (Vdim + BV - 1) // BV
        N = (T + CHUNK - 1) // CHUNK
        BH = B * H
        d = torch.exp(g.contiguous()).contiguous()
        qc = q.contiguous()
        kc = k.contiguous()
        vc = v.contiguous()
        bc = beta.contiguous()
        out = torch.empty(B * H * T * Vdim, device=dev, dtype=torch.float32)
        bounds = torch.zeros(BH * (N + 1) * Kdim * Vdim, device=dev, dtype=torch.float32)
        grid = (BH * NV,)
        _fwd[grid](qc, kc, vc, d, bc, out, bounds, BH, T, Kdim, Vdim, NV, N, scale_f, BLOCK_K=BK, BLOCK_V=BV, num_warps=4)
        out = out.view(B, H, T, Vdim)
        ctx.BH, ctx.NV, ctx.N, ctx.BK = BH, NV, N, BK
        ctx.save_for_backward(qc, kc, vc, d, bc, bounds)
        return out

    @staticmethod
    def backward(ctx, go):
        qc, kc, vc, d, bc, bounds = ctx.saved_tensors
        B, H, T, Kdim, Vdim = ctx.B, ctx.H, ctx.T, ctx.Kdim, ctx.Vdim
        BH, NV, N, BK, scale_f = ctx.BH, ctx.NV, ctx.N, ctx.BK, ctx.scale_f
        dev = qc.device
        if T == 0:
            dq = torch.empty(B, H, 0, Kdim, device=dev, dtype=qc.dtype)
            dk = torch.empty(B, H, 0, Kdim, device=dev, dtype=kc.dtype)
            dv = torch.empty(B, H, 0, Vdim, device=dev, dtype=vc.dtype)
            dg = torch.empty(B, H, 0, Kdim, device=dev, dtype=d.dtype if d.numel() else torch.float32)
            db = torch.empty(B, H, 0, device=dev, dtype=bc.dtype)
            de = torch.zeros(ctx.e_shape, dtype=ctx.e_dtype, device=dev)
            dga = torch.zeros(ctx.ga_shape, dtype=ctx.ga_dtype, device=dev)
            return dq, dk, dv, de, dg, db, dga, None
        go_c = go.contiguous().to(torch.float32).view(-1).contiguous()
        qc_f = qc.view(-1).contiguous()
        kc_f = kc.view(-1).contiguous()
        vc_f = vc.view(-1).contiguous()
        d_f = d.view(-1).contiguous()
        bc_f = bc.view(-1).contiguous()
        dqP = torch.empty(BH * T * NV * Kdim, device=dev, dtype=torch.float32)
        dkP = torch.empty(BH * T * NV * Kdim, device=dev, dtype=torch.float32)
        dgP = torch.empty(BH * T * NV * Kdim, device=dev, dtype=torch.float32)
        dbP = torch.empty(BH * T * NV, device=dev, dtype=torch.float32)
        dv_o = torch.empty(BH * T * Vdim, device=dev, dtype=torch.float32)
        cur = torch.zeros(BH * Kdim * Vdim, device=dev, dtype=torch.float32)
        scratch = torch.empty(BH * CHUNK * Kdim * Vdim, device=dev, dtype=torch.float32)
        grid = (BH * NV,)
        for c in range(N - 1, -1, -1):
            t0 = c * CHUNK
            L = T - t0
            if L > CHUNK:
                L = CHUNK
            _bwd[grid](qc_f, kc_f, vc_f, d_f, bc_f, go_c, bounds, scratch, cur,
                       dqP, dkP, dgP, dbP, dv_o,
                       BH, T, Kdim, Vdim, NV, N, t0, L, c, scale_f,
                       BLOCK_K=BK, BLOCK_V=BV, num_warps=4)
        dqP = dqP.view(B, H, T, NV, Kdim).sum(dim=3)
        dkP = dkP.view(B, H, T, NV, Kdim).sum(dim=3)
        dgP = dgP.view(B, H, T, NV, Kdim).sum(dim=3)
        dbP = dbP.view(B, H, T, NV).sum(dim=3)
        dv_o = dv_o.view(B, H, T, Vdim)
        dq = dqP.to(qc.dtype)
        dk = dkP.to(kc.dtype)
        dv = dv_o.to(vc.dtype)
        dg = dgP.to(d.dtype)
        db = dbP.to(bc.dtype)
        de = torch.zeros(ctx.e_shape, dtype=ctx.e_dtype, device=dev)
        dga = torch.zeros(ctx.ga_shape, dtype=ctx.ga_dtype, device=dev)
        return dq, dk, dv, de, dg, db, dga, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    scale_f = float(scale.item() if torch.is_tensor(scale) else scale)
    return _Func.apply(q, k, v, e, g, beta, gamma, scale_f)
