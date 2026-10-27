```python
import torch
import triton
import triton.language as tl

CHUNK_C = 64
BV_DEF = 32

@triton.jit
def _fwd_ckpt(
    Q, Kp, Vp, Gp, Bp, Out, Bounds,
    sQb, sQh, sQt, sQk,
    sKb, sKh, sKt, sKk,
    sVb, sVh, sVt, sVv,
    sGb, sGh, sGt, sGk,
    sBb, sBh, sBt,
    H, T, Kdim, Vdim, NV, N,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    iv = pid % NV
    b = bh // H
    h = bh % H
    offK = tl.arange(0, BLOCK_K)
    offV = tl.arange(0, BLOCK_V)
    maskK = offK < Kdim
    vIdx = iv * BLOCK_V + offV
    maskV = vIdx < Vdim
    mask2d = maskK[:, None] & maskV[None, :]

    qBase = Q + b * sQb + h * sQh
    kBase = Kp + b * sKb + h * sKh
    vBase = Vp + b * sVb + h * sVh
    gBase = Gp + b * sGb + h * sGh
    bBase = Bp + b * sBb + h * sBh

    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    for t in range(T):
        q = tl.load(qBase + t * sQt + offK * sQk, mask=maskK, other=0.0).to(tl.float32)
        kk = tl.load(kBase + t * sKt + offK * sKk, mask=maskK, other=0.0).to(tl.float32)
        gg = tl.load(gBase + t * sGt + offK * sGk, mask=maskK, other=0.0)
        vv = tl.load(vBase + t * sVt + vIdx * sVv, mask=maskV, other=0.0).to(tl.float32)
        bt = tl.load(bBase + t * sBt)
        dd = tl.exp(gg)
        pred = tl.sum(S * kk[:, None], axis=0)
        delta = (vv - pred) * bt
        S = (S + kk[:, None] * delta[None, :]) * dd[:, None]
        o = tl.sum(S * (q * scale)[:, None], axis=0)
        tl.store(Out + (bh * T + t) * Vdim + vIdx, o, mask=maskV)
        t1 = t + 1
        if (t1 % CHUNK) == 0:
            c = t1 // CHUNK
            bptr = (bh * (N + 1) + c) * Kdim * Vdim
            tl.store(Bounds + bptr + offK[:, None] * Vdim + vIdx[None, :], S, mask=mask2d)


@triton.jit
def _bwd_chunk(
    Q, Kp, Vp, Gp, Bp, GOut,
    Bounds, SC, SP, Cur,
    DqP, DkP, DgP, DbP, Dv,
    sQb, sQh, sQt, sQk,
    sKb, sKh, sKt, sKk,
    sVb, sVh, sVt, sVv,
    sGb, sGh, sGt, sGk,
    sBb, sBh, sBt,
    H, T, Kdim, Vdim, NV, N, t0, L, chunk,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    iv = pid % NV
    b = bh // H
    h = bh % H
    offK = tl.arange(0, BLOCK_K)
    offV = tl.arange(0, BLOCK_V)
    maskK = offK < Kdim
    vIdx = iv * BLOCK_V + offV
    maskV = vIdx < Vdim
    mask2d = maskK[:, None] & maskV[None, :]

    qBase = Q + b * sQb + h * sQh
    kBase = Kp + b * sKb + h * sKh
    vBase = Vp + b * sVb + h * sVh
    gBase = Gp + b * sGb + h * sGh
    bBase = Bp + b * sBb + h * sBh

    bstart = (bh * (N + 1) + chunk) * Kdim * Vdim
    S = tl.load(Bounds + bstart + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d, other=0.0)

    for j in range(L):
        t = t0 + j
        kk = tl.load(kBase + t * sKt + offK * sKk, mask=maskK, other=0.0).to(tl.float32)
        gg = tl.load(gBase + t * sGt + offK * sGk, mask=maskK, other=0.0)
        vv = tl.load(vBase + t * sVt + vIdx * sVv, mask=maskV, other=0.0).to(tl.float32)
        bt = tl.load(bBase + t * sBt)
        dd = tl.exp(gg)
        sptr = (bh * CHUNK + j) * Kdim * Vdim
        tl.store(SP + sptr + offK[:, None] * Vdim + vIdx[None, :], S, mask=mask2d)
        pred = tl.sum(S * kk[:, None], axis=0)
        delta = (vv - pred) * bt
        S = (S + kk[:, None] * delta[None, :]) * dd[:, None]
        tl.store(SC + sptr + offK[:, None] * Vdim + vIdx[None, :], S, mask=mask2d)

    cptr = bh * Kdim * Vdim
    cur = tl.load(Cur + cptr + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d, other=0.0)

    for jj in range(L):
        j = L - 1 - jj
        t = t0 + j
        q = tl.load(qBase + t * sQt + offK * sQk, mask=maskK, other=0.0).to(tl.float32)
        kk = tl.load(kBase + t * sKt + offK * sKk, mask=maskK, other=0.0).to(tl.float32)
        gg = tl.load(gBase + t * sGt + offK * sGk, mask=maskK, other=0.0)
        vv = tl.load(vBase + t * sVt + vIdx * sVv, mask=maskV, other=0.0).to(tl.float32)
        bt = tl.load(bBase + t * sBt)
        do = tl.load(GOut + (bh * T + t) * Vdim + vIdx, mask=maskV, other=0.0).to(tl.float32)
        sptr = (bh * CHUNK + j) * Kdim * Vdim
        Sc = tl.load(SC + sptr + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d, other=0.0)
        Sp = tl.load(SP + sptr + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d, other=0.0)
        dd = tl.exp(gg)
        qs = q * scale
        total = cur + qs[:, None] * do[None, :]
        dg_p = tl.sum(total * Sc, axis=1)
        dq_p = tl.sum(Sc * do[None, :], axis=1) * scale
        da = total * dd[:, None]
        c = tl.sum(da * kk[:, None], axis=0)
        pred2 = tl.sum(Sp * kk[:, None], axis=0)
        err = vv - pred2
        dv_v = bt * c
        tl.store(Dv + (bh * T + t) * Vdim + vIdx, dv_v, mask=maskV)
        db_p = tl.sum(c * err, axis=0)
        t1 = tl.sum(da * err[None, :], axis=1)
        t2 = tl.sum(Sp * c[None, :], axis=1)
        dk_p = (t1 - t2) * bt
        pbase = (bh * T + t) * NV + iv
        tl.store(DqP + pbase * Kdim + offK, dq_p, mask=maskK)
        tl.store(DkP + pbase * Kdim + offK, dk_p, mask=maskK)
        tl.store(DgP + pbase * Kdim + offK, dg_p, mask=maskK)
        tl.store(DbP + pbase, db_p)
        cur = da - bt * kk[:, None] * c[None, :]

    tl.store(Cur + cptr + offK[:, None] * Vdim + vIdx[None, :], cur, mask=mask2d)


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
        ctx.q_dtype = q.dtype
        ctx.k_dtype = k.dtype
        ctx.v_dtype = v.dtype
        ctx.g_dtype = g.dtype
        ctx.b_dtype = beta.dtype
        ctx.B, ctx.H, ctx.T, ctx.Kdim, ctx.Vdim = B, H, T, Kdim, Vdim
        ctx.scale_f = scale_f
        BK = triton.next_power_of_2(Kdim) if Kdim > 0 else 16
        if BK < 16:
            BK = 16
        if BK > 1024:
            BK = 1024
        if Vdim <= 16:
            BVV = 16
        elif Vdim <= 32:
            BVV = 32
        else:
            BVV = 64
        while BK * BVV > 16384 and BVV > 16:
            BVV //= 2
        NV = (Vdim + BVV - 1) // BVV if Vdim > 0 else 1
        if NV < 1:
            NV = 1
        BH = B * H
        N = (T + CHUNK_C - 1) // CHUNK_C if T > 0 else 0
        ctx.BK, ctx.BVV, ctx.NV, ctx.N, ctx.BH = BK, BVV, NV, N, BH
        if T == 0 or BH == 0 or Kdim == 0 or Vdim == 0:
            out = torch.empty(B, H, T, Vdim, device=dev, dtype=torch.float32)
            bounds = torch.zeros(max(BH, 1) * (N + 1) * max(Kdim, 1) * max(Vdim, 1), device=dev, dtype=torch.float32)
            ctx.save_for_backward(q, k, v, g, beta, bounds)
            ctx.strides = (q.stride(), k.stride(), v.stride(), g.stride(), beta.stride())
            return out
        outf = torch.empty(BH * T * Vdim, device=dev, dtype=torch.float32)
        bounds = torch.zeros(BH * (N + 1) * Kdim * Vdim, device=dev, dtype=torch.float32)
        grid = (BH * NV,)
        _fwd_ckpt[grid](
            q, k, v, g, beta, outf, bounds,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            H, T, Kdim, Vdim, NV, N, scale_f,
            BLOCK_K=BK, BLOCK_V=BVV, CHUNK=CHUNK_C, num_warps=4,
        )
        out = outf.view(B, H, T, Vdim)
        ctx.save_for_backward(q, k, v, g, beta, bounds)
        return out

    @staticmethod
    def backward(ctx, go):
        q, k, v, g, beta, bounds = ctx.saved_tensors
        B, H, T, Kdim, Vdim = ctx.B, ctx.H, ctx.T, ctx.Kdim, ctx.Vdim
        BK, BVV, NV, N, BH, scale_f = ctx.BK, ctx.BVV, ctx.NV, ctx.N, ctx.BH, ctx.scale_f
        dev = bounds.device
        if T == 0 or BH == 0 or Kdim == 0 or Vdim == 0:
            dq = torch.empty(B, H, T, Kdim, device=dev, dtype=ctx.q_dtype)
            dk = torch.empty(B, H, T, Kdim, device=dev, dtype=ctx.k_dtype)
            dv = torch.empty(B, H, T, Vdim, device=dev, dtype=ctx.v_dtype)
            dg = torch.empty(B, H, T, Kdim, device=dev, dtype=ctx.g_dtype)
            db = torch.empty(B, H, T, device=dev, dtype=ctx.b_dtype)
            de = torch.zeros(ctx.e_shape, dtype=ctx.e_dtype, device=dev)
            dga = torch.zeros(ctx.ga_shape, dtype=ctx.ga_dtype, device=dev)
            return dq, dk, dv, de, dg, db, dga, None
        go_c = go.contiguous().to(torch.float32).view(-1).contiguous()
        dqP = torch.empty(BH * T * NV * Kdim, device=dev, dtype=torch.float32)
        dkP = torch.empty(BH * T * NV * Kdim, device=dev, dtype=torch.float32)
        dgP = torch.empty(BH * T * NV * Kdim, device=dev, dtype=torch.float32)
        dbP = torch.empty(BH * T * NV, device=dev, dtype=torch.float32)
        dvf = torch.empty(BH * T * Vdim, device=dev, dtype=torch.float32)
        cur = torch.zeros(BH * Kdim * Vdim, device=dev, dtype=torch.float32)
        sc = torch.empty(BH * CHUNK_C * Kdim * Vdim, device=dev, dtype=torch.float32)
        sp = torch.empty(BH * CHUNK_C * Kdim * Vdim, device=dev, dtype=torch.float32)
        grid = (BH * NV,)
        for c in range(N - 1, -1, -1):
            t0 = c * CHUNK_C
            L = T - t0
            if L > CHUNK_C:
                L = CHUNK_C
            _bwd_chunk[grid](
                q, k, v, g, beta, go_c, bounds, sc, sp, cur,
                dqP, dkP, dgP, dbP, dvf,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                g.stride(0), g.stride(1), g.stride(2), g.stride(3),
                beta.stride(0), beta.stride(1), beta.stride(2),
                H, T, Kdim, Vdim, NV, N, t0, L, c, scale_f,
                BLOCK_K=BK, BLOCK_V=BVV, CHUNK=CHUNK_C, num_warps=4,
            )
        dq = dqP.view(B, H, T, NV, Kdim).sum(dim=3).to(ctx.q_dtype)
        dk = dkP.view(B, H, T, NV, Kdim).sum(dim=3).to(ctx.k_dtype)
        dg = dgP.view(B, H, T, NV, Kdim).sum(dim=3).to(ctx.g_dtype)
        db = dbP.view(B, H, T, NV).sum(dim=3).to(ctx.b_dtype)
        dv = dvf.view(B, H, T, Vdim).to(ctx.v_dtype)
        de = torch.zeros(ctx.e_shape, dtype=ctx.e_dtype, device=dev)
        dga = torch.zeros(ctx.ga_shape, dtype=ctx.ga_dtype, device=dev)
        return dq, dk, dv, de, dg, db, dga, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    scale_f = float(scale.item() if torch.is_tensor(scale) else scale)
    return _Func.apply(q, k, v, e, g, beta, gamma, scale_f)
```