```python
import torch
import triton
import triton.language as tl

CHUNK_C = 32

@triton.jit
def _fwd_chunk(Q, K, V, G, B, Out, Bounds,
               BH, T, Kdim, Vdim, NV, N, t0, L, chunk, scale,
               BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, CHUNK: tl.constexpr):
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
    S = tl.load(Bounds + bstart + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d, other=0.0)
    for j in range(CHUNK):
        valid = j < L
        t = t0 + j
        mk = maskK & valid
        mv = maskV & valid
        bq = (bh * T + t) * Kdim
        bv = (bh * T + t) * Vdim
        bb = bh * T + t
        q = tl.load(Q + bq + offK, mask=mk, other=0.0).to(tl.float32)
        kk = tl.load(K + bq + offK, mask=mk, other=0.0).to(tl.float32)
        gg = tl.load(G + bq + offK, mask=mk, other=0.0)
        vv = tl.load(V + bv + vIdx, mask=mv, other=0.0).to(tl.float32)
        bt = tl.load(B + bb, mask=valid, other=0.0)
        dd = tl.exp(gg)
        pred = tl.sum(S * kk[:, None], axis=0)
        delta = (vv - pred) * bt
        S = (S + kk[:, None] * delta[None, :]) * dd[:, None]
        o = tl.sum(S * (q * scale)[:, None], axis=0)
        tl.store(Out + bv + vIdx, o, mask=mv)
    bend = (bh * (N + 1) + chunk + 1) * Kdim * Vdim
    tl.store(Bounds + bend + offK[:, None] * Vdim + vIdx[None, :], S, mask=mask2d)


@triton.jit
def _bwd_chunk(Q, K, V, G, B, GOut, Bounds, Scratch, Cur,
               DqP, DkP, DgP, DbP, Dv,
               BH, T, Kdim, Vdim, NV, N, t0, L, chunk, scale,
               BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, CHUNK: tl.constexpr):
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
    S_start = tl.load(Bounds + bstart + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d, other=0.0)
    S = S_start
    for j in range(CHUNK):
        valid = j < L
        t = t0 + j
        mk = maskK & valid
        mv = maskV & valid
        bq = (bh * T + t) * Kdim
        bv = (bh * T + t) * Vdim
        bb = bh * T + t
        kk = tl.load(K + bq + offK, mask=mk, other=0.0).to(tl.float32)
        gg = tl.load(G + bq + offK, mask=mk, other=0.0)
        vv = tl.load(V + bv + vIdx, mask=mv, other=0.0).to(tl.float32)
        bt = tl.load(B + bb, mask=valid, other=0.0)
        dd = tl.exp(gg)
        pred = tl.sum(S * kk[:, None], axis=0)
        delta = (vv - pred) * bt
        S = (S + kk[:, None] * delta[None, :]) * dd[:, None]
        sptr = (bh * CHUNK + j) * Kdim * Vdim
        tl.store(Scratch + sptr + offK[:, None] * Vdim + vIdx[None, :], S, mask=mask2d & valid)
    cptr = bh * Kdim * Vdim
    cur = tl.load(Cur + cptr + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d, other=0.0)
    for jj in range(CHUNK):
        j = CHUNK - 1 - jj
        valid = j < L
        t = t0 + j
        mk = maskK & valid
        mv = maskV & valid
        bq = (bh * T + t) * Kdim
        bv = (bh * T + t) * Vdim
        bb = bh * T + t
        q = tl.load(Q + bq + offK, mask=mk, other=0.0).to(tl.float32)
        kk = tl.load(K + bq + offK, mask=mk, other=0.0).to(tl.float32)
        gg = tl.load(G + bq + offK, mask=mk, other=0.0)
        vv = tl.load(V + bv + vIdx, mask=mv, other=0.0).to(tl.float32)
        bt = tl.load(B + bb, mask=valid, other=0.0)
        do = tl.load(GOut + bv + vIdx, mask=mv, other=0.0)
        sptr = (bh * CHUNK + j) * Kdim * Vdim
        Sc = tl.load(Scratch + sptr + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d & valid, other=0.0)
        if j == 0:
            Sp = S_start
        else:
            sptr2 = (bh * CHUNK + j - 1) * Kdim * Vdim
            Sp = tl.load(Scratch + sptr2 + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d & valid, other=0.0)
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
        db_p = tl.sum(c * err, axis=0)
        t1 = tl.sum(da * err[None, :], axis=1)
        t2 = tl.sum(Sp * c[None, :], axis=1)
        dk_p = (t1 - t2) * bt
        pbase = (bh * T + t) * NV + iv
        tl.store(DqP + pbase * Kdim + offK, dq_p, mask=mk)
        tl.store(DkP + pbase * Kdim + offK, dk_p, mask=mk)
        tl.store(DgP + pbase * Kdim + offK, dg_p, mask=mk)
        tl.store(DbP + pbase, db_p, mask=valid)
        tl.store(Dv + bv + vIdx, dv_v, mask=mv)
        cur = da - bt * kk[:, None] * c[None, :]
    tl.store(Cur + cptr + offK[:, None] * Vdim + vIdx[None, :], cur, mask=mask2d)


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale_f):
        B, H, T, Kdim = q.shape
        Vdim = v.shape[-1]
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
        ctx.scale_f = float(scale_f)
        ctx.BH = B * H
        if T == 0 or Kdim == 0 or Vdim == 0:
            if Kdim == 0 and T > 0 and Vdim > 0:
                out = torch.zeros(B, H, T, Vdim, device=dev, dtype=torch.float32)
            else:
                out = torch.empty(B, H, T, Vdim, device=dev, dtype=torch.float32)
                if out.numel() > 0:
                    out.zero_()
            ctx.BK, ctx.BVV, ctx.NV, ctx.N = 16, 32, 1, 0
            empty = torch.empty(0, device=dev, dtype=torch.float32)
            ctx.save_for_backward(empty, empty, empty, empty, empty, empty)
            return out
        BK = triton.next_power_of_2(Kdim)
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
        NV = (Vdim + BVV - 1) // BVV
        BH = B * H
        N = (T + CHUNK_C - 1) // CHUNK_C
        qc = q.contiguous()
        kc = k.contiguous()
        vc = v.contiguous()
        gc = g.contiguous()
        bc = beta.contiguous()
        outf = torch.empty(BH * T * Vdim, device=dev, dtype=torch.float32)
        bounds = torch.zeros(BH * (N + 1) * Kdim * Vdim, device=dev, dtype=torch.float32)
        grid = (BH * NV,)
        sf = float(scale_f)
        for chunk in range(N):
            t0 = chunk * CHUNK_C
            Lc = T - t0
            if Lc > CHUNK_C:
                Lc = CHUNK_C
            _fwd_chunk[grid](qc, kc, vc, gc, bc, outf, bounds,
                             BH, T, Kdim, Vdim, NV, N, t0, Lc, chunk, sf,
                             BLOCK_K=BK, BLOCK_V=BVV, CHUNK=CHUNK_C, num_warps=4)
        out = outf.view(B, H, T, Vdim)
        ctx.BK, ctx.BVV, ctx.NV, ctx.N = BK, BVV, NV, N
        ctx.save_for_backward(qc, kc, vc, gc, bc, bounds)
        return out

    @staticmethod
    def backward(ctx, go):
        B, H, T, Kdim, Vdim = ctx.B, ctx.H, ctx.T, ctx.Kdim, ctx.Vdim
        dev = go.device
        if T == 0 or Kdim == 0 or Vdim == 0:
            dq = torch.zeros(B, H, T, Kdim, device=dev, dtype=ctx.q_dtype)
            dk = torch.zeros(B, H, T, Kdim, device=dev, dtype=ctx.k_dtype)
            dv = torch.zeros(B, H, T, Vdim, device=dev, dtype=ctx.v_dtype)
            dg = torch.zeros(B, H, T, Kdim, device=dev, dtype=ctx.g_dtype)
            db = torch.zeros(B, H, T, device=dev, dtype=ctx.b_dtype)
            de = torch.zeros(ctx.e_shape, device=dev, dtype=ctx.e_dtype)
            dga = torch.zeros(ctx.ga_shape, device=dev, dtype=ctx.ga_dtype)
            return dq, dk, dv, de, dg, db, dga, None
        qc, kc, vc, gc, bc, bounds = ctx.saved_tensors
        BK, BVV, NV, N, BH = ctx.BK, ctx.BVV, ctx.NV, ctx.N, ctx.BH
        sf = ctx.scale_f
        go_c = go.contiguous().to(torch.float32).view(-1).contiguous()
        dqP = torch.empty(BH * T * NV * Kdim, device=dev, dtype=torch.float32)
        dkP = torch.empty(BH * T * NV * Kdim, device=dev, dtype=torch.float32)
        dgP = torch.empty(BH * T * NV * Kdim, device=dev, dtype=torch.float32)
        dbP = torch.empty(BH * T * NV, device=dev, dtype=torch.float32)
        dvf = torch.empty(BH * T * Vdim, device=dev, dtype=torch.float32)
        cur = torch.zeros(BH * Kdim * Vdim, device=dev, dtype=torch.float32)
        scratch = torch.empty(BH * CHUNK_C * Kdim * Vdim, device=dev, dtype=torch.float32)
        grid = (BH * NV,)
        for c in range(N - 1, -1, -1):
            t0 = c * CHUNK_C
            Lc = T - t0
            if Lc > CHUNK_C:
                Lc = CHUNK_C
            _bwd_chunk[grid](qc, kc, vc, gc, bc, go_c, bounds, scratch, cur,
                             dqP, dkP, dgP, dbP, dvf,
                             BH, T, Kdim, Vdim, NV, N, t0, Lc, c, sf,
                             BLOCK_K=BK, BLOCK_V=BVV, CHUNK=CHUNK_C, num_warps=4)
        dq = dqP.view(B, H, T, NV, Kdim).sum(dim=3).to(ctx.q_dtype)
        dk = dkP.view(B, H, T, NV, Kdim).sum(dim=3).to(ctx.k_dtype)
        dg = dgP.view(B, H, T, NV, Kdim).sum(dim=3).to(ctx.g_dtype)
        db = dbP.view(B, H, T, NV).sum(dim=3).to(ctx.b_dtype)
        dv = dvf.view(B, H, T, Vdim).to(ctx.v_dtype)
        de = torch.zeros(ctx.e_shape, device=dev, dtype=ctx.e_dtype)
        dga = torch.zeros(ctx.ga_shape, device=dev, dtype=ctx.ga_dtype)
        return dq, dk, dv, de, dg, db, dga, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    scale_f = float(scale.item() if torch.is_tensor(scale) else scale)
    return _Func.apply(q, k, v, e, g, beta, gamma, scale_f)
```