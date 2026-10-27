```python
import torch
import triton
import triton.language as tl

CHUNK = 64
BV = 32

@triton.jit
def _fwd(Q, Kptr, Vptr, Dptr, Bptr, Out, Bounds,
         H, T, Kdim, Vdim, NV, N, scale,
         BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, CHUNK_C: tl.constexpr):
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
    seq_q = (b * H + h) * T * Kdim
    seq_v = (b * H + h) * T * Vdim
    seq_b = (b * H + h) * T
    bounds_seq = (b * H + h) * (N + 1) * Kdim * Vdim
    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    for t in range(T):
        q = tl.load(Q + seq_q + t * Kdim + offK, mask=maskK, other=0.0).to(tl.float32)
        k = tl.load(Kptr + seq_q + t * Kdim + offK, mask=maskK, other=0.0).to(tl.float32)
        d = tl.load(Dptr + seq_q + t * Kdim + offK, mask=maskK, other=1.0)
        vv = tl.load(Vptr + seq_v + t * Vdim + vIdx, mask=maskV, other=0.0).to(tl.float32)
        bt = tl.load(Bptr + seq_b + t)
        pred = tl.sum(S * k[:, None], axis=0)
        delta = (vv - pred) * bt
        S = (S + k[:, None] * delta[None, :]) * d[:, None]
        o = tl.sum(S * (q * scale)[:, None], axis=0)
        tl.store(Out + seq_v + t * Vdim + vIdx, o, mask=maskV)
        c_next = (t + 1) // CHUNK_C
        is_b = (t + 1) - c_next * CHUNK_C == 0
        valid = is_b & (c_next < N)
        bptr = Bounds + bounds_seq + c_next * Kdim * Vdim
        tl.store(bptr + offK[:, None] * Vdim + vIdx[None, :], S, mask=mask2d & valid)


@triton.jit
def _bwd(Q, Kptr, Vptr, Dptr, Bptr, GOut, Bounds, Scratch, CurDS, Dq, Dk, Dg, Dbeta, Dv,
         H, T, Kdim, Vdim, NV, N, t0, L, chunk, scale,
         BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, CHUNK_C: tl.constexpr):
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
    seq_q = (b * H + h) * T * Kdim
    seq_v = (b * H + h) * T * Vdim
    seq_b = (b * H + h) * T
    bounds_seq = (b * H + h) * (N + 1) * Kdim * Vdim
    scratch_seq = (b * H + h) * CHUNK_C * Kdim * Vdim
    cur_seq = (b * H + h) * Kdim * Vdim
    dq_seq = (b * H + h) * T * Kdim
    dv_seq = (b * H + h) * T * Vdim
    db_seq = (b * H + h) * T
    has_start = chunk != 0
    bstart = Bounds + bounds_seq + chunk * Kdim * Vdim
    S_start = tl.load(bstart + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d & has_start, other=0.0)
    S = S_start
    for j in range(L):
        t = t0 + j
        k = tl.load(Kptr + seq_q + t * Kdim + offK, mask=maskK, other=0.0).to(tl.float32)
        d = tl.load(Dptr + seq_q + t * Kdim + offK, mask=maskK, other=1.0)
        vv = tl.load(Vptr + seq_v + t * Vdim + vIdx, mask=maskV, other=0.0).to(tl.float32)
        bt = tl.load(Bptr + seq_b + t)
        pred = tl.sum(S * k[:, None], axis=0)
        delta = (vv - pred) * bt
        S = (S + k[:, None] * delta[None, :]) * d[:, None]
        sptr = Scratch + scratch_seq + j * Kdim * Vdim
        tl.store(sptr + offK[:, None] * Vdim + vIdx[None, :], S, mask=mask2d)
    cur = tl.load(CurDS + cur_seq + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d, other=0.0)
    for jj in range(L):
        j = L - 1 - jj
        t = t0 + j
        q = tl.load(Q + seq_q + t * Kdim + offK, mask=maskK, other=0.0).to(tl.float32)
        k = tl.load(Kptr + seq_q + t * Kdim + offK, mask=maskK, other=0.0).to(tl.float32)
        d = tl.load(Dptr + seq_q + t * Kdim + offK, mask=maskK, other=1.0)
        vv = tl.load(Vptr + seq_v + t * Vdim + vIdx, mask=maskV, other=0.0).to(tl.float32)
        bt = tl.load(Bptr + seq_b + t)
        do = tl.load(GOut + seq_v + t * Vdim + vIdx, mask=maskV, other=0.0).to(tl.float32)
        Sc = tl.load(Scratch + scratch_seq + j * Kdim * Vdim + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d, other=0.0)
        j_prev = j - 1
        not_first = j != 0
        is_first = j == 0
        Sp_loaded = tl.load(Scratch + scratch_seq + j_prev * Kdim * Vdim + offK[:, None] * Vdim + vIdx[None, :], mask=mask2d & not_first, other=0.0)
        Sp = tl.where(is_first, S_start, Sp_loaded)
        qs = q * scale
        total = cur + qs[:, None] * do[None, :]
        dg_part = tl.sum(total * Sc, axis=1)
        tl.atomic_add(Dg + dq_seq + t * Kdim + offK, dg_part, mask=maskK)
        da = total * d[:, None]
        c = tl.sum(da * k[:, None], axis=0)
        pred = tl.sum(Sp * k[:, None], axis=0)
        err = vv - pred
        tl.store(Dv + dv_seq + t * Vdim + vIdx, bt * c, mask=maskV)
        db_part = tl.sum(c * err, axis=0)
        tl.atomic_add(Dbeta + db_seq + t, db_part)
        dq_part = tl.sum(Sc * do[None, :], axis=1) * scale
        tl.atomic_add(Dq + dq_seq + t * Kdim + offK, dq_part, mask=maskK)
        term1 = tl.sum(da * err[None, :], axis=1)
        term2 = tl.sum(Sp * c[None, :], axis=1)
        dk_part = (term1 - term2) * bt
        tl.atomic_add(Dk + dq_seq + t * Kdim + offK, dk_part, mask=maskK)
        cur = da - bt * k[:, None] * c[None, :]
    tl.store(CurDS + cur_seq + offK[:, None] * Vdim + vIdx[None, :], cur, mask=mask2d)


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, Kdim = q.shape
        Vdim = v.shape[-1]
        scale_f = float(scale.item() if torch.is_tensor(scale) else scale)
        dev = q.device
        if T == 0:
            out = torch.empty(B, H, 0, Vdim, device=dev, dtype=torch.float32)
            d = torch.empty(B, H, 0, Kdim, device=dev, dtype=torch.float32)
            bounds = torch.zeros(B, H, 1, Kdim, Vdim, device=dev, dtype=torch.float32)
            ctx.B, ctx.H, ctx.T, ctx.Kdim, ctx.Vdim = B, H, T, Kdim, Vdim
            ctx.NV, ctx.N, ctx.BK, ctx.scale_f = 1, 0, 16, scale_f
            ctx.e_shape, ctx.e_dtype = e.shape, e.dtype
            ctx.ga_shape, ctx.ga_dtype = gamma.shape, gamma.dtype
            ctx.dev = dev
            ctx.save_for_backward(q, k, v, d, beta, bounds)
            return out
        d = torch.exp(g)
        out = torch.empty(B, H, T, Vdim, device=dev, dtype=torch.float32)
        N = (T + CHUNK - 1) // CHUNK
        bounds = torch.zeros(B, H, N + 1, Kdim, Vdim, device=dev, dtype=torch.float32)
        BK = triton.next_power_of_2(Kdim)
        if BK < 16:
            BK = 16
        NV = (Vdim + BV - 1) // BV
        grid = (B * H * NV,)
        _fwd[grid](q, k, v, d, beta, out, bounds,
                   H, T, Kdim, Vdim, NV, N, scale_f,
                   BLOCK_K=BK, BLOCK_V=BV, CHUNK_C=CHUNK, num_warps=4)
        ctx.B, ctx.H, ctx.T, ctx.Kdim, ctx.Vdim = B, H, T, Kdim, Vdim
        ctx.NV, ctx.N, ctx.BK, ctx.scale_f = NV, N, BK, scale_f
        ctx.e_shape, ctx.e_dtype = e.shape, e.dtype
        ctx.ga_shape, ctx.ga_dtype = gamma.shape, gamma.dtype
        ctx.dev = dev
        ctx.save_for_backward(q, k, v, d, beta, bounds)
        return out

    @staticmethod
    def backward(ctx, go):
        q, k, v, d, beta, bounds = ctx.saved_tensors
        B, H, T, Kdim, Vdim = ctx.B, ctx.H, ctx.T, ctx.Kdim, ctx.Vdim
        NV, N, BK, scale_f = ctx.NV, ctx.N, ctx.BK, ctx.scale_f
        dev = ctx.dev
        if T == 0:
            dq = torch.empty(ctx.B, ctx.H, 0, ctx.Kdim, device=dev, dtype=q.dtype)
            dk = torch.empty(ctx.B, ctx.H, 0, ctx.Kdim, device=dev, dtype=k.dtype)
            dv = torch.empty(ctx.B, ctx.H, 0, ctx.Vdim, device=dev, dtype=v.dtype)
            dg = torch.empty(ctx.B, ctx.H, 0, ctx.Kdim, device=dev, dtype=d.dtype)
            db = torch.empty(ctx.B, ctx.H, 0, device=dev, dtype=beta.dtype)
            de = torch.zeros(ctx.e_shape, dtype=ctx.e_dtype, device=dev)
            dga = torch.zeros(ctx.ga_shape, dtype=ctx.ga_dtype, device=dev)
            return dq, dk, dv, de, dg, db, dga, None
        go_c = go.to(torch.float32)
        if not go_c.is_contiguous():
            go_c = go_c.contiguous()
        dq = torch.zeros(B, H, T, Kdim, device=dev, dtype=torch.float32)
        dk = torch.zeros(B, H, T, Kdim, device=dev, dtype=torch.float32)
        dg = torch.zeros(B, H, T, Kdim, device=dev, dtype=torch.float32)
        dbeta = torch.zeros(B, H, T, device=dev, dtype=torch.float32)
        dv = torch.empty(B, H, T, Vdim, device=dev, dtype=torch.float32)
        cur = torch.zeros(B, H, Kdim, Vdim, device=dev, dtype=torch.float32)
        scratch = torch.empty(B, H, CHUNK, Kdim, Vdim, device=dev, dtype=torch.float32)
        grid = (B * H * NV,)
        for c in range(N - 1, -1, -1):
            t0 = c * CHUNK
            L = T - t0
            if L > CHUNK:
                L = CHUNK
            _bwd[grid](q, k, v, d, beta, go_c, bounds, scratch, cur, dq, dk, dg, dbeta, dv,
                       H, T, Kdim, Vdim, NV, N, t0, L, c, scale_f,
                       BLOCK_K=BK, BLOCK_V=BV, CHUNK_C=CHUNK, num_warps=4)
        dq_o = dq.to(q.dtype)
        dk_o = dk.to(k.dtype)
        dv_o = dv.to(v.dtype)
        dg_o = dg.to(d.dtype)
        db_o = dbeta.to(beta.dtype)
        de = torch.zeros(ctx.e_shape, dtype=ctx.e_dtype, device=dev)
        dga = torch.zeros(ctx.ga_shape, dtype=ctx.ga_dtype, device=dev)
        return dq_o, dk_o, dv_o, de, dg_o, db_o, dga, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    qc = q.contiguous() if isinstance(q, torch.Tensor) else q
    kc = k.contiguous() if isinstance(k, torch.Tensor) else k
    vc = v.contiguous() if isinstance(v, torch.Tensor) else v
    ec = e.contiguous() if isinstance(e, torch.Tensor) else e
    gc = g.contiguous() if isinstance(g, torch.Tensor) else g
    bc = beta.contiguous() if isinstance(beta, torch.Tensor) else beta
    gac = gamma.contiguous() if isinstance(gamma, torch.Tensor) else gamma
    scale_f = float(scale.item() if torch.is_tensor(scale) else scale)
    return _Func.apply(qc, kc, vc, ec, gc, bc, gac, scale_f)
```
My approach fuses the forward recurrence with precomputed exp decay and periodic state checkpointing, then reuses checkpoints for a chunked reverse pass that recomputes forward states exactly for numerical stability. The backward streams KxV states sharded over V and derives closed-form gradients for q,k,v,g,beta with atomic reduction over shards for K-reductions. I am unsure about Triton support for scalar-bool masked loads/stores, tl.where broadcasting, and atomic-add throughput, plus optimal BLOCK/CHUNK tuning on L4.