```python
import torch
import triton
import triton.language as tl


def _npo2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


@triton.jit
def _fwd_generic(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, b_ptr, c_ptr, o_ptr, bnd_ptr,
    T, Nc, C, scale,
    K, V,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    mask_k = off_k < K
    mask_v = off_v < V
    mask2d = mask_k[:, None] & mask_v[None, :]
    off2d = off_k[:, None] * V + off_v[None, :]
    KV = K * V

    q_base = q_ptr + pid * T * K
    k_base = k_ptr + pid * T * K
    e_base = e_ptr + pid * T * K
    g_base = g_ptr + pid * T * K
    v_base = v_ptr + pid * T * V
    b_base = b_ptr + pid * T
    c_base = c_ptr + pid * T
    o_base = o_ptr + pid * T * V

    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    for t in range(T):
        q = tl.load(q_base + t * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        k = tl.load(k_base + t * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        e = tl.load(e_base + t * K + off_v, mask=mask_v, other=0.0).to(tl.float32)
        g = tl.load(g_base + t * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        v = tl.load(v_base + t * V + off_v, mask=mask_v, other=0.0).to(tl.float32)
        b = tl.load(b_base + t)
        cc = tl.load(c_base + t)
        d = tl.exp(g)
        S = S * d[:, None]
        u = tl.sum(S * e[None, :], axis=1)
        S = S - cc * u[:, None] * e[None, :]
        pred = tl.sum(S * k[:, None], axis=0)
        diff = v - pred
        S = S + b * k[:, None] * diff[None, :]
        o = tl.sum(S * q[:, None], axis=0) * scale
        tl.store(o_base + t * V + off_v, o, mask=mask_v)
        do_store = (((t + 1) % C) == 0) | (t == T - 1)
        if do_store:
            chunk = t // C
            tl.store(bnd_ptr + pid * Nc * KV + chunk * KV + off2d, S, mask=mask2d)


@triton.jit
def _bwd_generic(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, b_ptr, c_ptr, do_ptr,
    bnd_ptr, scratch_ptr, carry_ptr,
    dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, db_ptr, dc_ptr,
    chunk_start, chunk_len,
    T, Nc, C, scale,
    K, V,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    mask_k = off_k < K
    mask_v = off_v < V
    mask2d = mask_k[:, None] & mask_v[None, :]
    off2d = off_k[:, None] * V + off_v[None, :]
    KV = K * V

    q_base = q_ptr + pid * T * K
    k_base = k_ptr + pid * T * K
    e_base = e_ptr + pid * T * K
    g_base = g_ptr + pid * T * K
    v_base = v_ptr + pid * T * V
    b_base = b_ptr + pid * T
    c_base = c_ptr + pid * T
    do_base = do_ptr + pid * T * V
    dq_base = dq_ptr + pid * T * K
    dk_base = dk_ptr + pid * T * K
    dv_base = dv_ptr + pid * T * V
    de_base = de_ptr + pid * T * K
    dg_base = dg_ptr + pid * T * K
    db_base = db_ptr + pid * T
    dc_base = dc_ptr + pid * T

    # S_init -> S_cur for forward recompute
    S_cur = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    if chunk_start != 0:
        prev = (chunk_start // C) - 1
        S_cur = tl.load(bnd_ptr + pid * Nc * KV + prev * KV + off2d, mask=mask2d, other=0.0).to(tl.float32)

    for t in range(chunk_len):
        gt = chunk_start + t
        q = tl.load(q_base + gt * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        k = tl.load(k_base + gt * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        e = tl.load(e_base + gt * K + off_v, mask=mask_v, other=0.0).to(tl.float32)
        g = tl.load(g_base + gt * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        v = tl.load(v_base + gt * V + off_v, mask=mask_v, other=0.0).to(tl.float32)
        b = tl.load(b_base + gt)
        cc = tl.load(c_base + gt)
        d = tl.exp(g)
        S_cur = S_cur * d[:, None]
        u = tl.sum(S_cur * e[None, :], axis=1)
        S_cur = S_cur - cc * u[:, None] * e[None, :]
        pred = tl.sum(S_cur * k[:, None], axis=0)
        diff = v - pred
        S_cur = S_cur + b * k[:, None] * diff[None, :]
        tl.store(scratch_ptr + pid * C * KV + t * KV + off2d, S_cur, mask=mask2d)

    carry_base = carry_ptr + pid * KV
    Carry = tl.load(carry_base + off2d, mask=mask2d, other=0.0).to(tl.float32)

    for _i in range(chunk_len):
        t = chunk_len - 1 - _i
        gt = chunk_start + t
        q = tl.load(q_base + gt * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        k = tl.load(k_base + gt * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        e = tl.load(e_base + gt * K + off_v, mask=mask_v, other=0.0).to(tl.float32)
        g = tl.load(g_base + gt * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        v = tl.load(v_base + gt * V + off_v, mask=mask_v, other=0.0).to(tl.float32)
        b = tl.load(b_base + gt)
        cc = tl.load(c_base + gt)
        do = tl.load(do_base + gt * V + off_v, mask=mask_v, other=0.0).to(tl.float32)
        d = tl.exp(g)

        S = tl.load(scratch_ptr + pid * C * KV + t * KV + off2d, mask=mask2d, other=0.0).to(tl.float32)
        # S_prev
        S_prev = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
        if t > 0:
            S_prev = tl.load(scratch_ptr + pid * C * KV + (t - 1) * KV + off2d, mask=mask2d, other=0.0).to(tl.float32)
        else:
            if chunk_start != 0:
                prev = (chunk_start // C) - 1
                S_prev = tl.load(bnd_ptr + pid * Nc * KV + prev * KV + off2d, mask=mask2d, other=0.0).to(tl.float32)

        dqv = scale * tl.sum(S * do[None, :], axis=1)
        tl.store(dq_base + gt * K + off_k, dqv, mask=mask_k)

        S1 = S_prev * d[:, None]
        u = tl.sum(S1 * e[None, :], axis=1)
        S2 = S1 - cc * u[:, None] * e[None, :]
        pred = tl.sum(S2 * k[:, None], axis=0)
        r = v - pred

        G = Carry + scale * q[:, None] * do[None, :]
        w = tl.sum(G * k[:, None], axis=0)
        Gr = tl.sum(G * r[None, :], axis=1)
        S2w = tl.sum(S2 * w[None, :], axis=1)
        dkv = b * (Gr - S2w)
        tl.store(dk_base + gt * K + off_k, dkv, mask=mask_k)
        dvv = b * w
        tl.store(dv_base + gt * V + off_v, dvv, mask=mask_v)
        dbv = tl.sum(w * r)
        tl.store(db_base + gt, dbv)
        G2 = G - b * k[:, None] * w[None, :]
        z = tl.sum(G2 * e[None, :], axis=1)
        G2Tu = tl.sum(G2 * u[:, None], axis=0)
        S1Tz = tl.sum(S1 * z[:, None], axis=0)
        dcv = -tl.sum(u * z)
        tl.store(dc_base + gt, dcv)
        dev = -cc * (S1Tz + G2Tu)
        tl.store(de_base + gt * K + off_v, dev, mask=mask_v)
        G2 = G2 - cc * z[:, None] * e[None, :]
        dgv = tl.sum(G2 * S1, axis=1)
        tl.store(dg_base + gt * K + off_k, dgv, mask=mask_k)
        Carry = G2 * d[:, None]

    tl.store(carry_base + off2d, Carry, mask=mask2d)


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale_f):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        C = 64
        Nc = (T + C - 1) // C if T > 0 else 0
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        e_c = e.contiguous()
        g_c = g.contiguous()
        b_c = beta.contiguous()
        c_c = gamma.contiguous()
        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        if BH == 0 or T == 0 or K == 0 or V == 0:
            ctx.save_for_backward(q_c, k_c, v_c, e_c, g_c, b_c, c_c)
            ctx.B, ctx.H, ctx.T, ctx.K, ctx.V, ctx.BH, ctx.Nc, ctx.C = B, H, T, K, V, BH, Nc, C
            ctx.scale_f = scale_f
            ctx.bnd = torch.empty((0,), device=q.device, dtype=torch.float32)
            return out
        bnd = torch.empty((BH, Nc, K, V), device=q.device, dtype=torch.float32)
        BK = max(16, _npo2(K))
        BV = max(16, _npo2(V))
        nw = 8 if max(BK, BV) >= 64 else 4
        _fwd_generic[(BH,)](
            q_c, k_c, v_c, e_c, g_c, b_c, c_c, out, bnd,
            T, Nc, C, float(scale_f), K, V,
            BLOCK_K=BK, BLOCK_V=BV, num_warps=nw,
        )
        ctx.save_for_backward(q_c, k_c, v_c, e_c, g_c, b_c, c_c)
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V, ctx.BH, ctx.Nc, ctx.C = B, H, T, K, V, BH, Nc, C
        ctx.scale_f = float(scale_f)
        ctx.bnd = bnd
        ctx.q_dtype, ctx.k_dtype, ctx.v_dtype, ctx.e_dtype = q.dtype, k.dtype, v.dtype, e.dtype
        ctx.g_dtype, ctx.b_dtype, ctx.c_dtype = g.dtype, beta.dtype, gamma.dtype
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q_c, k_c, v_c, e_c, g_c, b_c, c_c = ctx.saved_tensors
        B, H, T, K, V, BH, Nc, C = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V, ctx.BH, ctx.Nc, ctx.C
        scale_f = ctx.scale_f
        if BH == 0 or T == 0 or K == 0 or V == 0:
            zq = torch.zeros_like(q_c).to(ctx.q_dtype) if BH != 0 else torch.empty_like(q_c)
            zk = torch.zeros_like(k_c).to(ctx.k_dtype) if BH != 0 else torch.empty_like(k_c)
            zv = torch.zeros_like(v_c).to(ctx.v_dtype) if BH != 0 else torch.empty_like(v_c)
            ze = torch.zeros_like(e_c).to(ctx.e_dtype) if BH != 0 else torch.empty_like(e_c)
            zg = torch.zeros_like(g_c)
            zb = torch.zeros_like(b_c)
            zc = torch.zeros_like(c_c)
            return zq, zk, zv, ze, zg, zb, zc, None
        do_c = grad_output.contiguous().to(torch.float32)
        dq = torch.empty((B, H, T, K), device=q_c.device, dtype=torch.float32)
        dk = torch.empty((B, H, T, K), device=q_c.device, dtype=torch.float32)
        dv = torch.empty((B, H, T, V), device=q_c.device, dtype=torch.float32)
        de = torch.empty((B, H, T, K), device=q_c.device, dtype=torch.float32)
        dg = torch.empty((B, H, T, K), device=q_c.device, dtype=torch.float32)
        db = torch.empty((B, H, T), device=q_c.device, dtype=torch.float32)
        dc = torch.empty((B, H, T), device=q_c.device, dtype=torch.float32)
        carry = torch.zeros((BH, K, V), device=q_c.device, dtype=torch.float32)
        scratch = torch.empty((BH, C, K, V), device=q_c.device, dtype=torch.float32)
        BK = max(16, _npo2(K))
        BV = max(16, _npo2(V))
        nw = 8 if max(BK, BV) >= 64 else 4
        for idx in range(Nc - 1, -1, -1):
            cs = idx * C
            cl = min(C, T - cs)
            _bwd_generic[(BH,)](
                q_c, k_c, v_c, e_c, g_c, b_c, c_c, do_c,
                ctx.bnd, scratch, carry,
                dq, dk, dv, de, dg, db, dc,
                cs, cl, T, Nc, C, float(scale_f), K, V,
                BLOCK_K=BK, BLOCK_V=BV, num_warps=nw,
            )
        return (
            dq.to(ctx.q_dtype),
            dk.to(ctx.k_dtype),
            dv.to(ctx.v_dtype),
            de.to(ctx.e_dtype),
            dg.to(ctx.g_dtype),
            db.to(ctx.b_dtype),
            dc.to(ctx.c_dtype),
            None,
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if torch.is_tensor(scale):
        try:
            scale_f = float(scale.detach().reshape(-1)[0].to(torch.float32).item())
        except Exception:
            scale_f = float(scale.item())
    else:
        scale_f = float(scale)
    return _Func.apply(q, k, v, e, g, beta, gamma, scale_f)
```
My approach fuses the exact fp32 recurrence in Triton with chunk-boundary checkpointing so backward recomputes each 64-step forward slice into scratch then scans adjoints in reverse, keeping only one KxV adjoint tile live to stay correct with no decay. I keep all reductions in fp32 and derive closed-form vector grads for q,k,v,e,g,beta,gamma to avoid storing full trajectories. I am unsure about Triton masking and register spilling for 128x128 fp32 tiles and about strict handling of non-square or large-K shapes.