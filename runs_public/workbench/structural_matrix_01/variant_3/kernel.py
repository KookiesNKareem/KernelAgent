import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_save_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr, s_ptr,
    T, K, V, scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < V

    q_base = pid * T * K
    k_base = pid * T * K
    v_base = pid * T * V
    e_base = pid * T * V
    g_base = pid * T * K
    b_base = pid * T
    o_base = pid * T * V
    s_base = pid * T * K * V

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in range(T):
        kt = tl.load(k_ptr + k_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        qt = tl.load(q_ptr + q_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        gt = tl.load(g_ptr + g_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        vt = tl.load(v_ptr + v_base + t * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        et = tl.load(e_ptr + e_base + t * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        bt = tl.load(beta_ptr + b_base + t).to(tl.float32)
        ct = tl.load(gamma_ptr + b_base + t).to(tl.float32)

        d = tl.exp(gt)
        S = S * d[:, None]
        p = tl.sum(S * et[None, :], axis=1)
        S = S - (ct * p)[:, None] * et[None, :]
        pred = tl.sum(S * kt[:, None], axis=0)
        err = (vt - pred) * bt
        S = S + kt[:, None] * err[None, :]
        qts = qt * scale
        out = tl.sum(S * qts[:, None], axis=0)
        tl.store(o_ptr + o_base + t * V + offs_v, out, mask=mask_v)

        s_ptrs = s_ptr + s_base + t * K * V + offs_k[:, None] * V + offs_v[None, :]
        s_mask = mask_k[:, None] & mask_v[None, :]
        tl.store(s_ptrs, S, mask=s_mask)


@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr,
    do_ptr, s_ptr,
    dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
    T, K, V, scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < V
    s_mask = mask_k[:, None] & mask_v[None, :]

    q_base = pid * T * K
    k_base = pid * T * K
    v_base = pid * T * V
    e_base = pid * T * V
    g_base = pid * T * K
    b_base = pid * T
    o_base = pid * T * V
    s_base = pid * T * K * V

    C = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for ti in range(T):
        t = T - 1 - ti
        kt = tl.load(k_ptr + k_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        qt = tl.load(q_ptr + q_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        vt = tl.load(v_ptr + v_base + t * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        et = tl.load(e_ptr + e_base + t * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        gt = tl.load(g_ptr + g_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        bt = tl.load(beta_ptr + b_base + t).to(tl.float32)
        ct = tl.load(gamma_ptr + b_base + t).to(tl.float32)
        dot = tl.load(do_ptr + o_base + t * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)

        s_ptrs_t = s_ptr + s_base + t * K * V + offs_k[:, None] * V + offs_v[None, :]
        St = tl.load(s_ptrs_t, mask=s_mask, other=0.0).to(tl.float32)
        if t > 0:
            s_ptrs_p = s_ptr + s_base + (t - 1) * K * V + offs_k[:, None] * V + offs_v[None, :]
            Sp = tl.load(s_ptrs_p, mask=s_mask, other=0.0).to(tl.float32)
        else:
            Sp = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

        d = tl.exp(gt)
        U = Sp * d[:, None]
        w = tl.sum(U * et[None, :], axis=1)
        P = U - (ct * w)[:, None] * et[None, :]
        pred = tl.sum(P * kt[:, None], axis=0)
        r = vt - pred
        err = r * bt

        DS = C + scale * (qt[:, None] * dot[None, :])
        s_vec = tl.sum(DS * kt[:, None], axis=0)

        dq = scale * tl.sum(St * dot[None, :], axis=1)
        dv = bt * s_vec
        dbeta = tl.sum(s_vec * r, axis=0)

        t1 = tl.sum(DS * err[None, :], axis=1)
        t2 = tl.sum(P * s_vec[None, :], axis=1)
        dk = t1 - bt * t2

        DP = DS - (bt * kt)[:, None] * s_vec[None, :]
        a = tl.sum(DP * et[None, :], axis=1)
        u1 = tl.sum(U * a[:, None], axis=0)
        u2 = tl.sum(DP * w[:, None], axis=0)
        de = (-ct) * (u1 + u2)
        dgamma = -tl.sum(w * a, axis=0)

        DU = DP - (ct * a)[:, None] * et[None, :]
        dd = tl.sum(DU * Sp, axis=1)
        dg = d * dd

        C = DU * d[:, None]

        tl.store(dq_ptr + q_base + t * K + offs_k, dq, mask=mask_k)
        tl.store(dk_ptr + k_base + t * K + offs_k, dk, mask=mask_k)
        tl.store(dg_ptr + g_base + t * K + offs_k, dg, mask=mask_k)
        tl.store(dv_ptr + v_base + t * V + offs_v, dv, mask=mask_v)
        tl.store(de_ptr + e_base + t * V + offs_v, de, mask=mask_v)
        tl.store(dbeta_ptr + b_base + t, dbeta)
        tl.store(dgamma_ptr + b_base + t, dgamma)


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        Vv = v.shape[-1]
        qc = q.contiguous()
        kc = k.contiguous()
        vc = v.contiguous()
        ec = e.contiguous()
        gc = g.contiguous()
        betac = beta.contiguous()
        gammac = gamma.contiguous()
        out_dtype = torch.promote_types(q.dtype, torch.float32)
        o = torch.empty(B, H, T, Vv, dtype=out_dtype, device=q.device)
        if T == 0:
            ctx.saved = None
            ctx.dims = (B, H, T, K, Vv)
            ctx.scale = float(scale) if not isinstance(scale, float) else scale
            ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
            ctx.has_saved = False
            return o
        Ssave = torch.empty(B, H, T, K, Vv, dtype=torch.float32, device=q.device)
        def _np2(x):
            return 1 if x <= 1 else 1 << (x - 1).bit_length()
        BK = _np2(K)
        BV = _np2(Vv)
        scale_f = float(scale)
        grid = (B * H,)
        nw = 4
        if BK * BV > 2048:
            nw = 8
        _fwd_save_kernel[grid](
            qc, kc, vc, ec, gc, betac, gammac, o, Ssave,
            T, K, Vv, scale_f, BK, BV, num_warps=nw,
        )
        ctx.save_for_backward(qc, kc, vc, ec, gc, betac, gammac, Ssave)
        ctx.dims = (B, H, T, K, Vv, BK, BV)
        ctx.scale = scale_f
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
        ctx.has_saved = True
        return o

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.has_saved:
            B, H, T, K, Vv = ctx.dims
            qd, kd, vd, ed, gd, betad, gammad = ctx.dtypes
            def _empty(sh, dt, dev):
                return torch.empty(sh, dtype=dt, device=dev)
            dev = grad_output.device
            return (
                _empty((B, H, T, K), qd, dev),
                _empty((B, H, T, K), kd, dev),
                _empty((B, H, T, Vv), vd, dev),
                _empty((B, H, T, Vv), ed, dev),
                _empty((B, H, T, K), gd, dev),
                _empty((B, H, T), betad, dev),
                _empty((B, H, T), gammad, dev),
                None,
            )
        qc, kc, vc, ec, gc, betac, gammac, Ssave = ctx.saved_tensors
        B, H, T, K, Vv, BK, BV = ctx.dims
        scale_f = ctx.scale
        qd, kd, vd, ed, gd, betad, gammad = ctx.dtypes
        dev = qc.device
        go = grad_output.contiguous().to(torch.float32)
        dq32 = torch.empty_like(qc, dtype=torch.float32)
        dk32 = torch.empty_like(kc, dtype=torch.float32)
        dv32 = torch.empty_like(vc, dtype=torch.float32)
        de32 = torch.empty_like(ec, dtype=torch.float32)
        dg32 = torch.empty_like(gc, dtype=torch.float32)
        db32 = torch.empty_like(betac, dtype=torch.float32)
        dgam32 = torch.empty_like(gammac, dtype=torch.float32)
        grid = (B * H,)
        nw = 4
        if BK * BV > 2048:
            nw = 8
        _bwd_kernel[grid](
            qc, kc, vc, ec, gc, betac, gammac,
            go, Ssave,
            dq32, dk32, dv32, de32, dg32, db32, dgam32,
            T, K, Vv, scale_f, BK, BV, num_warps=nw,
        )
        def _cast(x32, dt):
            if x32.dtype == dt:
                return x32
            return x32.to(dt)
        return (
            _cast(dq32, qd),
            _cast(dk32, kd),
            _cast(dv32, vd),
            _cast(de32, ed),
            _cast(dg32, gd),
            _cast(db32, betad),
            _cast(dgam32, gammad),
            None,
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _Func.apply(q, k, v, e, g, beta, gamma, scale)
