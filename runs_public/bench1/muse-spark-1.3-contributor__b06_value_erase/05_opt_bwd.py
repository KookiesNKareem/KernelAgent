import torch
import triton
import triton.language as tl


def _npo2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


# ---------------- fast 128 path, C=64 hard coded ----------------

@triton.jit
def _fwd_fast128(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, b_ptr, c_ptr, o_ptr, bnd_ptr,
                 T, Nc, scale):
    pid = tl.program_id(0)
    off = tl.arange(0, 128)
    off2d = off[:, None] * 128 + off[None, :]
    q_base = q_ptr + pid * T * 128
    k_base = k_ptr + pid * T * 128
    e_base = e_ptr + pid * T * 128
    d_base = d_ptr + pid * T * 128
    v_base = v_ptr + pid * T * 128
    b_base = b_ptr + pid * T
    c_base = c_ptr + pid * T
    o_base = o_ptr + pid * T * 128
    S = tl.zeros([128, 128], dtype=tl.float32)
    for t in range(T):
        q = tl.load(q_base + t * 128 + off).to(tl.float32)
        k = tl.load(k_base + t * 128 + off).to(tl.float32)
        e = tl.load(e_base + t * 128 + off).to(tl.float32)
        v = tl.load(v_base + t * 128 + off).to(tl.float32)
        d = tl.load(d_base + t * 128 + off)
        b = tl.load(b_base + t)
        cc = tl.load(c_base + t)
        S = S * d[:, None]
        u = tl.sum(S * e[None, :], axis=1)
        S = S - cc * u[:, None] * e[None, :]
        pred = tl.sum(S * k[:, None], axis=0)
        diff = v - pred
        S = S + b * k[:, None] * diff[None, :]
        o = tl.sum(S * q[:, None], axis=0) * scale
        tl.store(o_base + t * 128 + off, o)
        do_store = ((t & 63) == 63) | (t == T - 1)
        if do_store:
            chunk = t >> 6
            tl.store(bnd_ptr + pid * Nc * 16384 + chunk * 16384 + off2d, S)


@triton.jit
def _bwd_fast128(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, b_ptr, c_ptr, do_ptr,
                 bnd_ptr, scratch_ptr, carry_ptr,
                 dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, db_ptr, dc_ptr,
                 chunk_start, chunk_len, T, Nc, scale):
    pid = tl.program_id(0)
    off = tl.arange(0, 128)
    off2d = off[:, None] * 128 + off[None, :]
    q_base = q_ptr + pid * T * 128
    k_base = k_ptr + pid * T * 128
    e_base = e_ptr + pid * T * 128
    d_base = d_ptr + pid * T * 128
    v_base = v_ptr + pid * T * 128
    b_base = b_ptr + pid * T
    c_base = c_ptr + pid * T
    do_base = do_ptr + pid * T * 128
    dq_base = dq_ptr + pid * T * 128
    dk_base = dk_ptr + pid * T * 128
    dv_base = dv_ptr + pid * T * 128
    de_base = de_ptr + pid * T * 128
    dg_base = dg_ptr + pid * T * 128
    db_base = db_ptr + pid * T
    dc_base = dc_ptr + pid * T

    S_init = tl.zeros([128, 128], dtype=tl.float32)
    if chunk_start != 0:
        prev = (chunk_start >> 6) - 1
        S_init = tl.load(bnd_ptr + pid * Nc * 16384 + prev * 16384 + off2d).to(tl.float32)
    S_cur = S_init
    for t in range(chunk_len):
        gt = chunk_start + t
        k = tl.load(k_base + gt * 128 + off).to(tl.float32)
        e = tl.load(e_base + gt * 128 + off).to(tl.float32)
        v = tl.load(v_base + gt * 128 + off).to(tl.float32)
        d = tl.load(d_base + gt * 128 + off)
        b = tl.load(b_base + gt)
        cc = tl.load(c_base + gt)
        S_cur = S_cur * d[:, None]
        u = tl.sum(S_cur * e[None, :], axis=1)
        S_cur = S_cur - cc * u[:, None] * e[None, :]
        pred = tl.sum(S_cur * k[:, None], axis=0)
        diff = v - pred
        S_cur = S_cur + b * k[:, None] * diff[None, :]
        tl.store(scratch_ptr + pid * 64 * 16384 + t * 16384 + off2d,
                 S_cur.to(tl.bfloat16))

    Carry = tl.load(carry_ptr + pid * 16384 + off2d).to(tl.float32)

    S_cur_reg = S_cur
    S_prev_reg = S_init
    if chunk_len > 1:
        S_prev_reg = tl.load(scratch_ptr + pid * 64 * 16384 + (chunk_len - 2) * 16384 + off2d).to(tl.float32)

    for _i in range(chunk_len):
        t = chunk_len - 1 - _i
        gt = chunk_start + t
        q = tl.load(q_base + gt * 128 + off).to(tl.float32)
        k = tl.load(k_base + gt * 128 + off).to(tl.float32)
        e = tl.load(e_base + gt * 128 + off).to(tl.float32)
        v = tl.load(v_base + gt * 128 + off).to(tl.float32)
        d = tl.load(d_base + gt * 128 + off)
        b = tl.load(b_base + gt)
        cc = tl.load(c_base + gt)
        do = tl.load(do_base + gt * 128 + off).to(tl.float32)

        S = S_cur_reg
        S_prev = S_prev_reg

        dqv = scale * tl.sum(S * do[None, :], axis=1)
        tl.store(dq_base + gt * 128 + off, dqv)

        S1 = S_prev * d[:, None]
        u = tl.sum(S1 * e[None, :], axis=1)
        S2 = S1 - cc * u[:, None] * e[None, :]
        pred = tl.sum(S2 * k[:, None], axis=0)
        r = v - pred

        G = Carry + scale * q[:, None] * do[None, :]
        w = tl.sum(G * k[:, None], axis=0)
        Gr = tl.sum(G * r[None, :], axis=1)
        S2w = tl.sum(S2 * w[None, :], axis=1)
        tl.store(dk_base + gt * 128 + off, b * (Gr - S2w))
        tl.store(dv_base + gt * 128 + off, b * w)
        tl.store(db_base + gt, tl.sum(w * r))

        G = G - b * k[:, None] * w[None, :]
        z = tl.sum(G * e[None, :], axis=1)
        Gtu = tl.sum(G * u[:, None], axis=0)
        Stz = tl.sum(S1 * z[:, None], axis=0)
        tl.store(dc_base + gt, -tl.sum(u * z))
        tl.store(de_base + gt * 128 + off, -cc * (Stz + Gtu))
        G = G - cc * z[:, None] * e[None, :]
        tl.store(dg_base + gt * 128 + off, tl.sum(G * S1, axis=1))
        Carry = G * d[:, None]

        if t > 0:
            S_cur_reg = S_prev_reg
            if t >= 2:
                S_prev_reg = tl.load(scratch_ptr + pid * 64 * 16384 + (t - 2) * 16384 + off2d).to(tl.float32)
            else:
                S_prev_reg = S_init

    tl.store(carry_ptr + pid * 16384 + off2d, Carry)


# ---------------- generic fallback (d precomputed, no exp) ----------------

@triton.jit
def _fwd_generic(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, b_ptr, c_ptr, o_ptr, bnd_ptr,
                 T, Nc, C, scale, K, V,
                 BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
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
    d_base = d_ptr + pid * T * K
    v_base = v_ptr + pid * T * V
    b_base = b_ptr + pid * T
    c_base = c_ptr + pid * T
    o_base = o_ptr + pid * T * V
    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    for t in range(T):
        q = tl.load(q_base + t * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        k = tl.load(k_base + t * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        e = tl.load(e_base + t * K + off_v, mask=mask_v, other=0.0).to(tl.float32)
        d = tl.load(d_base + t * K + off_k, mask=mask_k, other=1.0)
        v = tl.load(v_base + t * V + off_v, mask=mask_v, other=0.0).to(tl.float32)
        b = tl.load(b_base + t)
        cc = tl.load(c_base + t)
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
def _bwd_generic(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, b_ptr, c_ptr, do_ptr,
                 bnd_ptr, scratch_ptr, carry_ptr,
                 dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, db_ptr, dc_ptr,
                 chunk_start, chunk_len, T, Nc, C, scale, K, V,
                 BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
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
    d_base = d_ptr + pid * T * K
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
    S_cur = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    if chunk_start != 0:
        prev = (chunk_start // C) - 1
        S_cur = tl.load(bnd_ptr + pid * Nc * KV + prev * KV + off2d, mask=mask2d, other=0.0).to(tl.float32)
    for t in range(chunk_len):
        gt = chunk_start + t
        k = tl.load(k_base + gt * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        e = tl.load(e_base + gt * K + off_v, mask=mask_v, other=0.0).to(tl.float32)
        v = tl.load(v_base + gt * V + off_v, mask=mask_v, other=0.0).to(tl.float32)
        d = tl.load(d_base + gt * K + off_k, mask=mask_k, other=1.0)
        b = tl.load(b_base + gt)
        cc = tl.load(c_base + gt)
        S_cur = S_cur * d[:, None]
        u = tl.sum(S_cur * e[None, :], axis=1)
        S_cur = S_cur - cc * u[:, None] * e[None, :]
        pred = tl.sum(S_cur * k[:, None], axis=0)
        diff = v - pred
        S_cur = S_cur + b * k[:, None] * diff[None, :]
        tl.store(scratch_ptr + pid * C * KV + t * KV + off2d, S_cur, mask=mask2d)
    Carry = tl.load(carry_ptr + pid * KV + off2d, mask=mask2d, other=0.0).to(tl.float32)
    for _i in range(chunk_len):
        t = chunk_len - 1 - _i
        gt = chunk_start + t
        q = tl.load(q_base + gt * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        k = tl.load(k_base + gt * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
        e = tl.load(e_base + gt * K + off_v, mask=mask_v, other=0.0).to(tl.float32)
        v = tl.load(v_base + gt * V + off_v, mask=mask_v, other=0.0).to(tl.float32)
        d = tl.load(d_base + gt * K + off_k, mask=mask_k, other=1.0)
        b = tl.load(b_base + gt)
        cc = tl.load(c_base + gt)
        do = tl.load(do_base + gt * V + off_v, mask=mask_v, other=0.0).to(tl.float32)
        S = tl.load(scratch_ptr + pid * C * KV + t * KV + off2d, mask=mask2d, other=0.0).to(tl.float32)
        S_prev = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
        if t > 0:
            S_prev = tl.load(scratch_ptr + pid * C * KV + (t - 1) * KV + off2d, mask=mask2d, other=0.0).to(tl.float32)
        else:
            if chunk_start != 0:
                prev = (chunk_start // C) - 1
                S_prev = tl.load(bnd_ptr + pid * Nc * KV + prev * KV + off2d, mask=mask2d, other=0.0).to(tl.float32)
        tl.store(dq_base + gt * K + off_k, scale * tl.sum(S * do[None, :], axis=1), mask=mask_k)
        S1 = S_prev * d[:, None]
        u = tl.sum(S1 * e[None, :], axis=1)
        S2 = S1 - cc * u[:, None] * e[None, :]
        pred = tl.sum(S2 * k[:, None], axis=0)
        r = v - pred
        G = Carry + scale * q[:, None] * do[None, :]
        w = tl.sum(G * k[:, None], axis=0)
        Gr = tl.sum(G * r[None, :], axis=1)
        S2w = tl.sum(S2 * w[None, :], axis=1)
        tl.store(dk_base + gt * K + off_k, b * (Gr - S2w), mask=mask_k)
        tl.store(dv_base + gt * V + off_v, b * w, mask=mask_v)
        tl.store(db_base + gt, tl.sum(w * r))
        G = G - b * k[:, None] * w[None, :]
        z = tl.sum(G * e[None, :], axis=1)
        Gtu = tl.sum(G * u[:, None], axis=0)
        Stz = tl.sum(S1 * z[:, None], axis=0)
        tl.store(dc_base + gt, -tl.sum(u * z))
        tl.store(de_base + gt * K + off_v, -cc * (Stz + Gtu), mask=mask_v)
        G = G - cc * z[:, None] * e[None, :]
        tl.store(dg_base + gt * K + off_k, tl.sum(G * S1, axis=1), mask=mask_k)
        Carry = G * d[:, None]
    tl.store(carry_ptr + pid * KV + off2d, Carry, mask=mask2d)


class _FastFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale_f):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        Nc = (T + 63) // 64
        q_c = q if q.is_contiguous() else q.contiguous()
        k_c = k if k.is_contiguous() else k.contiguous()
        v_c = v if v.is_contiguous() else v.contiguous()
        e_c = e if e.is_contiguous() else e.contiguous()
        g_c = g if g.is_contiguous() else g.contiguous()
        b_c = beta if beta.is_contiguous() else beta.contiguous()
        c_c = gamma if gamma.is_contiguous() else gamma.contiguous()
        with torch.no_grad():
            d_c = torch.exp(g_c).contiguous()
        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        bnd = torch.empty((BH, Nc, K, V), device=q.device, dtype=torch.float32)
        if BH and T:
            _fwd_fast128[(BH,)](q_c, k_c, v_c, e_c, d_c, b_c, c_c, out, bnd,
                                 T, Nc, float(scale_f), num_warps=8)
        ctx.save_for_backward(q_c, k_c, v_c, e_c, d_c, b_c, c_c, bnd)
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V, ctx.BH, ctx.Nc = B, H, T, K, V, BH, Nc
        ctx.scale_f = float(scale_f)
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q_c, k_c, v_c, e_c, d_c, b_c, c_c, bnd = ctx.saved_tensors
        B, H, T, K, V, BH, Nc = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V, ctx.BH, ctx.Nc
        scale_f = ctx.scale_f
        if BH == 0 or T == 0:
            z = lambda t, dt: torch.empty_like(t).to(dt) if hasattr(t, 'to') else t
            return (torch.empty_like(q_c), torch.empty_like(k_c), torch.empty_like(v_c),
                    torch.empty_like(e_c), torch.empty_like(d_c), torch.empty_like(b_c),
                    torch.empty_like(c_c), None)
        do_c = grad_output.contiguous().to(torch.float32) if grad_output.dtype != torch.float32 else (grad_output if grad_output.is_contiguous() else grad_output.contiguous())
        dq = torch.empty((B, H, T, K), device=q_c.device, dtype=torch.float32)
        dk = torch.empty((B, H, T, K), device=q_c.device, dtype=torch.float32)
        dv = torch.empty((B, H, T, V), device=q_c.device, dtype=torch.float32)
        de = torch.empty((B, H, T, K), device=q_c.device, dtype=torch.float32)
        dg = torch.empty((B, H, T, K), device=q_c.device, dtype=torch.float32)
        db = torch.empty((B, H, T), device=q_c.device, dtype=torch.float32)
        dc = torch.empty((B, H, T), device=q_c.device, dtype=torch.float32)
        carry = torch.zeros((BH, K, V), device=q_c.device, dtype=torch.float32)
        scratch = torch.empty((BH, 64, K, V), device=q_c.device, dtype=torch.bfloat16)
        for idx in range(Nc - 1, -1, -1):
            cs = idx * 64
            cl = min(64, T - cs)
            _bwd_fast128[(BH,)](q_c, k_c, v_c, e_c, d_c, b_c, c_c, do_c, bnd, scratch, carry,
                                dq, dk, dv, de, dg, db, dc, cs, cl, T, Nc, float(scale_f), num_warps=8)
        qd, kd, vd, ed, gd, bd, cd = ctx.dtypes
        return (dq.to(qd), dk.to(kd), dv.to(vd), de.to(ed), dg.to(gd), db.to(bd), dc.to(cd), None)


class _GenFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale_f):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        C = 64
        Nc = (T + C - 1) // C if T else 0
        q_c = q if q.is_contiguous() else q.contiguous()
        k_c = k if k.is_contiguous() else k.contiguous()
        v_c = v if v.is_contiguous() else v.contiguous()
        e_c = e if e.is_contiguous() else e.contiguous()
        g_c = g if g.is_contiguous() else g.contiguous()
        b_c = beta if beta.is_contiguous() else beta.contiguous()
        c_c = gamma if gamma.is_contiguous() else gamma.contiguous()
        with torch.no_grad():
            d_c = torch.exp(g_c.to(torch.float32)).contiguous() if g_c.dtype != torch.float32 else torch.exp(g_c).contiguous()
        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        if BH == 0 or T == 0 or K == 0 or V == 0:
            ctx.save_for_backward(q_c, k_c, v_c, e_c, d_c, b_c, c_c)
            ctx.args = (B, H, T, K, V, BH, Nc, C)
            ctx.scale_f = float(scale_f)
            ctx.bnd = torch.empty((0,), device=q.device, dtype=torch.float32)
            ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
            return out
        bnd = torch.empty((BH, Nc, K, V), device=q.device, dtype=torch.float32)
        BK = max(16, _npo2(K))
        BV = max(16, _npo2(V))
        nw = 8 if max(BK, BV) >= 64 else 4
        _fwd_generic[(BH,)](q_c, k_c, v_c, e_c, d_c, b_c, c_c, out, bnd, T, Nc, C,
                            float(scale_f), K, V, BLOCK_K=BK, BLOCK_V=BV, num_warps=nw)
        ctx.save_for_backward(q_c, k_c, v_c, e_c, d_c, b_c, c_c)
        ctx.args = (B, H, T, K, V, BH, Nc, C)
        ctx.scale_f = float(scale_f)
        ctx.bnd = bnd
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q_c, k_c, v_c, e_c, d_c, b_c, c_c = ctx.saved_tensors
        B, H, T, K, V, BH, Nc, C = ctx.args
        scale_f = ctx.scale_f
        if BH == 0 or T == 0 or K == 0 or V == 0:
            return (torch.zeros_like(q_c), torch.zeros_like(k_c), torch.zeros_like(v_c),
                    torch.zeros_like(e_c), torch.zeros_like(d_c), torch.zeros_like(b_c),
                    torch.zeros_like(c_c), None)
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
            _bwd_generic[(BH,)](q_c, k_c, v_c, e_c, d_c, b_c, c_c, do_c, ctx.bnd, scratch, carry,
                                dq, dk, dv, de, dg, db, dc, cs, cl, T, Nc, C,
                                float(scale_f), K, V, BLOCK_K=BK, BLOCK_V=BV, num_warps=nw)
        qd, kd, vd, ed, gd, bd, cd = ctx.dtypes
        return (dq.to(qd), dk.to(kd), dv.to(vd), de.to(ed), dg.to(gd), db.to(bd), dc.to(cd), None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if torch.is_tensor(scale):
        try:
            scale_f = float(scale.detach().reshape(-1)[0].to(torch.float32).item())
        except Exception:
            scale_f = float(scale.item())
    else:
        scale_f = float(scale)
    B, H, T, K = q.shape
    V = v.shape[-1]
    if K == 128 and V == 128:
        return _FastFunc.apply(q, k, v, e, g, beta, gamma, scale_f)
    return _GenFunc.apply(q, k, v, e, g, beta, gamma, scale_f)
