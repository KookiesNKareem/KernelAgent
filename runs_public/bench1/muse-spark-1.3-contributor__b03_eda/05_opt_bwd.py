import torch
import triton
import triton.language as tl


def _npo2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


def _choose_blocks(Kdim: int, Vdim: int):
    BK = _npo2(max(1, Kdim))
    if BK < 16:
        BK = 16
    BV = _npo2(max(1, Vdim))
    if BV < 16:
        BV = 16
    if BK * BV <= 16384:
        NV = (Vdim + BV - 1) // BV if Vdim > 0 else 1
        if NV < 1:
            NV = 1
        return BK, BV, NV
    # large product: shard V to keep product <= 8192
    BK = _npo2(max(1, Kdim))
    if BK < 16:
        BK = 16
    target = 8192 // BK
    if target < 16:
        target = 16
    BV = 1 << (target.bit_length() - 1)
    if BV < 16:
        BV = 16
    if BV > _npo2(max(1, Vdim)) and Vdim <= 8192:
        BV = _npo2(max(1, Vdim))
    NV = (Vdim + BV - 1) // BV if Vdim > 0 else 1
    return BK, BV, NV


@triton.jit
def _fwd_train_kernel(
    Q, Kptr, Vptr, Eptr, Gptr, Bptr, Cptr, Optr, Ckptptr,
    scale,
    T, Kdim, Vdim, Nc, NV,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    EVEN_K: tl.constexpr, EVEN_V: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    v_off = pid_v * BLOCK_V + off_v
    if EVEN_K:
        k_mask = None
    else:
        k_mask = off_k < Kdim
    if EVEN_V:
        v_mask = None
    else:
        v_mask = v_off < Vdim

    Qbh = Q + pid_bh * T * Kdim
    Kb = Kptr + pid_bh * T * Kdim
    Eb = Eptr + pid_bh * T * Kdim
    Vb = Vptr + pid_bh * T * Vdim
    Gb = Gptr + pid_bh * T
    Bb = Bptr + pid_bh * T
    Cb = Cptr + pid_bh * T
    Ob = Optr + pid_bh * T * Vdim

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    cidx = 0
    cnt = CHUNK
    for t in range(T):
        if EVEN_K:
            qk = tl.load(Qbh + t * Kdim + off_k).to(tl.float32)
            kk = tl.load(Kb + t * Kdim + off_k).to(tl.float32)
            ek = tl.load(Eb + t * Kdim + off_k).to(tl.float32)
        else:
            qk = tl.load(Qbh + t * Kdim + off_k, mask=k_mask, other=0.0).to(tl.float32)
            kk = tl.load(Kb + t * Kdim + off_k, mask=k_mask, other=0.0).to(tl.float32)
            ek = tl.load(Eb + t * Kdim + off_k, mask=k_mask, other=0.0).to(tl.float32)
        if EVEN_V:
            vv = tl.load(Vb + t * Vdim + pid_v * BLOCK_V + off_v).to(tl.float32)
        else:
            vv = tl.load(Vb + t * Vdim + pid_v * BLOCK_V + off_v, mask=v_mask, other=0.0).to(tl.float32)
        d = tl.load(Gb + t).to(tl.float32)
        bb = tl.load(Bb + t).to(tl.float32)
        cc = tl.load(Cb + t).to(tl.float32)
        S = S * d
        se = tl.sum(S * ek[:, None], axis=0)
        S = S - cc * ek[:, None] * se[None, :]
        pred = tl.sum(S * kk[:, None], axis=0)
        diff = (vv - pred) * bb
        S = S + kk[:, None] * diff[None, :]
        o = tl.sum(S * (qk * scale)[:, None], axis=0)
        if EVEN_V:
            tl.store(Ob + t * Vdim + pid_v * BLOCK_V + off_v, o)
        else:
            tl.store(Ob + t * Vdim + pid_v * BLOCK_V + off_v, o, mask=v_mask)
        cnt = cnt - 1
        last = (t == T - 1)
        do_store = (cnt == 0) | last
        if do_store:
            base = Ckptptr + ((pid_bh * NV + pid_v) * Nc + cidx) * BLOCK_K * BLOCK_V
            ptrs = base + off_k[:, None] * BLOCK_V + off_v[None, :]
            tl.store(ptrs, S)
            cidx = cidx + 1
            cnt = CHUNK


@triton.jit
def _bwd_direct_kernel(
    Q, Kptr, Vptr, Eptr, Gptr, Bptr, Cptr, DOptr,
    DQptr, DKptr, DEptr, DVptr, DBptr, DCptr, DGptr,
    Ckptptr, Scratchptr, Carryptr,
    scale,
    T, Kdim, Vdim, s, C, Nc, cidx,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    EVEN_K: tl.constexpr, EVEN_V: tl.constexpr,
    CMAX: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    if EVEN_K:
        k_mask = None
    else:
        k_mask = off_k < Kdim
    if EVEN_V:
        v_mask = None
    else:
        v_mask = off_v < Vdim

    Qbh = Q + pid_bh * T * Kdim + s * Kdim
    Kb = Kptr + pid_bh * T * Kdim + s * Kdim
    Eb = Eptr + pid_bh * T * Kdim + s * Kdim
    Vb = Vptr + pid_bh * T * Vdim + s * Vdim
    Db = DOptr + pid_bh * T * Vdim + s * Vdim
    Gb = Gptr + pid_bh * T + s
    Bb = Bptr + pid_bh * T + s
    Cb = Cptr + pid_bh * T + s
    DQb = DQptr + pid_bh * T * Kdim + s * Kdim
    DKb = DKptr + pid_bh * T * Kdim + s * Kdim
    DEb = DEptr + pid_bh * T * Kdim + s * Kdim
    DVb = DVptr + pid_bh * T * Vdim + s * Vdim
    DBb = DBptr + pid_bh * T + s
    DCb = DCptr + pid_bh * T + s
    DGb = DGptr + pid_bh * T + s

    S_start = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    if cidx != 0:
        cbase = Ckptptr + (pid_bh * Nc + (cidx - 1)) * BLOCK_K * BLOCK_V
        cptrs = cbase + off_k[:, None] * BLOCK_V + off_v[None, :]
        S_start = tl.load(cptrs).to(tl.float32)

    sbase = Scratchptr + pid_bh * CMAX * BLOCK_K * BLOCK_V
    S_prev = S_start
    for t in range(C):
        if EVEN_K:
            kk = tl.load(Kb + t * Kdim + off_k).to(tl.float32)
            ek = tl.load(Eb + t * Kdim + off_k).to(tl.float32)
        else:
            kk = tl.load(Kb + t * Kdim + off_k, mask=k_mask, other=0.0).to(tl.float32)
            ek = tl.load(Eb + t * Kdim + off_k, mask=k_mask, other=0.0).to(tl.float32)
        if EVEN_V:
            vv = tl.load(Vb + t * Vdim + off_v).to(tl.float32)
        else:
            vv = tl.load(Vb + t * Vdim + off_v, mask=v_mask, other=0.0).to(tl.float32)
        d = tl.load(Gb + t).to(tl.float32)
        bb = tl.load(Bb + t).to(tl.float32)
        cc = tl.load(Cb + t).to(tl.float32)
        A = S_prev * d
        ae = tl.sum(A * ek[:, None], axis=0)
        Bmat = A - cc * ek[:, None] * ae[None, :]
        pred = tl.sum(Bmat * kk[:, None], axis=0)
        delta = vv - pred
        S_cur = Bmat + bb * kk[:, None] * delta[None, :]
        ptrs = sbase + t * BLOCK_K * BLOCK_V + off_k[:, None] * BLOCK_V + off_v[None, :]
        tl.store(ptrs, S_cur)
        S_prev = S_cur

    carry_base = Carryptr + pid_bh * BLOCK_K * BLOCK_V
    carry_ptrs = carry_base + off_k[:, None] * BLOCK_V + off_v[None, :]
    cur = tl.load(carry_ptrs).to(tl.float32)

    for ti in range(C):
        t = C - 1 - ti
        ptr_t = sbase + t * BLOCK_K * BLOCK_V + off_k[:, None] * BLOCK_V + off_v[None, :]
        S_t = tl.load(ptr_t).to(tl.float32)
        Sp = S_start
        if t != 0:
            ptr_p = sbase + (t - 1) * BLOCK_K * BLOCK_V + off_k[:, None] * BLOCK_V + off_v[None, :]
            Sp = tl.load(ptr_p).to(tl.float32)
        if EVEN_K:
            kk = tl.load(Kb + t * Kdim + off_k).to(tl.float32)
            ek = tl.load(Eb + t * Kdim + off_k).to(tl.float32)
            qk = tl.load(Qbh + t * Kdim + off_k).to(tl.float32)
        else:
            kk = tl.load(Kb + t * Kdim + off_k, mask=k_mask, other=0.0).to(tl.float32)
            ek = tl.load(Eb + t * Kdim + off_k, mask=k_mask, other=0.0).to(tl.float32)
            qk = tl.load(Qbh + t * Kdim + off_k, mask=k_mask, other=0.0).to(tl.float32)
        if EVEN_V:
            vv = tl.load(Vb + t * Vdim + off_v).to(tl.float32)
            do = tl.load(Db + t * Vdim + off_v).to(tl.float32)
        else:
            vv = tl.load(Vb + t * Vdim + off_v, mask=v_mask, other=0.0).to(tl.float32)
            do = tl.load(Db + t * Vdim + off_v, mask=v_mask, other=0.0).to(tl.float32)
        d = tl.load(Gb + t).to(tl.float32)
        bb = tl.load(Bb + t).to(tl.float32)
        cc = tl.load(Cb + t).to(tl.float32)

        A = Sp * d
        ae = tl.sum(A * ek[:, None], axis=0)
        Bmat = A - cc * ek[:, None] * ae[None, :]
        pred = tl.sum(Bmat * kk[:, None], axis=0)
        delta = vv - pred
        qs = qk * scale
        G = qs[:, None] * do[None, :] + cur
        s_vec = tl.sum(G * kk[:, None], axis=0)
        Gd = tl.sum(G * delta[None, :], axis=1)
        Bs = tl.sum(Bmat * s_vec[None, :], axis=1)
        dv = bb * s_vec
        if EVEN_V:
            tl.store(DVb + t * Vdim + off_v, dv)
        else:
            tl.store(DVb + t * Vdim + off_v, dv, mask=v_mask)
        db = tl.sum(s_vec * delta, axis=0)
        dk = bb * (Gd - Bs)
        dq = scale * tl.sum(S_t * do[None, :], axis=1)
        if EVEN_K:
            tl.store(DQb + t * Kdim + off_k, dq)
            tl.store(DKb + t * Kdim + off_k, dk)
        else:
            tl.store(DQb + t * Kdim + off_k, dq, mask=k_mask)
            tl.store(DKb + t * Kdim + off_k, dk, mask=k_mask)
        GB = G - bb * kk[:, None] * s_vec[None, :]
        ge = tl.sum(GB * ek[:, None], axis=0)
        dc = -tl.sum(ge * ae, axis=0)
        GBae = tl.sum(GB * ae[None, :], axis=1)
        Age = tl.sum(A * ge[None, :], axis=1)
        de = -cc * (GBae + Age)
        if EVEN_K:
            tl.store(DEb + t * Kdim + off_k, de)
        else:
            tl.store(DEb + t * Kdim + off_k, de, mask=k_mask)
        tl.store(DBb + t, db)
        tl.store(DCb + t, dc)
        GA = GB - cc * ek[:, None] * ge[None, :]
        tmp = tl.sum(GA * A, axis=0)
        dg = tl.sum(tmp, axis=0)
        tl.store(DGb + t, dg)
        cur = d * GA
    tl.store(carry_ptrs, cur)


@triton.jit
def _bwd_chunk_kernel(
    Q, Kptr, Vptr, Eptr, Gptr, Bptr, Cptr, DOptr, DVptr,
    DQp, DKp, DEp, DBp, DCp, DGp,
    Ckptptr, Scratchptr, Carryptr,
    scale,
    T, Kdim, Vdim, s, C, Cmax, Nc, cidx, NV,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    v_off_full = pid_v * BLOCK_V + off_v
    k_mask = off_k < Kdim
    v_mask = v_off_full < Vdim
    Qbh = Q + pid_bh * T * Kdim + s * Kdim
    Kb = Kptr + pid_bh * T * Kdim + s * Kdim
    Eb = Eptr + pid_bh * T * Kdim + s * Kdim
    Vb = Vptr + pid_bh * T * Vdim + s * Vdim + pid_v * BLOCK_V
    Db = DOptr + pid_bh * T * Vdim + s * Vdim + pid_v * BLOCK_V
    Gb = Gptr + pid_bh * T + s
    Bb = Bptr + pid_bh * T + s
    Cb = Cptr + pid_bh * T + s
    DVb = DVptr + pid_bh * T * Vdim + s * Vdim + pid_v * BLOCK_V
    S_start = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    if cidx != 0:
        cbase = Ckptptr + ((pid_bh * NV + pid_v) * Nc + (cidx - 1)) * BLOCK_K * BLOCK_V
        cptrs = cbase + off_k[:, None] * BLOCK_V + off_v[None, :]
        S_start = tl.load(cptrs).to(tl.float32)
    sbase = Scratchptr + (pid_bh * NV + pid_v) * Cmax * BLOCK_K * BLOCK_V
    S_prev = S_start
    for t in range(C):
        kk = tl.load(Kb + t * Kdim + off_k, mask=k_mask, other=0.0).to(tl.float32)
        ek = tl.load(Eb + t * Kdim + off_k, mask=k_mask, other=0.0).to(tl.float32)
        vv = tl.load(Vb + t * Vdim + off_v, mask=v_mask, other=0.0).to(tl.float32)
        d = tl.load(Gb + t).to(tl.float32)
        bb = tl.load(Bb + t).to(tl.float32)
        cc = tl.load(Cb + t).to(tl.float32)
        A = S_prev * d
        ae = tl.sum(A * ek[:, None], axis=0)
        Bmat = A - cc * ek[:, None] * ae[None, :]
        pred = tl.sum(Bmat * kk[:, None], axis=0)
        delta = vv - pred
        S_cur = Bmat + bb * kk[:, None] * delta[None, :]
        ptrs = sbase + t * BLOCK_K * BLOCK_V + off_k[:, None] * BLOCK_V + off_v[None, :]
        tl.store(ptrs, S_cur)
        S_prev = S_cur
    carry_base = Carryptr + (pid_bh * NV + pid_v) * BLOCK_K * BLOCK_V
    carry_ptrs = carry_base + off_k[:, None] * BLOCK_V + off_v[None, :]
    cur = tl.load(carry_ptrs).to(tl.float32)
    dq_base = DQp + (pid_bh * NV + pid_v) * Cmax * BLOCK_K
    dk_base = DKp + (pid_bh * NV + pid_v) * Cmax * BLOCK_K
    de_base = DEp + (pid_bh * NV + pid_v) * Cmax * BLOCK_K
    db_base = DBp + (pid_bh * NV + pid_v) * Cmax
    dc_base = DCp + (pid_bh * NV + pid_v) * Cmax
    dg_base = DGp + (pid_bh * NV + pid_v) * Cmax
    for ti in range(C):
        t = C - 1 - ti
        ptr_t = sbase + t * BLOCK_K * BLOCK_V + off_k[:, None] * BLOCK_V + off_v[None, :]
        S_t = tl.load(ptr_t).to(tl.float32)
        Sp = S_start
        if t != 0:
            ptr_p = sbase + (t - 1) * BLOCK_K * BLOCK_V + off_k[:, None] * BLOCK_V + off_v[None, :]
            Sp = tl.load(ptr_p).to(tl.float32)
        kk = tl.load(Kb + t * Kdim + off_k, mask=k_mask, other=0.0).to(tl.float32)
        ek = tl.load(Eb + t * Kdim + off_k, mask=k_mask, other=0.0).to(tl.float32)
        qk = tl.load(Qbh + t * Kdim + off_k, mask=k_mask, other=0.0).to(tl.float32)
        vv = tl.load(Vb + t * Vdim + off_v, mask=v_mask, other=0.0).to(tl.float32)
        do = tl.load(Db + t * Vdim + off_v, mask=v_mask, other=0.0).to(tl.float32)
        d = tl.load(Gb + t).to(tl.float32)
        bb = tl.load(Bb + t).to(tl.float32)
        cc = tl.load(Cb + t).to(tl.float32)
        A = Sp * d
        ae = tl.sum(A * ek[:, None], axis=0)
        Bmat = A - cc * ek[:, None] * ae[None, :]
        pred = tl.sum(Bmat * kk[:, None], axis=0)
        delta = vv - pred
        qs = qk * scale
        G = qs[:, None] * do[None, :] + cur
        s_vec = tl.sum(G * kk[:, None], axis=0)
        Gd = tl.sum(G * delta[None, :], axis=1)
        Bs = tl.sum(Bmat * s_vec[None, :], axis=1)
        dv = bb * s_vec
        tl.store(DVb + t * Vdim + off_v, dv, mask=v_mask)
        db = tl.sum(s_vec * delta, axis=0)
        dk = bb * (Gd - Bs)
        dq = scale * tl.sum(S_t * do[None, :], axis=1)
        tl.store(dq_base + t * BLOCK_K + off_k, dq)
        tl.store(dk_base + t * BLOCK_K + off_k, dk)
        GB = G - bb * kk[:, None] * s_vec[None, :]
        ge = tl.sum(GB * ek[:, None], axis=0)
        dc = -tl.sum(ge * ae, axis=0)
        GBae = tl.sum(GB * ae[None, :], axis=1)
        Age = tl.sum(A * ge[None, :], axis=1)
        de = -cc * (GBae + Age)
        tl.store(de_base + t * BLOCK_K + off_k, de)
        tl.store(db_base + t, db)
        tl.store(dc_base + t, dc)
        GA = GB - cc * ek[:, None] * ge[None, :]
        tmp = tl.sum(GA * A, axis=0)
        dg = tl.sum(tmp, axis=0)
        tl.store(dg_base + t, dg)
        cur = d * GA
    tl.store(carry_ptrs, cur)


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale_f):
        B, H, T, Kdim = q.shape
        Vdim = v.shape[-1]
        BH = B * H
        BLOCK_K, BLOCK_V, NV = _choose_blocks(Kdim, Vdim)
        if BH == 0 or T == 0 or Vdim == 0 or Kdim == 0:
            out = torch.empty(B, H, T, Vdim, dtype=torch.float32, device=q.device)
            ctx.meta = (B, H, T, Kdim, Vdim, BH, BLOCK_K, BLOCK_V, NV, 16, 0, scale_f,
                        q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype, q.device)
            ctx.save_for_backward(q, k, v, e, g, beta, gamma, torch.empty(0, dtype=torch.float32, device=q.device))
            return out
        denom = BH * NV * BLOCK_K * BLOCK_V
        CHUNK = 8000000 // denom if denom > 0 else 64
        if CHUNK < 16:
            CHUNK = 16
        if CHUNK > 128:
            CHUNK = 128
        if CHUNK > T:
            CHUNK = T
        Nc = (T + CHUNK - 1) // CHUNK
        out = torch.empty(B, H, T, Vdim, dtype=torch.float32, device=q.device)
        qc = q.contiguous()
        kc = k.contiguous()
        vc = v.contiguous()
        ec = e.contiguous()
        gc = g.contiguous().to(torch.float32)
        bc = beta.contiguous().to(torch.float32)
        cc = gamma.contiguous().to(torch.float32)
        gexp = torch.exp(gc)
        ckpt = torch.empty(BH * NV * Nc * BLOCK_K * BLOCK_V, dtype=torch.float32, device=q.device)
        grid = (BH, NV)
        nw = 4 if (BLOCK_K * BLOCK_V <= 2048) else 8
        EVEN_K = (Kdim == BLOCK_K)
        EVEN_V = (Vdim % BLOCK_V == 0) and (BLOCK_V <= Vdim or Vdim % BLOCK_V == 0)
        if Vdim <= BLOCK_V and _npo2(max(1, Vdim)) == BLOCK_V and Vdim != BLOCK_V:
            EVEN_V = False
        if NV == 1 and Vdim == BLOCK_V:
            EVEN_V = True
        if NV == 1 and Vdim < BLOCK_V:
            EVEN_V = False
        _fwd_train_kernel[grid](
            qc, kc, vc, ec, gexp, bc, cc, out, ckpt,
            scale_f, T, Kdim, Vdim, Nc, NV,
            BLOCK_K, BLOCK_V, EVEN_K, EVEN_V, CHUNK, num_warps=nw,
        )
        ctx.meta = (B, H, T, Kdim, Vdim, BH, BLOCK_K, BLOCK_V, NV, CHUNK, Nc, scale_f,
                    q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype, q.device)
        ctx.save_for_backward(qc, kc, vc, ec, gexp, bc, cc, ckpt)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, e, gexp, beta, gamma, ckpt = ctx.saved_tensors
        (B, H, T, Kdim, Vdim, BH, BLOCK_K, BLOCK_V, NV, CHUNK, Nc, scale_f,
         qd, kd, vd, ed, gd, betad, gammad, dev) = ctx.meta
        needs = ctx.needs_input_grad
        go = grad_output.contiguous().to(torch.float32)
        dq_full = torch.empty(B, H, T, Kdim, dtype=torch.float32, device=dev)
        dk_full = torch.empty(B, H, T, Kdim, dtype=torch.float32, device=dev)
        de_full = torch.empty(B, H, T, Kdim, dtype=torch.float32, device=dev)
        dv_full = torch.empty(B, H, T, Vdim, dtype=torch.float32, device=dev)
        dg_full = torch.empty(B, H, T, dtype=torch.float32, device=dev)
        db_full = torch.empty(B, H, T, dtype=torch.float32, device=dev)
        dc_full = torch.empty(B, H, T, dtype=torch.float32, device=dev)
        if BH == 0 or T == 0 or Nc == 0:
            res = []
            for need, full, dt in [(needs[0], dq_full, qd), (needs[1], dk_full, kd),
                                   (needs[2], dv_full, vd), (needs[3], de_full, ed),
                                   (needs[4], dg_full, gd), (needs[5], db_full, betad),
                                   (needs[6], dc_full, gammad)]:
                res.append(full.to(dt) if need else None)
            res.append(None)
            return tuple(res)
        Cmax = CHUNK
        EVEN_K = (Kdim == BLOCK_K)
        EVEN_V = True
        if NV == 1:
            if Vdim == BLOCK_V:
                EVEN_V = True
            elif Vdim < BLOCK_V:
                EVEN_V = False
            else:
                EVEN_V = (Vdim % BLOCK_V == 0)
        else:
            EVEN_V = False
        if NV == 1:
            carry = torch.zeros(BH * BLOCK_K * BLOCK_V, dtype=torch.float32, device=dev)
            scratch = torch.empty(BH * Cmax * BLOCK_K * BLOCK_V, dtype=torch.float32, device=dev)
            grid = (BH,)
            nw = 8 if (BLOCK_K * BLOCK_V > 4096) else 4
            for cidx in range(Nc - 1, -1, -1):
                s = cidx * CHUNK
                C = T - s if (s + CHUNK > T) else CHUNK
                _bwd_direct_kernel[grid](
                    q, k, v, e, gexp, beta, gamma, go,
                    dq_full, dk_full, de_full, dv_full, db_full, dc_full, dg_full,
                    ckpt, scratch, carry,
                    scale_f, T, Kdim, Vdim, s, C, Nc, cidx,
                    BLOCK_K, BLOCK_V, EVEN_K, EVEN_V, Cmax, num_warps=nw,
                )
        else:
            carry = torch.zeros(BH * NV * BLOCK_K * BLOCK_V, dtype=torch.float32, device=dev)
            scratch = torch.empty(BH * NV * Cmax * BLOCK_K * BLOCK_V, dtype=torch.float32, device=dev)
            dq_p = torch.empty(BH * NV * Cmax * BLOCK_K, dtype=torch.float32, device=dev)
            dk_p = torch.empty(BH * NV * Cmax * BLOCK_K, dtype=torch.float32, device=dev)
            de_p = torch.empty(BH * NV * Cmax * BLOCK_K, dtype=torch.float32, device=dev)
            db_p = torch.empty(BH * NV * Cmax, dtype=torch.float32, device=dev)
            dc_p = torch.empty(BH * NV * Cmax, dtype=torch.float32, device=dev)
            dg_p = torch.empty(BH * NV * Cmax, dtype=torch.float32, device=dev)
            grid = (BH, NV)
            dq_p4 = dq_p.view(BH, NV, Cmax, BLOCK_K)
            dk_p4 = dk_p.view(BH, NV, Cmax, BLOCK_K)
            de_p4 = de_p.view(BH, NV, Cmax, BLOCK_K)
            db_p3 = db_p.view(BH, NV, Cmax)
            dc_p3 = dc_p.view(BH, NV, Cmax)
            dg_p3 = dg_p.view(BH, NV, Cmax)
            dq_f3 = dq_full.view(BH, T, Kdim)
            dk_f3 = dk_full.view(BH, T, Kdim)
            de_f3 = de_full.view(BH, T, Kdim)
            dg_f2 = dg_full.view(BH, T)
            db_f2 = db_full.view(BH, T)
            dc_f2 = dc_full.view(BH, T)
            for cidx in range(Nc - 1, -1, -1):
                s = cidx * CHUNK
                C = T - s if (s + CHUNK > T) else CHUNK
                _bwd_chunk_kernel[grid](
                    q, k, v, e, gexp, beta, gamma, go, dv_full,
                    dq_p, dk_p, de_p, db_p, dc_p, dg_p,
                    ckpt, scratch, carry,
                    scale_f, T, Kdim, Vdim, s, C, Cmax, Nc, cidx, NV,
                    BLOCK_K, BLOCK_V, num_warps=8,
                )
                dq_sum = dq_p4[:, :, :C, :Kdim].sum(dim=1)
                dk_sum = dk_p4[:, :, :C, :Kdim].sum(dim=1)
                de_sum = de_p4[:, :, :C, :Kdim].sum(dim=1)
                db_sum = db_p3[:, :, :C].sum(dim=1)
                dc_sum = dc_p3[:, :, :C].sum(dim=1)
                dg_sum = dg_p3[:, :, :C].sum(dim=1)
                dq_f3[:, s:s + C, :].copy_(dq_sum)
                dk_f3[:, s:s + C, :].copy_(dk_sum)
                de_f3[:, s:s + C, :].copy_(de_sum)
                db_f2[:, s:s + C].copy_(db_sum)
                dc_f2[:, s:s + C].copy_(dc_sum)
                dg_f2[:, s:s + C].copy_(dg_sum)
        res = []
        for need, full, dt in [(needs[0], dq_full, qd), (needs[1], dk_full, kd),
                               (needs[2], dv_full, vd), (needs[3], de_full, ed),
                               (needs[4], dg_full, gd), (needs[5], db_full, betad),
                               (needs[6], dc_full, gammad)]:
            if need:
                res.append(full.to(dt) if full.dtype != dt else full)
            else:
                res.append(None)
        res.append(None)
        return tuple(res)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.to(torch.float32).reshape(-1)[0].item()) if scale.numel() > 0 else 1.0
    else:
        scale_f = float(scale)
    return _Func.apply(q, k, v, e, g, beta, gamma, scale_f)
