import torch
import triton
import triton.language as tl


def _np2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


@triton.jit
def _fwd_ckpt_kernel(
    q_ptr, k_ptr, v_ptr, q2_ptr, k2_ptr, dec_ptr, beta_ptr, out_ptr, ckpt_ptr,
    T, D, E, V, BH, NV, C, nC, K, scale,
    BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    d_idx = tl.arange(0, BLOCK_D)
    e_idx = tl.arange(0, BLOCK_E)
    v_idx = vid * BLOCK_V + tl.arange(0, BLOCK_V)
    d_mask = d_idx < D
    e_mask = e_idx < E
    v_mask = v_idx < V
    de_v_mask = tl.expand_dims(tl.expand_dims(d_mask, 1), 2) & tl.expand_dims(tl.expand_dims(e_mask, 0), 2) & tl.expand_dims(tl.expand_dims(v_mask, 0), 0)
    S = tl.zeros((BLOCK_D, BLOCK_E, BLOCK_V), dtype=tl.float32)
    for t in range(T):
        k_d = tl.load(k_ptr + bh * T * D + t * D + d_idx, mask=d_mask, other=0.0).to(tl.float32)
        q_d = tl.load(q_ptr + bh * T * D + t * D + d_idx, mask=d_mask, other=0.0).to(tl.float32)
        k2_e = tl.load(k2_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=0.0).to(tl.float32)
        q2_e = tl.load(q2_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=0.0).to(tl.float32)
        dec_e = tl.load(dec_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=1.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + bh * T + t).to(tl.float32)
        v_v = tl.load(v_ptr + bh * T * V + t * V + v_idx, mask=v_mask, other=0.0).to(tl.float32)
        k_3 = tl.expand_dims(tl.expand_dims(k_d, 1), 2)
        q_3 = tl.expand_dims(tl.expand_dims(q_d, 1), 2)
        k2_3 = tl.expand_dims(tl.expand_dims(k2_e, 0), 2)
        q2_3 = tl.expand_dims(tl.expand_dims(q2_e, 0), 2)
        dec_3 = tl.expand_dims(tl.expand_dims(dec_e, 0), 2)
        Shat = S * dec_3
        tmp_p = Shat * k_3 * k2_3
        pred = tl.sum(tl.sum(tmp_p, 0), 0)
        delta = (v_v - pred) * beta_t
        S = Shat + k_3 * k2_3 * tl.expand_dims(tl.expand_dims(delta, 0), 0)
        tmp_o = S * q_3 * q2_3
        out_v = tl.sum(tl.sum(tmp_o, 0), 0) * scale
        tl.store(out_ptr + bh * T * V + t * V + v_idx, out_v, mask=v_mask)
        do_ckpt = False
        cidx = 0
        if (t + 1) % C == 0:
            do_ckpt = True
            cidx = (t + 1) // C
        elif t == T - 1:
            do_ckpt = True
            cidx = nC
        if do_ckpt:
            d_e = tl.expand_dims(tl.expand_dims(d_idx, 1), 2) * E + tl.expand_dims(tl.expand_dims(e_idx, 0), 2)
            vv_e = tl.expand_dims(tl.expand_dims(v_idx, 0), 0)
            offs = (d_e * V + vv_e)
            base = (bh * (nC + 1) + cidx) * K * V
            tl.store(ckpt_ptr + base + offs, S, mask=de_v_mask)


@triton.jit
def _fwd_recomp_kernel(
    k_ptr, k2_ptr, v_ptr, dec_ptr, beta_ptr, ckpt_ptr, scratch_ptr,
    T, D, E, V, BH, NV, nC, K, Cmax, t0, cur_C, chunk_idx,
    BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    d_idx = tl.arange(0, BLOCK_D)
    e_idx = tl.arange(0, BLOCK_E)
    v_idx = vid * BLOCK_V + tl.arange(0, BLOCK_V)
    d_mask = d_idx < D
    e_mask = e_idx < E
    v_mask = v_idx < V
    de_v_mask = tl.expand_dims(tl.expand_dims(d_mask, 1), 2) & tl.expand_dims(tl.expand_dims(e_mask, 0), 2) & tl.expand_dims(tl.expand_dims(v_mask, 0), 0)
    d_e = tl.expand_dims(tl.expand_dims(d_idx, 1), 2) * E + tl.expand_dims(tl.expand_dims(e_idx, 0), 2)
    vv_e = tl.expand_dims(tl.expand_dims(v_idx, 0), 0)
    offs_kv = (d_e * V + vv_e)
    sbase = (bh * (nC + 1) + chunk_idx) * K * V
    S = tl.load(ckpt_ptr + sbase + offs_kv, mask=de_v_mask, other=0.0).to(tl.float32)
    for ii in range(cur_C):
        t = t0 + ii
        k_d = tl.load(k_ptr + bh * T * D + t * D + d_idx, mask=d_mask, other=0.0).to(tl.float32)
        k2_e = tl.load(k2_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=0.0).to(tl.float32)
        dec_e = tl.load(dec_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=1.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + bh * T + t).to(tl.float32)
        v_v = tl.load(v_ptr + bh * T * V + t * V + v_idx, mask=v_mask, other=0.0).to(tl.float32)
        k_3 = tl.expand_dims(tl.expand_dims(k_d, 1), 2)
        k2_3 = tl.expand_dims(tl.expand_dims(k2_e, 0), 2)
        dec_3 = tl.expand_dims(tl.expand_dims(dec_e, 0), 2)
        Shat = S * dec_3
        tmp_p = Shat * k_3 * k2_3
        pred = tl.sum(tl.sum(tmp_p, 0), 0)
        delta = (v_v - pred) * beta_t
        S = Shat + k_3 * k2_3 * tl.expand_dims(tl.expand_dims(delta, 0), 0)
        sbase2 = (bh * Cmax + ii) * K * V
        tl.store(scratch_ptr + sbase2 + offs_kv, S, mask=de_v_mask)


@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, q2_ptr, k2_ptr, dec_ptr, beta_ptr, dO_ptr, scratch_ptr, ckpt_ptr, carry_ptr,
    dq_ptr, dk_ptr, dv_ptr, dq2_ptr, dk2_ptr, dg_ptr, dbeta_ptr,
    T, D, E, V, BH, NV, nC, K, Cmax, t0, cur_C, chunk_idx, scale,
    BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    d_idx = tl.arange(0, BLOCK_D)
    e_idx = tl.arange(0, BLOCK_E)
    v_idx = vid * BLOCK_V + tl.arange(0, BLOCK_V)
    d_mask = d_idx < D
    e_mask = e_idx < E
    v_mask = v_idx < V
    de_mask = tl.expand_dims(d_mask, 1) & tl.expand_dims(e_mask, 0)
    de_v_mask = tl.expand_dims(tl.expand_dims(d_mask, 1), 2) & tl.expand_dims(tl.expand_dims(e_mask, 0), 2) & tl.expand_dims(tl.expand_dims(v_mask, 0), 0)
    d_e = tl.expand_dims(tl.expand_dims(d_idx, 1), 2) * E + tl.expand_dims(tl.expand_dims(e_idx, 0), 2)
    vv_e = tl.expand_dims(tl.expand_dims(v_idx, 0), 0)
    offs_kv = (d_e * V + vv_e)
    cbase = bh * K * V
    carry = tl.load(carry_ptr + cbase + offs_kv, mask=de_v_mask, other=0.0).to(tl.float32)
    sbase0 = (bh * (nC + 1) + chunk_idx) * K * V
    S_start = tl.load(ckpt_ptr + sbase0 + offs_kv, mask=de_v_mask, other=0.0).to(tl.float32)
    for ii in range(cur_C - 1):
        t_local = cur_C - 1 - ii
        t = t0 + t_local
        k_d = tl.load(k_ptr + bh * T * D + t * D + d_idx, mask=d_mask, other=0.0).to(tl.float32)
        q_d = tl.load(q_ptr + bh * T * D + t * D + d_idx, mask=d_mask, other=0.0).to(tl.float32)
        k2_e = tl.load(k2_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=0.0).to(tl.float32)
        q2_e = tl.load(q2_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=0.0).to(tl.float32)
        dec_e = tl.load(dec_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=1.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + bh * T + t).to(tl.float32)
        v_v = tl.load(v_ptr + bh * T * V + t * V + v_idx, mask=v_mask, other=0.0).to(tl.float32)
        dO_v = tl.load(dO_ptr + bh * T * V + t * V + v_idx, mask=v_mask, other=0.0).to(tl.float32)
        St = tl.load(scratch_ptr + (bh * Cmax + t_local) * K * V + offs_kv, mask=de_v_mask, other=0.0).to(tl.float32)
        Sp = tl.load(scratch_ptr + (bh * Cmax + t_local - 1) * K * V + offs_kv, mask=de_v_mask, other=0.0).to(tl.float32)
        k_3 = tl.expand_dims(tl.expand_dims(k_d, 1), 2)
        q_3 = tl.expand_dims(tl.expand_dims(q_d, 1), 2)
        k2_3 = tl.expand_dims(tl.expand_dims(k2_e, 0), 2)
        q2_3 = tl.expand_dims(tl.expand_dims(q2_e, 0), 2)
        dec_3 = tl.expand_dims(tl.expand_dims(dec_e, 0), 2)
        kap = k_3 * k2_3
        qq = q_3 * q2_3
        dO_3 = tl.expand_dims(tl.expand_dims(dO_v, 0), 0)
        direct = qq * dO_3 * scale
        DS = direct + carry
        Shat = Sp * dec_3
        pred = tl.sum(tl.sum(Shat * kap, 0), 0)
        vmp = v_v - pred
        a = tl.sum(tl.sum(DS * kap, 0), 0)
        ba = beta_t * a
        ba_3 = tl.expand_dims(tl.expand_dims(ba, 0), 0)
        vmp_3 = tl.expand_dims(tl.expand_dims(vmp, 0), 0)
        a_3 = tl.expand_dims(tl.expand_dims(a, 0), 0)
        Emat = DS - kap * ba_3
        tl.store(dv_ptr + bh * T * V + t * V + v_idx, ba, mask=v_mask)
        db_p = tl.sum(a * vmp)
        tl.atomic_add(dbeta_ptr + bh * T + t, db_p)
        dot1 = tl.sum(DS * vmp_3, 2)
        dot2 = tl.sum(Shat * a_3, 2)
        dkap = (dot1 - dot2) * beta_t
        dqq = tl.sum(St * dO_3, 2) * scale
        dDmat = tl.sum(Emat * Sp, 2)
        dk_p = tl.sum(dkap * tl.expand_dims(e_mask, 0) * tl.expand_dims(k2_e, 0) if False else dkap * tl.expand_dims(k2_e, 0), 1)
        tl.atomic_add(dk_ptr + bh * T * D + t * D + d_idx, dk_p, mask=d_mask)
        dk2_p = tl.sum(dkap * tl.expand_dims(k_d, 1), 0)
        tl.atomic_add(dk2_ptr + bh * T * E + t * E + e_idx, dk2_p, mask=e_mask)
        dq_p = tl.sum(dqq * tl.expand_dims(q2_e, 0), 1)
        tl.atomic_add(dq_ptr + bh * T * D + t * D + d_idx, dq_p, mask=d_mask)
        dq2_p = tl.sum(dqq * tl.expand_dims(q_d, 1), 0)
        tl.atomic_add(dq2_ptr + bh * T * E + t * E + e_idx, dq2_p, mask=e_mask)
        dg_p = tl.sum(dDmat, 0) * dec_e
        tl.atomic_add(dg_ptr + bh * T * E + t * E + e_idx, dg_p, mask=e_mask)
        carry = Emat * dec_3
    # tail t_local = 0
    t_local = 0
    t = t0
    k_d = tl.load(k_ptr + bh * T * D + t * D + d_idx, mask=d_mask, other=0.0).to(tl.float32)
    q_d = tl.load(q_ptr + bh * T * D + t * D + d_idx, mask=d_mask, other=0.0).to(tl.float32)
    k2_e = tl.load(k2_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=0.0).to(tl.float32)
    q2_e = tl.load(q2_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=0.0).to(tl.float32)
    dec_e = tl.load(dec_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=1.0).to(tl.float32)
    beta_t = tl.load(beta_ptr + bh * T + t).to(tl.float32)
    v_v = tl.load(v_ptr + bh * T * V + t * V + v_idx, mask=v_mask, other=0.0).to(tl.float32)
    dO_v = tl.load(dO_ptr + bh * T * V + t * V + v_idx, mask=v_mask, other=0.0).to(tl.float32)
    St = tl.load(scratch_ptr + (bh * Cmax + 0) * K * V + offs_kv, mask=de_v_mask, other=0.0).to(tl.float32)
    Sp = S_start
    k_3 = tl.expand_dims(tl.expand_dims(k_d, 1), 2)
    q_3 = tl.expand_dims(tl.expand_dims(q_d, 1), 2)
    k2_3 = tl.expand_dims(tl.expand_dims(k2_e, 0), 2)
    q2_3 = tl.expand_dims(tl.expand_dims(q2_e, 0), 2)
    dec_3 = tl.expand_dims(tl.expand_dims(dec_e, 0), 2)
    kap = k_3 * k2_3
    qq = q_3 * q2_3
    dO_3 = tl.expand_dims(tl.expand_dims(dO_v, 0), 0)
    direct = qq * dO_3 * scale
    DS = direct + carry
    Shat = Sp * dec_3
    pred = tl.sum(tl.sum(Shat * kap, 0), 0)
    vmp = v_v - pred
    a = tl.sum(tl.sum(DS * kap, 0), 0)
    ba = beta_t * a
    ba_3 = tl.expand_dims(tl.expand_dims(ba, 0), 0)
    vmp_3 = tl.expand_dims(tl.expand_dims(vmp, 0), 0)
    a_3 = tl.expand_dims(tl.expand_dims(a, 0), 0)
    Emat = DS - kap * ba_3
    tl.store(dv_ptr + bh * T * V + t * V + v_idx, ba, mask=v_mask)
    db_p = tl.sum(a * vmp)
    tl.atomic_add(dbeta_ptr + bh * T + t, db_p)
    dot1 = tl.sum(DS * vmp_3, 2)
    dot2 = tl.sum(Shat * a_3, 2)
    dkap = (dot1 - dot2) * beta_t
    dqq = tl.sum(St * dO_3, 2) * scale
    dDmat = tl.sum(Emat * Sp, 2)
    dk_p = tl.sum(dkap * tl.expand_dims(k2_e, 0), 1)
    tl.atomic_add(dk_ptr + bh * T * D + t * D + d_idx, dk_p, mask=d_mask)
    dk2_p = tl.sum(dkap * tl.expand_dims(k_d, 1), 0)
    tl.atomic_add(dk2_ptr + bh * T * E + t * E + e_idx, dk2_p, mask=e_mask)
    dq_p = tl.sum(dqq * tl.expand_dims(q2_e, 0), 1)
    tl.atomic_add(dq_ptr + bh * T * D + t * D + d_idx, dq_p, mask=d_mask)
    dq2_p = tl.sum(dqq * tl.expand_dims(q_d, 1), 0)
    tl.atomic_add(dq2_ptr + bh * T * E + t * E + e_idx, dq2_p, mask=e_mask)
    dg_p = tl.sum(dDmat, 0) * dec_e
    tl.atomic_add(dg_ptr + bh * T * E + t * E + e_idx, dg_p, mask=e_mask)
    carry = Emat * dec_3
    tl.store(carry_ptr + cbase + offs_kv, carry, mask=de_v_mask)


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q2, k2, g, beta, scale):
        B, H, T, D = q.shape
        E = q2.shape[-1]
        Vd = v.shape[-1]
        BH = B * H
        K = D * E
        q3 = q.contiguous().view(BH, T, D)
        k3 = k.contiguous().view(BH, T, D)
        v3 = v.contiguous().view(BH, T, Vd)
        q23 = q2.contiguous().view(BH, T, E)
        k23 = k2.contiguous().view(BH, T, E)
        dec = torch.exp(g.to(torch.float32)).contiguous().view(BH, T, E)
        b3 = beta.contiguous().view(BH, T)
        out3 = torch.empty((BH, T, Vd), device=q.device, dtype=torch.float32)
        scale_f = float(scale.item()) if torch.is_tensor(scale) else float(scale)
        C = 64
        if T <= 32:
            C = T
        nC = (T + C - 1) // C
        ckpt = torch.zeros((BH * (nC + 1) * K * Vd), device=q.device, dtype=torch.float32)
        BLOCK_D = _np2(D)
        BLOCK_E = _np2(E)
        BLOCK_V = 16
        NV = (Vd + BLOCK_V - 1) // BLOCK_V
        grid = (BH * NV,)
        _fwd_ckpt_kernel[grid](
            q3, k3, v3, q23, k23, dec, b3, out3, ckpt,
            T, D, E, Vd, BH, NV, C, nC, K, scale_f,
            BLOCK_D=BLOCK_D, BLOCK_E=BLOCK_E, BLOCK_V=BLOCK_V,
            num_warps=4, num_stages=1,
        )
        ctx.save_for_backward(q3, k3, v3, q23, k23, dec, b3, ckpt)
        ctx.B, ctx.H, ctx.T, ctx.D, ctx.E, ctx.Vd, ctx.BH, ctx.K = B, H, T, D, E, Vd, BH, K
        ctx.C, ctx.nC, ctx.scale_f = C, nC, scale_f
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, q2.dtype, k2.dtype, g.dtype, beta.dtype)
        return out3.view(B, H, T, Vd)

    @staticmethod
    def backward(ctx, grad_out):
        q3, k3, v3, q23, k23, dec, b3, ckpt = ctx.saved_tensors
        B, H, T, D, E, Vd, BH, K = ctx.B, ctx.H, ctx.T, ctx.D, ctx.E, ctx.Vd, ctx.BH, ctx.K
        C, nC, scale_f = ctx.C, ctx.nC, ctx.scale_f
        g_out = grad_out.contiguous().view(BH, T, Vd).to(torch.float32)
        dq = torch.zeros((BH, T, D), device=q3.device, dtype=torch.float32)
        dk = torch.zeros((BH, T, D), device=q3.device, dtype=torch.float32)
        dv = torch.zeros((BH, T, Vd), device=q3.device, dtype=torch.float32)
        dq2 = torch.zeros((BH, T, E), device=q3.device, dtype=torch.float32)
        dk2 = torch.zeros((BH, T, E), device=q3.device, dtype=torch.float32)
        dg = torch.zeros((BH, T, E), device=q3.device, dtype=torch.float32)
        db = torch.zeros((BH, T), device=q3.device, dtype=torch.float32)
        carry = torch.zeros((BH * K * Vd), device=q3.device, dtype=torch.float32)
        scratch = torch.empty((BH * C * K * Vd), device=q3.device, dtype=torch.float32)
        BLOCK_D = _np2(D)
        BLOCK_E = _np2(E)
        BLOCK_V = 16
        NV = (Vd + BLOCK_V - 1) // BLOCK_V
        grid = (BH * NV,)
        for ci in range(nC - 1, -1, -1):
            t0 = ci * C
            cur_C = min(C, T - t0)
            _fwd_recomp_kernel[grid](
                k3, k23, v3, dec, b3, ckpt, scratch,
                T, D, E, Vd, BH, NV, nC, K, C, t0, cur_C, ci,
                BLOCK_D=BLOCK_D, BLOCK_E=BLOCK_E, BLOCK_V=BLOCK_V,
                num_warps=4, num_stages=1,
            )
            _bwd_kernel[grid](
                q3, k3, v3, q23, k23, dec, b3, g_out, scratch, ckpt, carry,
                dq, dk, dv, dq2, dk2, dg, db,
                T, D, E, Vd, BH, NV, nC, K, C, t0, cur_C, ci, scale_f,
                BLOCK_D=BLOCK_D, BLOCK_E=BLOCK_E, BLOCK_V=BLOCK_V,
                num_warps=4, num_stages=1,
            )
        dt = ctx.dtypes
        dq = dq.view(B, H, T, D).to(dt[0])
        dk = dk.view(B, H, T, D).to(dt[1])
        dv = dv.view(B, H, T, Vd).to(dt[2])
        dq2 = dq2.view(B, H, T, E).to(dt[3])
        dk2 = dk2.view(B, H, T, E).to(dt[4])
        dg = dg.view(B, H, T, E).to(dt[5])
        db = db.view(B, H, T).to(dt[6])
        return dq, dk, dv, dq2, dk2, dg, db, None


def kernel_function(q, k, v, q2, k2, g, beta, scale):
    return _Func.apply(q, k, v, q2, k2, g, beta, scale)
