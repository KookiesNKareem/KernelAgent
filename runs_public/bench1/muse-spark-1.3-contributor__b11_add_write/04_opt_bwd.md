```python
import torch
import triton
import triton.language as tl


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


# ---------------- fast path K=V=128, Ct=32 ----------------

@triton.jit
def _fwd_fast_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, b_ptr, c_ptr,
                     out_ptr, ckpt_ptr, d_ptr, qs_ptr,
                     BH, T, Nc, Ct, scale):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    k_offs = tl.arange(0, 128)
    v_offs = pid_v * 64 + tl.arange(0, 64)
    S = tl.zeros((128, 64), dtype=tl.float32)
    for cc in tl.range(Nc):
        for ctt in tl.range(Ct):
            t = cc * 32 + ctt
            base_q = (bh * T + t) * 128
            base_v = (bh * T + t) * 128
            qb = tl.load(q_ptr + base_q + k_offs)
            qf = qb.to(tl.float32) * scale
            tl.store(qs_ptr + base_q + k_offs, qf)
            kf = tl.load(k_ptr + base_q + k_offs).to(tl.float32)
            ef = tl.load(e_ptr + base_q + k_offs).to(tl.float32)
            gf = tl.load(g_ptr + base_q + k_offs)
            df = tl.exp(gf)
            tl.store(d_ptr + base_q + k_offs, df)
            vf = tl.load(v_ptr + base_v + v_offs).to(tl.float32)
            bf = tl.load(b_ptr + bh * T + t)
            cf = tl.load(c_ptr + bh * T + t)
            de = tl.expand_dims(df, 1)
            ke = tl.expand_dims(kf, 1)
            ee = tl.expand_dims(ef, 1)
            qe = tl.expand_dims(qf, 1)
            ve = tl.expand_dims(vf, 0)
            S = S * de
            pred = tl.sum(S * ke, 0)
            diff = (vf - pred) * bf
            S = S + ke * tl.expand_dims(diff, 0) + cf * ee * ve
            outv = tl.sum(S * qe, 0)
            tl.store(out_ptr + base_v + v_offs, outv)
        ckpt_base = (bh * (Nc + 1) + (cc + 1)) * 16384
        off2d = ckpt_base + tl.expand_dims(k_offs, 1) * 128 + tl.expand_dims(v_offs, 0)
        tl.store(ckpt_ptr + off2d, S)


@triton.jit
def _bwd_fast_kernel(qs_ptr, k_ptr, e_ptr, d_ptr, v_ptr, b_ptr, c_ptr, do_ptr,
                     ckpt_ptr, dq_p_ptr, dk_p_ptr, de_p_ptr, dg_p_ptr,
                     dv_ptr, db_p_ptr, dc_p_ptr,
                     BH, T, Nc, Ct, scale):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    k_offs = tl.arange(0, 128)
    v_offs = pid_v * 64 + tl.arange(0, 64)
    DS = tl.zeros((128, 64), dtype=tl.float32)
    k_next = tl.zeros((128,), dtype=tl.float32)
    d_next = tl.zeros((128,), dtype=tl.float32)
    b_next = 0.0
    for c_rev in tl.range(Nc):
        c = Nc - 1 - c_rev
        t0 = c * 32
        for idx in tl.range(Ct):
            idx_rev = 31 - idx
            t = t0 + idx_rev
            base_q = (bh * T + t) * 128
            base_v = (bh * T + t) * 128
            qs_t = tl.load(qs_ptr + base_q + k_offs)
            do_t = tl.load(do_ptr + base_v + v_offs)
            k_t = tl.load(k_ptr + base_q + k_offs).to(tl.float32)
            e_t = tl.load(e_ptr + base_q + k_offs).to(tl.float32)
            d_t = tl.load(d_ptr + base_q + k_offs)
            v_t = tl.load(v_ptr + base_v + v_offs).to(tl.float32)
            b_t = tl.load(b_ptr + bh * T + t)
            c_t = tl.load(c_ptr + bh * T + t)
            qdo = tl.expand_dims(qs_t, 1) * tl.expand_dims(do_t, 0)
            s_next = tl.sum(tl.expand_dims(k_next, 1) * DS, 0)
            DS_scaled = DS * tl.expand_dims(d_next, 1)
            dk_n = d_next * k_next * b_next
            corr = tl.expand_dims(dk_n, 1) * tl.expand_dims(s_next, 0)
            DS_cur = qdo + DS_scaled - corr
            ckpt_base_s = (bh * (Nc + 1) + c) * 16384
            off2d_s = ckpt_base_s + tl.expand_dims(k_offs, 1) * 128 + tl.expand_dims(v_offs, 0)
            S_tmp = tl.load(ckpt_ptr + off2d_s)
            for f in tl.range(idx_rev):
                tf = t0 + f
                bqf = (bh * T + tf) * 128
                bvf = (bh * T + tf) * 128
                k_f = tl.load(k_ptr + bqf + k_offs).to(tl.float32)
                e_f = tl.load(e_ptr + bqf + k_offs).to(tl.float32)
                d_f = tl.load(d_ptr + bqf + k_offs)
                v_f = tl.load(v_ptr + bvf + v_offs).to(tl.float32)
                b_f = tl.load(b_ptr + bh * T + tf)
                c_f = tl.load(c_ptr + bh * T + tf)
                S_tmp = S_tmp * tl.expand_dims(d_f, 1)
                pred_f = tl.sum(S_tmp * tl.expand_dims(k_f, 1), 0)
                diff_f = (v_f - pred_f) * b_f
                S_tmp = S_tmp + tl.expand_dims(k_f, 1) * tl.expand_dims(diff_f, 0) + c_f * tl.expand_dims(e_f, 1) * tl.expand_dims(v_f, 0)
            s_cur = tl.sum(tl.expand_dims(k_t, 1) * DS_cur, 0)
            kd = k_t * d_t
            rvec = tl.sum(tl.expand_dims(kd, 1) * S_tmp, 0)
            vmr = v_t - rvec
            wvec = b_t * k_t + c_t * e_t
            dv = tl.sum(tl.expand_dims(wvec, 1) * DS_cur, 0)
            tl.store(dv_ptr + base_v + v_offs, dv.to(tl.bfloat16))
            S_do = tl.sum(S_tmp * tl.expand_dims(do_t, 0), 1)
            v_do = tl.sum(v_t * do_t, 0)
            vmr_do = tl.sum(vmr * do_t, 0)
            diff_do = b_t * vmr_do
            dq_part = scale * (d_t * S_do + k_t * diff_do + c_t * e_t * v_do)
            base_p = ((bh * T + t) * 2 + pid_v) * 128
            tl.store(dq_p_ptr + base_p + k_offs, dq_part)
            tmpv = tl.sum(DS_cur * tl.expand_dims(v_t, 0), 1)
            de_part = c_t * tmpv
            tl.store(de_p_ptr + base_p + k_offs, de_part)
            dc_part = tl.sum(e_t * tmpv, 0)
            tl.store(dc_p_ptr + (bh * T + t) * 2 + pid_v, dc_part)
            a_part = tl.sum(S_tmp * DS_cur, 1)
            b2_part = tl.sum(S_tmp * tl.expand_dims(s_cur, 0), 1)
            dg_part = d_t * (a_part - b_t * k_t * b2_part)
            tl.store(dg_p_ptr + base_p + k_offs, dg_part)
            c1_part = tl.sum(DS_cur * tl.expand_dims(vmr, 0), 1)
            c2_part = d_t * b2_part
            dk_part = b_t * (c1_part - c2_part)
            tl.store(dk_p_ptr + base_p + k_offs, dk_part)
            db_part = tl.sum(s_cur * vmr, 0)
            tl.store(db_p_ptr + (bh * T + t) * 2 + pid_v, db_part)
            DS = DS_cur
            k_next = k_t
            d_next = d_t
            b_next = b_t


@triton.jit
def _reduce_fast_kernel(dq_p_ptr, dk_p_ptr, de_p_ptr, dg_p_ptr, db_p_ptr, dc_p_ptr,
                        dq_o_ptr, dk_o_ptr, de_o_ptr, dg_o_ptr, db_o_ptr, dc_o_ptr,
                        BHT):
    bt = tl.program_id(0)
    k_offs = tl.arange(0, 128)
    base = bt * 256
    dq0 = tl.load(dq_p_ptr + base + k_offs)
    dq1 = tl.load(dq_p_ptr + base + 128 + k_offs)
    tl.store(dq_o_ptr + bt * 128 + k_offs, (dq0 + dq1).to(tl.bfloat16))
    dk0 = tl.load(dk_p_ptr + base + k_offs)
    dk1 = tl.load(dk_p_ptr + base + 128 + k_offs)
    tl.store(dk_o_ptr + bt * 128 + k_offs, (dk0 + dk1).to(tl.bfloat16))
    de0 = tl.load(de_p_ptr + base + k_offs)
    de1 = tl.load(de_p_ptr + base + 128 + k_offs)
    tl.store(de_o_ptr + bt * 128 + k_offs, (de0 + de1).to(tl.bfloat16))
    dg0 = tl.load(dg_p_ptr + base + k_offs)
    dg1 = tl.load(dg_p_ptr + base + 128 + k_offs)
    tl.store(dg_o_ptr + bt * 128 + k_offs, dg0 + dg1)
    db0 = tl.load(db_p_ptr + bt * 2)
    db1 = tl.load(db_p_ptr + bt * 2 + 1)
    tl.store(db_o_ptr + bt, db0 + db1)
    dc0 = tl.load(dc_p_ptr + bt * 2)
    dc1 = tl.load(dc_p_ptr + bt * 2 + 1)
    tl.store(dc_o_ptr + bt, dc0 + dc1)


class _RecFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        device = q.device
        scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        Ct = 32
        Nc = T // Ct
        BH = B * H
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        e_c = e.contiguous()
        g_c = g.contiguous()
        b_c = beta.contiguous()
        c_c = gamma.contiguous()
        out = torch.empty((B, H, T, V), dtype=torch.float32, device=device)
        ckpt = torch.zeros((BH * (Nc + 1), K, V), dtype=torch.float32, device=device)
        d_mat = torch.empty((B, H, T, K), dtype=torch.float32, device=device)
        qs_mat = torch.empty((B, H, T, K), dtype=torch.float32, device=device)
        _fwd_fast_kernel[(BH, 2)](q_c, k_c, v_c, e_c, g_c, b_c, c_c,
                                  out, ckpt, d_mat, qs_mat,
                                  BH, T, Nc, Ct, scale_f, num_warps=8)
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V, ctx.BH, ctx.Nc, ctx.Ct = B, H, T, K, V, BH, Nc, Ct
        ctx.scale_f = scale_f
        ctx.qd, ctx.kd, ctx.vd, ctx.ed = q.dtype, k.dtype, v.dtype, e.dtype
        ctx.gd, ctx.bd, ctx.cd = g.dtype, beta.dtype, gamma.dtype
        ctx.k_c, ctx.e_c, ctx.v_c = k_c, e_c, v_c
        ctx.b_c, ctx.c_c = b_c, c_c
        ctx.d_mat, ctx.qs_mat, ctx.ckpt = d_mat, qs_mat, ckpt
        return out

    @staticmethod
    def backward(ctx, grad_out):
        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        BH, Nc, Ct = ctx.BH, ctx.Nc, ctx.Ct
        device = grad_out.device
        do_c = grad_out.contiguous().to(torch.float32)
        N2 = BH * T * 2
        dq_p = torch.empty((N2, K), dtype=torch.float32, device=device)
        dk_p = torch.empty((N2, K), dtype=torch.float32, device=device)
        de_p = torch.empty((N2, K), dtype=torch.float32, device=device)
        dg_p = torch.empty((N2, K), dtype=torch.float32, device=device)
        db_p = torch.empty((N2,), dtype=torch.float32, device=device)
        dc_p = torch.empty((N2,), dtype=torch.float32, device=device)
        dv = torch.empty((B, H, T, V), dtype=torch.bfloat16, device=device)
        _bwd_fast_kernel[(BH, 2)](ctx.qs_mat, ctx.k_c, ctx.e_c, ctx.d_mat, ctx.v_c,
                                   ctx.b_c, ctx.c_c, do_c, ctx.ckpt,
                                   dq_p, dk_p, de_p, dg_p, dv, db_p, dc_p,
                                   BH, T, Nc, Ct, ctx.scale_f, num_warps=8)
        dq_f = torch.empty((B, H, T, K), dtype=ctx.qd, device=device)
        dk_f = torch.empty((B, H, T, K), dtype=ctx.kd, device=device)
        de_f = torch.empty((B, H, T, K), dtype=ctx.ed, device=device)
        dg_f = torch.empty((B, H, T, K), dtype=ctx.gd, device=device)
        db_f = torch.empty((B, H, T), dtype=ctx.bd, device=device)
        dc_f = torch.empty((B, H, T), dtype=ctx.cd, device=device)
        _reduce_fast_kernel[(BH * T,)](dq_p, dk_p, de_p, dg_p, db_p, dc_p,
                                           dq_f, dk_f, de_f, dg_f, db_f, dc_f,
                                           BH * T, num_warps=4)
        gq = dq_f if ctx.needs_input_grad[0] else None
        gk = dk_f if ctx.needs_input_grad[1] else None
        gv = dv.to(ctx.vd) if ctx.needs_input_grad[2] else None
        if gv is not None and gv.dtype != ctx.vd:
            gv = gv.to(ctx.vd)
        # dv already bfloat16; convert if needed
        if ctx.needs_input_grad[2]:
            if dv.dtype != ctx.vd:
                gv = dv.to(ctx.vd)
            else:
                gv = dv
        else:
            gv = None
        ge = de_f if ctx.needs_input_grad[3] else None
        gg = dg_f if ctx.needs_input_grad[4] else None
        gb = db_f if ctx.needs_input_grad[5] else None
        gc = dc_f if ctx.needs_input_grad[6] else None
        return (gq, gk, gv, ge, gg, gb, gc, None)


# ---------------- generic fallback (correct for any shape) ----------------

@triton.jit
def _g_fwd_ckpt_kernel(qs_ptr, k_ptr, v_ptr, e_ptr, d_ptr, b_ptr, c_ptr, out_ptr, ckpt_ptr,
                       BH, T, K, V, Ct, Nc,
                       BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = k_offs < K
    mask_v = v_offs < V
    k_col = tl.expand_dims(k_offs, 1)
    v_row = tl.expand_dims(v_offs, 0)
    mask_2d = tl.expand_dims(mask_k, 1) & tl.expand_dims(mask_v, 0)
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    for cc in tl.range(Nc):
        for ctt in tl.range(Ct):
            t = cc * Ct + ctt
            if t < T:
                qv = tl.load(qs_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0)
                kv = tl.load(k_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
                ev = tl.load(e_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
                dvv = tl.load(d_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0)
                vv = tl.load(v_ptr + (bh * T + t) * V + v_offs, mask=mask_v, other=0.0).to(tl.float32)
                bv = tl.load(b_ptr + bh * T + t)
                cv = tl.load(c_ptr + bh * T + t)
                S = S * tl.expand_dims(dvv, 1)
                pred = tl.sum(S * tl.expand_dims(kv, 1), 0)
                diff = (vv - pred) * bv
                S = S + tl.expand_dims(kv, 1) * tl.expand_dims(diff, 0) + cv * tl.expand_dims(ev, 1) * tl.expand_dims(vv, 0)
                outv = tl.sum(S * tl.expand_dims(qv, 1), 0)
                tl.store(out_ptr + (bh * T + t) * V + v_offs, outv, mask=mask_v)
        ckpt_base = (bh * (Nc + 1) + (cc + 1)) * K * V
        tl.store(ckpt_ptr + ckpt_base + k_col * V + v_row, S, mask=mask_2d)


@triton.jit
def _g_bwd_chunk_kernel(qs_ptr, k_ptr, e_ptr, d_ptr, b_ptr, c_ptr, v_ptr, do_ptr,
                        ckpt_ptr, scratch_ptr, carry_ptr,
                        dq_ptr, dk_ptr, de_ptr, dg_ptr, dv_ptr, db_ptr, dc_ptr,
                        BH, T, K, V, Ct, Nc, NV, t0, Ct_len, c_idx, scale,
                        BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = k_offs < K
    mask_v = v_offs < V
    k_col = tl.expand_dims(k_offs, 1)
    v_row = tl.expand_dims(v_offs, 0)
    mask_2d = tl.expand_dims(mask_k, 1) & tl.expand_dims(mask_v, 0)
    ckpt_base_start = (bh * (Nc + 1) + c_idx) * K * V
    s_start_ptrs = ckpt_ptr + ckpt_base_start + k_col * V + v_row
    S = tl.load(s_start_ptrs, mask=mask_2d, other=0.0)
    for ct in tl.range(Ct_len):
        t = t0 + ct
        kv = tl.load(k_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
        ev = tl.load(e_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
        dvv = tl.load(d_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0)
        vv = tl.load(v_ptr + (bh * T + t) * V + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        bv = tl.load(b_ptr + bh * T + t)
        cv = tl.load(c_ptr + bh * T + t)
        S = S * tl.expand_dims(dvv, 1)
        pred = tl.sum(S * tl.expand_dims(kv, 1), 0)
        diff = (vv - pred) * bv
        S = S + tl.expand_dims(kv, 1) * tl.expand_dims(diff, 0) + cv * tl.expand_dims(ev, 1) * tl.expand_dims(vv, 0)
        tl.store(scratch_ptr + (bh * Ct + ct) * K * V + k_col * V + v_row, S, mask=mask_2d)
    carry_ptrs = carry_ptr + bh * K * V + k_col * V + v_row
    DS_next = tl.load(carry_ptrs, mask=mask_2d, other=0.0)
    t1 = t0 + Ct_len
    k_next = tl.zeros((BLOCK_K,), dtype=tl.float32)
    d_next = tl.zeros((BLOCK_K,), dtype=tl.float32)
    b_next = 0.0
    if t1 < T:
        k_next = tl.load(k_ptr + (bh * T + t1) * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
        d_next = tl.load(d_ptr + (bh * T + t1) * K + k_offs, mask=mask_k, other=0.0)
        b_next = tl.load(b_ptr + bh * T + t1)
    for idx in tl.range(Ct_len):
        ct = Ct_len - 1 - idx
        t = t0 + ct
        q_t = tl.load(qs_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0)
        k_t = tl.load(k_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
        e_t = tl.load(e_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
        d_t = tl.load(d_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0)
        v_t = tl.load(v_ptr + (bh * T + t) * V + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        do_t = tl.load(do_ptr + (bh * T + t) * V + v_offs, mask=mask_v, other=0.0)
        b_t = tl.load(b_ptr + bh * T + t)
        c_t = tl.load(c_ptr + bh * T + t)
        S_t = tl.load(scratch_ptr + (bh * Ct + ct) * K * V + k_col * V + v_row, mask=mask_2d, other=0.0)
        if ct == 0:
            S_prev = tl.load(s_start_ptrs, mask=mask_2d, other=0.0)
        else:
            S_prev = tl.load(scratch_ptr + (bh * Ct + ct - 1) * K * V + k_col * V + v_row, mask=mask_2d, other=0.0)
        qdo = tl.expand_dims(q_t, 1) * tl.expand_dims(do_t, 0)
        if t + 1 < T:
            s_nxt = tl.sum(tl.expand_dims(k_next, 1) * DS_next, 0)
            DS_trans = tl.expand_dims(d_next, 1) * DS_next - b_next * tl.expand_dims(d_next, 1) * tl.expand_dims(k_next, 1) * tl.expand_dims(s_nxt, 0)
            DS_cur = qdo + DS_trans
        else:
            DS_cur = qdo
        k_te = tl.expand_dims(k_t, 1)
        s_vec = tl.sum(k_te * DS_cur, 0)
        Zmat = S_prev * tl.expand_dims(d_t, 1)
        r_vec = tl.sum(k_te * Zmat, 0)
        vmr = v_t - r_vec
        wvec = b_t * k_t + c_t * e_t
        dv_vec = tl.sum(DS_cur * tl.expand_dims(wvec, 1), 0)
        tl.store(dv_ptr + (bh * T + t) * V + v_offs, dv_vec, mask=mask_v)
        tmp_k = tl.sum(DS_cur * tl.expand_dims(v_t, 0), 1)
        tl.store(de_ptr + ((bh * NV + pid_v) * T + t) * K + k_offs, c_t * tmp_k, mask=mask_k)
        tl.store(dc_ptr + (bh * NV + pid_v) * T + t, tl.sum(e_t * tmp_k, 0))
        tl.store(db_ptr + (bh * NV + pid_v) * T + t, tl.sum(s_vec * vmr, 0))
        tl.store(dq_ptr + ((bh * NV + pid_v) * T + t) * K + k_offs, scale * tl.sum(S_t * tl.expand_dims(do_t, 0), 1), mask=mask_k)
        a_k = tl.sum(S_prev * DS_cur, 1)
        b2_k = tl.sum(S_prev * tl.expand_dims(s_vec, 0), 1)
        tl.store(dg_ptr + ((bh * NV + pid_v) * T + t) * K + k_offs, d_t * (a_k - b_t * k_t * b2_k), mask=mask_k)
        c1_k = tl.sum(DS_cur * tl.expand_dims(vmr, 0), 1)
        c2_k = tl.sum(Zmat * tl.expand_dims(s_vec, 0), 1)
        tl.store(dk_ptr + ((bh * NV + pid_v) * T + t) * K + k_offs, b_t * (c1_k - c2_k), mask=mask_k)
        DS_next = DS_cur
        k_next = k_t
        d_next = d_t
        b_next = b_t
    tl.store(carry_ptrs, DS_next, mask=mask_2d)


class _RecGeneric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        device = q.device
        scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        if B == 0 or H == 0 or T == 0 or V == 0 or K == 0:
            out = torch.zeros((B, H, T, V), dtype=torch.float32, device=device)
            ctx.empty = True
            ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
            ctx.qd, ctx.kd, ctx.vd, ctx.ed = q.dtype, k.dtype, v.dtype, e.dtype
            ctx.gd, ctx.bd, ctx.cd = g.dtype, beta.dtype, gamma.dtype
            ctx.scale_f = scale_f
            return out
        ctx.empty = False
        with torch.no_grad():
            q_scaled = (q.detach().to(torch.float32) * scale_f).contiguous()
            d_mat = torch.exp(g.detach().to(torch.float32)).contiguous()
            k_c = k.detach().contiguous()
            e_c = e.detach().contiguous()
            v_c = v.detach().contiguous()
            b_c = beta.detach().to(torch.float32).contiguous()
            c_c = gamma.detach().to(torch.float32).contiguous()
        BH = B * H
        Ct = 64 if T > 64 else T
        Nc = (T + Ct - 1) // Ct
        BK = _next_pow2(K)
        if BK < 16:
            BK = 16
        BV = 64 if BK <= 32 else (32 if BK <= 128 else 16)
        NV = (V + BV - 1) // BV
        out = torch.empty((B, H, T, V), dtype=torch.float32, device=device)
        ckpt = torch.zeros((B, H, Nc + 1, K, V), dtype=torch.float32, device=device)
        if BH > 0 and NV > 0 and Nc > 0:
            _g_fwd_ckpt_kernel[(BH, NV)](q_scaled, k_c, v_c, e_c, d_mat, b_c, c_c, out, ckpt,
                                         BH, T, K, V, Ct, Nc, BK, BV, num_warps=4)
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
        ctx.BH, ctx.Ct, ctx.Nc, ctx.BK, ctx.BV, ctx.NV = BH, Ct, Nc, BK, BV, NV
        ctx.scale_f = scale_f
        ctx.qd, ctx.kd, ctx.vd, ctx.ed = q.dtype, k.dtype, v.dtype, e.dtype
        ctx.gd, ctx.bd, ctx.cd = g.dtype, beta.dtype, gamma.dtype
        ctx.q_scaled, ctx.k_c, ctx.e_c, ctx.d_mat = q_scaled, k_c, e_c, d_mat
        ctx.v_c, ctx.b_c, ctx.c_c, ctx.ckpt = v_c, b_c, c_c, ckpt
        return out

    @staticmethod
    def backward(ctx, grad_out):
        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        needs = ctx.needs_input_grad
        if ctx.empty:
            outs = []
            for i, (sh, dt) in enumerate([((B, H, T, K), ctx.qd), ((B, H, T, K), ctx.kd),
                                         ((B, H, T, V), ctx.vd), ((B, H, T, K), ctx.ed),
                                         ((B, H, T, K), ctx.gd), ((B, H, T), ctx.bd),
                                         ((B, H, T), ctx.cd)]):
                outs.append(torch.zeros(sh, dtype=dt, device=grad_out.device) if needs[i] else None)
            outs.append(None)
            return tuple(outs)
        device = grad_out.device
        BH, Ct, Nc, BK, BV, NV = ctx.BH, ctx.Ct, ctx.Nc, ctx.BK, ctx.BV, ctx.NV
        scale_f = ctx.scale_f
        with torch.no_grad():
            do_c = grad_out.detach().to(torch.float32).contiguous()
            dq_p = torch.empty((B, H, NV, T, K), dtype=torch.float32, device=device)
            dk_p = torch.empty((B, H, NV, T, K), dtype=torch.float32, device=device)
            de_p = torch.empty((B, H, NV, T, K), dtype=torch.float32, device=device)
            dg_p = torch.empty((B, H, NV, T, K), dtype=torch.float32, device=device)
            db_p = torch.empty((B, H, NV, T), dtype=torch.float32, device=device)
            dc_p = torch.empty((B, H, NV, T), dtype=torch.float32, device=device)
            dv_full = torch.empty((B, H, T, V), dtype=torch.float32, device=device)
            carry = torch.zeros((B, H, K, V), dtype=torch.float32, device=device)
            scratch = torch.empty((B, H, Ct, K, V), dtype=torch.float32, device=device)
            qs, kk, ee, dd = ctx.q_scaled, ctx.k_c, ctx.e_c, ctx.d_mat
            vv, bb, cc = ctx.v_c, ctx.b_c, ctx.c_c
            ckpt = ctx.ckpt
            for c in range(Nc - 1, -1, -1):
                t0 = c * Ct
                clen = T - t0 if T - t0 < Ct else Ct
                if clen <= 0:
                    continue
                _g_bwd_chunk_kernel[(BH, NV)](qs, kk, ee, dd, bb, cc, vv, do_c,
                                              ckpt, scratch, carry,
                                              dq_p, dk_p, de_p, dg_p, dv_full, db_p, dc_p,
                                              BH, T, K, V, Ct, Nc, NV, t0, clen, c, scale_f,
                                              BK, BV, num_warps=4)
            dq_sum = dq_p.sum(dim=2)
            dk_sum = dk_p.sum(dim=2)
            de_sum = de_p.sum(dim=2)
            dg_sum = dg_p.sum(dim=2)
            db_sum = db_p.sum(dim=2)
            dc_sum = dc_p.sum(dim=2)
        gq = dq_sum.to(ctx.qd) if needs[0] else None
        gk = dk_sum.to(ctx.kd) if needs[1] else None
        gv = dv_full.to(ctx.vd) if needs[2] else None
        ge = de_sum.to(ctx.ed) if needs[3] else None
        gg = dg_sum.to(ctx.gd) if needs[4] else None
        gb = db_sum.to(ctx.bd) if needs[5] else None
        gc = dc_sum.to(ctx.cd) if needs[6] else None
        return (gq, gk, gv, ge, gg, gb, gc, None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    if K == 128 and V == 128 and T % 32 == 0 and T > 0 and B > 0 and H > 0:
        return _RecFast.apply(q, k, v, e, g, beta, gamma, scale)
    return _RecGeneric.apply(q, k, v, e, g, beta, gamma, scale)
```
My forward fuses the recurrence with V-sharding and checkpoints every 32 steps while also materializing exp decay and scaled queries for reuse. My backward runs in a single Triton launch that walks chunks reverse, keeping the DS adjoint in registers and recomputing S on the fly with a nested forward loop bounded by the intra-chunk index, avoiding any global scratch. K-grad partials are reduced across the two V shards with a fused Triton reduction that also casts to input dtypes. I am unsure about Triton 3.5 support for dynamic tl.range bounds, scalar if-free control flow, and 128x64 fp32 live state causing register spilling on L4.