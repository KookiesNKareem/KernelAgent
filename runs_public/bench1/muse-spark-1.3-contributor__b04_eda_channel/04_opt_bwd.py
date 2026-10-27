import torch
import triton
import triton.language as tl


# ---------------- fast 128x128 sharded path ----------------

@triton.jit
def _fwd_shard128(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr, chk_ptr,
    H, scale,
    T: tl.constexpr, Nc: tl.constexpr, C: tl.constexpr,
):
    pid = tl.program_id(0)
    nV: tl.constexpr = 4
    bh = pid // nV
    iv = pid % nV
    b = bh // H
    h = bh % H
    offs_k = tl.arange(0, 128)
    offs_v = iv * 32 + tl.arange(0, 32)
    BH = b * H + h
    q_base = (BH * T) * 128
    v_base = (BH * T) * 128
    o_base = (BH * T) * 128
    beta_base = BH * T
    chk_base = BH * Nc * 128 * 128
    S = tl.zeros([128, 32], dtype=tl.float32)
    for t in tl.range(0, T, num_stages=1):
        q_vec = tl.load(q_ptr + q_base + t * 128 + offs_k).to(tl.float32)
        k_vec = tl.load(k_ptr + q_base + t * 128 + offs_k).to(tl.float32)
        e_vec = tl.load(e_ptr + q_base + t * 128 + offs_k).to(tl.float32)
        g_vec = tl.load(g_ptr + q_base + t * 128 + offs_k).to(tl.float32)
        v_vec = tl.load(v_ptr + v_base + t * 128 + offs_v).to(tl.float32)
        bt = tl.load(beta_ptr + beta_base + t).to(tl.float32)
        ct = tl.load(gamma_ptr + beta_base + t).to(tl.float32)
        d_vec = tl.exp(g_vec)
        S = S * tl.expand_dims(d_vec, 1)
        a_vec = tl.sum(S * tl.expand_dims(e_vec, 1), axis=0)
        S = S - tl.expand_dims(ct * e_vec, 1) * tl.expand_dims(a_vec, 0)
        pred_vec = tl.sum(S * tl.expand_dims(k_vec, 1), axis=0)
        delta_vec = (v_vec - pred_vec) * bt
        S = S + tl.expand_dims(k_vec, 1) * tl.expand_dims(delta_vec, 0)
        out_vec = tl.sum(S * tl.expand_dims(q_vec * scale, 1), axis=0)
        tl.store(o_ptr + o_base + t * 128 + offs_v, out_vec)
        nxt = t + 1
        if nxt % C == 0:
            cc = nxt // C - 1
            if cc < Nc:
                cbase = chk_ptr + chk_base + cc * 128 * 128
                ptrs = cbase + tl.expand_dims(offs_k, 1) * 128 + tl.expand_dims(offs_v, 0)
                tl.store(ptrs, S)


@triton.jit
def _bwd_shard128(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, do_ptr,
    dq_ptr, dk_ptr, de_ptr, dg_ptr, dv_ptr, dbeta_ptr, dgamma_ptr, aw_ptr,
    chk_ptr, tmp_ptr,
    H, scale,
    T: tl.constexpr, Nc: tl.constexpr, C: tl.constexpr,
):
    pid = tl.program_id(0)
    nV: tl.constexpr = 4
    bh = pid // nV
    iv = pid % nV
    b = bh // H
    h = bh % H
    offs_k = tl.arange(0, 128)
    offs_v = iv * 32 + tl.arange(0, 32)
    BH = b * H + h
    q_base = (BH * T) * 128
    v_base = (BH * T) * 128
    beta_base = BH * T
    chk_base = BH * Nc * 128 * 128
    tmp_base = BH * C * 128 * 128
    cur = tl.zeros([128, 32], dtype=tl.float32)
    for ii in tl.range(0, Nc, num_stages=1):
        c = Nc - 1 - ii
        c_start = c * C
        remain = T - c_start
        cur_len = C
        if remain < C:
            cur_len = remain
        # init
        S_init = tl.zeros([128, 32], dtype=tl.float32)
        if c > 0:
            cbase = chk_ptr + chk_base + (c - 1) * 128 * 128
            ptrs = cbase + tl.expand_dims(offs_k, 1) * 128 + tl.expand_dims(offs_v, 0)
            S_init = tl.load(ptrs)
        S = S_init
        # forward recompute
        for tl_local in tl.range(0, C, num_stages=1):
            if tl_local < cur_len:
                t = c_start + tl_local
                k_vec = tl.load(k_ptr + q_base + t * 128 + offs_k).to(tl.float32)
                e_vec = tl.load(e_ptr + q_base + t * 128 + offs_k).to(tl.float32)
                g_vec = tl.load(g_ptr + q_base + t * 128 + offs_k).to(tl.float32)
                v_vec = tl.load(v_ptr + v_base + t * 128 + offs_v).to(tl.float32)
                bt = tl.load(beta_ptr + beta_base + t).to(tl.float32)
                ct = tl.load(gamma_ptr + beta_base + t).to(tl.float32)
                d_vec = tl.exp(g_vec)
                S = S * tl.expand_dims(d_vec, 1)
                a_vec = tl.sum(S * tl.expand_dims(e_vec, 1), axis=0)
                S = S - tl.expand_dims(ct * e_vec, 1) * tl.expand_dims(a_vec, 0)
                pred_vec = tl.sum(S * tl.expand_dims(k_vec, 1), axis=0)
                delta_vec = (v_vec - pred_vec) * bt
                S = S + tl.expand_dims(k_vec, 1) * tl.expand_dims(delta_vec, 0)
                tbase = tmp_ptr + tmp_base + tl_local * 128 * 128
                ptrs2 = tbase + tl.expand_dims(offs_k, 1) * 128 + tl.expand_dims(offs_v, 0)
                tl.store(ptrs2, S)
        # reverse
        for jj in tl.range(0, C, num_stages=1):
            tl_local = cur_len - 1 - jj
            if tl_local >= 0 and tl_local < cur_len:
                t = c_start + tl_local
                tbase = tmp_ptr + tmp_base + tl_local * 128 * 128
                ptrs2 = tbase + tl.expand_dims(offs_k, 1) * 128 + tl.expand_dims(offs_v, 0)
                S_cur = tl.load(ptrs2)
                if tl_local > 0:
                    pbase = tmp_ptr + tmp_base + (tl_local - 1) * 128 * 128
                    ptrs3 = pbase + tl.expand_dims(offs_k, 1) * 128 + tl.expand_dims(offs_v, 0)
                    S_prev = tl.load(ptrs3)
                else:
                    S_prev = S_init
                q_vec = tl.load(q_ptr + q_base + t * 128 + offs_k).to(tl.float32)
                k_vec = tl.load(k_ptr + q_base + t * 128 + offs_k).to(tl.float32)
                e_vec = tl.load(e_ptr + q_base + t * 128 + offs_k).to(tl.float32)
                g_vec = tl.load(g_ptr + q_base + t * 128 + offs_k).to(tl.float32)
                v_vec = tl.load(v_ptr + v_base + t * 128 + offs_v).to(tl.float32)
                do_vec = tl.load(do_ptr + v_base + t * 128 + offs_v).to(tl.float32)
                bt = tl.load(beta_ptr + beta_base + t).to(tl.float32)
                ct = tl.load(gamma_ptr + beta_base + t).to(tl.float32)
                d_vec = tl.exp(g_vec)
                DS = cur + tl.expand_dims(q_vec * scale, 1) * tl.expand_dims(do_vec, 0)
                S1 = S_prev * tl.expand_dims(d_vec, 1)
                a_vec = tl.sum(S1 * tl.expand_dims(e_vec, 1), axis=0)
                S2 = S1 - tl.expand_dims(ct * e_vec, 1) * tl.expand_dims(a_vec, 0)
                pred_vec = tl.sum(S2 * tl.expand_dims(k_vec, 1), axis=0)
                r_vec = v_vec - pred_vec
                w_vec = tl.sum(DS * tl.expand_dims(k_vec, 1), axis=0)
                se_vec = tl.sum(DS * tl.expand_dims(e_vec, 1), axis=0)
                ke = tl.sum(k_vec * e_vec)
                u_vec = se_vec - bt * w_vec * ke
                dv_vec = bt * w_vec
                tl.store(dv_ptr + v_base + t * 128 + offs_v, dv_vec)
                dq_part = tl.sum(S_cur * tl.expand_dims(do_vec, 0), axis=1) * scale
                DSr = tl.sum(DS * tl.expand_dims(r_vec, 0), axis=1)
                Spw = tl.sum(S_prev * tl.expand_dims(w_vec, 0), axis=1)
                DSa = tl.sum(DS * tl.expand_dims(a_vec, 0), axis=1)
                Spu = tl.sum(S_prev * tl.expand_dims(u_vec, 0), axis=1)
                row = tl.sum(DS * S_prev, axis=1)
                dk_part = bt * (DSr - d_vec * Spw)
                de_part = -ct * (DSa + d_vec * Spu)
                dg_part = d_vec * (row - bt * k_vec * Spw - ct * e_vec * Spu)
                db_part = tl.sum(w_vec * r_vec)
                dgam_part = -tl.sum(u_vec * a_vec)
                aw_part = tl.sum(a_vec * w_vec)
                dq_ptrs = dq_ptr + q_base + t * 128 + offs_k
                tl.atomic_add(dq_ptrs, dq_part)
                dk_ptrs = dk_ptr + q_base + t * 128 + offs_k
                tl.atomic_add(dk_ptrs, dk_part)
                de_ptrs = de_ptr + q_base + t * 128 + offs_k
                tl.atomic_add(de_ptrs, de_part)
                dgp = dg_ptr + q_base + t * 128 + offs_k
                tl.atomic_add(dgp, dg_part)
                tl.atomic_add(dbeta_ptr + beta_base + t, db_part)
                tl.atomic_add(dgamma_ptr + beta_base + t, dgam_part)
                tl.atomic_add(aw_ptr + beta_base + t, aw_part)
                cur = (DS - tl.expand_dims(bt * k_vec, 1) * tl.expand_dims(w_vec, 0) - tl.expand_dims(ct * e_vec, 1) * tl.expand_dims(u_vec, 0)) * tl.expand_dims(d_vec, 1)


@triton.jit
def _corr128(
    k_ptr, e_ptr, beta_ptr, gamma_ptr, aw_ptr, dk_ptr, de_ptr,
    H,
    T: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    BH = b * H + h
    offs_k = tl.arange(0, 128)
    q_base = (BH * T) * 128
    beta_base = BH * T
    for t in tl.range(0, T, num_stages=1):
        aw = tl.load(aw_ptr + beta_base + t)
        bt = tl.load(beta_ptr + beta_base + t).to(tl.float32)
        ct = tl.load(gamma_ptr + beta_base + t).to(tl.float32)
        factor = bt * ct * aw
        if factor != 0.0:
            k_vec = tl.load(k_ptr + q_base + t * 128 + offs_k).to(tl.float32)
            e_vec = tl.load(e_ptr + q_base + t * 128 + offs_k).to(tl.float32)
            dk_ptrs = dk_ptr + q_base + t * 128 + offs_k
            de_ptrs = de_ptr + q_base + t * 128 + offs_k
            dk_vec = tl.load(dk_ptrs)
            de_vec = tl.load(de_ptrs)
            dk_vec = dk_vec + factor * e_vec
            de_vec = de_vec + factor * k_vec
            tl.store(dk_ptrs, dk_vec)
            tl.store(de_ptrs, de_vec)


class _Fast128Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale_f):
        B, H, T, K = q.shape
        V = v.shape[-1]
        device = q.device
        C = 64
        Nc = (T + C - 1) // C
        out = torch.empty(B, H, T, V, dtype=torch.float32, device=device)
        chk = torch.zeros(B, H, Nc, K, V, dtype=torch.float32, device=device)
        grid = (B * H * 4,)
        _fwd_shard128[grid](q, k, v, e, g, beta, gamma, out, chk, H, float(scale_f), T, Nc, C, num_warps=4, num_stages=1)
        ctx.save_for_backward(q, k, v, e, g, beta, gamma, chk)
        ctx.B, ctx.H, ctx.T, ctx.C, ctx.Nc = B, H, T, C, Nc
        ctx.scale_f = float(scale_f)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma, chk = ctx.saved_tensors
        B, H, T, C, Nc = ctx.B, ctx.H, ctx.T, ctx.C, ctx.Nc
        device = q.device
        do = grad_out.contiguous().to(torch.float32)
        dq = torch.zeros(B, H, T, 128, dtype=torch.float32, device=device)
        dk = torch.zeros(B, H, T, 128, dtype=torch.float32, device=device)
        de = torch.zeros(B, H, T, 128, dtype=torch.float32, device=device)
        dg = torch.zeros(B, H, T, 128, dtype=torch.float32, device=device)
        dv = torch.empty(B, H, T, 128, dtype=torch.float32, device=device)
        dbeta = torch.zeros(B, H, T, dtype=torch.float32, device=device)
        dgamma = torch.zeros(B, H, T, dtype=torch.float32, device=device)
        aw = torch.zeros(B, H, T, dtype=torch.float32, device=device)
        tmp = torch.empty(B, H, C, 128, 128, dtype=torch.float32, device=device)
        grid = (B * H * 4,)
        _bwd_shard128[grid](q, k, v, e, g, beta, gamma, do, dq, dk, de, dg, dv, dbeta, dgamma, aw, chk, tmp, H, float(ctx.scale_f), T, Nc, C, num_warps=4, num_stages=1)
        grid2 = (B * H,)
        _corr128[grid2](k, e, beta, gamma, aw, dk, de, H, T, num_warps=4, num_stages=1)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), de.to(e.dtype), dg.to(g.dtype), dbeta.to(beta.dtype), dgamma.to(gamma.dtype), None)


# ---------------- generic fallback (single-CTA) ----------------

@triton.jit
def _fwd_ckpt_single(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr, chk_ptr,
    K, V, Nc, scale,
    T: tl.constexpr, C: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < V
    ek = tl.expand_dims(offs_k, 1)
    ev = tl.expand_dims(offs_v, 0)
    mask_2d = tl.expand_dims(mask_k, 1) & tl.expand_dims(mask_v, 0)
    KV = K * V
    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    for t in tl.range(0, T, num_stages=1):
        k_vec = tl.load(k_ptr + pid * T * K + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        e_vec = tl.load(e_ptr + pid * T * K + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        g_vec = tl.load(g_ptr + pid * T * K + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        q_vec = tl.load(q_ptr + pid * T * K + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptr + pid * T * V + t * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        bt = tl.load(beta_ptr + pid * T + t).to(tl.float32)
        ct = tl.load(gamma_ptr + pid * T + t).to(tl.float32)
        d_vec = tl.exp(g_vec)
        S = S * tl.expand_dims(d_vec, 1)
        a_vec = tl.sum(S * tl.expand_dims(e_vec, 1), axis=0)
        S = S - tl.expand_dims(ct * e_vec, 1) * tl.expand_dims(a_vec, 0)
        pred_vec = tl.sum(S * tl.expand_dims(k_vec, 1), axis=0)
        delta_vec = (v_vec - pred_vec) * bt
        S = S + tl.expand_dims(k_vec, 1) * tl.expand_dims(delta_vec, 0)
        qs_vec = q_vec * scale
        out_vec = tl.sum(S * tl.expand_dims(qs_vec, 1), axis=0)
        tl.store(o_ptr + pid * T * V + t * V + offs_v, out_vec, mask=mask_v)
        if (t + 1) % C == 0:
            cc = (t + 1) // C
            if cc < Nc:
                base = chk_ptr + (pid * Nc + cc) * KV
                ptrs = base + ek * V + ev
                tl.store(ptrs, S, mask=mask_2d)


@triton.jit
def _bwd_fused_single(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, do_ptr,
    dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
    chk_ptr, tmp_ptr, carry_ptr,
    T, K, V, Nc, c, scale,
    C: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < V
    ek = tl.expand_dims(offs_k, 1)
    ev = tl.expand_dims(offs_v, 0)
    mask_2d = tl.expand_dims(mask_k, 1) & tl.expand_dims(mask_v, 0)
    KV = K * V
    chk_init_base = chk_ptr + (pid * Nc + c) * KV
    chk_init_ptrs = chk_init_base + ek * V + ev
    S = tl.load(chk_init_ptrs, mask=mask_2d, other=0.0)
    for tl_local in tl.range(0, C, num_stages=1):
        t_global = c * C + tl_local
        if t_global < T:
            k_vec = tl.load(k_ptr + pid * T * K + t_global * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            e_vec = tl.load(e_ptr + pid * T * K + t_global * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            g_vec = tl.load(g_ptr + pid * T * K + t_global * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            v_vec = tl.load(v_ptr + pid * T * V + t_global * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            bt = tl.load(beta_ptr + pid * T + t_global).to(tl.float32)
            ct = tl.load(gamma_ptr + pid * T + t_global).to(tl.float32)
            d_vec = tl.exp(g_vec)
            S = S * tl.expand_dims(d_vec, 1)
            a_vec = tl.sum(S * tl.expand_dims(e_vec, 1), axis=0)
            S = S - tl.expand_dims(ct * e_vec, 1) * tl.expand_dims(a_vec, 0)
            pred_vec = tl.sum(S * tl.expand_dims(k_vec, 1), axis=0)
            delta_vec = (v_vec - pred_vec) * bt
            S = S + tl.expand_dims(k_vec, 1) * tl.expand_dims(delta_vec, 0)
            base = tmp_ptr + (pid * C + tl_local) * KV
            tl.store(base + ek * V + ev, S, mask=mask_2d)
    carry_base = carry_ptr + pid * KV
    carry_ptrs = carry_base + ek * V + ev
    cur = tl.load(carry_ptrs, mask=mask_2d, other=0.0)
    for ii in tl.range(0, C, num_stages=1):
        tl_local = C - 1 - ii
        t_global = c * C + tl_local
        if t_global < T:
            q_vec = tl.load(q_ptr + pid * T * K + t_global * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            k_vec = tl.load(k_ptr + pid * T * K + t_global * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            e_vec = tl.load(e_ptr + pid * T * K + t_global * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            g_vec = tl.load(g_ptr + pid * T * K + t_global * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            v_vec = tl.load(v_ptr + pid * T * V + t_global * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            do_vec = tl.load(do_ptr + pid * T * V + t_global * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            bt = tl.load(beta_ptr + pid * T + t_global).to(tl.float32)
            ct = tl.load(gamma_ptr + pid * T + t_global).to(tl.float32)
            cur_base = tmp_ptr + (pid * C + tl_local) * KV
            S_cur = tl.load(cur_base + ek * V + ev, mask=mask_2d, other=0.0)
            if tl_local > 0:
                prev_base = tmp_ptr + (pid * C + tl_local - 1) * KV
                S_prev = tl.load(prev_base + ek * V + ev, mask=mask_2d, other=0.0)
            else:
                S_prev = tl.load(chk_init_ptrs, mask=mask_2d, other=0.0)
            d_vec = tl.exp(g_vec)
            qs_vec = q_vec * scale
            DS = cur + tl.expand_dims(qs_vec, 1) * tl.expand_dims(do_vec, 0)
            dq_vec = tl.sum(S_cur * tl.expand_dims(do_vec, 0), axis=1) * scale
            tl.store(dq_ptr + pid * T * K + t_global * K + offs_k, dq_vec, mask=mask_k)
            ed_vec = e_vec * d_vec
            kd_vec = k_vec * d_vec
            a_vec = tl.sum(S_prev * tl.expand_dims(ed_vec, 1), axis=0)
            w_vec = tl.sum(DS * tl.expand_dims(k_vec, 1), axis=0)
            se_vec = tl.sum(DS * tl.expand_dims(e_vec, 1), axis=0)
            ke = tl.sum(k_vec * e_vec)
            u_vec = se_vec - bt * w_vec * ke
            yk_vec = tl.sum(S_prev * tl.expand_dims(kd_vec, 1), axis=0)
            pred_vec = yk_vec - ct * a_vec * ke
            r_vec = v_vec - pred_vec
            dv_vec = bt * w_vec
            tl.store(dv_ptr + pid * T * V + t_global * V + offs_v, dv_vec, mask=mask_v)
            dbeta_val = tl.sum(w_vec * r_vec)
            tl.store(dbeta_ptr + pid * T + t_global, dbeta_val)
            dgamma_val = -tl.sum(u_vec * a_vec)
            tl.store(dgamma_ptr + pid * T + t_global, dgamma_val)
            DSa_vec = tl.sum(DS * tl.expand_dims(a_vec, 0), axis=1)
            Spw_vec = tl.sum(S_prev * tl.expand_dims(w_vec, 0), axis=1)
            Spu_vec = tl.sum(S_prev * tl.expand_dims(u_vec, 0), axis=1)
            DSr_vec = tl.sum(DS * tl.expand_dims(r_vec, 0), axis=1)
            aw = tl.sum(a_vec * w_vec)
            Yw_vec = d_vec * Spw_vec
            Yu_vec = d_vec * Spu_vec
            Mw_vec = Yw_vec - ct * e_vec * aw
            dk_vec = bt * (DSr_vec - Mw_vec)
            tl.store(dk_ptr + pid * T * K + t_global * K + offs_k, dk_vec, mask=mask_k)
            DMa_vec = DSa_vec - bt * k_vec * aw
            de_vec = -ct * (DMa_vec + Yu_vec)
            tl.store(de_ptr + pid * T * K + t_global * K + offs_k, de_vec, mask=mask_k)
            row_DS_S = tl.sum(DS * S_prev, axis=1)
            dg_vec = d_vec * (row_DS_S - bt * k_vec * Spw_vec - ct * e_vec * Spu_vec)
            tl.store(dg_ptr + pid * T * K + t_global * K + offs_k, dg_vec, mask=mask_k)
            cur = (DS - tl.expand_dims(bt * k_vec, 1) * tl.expand_dims(w_vec, 0) - tl.expand_dims(ct * e_vec, 1) * tl.expand_dims(u_vec, 0)) * tl.expand_dims(d_vec, 1)
    tl.store(carry_ptrs, cur, mask=mask_2d)


def _np2(x, minimum=16):
    if x <= minimum:
        return minimum
    return 1 << ((x - 1).bit_length())


class _GenericFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale_f):
        B, H, T, K = q.shape
        V = v.shape[-1]
        device = q.device
        C = 64
        Nc = (T + C - 1) // C if T > 0 else 0
        BLOCK_K = _np2(K)
        BLOCK_V = _np2(V)
        out = torch.empty(B, H, T, V, dtype=torch.float32, device=device)
        if T == 0 or K == 0 or V == 0:
            ctx._empty = True
            ctx._dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
            ctx._shapes = (B, H, T, K, V)
            return out
        ctx._empty = False
        chk = torch.zeros(B, H, Nc, K, V, dtype=torch.float32, device=device)
        grid = (B * H,)
        _fwd_ckpt_single[grid](q, k, v, e, g, beta, gamma, out, chk, K, V, Nc, float(scale_f), T, C, BLOCK_K, BLOCK_V, num_warps=8, num_stages=1)
        ctx.save_for_backward(q, k, v, e, g, beta, gamma, chk)
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V, ctx.Nc, ctx.C = B, H, T, K, V, Nc, C
        ctx.BLOCK_K, ctx.BLOCK_V = BLOCK_K, BLOCK_V
        ctx.scale_f = float(scale_f)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if getattr(ctx, "_empty", False):
            B, H, T, K, V = ctx._shapes
            qd, kd, vd, ed, gd, bd, cd = ctx._dtypes
            device = grad_out.device
            return (torch.empty(B, H, T, K, dtype=qd, device=device), torch.empty(B, H, T, K, dtype=kd, device=device), torch.empty(B, H, T, V, dtype=vd, device=device), torch.empty(B, H, T, K, dtype=ed, device=device), torch.empty(B, H, T, K, dtype=gd, device=device), torch.empty(B, H, T, dtype=bd, device=device), torch.empty(B, H, T, dtype=cd, device=device), None)
        q, k, v, e, g, beta, gamma, chk = ctx.saved_tensors
        B, H, T, K, V, Nc, C = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V, ctx.Nc, ctx.C
        BLOCK_K, BLOCK_V = ctx.BLOCK_K, ctx.BLOCK_V
        device = q.device
        do = grad_out.contiguous().to(torch.float32)
        dq = torch.empty(B, H, T, K, dtype=torch.float32, device=device)
        dk = torch.empty(B, H, T, K, dtype=torch.float32, device=device)
        dv = torch.empty(B, H, T, V, dtype=torch.float32, device=device)
        de = torch.empty(B, H, T, K, dtype=torch.float32, device=device)
        dg = torch.empty(B, H, T, K, dtype=torch.float32, device=device)
        dbeta = torch.empty(B, H, T, dtype=torch.float32, device=device)
        dgamma = torch.empty(B, H, T, dtype=torch.float32, device=device)
        tmp = torch.empty(B, H, C, K, V, dtype=torch.float32, device=device)
        carry = torch.zeros(B, H, K, V, dtype=torch.float32, device=device)
        grid = (B * H,)
        for c in range(Nc - 1, -1, -1):
            _bwd_fused_single[grid](q, k, v, e, g, beta, gamma, do, dq, dk, dv, de, dg, dbeta, dgamma, chk, tmp, carry, T, K, V, Nc, c, float(ctx.scale_f), C, BLOCK_K, BLOCK_V, num_warps=8, num_stages=1)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), de.to(e.dtype), dg.to(g.dtype), dbeta.to(beta.dtype), dgamma.to(gamma.dtype), None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
    B, H, T, K = q.shape
    V = v.shape[-1]
    if K == 128 and V == 128 and T > 0:
        qc = q.contiguous()
        kc = k.contiguous()
        vc = v.contiguous()
        ec = e.contiguous()
        gc = g.contiguous()
        bc = beta.contiguous()
        cc = gamma.contiguous()
        return _Fast128Func.apply(qc, kc, vc, ec, gc, bc, cc, scale_f)
    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    ec = e.contiguous()
    gc = g.contiguous()
    bc = beta.contiguous()
    cc = gamma.contiguous()
    return _GenericFunc.apply(qc, kc, vc, ec, gc, bc, cc, scale_f)
