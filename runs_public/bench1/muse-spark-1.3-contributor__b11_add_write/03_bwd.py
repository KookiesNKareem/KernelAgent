import torch
import triton
import triton.language as tl


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


@triton.jit
def _fwd_ckpt_kernel(
    qs_ptr, k_ptr, v_ptr, e_ptr, d_ptr, b_ptr, c_ptr, out_ptr, ckpt_ptr,
    BH, T, K, V, Ct, Nc,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
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
    for c in tl.range(Nc):
        for ct in tl.range(Ct):
            t = c * Ct + ct
            if t < T:
                qv = tl.load(qs_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0)
                kv = tl.load(k_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
                ev = tl.load(e_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
                dvv = tl.load(d_ptr + (bh * T + t) * K + k_offs, mask=mask_k, other=0.0)
                vv = tl.load(v_ptr + (bh * T + t) * V + v_offs, mask=mask_v, other=0.0).to(tl.float32)
                bv = tl.load(b_ptr + bh * T + t)
                cv = tl.load(c_ptr + bh * T + t)
                d_e = tl.expand_dims(dvv, 1)
                k_e = tl.expand_dims(kv, 1)
                e_e = tl.expand_dims(ev, 1)
                q_e = tl.expand_dims(qv, 1)
                v_e = tl.expand_dims(vv, 0)
                S = S * d_e
                pred = tl.sum(S * k_e, 0)
                diff = (vv - pred) * bv
                S = S + k_e * tl.expand_dims(diff, 0) + cv * e_e * v_e
                outv = tl.sum(S * q_e, 0)
                tl.store(out_ptr + (bh * T + t) * V + v_offs, outv, mask=mask_v)
        ckpt_base = (bh * (Nc + 1) + (c + 1)) * K * V
        ckpt_ptrs = ckpt_ptr + ckpt_base + k_col * V + v_row
        tl.store(ckpt_ptrs, S, mask=mask_2d)


@triton.jit
def _bwd_chunk_kernel(
    qs_ptr, k_ptr, e_ptr, d_ptr, b_ptr, c_ptr, v_ptr, do_ptr,
    ckpt_ptr, scratch_ptr, carry_ptr,
    dq_ptr, dk_ptr, de_ptr, dg_ptr, dv_ptr, db_ptr, dc_ptr,
    BH, T, K, V, Ct, Nc, NV, t0, Ct_len, c_idx, scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
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
        scr_ptrs = scratch_ptr + (bh * Ct + ct) * K * V + k_col * V + v_row
        tl.store(scr_ptrs, S, mask=mask_2d)
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
        k_t_e = tl.expand_dims(k_t, 1)
        s_vec = tl.sum(k_t_e * DS_cur, 0)
        Zmat = S_prev * tl.expand_dims(d_t, 1)
        r_vec = tl.sum(k_t_e * Zmat, 0)
        vmr = v_t - r_vec
        wvec = b_t * k_t + c_t * e_t
        dv_vec = tl.sum(DS_cur * tl.expand_dims(wvec, 1), 0)
        tl.store(dv_ptr + (bh * T + t) * V + v_offs, dv_vec, mask=mask_v)
        tmp_k = tl.sum(DS_cur * tl.expand_dims(v_t, 0), 1)
        de_part = c_t * tmp_k
        tl.store(de_ptr + ((bh * NV + pid_v) * T + t) * K + k_offs, de_part, mask=mask_k)
        dc_part = tl.sum(e_t * tmp_k, 0)
        tl.store(dc_ptr + (bh * NV + pid_v) * T + t, dc_part)
        db_part = tl.sum(s_vec * vmr, 0)
        tl.store(db_ptr + (bh * NV + pid_v) * T + t, db_part)
        dq_part = scale * tl.sum(S_t * tl.expand_dims(do_t, 0), 1)
        tl.store(dq_ptr + ((bh * NV + pid_v) * T + t) * K + k_offs, dq_part, mask=mask_k)
        a_k = tl.sum(S_prev * DS_cur, 1)
        b2_k = tl.sum(S_prev * tl.expand_dims(s_vec, 0), 1)
        dg_part = d_t * (a_k - b_t * k_t * b2_k)
        tl.store(dg_ptr + ((bh * NV + pid_v) * T + t) * K + k_offs, dg_part, mask=mask_k)
        c1_k = tl.sum(DS_cur * tl.expand_dims(vmr, 0), 1)
        c2_k = tl.sum(Zmat * tl.expand_dims(s_vec, 0), 1)
        dk_part = b_t * (c1_k - c2_k)
        tl.store(dk_ptr + ((bh * NV + pid_v) * T + t) * K + k_offs, dk_part, mask=mask_k)
        DS_next = DS_cur
        k_next = k_t
        d_next = d_t
        b_next = b_t
    tl.store(carry_ptrs, DS_next, mask=mask_2d)


class _RecFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        device = q.device
        if isinstance(scale, torch.Tensor):
            scale_f = float(scale.detach().item())
        else:
            scale_f = float(scale)
        if B == 0 or H == 0 or T == 0 or V == 0 or K == 0:
            out = torch.zeros((B, H, T, V), dtype=torch.float32, device=device)
            ctx.empty = True
            ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
            ctx.q_dtype, ctx.k_dtype, ctx.v_dtype, ctx.e_dtype = q.dtype, k.dtype, v.dtype, e.dtype
            ctx.g_dtype, ctx.b_dtype, ctx.c_dtype = g.dtype, beta.dtype, gamma.dtype
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
        if BK <= 32:
            BV = 64
        elif BK <= 128:
            BV = 32
        else:
            BV = 16
        NV = (V + BV - 1) // BV
        out = torch.empty((B, H, T, V), dtype=torch.float32, device=device)
        ckpt = torch.zeros((B, H, Nc + 1, K, V), dtype=torch.float32, device=device)
        if BH > 0 and NV > 0 and Nc > 0:
            _fwd_ckpt_kernel[(BH, NV)](
                q_scaled, k_c, v_c, e_c, d_mat, b_c, c_c, out, ckpt,
                BH, T, K, V, Ct, Nc,
                BK, BV,
                num_warps=4,
            )
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
        ctx.BH, ctx.Ct, ctx.Nc, ctx.BK, ctx.BV, ctx.NV = BH, Ct, Nc, BK, BV, NV
        ctx.scale_f = scale_f
        ctx.q_dtype, ctx.k_dtype, ctx.v_dtype, ctx.e_dtype = q.dtype, k.dtype, v.dtype, e.dtype
        ctx.g_dtype, ctx.b_dtype, ctx.c_dtype = g.dtype, beta.dtype, gamma.dtype
        ctx.q_scaled, ctx.k_c, ctx.e_c, ctx.d_mat = q_scaled, k_c, e_c, d_mat
        ctx.v_c, ctx.b_c, ctx.c_c, ctx.ckpt = v_c, b_c, c_c, ckpt
        return out

    @staticmethod
    def backward(ctx, grad_out):
        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        needs = ctx.needs_input_grad
        def _none(shape, dtype):
            return None
        if ctx.empty:
            outs = []
            for i, (sh, dt) in enumerate([
                ((B, H, T, K), ctx.q_dtype), ((B, H, T, K), ctx.k_dtype),
                ((B, H, T, V), ctx.v_dtype), ((B, H, T, K), ctx.e_dtype),
                ((B, H, T, K), ctx.g_dtype), ((B, H, T), ctx.b_dtype),
                ((B, H, T), ctx.c_dtype),
            ]):
                if needs[i]:
                    outs.append(torch.zeros(sh, dtype=dt, device=grad_out.device))
                else:
                    outs.append(None)
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
                _bwd_chunk_kernel[(BH, NV)](
                    qs, kk, ee, dd, bb, cc, vv, do_c,
                    ckpt, scratch, carry,
                    dq_p, dk_p, de_p, dg_p, dv_full, db_p, dc_p,
                    BH, T, K, V, Ct, Nc, NV, t0, clen, c, scale_f,
                    BK, BV,
                    num_warps=4,
                )
            dq_sum = dq_p.sum(dim=2)
            dk_sum = dk_p.sum(dim=2)
            de_sum = de_p.sum(dim=2)
            dg_sum = dg_p.sum(dim=2)
            db_sum = db_p.sum(dim=2)
            dc_sum = dc_p.sum(dim=2)
        gq = dq_sum.to(ctx.q_dtype) if needs[0] else None
        gk = dk_sum.to(ctx.k_dtype) if needs[1] else None
        gv = dv_full.to(ctx.v_dtype) if needs[2] else None
        ge = de_sum.to(ctx.e_dtype) if needs[3] else None
        gg = dg_sum.to(ctx.g_dtype) if needs[4] else None
        gb = db_sum.to(ctx.b_dtype) if needs[5] else None
        gc = dc_sum.to(ctx.c_dtype) if needs[6] else None
        return (gq, gk, gv, ge, gg, gb, gc, None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _RecFunc.apply(q, k, v, e, g, beta, gamma, scale)
