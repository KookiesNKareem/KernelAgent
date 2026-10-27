import torch
import triton
import triton.language as tl


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


class _RecurrenceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        device = q.device
        C = 64
        if T == 0:
            C = 64
            Nc = 0
        else:
            if T < C:
                pass
            Nc = (T + C - 1) // C
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
        _fwd_ckpt_single[grid](
            q, k, v, e, g, beta, gamma, out, chk,
            K, V, Nc, float(scale),
            T, C, BLOCK_K, BLOCK_V,
            num_warps=8, num_stages=1,
        )
        ctx.save_for_backward(q, k, v, e, g, beta, gamma, chk)
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V, ctx.Nc, ctx.C = B, H, T, K, V, Nc, C
        ctx.BLOCK_K, ctx.BLOCK_V = BLOCK_K, BLOCK_V
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if getattr(ctx, "_empty", False):
            B, H, T, K, V = ctx._shapes
            qd, kd, vd, ed, gd, bd, cd = ctx._dtypes
            device = grad_out.device
            return (
                torch.empty(B, H, T, K, dtype=qd, device=device),
                torch.empty(B, H, T, K, dtype=kd, device=device),
                torch.empty(B, H, T, V, dtype=vd, device=device),
                torch.empty(B, H, T, K, dtype=ed, device=device),
                torch.empty(B, H, T, K, dtype=gd, device=device),
                torch.empty(B, H, T, dtype=bd, device=device),
                torch.empty(B, H, T, dtype=cd, device=device),
                None,
            )
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
            _bwd_fused_single[grid](
                q, k, v, e, g, beta, gamma, do,
                dq, dk, dv, de, dg, dbeta, dgamma,
                chk, tmp, carry,
                T, K, V, Nc, c, float(ctx.scale),
                C, BLOCK_K, BLOCK_V,
                num_warps=8, num_stages=1,
            )
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            de.to(e.dtype),
            dg.to(g.dtype),
            dbeta.to(beta.dtype),
            dgamma.to(gamma.dtype),
            None,
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)
    q_c = q.contiguous() if isinstance(q, torch.Tensor) else q
    k_c = k.contiguous()
    v_c = v.contiguous()
    e_c = e.contiguous()
    g_c = g.contiguous()
    beta_c = beta.contiguous()
    gamma_c = gamma.contiguous()
    return _RecurrenceFunc.apply(q_c, k_c, v_c, e_c, g_c, beta_c, gamma_c, scale_f)
