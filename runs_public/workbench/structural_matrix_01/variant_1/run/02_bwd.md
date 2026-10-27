```python
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr, s_ptr,
    s_qb, s_qh, s_qt, s_qk,
    s_kb, s_kh, s_kt, s_kk,
    s_vb, s_vh, s_vt, s_vv,
    s_eb, s_eh, s_et, s_ek,
    s_gb, s_gh, s_gt, s_gk,
    s_betab, s_betah, s_betat,
    s_gammab, s_gammah, s_gammat,
    s_ob, s_oh, s_ot, s_ov,
    s_sb, s_sh, s_st, s_sk, s_sv,
    B, H, T, Ksz, Vsz,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    m_k = off_k < Ksz
    m_v = off_v < Vsz
    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    for t in range(T):
        k_vec = tl.load(k_ptr + b * s_kb + h * s_kh + t * s_kt + off_k * s_kk, mask=m_k, other=0.0).to(tl.float32)
        e_vec = tl.load(e_ptr + b * s_eb + h * s_eh + t * s_et + off_k * s_ek, mask=m_k, other=0.0).to(tl.float32)
        g_vec = tl.load(g_ptr + b * s_gb + h * s_gh + t * s_gt + off_k * s_gk, mask=m_k, other=0.0).to(tl.float32)
        q_vec = tl.load(q_ptr + b * s_qb + h * s_qh + t * s_qt + off_k * s_qk, mask=m_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptr + b * s_vb + h * s_vh + t * s_vt + off_v * s_vv, mask=m_v, other=0.0).to(tl.float32)
        bt = tl.load(beta_ptr + b * s_betab + h * s_betah + t * s_betat).to(tl.float32)
        ct = tl.load(gamma_ptr + b * s_gammab + h * s_gammah + t * s_gammat).to(tl.float32)
        d_vec = tl.exp(g_vec)
        S = S * d_vec[:, None]
        a_vec = tl.sum(S * e_vec[:, None], axis=0)
        S = S - ct * e_vec[:, None] * a_vec[None, :]
        pred_vec = tl.sum(S * k_vec[:, None], axis=0)
        delta_vec = (v_vec - pred_vec) * bt
        S = S + k_vec[:, None] * delta_vec[None, :]
        qs = q_vec * scale
        out_vec = tl.sum(S * qs[:, None], axis=0)
        tl.store(o_ptr + b * s_ob + h * s_oh + t * s_ot + off_v * s_ov, out_vec, mask=m_v)
        s_base = s_ptr + b * s_sb + h * s_sh + t * s_st
        s_ptrs = s_base + off_k[:, None] * s_sk + off_v[None, :] * s_sv
        s_mask = m_k[:, None] & m_v[None, :]
        tl.store(s_ptrs, S, mask=s_mask)


@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, do_ptr, s_ptr,
    dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
    s_qb, s_qh, s_qt, s_qk,
    s_kb, s_kh, s_kt, s_kk,
    s_vb, s_vh, s_vt, s_vv,
    s_eb, s_eh, s_et, s_ek,
    s_gb, s_gh, s_gt, s_gk,
    s_betab, s_betah, s_betat,
    s_gammab, s_gammah, s_gammat,
    s_dob, s_doh, s_dot, s_dov,
    s_sb, s_sh, s_st, s_sk, s_sv,
    s_dqb, s_dqh, s_dqt, s_dqk,
    s_dkb, s_dkh, s_dkt, s_dkk,
    s_dvb, s_dvh, s_dvt, s_dvv,
    s_deb, s_deh, s_det, s_dek,
    s_dgb, s_dgh, s_dgt, s_dgk,
    s_dbetab, s_dbetah, s_dbetat,
    s_dgammab, s_dgammah, s_dgammat,
    B, H, T, Ksz, Vsz,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    m_k = off_k < Ksz
    m_v = off_v < Vsz
    m_2d = m_k[:, None] & m_v[None, :]
    carry = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    for tt in range(T):
        t = T - 1 - tt
        k_vec = tl.load(k_ptr + b * s_kb + h * s_kh + t * s_kt + off_k * s_kk, mask=m_k, other=0.0).to(tl.float32)
        e_vec = tl.load(e_ptr + b * s_eb + h * s_eh + t * s_et + off_k * s_ek, mask=m_k, other=0.0).to(tl.float32)
        g_vec = tl.load(g_ptr + b * s_gb + h * s_gh + t * s_gt + off_k * s_gk, mask=m_k, other=0.0).to(tl.float32)
        q_vec = tl.load(q_ptr + b * s_qb + h * s_qh + t * s_qt + off_k * s_qk, mask=m_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptr + b * s_vb + h * s_vh + t * s_vt + off_v * s_vv, mask=m_v, other=0.0).to(tl.float32)
        bt = tl.load(beta_ptr + b * s_betab + h * s_betah + t * s_betat).to(tl.float32)
        ct = tl.load(gamma_ptr + b * s_gammab + h * s_gammah + t * s_gammat).to(tl.float32)
        do_vec = tl.load(do_ptr + b * s_dob + h * s_doh + t * s_dot + off_v * s_dov, mask=m_v, other=0.0).to(tl.float32)
        s_base_t = s_ptr + b * s_sb + h * s_sh + t * s_st
        s_ptrs_t = s_base_t + off_k[:, None] * s_sk + off_v[None, :] * s_sv
        St = tl.load(s_ptrs_t, mask=m_2d, other=0.0).to(tl.float32)
        Sprev = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
        if t > 0:
            s_base_p = s_ptr + b * s_sb + h * s_sh + (t - 1) * s_st
            s_ptrs_p = s_base_p + off_k[:, None] * s_sk + off_v[None, :] * s_sv
            Sprev = tl.load(s_ptrs_p, mask=m_2d, other=0.0).to(tl.float32)
        d_vec = tl.exp(g_vec)
        S1 = Sprev * d_vec[:, None]
        a_vec = tl.sum(S1 * e_vec[:, None], axis=0)
        S2 = S1 - ct * e_vec[:, None] * a_vec[None, :]
        pred_vec = tl.sum(S2 * k_vec[:, None], axis=0)
        r_vec = v_vec - pred_vec
        qs_vec = q_vec * scale
        G = carry + qs_vec[:, None] * do_vec[None, :]
        w_vec = tl.sum(G * k_vec[:, None], axis=0)
        Hm = G - bt * k_vec[:, None] * w_vec[None, :]
        y_vec = tl.sum(Hm * e_vec[:, None], axis=0)
        Fm = Hm - ct * e_vec[:, None] * y_vec[None, :]
        dq_vec = tl.sum(St * do_vec[None, :], axis=1) * scale
        dv_vec = bt * w_vec
        Gr_vec = tl.sum(G * r_vec[None, :], axis=1)
        S2w_vec = tl.sum(S2 * w_vec[None, :], axis=1)
        dk_vec = bt * (Gr_vec - S2w_vec)
        dbt = tl.sum(w_vec * r_vec, axis=0)
        Ha_vec = tl.sum(Hm * a_vec[None, :], axis=1)
        S1y_vec = tl.sum(S1 * y_vec[None, :], axis=1)
        de_vec = -ct * (Ha_vec + S1y_vec)
        dct = -tl.sum(y_vec * a_vec, axis=0)
        dg_vec = tl.sum(Fm * S1, axis=1)
        tl.store(dq_ptr + b * s_dqb + h * s_dqh + t * s_dqt + off_k * s_dqk, dq_vec, mask=m_k)
        tl.store(dk_ptr + b * s_dkb + h * s_dkh + t * s_dkt + off_k * s_dkk, dk_vec, mask=m_k)
        tl.store(de_ptr + b * s_deb + h * s_deh + t * s_det + off_k * s_dek, de_vec, mask=m_k)
        tl.store(dg_ptr + b * s_dgb + h * s_dgh + t * s_dgt + off_k * s_dgk, dg_vec, mask=m_k)
        tl.store(dv_ptr + b * s_dvb + h * s_dvh + t * s_dvt + off_v * s_dvv, dv_vec, mask=m_v)
        tl.store(dbeta_ptr + b * s_dbetab + h * s_dbetah + t * s_dbetat, dbt)
        tl.store(dgamma_ptr + b * s_dgammab + h * s_dgammah + t * s_dgammat, dct)
        carry = Fm * d_vec[:, None]


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    if p < 16:
        p = 16
    return p


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        Vd = v.shape[-1]
        scale_f = float(scale) if not isinstance(scale, torch.Tensor) else float(scale.item())
        if g.dim() == 3:
            g_exp = g.unsqueeze(-1).expand(B, H, T, K)
        elif g.dim() == 4 and g.shape[-1] != K:
            g_exp = g.expand(B, H, T, K)
        else:
            g_exp = g
        out = torch.empty(B, H, T, Vd, dtype=torch.float32, device=q.device)
        s_save = torch.empty(B, H, T, K, Vd, dtype=torch.float32, device=q.device)
        BK = _next_pow2(K)
        BV = _next_pow2(Vd)
        nw = 8 if BK * BV > 1024 else 4
        grid = (B * H,)
        _fwd_kernel[grid](
            q, k, v, e, g_exp, beta, gamma, out, s_save,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            e.stride(0), e.stride(1), e.stride(2), e.stride(3),
            g_exp.stride(0), g_exp.stride(1), g_exp.stride(2), g_exp.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            s_save.stride(0), s_save.stride(1), s_save.stride(2), s_save.stride(3), s_save.stride(4),
            B, H, T, K, Vd, scale_f,
            BLOCK_K=BK, BLOCK_V=BV, num_warps=nw,
        )
        ctx.save_for_backward(q, k, v, e, g, g_exp, beta, gamma, s_save)
        ctx.BHKTV = (B, H, T, K, Vd)
        ctx.BK = BK
        ctx.BV = BV
        ctx.scale_f = scale_f
        ctx.nw = nw
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g_orig, g_exp, beta, gamma, s_save = ctx.saved_tensors
        B, H, T, K, Vd = ctx.BHKTV
        BK = ctx.BK
        BV = ctx.BV
        scale_f = ctx.scale_f
        go = grad_out.contiguous()
        dq32 = torch.empty(B, H, T, K, dtype=torch.float32, device=q.device)
        dk32 = torch.empty(B, H, T, K, dtype=torch.float32, device=q.device)
        de32 = torch.empty(B, H, T, K, dtype=torch.float32, device=q.device)
        dg_exp = torch.empty(B, H, T, K, dtype=torch.float32, device=q.device)
        dv32 = torch.empty(B, H, T, Vd, dtype=torch.float32, device=q.device)
        db32 = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
        dgm32 = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
        grid = (B * H,)
        _bwd_kernel[grid](
            q, k, v, e, g_exp, beta, gamma, go, s_save,
            dq32, dk32, dv32, de32, dg_exp, db32, dgm32,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            e.stride(0), e.stride(1), e.stride(2), e.stride(3),
            g_exp.stride(0), g_exp.stride(1), g_exp.stride(2), g_exp.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            go.stride(0), go.stride(1), go.stride(2), go.stride(3),
            s_save.stride(0), s_save.stride(1), s_save.stride(2), s_save.stride(3), s_save.stride(4),
            dq32.stride(0), dq32.stride(1), dq32.stride(2), dq32.stride(3),
            dk32.stride(0), dk32.stride(1), dk32.stride(2), dk32.stride(3),
            dv32.stride(0), dv32.stride(1), dv32.stride(2), dv32.stride(3),
            de32.stride(0), de32.stride(1), de32.stride(2), de32.stride(3),
            dg_exp.stride(0), dg_exp.stride(1), dg_exp.stride(2), dg_exp.stride(3),
            db32.stride(0), db32.stride(1), db32.stride(2),
            dgm32.stride(0), dgm32.stride(1), dgm32.stride(2),
            B, H, T, K, Vd, scale_f,
            BLOCK_K=BK, BLOCK_V=BV, num_warps=ctx.nw,
        )
        dq = dq32.to(q.dtype)
        dk = dk32.to(k.dtype)
        dv = dv32.to(v.dtype)
        de = de32.to(e.dtype)
        db = db32.to(beta.dtype)
        dgm = dgm32.to(gamma.dtype)
        if g_orig.shape == dg_exp.shape:
            dg = dg_exp.to(g_orig.dtype)
        elif g_orig.dim() == 3:
            dg = dg_exp.sum(-1).to(g_orig.dtype)
        else:
            try:
                dg = torch.zeros_like(g_orig, dtype=torch.float32)
                dg = dg + dg_exp.sum(-1, keepdim=False).to(torch.float32) if g_orig.dim() == 3 else dg
                if dg.shape != g_orig.shape:
                    dg = dg_exp.sum(-1).reshape(g_orig.shape).to(g_orig.dtype)
                else:
                    dg = dg.to(g_orig.dtype)
            except Exception:
                dg = dg_exp.sum(-1).to(g_orig.dtype)
        return dq, dk, dv, de, dg, db, dgm, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _Fn.apply(q, k, v, e, g, beta, gamma, scale)
```
I fuse the whole time loop per batch-head in Triton, keeping the KxV state in registers; forward saves all S_t for reuse. Backward runs a single reverse Triton loop carrying the adjoint G_t = scale*q_t*do_t^T + M_{t+1}^T G_{t+1}, recomputing S1/S2/pred from saved S_{t-1} and deriving closed-form vector grads for q,k,v,e,g,beta,gamma. Python only allocates, launches, handles g-broadcast reduction and dtype casts. Unsure about Triton support for dynamic `range(T)` reverse indexing and runtime `if t>0`, and exact bf16 grad casting/accumulation order versus PyTorch autograd.