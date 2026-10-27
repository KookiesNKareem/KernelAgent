```python
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr, ckpt_ptr,
    T, K, Vd, scale, LC, NCKPT,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(1)
    vb = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    v_offs = vb * BLOCK_V + offs_v
    mask_v = v_offs < Vd

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    g_base = g_ptr + bh * T * K
    v_base = v_ptr + bh * T * Vd
    o_base = out_ptr + bh * T * Vd
    beta_base = beta_ptr + bh * T
    gamma_base = gamma_ptr + bh * T

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in range(T):
        k_t = tl.load(k_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        e_t = tl.load(e_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        g_t = tl.load(g_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        q_t = tl.load(q_base + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_t = tl.load(v_base + t * Vd + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        b = tl.load(beta_base + t).to(tl.float32)
        gm = tl.load(gamma_base + t).to(tl.float32)
        c = gm * (1.0 - b)
        d_t = tl.exp(g_t)
        d_col = tl.expand_dims(d_t, 1)
        S = S * d_col
        e_col = tl.expand_dims(e_t, 1)
        s_e = tl.sum(S * e_col, axis=0)
        S = S - c * e_col * tl.expand_dims(s_e, 0)
        k_col = tl.expand_dims(k_t, 1)
        pred = tl.sum(S * k_col, axis=0)
        delta = b * (v_t - pred)
        S = S + k_col * tl.expand_dims(delta, 0)
        q_col = tl.expand_dims(q_t * scale, 1)
        o_t = tl.sum(S * q_col, axis=0)
        tl.store(o_base + t * Vd + v_offs, o_t, mask=mask_v)
        is_bnd = ((t + 1) % LC == 0) | (t == T - 1)
        if is_bnd:
            idx_div = (t + 1) // LC
            is_div = ((t + 1) % LC) == 0
            idx = tl.where(is_div, idx_div, NCKPT - 1)
            ckpt_base = ckpt_ptr + (bh * NCKPT + idx) * K * Vd
            ptrs = ckpt_base + offs_k[:, None] * Vd + v_offs[None, :]
            m2d = mask_k[:, None] & mask_v[None, :]
            tl.store(ptrs, S, mask=m2d)


@triton.jit
def _bwd_chunk_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, do_ptr,
    ckpt_ptr, dS_ptr, scratch_ptr,
    dq_ptr, dk_ptr, de_ptr, dg_ptr, dv_ptr, dbeta_ptr, dgamma_ptr,
    T, K, Vd, scale, chunk_idx, cur_L, LC, NCKPT,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(1)
    vb = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    v_offs = vb * BLOCK_V + offs_v
    mask_v = v_offs < Vd
    m2d = mask_k[:, None] & mask_v[None, :]
    chunk_start = chunk_idx * LC

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    g_base = g_ptr + bh * T * K
    v_base = v_ptr + bh * T * Vd
    do_base = do_ptr + bh * T * Vd
    beta_base = beta_ptr + bh * T
    gamma_base = gamma_ptr + bh * T
    dq_base = dq_ptr + bh * T * K
    dk_base = dk_ptr + bh * T * K
    de_base = de_ptr + bh * T * K
    dg_base = dg_ptr + bh * T * K
    dv_base = dv_ptr + bh * T * Vd
    db_base = dbeta_ptr + bh * T
    dgma_base = dgamma_ptr + bh * T

    ckpt_start_base = ckpt_ptr + (bh * NCKPT + chunk_idx) * K * Vd
    s_ptrs = ckpt_start_base + offs_k[:, None] * Vd + v_offs[None, :]
    S_cur = tl.load(s_ptrs, mask=m2d, other=0.0).to(tl.float32)

    scratch_base = scratch_ptr + bh * LC * K * Vd

    for tr in range(cur_L):
        t_g = chunk_start + tr
        k_t = tl.load(k_base + t_g * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        e_t = tl.load(e_base + t_g * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        g_t = tl.load(g_base + t_g * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_t = tl.load(v_base + t_g * Vd + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        b = tl.load(beta_base + t_g).to(tl.float32)
        gm = tl.load(gamma_base + t_g).to(tl.float32)
        c = gm * (1.0 - b)
        d_t = tl.exp(g_t)
        d_col = tl.expand_dims(d_t, 1)
        e_col = tl.expand_dims(e_t, 1)
        k_col = tl.expand_dims(k_t, 1)
        Y = S_cur * d_col
        s_vec = tl.sum(Y * e_col, axis=0)
        Z = Y - c * e_col * tl.expand_dims(s_vec, 0)
        pred = tl.sum(Z * k_col, axis=0)
        dvec = v_t - pred
        S_next = Z + k_col * tl.expand_dims(b * dvec, 0)
        scr_ptrs = scratch_base + tr * K * Vd + offs_k[:, None] * Vd + v_offs[None, :]
        tl.store(scr_ptrs, S_cur, mask=m2d)
        S_cur = S_next

    dS_base = dS_ptr + bh * K * Vd
    dS_ptrs = dS_base + offs_k[:, None] * Vd + v_offs[None, :]
    dS = tl.load(dS_ptrs, mask=m2d, other=0.0).to(tl.float32)

    for _i in range(cur_L):
        tr = cur_L - 1 - _i
        t_g = chunk_start + tr
        k_t = tl.load(k_base + t_g * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        e_t = tl.load(e_base + t_g * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        g_t = tl.load(g_base + t_g * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        q_t = tl.load(q_base + t_g * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_t = tl.load(v_base + t_g * Vd + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        do_t = tl.load(do_base + t_g * Vd + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        b = tl.load(beta_base + t_g).to(tl.float32)
        gm = tl.load(gamma_base + t_g).to(tl.float32)
        c = gm * (1.0 - b)
        d_t = tl.exp(g_t)
        d_col = tl.expand_dims(d_t, 1)
        e_col = tl.expand_dims(e_t, 1)
        k_col = tl.expand_dims(k_t, 1)
        scr_ptrs = scratch_base + tr * K * Vd + offs_k[:, None] * Vd + v_offs[None, :]
        S_prev = tl.load(scr_ptrs, mask=m2d, other=0.0).to(tl.float32)
        Y = S_prev * d_col
        s_vec = tl.sum(Y * e_col, axis=0)
        Z = Y - c * e_col * tl.expand_dims(s_vec, 0)
        pred = tl.sum(Z * k_col, axis=0)
        dvec = v_t - pred
        S_t = Z + k_col * tl.expand_dims(b * dvec, 0)

        qs = q_t * scale
        G = dS + tl.expand_dims(qs, 1) * tl.expand_dims(do_t, 0)
        h_vec = tl.sum(G * k_col, axis=0)
        dv_vec = b * h_vec
        tl.store(dv_base + t_g * Vd + v_offs, dv_vec, mask=mask_v)

        dq_part = scale * tl.sum(S_t * tl.expand_dims(do_t, 0), axis=1)
        tl.atomic_add(dq_base + t_g * K + offs_k, dq_part, mask=mask_k)

        dZ = G - tl.expand_dims(k_t * b, 1) * tl.expand_dims(h_vec, 0)
        u_vec = tl.sum(dZ * e_col, axis=0)
        dZ_s = tl.sum(dZ * tl.expand_dims(s_vec, 0), axis=1)
        Y_u = tl.sum(Y * tl.expand_dims(u_vec, 0), axis=1)
        de_part = -c * (dZ_s + Y_u)
        tl.atomic_add(de_base + t_g * K + offs_k, de_part, mask=mask_k)

        G_d = tl.sum(G * tl.expand_dims(dvec, 0), axis=1)
        Z_h = tl.sum(Z * tl.expand_dims(h_vec, 0), axis=1)
        dk_part = b * (G_d - Z_h)
        tl.atomic_add(dk_base + t_g * K + offs_k, dk_part, mask=mask_k)

        dY = dZ - c * e_col * tl.expand_dims(u_vec, 0)
        dg_part = tl.sum(Y * dY, axis=1)
        tl.atomic_add(dg_base + t_g * K + offs_k, dg_part, mask=mask_k)

        hdot = tl.sum(h_vec * dvec, axis=0)
        usdot = tl.sum(u_vec * s_vec, axis=0)
        db_part = hdot + gm * usdot
        tl.atomic_add(db_base + t_g, db_part)
        dgma_part = -(1.0 - b) * usdot
        tl.atomic_add(dgma_base + t_g, dgma_part)

        dS = d_col * dY

    tl.store(dS_ptrs, dS, mask=m2d)


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        Vd = v.shape[-1]
        if isinstance(scale, torch.Tensor):
            scale_f = float(scale.detach().to(torch.float32).reshape(-1)[0].item())
        else:
            scale_f = float(scale)
        if T == 0 or K == 0 or Vd == 0:
            out = torch.empty(B, H, T, Vd, device=q.device, dtype=torch.float32)
            ctx.meta = (B, H, T, K, Vd, scale_f, 64, 1, 16, 16, True)
            ctx.save_for_backward(q, k, v, e, g, beta, gamma,
                                  torch.empty(0, device=q.device, dtype=torch.float32))
            ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
            return out
        BH = B * H
        q_ = q.contiguous().view(BH, T, K)
        k_ = k.contiguous().view(BH, T, K)
        e_ = e.contiguous().view(BH, T, K)
        g_ = g.contiguous().view(BH, T, K)
        v_ = v.contiguous().view(BH, T, Vd)
        beta_ = beta.contiguous().view(BH, T)
        gamma_ = gamma.contiguous().view(BH, T)
        out = torch.empty(BH, T, Vd, device=q.device, dtype=torch.float32)
        BK = triton.next_power_of_2(K)
        if BK < 16:
            BK = 16
        if Vd >= 32:
            BV = 32
        else:
            BV = triton.next_power_of_2(Vd)
            if BV < 16:
                BV = 16
        while BK * BV > 4096 and BV > 8:
            BV //= 2
        LC = 64
        n_chunks = (T + LC - 1) // LC
        NCKPT = n_chunks + 1
        ckpt = torch.zeros(BH, NCKPT, K, Vd, device=q.device, dtype=torch.float32)
        n_vb = (Vd + BV - 1) // BV
        grid = (n_vb, BH)
        nw = 8 if BK * BV > 2048 else 4
        _fwd_kernel[grid](q_, k_, v_, e_, g_, beta_, gamma_, out, ckpt,
                          T, K, Vd, scale_f, LC, NCKPT,
                          BLOCK_K=BK, BLOCK_V=BV, num_warps=nw)
        ctx.save_for_backward(q_, k_, v_, e_, g_, beta_, gamma_, ckpt)
        ctx.meta = (B, H, T, K, Vd, scale_f, LC, n_chunks, NCKPT, BK, BV, n_vb)
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
        return out.view(B, H, T, Vd)

    @staticmethod
    def backward(ctx, grad_out):
        q_, k_, v_, e_, g_, beta_, gamma_, ckpt = ctx.saved_tensors
        B, H, T, K, Vd, scale_f, LC, n_chunks, NCKPT, BK, BV, n_vb = ctx.meta
        q_dt, k_dt, v_dt, e_dt, g_dt, beta_dt, gamma_dt = ctx.dtypes
        if T == 0 or K == 0 or Vd == 0:
            zq = torch.empty_like(q_)
            zk = torch.empty_like(k_)
            zv = torch.empty_like(v_)
            ze = torch.empty_like(e_)
            zg = torch.empty_like(g_)
            zb = torch.empty_like(beta_)
            zgm = torch.empty_like(gamma_)
            def _v(x, d, s):
                return x.view(s).to(d)
            return (_v(zq, q_dt, (B, H, T, K)), _v(zk, k_dt, (B, H, T, K)),
                    _v(zv, v_dt, (B, H, T, Vd)), _v(ze, e_dt, (B, H, T, K)),
                    _v(zg, g_dt, (B, H, T, K)), _v(zb, beta_dt, (B, H, T)),
                    _v(zgm, gamma_dt, (B, H, T)), None)
        BH = B * H
        do_ = grad_out.contiguous().view(BH, T, Vd).to(torch.float32).contiguous()
        dq = torch.zeros(BH, T, K, device=q_.device, dtype=torch.float32)
        dk = torch.zeros(BH, T, K, device=q_.device, dtype=torch.float32)
        de = torch.zeros(BH, T, K, device=q_.device, dtype=torch.float32)
        dg = torch.zeros(BH, T, K, device=q_.device, dtype=torch.float32)
        dv = torch.zeros(BH, T, Vd, device=q_.device, dtype=torch.float32)
        db = torch.zeros(BH, T, device=q_.device, dtype=torch.float32)
        dgm = torch.zeros(BH, T, device=q_.device, dtype=torch.float32)
        dS = torch.zeros(BH, K, Vd, device=q_.device, dtype=torch.float32)
        scratch = torch.empty(BH, LC, K, Vd, device=q_.device, dtype=torch.float32)
        grid = (n_vb, BH)
        nw = 8 if BK * BV > 2048 else 4
        for c in range(n_chunks - 1, -1, -1):
            cur_L = min(LC, T - c * LC)
            _bwd_chunk_kernel[grid](q_, k_, v_, e_, g_, beta_, gamma_, do_,
                                    ckpt, dS, scratch,
                                    dq, dk, de, dg, dv, db, dgm,
                                    T, K, Vd, scale_f, c, cur_L, LC, NCKPT,
                                    BLOCK_K=BK, BLOCK_V=BV, num_warps=nw)
        dq_o = dq.view(B, H, T, K).to(q_dt)
        dk_o = dk.view(B, H, T, K).to(k_dt)
        dv_o = dv.view(B, H, T, Vd).to(v_dt)
        de_o = de.view(B, H, T, K).to(e_dt)
        dg_o = dg.view(B, H, T, K).to(g_dt)
        db_o = db.view(B, H, T).to(beta_dt)
        dgm_o = dgm.view(B, H, T).to(gamma_dt)
        return (dq_o, dk_o, dv_o, de_o, dg_o, db_o, dgm_o, None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _Func.apply(q, k, v, e, g, beta, gamma, scale)
```