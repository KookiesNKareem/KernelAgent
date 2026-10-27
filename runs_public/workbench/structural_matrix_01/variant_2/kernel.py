import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr, cache_ptr,
    s_qb, s_qh, s_qt, s_qk,
    s_kb, s_kh, s_kt, s_kk,
    s_vb, s_vh, s_vt, s_vv,
    s_eb, s_eh, s_et, s_ek,
    s_gb, s_gh, s_gt, s_gk,
    s_betab, s_betah, s_betat,
    s_gammab, s_gammah, s_gammat,
    B, H, T, K, V, NV,
    scale,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    iv = pid % NV
    b = bh // H
    h = bh % H
    rk = tl.arange(0, BLOCK_K)
    rv = tl.arange(0, BLOCK_V)
    mask_k = rk < K
    v_idx = iv * BLOCK_V + rv
    mask_v = v_idx < V
    mask_2d = mask_k[:, None] & mask_v[None, :]

    q_base = q_ptr + b * s_qb + h * s_qh
    k_base = k_ptr + b * s_kb + h * s_kh
    v_base = v_ptr + b * s_vb + h * s_vh
    e_base = e_ptr + b * s_eb + h * s_eh
    g_base = g_ptr + b * s_gb + h * s_gh
    beta_base = beta_ptr + b * s_betab + h * s_betah
    gamma_base = gamma_ptr + b * s_gammab + h * s_gammah

    S = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)

    for t in range(T):
        q_t = tl.load(q_base + t * s_qt + rk * s_qk, mask=mask_k, other=0.0).to(tl.float32)
        k_t = tl.load(k_base + t * s_kt + rk * s_kk, mask=mask_k, other=0.0).to(tl.float32)
        e_t = tl.load(e_base + t * s_et + rk * s_ek, mask=mask_k, other=0.0).to(tl.float32)
        g_t = tl.load(g_base + t * s_gt + rk * s_gk, mask=mask_k, other=0.0).to(tl.float32)
        v_t = tl.load(v_base + t * s_vt + v_idx * s_vv, mask=mask_v, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_base + t * s_betat).to(tl.float32)
        gamma_t = tl.load(gamma_base + t * s_gammat).to(tl.float32)

        decay = tl.exp(g_t)
        S = S * tl.expand_dims(decay, 1)
        k_col = tl.expand_dims(k_t, 1)
        pred = tl.sum(S * k_col, 0)
        delta = (v_t - pred) * beta_t
        S = S + k_col * tl.expand_dims(delta, 0)
        e_col = tl.expand_dims(e_t, 1)
        s_vec = tl.sum(S * e_col, 0)
        ge = e_t * gamma_t
        S = S - tl.expand_dims(ge, 1) * tl.expand_dims(s_vec, 0)

        qs = q_t * scale
        o_t = tl.sum(S * tl.expand_dims(qs, 1), 0)
        o_off = ((b * H + h) * T + t) * V + v_idx
        tl.store(out_ptr + o_off, o_t, mask=mask_v)
        c_ptrs = cache_ptr + (((b * H + h) * T + t) * K * V) + rk[:, None] * V + v_idx[None, :]
        tl.store(c_ptrs, S, mask=mask_2d)


@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, dO_ptr, cache_ptr,
    dq_ptr, dk_ptr, de_ptr, dg_ptr, dv_ptr, dbeta_ptr, dgamma_ptr,
    s_qb, s_qh, s_qt, s_qk,
    s_kb, s_kh, s_kt, s_kk,
    s_vb, s_vh, s_vt, s_vv,
    s_eb, s_eh, s_et, s_ek,
    s_gb, s_gh, s_gt, s_gk,
    s_betab, s_betah, s_betat,
    s_gammab, s_gammah, s_gammat,
    s_dOb, s_dOh, s_dOt, s_dOv,
    B, H, T, K, V, NV,
    scale,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    iv = pid % NV
    b = bh // H
    h = bh % H
    rk = tl.arange(0, BLOCK_K)
    rv = tl.arange(0, BLOCK_V)
    mask_k = rk < K
    v_idx = iv * BLOCK_V + rv
    mask_v = v_idx < V
    mask_2d = mask_k[:, None] & mask_v[None, :]

    q_base = q_ptr + b * s_qb + h * s_qh
    k_base = k_ptr + b * s_kb + h * s_kh
    v_base = v_ptr + b * s_vb + h * s_vh
    e_base = e_ptr + b * s_eb + h * s_eh
    g_base = g_ptr + b * s_gb + h * s_gh
    beta_base = beta_ptr + b * s_betab + h * s_betah
    gamma_base = gamma_ptr + b * s_gammab + h * s_gammah
    dO_base = dO_ptr + b * s_dOb + h * s_dOh

    Gprop = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)

    for tt in range(T):
        t = T - 1 - tt
        q_t = tl.load(q_base + t * s_qt + rk * s_qk, mask=mask_k, other=0.0).to(tl.float32)
        k_t = tl.load(k_base + t * s_kt + rk * s_kk, mask=mask_k, other=0.0).to(tl.float32)
        e_t = tl.load(e_base + t * s_et + rk * s_ek, mask=mask_k, other=0.0).to(tl.float32)
        g_t = tl.load(g_base + t * s_gt + rk * s_gk, mask=mask_k, other=0.0).to(tl.float32)
        v_t = tl.load(v_base + t * s_vt + v_idx * s_vv, mask=mask_v, other=0.0).to(tl.float32)
        dO_t = tl.load(dO_base + t * s_dOt + v_idx * s_dOv, mask=mask_v, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_base + t * s_betat).to(tl.float32)
        gamma_t = tl.load(gamma_base + t * s_gammat).to(tl.float32)

        c_base_cur = (((b * H + h) * T + t) * K * V)
        S_cur = tl.load(cache_ptr + c_base_cur + rk[:, None] * V + v_idx[None, :], mask=mask_2d, other=0.0)

        if t > 0:
            c_base_prev = (((b * H + h) * T + (t - 1)) * K * V)
            S_prev = tl.load(cache_ptr + c_base_prev + rk[:, None] * V + v_idx[None, :], mask=mask_2d, other=0.0)
        else:
            S_prev = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)

        dvec = tl.exp(g_t)
        S_d = S_prev * tl.expand_dims(dvec, 1)
        k_col = tl.expand_dims(k_t, 1)
        pred = tl.sum(S_d * k_col, 0)
        diff = v_t - pred
        delta = diff * beta_t
        S_mid = S_d + k_col * tl.expand_dims(delta, 0)
        e_col = tl.expand_dims(e_t, 1)
        s_vec = tl.sum(S_mid * e_col, 0)

        qs = q_t * scale
        Gcur = Gprop + tl.expand_dims(qs, 1) * tl.expand_dims(dO_t, 0)
        h_vec = tl.sum(Gcur * e_col, 0)
        ge = e_t * gamma_t
        Gmid = Gcur - tl.expand_dims(ge, 1) * tl.expand_dims(h_vec, 0)
        p_vec = tl.sum(Gmid * k_col, 0)
        kb = k_t * beta_t
        Gd = Gmid - tl.expand_dims(kb, 1) * tl.expand_dims(p_vec, 0)
        Gprop = Gd * tl.expand_dims(dvec, 1)

        # dq: out = sum_k S_cur[k,v] * q[k]*scale -> d/dq = scale*sum_v S_cur*dO
        dq_part = tl.sum(S_cur * tl.expand_dims(dO_t, 0), 1) * scale
        dq_ptrs = dq_ptr + (((b * H + h) * T + t) * K + rk)
        tl.atomic_add(dq_ptrs, dq_part, mask=mask_k)

        dv_vec = p_vec * beta_t
        dv_off = ((b * H + h) * T + t) * V + v_idx
        tl.store(dv_ptr + dv_off, dv_vec, mask=mask_v)

        dbeta_part = tl.sum(p_vec * diff, 0)
        tl.atomic_add(dbeta_ptr + (b * H + h) * T + t, dbeta_part)

        dgamma_part = -tl.sum(h_vec * s_vec, 0)
        tl.atomic_add(dgamma_ptr + (b * H + h) * T + t, dgamma_part)

        tmp1 = tl.sum(Gmid * tl.expand_dims(diff, 0), 1)
        tmp2 = tl.sum(S_d * tl.expand_dims(p_vec, 0), 1)
        dk_part = (tmp1 - tmp2) * beta_t
        dk_ptrs = dk_ptr + (((b * H + h) * T + t) * K + rk)
        tl.atomic_add(dk_ptrs, dk_part, mask=mask_k)

        tmp3 = tl.sum(Gcur * tl.expand_dims(s_vec, 0), 1)
        tmp4 = tl.sum(S_mid * tl.expand_dims(h_vec, 0), 1)
        de_part = (tmp3 + tmp4) * (-gamma_t)
        de_ptrs = de_ptr + (((b * H + h) * T + t) * K + rk)
        tl.atomic_add(de_ptrs, de_part, mask=mask_k)

        tmp5 = tl.sum(Gd * S_prev, 1)
        dg_part = tmp5 * dvec
        dg_ptrs = dg_ptr + (((b * H + h) * T + t) * K + rk)
        tl.atomic_add(dg_ptrs, dg_part, mask=mask_k)


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        device = q.device
        out = torch.empty(B, H, T, V, dtype=torch.float32, device=device)
        if T == 0 or K == 0 or V == 0 or out.numel() == 0:
            cache = torch.empty(B, H, T, K, V, dtype=torch.float32, device=device)
            ctx.save_for_backward(q, k, v, e, g, beta, gamma, cache)
            ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
            ctx.scale = float(scale)
            return out
        BLOCK_K = triton.next_power_of_2(K)
        BLOCK_V = triton.next_power_of_2(V) if V <= 32 else 32
        NV = (V + BLOCK_V - 1) // BLOCK_V
        cache = torch.empty(B, H, T, K, V, dtype=torch.float32, device=device)
        grid = (B * H * NV,)
        scale_f = float(scale)
        _fwd_kernel[grid](
            q, k, v, e, g, beta, gamma, out, cache,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            e.stride(0), e.stride(1), e.stride(2), e.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            B, H, T, K, V, NV,
            scale_f, BLOCK_K, BLOCK_V, num_warps=4,
        )
        ctx.save_for_backward(q, k, v, e, g, beta, gamma, cache)
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
        ctx.scale = scale_f
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, e, g, beta, gamma, cache = ctx.saved_tensors
        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        scale_f = ctx.scale
        needs = ctx.needs_input_grad
        if grad_output.dtype != torch.float32:
            grad_output = grad_output.to(torch.float32)
        device = q.device
        if T == 0 or K == 0 or V == 0:
            def _z(shape, dtype):
                if shape.numel() == 0:
                    return torch.empty(shape, dtype=dtype, device=device)
                return torch.zeros(shape, dtype=torch.float32, device=device).to(dtype)
            gq = _z(torch.Size([B, H, T, K]), q.dtype) if needs[0] else None
            gk = _z(torch.Size([B, H, T, K]), k.dtype) if needs[1] else None
            gv = _z(torch.Size([B, H, T, V]), v.dtype) if needs[2] else None
            ge = _z(torch.Size([B, H, T, K]), e.dtype) if needs[3] else None
            gg = _z(torch.Size([B, H, T, K]), g.dtype) if needs[4] else None
            gb = _z(torch.Size([B, H, T]), beta.dtype) if needs[5] else None
            gg2 = _z(torch.Size([B, H, T]), gamma.dtype) if needs[6] else None
            return gq, gk, gv, ge, gg, gb, gg2, None
        BLOCK_K = triton.next_power_of_2(K)
        BLOCK_V = triton.next_power_of_2(V) if V <= 32 else 32
        NV = (V + BLOCK_V - 1) // BLOCK_V
        dq = torch.zeros(B, H, T, K, dtype=torch.float32, device=device)
        dk = torch.zeros(B, H, T, K, dtype=torch.float32, device=device)
        de = torch.zeros(B, H, T, K, dtype=torch.float32, device=device)
        dg = torch.zeros(B, H, T, K, dtype=torch.float32, device=device)
        dbeta = torch.zeros(B, H, T, dtype=torch.float32, device=device)
        dgamma = torch.zeros(B, H, T, dtype=torch.float32, device=device)
        dv = torch.zeros(B, H, T, V, dtype=torch.float32, device=device)
        grid = (B * H * NV,)
        _bwd_kernel[grid](
            q, k, v, e, g, beta, gamma, grad_output, cache,
            dq, dk, de, dg, dv, dbeta, dgamma,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            e.stride(0), e.stride(1), e.stride(2), e.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            B, H, T, K, V, NV,
            scale_f, BLOCK_K, BLOCK_V, num_warps=4,
        )
        out = []
        for tnsr, need, dtype in zip(
            [dq, dk, dv, de, dg, dbeta, dgamma],
            needs[:7],
            [q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype],
        ):
            if not need:
                out.append(None)
            else:
                out.append(tnsr.to(dtype) if tnsr.dtype != dtype else tnsr)
        out.append(None)
        return tuple(out)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _Func.apply(q, k, v, e, g, beta, gamma, scale)
