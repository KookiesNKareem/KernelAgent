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
    s_gamab, s_gamah, s_gamat,
    H, T, K_DIM, V_DIM,
    scale: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    mask_k = off_k < K_DIM
    mask_v = off_v < V_DIM
    mask_2d = tl.expand_dims(mask_k, 1) & tl.expand_dims(mask_v, 0)
    off_k_2d = tl.expand_dims(off_k, 1)
    off_v_2d = tl.expand_dims(off_v, 0)
    state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    q_base = q_ptr + b * s_qb + h * s_qh
    k_base = k_ptr + b * s_kb + h * s_kh
    v_base = v_ptr + b * s_vb + h * s_vh
    e_base = e_ptr + b * s_eb + h * s_eh
    g_base = g_ptr + b * s_gb + h * s_gh
    beta_base = beta_ptr + b * s_betab + h * s_betah
    gamma_base = gamma_ptr + b * s_gamab + h * s_gamah
    HTV = H * T * V_DIM
    TV = T * V_DIM
    HTKV = H * T * K_DIM * V_DIM
    TKV = T * K_DIM * V_DIM
    KV = K_DIM * V_DIM
    for t in range(T):
        qv = tl.load(q_base + t * s_qt + off_k * s_qk, mask=mask_k, other=0.0)
        kv = tl.load(k_base + t * s_kt + off_k * s_kk, mask=mask_k, other=0.0)
        vv = tl.load(v_base + t * s_vt + off_v * s_vv, mask=mask_v, other=0.0)
        ev = tl.load(e_base + t * s_et + off_k * s_ek, mask=mask_k, other=0.0)
        gv = tl.load(g_base + t * s_gt + off_k * s_gk, mask=mask_k, other=0.0)
        beta_val = tl.load(beta_base + t * s_betat)
        gamma_val = tl.load(gamma_base + t * s_gamat)
        decay = tl.exp(gv)
        state = state * tl.expand_dims(decay, 1)
        e_col = tl.expand_dims(ev, 1)
        a_vec = tl.sum(state * e_col, 0)
        state = state - gamma_val * e_col * tl.expand_dims(a_vec, 0)
        k_col = tl.expand_dims(kv, 1)
        p_vec = tl.sum(state * k_col, 0)
        diff = beta_val * (vv - p_vec)
        state = state + k_col * tl.expand_dims(diff, 0)
        o_vec = tl.sum(state * tl.expand_dims(qv * scale, 1), 0)
        o_ptrs = o_ptr + b * HTV + h * TV + t * V_DIM + off_v
        tl.store(o_ptrs, o_vec, mask=mask_v)
        s_ptrs = s_ptr + b * HTKV + h * TKV + t * KV + off_k_2d * V_DIM + off_v_2d
        tl.store(s_ptrs, state, mask=mask_2d)


@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, s_ptr, do_ptr,
    dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
    s_qb, s_qh, s_qt, s_qk,
    s_kb, s_kh, s_kt, s_kk,
    s_vb, s_vh, s_vt, s_vv,
    s_eb, s_eh, s_et, s_ek,
    s_gb, s_gh, s_gt, s_gk,
    s_betab, s_betah, s_betat,
    s_gamab, s_gamah, s_gamat,
    s_dob, s_doh, s_dot, s_dov,
    H, T, K_DIM, V_DIM,
    scale: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    mask_k = off_k < K_DIM
    mask_v = off_v < V_DIM
    mask_2d = tl.expand_dims(mask_k, 1) & tl.expand_dims(mask_v, 0)
    off_k_2d = tl.expand_dims(off_k, 1)
    off_v_2d = tl.expand_dims(off_v, 0)
    F = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    q_base = q_ptr + b * s_qb + h * s_qh
    k_base = k_ptr + b * s_kb + h * s_kh
    v_base = v_ptr + b * s_vb + h * s_vh
    e_base = e_ptr + b * s_eb + h * s_eh
    g_base = g_ptr + b * s_gb + h * s_gh
    beta_base = beta_ptr + b * s_betab + h * s_betah
    gamma_base = gamma_ptr + b * s_gamab + h * s_gamah
    do_base = do_ptr + b * s_dob + h * s_doh
    HTKV = H * T * K_DIM * V_DIM
    TKV = T * K_DIM * V_DIM
    KV = K_DIM * V_DIM
    s_bh = s_ptr + b * HTKV + h * TKV
    HTK = H * T * K_DIM
    TK = T * K_DIM
    HTV = H * T * V_DIM
    TV = T * V_DIM
    HTT = H * T
    TT = T
    for ti in range(T):
        t = T - 1 - ti
        qv = tl.load(q_base + t * s_qt + off_k * s_qk, mask=mask_k, other=0.0)
        kv = tl.load(k_base + t * s_kt + off_k * s_kk, mask=mask_k, other=0.0)
        vv = tl.load(v_base + t * s_vt + off_v * s_vv, mask=mask_v, other=0.0)
        ev = tl.load(e_base + t * s_et + off_k * s_ek, mask=mask_k, other=0.0)
        gv = tl.load(g_base + t * s_gt + off_k * s_gk, mask=mask_k, other=0.0)
        beta_val = tl.load(beta_base + t * s_betat)
        gamma_val = tl.load(gamma_base + t * s_gamat)
        do_vec = tl.load(do_base + t * s_dot + off_v * s_dov, mask=mask_v, other=0.0)
        s_ptrs_t = s_bh + t * KV + off_k_2d * V_DIM + off_v_2d
        St = tl.load(s_ptrs_t, mask=mask_2d, other=0.0)
        Sprev = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        if t > 0:
            s_ptrs_m = s_bh + (t - 1) * KV + off_k_2d * V_DIM + off_v_2d
            Sprev = tl.load(s_ptrs_m, mask=mask_2d, other=0.0)
        dvec = tl.exp(gv)
        S1 = Sprev * tl.expand_dims(dvec, 1)
        e_col = tl.expand_dims(ev, 1)
        a_vec = tl.sum(S1 * e_col, 0)
        S2 = S1 - gamma_val * e_col * tl.expand_dims(a_vec, 0)
        k_col = tl.expand_dims(kv, 1)
        p_vec = tl.sum(S2 * k_col, 0)
        delta = beta_val * (vv - p_vec)
        DS = F + scale * tl.expand_dims(qv, 1) * tl.expand_dims(do_vec, 0)
        dqv = scale * tl.sum(St * tl.expand_dims(do_vec, 0), 1)
        dd = tl.sum(DS * k_col, 0)
        dp = -beta_val * dd
        dk1 = tl.sum(DS * tl.expand_dims(delta, 0), 1)
        dk2 = tl.sum(S2 * tl.expand_dims(dp, 0), 1)
        dkv = dk1 + dk2
        dvv = beta_val * dd
        dbeta_val = tl.sum(dd * (vv - p_vec), 0)
        dS2 = DS + k_col * tl.expand_dims(dp, 0)
        bvec = tl.sum(dS2 * e_col, 0)
        dgamma_val = -tl.sum(a_vec * bvec, 0)
        dev = -gamma_val * (tl.sum(dS2 * tl.expand_dims(a_vec, 0), 1) + tl.sum(S1 * tl.expand_dims(bvec, 0), 1))
        dS1 = dS2 - gamma_val * e_col * tl.expand_dims(bvec, 0)
        dgv = tl.sum(dS1 * S1, 1)
        F = dS1 * tl.expand_dims(dvec, 1)
        dq_ptrs = dq_ptr + b * HTK + h * TK + t * K_DIM + off_k
        dk_ptrs = dk_ptr + b * HTK + h * TK + t * K_DIM + off_k
        de_ptrs = de_ptr + b * HTK + h * TK + t * K_DIM + off_k
        dg_ptrs = dg_ptr + b * HTK + h * TK + t * K_DIM + off_k
        dv_ptrs = dv_ptr + b * HTV + h * TV + t * V_DIM + off_v
        tl.store(dq_ptrs, dqv, mask=mask_k)
        tl.store(dk_ptrs, dkv, mask=mask_k)
        tl.store(de_ptrs, dev, mask=mask_k)
        tl.store(dg_ptrs, dgv, mask=mask_k)
        tl.store(dv_ptrs, dvv, mask=mask_v)
        tl.store(dbeta_ptr + b * HTT + h * TT + t, dbeta_val)
        tl.store(dgamma_ptr + b * HTT + h * TT + t, dgamma_val)


class _F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        def _prom(x):
            return x.to(torch.promote_types(x.dtype, torch.float32))
        q32 = _prom(q)
        k32 = _prom(k)
        v32 = _prom(v)
        e32 = _prom(e)
        g32 = _prom(g)
        beta32 = _prom(beta)
        gamma32 = _prom(gamma)
        B, H, T, K = q.shape
        V_DIM = v.shape[-1]
        out = torch.empty((B, H, T, V_DIM), device=q.device, dtype=torch.float32)
        S = torch.empty((B, H, T, K, V_DIM), device=q.device, dtype=torch.float32)
        BLOCK_K = triton.next_power_of_2(K)
        BLOCK_V = triton.next_power_of_2(V_DIM)
        grid = (B * H,)
        _fwd_kernel[grid](
            q32, k32, v32, e32, g32, beta32, gamma32, out, S,
            q32.stride(0), q32.stride(1), q32.stride(2), q32.stride(3),
            k32.stride(0), k32.stride(1), k32.stride(2), k32.stride(3),
            v32.stride(0), v32.stride(1), v32.stride(2), v32.stride(3),
            e32.stride(0), e32.stride(1), e32.stride(2), e32.stride(3),
            g32.stride(0), g32.stride(1), g32.stride(2), g32.stride(3),
            beta32.stride(0), beta32.stride(1), beta32.stride(2),
            gamma32.stride(0), gamma32.stride(1), gamma32.stride(2),
            H, T, K, V_DIM, float(scale), BLOCK_K, BLOCK_V,
            num_warps=4, num_stages=1,
        )
        ctx.save_for_backward(q32, k32, v32, e32, g32, beta32, gamma32, S)
        ctx.B = B
        ctx.H = H
        ctx.T = T
        ctx.K = K
        ctx.V = V_DIM
        ctx.scale = float(scale)
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q32, k32, v32, e32, g32, beta32, gamma32, S = ctx.saved_tensors
        B = ctx.B
        H = ctx.H
        T = ctx.T
        K = ctx.K
        V_DIM = ctx.V
        scale = ctx.scale
        def _prom(x):
            return x.to(torch.promote_types(x.dtype, torch.float32))
        go32 = _prom(grad_out.contiguous() if grad_out.is_cuda else grad_out)
        # ensure fp32 contiguous grad input for simplicity; keep strides general
        if go32.dtype != torch.float32:
            go32 = go32.to(torch.float32)
        dq32 = torch.empty((B, H, T, K), device=q32.device, dtype=torch.float32)
        dk32 = torch.empty((B, H, T, K), device=q32.device, dtype=torch.float32)
        dv32 = torch.empty((B, H, T, V_DIM), device=q32.device, dtype=torch.float32)
        de32 = torch.empty((B, H, T, K), device=q32.device, dtype=torch.float32)
        dg32 = torch.empty((B, H, T, K), device=q32.device, dtype=torch.float32)
        dbeta32 = torch.empty((B, H, T), device=q32.device, dtype=torch.float32)
        dgamma32 = torch.empty((B, H, T), device=q32.device, dtype=torch.float32)
        BLOCK_K = triton.next_power_of_2(K)
        BLOCK_V = triton.next_power_of_2(V_DIM)
        grid = (B * H,)
        _bwd_kernel[grid](
            q32, k32, v32, e32, g32, beta32, gamma32, S, go32,
            dq32, dk32, dv32, de32, dg32, dbeta32, dgamma32,
            q32.stride(0), q32.stride(1), q32.stride(2), q32.stride(3),
            k32.stride(0), k32.stride(1), k32.stride(2), k32.stride(3),
            v32.stride(0), v32.stride(1), v32.stride(2), v32.stride(3),
            e32.stride(0), e32.stride(1), e32.stride(2), e32.stride(3),
            g32.stride(0), g32.stride(1), g32.stride(2), g32.stride(3),
            beta32.stride(0), beta32.stride(1), beta32.stride(2),
            gamma32.stride(0), gamma32.stride(1), gamma32.stride(2),
            go32.stride(0), go32.stride(1), go32.stride(2), go32.stride(3),
            H, T, K, V_DIM, float(scale), BLOCK_K, BLOCK_V,
            num_warps=4, num_stages=1,
        )
        dts = ctx.dtypes
        outs = (dq32, dk32, dv32, de32, dg32, dbeta32, dgamma32)
        casted = tuple(o.to(dt) if o.dtype != dt else o for o, dt in zip(outs, dts))
        return casted[0], casted[1], casted[2], casted[3], casted[4], casted[5], casted[6], None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _F.apply(q, k, v, e, g, beta, gamma, scale)
```
I keep the whole KxV state in registers with one Triton program per BH slice, fusing decay, Householder erase, delta write and readout forward, and saving the S trajectory for training. Backward replays the step in reverse with an adjoint state F, recomputing S1/S2/a/p per step and applying the transposed erase/decay to propagate F while emitting all seven gradients, then casting to input dtypes. I am unsure about unrolled versus pipelined loop code size for T=65 and about exact fp32 summation-order tolerance versus torch autograd.