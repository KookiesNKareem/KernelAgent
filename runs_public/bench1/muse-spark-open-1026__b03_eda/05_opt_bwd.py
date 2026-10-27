import torch
import triton
import triton.language as tl

# ---------- fast path for K=V=128 contiguous (benchmark) ----------
@triton.jit
def fwd_fast_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr, states_ptr,
                    BH, T, scale):
    pid = tl.program_id(0)
    bh = pid
    offs_k = tl.arange(0, 128)
    offs_v0 = tl.arange(0, 64)
    offs_v1 = tl.arange(0, 64) + 64
    S0 = tl.zeros((128, 64), dtype=tl.float32)
    S1 = tl.zeros((128, 64), dtype=tl.float32)
    for t in range(T):
        b = bh * T + t
        g = tl.load(g_ptr + b)
        beta = tl.load(beta_ptr + b)
        gamma = tl.load(gamma_ptr + b)
        d = tl.exp(g)
        S0 = S0 * d
        S1 = S1 * d
        e = tl.load(e_ptr + b * 128 + offs_k).to(tl.float32)
        se0 = tl.sum(e[:, None] * S0, axis=0)
        se1 = tl.sum(e[:, None] * S1, axis=0)
        S0 = S0 - gamma * (e[:, None] * se0[None, :])
        S1 = S1 - gamma * (e[:, None] * se1[None, :])
        k = tl.load(k_ptr + b * 128 + offs_k).to(tl.float32)
        sk0 = tl.sum(k[:, None] * S0, axis=0)
        sk1 = tl.sum(k[:, None] * S1, axis=0)
        v0 = tl.load(v_ptr + b * 128 + offs_v0).to(tl.float32)
        v1 = tl.load(v_ptr + b * 128 + offs_v1).to(tl.float32)
        S0 = S0 + beta * (k[:, None] * (v0 - sk0)[None, :])
        S1 = S1 + beta * (k[:, None] * (v1 - sk1)[None, :])
        q = tl.load(q_ptr + b * 128 + offs_k).to(tl.float32)
        o0 = tl.sum(q[:, None] * S0, axis=0) * scale
        o1 = tl.sum(q[:, None] * S1, axis=0) * scale
        tl.store(o_ptr + b * 128 + offs_v0, o0)
        tl.store(o_ptr + b * 128 + offs_v1, o1)
        base = states_ptr + (bh * T + t) * 16384
        tl.store(base + offs_k[:, None] * 128 + offs_v0[None, :], S0)
        tl.store(base + offs_k[:, None] * 128 + offs_v1[None, :], S1)

# fused bwd fast: BH program, writes grads directly, no reduction
@triton.jit
def bwd_fast_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, do_ptr,
                    states_ptr,
                    dq_ptr, dk_ptr, de_ptr, dv_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
                    BH, T, scale):
    pid = tl.program_id(0)
    bh = pid
    offs_k = tl.arange(0, 128)
    offs_v0 = tl.arange(0, 64)
    offs_v1 = tl.arange(0, 64) + 64
    P0 = tl.zeros((128, 64), dtype=tl.float32)
    P1 = tl.zeros((128, 64), dtype=tl.float32)
    for idx in range(T):
        t = T - 1 - idx
        b = bh * T + t
        g = tl.load(g_ptr + b)
        beta = tl.load(beta_ptr + b)
        gamma = tl.load(gamma_ptr + b)
        d = tl.exp(g)
        q = tl.load(q_ptr + b * 128 + offs_k).to(tl.float32)
        k = tl.load(k_ptr + b * 128 + offs_k).to(tl.float32)
        e = tl.load(e_ptr + b * 128 + offs_k).to(tl.float32)
        do0 = tl.load(do_ptr + b * 128 + offs_v0).to(tl.float32)
        do1 = tl.load(do_ptr + b * 128 + offs_v1).to(tl.float32)
        v0 = tl.load(v_ptr + b * 128 + offs_v0).to(tl.float32)
        v1 = tl.load(v_ptr + b * 128 + offs_v1).to(tl.float32)
        # S_prev
        if t == 0:
            Sp0 = tl.zeros((128, 64), dtype=tl.float32)
            Sp1 = tl.zeros((128, 64), dtype=tl.float32)
        else:
            basep = states_ptr + (bh * T + t - 1) * 16384
            Sp0 = tl.load(basep + offs_k[:, None] * 128 + offs_v0[None, :])
            Sp1 = tl.load(basep + offs_k[:, None] * 128 + offs_v1[None, :])
        S0a = Sp0 * d
        S1a = Sp1 * d
        se0 = tl.sum(e[:, None] * S0a, axis=0)
        se1 = tl.sum(e[:, None] * S1a, axis=0)
        S0b = S0a - gamma * (e[:, None] * se0[None, :])
        S1b = S1a - gamma * (e[:, None] * se1[None, :])
        sk0 = tl.sum(k[:, None] * S0b, axis=0)
        sk1 = tl.sum(k[:, None] * S1b, axis=0)
        r0 = v0 - sk0
        r1 = v1 - sk1
        St0 = S0b + beta * (k[:, None] * r0[None, :])
        St1 = S1b + beta * (k[:, None] * r1[None, :])

        C0 = scale * (q[:, None] * do0[None, :])
        C1 = scale * (q[:, None] * do1[None, :])
        D0 = C0 + P0
        D1 = C1 + P1

        # dq
        dq0 = scale * tl.sum(St0 * do0[None, :], axis=1)
        dq1 = scale * tl.sum(St1 * do1[None, :], axis=1)
        dq = dq0 + dq1
        tl.store(dq_ptr + b * 128 + offs_k, dq)

        w0 = tl.sum(k[:, None] * D0, axis=0)
        w1 = tl.sum(k[:, None] * D1, axis=0)
        # dv
        tl.store(dv_ptr + b * 128 + offs_v0, beta * w0)
        tl.store(dv_ptr + b * 128 + offs_v1, beta * w1)
        dbeta = tl.sum(w0 * r0, axis=0) + tl.sum(w1 * r1, axis=0)
        tl.store(dbeta_ptr + b, dbeta)
        # dk
        Dr0 = tl.sum(D0 * r0[None, :], axis=1)
        Dr1 = tl.sum(D1 * r1[None, :], axis=1)
        Sw0 = tl.sum(S0b * w0[None, :], axis=1)
        Sw1 = tl.sum(S1b * w1[None, :], axis=1)
        dk = beta * ((Dr0 + Dr1) - (Sw0 + Sw1))
        tl.store(dk_ptr + b * 128 + offs_k, dk)
        # propagate beta
        D0a = D0 - beta * (k[:, None] * w0[None, :])
        D1a = D1 - beta * (k[:, None] * w1[None, :])
        we0 = tl.sum(e[:, None] * D0a, axis=0)
        we1 = tl.sum(e[:, None] * D1a, axis=0)
        dgamma = -tl.sum(we0 * se0, axis=0) - tl.sum(we1 * se1, axis=0)
        tl.store(dgamma_ptr + b, dgamma)
        # de
        tmp0 = tl.sum(D0a * se0[None, :], axis=1)
        tmp1 = tl.sum(D1a * se1[None, :], axis=1)
        tmp2 = tl.sum(S0a * we0[None, :], axis=1)
        tmp3 = tl.sum(S1a * we1[None, :], axis=1)
        de = -gamma * ((tmp0 + tmp1) + (tmp2 + tmp3))
        tl.store(de_ptr + b * 128 + offs_k, de)
        D0b = D0a - gamma * (e[:, None] * we0[None, :])
        D1b = D1a - gamma * (e[:, None] * we1[None, :])
        dg = tl.sum(D0b * S0a) + tl.sum(D1b * S1a)
        tl.store(dg_ptr + b, dg)
        P0 = D0b * d
        P1 = D1b * d

# ---------- generic strided path (correctness) ----------
@triton.jit
def fwd_generic(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr, states_ptr,
                B, H, T, K, V, scale,
                q_sb, q_sh, q_st, q_sk,
                k_sb, k_sh, k_st, k_sk,
                v_sb, v_sh, v_st, v_sv,
                e_sb, e_sh, e_st, e_sk,
                g_sb, g_sh, g_st, b_sb, b_sh, b_st, ga_sb, ga_sh, ga_st,
                o_sb, o_sh, o_st, o_sv,
                BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, VBLOCKS: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // VBLOCKS
    vb = pid % VBLOCKS
    b = bh // H
    h = bh % H
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < V
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    for t in range(T):
        g = tl.load(g_ptr + b * g_sb + h * g_sh + t * g_st)
        beta = tl.load(beta_ptr + b * b_sb + h * b_sh + t * b_st)
        gamma = tl.load(gamma_ptr + b * ga_sb + h * ga_sh + t * ga_st)
        d = tl.exp(g)
        S = S * d
        e = tl.load(e_ptr + b * e_sb + h * e_sh + t * e_st + offs_k * e_sk, mask=mask_k, other=0.0).to(tl.float32)
        se = tl.sum(e[:, None] * S, axis=0)
        S = S - gamma * (e[:, None] * se[None, :])
        k = tl.load(k_ptr + b * k_sb + h * k_sh + t * k_st + offs_k * k_sk, mask=mask_k, other=0.0).to(tl.float32)
        sk = tl.sum(k[:, None] * S, axis=0)
        v = tl.load(v_ptr + b * v_sb + h * v_sh + t * v_st + offs_v * v_sv, mask=mask_v, other=0.0).to(tl.float32)
        S = S + beta * (k[:, None] * (v - sk)[None, :])
        q = tl.load(q_ptr + b * q_sb + h * q_sh + t * q_st + offs_k * q_sk, mask=mask_k, other=0.0).to(tl.float32)
        o = tl.sum(q[:, None] * S, axis=0) * scale
        tl.store(o_ptr + b * o_sb + h * o_sh + t * o_st + offs_v * o_sv, o, mask=mask_v)
        base = states_ptr + (bh * T + t) * K * V
        tl.store(base + offs_k[:, None] * V + offs_v[None, :], S, mask=mask_k[:, None] & mask_v[None, :])

@triton.jit
def bwd_generic(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, do_ptr,
                states_ptr, dv_ptr,
                dq_part_ptr, dk_part_ptr, de_part_ptr, dbeta_part_ptr, dgamma_part_ptr, dg_part_ptr,
                B, H, T, K, V, scale,
                q_sb, q_sh, q_st, q_sk,
                k_sb, k_sh, k_st, k_sk,
                v_sb, v_sh, v_st, v_sv,
                e_sb, e_sh, e_st, e_sk,
                g_sb, g_sh, g_st, b_sb, b_sh, b_st, ga_sb, ga_sh, ga_st,
                do_sb, do_sh, do_st, do_sv,
                dv_sb, dv_sh, dv_st, dv_sv,
                BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, VBLOCKS: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // VBLOCKS
    vb = pid % VBLOCKS
    b = bh // H
    h = bh % H
    offs_k = tl.arange(0, BLOCK_K)
    offs_bv = tl.arange(0, BLOCK_V)
    offs_v = vb * BLOCK_V + offs_bv
    mask_k = offs_k < K
    mask_v = offs_v < V
    P = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    for idx in range(T):
        t = T - 1 - idx
        g = tl.load(g_ptr + b * g_sb + h * g_sh + t * g_st)
        beta = tl.load(beta_ptr + b * b_sb + h * b_sh + t * b_st)
        gamma = tl.load(gamma_ptr + b * ga_sb + h * ga_sh + t * ga_st)
        d = tl.exp(g)
        q = tl.load(q_ptr + b * q_sb + h * q_sh + t * q_st + offs_k * q_sk, mask=mask_k, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + b * k_sb + h * k_sh + t * k_st + offs_k * k_sk, mask=mask_k, other=0.0).to(tl.float32)
        e = tl.load(e_ptr + b * e_sb + h * e_sh + t * e_st + offs_k * e_sk, mask=mask_k, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + b * v_sb + h * v_sh + t * v_st + offs_v * v_sv, mask=mask_v, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + b * do_sb + h * do_sh + t * do_st + offs_v * do_sv, mask=mask_v, other=0.0).to(tl.float32)
        if t == 0:
            Sp = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        else:
            basep = states_ptr + (bh * T + t - 1) * K * V
            Sp = tl.load(basep + offs_k[:, None] * V + offs_v[None, :], mask=mask_k[:, None] & mask_v[None, :], other=0.0)
        S0 = Sp * d
        se = tl.sum(e[:, None] * S0, axis=0)
        S1 = S0 - gamma * (e[:, None] * se[None, :])
        sk = tl.sum(k[:, None] * S1, axis=0)
        r = v - sk
        St = S1 + beta * (k[:, None] * r[None, :])
        C = scale * (q[:, None] * do[None, :])
        D = C + P
        dq = scale * tl.sum(St * do[None, :], axis=1)
        w = tl.sum(k[:, None] * D, axis=0)
        tl.store(dv_ptr + b * dv_sb + h * dv_sh + t * dv_st + offs_v * dv_sv, beta * w, mask=mask_v)
        dbeta = tl.sum(w * r, axis=0)
        Dr = tl.sum(D * r[None, :], axis=1)
        Sw = tl.sum(S1 * w[None, :], axis=1)
        dk = beta * (Dr - Sw)
        D1 = D - beta * (k[:, None] * w[None, :])
        we = tl.sum(e[:, None] * D1, axis=0)
        dgamma = -tl.sum(we * se, axis=0)
        De = tl.sum(D1 * se[None, :], axis=1)
        Sw2 = tl.sum(S0 * we[None, :], axis=1)
        de = -gamma * (De + Sw2)
        D0 = D1 - gamma * (e[:, None] * we[None, :])
        dg = tl.sum(D0 * S0)
        off = (bh * T + t) * VBLOCKS + vb
        tl.store(dq_part_ptr + off * K + offs_k, dq, mask=mask_k)
        tl.store(dk_part_ptr + off * K + offs_k, dk, mask=mask_k)
        tl.store(de_part_ptr + off * K + offs_k, de, mask=mask_k)
        tl.store(dbeta_part_ptr + off, dbeta)
        tl.store(dgamma_part_ptr + off, dgamma)
        tl.store(dg_part_ptr + off, dg)
        P = D0 * d

class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        Vd = v.shape[-1]
        BH = B * H
        # fast path check
        is_fast = (K == 128 and Vd == 128 and q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and e.is_contiguous() and g.is_contiguous() and beta.is_contiguous() and gamma.is_contiguous())
        if is_fast:
            out = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)
            states = torch.empty((BH * T * K * Vd,), device=q.device, dtype=torch.float32)
            grid = (BH,)
            fwd_fast_kernel[grid](q, k, v, e, g, beta, gamma, out, states, BH, T, float(scale), num_warps=4, num_stages=2)
            ctx.fast = True
            ctx.save_for_backward(q, k, v, e, g, beta, gamma, states)
        else:
            BLOCK_K = 32 if K > 64 else K
            if K <= 32:
                BLOCK_K = K
            BLOCK_V = 32
            if Vd < BLOCK_V:
                BLOCK_V = Vd
            VBLOCKS = (Vd + BLOCK_V - 1)//BLOCK_V
            # for generic we ensure BLOCK_K covers K (if K<=32 already, else 32 with tail mask)
            # if K not multiple of 32 we keep masking
            out = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)
            states = torch.empty((BH * T * K * Vd,), device=q.device, dtype=torch.float32)
            grid = (BH * VBLOCKS,)
            q_sb, q_sh, q_st, q_sk = q.stride()
            k_sb, k_sh, k_st, k_sk = k.stride()
            v_sb, v_sh, v_st, v_sv = v.stride()
            e_sb, e_sh, e_st, e_sk = e.stride()
            g_sb, g_sh, g_st = g.stride()
            b_sb, b_sh, b_st = beta.stride()
            ga_sb, ga_sh, ga_st = gamma.stride()
            o_sb, o_sh, o_st, o_sv = out.stride()
            fwd_generic[grid](q, k, v, e, g, beta, gamma, out, states, B, H, T, K, Vd, float(scale),
                              q_sb, q_sh, q_st, q_sk, k_sb, k_sh, k_st, k_sk, v_sb, v_sh, v_st, v_sv,
                              e_sb, e_sh, e_st, e_sk, g_sb, g_sh, g_st, b_sb, b_sh, b_st, ga_sb, ga_sh, ga_st,
                              o_sb, o_sh, o_st, o_sv, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, VBLOCKS=VBLOCKS, num_warps=4, num_stages=2)
            ctx.fast = False
            ctx.save_for_backward(q, k, v, e, g, beta, gamma, states)
            ctx.BLOCK_K = BLOCK_K; ctx.BLOCK_V = BLOCK_V; ctx.VBLOCKS = VBLOCKS
        ctx.B = B; ctx.H = H; ctx.T = T; ctx.K = K; ctx.Vd = Vd; ctx.BH = BH; ctx.scale = float(scale)
        ctx.q_dtype = q.dtype; ctx.k_dtype = k.dtype; ctx.v_dtype = v.dtype; ctx.e_dtype = e.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma, states = ctx.saved_tensors
        B = ctx.B; H = ctx.H; T = ctx.T; K = ctx.K; Vd = ctx.Vd; BH = ctx.BH; scale = ctx.scale
        go = grad_out.contiguous() if not grad_out.is_contiguous() else grad_out
        if ctx.fast:
            dv = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)
            dq = torch.empty((B, H, T, K), device=q.device, dtype=torch.float32)
            dk = torch.empty((B, H, T, K), device=q.device, dtype=torch.float32)
            de = torch.empty((B, H, T, K), device=q.device, dtype=torch.float32)
            dg = torch.empty((B, H, T), device=q.device, dtype=torch.float32)
            dbeta = torch.empty((B, H, T), device=q.device, dtype=torch.float32)
            dgamma = torch.empty((B, H, T), device=q.device, dtype=torch.float32)
            grid = (BH,)
            bwd_fast_kernel[grid](q, k, v, e, g, beta, gamma, go, states, dq, dk, de, dv, dg, dbeta, dgamma, BH, T, scale, num_warps=4, num_stages=2)
            return dq.to(ctx.q_dtype), dk.to(ctx.k_dtype), dv.to(ctx.v_dtype), de.to(ctx.e_dtype), dg, dbeta, dgamma, None
        else:
            BLOCK_K = ctx.BLOCK_K; BLOCK_V = ctx.BLOCK_V; VBLOCKS = ctx.VBLOCKS
            num = BH * T
            dv = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)
            dq_part = torch.empty((num * VBLOCKS, K), device=q.device, dtype=torch.float32)
            dk_part = torch.empty((num * VBLOCKS, K), device=q.device, dtype=torch.float32)
            de_part = torch.empty((num * VBLOCKS, K), device=q.device, dtype=torch.float32)
            dbeta_part = torch.empty((num * VBLOCKS,), device=q.device, dtype=torch.float32)
            dgamma_part = torch.empty((num * VBLOCKS,), device=q.device, dtype=torch.float32)
            dg_part = torch.empty((num * VBLOCKS,), device=q.device, dtype=torch.float32)
            q_sb, q_sh, q_st, q_sk = q.stride(); k_sb, k_sh, k_st, k_sk = k.stride()
            v_sb, v_sh, v_st, v_sv = v.stride(); e_sb, e_sh, e_st, e_sk = e.stride()
            g_sb, g_sh, g_st = g.stride(); b_sb, b_sh, b_st = beta.stride(); ga_sb, ga_sh, ga_st = gamma.stride()
            do_sb, do_sh, do_st, do_sv = go.stride(); dv_sb, dv_sh, dv_st, dv_sv = dv.stride()
            grid = (BH * VBLOCKS,)
            bwd_generic[grid](q, k, v, e, g, beta, gamma, go, states, dv, dq_part, dk_part, de_part, dbeta_part, dgamma_part, dg_part,
                              B, H, T, K, Vd, scale, q_sb, q_sh, q_st, q_sk, k_sb, k_sh, k_st, k_sk, v_sb, v_sh, v_st, v_sv,
                              e_sb, e_sh, e_st, e_sk, g_sb, g_sh, g_st, b_sb, b_sh, b_st, ga_sb, ga_sh, ga_st,
                              do_sb, do_sh, do_st, do_sv, dv_sb, dv_sh, dv_st, dv_sv, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, VBLOCKS=VBLOCKS, num_warps=4, num_stages=2)
            dq = dq_part.view(num, VBLOCKS, K).sum(dim=1).view(B, H, T, K).to(ctx.q_dtype)
            dk = dk_part.view(num, VBLOCKS, K).sum(dim=1).view(B, H, T, K).to(ctx.k_dtype)
            de = de_part.view(num, VBLOCKS, K).sum(dim=1).view(B, H, T, K).to(ctx.e_dtype)
            dv = dv.to(ctx.v_dtype)
            dg = dg_part.view(num, VBLOCKS).sum(dim=1).view(B, H, T)
            dbeta = dbeta_part.view(num, VBLOCKS).sum(dim=1).view(B, H, T)
            dgamma = dgamma_part.view(num, VBLOCKS).sum(dim=1).view(B, H, T)
            return dq, dk, dv, de, dg, dbeta, dgamma, None

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        scale = float(scale)
    return _Func.apply(q, k, v, e, g, beta, gamma, scale)
