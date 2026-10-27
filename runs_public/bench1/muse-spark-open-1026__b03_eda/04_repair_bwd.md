```python
import torch
import triton
import triton.language as tl

# ---------- forward : keep S = K x BLOCK_V in SRAM, V-sharded ----------
@triton.jit
def fwd_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr, states_ptr,
               B, H, T, K, V, scale,
               q_sb, q_sh, q_st, q_sk,
               k_sb, k_sh, k_st, k_sk,
               v_sb, v_sh, v_st, v_sv,
               e_sb, e_sh, e_st, e_sk,
               g_sb, g_sh, g_st,
               b_sb, b_sh, b_st,
               ga_sb, ga_sh, ga_st,
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
    # strides for V-block loads are in elements
    for t in range(T):
        g_val = tl.load(g_ptr + b * g_sb + h * g_sh + t * g_st)
        beta_val = tl.load(beta_ptr + b * b_sb + h * b_sh + t * b_st)
        gamma_val = tl.load(gamma_ptr + b * ga_sb + h * ga_sh + t * ga_st)
        d = tl.exp(g_val)
        S = S * d

        e_base = e_ptr + b * e_sb + h * e_sh + t * e_st
        e_vec = tl.load(e_base + offs_k * e_sk, mask=mask_k, other=0.0).to(tl.float32)
        s_e = tl.sum(e_vec[:, None] * S, axis=0)  # BLOCK_V
        # masked columns stay 0
        S = S - gamma_val * (e_vec[:, None] * s_e[None, :])

        k_base = k_ptr + b * k_sb + h * k_sh + t * k_st
        k_vec = tl.load(k_base + offs_k * k_sk, mask=mask_k, other=0.0).to(tl.float32)
        s_k = tl.sum(k_vec[:, None] * S, axis=0)
        v_base = v_ptr + b * v_sb + h * v_sh + t * v_st
        v_vec = tl.load(v_base + offs_v * v_sv, mask=mask_v, other=0.0).to(tl.float32)
        # S = S + beta*k*(v - s_k)^T
        S = S + beta_val * (k_vec[:, None] * (v_vec - s_k)[None, :])

        q_base = q_ptr + b * q_sb + h * q_sh + t * q_st
        q_vec = tl.load(q_base + offs_k * q_sk, mask=mask_k, other=0.0).to(tl.float32)
        o_vec = tl.sum(q_vec[:, None] * S, axis=0) * scale
        o_base = o_ptr + b * o_sb + h * o_sh + t * o_st
        tl.store(o_base + offs_v * o_sv, o_vec, mask=mask_v)

        # write state shard [BH,T,K,V] contiguous
        st_base = states_ptr + (bh * T + t) * K * V
        ptrs = st_base + offs_k[:, None] * V + offs_v[None, :]
        tl.store(ptrs, S, mask=mask_k[:, None] & mask_v[None, :])


@triton.jit
def bwd_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, do_ptr,
               states_ptr,
               dv_ptr,
               dq_part_ptr, dk_part_ptr, de_part_ptr,
               dbeta_part_ptr, dgamma_part_ptr, dg_part_ptr,
               B, H, T, K, V, scale,
               q_sb, q_sh, q_st, q_sk,
               k_sb, k_sh, k_st, k_sk,
               v_sb, v_sh, v_st, v_sv,
               e_sb, e_sh, e_st, e_sk,
               g_sb, g_sh, g_st,
               b_sb, b_sh, b_st,
               ga_sb, ga_sh, ga_st,
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
    # Prop = A_{t+1}^T ...  K x BLOCK_V
    Prop = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for idx in range(T):
        t = T - 1 - idx

        # ----- load forward scalars -----
        g_val = tl.load(g_ptr + b * g_sb + h * g_sh + t * g_st)
        beta_val = tl.load(beta_ptr + b * b_sb + h * b_sh + t * b_st)
        gamma_val = tl.load(gamma_ptr + b * ga_sb + h * ga_sh + t * ga_st)
        d_val = tl.exp(g_val)

        q_base = q_ptr + b * q_sb + h * q_sh + t * q_st
        q_vec = tl.load(q_base + offs_k * q_sk, mask=mask_k, other=0.0).to(tl.float32)
        k_base = k_ptr + b * k_sb + h * k_sh + t * k_st
        k_vec = tl.load(k_base + offs_k * k_sk, mask=mask_k, other=0.0).to(tl.float32)
        e_base = e_ptr + b * e_sb + h * e_sh + t * e_st
        e_vec = tl.load(e_base + offs_k * e_sk, mask=mask_k, other=0.0).to(tl.float32)
        v_base = v_ptr + b * v_sb + h * v_sh + t * v_st
        v_vec = tl.load(v_base + offs_v * v_sv, mask=mask_v, other=0.0).to(tl.float32)
        do_base = do_ptr + b * do_sb + h * do_sh + t * do_st
        do_vec = tl.load(do_base + offs_v * do_sv, mask=mask_v, other=0.0).to(tl.float32)

        # ----- reconstruct forward intermediates from S_prev -----
        if t == 0:
            S_prev = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        else:
            st_prev_base = states_ptr + (bh * T + (t - 1)) * K * V
            S_prev = tl.load(st_prev_base + offs_k[:, None] * V + offs_v[None, :],
                             mask=mask_k[:, None] & mask_v[None, :], other=0.0)

        S0 = S_prev * d_val
        s_e = tl.sum(e_vec[:, None] * S0, axis=0)  # BLOCK_V
        S1 = S0 - gamma_val * (e_vec[:, None] * s_e[None, :])
        s_k1 = tl.sum(k_vec[:, None] * S1, axis=0)
        r_vec = v_vec - s_k1
        S_cur = S1 + beta_val * (k_vec[:, None] * r_vec[None, :])

        # ----- backward -----
        # C = scale * q do^T
        C = scale * (q_vec[:, None] * do_vec[None, :])
        D = C + Prop  # dS_t  K x BLOCK_V

        # dq partial : scale * S_cur * do
        dq_vec = scale * tl.sum(S_cur * do_vec[None, :], axis=1)  # K (sum over BLOCK_V)

        # dv and w = k^T D
        w = tl.sum(k_vec[:, None] * D, axis=0)  # BLOCK_V
        dv_vec = beta_val * w
        dv_base = dv_ptr + b * dv_sb + h * dv_sh + t * dv_st
        tl.store(dv_base + offs_v * dv_sv, dv_vec, mask=mask_v)

        dbeta_partial = tl.sum(w * r_vec, axis=0)  # scalar partial

        # dk = beta*( D*r - S1*w )
        Dr = tl.sum(D * r_vec[None, :], axis=1)      # K
        S1w = tl.sum(S1 * w[None, :], axis=1)       # K
        dk_vec = beta_val * (Dr - S1w)

        # propagate through beta: D1 = (I - beta k k^T) D
        D1 = D - beta_val * (k_vec[:, None] * w[None, :])

        # gamma branch
        w_e1 = tl.sum(e_vec[:, None] * D1, axis=0)  # BLOCK_V
        dgamma_partial = -tl.sum(w_e1 * s_e, axis=0)

        De_s = tl.sum(D1 * s_e[None, :], axis=1)    # K
        S0_we = tl.sum(S0 * w_e1[None, :], axis=1) # K
        de_vec = -gamma_val * (De_s + S0_we)

        D0 = D1 - gamma_val * (e_vec[:, None] * w_e1[None, :])

        dg_partial = tl.sum(D0 * S0, axis=0)  # sum over K -> BLOCK_V then need sum over KxV
        # actually tl.sum(D0*S0) without axis gives scalar? we did sum over K only -> BLOCK_V vector
        # sum over remaining V to get scalar partial for this shard
        # D0*S0 shape KxBLOCK_V, sum over K => BLOCK_V, sum over BLOCK_V => scalar
        dg_partial = tl.sum(dg_partial, axis=0)

        # part buffers
        part_base = (bh * T + t) * VBLOCKS + vb
        # K-vector parts: [num*VBLOCKS, K]
        off = part_base * K
        tl.store(dq_part_ptr + off + offs_k, dq_vec, mask=mask_k)
        tl.store(dk_part_ptr + off + offs_k, dk_vec, mask=mask_k)
        tl.store(de_part_ptr + off + offs_k, de_vec, mask=mask_k)
        tl.store(dbeta_part_ptr + part_base, dbeta_partial)
        tl.store(dgamma_part_ptr + part_base, dgamma_partial)
        tl.store(dg_part_ptr + part_base, dg_partial)

        Prop = d_val * D0


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        Vd = v.shape[-1]
        BH = B * H
        BLOCK_K = K
        BLOCK_V = 32
        if Vd < BLOCK_V:
            BLOCK_V = Vd
        VBLOCKS = (Vd + BLOCK_V - 1) // BLOCK_V
        out = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)
        states = torch.empty((BH * T * K * Vd), device=q.device, dtype=torch.float32)
        grid = (BH * VBLOCKS,)
        # strides in elements
        q_sb, q_sh, q_st, q_sk = q.stride()
        k_sb, k_sh, k_st, k_sk = k.stride()
        v_sb, v_sh, v_st, v_sv = v.stride()
        e_sb, e_sh, e_st, e_sk = e.stride()
        g_sb, g_sh, g_st = g.stride()
        b_sb, b_sh, b_st = beta.stride()
        ga_sb, ga_sh, ga_st = gamma.stride()
        o_sb, o_sh, o_st, o_sv = out.stride()

        fwd_kernel[grid](q, k, v, e, g, beta, gamma, out, states,
                         B, H, T, K, Vd, float(scale),
                         q_sb, q_sh, q_st, q_sk,
                         k_sb, k_sh, k_st, k_sk,
                         v_sb, v_sh, v_st, v_sv,
                         e_sb, e_sh, e_st, e_sk,
                         g_sb, g_sh, g_st,
                         b_sb, b_sh, b_st,
                         ga_sb, ga_sh, ga_st,
                         o_sb, o_sh, o_st, o_sv,
                         BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, VBLOCKS=VBLOCKS,
                         num_warps=4, num_stages=2)
        ctx.save_for_backward(q, k, v, e, g, beta, gamma, states)
        ctx.B = B; ctx.H = H; ctx.T = T; ctx.K = K; ctx.Vd = Vd
        ctx.BH = BH; ctx.BLOCK_K = BLOCK_K; ctx.BLOCK_V = BLOCK_V; ctx.VBLOCKS = VBLOCKS
        ctx.scale = float(scale)
        ctx.q_dtype = q.dtype; ctx.k_dtype = k.dtype; ctx.v_dtype = v.dtype; ctx.e_dtype = e.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma, states = ctx.saved_tensors
        B = ctx.B; H = ctx.H; T = ctx.T; K = ctx.K; Vd = ctx.Vd
        BH = ctx.BH; BLOCK_K = ctx.BLOCK_K; BLOCK_V = ctx.BLOCK_V; VBLOCKS = ctx.VBLOCKS
        scale = ctx.scale
        go = grad_out.contiguous() if not grad_out.is_contiguous() else grad_out
        # grad_out is [B,H,T,Vd] float32 contiguous
        num = BH * T
        dv = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)
        dq_part = torch.empty((num * VBLOCKS, K), device=q.device, dtype=torch.float32)
        dk_part = torch.empty((num * VBLOCKS, K), device=q.device, dtype=torch.float32)
        de_part = torch.empty((num * VBLOCKS, K), device=q.device, dtype=torch.float32)
        dbeta_part = torch.empty((num * VBLOCKS), device=q.device, dtype=torch.float32)
        dgamma_part = torch.empty((num * VBLOCKS), device=q.device, dtype=torch.float32)
        dg_part = torch.empty((num * VBLOCKS), device=q.device, dtype=torch.float32)

        q_sb, q_sh, q_st, q_sk = q.stride()
        k_sb, k_sh, k_st, k_sk = k.stride()
        v_sb, v_sh, v_st, v_sv = v.stride()
        e_sb, e_sh, e_st, e_sk = e.stride()
        g_sb, g_sh, g_st = g.stride()
        b_sb, b_sh, b_st = beta.stride()
        ga_sb, ga_sh, ga_st = gamma.stride()
        do_sb, do_sh, do_st, do_sv = go.stride()
        dv_sb, dv_sh, dv_st, dv_sv = dv.stride()

        grid = (BH * VBLOCKS,)
        bwd_kernel[grid](q, k, v, e, g, beta, gamma, go,
                         states, dv,
                         dq_part, dk_part, de_part,
                         dbeta_part, dgamma_part, dg_part,
                         B, H, T, K, Vd, scale,
                         q_sb, q_sh, q_st, q_sk,
                         k_sb, k_sh, k_st, k_sk,
                         v_sb, v_sh, v_st, v_sv,
                         e_sb, e_sh, e_st, e_sk,
                         g_sb, g_sh, g_st,
                         b_sb, b_sh, b_st,
                         ga_sb, ga_sh, ga_st,
                         do_sb, do_sh, do_st, do_sv,
                         dv_sb, dv_sh, dv_st, dv_sv,
                         BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, VBLOCKS=VBLOCKS,
                         num_warps=4, num_stages=2)

        # reduction over VBLOCKS (cheap)
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
```
