import torch
import triton
import triton.language as tl

# ---------- forward (SRAM S, V-sharded) ----------
@triton.jit
def fwd_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr, s_final_ptr,
               BH, T, K, V, scale,
               BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, VBLOCKS: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // VBLOCKS
    vb = pid % VBLOCKS
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    bh_T = bh * T
    for t in range(T):
        bh_t = bh_T + t
        g_val = tl.load(g_ptr + bh_t)
        beta_val = tl.load(beta_ptr + bh_t)
        gamma_val = tl.load(gamma_ptr + bh_t)
        d = tl.exp(g_val)
        S = S * d
        e_vec = tl.load(e_ptr + bh_t * K + offs_k).to(tl.float32)
        s_e = tl.sum(e_vec[:, None] * S, axis=0)
        S = S - gamma_val * (e_vec[:, None] * s_e[None, :])
        k_vec = tl.load(k_ptr + bh_t * K + offs_k).to(tl.float32)
        s_k = tl.sum(k_vec[:, None] * S, axis=0)
        v_vec = tl.load(v_ptr + bh_t * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        S = S + beta_val * (k_vec[:, None] * (v_vec - s_k)[None, :])
        q_vec = tl.load(q_ptr + bh_t * K + offs_k).to(tl.float32)
        o_vec = tl.sum(q_vec[:, None] * S, axis=0) * scale
        tl.store(o_ptr + bh_t * V + offs_v, o_vec, mask=mask_v)
    # write final S_T for backward ( [BH,K,V] row-major K*V per bh )
    base = s_final_ptr + bh * K * V
    ptrs = base + offs_k[:, None] * V + offs_v[None, :]
    tl.store(ptrs, S, mask=mask_v[None, :])

@triton.jit
def bwd_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, do_ptr,
               dv_ptr,
               dq_part_ptr, dk_part_ptr, de_part_ptr,
               dbeta_part_ptr, dgamma_part_ptr, dg_part_ptr,
               s_final_ptr,
               BH, T, K, V, scale,
               BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, VBLOCKS: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // VBLOCKS
    vb = pid % VBLOCKS
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V
    # load S_T shard
    base_s = s_final_ptr + bh * K * V
    s_ptrs = base_s + offs_k[:, None] * V + offs_v[None, :]
    S = tl.load(s_ptrs, mask=mask_v[None, :], other=0.0)
    Prop = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    bh_T = bh * T
    for i in range(T):
        t = T - 1 - i
        bh_t = bh_T + t
        k_vec = tl.load(k_ptr + bh_t * K + offs_k).to(tl.float32)
        v_vec = tl.load(v_ptr + bh_t * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        e_vec = tl.load(e_ptr + bh_t * K + offs_k).to(tl.float32)
        q_vec = tl.load(q_ptr + bh_t * K + offs_k).to(tl.float32)
        do_vec = tl.load(do_ptr + bh_t * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        beta_val = tl.load(beta_ptr + bh_t)
        gamma_val = tl.load(gamma_ptr + bh_t)
        g_val = tl.load(g_ptr + bh_t)
        d_val = tl.exp(g_val)

        # D_t = scale*q do^T + Prop
        C = scale * (q_vec[:, None] * do_vec[None, :])
        D = C + Prop

        # ---- S inversion intermediates for this t ----
        # U = S - beta*k v^T
        U = S - beta_val * (k_vec[:, None] * v_vec[None, :])
        sku = tl.sum(k_vec[:, None] * U, axis=0)  # BV
        beta_p = beta_val / (1.0 - beta_val)
        S_mid = U + beta_p * (k_vec[:, None] * sku[None, :])
        sk_mid = tl.sum(k_vec[:, None] * S_mid, axis=0)
        r_vec = v_vec - sk_mid
        sD = tl.sum(k_vec[:, None] * D, axis=0)

        # dv (owned per V)
        dv_vec = beta_val * sD
        tl.store(dv_ptr + bh_t * V + offs_v, dv_vec, mask=mask_v)

        # dbeta partial
        dbeta_p = tl.sum(sD * r_vec, axis=0)
        tl.store(dbeta_part_ptr + bh_t * VBLOCKS + vb, dbeta_p)

        # dk partial  beta*( D r - S_mid sD )
        D_r = tl.sum(D * r_vec[None, :], axis=1)          # K
        S_mid_sD = tl.sum(S_mid * sD[None, :], axis=1)    # K
        dk_vec = beta_val * (D_r - S_mid_sD)

        # D_mid1 / D_mid0
        D_mid1 = D - beta_val * (k_vec[:, None] * sD[None, :])
        s_e_mid = tl.sum(e_vec[:, None] * S_mid, axis=0)
        gamma_p = gamma_val / (1.0 - gamma_val)
        Se = S_mid + gamma_p * (e_vec[:, None] * s_e_mid[None, :])
        s_e0 = tl.sum(e_vec[:, None] * Se, axis=0)
        s_eD = tl.sum(e_vec[:, None] * D_mid1, axis=0)
        D_mid0 = D_mid1 - gamma_val * (e_vec[:, None] * s_eD[None, :])

        dgamma_p = -tl.sum(s_e0 * s_eD, axis=0)
        tl.store(dgamma_part_ptr + bh_t * VBLOCKS + vb, dgamma_p)

        dg_p = tl.sum(D_mid0 * Se, axis=0)
        dg_p = tl.sum(dg_p, axis=0)
        tl.store(dg_part_ptr + bh_t * VBLOCKS + vb, dg_p)

        D_mid1_s_e0 = tl.sum(D_mid1 * s_e0[None, :], axis=1)
        Se_s_eD = tl.sum(Se * s_eD[None, :], axis=1)
        de_vec = -gamma_val * (D_mid1_s_e0 + Se_s_eD)

        dq_vec = scale * tl.sum(S * do_vec[None, :], axis=1)

        # store K-vector partials  [BH*T*VBLOCKS, K]
        part_base = (bh_t * VBLOCKS + vb) * K
        tl.store(dq_part_ptr + part_base + offs_k, dq_vec)
        tl.store(dk_part_ptr + part_base + offs_k, dk_vec)
        tl.store(de_part_ptr + part_base + offs_k, de_vec)

        # propagate to t-1
        Prop = d_val * D_mid0
        S = Se * tl.exp(-g_val)

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
        s_final = torch.empty((BH, K, Vd), device=q.device, dtype=torch.float32)
        grid = (BH * VBLOCKS,)
        fwd_kernel[grid](q, k, v, e, g, beta, gamma, out, s_final,
                         BH, T, K, Vd, float(scale),
                         BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, VBLOCKS=VBLOCKS,
                         num_warps=4, num_stages=2)
        ctx.save_for_backward(q, k, v, e, g, beta, gamma, s_final)
        ctx.BH = BH; ctx.B = B; ctx.H = H; ctx.T = T; ctx.K = K; ctx.Vd = Vd
        ctx.BLOCK_K = BLOCK_K; ctx.BLOCK_V = BLOCK_V; ctx.VBLOCKS = VBLOCKS
        ctx.scale = float(scale)
        ctx.q_dtype = q.dtype; ctx.k_dtype = k.dtype; ctx.v_dtype = v.dtype; ctx.e_dtype = e.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma, s_final = ctx.saved_tensors
        B = ctx.B; H = ctx.H; T = ctx.T; K = ctx.K; Vd = ctx.Vd
        BH = ctx.BH; BLOCK_K = ctx.BLOCK_K; BLOCK_V = ctx.BLOCK_V; VBLOCKS = ctx.VBLOCKS
        scale = ctx.scale
        go = grad_out.contiguous().to(torch.float32)
        num = BH * T
        dv = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)
        dq_part = torch.empty((num * VBLOCKS, K), device=q.device, dtype=torch.float32)
        dk_part = torch.empty((num * VBLOCKS, K), device=q.device, dtype=torch.float32)
        de_part = torch.empty((num * VBLOCKS, K), device=q.device, dtype=torch.float32)
        dbeta_part = torch.empty((num, VBLOCKS), device=q.device, dtype=torch.float32)
        dgamma_part = torch.empty((num, VBLOCKS), device=q.device, dtype=torch.float32)
        dg_part = torch.empty((num, VBLOCKS), device=q.device, dtype=torch.float32)
        grid = (BH * VBLOCKS,)
        bwd_kernel[grid](q, k, v, e, g, beta, gamma, go,
                         dv, dq_part, dk_part, de_part,
                         dbeta_part, dgamma_part, dg_part,
                         s_final,
                         BH, T, K, Vd, scale,
                         BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, VBLOCKS=VBLOCKS,
                         num_warps=4, num_stages=2)
        dq = dq_part.view(num, VBLOCKS, K).sum(dim=1).view(B, H, T, K).to(ctx.q_dtype)
        dk = dk_part.view(num, VBLOCKS, K).sum(dim=1).view(B, H, T, K).to(ctx.k_dtype)
        de = de_part.view(num, VBLOCKS, K).sum(dim=1).view(B, H, T, K).to(ctx.e_dtype)
        dv = dv.to(ctx.v_dtype)
        dg = dg_part.sum(dim=1).view(B, H, T)
        dbeta = dbeta_part.sum(dim=1).view(B, H, T)
        dgamma = dgamma_part.sum(dim=1).view(B, H, T)
        return dq, dk, dv, de, dg, dbeta, dgamma, None

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); e = e.contiguous()
    g = g.contiguous(); beta = beta.contiguous(); gamma = gamma.contiguous()
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    return _Func.apply(q, k, v, e, g, beta, gamma, float(scale))
