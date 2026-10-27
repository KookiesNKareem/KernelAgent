import torch
import triton
import triton.language as tl

def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 1 else 1

@triton.jit
def fwd_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
               BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, scale: tl.constexpr,
               BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, NTILES: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // NTILES
    tile = pid % NTILES
    v_start = tile * BLOCK_V
    k_idx = tl.arange(0, BLOCK_K)
    v_idx = tl.arange(0, BLOCK_V)
    k_mask = k_idx < K
    v_mask = (v_start + v_idx) < V
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    q_base = bh * T * K
    k_base = bh * T * K
    e_base = bh * T * K
    g_base = bh * T * K
    v_base = bh * T * V
    out_base = bh * T * V
    beta_base = bh * T
    gamma_base = bh * T
    for t in range(T):
        q = tl.load(q_ptr + q_base + t * K + k_idx, mask=k_mask, other=0.0)
        q = q.to(tl.float32)
        k = tl.load(k_ptr + k_base + t * K + k_idx, mask=k_mask, other=0.0)
        k = k.to(tl.float32)
        e = tl.load(e_ptr + e_base + t * K + k_idx, mask=k_mask, other=0.0)
        e = e.to(tl.float32)
        g = tl.load(g_ptr + g_base + t * K + k_idx, mask=k_mask, other=0.0)
        g = g.to(tl.float32)
        beta = tl.load(beta_ptr + beta_base + t)
        gamma = tl.load(gamma_ptr + gamma_base + t)
        v = tl.load(v_ptr + v_base + t * V + v_start + v_idx, mask=v_mask, other=0.0)
        v = v.to(tl.float32)
        d = tl.exp(g)
        S = S * d[:, None]
        dot_e = tl.sum(e[:, None] * S, axis=0)
        S = S - gamma * e[:, None] * dot_e[None, :]
        pred = tl.sum(k[:, None] * S, axis=0)
        delta = beta * (v - pred)
        delta = tl.where(v_mask, delta, 0.0)
        S = S + k[:, None] * delta[None, :]
        out = tl.sum(S * q[:, None], axis=0) * scale
        tl.store(out_ptr + out_base + t * V + v_start + v_idx, out, mask=v_mask)

# checkpointed forward: same math but exp-hoisted (d loaded) and S_t stored
@triton.jit
def fwd_store_kernel(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, out_ptr, s_ptr,
                     BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, scale: tl.constexpr,
                     BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, NTILES: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // NTILES
    tile = pid % NTILES
    v_start = tile * BLOCK_V
    k_idx = tl.arange(0, BLOCK_K)
    v_idx = tl.arange(0, BLOCK_V)
    k_mask = k_idx < K
    v_mask = (v_start + v_idx) < V
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    q_base = bh * T * K
    k_base = bh * T * K
    e_base = bh * T * K
    d_base = bh * T * K
    v_base = bh * T * V
    out_base = bh * T * V
    b_base = bh * T
    s_base = bh * (T + 1) * K * V
    # 2-D ptr helpers for checkpoint store
    for t in range(T):
        q = tl.load(q_ptr + q_base + t * K + k_idx, mask=k_mask, other=0.0)
        q = q.to(tl.float32)
        kk = tl.load(k_ptr + k_base + t * K + k_idx, mask=k_mask, other=0.0)
        kk = kk.to(tl.float32)
        ee = tl.load(e_ptr + e_base + t * K + k_idx, mask=k_mask, other=0.0)
        ee = ee.to(tl.float32)
        dd = tl.load(d_ptr + d_base + t * K + k_idx, mask=k_mask, other=0.0)
        dd = dd.to(tl.float32)
        dd = tl.where(k_mask, dd, 1.0)
        beta = tl.load(beta_ptr + b_base + t)
        gamma = tl.load(gamma_ptr + b_base + t)
        vv = tl.load(v_ptr + v_base + t * V + v_start + v_idx, mask=v_mask, other=0.0)
        vv = vv.to(tl.float32)
        S = S * dd[:, None]
        dot_e = tl.sum(ee[:, None] * S, axis=0)
        S = S - gamma * ee[:, None] * dot_e[None, :]
        pred = tl.sum(kk[:, None] * S, axis=0)
        delta = beta * (vv - pred)
        delta = tl.where(v_mask, delta, 0.0)
        S = S + kk[:, None] * delta[None, :]
        out = tl.sum(S * q[:, None], axis=0) * scale
        tl.store(out_ptr + out_base + t * V + v_start + v_idx, out, mask=v_mask)
        # store S_t checkpoint (t+1)
        s_off = s_base + (t + 1) * K * V
        s_ptrs = s_ptr + s_off + k_idx[:, None] * V + (v_start + v_idx[None, :])
        s_mask = k_mask[:, None] & v_mask[None, :]
        tl.store(s_ptrs, S, mask=s_mask)

@triton.jit
def bwd_kernel(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, s_ptr, grad_out_ptr,
               dq_part_ptr, dk_part_ptr, de_part_ptr, dg_part_ptr, beta_part_ptr, gamma_part_ptr, grad_v_ptr,
               BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, scale: tl.constexpr,
               BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, NTILES: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // NTILES
    tile = pid % NTILES
    v_start = tile * BLOCK_V
    k_idx = tl.arange(0, BLOCK_K)
    v_idx = tl.arange(0, BLOCK_V)
    k_mask = k_idx < K
    v_mask = (v_start + v_idx) < V
    ds = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    q_base = bh * T * K
    k_base = bh * T * K
    e_base = bh * T * K
    d_base = bh * T * K
    v_base = bh * T * V
    out_base = bh * T * V
    b_base = bh * T
    s_base = bh * (T + 1) * K * V
    for t in range(T):
        rev = T - 1 - t
        q = tl.load(q_ptr + q_base + rev * K + k_idx, mask=k_mask, other=0.0)
        q = q.to(tl.float32)
        kk = tl.load(k_ptr + k_base + rev * K + k_idx, mask=k_mask, other=0.0)
        kk = kk.to(tl.float32)
        ee = tl.load(e_ptr + e_base + rev * K + k_idx, mask=k_mask, other=0.0)
        ee = ee.to(tl.float32)
        dd = tl.load(d_ptr + d_base + rev * K + k_idx, mask=k_mask, other=0.0)
        dd = dd.to(tl.float32)
        dd = tl.where(k_mask, dd, 1.0)
        beta = tl.load(beta_ptr + b_base + rev)
        gamma = tl.load(gamma_ptr + b_base + rev)
        vv = tl.load(v_ptr + v_base + rev * V + v_start + v_idx, mask=v_mask, other=0.0)
        vv = vv.to(tl.float32)
        dO = tl.load(grad_out_ptr + out_base + rev * V + v_start + v_idx, mask=v_mask, other=0.0)
        dO = dO.to(tl.float32)
        # load S_t and S_{t-1} checkpoints for this tile
        s_off_t = s_base + (rev + 1) * K * V
        s_ptrs_t = s_ptr + s_off_t + k_idx[:, None] * V + (v_start + v_idx[None, :])
        S_t = tl.load(s_ptrs_t, mask=k_mask[:, None] & v_mask[None, :], other=0.0)
        s_off_prev = s_base + rev * K * V
        s_ptrs_prev = s_ptr + s_off_prev + k_idx[:, None] * V + (v_start + v_idx[None, :])
        S_prev = tl.load(s_ptrs_prev, mask=k_mask[:, None] & v_mask[None, :], other=0.0)
        # recompute intermediates needed for VJP
        S_a = S_prev * dd[:, None]
        ea = tl.sum(ee[:, None] * S_a, axis=0)
        S_b = S_a - gamma * ee[:, None] * ea[None, :]
        pred = tl.sum(kk[:, None] * S_b, axis=0)
        u = vv - pred
        # ds_t = ds_next + scale*q outer dO
        ds_out = scale * q[:, None] * dO[None, :]
        ds_t = ds + ds_out
        # grad_v and beta
        w_k = tl.sum(kk[:, None] * ds_t, axis=0)
        grad_v = beta * w_k
        tl.store(grad_v_ptr + out_base + rev * V + v_start + v_idx, grad_v, mask=v_mask)
        beta_part = tl.sum(w_k * u, axis=0)
        off_s = (bh * NTILES + tile) * T + rev
        tl.store(beta_part_ptr + off_s, beta_part)
        tmp1 = tl.sum(ds_t * u[None, :], axis=1)
        tmp2 = tl.sum(S_b * w_k[None, :], axis=1)
        grad_k_part = beta * (tmp1 - tmp2)
        off_k = ((bh * NTILES + tile) * T + rev) * K + k_idx
        tl.store(dk_part_ptr + off_k, grad_k_part, mask=k_mask)
        ds_b = ds_t - beta * kk[:, None] * w_k[None, :]
        we_b = tl.sum(ee[:, None] * ds_b, axis=0)
        gamma_part = -tl.sum(ea * we_b, axis=0)
        tl.store(gamma_part_ptr + off_s, gamma_part)
        ds_a = ds_b - gamma * ee[:, None] * we_b[None, :]
        dq_part = scale * tl.sum(S_t * dO[None, :], axis=1)
        tl.store(dq_part_ptr + off_k, dq_part, mask=k_mask)
        dg_part = tl.sum(ds_a * S_a, axis=1)
        tl.store(dg_part_ptr + off_k, dg_part, mask=k_mask)
        de_tmp1 = tl.sum(ds_b * ea[None, :], axis=1)
        de_tmp2 = tl.sum(S_a * we_b[None, :], axis=1)
        de_part = -gamma * (de_tmp1 + de_tmp2)
        tl.store(de_part_ptr + off_k, de_part, mask=k_mask)
        ds = ds_a * dd[:, None]

class _RecurrentFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g_exp, beta, gamma, scale_f):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        if T == 0:
            ctx.save_for_backward(q, k, v, e, g_exp, beta, gamma)
            ctx.BH = BH; ctx.T = T; ctx.K = K; ctx.V = V; ctx.scale_f = scale_f
            ctx.s_buf = None; ctx.B = B; ctx.H = H
            return torch.empty((B, H, 0, V), device=q.device, dtype=torch.float32)
        BLOCK_K = _next_pow2(K)
        BLOCK_V = 32 if V > 32 else _next_pow2(V)
        NTILES = (V + BLOCK_V - 1) // BLOCK_V
        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        s_buf = torch.zeros((B, H, T + 1, K, V), device=q.device, dtype=torch.float32)
        grid = (BH * NTILES,)
        fwd_store_kernel[grid](q, k, v, e, g_exp, beta, gamma, out, s_buf,
                               BH, T, K, V, scale_f, BLOCK_K, BLOCK_V, NTILES)
        ctx.save_for_backward(q, k, v, e, g_exp, beta, gamma, s_buf)
        ctx.BH = BH; ctx.T = T; ctx.K = K; ctx.V = V; ctx.scale_f = scale_f
        ctx.BLOCK_K = BLOCK_K; ctx.BLOCK_V = BLOCK_V; ctx.NTILES = NTILES
        ctx.B = B; ctx.H = H
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g_exp, beta, gamma, s_buf = ctx.saved_tensors
        B = ctx.B; H = ctx.H; T = ctx.T; K = ctx.K; V = ctx.V
        BH = ctx.BH; scale_f = ctx.scale_f
        BLOCK_K = ctx.BLOCK_K; BLOCK_V = ctx.BLOCK_V; NTILES = ctx.NTILES
        if T == 0:
            return (torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v),
                    torch.zeros_like(e), torch.zeros_like(g_exp), torch.zeros_like(beta),
                    torch.zeros_like(gamma), None)
        grad_out_c = grad_out.contiguous().to(torch.float32)
        dq_part = torch.zeros((BH, NTILES, T, K), device=q.device, dtype=torch.float32)
        dk_part = torch.zeros((BH, NTILES, T, K), device=q.device, dtype=torch.float32)
        de_part = torch.zeros((BH, NTILES, T, K), device=q.device, dtype=torch.float32)
        dg_part = torch.zeros((BH, NTILES, T, K), device=q.device, dtype=torch.float32)
        beta_part = torch.zeros((BH, NTILES, T), device=q.device, dtype=torch.float32)
        gamma_part = torch.zeros((BH, NTILES, T), device=q.device, dtype=torch.float32)
        grad_v = torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)
        grid = (BH * NTILES,)
        bwd_kernel[grid](q, k, v, e, g_exp, beta, gamma, s_buf, grad_out_c,
                         dq_part, dk_part, de_part, dg_part, beta_part, gamma_part, grad_v,
                         BH, T, K, V, scale_f, BLOCK_K, BLOCK_V, NTILES)
        dq = dq_part.sum(dim=1).view(B, H, T, K)
        dk = dk_part.sum(dim=1).view(B, H, T, K)
        de = de_part.sum(dim=1).view(B, H, T, K)
        dg = dg_part.sum(dim=1).view(B, H, T, K)
        dbeta = beta_part.sum(dim=1).view(B, H, T)
        dgamma = gamma_part.sum(dim=1).view(B, H, T)
        return dq.to(q.dtype), dk.to(k.dtype), grad_v.to(v.dtype), de.to(e.dtype), dg.to(g_exp.dtype), dbeta.to(beta.dtype), dgamma.to(gamma.dtype), None

def _fast_forward(q, k, v, e, g, beta, gamma, scale_f):
    B, H, T, K = q.shape
    V = v.shape[-1]
    if T == 0:
        return torch.empty((B, H, 0, V), device=q.device, dtype=torch.float32)
    q_c = q.contiguous(); k_c = k.contiguous(); v_c = v.contiguous()
    e_c = e.contiguous(); g_c = g.contiguous(); beta_c = beta.contiguous(); gamma_c = gamma.contiguous()
    BH = B * H
    BLOCK_K = _next_pow2(K)
    BLOCK_V = 32 if V > 32 else _next_pow2(V)
    NTILES = (V + BLOCK_V - 1) // BLOCK_V
    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    grid = (BH * NTILES,)
    fwd_kernel[grid](q_c, k_c, v_c, e_c, g_c, beta_c, gamma_c, out, BH, T, K, V, scale_f, BLOCK_K, BLOCK_V, NTILES)
    return out

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
    needs_grad = torch.is_grad_enabled() and any(isinstance(x, torch.Tensor) and x.requires_grad for x in (q, k, v, e, g, beta, gamma))
    if not needs_grad:
        return _fast_forward(q, k, v, e, g, beta, gamma, scale_f)
    # training path: hoist exp(g) to cheap elementwise prep, contiguous for Triton
    g_exp = torch.exp(g)
    q_c = q.contiguous(); k_c = k.contiguous(); v_c = v.contiguous()
    e_c = e.contiguous(); g_exp_c = g_exp.contiguous()
    beta_c = beta.contiguous(); gamma_c = gamma.contiguous()
    out = _RecurrentFunc.apply(q_c, k_c, v_c, e_c, g_exp_c, beta_c, gamma_c, scale_f)
    return out
