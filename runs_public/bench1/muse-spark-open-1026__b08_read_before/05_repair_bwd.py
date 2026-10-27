import torch
import triton
import triton.language as tl

# ---------- forward chunk : produces out and next checkpoint ----------
@triton.jit
def fwd_chunk_kernel(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,out_ptr,chk_ptr,
    BH, T, K, V,
    N: tl.constexpr, C: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    SCALE: tl.constexpr,
    t0: tl.constexpr, cur_T: tl.constexpr, chk_idx: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_vb = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_offs = pid_vb * BLOCK_V + offs_v
    mask_v = v_offs < V
    # load S start
    base_chk = (pid_bh * (N + 1) + chk_idx) * K * V
    S = tl.load(chk_ptr + base_chk + offs_k[:, None] * V + v_offs[None, :],
                mask=(offs_k[:, None] < K) & (v_offs[None, :] < V), other=0.0)
    for i in range(cur_T):
        t = t0 + i
        base_qk = (pid_bh * T + t) * K
        base_v = (pid_bh * T + t) * V
        q = tl.load(q_ptr + base_qk + offs_k).to(tl.float32) * SCALE
        k = tl.load(k_ptr + base_qk + offs_k).to(tl.float32)
        g = tl.load(g_ptr + base_qk + offs_k).to(tl.float32)
        d = tl.exp(g)
        v = tl.load(v_ptr + base_v + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + pid_bh * T + t).to(tl.float32)
        S = S * d[:, None]
        out = tl.sum(S * q[:, None], axis=0)
        pred = tl.sum(S * k[:, None], axis=0)
        tl.store(out_ptr + base_v + v_offs, out, mask=mask_v)
        S = S + k[:, None] * (beta * (v - pred))[None, :]
    base_next = (pid_bh * (N + 1) + chk_idx + 1) * K * V
    tl.store(chk_ptr + base_next + offs_k[:, None] * V + v_offs[None, :], S,
             mask=(offs_k[:, None] < K) & (v_offs[None, :] < V))

# ---------- scratch : recompute S_d for one chunk ----------
@triton.jit
def scratch_kernel(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,chk_ptr,scr_ptr,
    BH,T,K,V,N: tl.constexpr,C: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    t0: tl.constexpr, cur_T: tl.constexpr, chk_idx: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_vb = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_offs = pid_vb * BLOCK_V + offs_v
    base_chk = (pid_bh * (N + 1) + chk_idx) * K * V
    S = tl.load(chk_ptr + base_chk + offs_k[:, None] * V + v_offs[None, :],
                mask=(offs_k[:, None] < K) & (v_offs[None, :] < V), other=0.0)
    scr_base = pid_bh * C * K * V
    for i in range(cur_T):
        t = t0 + i
        base_qk = (pid_bh * T + t) * K
        base_v = (pid_bh * T + t) * V
        k = tl.load(k_ptr + base_qk + offs_k).to(tl.float32)
        g = tl.load(g_ptr + base_qk + offs_k).to(tl.float32)
        d = tl.exp(g)
        v = tl.load(v_ptr + base_v + v_offs, mask=v_offs < V, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + pid_bh * T + t).to(tl.float32)
        Sd = S * d[:, None]
        tl.store(scr_ptr + scr_base + i * K * V + offs_k[:, None] * V + v_offs[None, :], Sd,
                 mask=(offs_k[:, None] < K) & (v_offs[None, :] < V))
        pred = tl.sum(Sd * k[:, None], axis=0)
        S = Sd + k[:, None] * (beta * (v - pred))[None, :]

# ---------- backward chunk : uses scratch and G checkpoints ----------
@triton.jit
def bwd_chunk_kernel(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,do_ptr,scr_ptr, gchk_ptr, dq_ptr, dk_ptr, dg_ptr, dv_ptr, dbeta_ptr,
    BH,T,K,V,N: tl.constexpr,C: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, SCALE: tl.constexpr,
    t0: tl.constexpr, cur_T: tl.constexpr, chk_idx: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_vb = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_offs = pid_vb * BLOCK_V + offs_v
    mask_v = v_offs < V
    base_g_next = (pid_bh * (N + 1) + chk_idx + 1) * K * V
    G = tl.load(gchk_ptr + base_g_next + offs_k[:, None] * V + v_offs[None, :],
                mask=(offs_k[:, None] < K) & (v_offs[None, :] < V), other=0.0)
    scr_base = pid_bh * C * K * V
    # iterate backwards
    for ri in range(cur_T):
        i = cur_T - 1 - ri
        t = t0 + i
        base_qk = (pid_bh * T + t) * K
        base_v = (pid_bh * T + t) * V
        q = tl.load(q_ptr + base_qk + offs_k).to(tl.float32)
        k = tl.load(k_ptr + base_qk + offs_k).to(tl.float32)
        g = tl.load(g_ptr + base_qk + offs_k).to(tl.float32)
        d = tl.exp(g)
        v = tl.load(v_ptr + base_v + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + base_v + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + pid_bh * T + t).to(tl.float32)
        Sd = tl.load(scr_ptr + scr_base + i * K * V + offs_k[:, None] * V + v_offs[None, :],
                     mask=(offs_k[:, None] < K) & (v_offs[None, :] < V), other=0.0)
        pred = tl.sum(Sd * k[:, None], axis=0)
        u = tl.sum(G * k[:, None], axis=0)
        vmp = v - pred
        # dv
        dv = beta * u
        tl.store(dv_ptr + base_v + v_offs, dv, mask=mask_v)
        # dq partial
        dq_part = SCALE * tl.sum(Sd * do[None, :], axis=1)
        tl.atomic_add(dq_ptr + base_qk + offs_k, dq_part)
        # dk partial
        dk_part = beta * (tl.sum(G * vmp[None, :], axis=1) - tl.sum(Sd * u[None, :], axis=1))
        tl.atomic_add(dk_ptr + base_qk + offs_k, dk_part)
        # G_raw for dg and propagate
        G_raw = G + SCALE * q[:, None] * do[None, :] - beta * k[:, None] * u[None, :]
        dg_part = tl.sum(G_raw * Sd, axis=1)
        tl.atomic_add(dg_ptr + base_qk + offs_k, dg_part)
        db_part = tl.sum(u * vmp, axis=0)
        tl.atomic_add(dbeta_ptr + pid_bh * T + t, db_part)
        G = G_raw * d[:, None]
    base_g = (pid_bh * (N + 1) + chk_idx) * K * V
    tl.store(gchk_ptr + base_g + offs_k[:, None] * V + v_offs[None, :], G,
             mask=(offs_k[:, None] < K) & (v_offs[None, :] < V))

class GatedDeltaFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q,k,v,e,g,beta,gamma,scale):
        B,H,T,K = q.shape
        V = v.shape[-1]
        BH = B*H
        device = q.device
        scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        # contiguous flattened views for kernel
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        g_c = g.contiguous()
        beta_c = beta.contiguous()
        q_f = q_c.view(BH,T,K)
        k_f = k_c.view(BH,T,K)
        v_f = v_c.view(BH,T,V)
        g_f = g_c.view(BH,T,K)
        beta_f = beta_c.view(BH,T)
        out = torch.empty(B,H,T,V, dtype=torch.float32, device=device)
        out_f = out.view(BH,T,V)
        C = 64
        N = (T + C - 1)//C
        # choose BV
        if V % 64 == 0 and K * 64 <= 8192:
            BV = 64
        else:
            BV = 32
        BLOCK_K = K
        BLOCK_V = BV
        NV = (V + BV - 1)//BV
        chk = torch.zeros(BH, N+1, K, V, dtype=torch.float32, device=device)
        grid = (BH, NV)
        for chk_idx in range(N):
            t0 = chk_idx * C
            cur_T = C if t0 + C <= T else T - t0
            fwd_chunk_kernel[grid](q_f,k_f,v_f,g_f,beta_f,out_f,chk,
                BH,T,K,V,N,C,BLOCK_K,BLOCK_V,scale_f,t0,cur_T,chk_idx)
        # save for backward
        ctx.q_f=q_f; ctx.k_f=k_f; ctx.v_f=v_f; ctx.g_f=g_f; ctx.beta_f=beta_f
        ctx.chk=chk; ctx.BH=BH; ctx.B=B; ctx.H=H; ctx.T=T; ctx.K=K; ctx.V=V
        ctx.C=C; ctx.N=N; ctx.BV=BV; ctx.BLOCK_K=BLOCK_K; ctx.BLOCK_V=BLOCK_V; ctx.scale_f=scale_f
        ctx.q_dtype=q.dtype; ctx.k_dtype=k.dtype; ctx.v_dtype=v.dtype; ctx.e_dtype=e.dtype
        ctx.g_dtype=g.dtype; ctx.beta_dtype=beta.dtype; ctx.gamma_dtype=gamma.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        BH,T,K,V = ctx.BH, ctx.T, ctx.K, ctx.V
        B,H,C,N,BV = ctx.B, ctx.H, ctx.C, ctx.N, ctx.BV
        BLOCK_K, BLOCK_V, scale_f = ctx.BLOCK_K, ctx.BLOCK_V, ctx.scale_f
        device = grad_out.device
        go = grad_out.contiguous().view(BH,T,V)
        dq = torch.zeros(BH,T,K, dtype=torch.float32, device=device)
        dk = torch.zeros(BH,T,K, dtype=torch.float32, device=device)
        dg = torch.zeros(BH,T,K, dtype=torch.float32, device=device)
        dv = torch.zeros(BH,T,V, dtype=torch.float32, device=device)
        dbeta = torch.zeros(BH,T, dtype=torch.float32, device=device)
        gchk = torch.zeros(BH, N+1, K, V, dtype=torch.float32, device=device)
        scr = torch.empty(BH, C, K, V, dtype=torch.float32, device=device)
        NV = (V + BV - 1)//BV
        grid = (BH, NV)
        for chk_idx in range(N-1, -1, -1):
            t0 = chk_idx * C
            cur_T = C if t0 + C <= T else T - t0
            scratch_kernel[grid](ctx.q_f, ctx.k_f, ctx.v_f, ctx.g_f, ctx.beta_f, ctx.chk, scr,
                BH,T,K,V,N,C,BLOCK_K,BLOCK_V,t0,cur_T,chk_idx)
            bwd_chunk_kernel[grid](ctx.q_f, ctx.k_f, ctx.v_f, ctx.g_f, ctx.beta_f, go, scr, gchk, dq, dk, dg, dv, dbeta,
                BH,T,K,V,N,C,BLOCK_K,BLOCK_V,scale_f,t0,cur_T,chk_idx)
        dq_out = dq.view(B,H,T,K).to(ctx.q_dtype)
        dk_out = dk.view(B,H,T,K).to(ctx.k_dtype)
        dv_out = dv.view(B,H,T,V).to(ctx.v_dtype)
        dg_out = dg.view(B,H,T,K).to(ctx.g_dtype)
        dbeta_out = dbeta.view(B,H,T).to(ctx.beta_dtype)
        de = torch.zeros(B,H,T,K, dtype=ctx.e_dtype, device=device)
        dgamma = torch.zeros(B,H,T, dtype=ctx.gamma_dtype, device=device)
        return dq_out, dk_out, dv_out, de, dg_out, dbeta_out, dgamma, None

def kernel_function(q,k,v,e,g,beta,gamma,scale):
    return GatedDeltaFunc.apply(q,k,v,e,g,beta,gamma,scale)
