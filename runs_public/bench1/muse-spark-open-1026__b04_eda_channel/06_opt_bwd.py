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

@triton.jit
def fwd_out_chk_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr, chk_ptr,
               BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, scale: tl.constexpr,
               BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, NTILES: tl.constexpr, CHUNK: tl.constexpr, Nc: tl.constexpr):
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
    chk_base = bh * (Nc + 1) * K * V
    for t in range(T):
        q = tl.load(q_ptr + q_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + k_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        e = tl.load(e_ptr + e_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + g_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + beta_base + t)
        gamma = tl.load(gamma_ptr + gamma_base + t)
        v = tl.load(v_ptr + v_base + t * V + v_start + v_idx, mask=v_mask, other=0.0).to(tl.float32)
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
        if (t + 1) % CHUNK == 0:
            chk_idx = (t + 1) // CHUNK
            s_off = chk_base + chk_idx * K * V
            s_ptrs = chk_ptr + s_off + k_idx[:, None] * V + (v_start + v_idx[None, :])
            s_mask = k_mask[:, None] & v_mask[None, :]
            tl.store(s_ptrs, S, mask=s_mask)

@triton.jit
def chunk_fwd_bwd_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, chk_ptr, tmp_ptr, ds_ptr, grad_out_ptr,
               grad_q_ptr, grad_k_ptr, grad_e_ptr, grad_g_ptr, grad_beta_ptr, grad_gamma_ptr, grad_v_ptr,
               BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, scale: tl.constexpr,
               BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, NTILES: tl.constexpr, CHUNK: tl.constexpr, Nc: tl.constexpr,
               chunk_idx: tl.constexpr, cur_len: tl.constexpr, chunk_start: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // NTILES
    tile = pid % NTILES
    v_start = tile * BLOCK_V
    k_idx = tl.arange(0, BLOCK_K)
    v_idx = tl.arange(0, BLOCK_V)
    k_mask = k_idx < K
    v_mask = (v_start + v_idx) < V

    q_base = bh * T * K
    k_base = bh * T * K
    e_base = bh * T * K
    g_base = bh * T * K
    v_base = bh * T * V
    beta_base = bh * T
    gamma_base = bh * T
    out_base = bh * T * V
    chk_base = bh * (Nc + 1) * K * V
    tmp_base = bh * CHUNK * K * V
    ds_base = bh * K * V
    grad_q_base = bh * T * K
    grad_k_base = bh * T * K
    grad_e_base = bh * T * K
    grad_g_base = bh * T * K
    grad_b_base = bh * T
    grad_v_base = bh * T * V

    # load ds_next for this tile
    ds_ptrs = ds_ptr + ds_base + k_idx[:, None] * V + (v_start + v_idx[None, :])
    ds_mask = k_mask[:, None] & v_mask[None, :]
    ds = tl.load(ds_ptrs, mask=ds_mask, other=0.0)

    # load S_start
    s_off_start = chk_base + chunk_idx * K * V
    s_ptrs_start = chk_ptr + s_off_start + k_idx[:, None] * V + (v_start + v_idx[None, :])
    S = tl.load(s_ptrs_start, mask=ds_mask, other=0.0)

    # forward recompute storing to tmp
    for i in range(cur_len):
        t = chunk_start + i
        k_ = tl.load(k_ptr + k_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        e_ = tl.load(e_ptr + e_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        g_ = tl.load(g_ptr + g_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + beta_base + t)
        gamma = tl.load(gamma_ptr + gamma_base + t)
        v_ = tl.load(v_ptr + v_base + t * V + v_start + v_idx, mask=v_mask, other=0.0).to(tl.float32)
        d_ = tl.exp(g_)
        S = S * d_[:, None]
        dot_e = tl.sum(e_[:, None] * S, axis=0)
        S = S - gamma * e_[:, None] * dot_e[None, :]
        pred = tl.sum(k_[:, None] * S, axis=0)
        delta = beta * (v_ - pred)
        delta = tl.where(v_mask, delta, 0.0)
        S = S + k_[:, None] * delta[None, :]
        t_off = tmp_base + i * K * V
        t_ptrs = tmp_ptr + t_off + k_idx[:, None] * V + (v_start + v_idx[None, :])
        tl.store(t_ptrs, S, mask=ds_mask)

    # reload S_start for backward (need distinct variable)
    S_start = tl.load(s_ptrs_start, mask=ds_mask, other=0.0)

    for ri in range(cur_len):
        i = cur_len - 1 - ri
        t = chunk_start + i
        q_ = tl.load(q_ptr + q_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        k_ = tl.load(k_ptr + k_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        e_ = tl.load(e_ptr + e_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        g_ = tl.load(g_ptr + g_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        d_ = tl.exp(g_)
        beta = tl.load(beta_ptr + beta_base + t)
        gamma = tl.load(gamma_ptr + gamma_base + t)
        v_ = tl.load(v_ptr + v_base + t * V + v_start + v_idx, mask=v_mask, other=0.0).to(tl.float32)
        dO = tl.load(grad_out_ptr + out_base + t * V + v_start + v_idx, mask=v_mask, other=0.0).to(tl.float32)

        # S_t and S_prev from tmp
        t_off = tmp_base + i * K * V
        t_ptrs = tmp_ptr + t_off + k_idx[:, None] * V + (v_start + v_idx[None, :])
        S_t = tl.load(t_ptrs, mask=ds_mask, other=0.0)
        if i == 0:
            S_prev = S_start
        else:
            p_off = tmp_base + (i - 1) * K * V
            p_ptrs = tmp_ptr + p_off + k_idx[:, None] * V + (v_start + v_idx[None, :])
            S_prev = tl.load(p_ptrs, mask=ds_mask, other=0.0)

        S_a = S_prev * d_[:, None]
        ea = tl.sum(e_[:, None] * S_a, axis=0)
        S_b = S_a - gamma * e_[:, None] * ea[None, :]
        pred = tl.sum(k_[:, None] * S_b, axis=0)
        u_ = v_ - pred

        ds_out = scale * q_[:, None] * dO[None, :]
        ds_t = ds + ds_out

        w_k = tl.sum(k_[:, None] * ds_t, axis=0)
        # grad_v direct store (no reduction)
        grad_v = beta * w_k
        tl.store(grad_v_ptr + grad_v_base + t * V + v_start + v_idx, grad_v, mask=v_mask)

        # partials that need sum across V tiles -> atomic add
        beta_part = tl.sum(w_k * u_, axis=0)
        tl.atomic_add(grad_beta_ptr + grad_b_base + t, beta_part)

        tmp1 = tl.sum(ds_t * u_[None, :], axis=1)
        tmp2 = tl.sum(S_b * w_k[None, :], axis=1)
        grad_k_part = beta * (tmp1 - tmp2)
        for kk in range(0, BLOCK_K):  # will be unrolled? avoid loop - use vector atomic
            pass
        # vector atomic add for K dimension
        gk_ptrs = grad_k_ptr + grad_k_base + t * K + k_idx
        tl.atomic_add(gk_ptrs, grad_k_part, mask=k_mask)

        ds_b = ds_t - beta * k_[:, None] * w_k[None, :]
        we_b = tl.sum(e_[:, None] * ds_b, axis=0)
        gamma_part = -tl.sum(ea * we_b, axis=0)
        tl.atomic_add(grad_gamma_ptr + grad_b_base + t, gamma_part)

        ds_a = ds_b - gamma * e_[:, None] * we_b[None, :]

        dq_part = scale * tl.sum(S_t * dO[None, :], axis=1)
        dq_ptrs = grad_q_ptr + grad_q_base + t * K + k_idx
        tl.atomic_add(dq_ptrs, dq_part, mask=k_mask)

        dg_part = tl.sum(ds_a * S_prev, axis=1) * d_
        dg_ptrs = grad_g_ptr + grad_g_base + t * K + k_idx
        tl.atomic_add(dg_ptrs, dg_part, mask=k_mask)

        de_tmp1 = tl.sum(ds_b * ea[None, :], axis=1)
        de_tmp2 = tl.sum(S_a * we_b[None, :], axis=1)
        de_part = -gamma * (de_tmp1 + de_tmp2)
        de_ptrs = grad_e_ptr + grad_e_base + t * K + k_idx
        tl.atomic_add(de_ptrs, de_part, mask=k_mask)

        ds = ds_a * d_[:, None]

    # write back ds for previous chunk
    tl.store(ds_ptrs, ds, mask=ds_mask)

class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale_f):
        B,H,T,K = q.shape
        V = v.shape[-1]
        BH = B*H
        BLOCK_K = _next_pow2(K)
        BLOCK_V = 32 if V > 32 else _next_pow2(V)
        NTILES = (V + BLOCK_V - 1)//BLOCK_V
        CHUNK = 64
        if T < CHUNK:
            CHUNK = T if T>0 else 1
        Nc = (T + CHUNK -1)//CHUNK if T>0 else 0
        out = torch.empty((B,H,T,V), device=q.device, dtype=torch.float32)
        if T==0:
            ctx.save_for_backward(q,k,v,e,g,beta,gamma)
            ctx.BH=BH; ctx.T=T; ctx.K=K; ctx.V=V; ctx.scale_f=scale_f
            ctx.BLOCK_K=BLOCK_K; ctx.BLOCK_V=BLOCK_V; ctx.NTILES=NTILES; ctx.CHUNK=CHUNK; ctx.Nc=Nc
            ctx.B=B; ctx.H=H; ctx.chk=None; ctx.tmp=None
            return out
        chk = torch.zeros((BH, Nc+1, K, V), device=q.device, dtype=torch.float32)
        grid = (BH*NTILES,)
        fwd_out_chk_kernel[grid](q,k,v,e,g,beta,gamma,out,chk, BH,T,K,V,scale_f, BLOCK_K,BLOCK_V,NTILES,CHUNK,Nc)
        ctx.save_for_backward(q,k,v,e,g,beta,gamma)
        ctx.BH=BH; ctx.T=T; ctx.K=K; ctx.V=V; ctx.scale_f=scale_f
        ctx.BLOCK_K=BLOCK_K; ctx.BLOCK_V=BLOCK_V; ctx.NTILES=NTILES; ctx.CHUNK=CHUNK; ctx.Nc=Nc
        ctx.B=B; ctx.H=H; ctx.chk=chk
        tmp = torch.empty((BH, CHUNK, K, V), device=q.device, dtype=torch.float32)
        ctx.tmp=tmp
        return out
    @staticmethod
    def backward(ctx, grad_out):
        q,k,v,e,g,beta,gamma = ctx.saved_tensors
        B=ctx.B; H=ctx.H; T=ctx.T; K=ctx.K; V=ctx.V; BH=ctx.BH; scale_f=ctx.scale_f
        BLOCK_K=ctx.BLOCK_K; BLOCK_V=ctx.BLOCK_V; NTILES=ctx.NTILES; CHUNK=ctx.CHUNK; Nc=ctx.Nc
        chk=ctx.chk; tmp=ctx.tmp
        if T==0:
            return (torch.zeros_like(q),torch.zeros_like(k),torch.zeros_like(v),torch.zeros_like(e),torch.zeros_like(g),torch.zeros_like(beta),torch.zeros_like(gamma),None)
        grad_out_c = grad_out.contiguous().to(torch.float32)
        grad_q = torch.zeros((B,H,T,K), device=q.device, dtype=torch.float32)
        grad_k = torch.zeros((B,H,T,K), device=q.device, dtype=torch.float32)
        grad_e = torch.zeros((B,H,T,K), device=q.device, dtype=torch.float32)
        grad_g = torch.zeros((B,H,T,K), device=q.device, dtype=torch.float32)
        grad_beta = torch.zeros((B,H,T), device=q.device, dtype=torch.float32)
        grad_gamma = torch.zeros((B,H,T), device=q.device, dtype=torch.float32)
        grad_v = torch.zeros((B,H,T,V), device=q.device, dtype=torch.float32)
        ds = torch.zeros((BH, K, V), device=q.device, dtype=torch.float32)
        grid = (BH*NTILES,)
        # iterate chunks reverse
        for chunk_idx in range(Nc-1, -1, -1):
            chunk_start = chunk_idx*CHUNK
            cur_len = T - chunk_start
            if cur_len > CHUNK:
                cur_len = CHUNK
            chunk_fwd_bwd_kernel[grid](q,k,v,e,g,beta,gamma, chk, tmp, ds, grad_out_c, grad_q, grad_k, grad_e, grad_g, grad_beta, grad_gamma, grad_v, BH,T,K,V,scale_f, BLOCK_K,BLOCK_V,NTILES,CHUNK,Nc, chunk_idx, cur_len, chunk_start)
        return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype), grad_e.to(e.dtype), grad_g.to(g.dtype), grad_beta.to(beta.dtype), grad_gamma.to(gamma.dtype), None

def _fast_forward(q,k,v,e,g,beta,gamma,scale_f):
    B,H,T,K = q.shape
    V = v.shape[-1]
    if T==0:
        return torch.empty((B,H,0,V), device=q.device, dtype=torch.float32)
    q_c=q.contiguous(); k_c=k.contiguous(); v_c=v.contiguous(); e_c=e.contiguous(); g_c=g.contiguous(); beta_c=beta.contiguous(); gamma_c=gamma.contiguous()
    BH=B*H
    BLOCK_K=_next_pow2(K)
    BLOCK_V=32 if V>32 else _next_pow2(V)
    NTILES=(V+BLOCK_V-1)//BLOCK_V
    out=torch.empty((B,H,T,V), device=q.device, dtype=torch.float32)
    grid=(BH*NTILES,)
    fwd_kernel[grid](q_c,k_c,v_c,e_c,g_c,beta_c,gamma_c,out, BH,T,K,V,scale_f, BLOCK_K,BLOCK_V,NTILES)
    return out

def kernel_function(q,k,v,e,g,beta,gamma,scale):
    scale_f=float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
    needs_grad = torch.is_grad_enabled() and any(isinstance(x, torch.Tensor) and x.requires_grad for x in (q,k,v,e,g,beta,gamma))
    if not needs_grad:
        return _fast_forward(q,k,v,e,g,beta,gamma,scale_f)
    q_c=q.contiguous(); k_c=k.contiguous(); v_c=v.contiguous(); e_c=e.contiguous(); g_c=g.contiguous(); beta_c=beta.contiguous(); gamma_c=gamma.contiguous()
    return _Func.apply(q_c,k_c,v_c,e_c,g_c,beta_c,gamma_c, scale_f)
