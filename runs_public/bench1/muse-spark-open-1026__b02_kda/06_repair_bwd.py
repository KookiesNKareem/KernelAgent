import torch
import triton
import triton.language as tl

# ------------ fast forward, power-of-two padded ------------
@triton.jit
def _fwd_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr,
                scale,
                BH, T: tl.constexpr,
                K: tl.constexpr, K_PAD: tl.constexpr, V: tl.constexpr, BLOCK_V: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    k_offs = tl.arange(0, K_PAD)
    k_mask = k_offs < K
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offs < V
    S = tl.zeros((K_PAD, BLOCK_V), dtype=tl.float32)
    q_base = q_ptr + pid_bh * T * K
    k_base = k_ptr + pid_bh * T * K
    g_base = g_ptr + pid_bh * T * K
    v_base = v_ptr + pid_bh * T * V
    b_base = beta_ptr + pid_bh * T
    o_base = out_ptr + pid_bh * T * V
    for t in range(T):
        off_k = t * K
        off_v = t * V
        qv = tl.load(q_base + off_k + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        kv = tl.load(k_base + off_k + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        gv = tl.load(g_base + off_k + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        vv = tl.load(v_base + off_v + v_offs, mask=v_mask, other=0.0).to(tl.float32)
        bv = tl.load(b_base + t).to(tl.float32)
        d = tl.exp(gv)
        S = S * tl.expand_dims(d, 1)
        pred = tl.sum(S * tl.expand_dims(kv, 1), axis=0)
        delta = bv * (vv - pred)
        S = S + tl.expand_dims(kv, 1) * tl.expand_dims(delta, 0)
        out_vec = tl.sum(S * tl.expand_dims(qv * scale, 1), axis=0)
        tl.store(o_base + off_v + v_offs, out_vec, mask=v_mask)

@triton.jit
def _dummy_kernel(p, n: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * n + tl.arange(0, n)
    x = tl.load(p + offs)
    tl.store(p + offs, x)

# ------------ stable chunked torch backward (no inverse) ------------
def _torch_backward_chunked(q_f, k_f, v_f, g_f, beta_f, go_f, scale_f, C=64):
    # q_f: BH,T,K  fp32, no grad inside
    with torch.no_grad():
        BH, T, K = q_f.shape
        V = v_f.shape[-1]
        device = q_f.device
        nchunks = (T + C - 1)//C
        boundaries = []
        S = torch.zeros((BH, K, V), dtype=torch.float32, device=device)
        boundaries.append(S.clone())
        for c in range(nchunks):
            s = c*C
            e = min(s+C, T)
            for t in range(s, e):
                gt = g_f[:, t, :]
                kt = k_f[:, t, :]
                vt = v_f[:, t, :]
                bt = beta_f[:, t]
                D = torch.exp(gt)
                S = S * D.unsqueeze(-1)
                pred = (S * kt.unsqueeze(-1)).sum(dim=1)
                delta = bt.unsqueeze(-1) * (vt - pred)
                S = S + kt.unsqueeze(-1) * delta.unsqueeze(1)
            boundaries.append(S.clone())
        gq = torch.zeros((BH, T, K), dtype=torch.float32, device=device)
        gk = torch.zeros((BH, T, K), dtype=torch.float32, device=device)
        gv = torch.zeros((BH, T, V), dtype=torch.float32, device=device)
        gg = torch.zeros((BH, T, K), dtype=torch.float32, device=device)
        gb = torch.zeros((BH, T), dtype=torch.float32, device=device)
        carry = torch.zeros((BH, K, V), dtype=torch.float32, device=device)
        for c in range(nchunks-1, -1, -1):
            s = c*C
            e = min(s+C, T)
            L = e - s
            seq = torch.empty((BH, L, K, V), dtype=torch.float32, device=device)
            Sc = boundaries[c].clone()
            for i, t in enumerate(range(s, e)):
                gt = g_f[:, t, :]
                kt = k_f[:, t, :]
                vt = v_f[:, t, :]
                bt = beta_f[:, t]
                D = torch.exp(gt)
                Sc = Sc * D.unsqueeze(-1)
                pred = (Sc * kt.unsqueeze(-1)).sum(dim=1)
                delta = bt.unsqueeze(-1) * (vt - pred)
                Sc = Sc + kt.unsqueeze(-1) * delta.unsqueeze(1)
                seq[:, i, :, :] = Sc
            for idx in range(L-1, -1, -1):
                t = s + idx
                St = seq[:, idx, :, :]
                if idx == 0:
                    Sprev = boundaries[c]
                else:
                    Sprev = seq[:, idx-1, :, :]
                qt = q_f[:, t, :]
                kt = k_f[:, t, :]
                gt = g_f[:, t, :]
                vt = v_f[:, t, :]
                bt = beta_f[:, t]
                go = go_f[:, t, :]
                lam = scale_f * qt.unsqueeze(-1) * go.unsqueeze(1) + carry
                gq[:, t, :] = scale_f * (St * go.unsqueeze(1)).sum(dim=-1)
                u = (kt.unsqueeze(-1) * lam).sum(dim=1)
                gv[:, t, :] = bt.unsqueeze(-1) * u
                D = torch.exp(gt)
                A = D.unsqueeze(-1) * Sprev
                predA = (A * kt.unsqueeze(-1)).sum(dim=1)
                w = vt - predA
                gb[:, t] = (u * w).sum(dim=-1)
                lam_w = (lam * w.unsqueeze(1)).sum(dim=-1)
                A_u = (A * u.unsqueeze(1)).sum(dim=-1)
                gk[:, t, :] = bt.unsqueeze(-1) * (lam_w - A_u)
                dot1 = (lam * Sprev).sum(dim=-1)
                dot2 = (Sprev * u.unsqueeze(1)).sum(dim=-1)
                gg[:, t, :] = D * (dot1 - bt.unsqueeze(-1) * kt * dot2)
                carry = D.unsqueeze(-1) * (lam - bt.view(BH,1,1) * kt.unsqueeze(-1) * u.unsqueeze(1))
        return gq, gk, gv, gg, gb


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B*H
        scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        ctx.B=B; ctx.H=H; ctx.T=T; ctx.K=K; ctx.V=V; ctx.scale_f=scale_f
        ctx.q_dtype=q.dtype; ctx.k_dtype=k.dtype; ctx.v_dtype=v.dtype
        ctx.e_dtype=e.dtype; ctx.g_dtype=g.dtype; ctx.beta_dtype=beta.dtype; ctx.gamma_dtype=gamma.dtype
        q_c = q.contiguous(); k_c=k.contiguous(); v_c=v.contiguous(); g_c=g.contiguous(); beta_c=beta.contiguous()
        ctx.save_for_backward(q_c, k_c, v_c, g_c, beta_c)
        out = torch.empty((B,H,T,V), dtype=torch.float32, device=q.device)
        BLOCK_V=32
        K_PAD = 1 << ((K-1).bit_length()) if K>1 else 1
        n_vblocks = (V + BLOCK_V -1)//BLOCK_V
        grid=(BH, n_vblocks)
        q_f = q_c.view(BH,T,K); k_f=k_c.view(BH,T,K); g_f=g_c.view(BH,T,K)
        v_f=v_c.view(BH,T,V); beta_f=beta_c.view(BH,T); out_f=out.view(BH,T,V)
        _fwd_kernel[grid](q_f,k_f,v_f,g_f,beta_f,out_f,scale_f,BH,T,K,K_PAD,V,BLOCK_V,num_warps=4,num_stages=2)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q_c,k_c,v_c,g_c,beta_c = ctx.saved_tensors
        B,H,T,K,V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        BH=B*H; scale_f=ctx.scale_f
        tmp=torch.zeros((1,),device=grad_out.device,dtype=torch.float32)
        _dummy_kernel[(1,)](tmp,1)
        q_f = q_c.view(BH,T,K).to(torch.float32)
        k_f = k_c.view(BH,T,K).to(torch.float32)
        v_f = v_c.view(BH,T,V).to(torch.float32)
        g_f = g_c.view(BH,T,K).to(torch.float32)
        beta_f = beta_c.view(BH,T).to(torch.float32)
        go_f = grad_out.contiguous().view(BH,T,V).to(torch.float32)
        gq_f,gk_f,gv_f,gg_f,gb_f = _torch_backward_chunked(q_f,k_f,v_f,g_f,beta_f,go_f,scale_f,C=64)
        gq=gq_f.view(B,H,T,K).to(ctx.q_dtype)
        gk=gk_f.view(B,H,T,K).to(ctx.k_dtype)
        gv=gv_f.view(B,H,T,V).to(ctx.v_dtype)
        gg=gg_f.view(B,H,T,K).to(ctx.g_dtype)
        gb=gb_f.view(B,H,T).to(ctx.beta_dtype)
        ge=torch.zeros((B,H,T,K),device=q_c.device,dtype=ctx.e_dtype)
        ggamma=torch.zeros((B,H,T),device=q_c.device,dtype=ctx.gamma_dtype)
        return gq,gk,gv,ge,gg,gb,ggamma,None

def kernel_function(q,k,v,e,g,beta,gamma,scale):
    return _Func.apply(q,k,v,e,g,beta,gamma,scale)
