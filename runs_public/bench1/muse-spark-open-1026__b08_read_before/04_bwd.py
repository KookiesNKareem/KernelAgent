import torch
import triton
import triton.language as tl

# ---------- forward fast resident ----------
@triton.jit
def fwd_fast_kernel(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,out_ptr,state_ptr,
    BH,T,K,V, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, SCALE: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_vb = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_offs = pid_vb * BLOCK_V + offs_v
    mask_v = v_offs < V
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    for t in range(0, T):
        q = tl.load(q_ptr + (pid_bh*T + t)*K + offs_k).to(tl.float32) * SCALE
        k = tl.load(k_ptr + (pid_bh*T + t)*K + offs_k).to(tl.float32)
        g = tl.load(g_ptr + (pid_bh*T + t)*K + offs_k).to(tl.float32)
        d = tl.exp(g)
        v = tl.load(v_ptr + (pid_bh*T + t)*V + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + pid_bh*T + t).to(tl.float32)
        S = S * d[:, None]
        out = tl.sum(S * q[:, None], axis=0)
        pred = tl.sum(S * k[:, None], axis=0)
        tl.store(out_ptr + (pid_bh*T + t)*V + v_offs, out, mask=mask_v)
        S = S + k[:, None] * (beta * (v - pred))[None, :]
    # save final S for backward
    offs_k2 = tl.arange(0, BLOCK_K)[:, None]
    offs_v2 = (pid_vb*BLOCK_V + tl.arange(0, BLOCK_V))[None, :]
    s_ptr = state_ptr + pid_bh*K*V + offs_k2*V + offs_v2
    mask = (offs_k2 < K) & (offs_v2 < V)
    tl.store(s_ptr, S, mask=mask)

@triton.jit
def fwd_generic_kernel(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,out_ptr,state_ptr,
    BH,T,K,V,NK: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, SCALE: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_vb = tl.program_id(1)
    offs_v = tl.arange(0, BV)
    v_offs = pid_vb*BV + offs_v
    mask_v = v_offs < V
    for t in range(0, T):
        beta = tl.load(beta_ptr + pid_bh*T + t).to(tl.float32)
        v = tl.load(v_ptr + (pid_bh*T + t)*V + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        # phase A: decay + partial out/pred
        out_acc = tl.zeros((BV,), dtype=tl.float32)
        pred_acc = tl.zeros((BV,), dtype=tl.float32)
        for ki in range(NK):
            k_offs = ki*BK + tl.arange(0, BK)
            mask_k = k_offs < K
            S = tl.load(state_ptr + pid_bh*K*V + k_offs[:,None]*V + v_offs[None,:], mask=(k_offs[:,None]<K)&(v_offs[None,:]<V), other=0.0)
            qk = tl.load(q_ptr + (pid_bh*T + t)*K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
            kk = tl.load(k_ptr + (pid_bh*T + t)*K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
            gg = tl.load(g_ptr + (pid_bh*T + t)*K + k_offs, mask=mask_k, other=0.0).to(tl.float32)
            d = tl.exp(gg)
            S = S * d[:,None]
            tl.store(state_ptr + pid_bh*K*V + k_offs[:,None]*V + v_offs[None,:], S, mask=(k_offs[:,None]<K)&(v_offs[None,:]<V))
            out_acc += tl.sum(S * (qk*SCALE)[:,None], axis=0)
            pred_acc += tl.sum(S * kk[:,None], axis=0)
        tl.store(out_ptr + (pid_bh*T + t)*V + v_offs, out_acc, mask=mask_v)
        # phase B: write
        for ki in range(NK):
            k_offs = ki*BK + tl.arange(0, BK)
            kk = tl.load(k_ptr + (pid_bh*T + t)*K + k_offs, mask=k_offs<K, other=0.0).to(tl.float32)
            S = tl.load(state_ptr + pid_bh*K*V + k_offs[:,None]*V + v_offs[None,:], mask=(k_offs[:,None]<K)&(v_offs[None,:]<V), other=0.0)
            delta = beta * (v - pred_acc)
            S = S + kk[:,None] * delta[None,:]
            tl.store(state_ptr + pid_bh*K*V + k_offs[:,None]*V + v_offs[None,:], S, mask=(k_offs[:,None]<K)&(v_offs[None,:]<V))

# ---------- backward fast resident (inverse) ----------
@triton.jit
def bwd_fast_kernel(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,do_ptr,state_ptr,gstate_ptr,dq_ptr,dk_ptr,dg_ptr,dv_ptr,dbeta_ptr,
    BH,T,K,V, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, SCALE: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_vb = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_offs = pid_vb*BLOCK_V + offs_v
    mask_v = v_offs < V
    # load initial S = S_T , G=0
    S = tl.load(state_ptr + pid_bh*K*V + offs_k[:,None]*V + v_offs[None,:], mask=(offs_k[:,None]<K)&(v_offs[None,:]<V), other=0.0)
    G = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    for ri in range(0, T):
        t = T - 1 - ri
        q = tl.load(q_ptr + (pid_bh*T + t)*K + offs_k).to(tl.float32)
        k = tl.load(k_ptr + (pid_bh*T + t)*K + offs_k).to(tl.float32)
        g = tl.load(g_ptr + (pid_bh*T + t)*K + offs_k).to(tl.float32)
        d = tl.exp(g)
        inv_d = tl.exp(-g)
        beta = tl.load(beta_ptr + pid_bh*T + t).to(tl.float32)
        v = tl.load(v_ptr + (pid_bh*T + t)*V + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + (pid_bh*T + t)*V + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        s_ks = tl.sum(S * k[:,None], axis=0)
        u = tl.sum(G * k[:,None], axis=0)
        kTx = s_ks - beta * v
        den = 1.0 - beta
        # avoid div0, beta in (0,1)
        c = beta / (den + 1e-12)
        delta = c * kTx - beta * v
        tilde = S + k[:,None] * delta[None,:]
        pred = kTx / (den + 1e-12)
        tilde_G = G + SCALE * q[:,None] * do[None,:] - beta * k[:,None] * u[None,:]
        # grads
        dv = beta * u
        tl.store(dv_ptr + (pid_bh*T + t)*V + v_offs, dv, mask=mask_v)
        dq = SCALE * tl.sum(tilde * do[None,:], axis=1)
        tl.atomic_add(dq_ptr + (pid_bh*T + t)*K + offs_k, dq)
        term1 = tl.sum(G * (v - pred)[None,:], axis=1)
        term2 = tl.sum(tilde * u[None,:], axis=1)
        dk = beta * (term1 - term2)
        tl.atomic_add(dk_ptr + (pid_bh*T + t)*K + offs_k, dk)
        dg = tl.sum(tilde_G * tilde, axis=1)
        tl.atomic_add(dg_ptr + (pid_bh*T + t)*K + offs_k, dg)
        db = tl.sum(u * (v - pred), axis=0)
        tl.atomic_add(dbeta_ptr + pid_bh*T + t, db)
        # step
        G = tilde_G * d[:,None]
        S = tilde * inv_d[:,None]

@triton.jit
def bwd_generic_kernel(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,do_ptr,state_ptr,gstate_ptr,dq_ptr,dk_ptr,dg_ptr,dv_ptr,dbeta_ptr,
    BH,T,K,V,NK: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, SCALE: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_vb = tl.program_id(1)
    offs_v = tl.arange(0, BV)
    v_offs = pid_vb*BV + offs_v
    mask_v = v_offs < V
    for ri in range(0, T):
        t = T - 1 - ri
        beta = tl.load(beta_ptr + pid_bh*T + t).to(tl.float32)
        v = tl.load(v_ptr + (pid_bh*T + t)*V + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + (pid_bh*T + t)*V + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        # reduce s and u over K
        s_acc = tl.zeros((BV,), dtype=tl.float32)
        u_acc = tl.zeros((BV,), dtype=tl.float32)
        for ki in range(NK):
            k_offs = ki*BK + tl.arange(0, BK)
            k = tl.load(k_ptr + (pid_bh*T + t)*K + k_offs, mask=k_offs<K, other=0.0).to(tl.float32)
            S = tl.load(state_ptr + pid_bh*K*V + k_offs[:,None]*V + v_offs[None,:], mask=(k_offs[:,None]<K)&(v_offs[None,:]<V), other=0.0)
            Gc = tl.load(gstate_ptr + pid_bh*K*V + k_offs[:,None]*V + v_offs[None,:], mask=(k_offs[:,None]<K)&(v_offs[None,:]<V), other=0.0)
            s_acc += tl.sum(S * k[:,None], axis=0)
            u_acc += tl.sum(Gc * k[:,None], axis=0)
        kTx = s_acc - beta * v
        den = 1.0 - beta
        c = beta / (den + 1e-12)
        delta = c * kTx - beta * v
        pred = kTx / (den + 1e-12)
        # grad beta partial
        db = tl.sum(u_acc * (v - pred), axis=0)
        tl.atomic_add(dbeta_ptr + pid_bh*T + t, db)
        dv = beta * u_acc
        tl.store(dv_ptr + (pid_bh*T + t)*V + v_offs, dv, mask=mask_v)
        for ki in range(NK):
            k_offs = ki*BK + tl.arange(0, BK)
            k = tl.load(k_ptr + (pid_bh*T + t)*K + k_offs, mask=k_offs<K, other=0.0).to(tl.float32)
            q = tl.load(q_ptr + (pid_bh*T + t)*K + k_offs, mask=k_offs<K, other=0.0).to(tl.float32)
            g = tl.load(g_ptr + (pid_bh*T + t)*K + k_offs, mask=k_offs<K, other=0.0).to(tl.float32)
            d = tl.exp(g)
            inv_d = tl.exp(-g)
            S = tl.load(state_ptr + pid_bh*K*V + k_offs[:,None]*V + v_offs[None,:], mask=(k_offs[:,None]<K)&(v_offs[None,:]<V), other=0.0)
            Gc = tl.load(gstate_ptr + pid_bh*K*V + k_offs[:,None]*V + v_offs[None,:], mask=(k_offs[:,None]<K)&(v_offs[None,:]<V), other=0.0)
            tilde = S + k[:,None]*delta[None,:]
            tilde_G = Gc + SCALE * q[:,None]*do[None,:] - beta * k[:,None]*u_acc[None,:]
            dq = SCALE * tl.sum(tilde * do[None,:], axis=1)
            tl.atomic_add(dq_ptr + (pid_bh*T + t)*K + k_offs, dq, mask=k_offs<K)
            term1 = tl.sum(Gc * (v - pred)[None,:], axis=1)
            term2 = tl.sum(tilde * u_acc[None,:], axis=1)
            dk = beta * (term1 - term2)
            tl.atomic_add(dk_ptr + (pid_bh*T + t)*K + k_offs, dk, mask=k_offs<K)
            dg = tl.sum(tilde_G * tilde, axis=1)
            tl.atomic_add(dg_ptr + (pid_bh*T + t)*K + k_offs, dg, mask=k_offs<K)
            Gnext = tilde_G * d[:,None]
            Snext = tilde * inv_d[:,None]
            tl.store(gstate_ptr + pid_bh*K*V + k_offs[:,None]*V + v_offs[None,:], Gnext, mask=(k_offs[:,None]<K)&(v_offs[None,:]<V))
            tl.store(state_ptr + pid_bh*K*V + k_offs[:,None]*V + v_offs[None,:], Snext, mask=(k_offs[:,None]<K)&(v_offs[None,:]<V))

class GatedDeltaFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q,k,v,e,g,beta,gamma,scale):
        B,H,T,K = q.shape; V=v.shape[-1]; BH=B*H
        device=q.device
        scale_f=float(scale.item()) if isinstance(scale,torch.Tensor) else float(scale)
        q_c=q.contiguous(); k_c=k.contiguous(); v_c=v.contiguous()
        g_c=g.contiguous(); beta_c=beta.contiguous()
        out=torch.empty(B,H,T,V, dtype=torch.float32, device=device)
        qf=q_c.view(BH,T,K); kf=k_c.view(BH,T,K); vf=v_c.view(BH,T,V); gf=g_c.view(BH,T,K); bf=beta_c.view(BH,T)
        outf=out.view(BH,T,V)
        ctx.scale_f=scale_f; ctx.BH=BH; ctx.B=B; ctx.H=H; ctx.T=T; ctx.K=K; ctx.V=V
        ctx.q_dtype=q.dtype; ctx.k_dtype=k.dtype; ctx.v_dtype=v.dtype; ctx.e_dtype=e.dtype
        ctx.g_dtype=g.dtype; ctx.beta_dtype=beta.dtype; ctx.gamma_dtype=gamma.dtype
        # fast path K==128
        if K==128 and V%32==0:
            BV=64 if V%64==0 else 32
            NV=V//BV
            state=torch.zeros(BH,K,V, dtype=torch.float32, device=device)
            grid=(BH,NV)
            fwd_fast_kernel[grid](qf,kf,vf,gf,bf,outf,state,BH,T,K,V,128,BV,scale_f,num_warps=4,num_stages=2)
            ctx.state=state; ctx.fast=True; ctx.BV=BV
            ctx.save_for_backward(qf,kf,vf,gf,bf)
            return out
        else:
            BK=32; BV=32; NK=(K+BK-1)//BK
            state=torch.zeros(BH,K,V, dtype=torch.float32, device=device)
            NV=(V+BV-1)//BV
            grid=(BH,NV)
            fwd_generic_kernel[grid](qf,kf,vf,gf,bf,outf,state,BH,T,K,V,NK,BK,BV,scale_f,num_warps=2,num_stages=1)
            ctx.state=state; ctx.fast=False; ctx.BK=BK; ctx.BV=BV; ctx.NK=NK
            ctx.save_for_backward(qf,kf,vf,gf,bf)
            return out
    @staticmethod
    def backward(ctx, grad_out):
        qf,kf,vf,gf,bf = ctx.saved_tensors
        BH,T,K,V = ctx.BH, ctx.T, ctx.K, ctx.V
        B,H = ctx.B, ctx.H
        scale_f=ctx.scale_f
        device=grad_out.device
        go=grad_out.contiguous().view(BH,T,V)
        dq=torch.zeros(BH,T,K, dtype=torch.float32, device=device)
        dk=torch.zeros(BH,T,K, dtype=torch.float32, device=device)
        dg=torch.zeros(BH,T,K, dtype=torch.float32, device=device)
        dv=torch.zeros(BH,T,V, dtype=torch.float32, device=device)
        dbeta=torch.zeros(BH,T, dtype=torch.float32, device=device)
        gstate=torch.zeros(BH,K,V, dtype=torch.float32, device=device)
        if ctx.fast:
            BV=ctx.BV
            NV=V//BV
            grid=(BH,NV)
            bwd_fast_kernel[grid](qf,kf,vf,gf,bf,go,ctx.state,gstate,dq,dk,dg,dv,dbeta,BH,T,K,V,128,BV,scale_f,num_warps=4,num_stages=1)
        else:
            BK=ctx.BK; BV=ctx.BV; NK=ctx.NK
            NV=(V+BV-1)//BV
            grid=(BH,NV)
            bwd_generic_kernel[grid](qf,kf,vf,gf,bf,go,ctx.state,gstate,dq,dk,dg,dv,dbeta,BH,T,K,V,NK,BK,BV,scale_f,num_warps=2,num_stages=1)
        # reshape and cast
        dq_out=dq.view(B,H,T,K).to(ctx.q_dtype)
        dk_out=dk.view(B,H,T,K).to(ctx.k_dtype)
        dv_out=dv.view(B,H,T,V).to(ctx.v_dtype)
        dg_out=dg.view(B,H,T,K).to(ctx.g_dtype)
        dbeta_out=dbeta.view(B,H,T).to(ctx.beta_dtype)
        de=torch.zeros(B,H,T,K, dtype=ctx.e_dtype, device=device)
        dgamma=torch.zeros(B,H,T, dtype=ctx.gamma_dtype, device=device)
        return dq_out, dk_out, dv_out, de, dg_out, dbeta_out, dgamma, None

def kernel_function(q,k,v,e,g,beta,gamma,scale):
    return GatedDeltaFunc.apply(q,k,v,e,g,beta,gamma,scale)
