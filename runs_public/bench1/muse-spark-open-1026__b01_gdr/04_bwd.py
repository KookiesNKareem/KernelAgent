import torch
import triton
import triton.language as tl

# forward chunk: S=d*S+d*b*k*(v-d*p) , o=scale*S^T q  V-outer / C-inner
@triton.jit
def _fwd_kernel(q_ptr,k_ptr,v_ptr,d_ptr,beta_ptr,out_ptr,state_ptr,
                scale, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                Kb: tl.constexpr, Vb: tl.constexpr, START: tl.constexpr,
                BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
    pid = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    for v_off in range(0, Vb, BLOCK_V):
        offs_v = v_off + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        state_base = state_ptr + pid * Kb * Vb
        s = tl.load(state_base + offs_k[:,None]*Vb + offs_v[None,:],
                    mask=mask_k[:,None]&mask_v[None,:], other=0.0).to(tl.float32)
        for i in range(0, BLOCK_C):
            cur = START + i
            if cur >= T:
                continue
            d_val = tl.load(d_ptr + pid*T + cur).to(tl.float32)
            b_val = tl.load(beta_ptr + pid*T + cur).to(tl.float32)
            q_base = q_ptr + pid*T*Kb + cur*Kb
            k_base = k_ptr + pid*T*Kb + cur*Kb
            q_i = tl.load(q_base + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            k_i = tl.load(k_base + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            v_base = v_ptr + pid*T*Vb + cur*Vb + v_off
            v_i = tl.load(v_base + tl.arange(0,BLOCK_V), mask=mask_v, other=0.0).to(tl.float32)
            p = tl.sum(k_i[:,None]*s, axis=0)
            coeff = b_val*(v_i - d_val*p)
            s = s*d_val + k_i[:,None]*coeff[None,:]
            o = tl.sum(q_i[:,None]*s, axis=0)*scale
            out_base = out_ptr + pid*T*V + cur*V + v_off
            tl.store(out_base + tl.arange(0,BLOCK_V), o, mask=mask_v)
        tl.store(state_base + offs_k[:,None]*Vb + offs_v[None,:], s,
                 mask=mask_k[:,None]&mask_v[None,:])

# recompute S_t for a chunk and store to tmp
@triton.jit
def _recompute_kernel(k_ptr,v_ptr,d_ptr,beta_ptr,cur_state_ptr,tmp_ptr,
                      T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                      Kb: tl.constexpr, Vb: tl.constexpr, START: tl.constexpr,
                      BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
    pid = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    for v_off in range(0, Vb, BLOCK_V):
        offs_v = v_off + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        cur_base = cur_state_ptr + pid*Kb*Vb
        s = tl.load(cur_base + offs_k[:,None]*Vb + offs_v[None,:],
                    mask=mask_k[:,None]&mask_v[None,:], other=0.0).to(tl.float32)
        for i in range(0, BLOCK_C):
            cur = START + i
            if cur >= T:
                continue
            d_val = tl.load(d_ptr + pid*T + cur).to(tl.float32)
            b_val = tl.load(beta_ptr + pid*T + cur).to(tl.float32)
            k_base = k_ptr + pid*T*Kb + cur*Kb
            k_i = tl.load(k_base + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            v_base = v_ptr + pid*T*Vb + cur*Vb + v_off
            v_i = tl.load(v_base + tl.arange(0,BLOCK_V), mask=mask_v, other=0.0).to(tl.float32)
            p = tl.sum(k_i[:,None]*s, axis=0)
            coeff = b_val*(v_i - d_val*p)
            s = s*d_val + k_i[:,None]*coeff[None,:]
            tmp_base = tmp_ptr + pid*BLOCK_C*Kb*Vb + i*Kb*Vb
            tl.store(tmp_base + offs_k[:,None]*Vb + offs_v[None,:], s,
                     mask=mask_k[:,None]&mask_v[None,:])
        tl.store(cur_base + offs_k[:,None]*Vb + offs_v[None,:], s,
                 mask=mask_k[:,None]&mask_v[None,:])

# backward for one chunk, reverse order. updates g_state in place.
@triton.jit
def _bwd_kernel(q_ptr,k_ptr,v_ptr,d_ptr,beta_ptr,go_ptr,tmp_ptr,s0_ptr,g_state_ptr,
                dq_ptr,dk_ptr,dv_ptr,db_ptr,dg_ptr,
                scale, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                Kb: tl.constexpr, Vb: tl.constexpr, START: tl.constexpr,
                BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
    pid = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    for idx in range(0, BLOCK_C):
        i = BLOCK_C - 1 - idx
        cur = START + i
        if cur >= T:
            continue
        d_val = tl.load(d_ptr + pid*T + cur).to(tl.float32)
        b_val = tl.load(beta_ptr + pid*T + cur).to(tl.float32)
        k_base = k_ptr + pid*T*Kb + cur*Kb
        q_base = q_ptr + pid*T*Kb + cur*Kb
        k_i = tl.load(k_base + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        q_i = tl.load(q_base + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        gp = tl.zeros((BLOCK_K,), dtype=tl.float32)
        sr = tl.zeros((BLOCK_K,), dtype=tl.float32)
        gv = tl.zeros((BLOCK_K,), dtype=tl.float32)
        dq_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        tr_acc = 0.0
        rp_acc = 0.0
        rv_acc = 0.0
        for v_off in range(0, Vb, BLOCK_V):
            offs_v = v_off + tl.arange(0, BLOCK_V)
            mask_v = offs_v < V
            v_base = v_ptr + pid*T*Vb + cur*Vb + v_off
            v_i = tl.load(v_base + tl.arange(0, BLOCK_V), mask=mask_v, other=0.0).to(tl.float32)
            go_base = go_ptr + pid*T*V + cur*V + v_off
            go_i = tl.load(go_base + tl.arange(0, BLOCK_V), mask=mask_v, other=0.0).to(tl.float32)
            gs_base = g_state_ptr + pid*Kb*Vb
            g_next = tl.load(gs_base + offs_k[:,None]*Vb + offs_v[None,:],
                             mask=mask_k[:,None]&mask_v[None,:], other=0.0).to(tl.float32)
            g_t = g_next + scale * q_i[:,None] * go_i[None,:]
            # S_prev / S_cur
            if i == 0:
                s_prev = tl.load(s0_ptr + pid*Kb*Vb + offs_k[:,None]*Vb + offs_v[None,:],
                                 mask=mask_k[:,None]&mask_v[None,:], other=0.0).to(tl.float32)
            else:
                tp_prev = tmp_ptr + pid*BLOCK_C*Kb*Vb + (i-1)*Kb*Vb
                s_prev = tl.load(tp_prev + offs_k[:,None]*Vb + offs_v[None,:],
                                 mask=mask_k[:,None]&mask_v[None,:], other=0.0).to(tl.float32)
            tp_cur = tmp_ptr + pid*BLOCK_C*Kb*Vb + i*Kb*Vb
            s_cur = tl.load(tp_cur + offs_k[:,None]*Vb + offs_v[None,:],
                            mask=mask_k[:,None]&mask_v[None,:], other=0.0).to(tl.float32)
            p_tile = tl.sum(k_i[:,None]*s_prev, axis=0)
            r_tile = tl.sum(g_t * k_i[:,None], axis=0)
            # dv
            dv_base = dv_ptr + pid*T*V + cur*V + v_off
            tl.store(dv_base + tl.arange(0, BLOCK_V), b_val * r_tile, mask=mask_v)
            gp = gp + tl.sum(g_t * p_tile[None,:], axis=1)
            sr = sr + tl.sum(s_prev * r_tile[None,:], axis=1)
            gv = gv + tl.sum(g_t * v_i[None,:], axis=1)
            dq_acc = dq_acc + tl.sum(s_cur * go_i[None,:], axis=1)
            tr_acc = tr_acc + tl.sum(g_t * s_prev)
            rp_acc = rp_acc + tl.sum(r_tile * p_tile)
            rv_acc = rv_acc + tl.sum(r_tile * v_i)
            g_new = d_val * (g_t - b_val * k_i[:,None] * r_tile[None,:])
            tl.store(gs_base + offs_k[:,None]*Vb + offs_v[None,:], g_new,
                     mask=mask_k[:,None]&mask_v[None,:])
        dq_val = scale * dq_acc
        tl.store(dq_ptr + pid*T*K + cur*K + offs_k, dq_val, mask=mask_k)
        dk_val = b_val * gv - d_val * b_val * (gp + sr)
        tl.store(dk_ptr + pid*T*K + cur*K + offs_k, dk_val, mask=mask_k)
        tl.store(db_ptr + pid*T + cur, rv_acc - d_val * rp_acc)
        tl.store(dg_ptr + pid*T + cur, d_val * (tr_acc - b_val * rp_acc))

class DeltaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q,k,v,e,g,beta,gamma,scale):
        B,H,T,K = q.shape
        Vd = v.shape[-1]
        BH = B*H
        scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        Kb = ((K+31)//32)*32
        Vb = ((Vd+31)//32)*32
        if Kb==0: Kb=32
        if Vb==0: Vb=32
        BLOCK_C=32; BLOCK_K=Kb; BLOCK_V=32
        q_c = q.contiguous(); k_c=k.contiguous(); v_c=v.contiguous()
        g_c=g.contiguous().to(torch.float32); beta_c=beta.contiguous().to(torch.float32)
        q_r=q_c.view(BH,T,K); k_r=k_c.view(BH,T,K); v_r=v_c.view(BH,T,Vd)
        g_r=g_c.view(BH,T); beta_r=beta_c.view(BH,T)
        q_pad=torch.zeros((BH,T,Kb),device=q.device,dtype=q_r.dtype)
        k_pad=torch.zeros((BH,T,Kb),device=q.device,dtype=k_r.dtype)
        v_pad=torch.zeros((BH,T,Vb),device=q.device,dtype=v_r.dtype)
        q_pad[:,:,:K]=q_r; k_pad[:,:,:K]=k_r; v_pad[:,:,:Vd]=v_r
        d_pad=torch.exp(g_r)
        out_pad=torch.empty((BH,T,Vd),device=q.device,dtype=torch.float32)
        state=torch.zeros((BH,Kb,Vb),device=q.device,dtype=torch.float32)
        nC=(T+BLOCK_C-1)//BLOCK_C
        chunk_states=torch.empty((BH,nC,Kb,Vb),device=q.device,dtype=torch.float32)
        grid=(BH,)
        for ck in range(nC):
            START=ck*BLOCK_C
            chunk_states[:,ck]=state
            _fwd_kernel[grid](q_pad,k_pad,v_pad,d_pad,beta_r,out_pad,state,scale_f,T,K,Vd,Kb,Vb,START,BLOCK_C,BLOCK_K,BLOCK_V)
        out=out_pad.view(B,H,T,Vd)
        ctx.save_for_backward(q_pad,k_pad,v_pad,d_pad,beta_r,chunk_states)
        ctx.BH=BH; ctx.B=B; ctx.H=H; ctx.T=T; ctx.K=K; ctx.Vd=Vd; ctx.Kb=Kb; ctx.Vb=Vb
        ctx.BLOCK_C=BLOCK_C; ctx.BLOCK_K=BLOCK_K; ctx.BLOCK_V=BLOCK_V
        ctx.scale=scale_f; ctx.nC=nC
        ctx.q_dtype=q.dtype; ctx.k_dtype=k.dtype; ctx.v_dtype=v.dtype
        ctx.g_dtype=g.dtype; ctx.beta_dtype=beta.dtype
        ctx.e_dtype=e.dtype; ctx.gamma_dtype=gamma.dtype
        return out.to(q.dtype) if q.dtype!=torch.float32 else out
    @staticmethod
    def backward(ctx, go):
        q_pad,k_pad,v_pad,d_pad,beta_r,chunk_states=ctx.saved_tensors
        BH=ctx.BH; B=ctx.B; H=ctx.H; T=ctx.T; K=ctx.K; Vd=ctx.Vd; Kb=ctx.Kb; Vb=ctx.Vb
        BLOCK_C=ctx.BLOCK_C; BLOCK_K=ctx.BLOCK_K; BLOCK_V=ctx.BLOCK_V
        scale_f=ctx.scale
        go_r=go.contiguous().view(BH,T,Vd).to(torch.float32)
        dq_pad=torch.zeros((BH,T,Kb),device=q_pad.device,dtype=torch.float32)
        dk_pad=torch.zeros((BH,T,Kb),device=q_pad.device,dtype=torch.float32)
        dv_pad=torch.zeros((BH,T,Vb),device=q_pad.device,dtype=torch.float32)
        db_pad=torch.zeros((BH,T),device=q_pad.device,dtype=torch.float32)
        dg_pad=torch.zeros((BH,T),device=q_pad.device,dtype=torch.float32)
        tmp=torch.empty((BH,BLOCK_C,Kb,Vb),device=q_pad.device,dtype=torch.float32)
        cur_s=torch.empty((BH,Kb,Vb),device=q_pad.device,dtype=torch.float32)
        g_state=torch.zeros((BH,Kb,Vb),device=q_pad.device,dtype=torch.float32)
        grid=(BH,)
        for ck in range(ctx.nC-1,-1,-1):
            START=ck*BLOCK_C
            cur_s.copy_(chunk_states[:,ck])
            _recompute_kernel[grid](k_pad,v_pad,d_pad,beta_r,cur_s,tmp,T,K,Vd,Kb,Vb,START,BLOCK_C,BLOCK_K,BLOCK_V)
            _bwd_kernel[grid](q_pad,k_pad,v_pad,d_pad,beta_r,go_r,tmp,chunk_states[:,ck].contiguous(),g_state,dq_pad,dk_pad,dv_pad,db_pad,dg_pad,scale_f,T,K,Vd,Kb,Vb,START,BLOCK_C,BLOCK_K,BLOCK_V)
        dq=dq_pad[:,:,:K].view(B,H,T,K).to(ctx.q_dtype)
        dk=dk_pad[:,:,:K].view(B,H,T,K).to(ctx.k_dtype)
        dv=dv_pad[:,:,:Vd].view(B,H,T,Vd).to(ctx.v_dtype)
        db=db_pad.view(B,H,T).to(ctx.beta_dtype)
        dg=dg_pad.view(B,H,T).to(ctx.g_dtype)
        ge=torch.zeros((B,H,T,K),device=q_pad.device,dtype=ctx.e_dtype)
        gg=torch.zeros((B,H,T),device=q_pad.device,dtype=ctx.gamma_dtype)
        return dq,dk,dv,ge,dg,db,gg,None

def kernel_function(q,k,v,e,g,beta,gamma,scale):
    return DeltaFn.apply(q,k,v,e,g,beta,gamma,scale)
