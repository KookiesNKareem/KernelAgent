```python
import torch
import triton
import triton.language as tl

def _reference(q,k,v,e,g,beta,gamma,scale):
    B,H,T,K=q.shape;V=v.shape[-1]
    q,k,v,e,g,beta,gamma=(x.to(torch.promote_types(x.dtype,torch.float32)) for x in (q,k,v,e,g,beta,gamma))
    S=q.new_zeros(B,H,K,V);outs=[]
    for t in range(T):
        kt,vt,et,bt=k[:,:,t],v[:,:,t],e[:,:,t],beta[:,:,t]
        pred=(S*kt[...,None]).sum(-2)
        S=S+(bt[...,None]*(vt-pred))[...,None,:]*kt[...,None]
        S=S*g[:,:,t].exp()[...,None]
        outs.append((S*(q[:,:,t]*scale)[...,None]).sum(-2))
    return torch.stack(outs,dim=2)

@triton.jit
def _chunk_fwd_kernel(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,out_ptr,state_ptr,BH,T,K,V,t0,BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,CHUNK:tl.constexpr,SCALE:tl.constexpr,NUM_K_BLOCKS:tl.constexpr,NUM_V_BLOCKS:tl.constexpr):
    pid=tl.program_id(0);bh=pid//NUM_V_BLOCKS;vid=pid%NUM_V_BLOCKS
    offs_k=tl.arange(0,BLOCK_K);offs_v=tl.arange(0,BLOCK_V);cur_off_v=vid*BLOCK_V+offs_v
    base_state=state_ptr+bh*K*V
    if NUM_K_BLOCKS>0:
        S0_ptr=base_state+(offs_k[:,None]+0*BLOCK_K)*V+cur_off_v[None,:];S0=tl.load(S0_ptr).to(tl.float32)
    if NUM_K_BLOCKS>1:
        S1_ptr=base_state+(offs_k[:,None]+1*BLOCK_K)*V+cur_off_v[None,:];S1=tl.load(S1_ptr).to(tl.float32)
    if NUM_K_BLOCKS>2:
        S2_ptr=base_state+(offs_k[:,None]+2*BLOCK_K)*V+cur_off_v[None,:];S2=tl.load(S2_ptr).to(tl.float32)
    if NUM_K_BLOCKS>3:
        S3_ptr=base_state+(offs_k[:,None]+3*BLOCK_K)*V+cur_off_v[None,:];S3=tl.load(S3_ptr).to(tl.float32)
    if NUM_K_BLOCKS>4:
        S4_ptr=base_state+(offs_k[:,None]+4*BLOCK_K)*V+cur_off_v[None,:];S4=tl.load(S4_ptr).to(tl.float32)
    if NUM_K_BLOCKS>5:
        S5_ptr=base_state+(offs_k[:,None]+5*BLOCK_K)*V+cur_off_v[None,:];S5=tl.load(S5_ptr).to(tl.float32)
    if NUM_K_BLOCKS>6:
        S6_ptr=base_state+(offs_k[:,None]+6*BLOCK_K)*V+cur_off_v[None,:];S6=tl.load(S6_ptr).to(tl.float32)
    if NUM_K_BLOCKS>7:
        S7_ptr=base_state+(offs_k[:,None]+7*BLOCK_K)*V+cur_off_v[None,:];S7=tl.load(S7_ptr).to(tl.float32)
    base_q=q_ptr+bh*T*K;base_k=k_ptr+bh*T*K;base_g=g_ptr+bh*T*K;base_v=v_ptr+bh*T*V;base_o=out_ptr+bh*T*V;base_beta=beta_ptr+bh*T
    for c in range(CHUNK):
        t=t0+c
        if NUM_K_BLOCKS>0:
            k0=tl.load(base_k+t*K+0*BLOCK_K+offs_k).to(tl.float32);q0=tl.load(base_q+t*K+0*BLOCK_K+offs_k).to(tl.float32);g0=tl.load(base_g+t*K+0*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>1:
            k1=tl.load(base_k+t*K+1*BLOCK_K+offs_k).to(tl.float32);q1=tl.load(base_q+t*K+1*BLOCK_K+offs_k).to(tl.float32);g1=tl.load(base_g+t*K+1*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>2:
            k2=tl.load(base_k+t*K+2*BLOCK_K+offs_k).to(tl.float32);q2=tl.load(base_q+t*K+2*BLOCK_K+offs_k).to(tl.float32);g2=tl.load(base_g+t*K+2*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>3:
            k3=tl.load(base_k+t*K+3*BLOCK_K+offs_k).to(tl.float32);q3=tl.load(base_q+t*K+3*BLOCK_K+offs_k).to(tl.float32);g3=tl.load(base_g+t*K+3*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>4:
            k4=tl.load(base_k+t*K+4*BLOCK_K+offs_k).to(tl.float32);q4=tl.load(base_q+t*K+4*BLOCK_K+offs_k).to(tl.float32);g4=tl.load(base_g+t*K+4*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>5:
            k5=tl.load(base_k+t*K+5*BLOCK_K+offs_k).to(tl.float32);q5=tl.load(base_q+t*K+5*BLOCK_K+offs_k).to(tl.float32);g5=tl.load(base_g+t*K+5*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>6:
            k6=tl.load(base_k+t*K+6*BLOCK_K+offs_k).to(tl.float32);q6=tl.load(base_q+t*K+6*BLOCK_K+offs_k).to(tl.float32);g6=tl.load(base_g+t*K+6*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>7:
            k7=tl.load(base_k+t*K+7*BLOCK_K+offs_k).to(tl.float32);q7=tl.load(base_q+t*K+7*BLOCK_K+offs_k).to(tl.float32);g7=tl.load(base_g+t*K+7*BLOCK_K+offs_k).to(tl.float32)
        beta_t=tl.load(base_beta+t).to(tl.float32);v_cur=tl.load(base_v+t*V+cur_off_v).to(tl.float32)
        pred=tl.zeros((BLOCK_V,),dtype=tl.float32)
        if NUM_K_BLOCKS>0: pred+=tl.sum(S0*k0[:,None],axis=0)
        if NUM_K_BLOCKS>1: pred+=tl.sum(S1*k1[:,None],axis=0)
        if NUM_K_BLOCKS>2: pred+=tl.sum(S2*k2[:,None],axis=0)
        if NUM_K_BLOCKS>3: pred+=tl.sum(S3*k3[:,None],axis=0)
        if NUM_K_BLOCKS>4: pred+=tl.sum(S4*k4[:,None],axis=0)
        if NUM_K_BLOCKS>5: pred+=tl.sum(S5*k5[:,None],axis=0)
        if NUM_K_BLOCKS>6: pred+=tl.sum(S6*k6[:,None],axis=0)
        if NUM_K_BLOCKS>7: pred+=tl.sum(S7*k7[:,None],axis=0)
        b_err=beta_t*(v_cur-pred)
        if NUM_K_BLOCKS>0: S0=S0+k0[:,None]*b_err[None,:];S0=S0*tl.exp(g0)[:,None]
        if NUM_K_BLOCKS>1: S1=S1+k1[:,None]*b_err[None,:];S1=S1*tl.exp(g1)[:,None]
        if NUM_K_BLOCKS>2: S2=S2+k2[:,None]*b_err[None,:];S2=S2*tl.exp(g2)[:,None]
        if NUM_K_BLOCKS>3: S3=S3+k3[:,None]*b_err[None,:];S3=S3*tl.exp(g3)[:,None]
        if NUM_K_BLOCKS>4: S4=S4+k4[:,None]*b_err[None,:];S4=S4*tl.exp(g4)[:,None]
        if NUM_K_BLOCKS>5: S5=S5+k5[:,None]*b_err[None,:];S5=S5*tl.exp(g5)[:,None]
        if NUM_K_BLOCKS>6: S6=S6+k6[:,None]*b_err[None,:];S6=S6*tl.exp(g6)[:,None]
        if NUM_K_BLOCKS>7: S7=S7+k7[:,None]*b_err[None,:];S7=S7*tl.exp(g7)[:,None]
        out=tl.zeros((BLOCK_V,),dtype=tl.float32)
        if NUM_K_BLOCKS>0: out+=tl.sum(S0*q0[:,None],axis=0)
        if NUM_K_BLOCKS>1: out+=tl.sum(S1*q1[:,None],axis=0)
        if NUM_K_BLOCKS>2: out+=tl.sum(S2*q2[:,None],axis=0)
        if NUM_K_BLOCKS>3: out+=tl.sum(S3*q3[:,None],axis=0)
        if NUM_K_BLOCKS>4: out+=tl.sum(S4*q4[:,None],axis=0)
        if NUM_K_BLOCKS>5: out+=tl.sum(S5*q5[:,None],axis=0)
        if NUM_K_BLOCKS>6: out+=tl.sum(S6*q6[:,None],axis=0)
        if NUM_K_BLOCKS>7: out+=tl.sum(S7*q7[:,None],axis=0)
        out=out*SCALE;tl.store(base_o+t*V+cur_off_v,out)
    if NUM_K_BLOCKS>0: tl.store(S0_ptr,S0)
    if NUM_K_BLOCKS>1: tl.store(S1_ptr,S1)
    if NUM_K_BLOCKS>2: tl.store(S2_ptr,S2)
    if NUM_K_BLOCKS>3: tl.store(S3_ptr,S3)
    if NUM_K_BLOCKS>4: tl.store(S4_ptr,S4)
    if NUM_K_BLOCKS>5: tl.store(S5_ptr,S5)
    if NUM_K_BLOCKS>6: tl.store(S6_ptr,S6)
    if NUM_K_BLOCKS>7: tl.store(S7_ptr,S7)

@triton.jit
def _fwd_scratch_kernel(q_ptr,k_ptr,v_ptr,decay_ptr,beta_ptr,state_in_ptr,scratch_ptr,BH,T,K,V,t0,BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,CHUNK:tl.constexpr,NUM_K_BLOCKS:tl.constexpr,NUM_V_BLOCKS:tl.constexpr):
    pid=tl.program_id(0);bh=pid//NUM_V_BLOCKS;vid=pid%NUM_V_BLOCKS
    offs_k=tl.arange(0,BLOCK_K);offs_v=tl.arange(0,BLOCK_V);cur_off_v=vid*BLOCK_V+offs_v
    base_state=state_in_ptr+bh*K*V
    if NUM_K_BLOCKS>0:
        S0=tl.load(base_state+(offs_k[:,None]+0*BLOCK_K)*V+cur_off_v[None,:]).to(tl.float32)
    if NUM_K_BLOCKS>1:
        S1=tl.load(base_state+(offs_k[:,None]+1*BLOCK_K)*V+cur_off_v[None,:]).to(tl.float32)
    if NUM_K_BLOCKS>2:
        S2=tl.load(base_state+(offs_k[:,None]+2*BLOCK_K)*V+cur_off_v[None,:]).to(tl.float32)
    if NUM_K_BLOCKS>3:
        S3=tl.load(base_state+(offs_k[:,None]+3*BLOCK_K)*V+cur_off_v[None,:]).to(tl.float32)
    if NUM_K_BLOCKS>4:
        S4=tl.load(base_state+(offs_k[:,None]+4*BLOCK_K)*V+cur_off_v[None,:]).to(tl.float32)
    if NUM_K_BLOCKS>5:
        S5=tl.load(base_state+(offs_k[:,None]+5*BLOCK_K)*V+cur_off_v[None,:]).to(tl.float32)
    if NUM_K_BLOCKS>6:
        S6=tl.load(base_state+(offs_k[:,None]+6*BLOCK_K)*V+cur_off_v[None,:]).to(tl.float32)
    if NUM_K_BLOCKS>7:
        S7=tl.load(base_state+(offs_k[:,None]+7*BLOCK_K)*V+cur_off_v[None,:]).to(tl.float32)
    base_k=k_ptr+bh*T*K;base_v=v_ptr+bh*T*V;base_d=decay_ptr+bh*T*K;base_beta=beta_ptr+bh*T
    base_scratch=scratch_ptr+bh*CHUNK*K*V
    for c in range(CHUNK):
        t=t0+c
        if NUM_K_BLOCKS>0:
            k0=tl.load(base_k+t*K+0*BLOCK_K+offs_k).to(tl.float32);d0=tl.load(base_d+t*K+0*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>1:
            k1=tl.load(base_k+t*K+1*BLOCK_K+offs_k).to(tl.float32);d1=tl.load(base_d+t*K+1*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>2:
            k2=tl.load(base_k+t*K+2*BLOCK_K+offs_k).to(tl.float32);d2=tl.load(base_d+t*K+2*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>3:
            k3=tl.load(base_k+t*K+3*BLOCK_K+offs_k).to(tl.float32);d3=tl.load(base_d+t*K+3*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>4:
            k4=tl.load(base_k+t*K+4*BLOCK_K+offs_k).to(tl.float32);d4=tl.load(base_d+t*K+4*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>5:
            k5=tl.load(base_k+t*K+5*BLOCK_K+offs_k).to(tl.float32);d5=tl.load(base_d+t*K+5*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>6:
            k6=tl.load(base_k+t*K+6*BLOCK_K+offs_k).to(tl.float32);d6=tl.load(base_d+t*K+6*BLOCK_K+offs_k).to(tl.float32)
        if NUM_K_BLOCKS>7:
            k7=tl.load(base_k+t*K+7*BLOCK_K+offs_k).to(tl.float32);d7=tl.load(base_d+t*K+7*BLOCK_K+offs_k).to(tl.float32)
        beta_t=tl.load(base_beta+t).to(tl.float32);v_cur=tl.load(base_v+t*V+cur_off_v).to(tl.float32)
        pred=tl.zeros((BLOCK_V,),dtype=tl.float32)
        if NUM_K_BLOCKS>0: pred+=tl.sum(S0*k0[:,None],axis=0)
        if NUM_K_BLOCKS>1: pred+=tl.sum(S1*k1[:,None],axis=0)
        if NUM_K_BLOCKS>2: pred+=tl.sum(S2*k2[:,None],axis=0)
        if NUM_K_BLOCKS>3: pred+=tl.sum(S3*k3[:,None],axis=0)
        if NUM_K_BLOCKS>4: pred+=tl.sum(S4*k4[:,None],axis=0)
        if NUM_K_BLOCKS>5: pred+=tl.sum(S5*k5[:,None],axis=0)
        if NUM_K_BLOCKS>6: pred+=tl.sum(S6*k6[:,None],axis=0)
        if NUM_K_BLOCKS>7: pred+=tl.sum(S7*k7[:,None],axis=0)
        b_err=beta_t*(v_cur-pred)
        if NUM_K_BLOCKS>0: S0=S0+k0[:,None]*b_err[None,:];S0=S0*d0[:,None]
        if NUM_K_BLOCKS>1: S1=S1+k1[:,None]*b_err[None,:];S1=S1*d1[:,None]
        if NUM_K_BLOCKS>2: S2=S2+k2[:,None]*b_err[None,:];S2=S2*d2[:,None]
        if NUM_K_BLOCKS>3: S3=S3+k3[:,None]*b_err[None,:];S3=S3*d3[:,None]
        if NUM_K_BLOCKS>4: S4=S4+k4[:,None]*b_err[None,:];S4=S4*d4[:,None]
        if NUM_K_BLOCKS>5: S5=S5+k5[:,None]*b_err[None,:];S5=S5*d5[:,None]
        if NUM_K_BLOCKS>6: S6=S6+k6[:,None]*b_err[None,:];S6=S6*d6[:,None]
        if NUM_K_BLOCKS>7: S7=S7+k7[:,None]*b_err[None,:];S7=S7*d7[:,None]
        base_c=base_scratch+c*K*V
        if NUM_K_BLOCKS>0: tl.store(base_c+(offs_k[:,None]+0*BLOCK_K)*V+cur_off_v[None,:],S0)
        if NUM_K_BLOCKS>1: tl.store(base_c+(offs_k[:,None]+1*BLOCK_K)*V+cur_off_v[None,:],S1)
        if NUM_K_BLOCKS>2: tl.store(base_c+(offs_k[:,None]+2*BLOCK_K)*V+cur_off_v[None,:],S2)
        if NUM_K_BLOCKS>3: tl.store(base_c+(offs_k[:,None]+3*BLOCK_K)*V+cur_off_v[None,:],S3)
        if NUM_K_BLOCKS>4: tl.store(base_c+(offs_k[:,None]+4*BLOCK_K)*V+cur_off_v[None,:],S4)
        if NUM_K_BLOCKS>5: tl.store(base_c+(offs_k[:,None]+5*BLOCK_K)*V+cur_off_v[None,:],S5)
        if NUM_K_BLOCKS>6: tl.store(base_c+(offs_k[:,None]+6*BLOCK_K)*V+cur_off_v[None,:],S6)
        if NUM_K_BLOCKS>7: tl.store(base_c+(offs_k[:,None]+7*BLOCK_K)*V+cur_off_v[None,:],S7)

# backward BH kernel for K=V=128 (4x4 tiles) – no atomics, BH parallel
@triton.jit
def _bwd_bh_kernel(q_ptr,k_ptr,decay_ptr,beta_ptr,v_ptr,do_ptr,scratch_ptr,state_in_ptr,G_ptr,dq_ptr,dk_ptr,dg_ptr,dv_ptr,dbeta_ptr,BH,T,K,V,t0,BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,CHUNK:tl.constexpr,SCALE:tl.constexpr):
    bh=tl.program_id(0)
    offs_k=tl.arange(0,BLOCK_K)
    offs_v=tl.arange(0,BLOCK_V)
    base_q=q_ptr+bh*T*K; base_k=k_ptr+bh*T*K; base_d=decay_ptr+bh*T*K; base_beta=beta_ptr+bh*T; base_v=v_ptr+bh*T*V; base_do=do_ptr+bh*T*V
    base_scratch=scratch_ptr+bh*CHUNK*K*V; base_state=state_in_ptr+bh*K*V; base_G=G_ptr+bh*K*V
    base_dq=dq_ptr+bh*T*K; base_dk=dk_ptr+bh*T*K; base_dg=dg_ptr+bh*T*K; base_dv=dv_ptr+bh*T*V; base_db=dbeta_ptr+bh*T
    # load G 4x4
    G00=tl.load(base_G+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
    G01=tl.load(base_G+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
    G02=tl.load(base_G+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
    G03=tl.load(base_G+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
    G10=tl.load(base_G+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
    G11=tl.load(base_G+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
    G12=tl.load(base_G+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
    G13=tl.load(base_G+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
    G20=tl.load(base_G+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
    G21=tl.load(base_G+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
    G22=tl.load(base_G+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
    G23=tl.load(base_G+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
    G30=tl.load(base_G+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
    G31=tl.load(base_G+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
    G32=tl.load(base_G+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
    G33=tl.load(base_G+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
    for c in range(CHUNK):
        rev=CHUNK-1-c
        t=t0+rev
        q0=tl.load(base_q+t*K+0*BLOCK_K+offs_k).to(tl.float32)
        q1=tl.load(base_q+t*K+1*BLOCK_K+offs_k).to(tl.float32)
        q2=tl.load(base_q+t*K+2*BLOCK_K+offs_k).to(tl.float32)
        q3=tl.load(base_q+t*K+3*BLOCK_K+offs_k).to(tl.float32)
        k0=tl.load(base_k+t*K+0*BLOCK_K+offs_k).to(tl.float32)
        k1=tl.load(base_k+t*K+1*BLOCK_K+offs_k).to(tl.float32)
        k2=tl.load(base_k+t*K+2*BLOCK_K+offs_k).to(tl.float32)
        k3=tl.load(base_k+t*K+3*BLOCK_K+offs_k).to(tl.float32)
        d0=tl.load(base_d+t*K+0*BLOCK_K+offs_k).to(tl.float32)
        d1=tl.load(base_d+t*K+1*BLOCK_K+offs_k).to(tl.float32)
        d2=tl.load(base_d+t*K+2*BLOCK_K+offs_k).to(tl.float32)
        d3=tl.load(base_d+t*K+3*BLOCK_K+offs_k).to(tl.float32)
        beta_t=tl.load(base_beta+t).to(tl.float32)
        v0=tl.load(base_v+t*V+0*BLOCK_V+offs_v).to(tl.float32)
        v1=tl.load(base_v+t*V+1*BLOCK_V+offs_v).to(tl.float32)
        v2=tl.load(base_v+t*V+2*BLOCK_V+offs_v).to(tl.float32)
        v3=tl.load(base_v+t*V+3*BLOCK_V+offs_v).to(tl.float32)
        do0=tl.load(base_do+t*V+0*BLOCK_V+offs_v).to(tl.float32)
        do1=tl.load(base_do+t*V+1*BLOCK_V+offs_v).to(tl.float32)
        do2=tl.load(base_do+t*V+2*BLOCK_V+offs_v).to(tl.float32)
        do3=tl.load(base_do+t*V+3*BLOCK_V+offs_v).to(tl.float32)
        base_c=base_scratch+rev*K*V
        S00=tl.load(base_c+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
        S01=tl.load(base_c+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
        S02=tl.load(base_c+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
        S03=tl.load(base_c+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
        S10=tl.load(base_c+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
        S11=tl.load(base_c+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
        S12=tl.load(base_c+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
        S13=tl.load(base_c+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
        S20=tl.load(base_c+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
        S21=tl.load(base_c+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
        S22=tl.load(base_c+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
        S23=tl.load(base_c+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
        S30=tl.load(base_c+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
        S31=tl.load(base_c+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
        S32=tl.load(base_c+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
        S33=tl.load(base_c+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
        # S_prev
        if rev==0:
            SP00=tl.load(base_state+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
            SP01=tl.load(base_state+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
            SP02=tl.load(base_state+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
            SP03=tl.load(base_state+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
            SP10=tl.load(base_state+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
            SP11=tl.load(base_state+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
            SP12=tl.load(base_state+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
            SP13=tl.load(base_state+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
            SP20=tl.load(base_state+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
            SP21=tl.load(base_state+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
            SP22=tl.load(base_state+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
            SP23=tl.load(base_state+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
            SP30=tl.load(base_state+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
            SP31=tl.load(base_state+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
            SP32=tl.load(base_state+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
            SP33=tl.load(base_state+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
        else:
            base_p=base_scratch+(rev-1)*K*V
            SP00=tl.load(base_p+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
            SP01=tl.load(base_p+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
            SP02=tl.load(base_p+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
            SP03=tl.load(base_p+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
            SP10=tl.load(base_p+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
            SP11=tl.load(base_p+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
            SP12=tl.load(base_p+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
            SP13=tl.load(base_p+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
            SP20=tl.load(base_p+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
            SP21=tl.load(base_p+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
            SP22=tl.load(base_p+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
            SP23=tl.load(base_p+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
            SP30=tl.load(base_p+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V).to(tl.float32)
            SP31=tl.load(base_p+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V).to(tl.float32)
            SP32=tl.load(base_p+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V).to(tl.float32)
            SP33=tl.load(base_p+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V).to(tl.float32)
        # p = S_prev^T k
        p0=tl.sum(SP00*k0[:,None],axis=0)+tl.sum(SP10*k1[:,None],axis=0)+tl.sum(SP20*k2[:,None],axis=0)+tl.sum(SP30*k3[:,None],axis=0)
        p1=tl.sum(SP01*k0[:,None],axis=0)+tl.sum(SP11*k1[:,None],axis=0)+tl.sum(SP21*k2[:,None],axis=0)+tl.sum(SP31*k3[:,None],axis=0)
        p2=tl.sum(SP02*k0[:,None],axis=0)+tl.sum(SP12*k1[:,None],axis=0)+tl.sum(SP22*k2[:,None],axis=0)+tl.sum(SP32*k3[:,None],axis=0)
        p3=tl.sum(SP03*k0[:,None],axis=0)+tl.sum(SP13*k1[:,None],axis=0)+tl.sum(SP23*k2[:,None],axis=0)+tl.sum(SP33*k3[:,None],axis=0)
        s_scalar=tl.sum(k0*d0*q0)+tl.sum(k1*d1*q1)+tl.sum(k2*d2*q2)+tl.sum(k3*d3*q3)
        h0_0=tl.sum(k0[:,None]*d0[:,None]*G00,axis=0)+tl.sum(k1[:,None]*d1[:,None]*G10,axis=0)+tl.sum(k2[:,None]*d2[:,None]*G20,axis=0)+tl.sum(k3[:,None]*d3[:,None]*G30,axis=0)
        h0_1=tl.sum(k0[:,None]*d0[:,None]*G01,axis=0)+tl.sum(k1[:,None]*d1[:,None]*G11,axis=0)+tl.sum(k2[:,None]*d2[:,None]*G21,axis=0)+tl.sum(k3[:,None]*d3[:,None]*G31,axis=0)
        h0_2=tl.sum(k0[:,None]*d0[:,None]*G02,axis=0)+tl.sum(k1[:,None]*d1[:,None]*G12,axis=0)+tl.sum(k2[:,None]*d2[:,None]*G22,axis=0)+tl.sum(k3[:,None]*d3[:,None]*G32,axis=0)
        h0_3=tl.sum(k0[:,None]*d0[:,None]*G03,axis=0)+tl.sum(k1[:,None]*d1[:,None]*G13,axis=0)+tl.sum(k2[:,None]*d2[:,None]*G23,axis=0)+tl.sum(k3[:,None]*d3[:,None]*G33,axis=0)
        h_0=h0_0+SCALE*s_scalar*do0
        h_1=h0_1+SCALE*s_scalar*do1
        h_2=h0_2+SCALE*s_scalar*do2
        h_3=h0_3+SCALE*s_scalar*do3
        e0=v0-p0; e1=v1-p1; e2=v2-p2; e3=v3-p3
        dq0=(tl.sum(S00*do0[None,:],axis=1)+tl.sum(S01*do1[None,:],axis=1)+tl.sum(S02*do2[None,:],axis=1)+tl.sum(S03*do3[None,:],axis=1))*SCALE
        dq1=(tl.sum(S10*do0[None,:],axis=1)+tl.sum(S11*do1[None,:],axis=1)+tl.sum(S12*do2[None,:],axis=1)+tl.sum(S13*do3[None,:],axis=1))*SCALE
        dq2=(tl.sum(S20*do0[None,:],axis=1)+tl.sum(S21*do1[None,:],axis=1)+tl.sum(S22*do2[None,:],axis=1)+tl.sum(S23*do3[None,:],axis=1))*SCALE
        dq3=(tl.sum(S30*do0[None,:],axis=1)+tl.sum(S31*do1[None,:],axis=1)+tl.sum(S32*do2[None,:],axis=1)+tl.sum(S33*do3[None,:],axis=1))*SCALE
        tl.store(base_dq+t*K+0*BLOCK_K+offs_k,dq0)
        tl.store(base_dq+t*K+1*BLOCK_K+offs_k,dq1)
        tl.store(base_dq+t*K+2*BLOCK_K+offs_k,dq2)
        tl.store(base_dq+t*K+3*BLOCK_K+offs_k,dq3)
        dg0=tl.sum(G00*S00,axis=1)+tl.sum(G01*S01,axis=1)+tl.sum(G02*S02,axis=1)+tl.sum(G03*S03,axis=1)+q0*dq0
        dg1=tl.sum(G10*S10,axis=1)+tl.sum(G11*S11,axis=1)+tl.sum(G12*S12,axis=1)+tl.sum(G13*S13,axis=1)+q1*dq1
        dg2=tl.sum(G20*S20,axis=1)+tl.sum(G21*S21,axis=1)+tl.sum(G22*S22,axis=1)+tl.sum(G23*S23,axis=1)+q2*dq2
        dg3=tl.sum(G30*S30,axis=1)+tl.sum(G31*S31,axis=1)+tl.sum(G32*S32,axis=1)+tl.sum(G33*S33,axis=1)+q3*dq3
        tl.store(base_dg+t*K+0*BLOCK_K+offs_k,dg0)
        tl.store(base_dg+t*K+1*BLOCK_K+offs_k,dg1)
        tl.store(base_dg+t*K+2*BLOCK_K+offs_k,dg2)
        tl.store(base_dg+t*K+3*BLOCK_K+offs_k,dg3)
        tl.store(base_dv+t*V+0*BLOCK_V+offs_v,beta_t*h_0)
        tl.store(base_dv+t*V+1*BLOCK_V+offs_v,beta_t*h_1)
        tl.store(base_dv+t*V+2*BLOCK_V+offs_v,beta_t*h_2)
        tl.store(base_dv+t*V+3*BLOCK_V+offs_v,beta_t*h_3)
        db=tl.sum(e0*h_0)+tl.sum(e1*h_1)+tl.sum(e2*h_2)+tl.sum(e3*h_3)
        tl.store(base_db+t,db)
        dot_de=tl.sum(do0*e0)+tl.sum(do1*e1)+tl.sum(do2*e2)+tl.sum(do3*e3)
        Ge0=tl.sum(G00*e0[None,:],axis=1)+tl.sum(G01*e1[None,:],axis=1)+tl.sum(G02*e2[None,:],axis=1)+tl.sum(G03*e3[None,:],axis=1)
        Ge1=tl.sum(G10*e0[None,:],axis=1)+tl.sum(G11*e1[None,:],axis=1)+tl.sum(G12*e2[None,:],axis=1)+tl.sum(G13*e3[None,:],axis=1)
        Ge2=tl.sum(G20*e0[None,:],axis=1)+tl.sum(G21*e1[None,:],axis=1)+tl.sum(G22*e2[None,:],axis=1)+tl.sum(G23*e3[None,:],axis=1)
        Ge3=tl.sum(G30*e0[None,:],axis=1)+tl.sum(G31*e1[None,:],axis=1)+tl.sum(G32*e2[None,:],axis=1)+tl.sum(G33*e3[None,:],axis=1)
        SPdo0=tl.sum(SP00*do0[None,:],axis=1)+tl.sum(SP01*do1[None,:],axis=1)+tl.sum(SP02*do2[None,:],axis=1)+tl.sum(SP03*do3[None,:],axis=1)
        SPdo1=tl.sum(SP10*do0[None,:],axis=1)+tl.sum(SP11*do1[None,:],axis=1)+tl.sum(SP12*do2[None,:],axis=1)+tl.sum(SP13*do3[None,:],axis=1)
        SPdo2=tl.sum(SP20*do0[None,:],axis=1)+tl.sum(SP21*do1[None,:],axis=1)+tl.sum(SP22*do2[None,:],axis=1)+tl.sum(SP23*do3[None,:],axis=1)
        SPdo3=tl.sum(SP30*do0[None,:],axis=1)+tl.sum(SP31*do1[None,:],axis=1)+tl.sum(SP32*do2[None,:],axis=1)+tl.sum(SP33*do3[None,:],axis=1)
        SPh0_0=tl.sum(SP00*h0_0[None,:],axis=1)+tl.sum(SP01*h0_1[None,:],axis=1)+tl.sum(SP02*h0_2[None,:],axis=1)+tl.sum(SP03*h0_3[None,:],axis=1)
        SPh0_1=tl.sum(SP10*h0_0[None,:],axis=1)+tl.sum(SP11*h0_1[None,:],axis=1)+tl.sum(SP12*h0_2[None,:],axis=1)+tl.sum(SP13*h0_3[None,:],axis=1)
        SPh0_2=tl.sum(SP20*h0_0[None,:],axis=1)+tl.sum(SP21*h0_1[None,:],axis=1)+tl.sum(SP22*h0_2[None,:],axis=1)+tl.sum(SP23*h0_3[None,:],axis=1)
        SPh0_3=tl.sum(SP30*h0_0[None,:],axis=1)+tl.sum(SP31*h0_1[None,:],axis=1)+tl.sum(SP32*h0_2[None,:],axis=1)+tl.sum(SP33*h0_3[None,:],axis=1)
        Qe0=d0*(Ge0+SCALE*dot_de*q0)
        Qe1=d1*(Ge1+SCALE*dot_de*q1)
        Qe2=d2*(Ge2+SCALE*dot_de*q2)
        Qe3=d3*(Ge3+SCALE*dot_de*q3)
        SPh_0=SPh0_0+SCALE*s_scalar*SPdo0
        SPh_1=SPh0_1+SCALE*s_scalar*SPdo1
        SPh_2=SPh0_2+SCALE*s_scalar*SPdo2
        SPh_3=SPh0_3+SCALE*s_scalar*SPdo3
        dk0=beta_t*(Qe0 - SPh_0)
        dk1=beta_t*(Qe1 - SPh_1)
        dk2=beta_t*(Qe2 - SPh_2)
        dk3=beta_t*(Qe3 - SPh_3)
        tl.store(base_dk+t*K+0*BLOCK_K+offs_k,dk0)
        tl.store(base_dk+t*K+1*BLOCK_K+offs_k,dk1)
        tl.store(base_dk+t*K+2*BLOCK_K+offs_k,dk2)
        tl.store(base_dk+t*K+3*BLOCK_K+offs_k,dk3)
        # G update
        G00=d0[:,None]*G00+SCALE*d0[:,None]*q0[:,None]*do0[None,:]-beta_t*k0[:,None]*h_0[None,:]
        G01=d0[:,None]*G01+SCALE*d0[:,None]*q0[:,None]*do1[None,:]-beta_t*k0[:,None]*h_1[None,:]
        G02=d0[:,None]*G02+SCALE*d0[:,None]*q0[:,None]*do2[None,:]-beta_t*k0[:,None]*h_2[None,:]
        G03=d0[:,None]*G03+SCALE*d0[:,None]*q0[:,None]*do3[None,:]-beta_t*k0[:,None]*h_3[None,:]
        G10=d1[:,None]*G10+SCALE*d1[:,None]*q1[:,None]*do0[None,:]-beta_t*k1[:,None]*h_0[None,:]
        G11=d1[:,None]*G11+SCALE*d1[:,None]*q1[:,None]*do1[None,:]-beta_t*k1[:,None]*h_1[None,:]
        G12=d1[:,None]*G12+SCALE*d1[:,None]*q1[:,None]*do2[None,:]-beta_t*k1[:,None]*h_2[None,:]
        G13=d1[:,None]*G13+SCALE*d1[:,None]*q1[:,None]*do3[None,:]-beta_t*k1[:,None]*h_3[None,:]
        G20=d2[:,None]*G20+SCALE*d2[:,None]*q2[:,None]*do0[None,:]-beta_t*k2[:,None]*h_0[None,:]
        G21=d2[:,None]*G21+SCALE*d2[:,None]*q2[:,None]*do1[None,:]-beta_t*k2[:,None]*h_1[None,:]
        G22=d2[:,None]*G22+SCALE*d2[:,None]*q2[:,None]*do2[None,:]-beta_t*k2[:,None]*h_2[None,:]
        G23=d2[:,None]*G23+SCALE*d2[:,None]*q2[:,None]*do3[None,:]-beta_t*k2[:,None]*h_3[None,:]
        G30=d3[:,None]*G30+SCALE*d3[:,None]*q3[:,None]*do0[None,:]-beta_t*k3[:,None]*h_0[None,:]
        G31=d3[:,None]*G31+SCALE*d3[:,None]*q3[:,None]*do1[None,:]-beta_t*k3[:,None]*h_1[None,:]
        G32=d3[:,None]*G32+SCALE*d3[:,None]*q3[:,None]*do2[None,:]-beta_t*k3[:,None]*h_2[None,:]
        G33=d3[:,None]*G33+SCALE*d3[:,None]*q3[:,None]*do3[None,:]-beta_t*k3[:,None]*h_3[None,:]
    tl.store(base_G+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V,G00)
    tl.store(base_G+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V,G01)
    tl.store(base_G+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V,G02)
    tl.store(base_G+(offs_k[:,None]+0*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V,G03)
    tl.store(base_G+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V,G10)
    tl.store(base_G+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V,G11)
    tl.store(base_G+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V,G12)
    tl.store(base_G+(offs_k[:,None]+1*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V,G13)
    tl.store(base_G+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V,G20)
    tl.store(base_G+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V,G21)
    tl.store(base_G+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V,G22)
    tl.store(base_G+(offs_k[:,None]+2*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V,G23)
    tl.store(base_G+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+0*BLOCK_V,G30)
    tl.store(base_G+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+1*BLOCK_V,G31)
    tl.store(base_G+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+2*BLOCK_V,G32)
    tl.store(base_G+(offs_k[:,None]+3*BLOCK_K)*V+offs_v[None,:]+3*BLOCK_V,G33)

class _RecurrentFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,e,g,beta,gamma,scale):
        s=float(scale.item()) if isinstance(scale,torch.Tensor) else float(scale)
        if not q.is_cuda:
            out=_reference(q,k,v,e,g,beta,gamma,s);ctx.save_for_backward(q,k,v,g,beta);ctx.scale=s;ctx.use_ref=True;ctx.B=q.shape[0];ctx.H=q.shape[1];ctx.T=q.shape[2];ctx.K=q.shape[3];ctx.V=v.shape[-1];return out
        q_c=q.contiguous();k_c=k.contiguous();v_c=v.contiguous();g_c=g.contiguous();beta_c=beta.contiguous()
        B,H,T,K=q_c.shape;V=v_c.shape[-1];BH=B*H
        BLOCK_K=32;BLOCK_V=32;CHUNK=64
        K_pad=((K+BLOCK_K-1)//BLOCK_K)*BLOCK_K;V_pad=((V+BLOCK_V-1)//BLOCK_V)*BLOCK_V;T_pad=((T+CHUNK-1)//CHUNK)*CHUNK
        NUM_K_BLOCKS=K_pad//BLOCK_K;NUM_V_BLOCKS=V_pad//BLOCK_V
        if NUM_K_BLOCKS>8:
            out=_reference(q,k,v,e,g,beta,gamma,s);ctx.save_for_backward(q,k,v,g,beta);ctx.scale=s;ctx.use_ref=True;ctx.B=B;ctx.H=H;ctx.T=T;ctx.K=K;ctx.V=V;return out
        dev=q.device;need_pad=(K_pad!=K) or (V_pad!=V) or (T_pad!=T)
        if need_pad:
            q_f=torch.zeros((BH,T_pad,K_pad),device=dev,dtype=q_c.dtype);k_f=torch.zeros((BH,T_pad,K_pad),device=dev,dtype=k_c.dtype)
            g_f=torch.zeros((BH,T_pad,K_pad),device=dev,dtype=g_c.dtype);v_f=torch.zeros((BH,T_pad,V_pad),device=dev,dtype=v_c.dtype)
            beta_f=torch.zeros((BH,T_pad),device=dev,dtype=beta_c.dtype);decay_f=torch.ones((BH,T_pad,K_pad),device=dev,dtype=torch.float32)
            q_tmp=q_c.view(BH,T,K);k_tmp=k_c.view(BH,T,K);g_tmp=g_c.view(BH,T,K);v_tmp=v_c.view(BH,T,V);beta_tmp=beta_c.view(BH,T)
            q_f[:,:T,:K]=q_tmp;k_f[:,:T,:K]=k_tmp;g_f[:,:T,:K]=g_tmp;v_f[:,:T,:V]=v_tmp;beta_f[:,:T]=beta_tmp
            decay_f[:,:T,:K]=torch.exp(g_tmp.to(torch.float32))
            out_f=torch.empty((BH,T_pad,V_pad),device=dev,dtype=torch.float32);state=torch.zeros((BH,K_pad,V_pad),device=dev,dtype=torch.float32)
        else:
            q_f=q_c.view(BH,T,K);k_f=k_c.view(BH,T,K);g_f=g_c.view(BH,T,K);v_f=v_c.view(BH,T,V);beta_f=beta_c.view(BH,T)
            decay_f=torch.exp(g_f.to(torch.float32));out_f=torch.empty((BH,T_pad,V_pad),device=dev,dtype=torch.float32);state=torch.zeros((BH,K_pad,V_pad),device=dev,dtype=torch.float32)
        grid=(BH*NUM_V_BLOCKS,)
        n_chunk=T_pad//CHUNK
        cb_list=[state.clone()]
        for ci in range(n_chunk):
            t0=ci*CHUNK
            _chunk_fwd_kernel[grid](q_f,k_f,v_f,g_f,beta_f,out_f,state,BH,T_pad,K_pad,V_pad,t0,BLOCK_K,BLOCK_V,CHUNK,s,NUM_K_BLOCKS,NUM_V_BLOCKS,num_warps=4,num_stages=1)
            cb_list.append(state.clone())
        cb_stack=torch.stack(cb_list,dim=0)
        if need_pad:
            out=out_f[:,:T,:V].contiguous().view(B,H,T,V)
        else:
            out=out_f.view(B,H,T,V)
        ctx.save_for_backward(q_f,k_f,v_f,decay_f,beta_f,cb_stack)
        ctx.B=B;ctx.H=H;ctx.T=T;ctx.K=K;ctx.V=V;ctx.BH=BH;ctx.T_pad=T_pad;ctx.K_pad=K_pad;ctx.V_pad=V_pad;ctx.CHUNK=CHUNK;ctx.NUM_K_BLOCKS=NUM_K_BLOCKS;ctx.NUM_V_BLOCKS=NUM_V_BLOCKS;ctx.scale=s;ctx.need_pad=need_pad;ctx.dev=dev;ctx.dtype_q=q.dtype;ctx.dtype_k=k.dtype;ctx.dtype_v=v.dtype;ctx.dtype_g=g.dtype;ctx.dtype_beta=beta.dtype;ctx.use_ref=False
        return out
    @staticmethod
    def backward(ctx,grad_out):
        if getattr(ctx,'use_ref',False):
            q,k,v,g,beta=ctx.saved_tensors;B=ctx.B;H=ctx.H;T=ctx.T;K=ctx.K;V=ctx.V;s=ctx.scale
            q_l=q.view(B,H,T,K).detach().requires_grad_(True).to(torch.float32)
            k_l=k.view(B,H,T,K).detach().requires_grad_(True).to(torch.float32)
            v_l=v.view(B,H,T,V).detach().requires_grad_(True).to(torch.float32)
            g_l=g.view(B,H,T,K).detach().requires_grad_(True).to(torch.float32)
            beta_l=beta.view(B,H,T).detach().requires_grad_(True).to(torch.float32)
            e_d=torch.zeros_like(q_l);gamma_d=torch.zeros_like(beta_l)
            out=_reference(q_l,k_l,v_l,e_d,g_l,beta_l,gamma_d,s)
            out.backward(grad_out.to(torch.float32))
            dq=(q_l.grad if q_l.grad is not None else torch.zeros_like(q_l)).to(ctx.dtype_q if hasattr(ctx,'dtype_q') else q.dtype)
            dk=(k_l.grad if k_l.grad is not None else torch.zeros_like(k_l)).to(ctx.dtype_k if hasattr(ctx,'dtype_k') else k.dtype)
            dv=(v_l.grad if v_l.grad is not None else torch.zeros_like(v_l)).to(ctx.dtype_v if hasattr(ctx,'dtype_v') else v.dtype)
            dg=(g_l.grad if g_l.grad is not None else torch.zeros_like(g_l)).to(ctx.dtype_g if hasattr(ctx,'dtype_g') else g.dtype)
            dbeta=(beta_l.grad if beta_l.grad is not None else torch.zeros_like(beta_l)).to(ctx.dtype_beta if hasattr(ctx,'dtype_beta') else beta.dtype)
            de=torch.zeros((B,H,T,K),device=grad_out.device,dtype=dq.dtype);dgamma=torch.zeros((B,H,T),device=grad_out.device,dtype=dbeta.dtype)
            return dq,dk,dv,de,dg,dbeta,dgamma,None
        q_f,k_f,v_f,decay_f,beta_f,cb_stack=ctx.saved_tensors
        B=ctx.B;H=ctx.H;T=ctx.T;K=ctx.K;V=ctx.V;BH=ctx.BH;T_pad=ctx.T_pad;K_pad=ctx.K_pad;V_pad=ctx.V_pad;CHUNK=ctx.CHUNK;NUM_K_BLOCKS=ctx.NUM_K_BLOCKS;NUM_V_BLOCKS=ctx.NUM_V_BLOCKS;s=ctx.scale;dev=ctx.dev
        # fast path only for 128x128
        if K_pad!=128 or V_pad!=128 or NUM_K_BLOCKS!=4 or NUM_V_BLOCKS!=4:
            # fallback to reference grad
            q_l=q_f[:,:T,:K].contiguous().view(B,H,T,K).detach().requires_grad_(True).to(torch.float32)
            k_l=k_f[:,:T,:K].contiguous().view(B,H,T,K).detach().requires_grad_(True).to(torch.float32)
            v_l=v_f[:,:T,:V].contiguous().view(B,H,T,V).detach().requires_grad_(True).to(torch.float32)
            g_l=torch.log(decay_f[:,:T,:K].contiguous().view(B,H,T,K).clamp(min=1e-30)).detach().requires_grad_(True).to(torch.float32)
            beta_l=beta_f[:,:T].contiguous().view(B,H,T).detach().requires_grad_(True).to(torch.float32)
            e_d=torch.zeros_like(q_l);gamma_d=torch.zeros_like(beta_l)
            out=_reference(q_l,k_l,v_l,e_d,g_l,beta_l,gamma_d,s)
            out.backward(grad_out.to(torch.float32))
            dq=(q_l.grad if q_l.grad is not None else torch.zeros_like(q_l)).to(ctx.dtype_q)
            dk=(k_l.grad if k_l.grad is not None else torch.zeros_like(k_l)).to(ctx.dtype_k)
            dv=(v_l.grad if v_l.grad is not None else torch.zeros_like(v_l)).to(ctx.dtype_v)
            dg=(g_l.grad if g_l.grad is not None else torch.zeros_like(g_l)).to(ctx.dtype_g)
            dbeta=(beta_l.grad if beta_l.grad is not None else torch.zeros_like(beta_l)).to(ctx.dtype_beta)
            de=torch.zeros((B,H,T,K),device=dev,dtype=ctx.dtype_q);dgamma=torch.zeros((B,H,T),device=dev,dtype=ctx.dtype_beta)
            return dq,dk,dv,de,dg,dbeta,dgamma,None
        BLOCK_K=32;BLOCK_V=32
        go=grad_out.contiguous().view(BH,T,V).to(torch.float32)
        go_pad=torch.zeros((BH,T_pad,V_pad),device=dev,dtype=torch.float32);go_pad[:,:T,:V]=go
        dq_pad=torch.zeros((BH,T_pad,K_pad),device=dev,dtype=torch.float32)
        dk_pad=torch.zeros((BH,T_pad,K_pad),device=dev,dtype=torch.float32)
        dg_pad=torch.zeros((BH,T_pad,K_pad),device=dev,dtype=torch.float32)
        dv_pad=torch.zeros((BH,T_pad,V_pad),device=dev,dtype=torch.float32)
        dbeta_pad=torch.zeros((BH,T_pad),device=dev,dtype=torch.float32)
        G_state=torch.zeros((BH,K_pad,V_pad),device=dev,dtype=torch.float32)
        scratch=torch.empty((BH,CHUNK,K_pad,V_pad),device=dev,dtype=torch.float32)
        # need per chunk start states
        n_chunk=T_pad//CHUNK
        grid_f=(BH*NUM_V_BLOCKS,)
        grid_b=(BH,)
        for ci in range(n_chunk-1,-1,-1):
            t0=ci*CHUNK
            start_state=cb_stack[ci]
            _fwd_scratch_kernel[grid_f](q_f,k_f,v_f,decay_f,beta_f,start_state,scratch,BH,T_pad,K_pad,V_pad,t0,BLOCK_K,BLOCK_V,CHUNK,NUM_K_BLOCKS,NUM_V_BLOCKS,num_warps=4,num_stages=1)
            _bwd_bh_kernel[grid_b](q_f,k_f,decay_f,beta_f,v_f,go_pad,scratch,start_state,G_state,dq_pad,dk_pad,dg_pad,dv_pad,dbeta_pad,BH,T_pad,K_pad,V_pad,t0,BLOCK_K,BLOCK_V,CHUNK,s,num_warps=4,num_stages=1)
        dq=dq_pad[:,:T,:K].contiguous().view(B,H,T,K).to(ctx.dtype_q)
        dk=dk_pad[:,:T,:K].contiguous().view(B,H,T,K).to(ctx.dtype_k)
        dg=dg_pad[:,:T,:K].contiguous().view(B,H,T,K).to(ctx.dtype_g)
        dv=dv_pad[:,:T,:V].contiguous().view(B,H,T,V).to(ctx.dtype_v)
        dbeta=dbeta_pad[:,:T].contiguous().view(B,H,T).to(ctx.dtype_beta)
        de=torch.zeros((B,H,T,K),device=dev,dtype=ctx.dtype_q)
        dgamma=torch.zeros((B,H,T),device=dev,dtype=ctx.dtype_beta)
        return dq,dk,dv,de,dg,dbeta,dgamma,None

def kernel_function(q,k,v,e,g,beta,gamma,scale):
    return _RecurrentFunc.apply(q,k,v,e,g,beta,gamma,scale)
```