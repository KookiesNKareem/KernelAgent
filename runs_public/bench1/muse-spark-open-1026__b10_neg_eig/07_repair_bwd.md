```python
import torch
import triton
import triton.language as tl

# ---------- 128 fast -------------------------------------------------
@triton.jit
def fwd128_store(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,out_ptr,s_ptr,
    BH,T,K,V, scale, BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,NUM_V:tl.constexpr):
    pid=tl.program_id(0)
    bh=pid//NUM_V
    bv=pid%NUM_V
    off_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
    off_k0=tl.arange(0,BLOCK_K)
    off_k1=tl.arange(0,BLOCK_K)+BLOCK_K
    off_k2=tl.arange(0,BLOCK_K)+2*BLOCK_K
    off_k3=tl.arange(0,BLOCK_K)+3*BLOCK_K
    # S0 slice zero
    s0_base=s_ptr+bh*(T+1)*K*V
    z=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32)
    tl.store(tl.make_block_ptr(base=s0_base,shape=(K,V),strides=(V,1),offsets=(0,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),z,boundary_check=(0,1))
    tl.store(tl.make_block_ptr(base=s0_base,shape=(K,V),strides=(V,1),offsets=(BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),z,boundary_check=(0,1))
    tl.store(tl.make_block_ptr(base=s0_base,shape=(K,V),strides=(V,1),offsets=(2*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),z,boundary_check=(0,1))
    tl.store(tl.make_block_ptr(base=s0_base,shape=(K,V),strides=(V,1),offsets=(3*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),z,boundary_check=(0,1))
    S0=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32)
    S1=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32)
    S2=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32)
    S3=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32)
    q_base=q_ptr+bh*T*K
    k_base=k_ptr+bh*T*K
    v_base=v_ptr+bh*T*V
    g_base=g_ptr+bh*T
    b_base=beta_ptr+bh*T
    o_base=out_ptr+bh*T*V
    for t in range(T):
        d_t=tl.exp(tl.load(g_base+t).to(tl.float32))
        b2_t=tl.load(b_base+t).to(tl.float32)*2.0
        k0=tl.load(k_base+t*K+off_k0).to(tl.float32)
        k1=tl.load(k_base+t*K+off_k1).to(tl.float32)
        k2=tl.load(k_base+t*K+off_k2).to(tl.float32)
        k3=tl.load(k_base+t*K+off_k3).to(tl.float32)
        q0=tl.load(q_base+t*K+off_k0).to(tl.float32)
        q1=tl.load(q_base+t*K+off_k1).to(tl.float32)
        q2=tl.load(q_base+t*K+off_k2).to(tl.float32)
        q3=tl.load(q_base+t*K+off_k3).to(tl.float32)
        v_s=tl.load(v_base+t*V+off_v).to(tl.float32)
        S0=S0*d_t; S1=S1*d_t; S2=S2*d_t; S3=S3*d_t
        pred=tl.zeros((BLOCK_V,),dtype=tl.float32)
        pred+=tl.sum(S0*tl.expand_dims(k0,1),axis=0)
        pred+=tl.sum(S1*tl.expand_dims(k1,1),axis=0)
        pred+=tl.sum(S2*tl.expand_dims(k2,1),axis=0)
        pred+=tl.sum(S3*tl.expand_dims(k3,1),axis=0)
        delta=(v_s-pred)*b2_t
        drow=tl.expand_dims(delta,0)
        S0=S0+tl.expand_dims(k0,1)*drow
        S1=S1+tl.expand_dims(k1,1)*drow
        S2=S2+tl.expand_dims(k2,1)*drow
        S3=S3+tl.expand_dims(k3,1)*drow
        base=s_ptr+bh*(T+1)*K*V+(t+1)*K*V
        tl.store(tl.make_block_ptr(base=base,shape=(K,V),strides=(V,1),offsets=(0,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),S0,boundary_check=(0,1))
        tl.store(tl.make_block_ptr(base=base,shape=(K,V),strides=(V,1),offsets=(BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),S1,boundary_check=(0,1))
        tl.store(tl.make_block_ptr(base=base,shape=(K,V),strides=(V,1),offsets=(2*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),S2,boundary_check=(0,1))
        tl.store(tl.make_block_ptr(base=base,shape=(K,V),strides=(V,1),offsets=(3*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),S3,boundary_check=(0,1))
        acc=tl.zeros((BLOCK_V,),dtype=tl.float32)
        acc+=tl.sum(S0*tl.expand_dims(q0,1),axis=0)
        acc+=tl.sum(S1*tl.expand_dims(q1,1),axis=0)
        acc+=tl.sum(S2*tl.expand_dims(q2,1),axis=0)
        acc+=tl.sum(S3*tl.expand_dims(q3,1),axis=0)
        tl.store(o_base+t*V+off_v, acc*scale)

@triton.jit
def infer128(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,out_ptr,BH,T,K,V,scale,BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,NUM_V:tl.constexpr):
    pid=tl.program_id(0)
    bh=pid//NUM_V; bv=pid%NUM_V
    off_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
    off_k0=tl.arange(0,BLOCK_K); off_k1=tl.arange(0,BLOCK_K)+BLOCK_K; off_k2=tl.arange(0,BLOCK_K)+2*BLOCK_K; off_k3=tl.arange(0,BLOCK_K)+3*BLOCK_K
    S0=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32); S1=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32); S2=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32); S3=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32)
    q_base=q_ptr+bh*T*K; k_base=k_ptr+bh*T*K; v_base=v_ptr+bh*T*V; g_base=g_ptr+bh*T; b_base=beta_ptr+bh*T; o_base=out_ptr+bh*T*V
    for t in range(T):
        d_t=tl.exp(tl.load(g_base+t).to(tl.float32)); b2_t=tl.load(b_base+t).to(tl.float32)*2.0
        k0=tl.load(k_base+t*K+off_k0).to(tl.float32); k1=tl.load(k_base+t*K+off_k1).to(tl.float32); k2=tl.load(k_base+t*K+off_k2).to(tl.float32); k3=tl.load(k_base+t*K+off_k3).to(tl.float32)
        q0=tl.load(q_base+t*K+off_k0).to(tl.float32); q1=tl.load(q_base+t*K+off_k1).to(tl.float32); q2=tl.load(q_base+t*K+off_k2).to(tl.float32); q3=tl.load(q_base+t*K+off_k3).to(tl.float32)
        v_s=tl.load(v_base+t*V+off_v).to(tl.float32)
        S0=S0*d_t; S1=S1*d_t; S2=S2*d_t; S3=S3*d_t
        pred=tl.zeros((BLOCK_V,),dtype=tl.float32)
        pred+=tl.sum(S0*tl.expand_dims(k0,1),axis=0); pred+=tl.sum(S1*tl.expand_dims(k1,1),axis=0); pred+=tl.sum(S2*tl.expand_dims(k2,1),axis=0); pred+=tl.sum(S3*tl.expand_dims(k3,1),axis=0)
        delta=(v_s-pred)*b2_t; drow=tl.expand_dims(delta,0)
        S0=S0+tl.expand_dims(k0,1)*drow; S1=S1+tl.expand_dims(k1,1)*drow; S2=S2+tl.expand_dims(k2,1)*drow; S3=S3+tl.expand_dims(k3,1)*drow
        acc=tl.zeros((BLOCK_V,),dtype=tl.float32)
        acc+=tl.sum(S0*tl.expand_dims(q0,1),axis=0); acc+=tl.sum(S1*tl.expand_dims(q1,1),axis=0); acc+=tl.sum(S2*tl.expand_dims(q2,1),axis=0); acc+=tl.sum(S3*tl.expand_dims(q3,1),axis=0)
        tl.store(o_base+t*V+off_v, acc*scale)

@triton.jit
def bwd128(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,do_ptr,s_ptr,dq_ptr,dk_ptr,dv_ptr,db_ptr,dg_ptr,BH,T,K,V,scale,BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,NUM_V:tl.constexpr):
    pid=tl.program_id(0); bh=pid//NUM_V; bv=pid%NUM_V
    off_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
    off_k0=tl.arange(0,BLOCK_K); off_k1=tl.arange(0,BLOCK_K)+BLOCK_K; off_k2=tl.arange(0,BLOCK_K)+2*BLOCK_K; off_k3=tl.arange(0,BLOCK_K)+3*BLOCK_K
    H0=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32); H1=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32); H2=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32); H3=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32)
    k0_n=tl.zeros((BLOCK_K,),dtype=tl.float32); k1_n=tl.zeros((BLOCK_K,),dtype=tl.float32); k2_n=tl.zeros((BLOCK_K,),dtype=tl.float32); k3_n=tl.zeros((BLOCK_K,),dtype=tl.float32)
    d_n=0.0; b2_n=0.0; first=True
    q_base=q_ptr+bh*T*K; k_base=k_ptr+bh*T*K; v_base=v_ptr+bh*T*V; g_base=g_ptr+bh*T; b_base=beta_ptr+bh*T; do_base=do_ptr+bh*T*V
    dv_base=dv_ptr+bh*T*V; dq_base=dq_ptr+bh*T*K; dk_base=dk_ptr+bh*T*K; db_base=db_ptr+bh*T; dg_base=dg_ptr+bh*T
    for idx in range(T):
        t=T-1-idx
        if first==False:
            u_n=tl.zeros((BLOCK_V,),dtype=tl.float32)
            u_n+=tl.sum(H0*tl.expand_dims(k0_n,1),axis=0); u_n+=tl.sum(H1*tl.expand_dims(k1_n,1),axis=0); u_n+=tl.sum(H2*tl.expand_dims(k2_n,1),axis=0); u_n+=tl.sum(H3*tl.expand_dims(k3_n,1),axis=0)
            H0=d_n*(H0 - tl.expand_dims(k0_n,1)*tl.expand_dims(u_n,0)*b2_n)
            H1=d_n*(H1 - tl.expand_dims(k1_n,1)*tl.expand_dims(u_n,0)*b2_n)
            H2=d_n*(H2 - tl.expand_dims(k2_n,1)*tl.expand_dims(u_n,0)*b2_n)
            H3=d_n*(H3 - tl.expand_dims(k3_n,1)*tl.expand_dims(u_n,0)*b2_n)
        k0=tl.load(k_base+t*K+off_k0).to(tl.float32); k1=tl.load(k_base+t*K+off_k1).to(tl.float32); k2=tl.load(k_base+t*K+off_k2).to(tl.float32); k3=tl.load(k_base+t*K+off_k3).to(tl.float32)
        q0=tl.load(q_base+t*K+off_k0).to(tl.float32); q1=tl.load(q_base+t*K+off_k1).to(tl.float32); q2=tl.load(q_base+t*K+off_k2).to(tl.float32); q3=tl.load(q_base+t*K+off_k3).to(tl.float32)
        v_s=tl.load(v_base+t*V+off_v).to(tl.float32); do_s=tl.load(do_base+t*V+off_v).to(tl.float32)
        d_t=tl.exp(tl.load(g_base+t).to(tl.float32)); b2_t=tl.load(b_base+t).to(tl.float32)*2.0
        H0=H0+tl.expand_dims(q0,1)*tl.expand_dims(do_s,0)*scale; H1=H1+tl.expand_dims(q1,1)*tl.expand_dims(do_s,0)*scale; H2=H2+tl.expand_dims(q2,1)*tl.expand_dims(do_s,0)*scale; H3=H3+tl.expand_dims(q3,1)*tl.expand_dims(do_s,0)*scale
        u=tl.zeros((BLOCK_V,),dtype=tl.float32)
        u+=tl.sum(H0*tl.expand_dims(k0,1),axis=0); u+=tl.sum(H1*tl.expand_dims(k1,1),axis=0); u+=tl.sum(H2*tl.expand_dims(k2,1),axis=0); u+=tl.sum(H3*tl.expand_dims(k3,1),axis=0)
        # dv : each V shard writes its own slice directly (no atomic needed)
        tl.store(dv_base+t*V+off_v, u*b2_t)
        base_s=s_ptr+bh*(T+1)*K*V+(t+1)*K*V; base_sp=s_ptr+bh*(T+1)*K*V+t*K*V
        S0c=tl.load(tl.make_block_ptr(base=base_s,shape=(K,V),strides=(V,1),offsets=(0,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        S1c=tl.load(tl.make_block_ptr(base=base_s,shape=(K,V),strides=(V,1),offsets=(BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        S2c=tl.load(tl.make_block_ptr(base=base_s,shape=(K,V),strides=(V,1),offsets=(2*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        S3c=tl.load(tl.make_block_ptr(base=base_s,shape=(K,V),strides=(V,1),offsets=(3*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        Sp0=tl.load(tl.make_block_ptr(base=base_sp,shape=(K,V),strides=(V,1),offsets=(0,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        Sp1=tl.load(tl.make_block_ptr(base=base_sp,shape=(K,V),strides=(V,1),offsets=(BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        Sp2=tl.load(tl.make_block_ptr(base=base_sp,shape=(K,V),strides=(V,1),offsets=(2*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        Sp3=tl.load(tl.make_block_ptr(base=base_sp,shape=(K,V),strides=(V,1),offsets=(3*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        p=tl.zeros((BLOCK_V,),dtype=tl.float32)
        p+=tl.sum(Sp0*tl.expand_dims(k0,1),axis=0); p+=tl.sum(Sp1*tl.expand_dims(k1,1),axis=0); p+=tl.sum(Sp2*tl.expand_dims(k2,1),axis=0); p+=tl.sum(Sp3*tl.expand_dims(k3,1),axis=0)
        a=v_s - p*d_t
        db_s=tl.sum(u*a)*2.0
        tl.atomic_add(db_base+t, db_s)
        fdot=tl.sum(H0*Sp0)+tl.sum(H1*Sp1)+tl.sum(H2*Sp2)+tl.sum(H3*Sp3)
        updot=tl.sum(u*p)
        dg_s=d_t*(fdot - b2_t*updot)
        tl.atomic_add(dg_base+t, dg_s)
        dq0=tl.sum(S0c*tl.expand_dims(do_s,0),axis=1)*scale; dq1=tl.sum(S1c*tl.expand_dims(do_s,0),axis=1)*scale; dq2=tl.sum(S2c*tl.expand_dims(do_s,0),axis=1)*scale; dq3=tl.sum(S3c*tl.expand_dims(do_s,0),axis=1)*scale
        tl.atomic_add(dq_base+t*K+off_k0, dq0); tl.atomic_add(dq_base+t*K+off_k1, dq1); tl.atomic_add(dq_base+t*K+off_k2, dq2); tl.atomic_add(dq_base+t*K+off_k3, dq3)
        hv0=tl.sum(H0*tl.expand_dims(v_s,0),axis=1); hv1=tl.sum(H1*tl.expand_dims(v_s,0),axis=1); hv2=tl.sum(H2*tl.expand_dims(v_s,0),axis=1); hv3=tl.sum(H3*tl.expand_dims(v_s,0),axis=1)
        hp0=tl.sum(H0*tl.expand_dims(p,0),axis=1); hp1=tl.sum(H1*tl.expand_dims(p,0),axis=1); hp2=tl.sum(H2*tl.expand_dims(p,0),axis=1); hp3=tl.sum(H3*tl.expand_dims(p,0),axis=1)
        su0=tl.sum(Sp0*tl.expand_dims(u,0),axis=1); su1=tl.sum(Sp1*tl.expand_dims(u,0),axis=1); su2=tl.sum(Sp2*tl.expand_dims(u,0),axis=1); su3=tl.sum(Sp3*tl.expand_dims(u,0),axis=1)
        dk0=(hv0 - d_t*hp0 - d_t*su0)*b2_t; dk1=(hv1 - d_t*hp1 - d_t*su1)*b2_t; dk2=(hv2 - d_t*hp2 - d_t*su2)*b2_t; dk3=(hv3 - d_t*hp3 - d_t*su3)*b2_t
        tl.atomic_add(dk_base+t*K+off_k0, dk0); tl.atomic_add(dk_base+t*K+off_k1, dk1); tl.atomic_add(dk_base+t*K+off_k2, dk2); tl.atomic_add(dk_base+t*K+off_k3, dk3)
        k0_n=k0; k1_n=k1; k2_n=k2; k3_n=k3; d_n=d_t; b2_n=b2_t; first=False

# ---------- generic fallback ----------
@triton.jit
def generic_fwd(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,out_ptr,s_ptr,scratch_ptr,BH,T,K,V,scale,BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,NUM_K:tl.constexpr,NUM_V:tl.constexpr):
    pid=tl.program_id(0)
    q_base=q_ptr+pid*T*K; k_base=k_ptr+pid*T*K; v_base=v_ptr+pid*T*V; g_base=g_ptr+pid*T; b_base=beta_ptr+pid*T; o_base=out_ptr+pid*T*V
    s0=s_ptr+pid*(T+1)*K*V
    # zero S0
    for bk in range(NUM_K):
        for bv in range(NUM_V):
            p=tl.make_block_ptr(base=s0,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
            tl.store(p, tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32), boundary_check=(0,1))
    for t in range(T):
        d_t=tl.exp(tl.load(g_base+t).to(tl.float32)); b2_t=tl.load(b_base+t).to(tl.float32)*2.0
        for bk in range(NUM_K):
            for bv in range(NUM_V):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)*d_t; tl.store(sp,S,boundary_check=(0,1))
        for bv in range(NUM_V):
            pred=tl.zeros((BLOCK_V,),dtype=tl.float32)
            for bk in range(NUM_K):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                k_blk=tl.load(k_base+t*K+offs_k, mask=offs_k<K).to(tl.float32)
                pred+=tl.sum(S*tl.expand_dims(k_blk,1),axis=0)
            offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
            v_blk=tl.load(v_base+t*V+offs_v, mask=offs_v<V).to(tl.float32)
            delta=(v_blk-pred)*b2_t
            for bk in range(NUM_K):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                k_blk=tl.load(k_base+t*K+offs_k, mask=offs_k<K).to(tl.float32)
                S=S+tl.expand_dims(k_blk,1)*tl.expand_dims(delta,0)
                tl.store(sp,S,boundary_check=(0,1))
        for bv in range(NUM_V):
            offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
            acc=tl.zeros((BLOCK_V,),dtype=tl.float32)
            for bk in range(NUM_K):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                q_blk=tl.load(q_base+t*K+offs_k, mask=offs_k<K).to(tl.float32)
                acc+=tl.sum(S*tl.expand_dims(q_blk,1),axis=0)
            tl.store(o_base+t*V+offs_v, acc*scale, mask=offs_v<V)
        s_dst=s_ptr+pid*(T+1)*K*V+(t+1)*K*V
        for bk in range(NUM_K):
            for bv in range(NUM_V):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1))
                dp=tl.make_block_ptr(base=s_dst,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                tl.store(dp,S,boundary_check=(0,1))

@triton.jit
def generic_infer(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,out_ptr,scratch_ptr,BH,T,K,V,scale,BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,NUM_K:tl.constexpr,NUM_V:tl.constexpr):
    pid=tl.program_id(0)
    q_base=q_ptr+pid*T*K; k_base=k_ptr+pid*T*K; v_base=v_ptr+pid*T*V; g_base=g_ptr+pid*T; b_base=beta_ptr+pid*T; o_base=out_ptr+pid*T*V
    for t in range(T):
        d_t=tl.exp(tl.load(g_base+t).to(tl.float32)); b2_t=tl.load(b_base+t).to(tl.float32)*2.0
        for bk in range(NUM_K):
            for bv in range(NUM_V):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)*d_t; tl.store(sp,S,boundary_check=(0,1))
        for bv in range(NUM_V):
            pred=tl.zeros((BLOCK_V,),dtype=tl.float32)
            for bk in range(NUM_K):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                k_blk=tl.load(k_base+t*K+offs_k, mask=offs_k<K).to(tl.float32)
                pred+=tl.sum(S*tl.expand_dims(k_blk,1),axis=0)
            offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
            v_blk=tl.load(v_base+t*V+offs_v, mask=offs_v<V).to(tl.float32)
            delta=(v_blk-pred)*b2_t
            for bk in range(NUM_K):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                k_blk=tl.load(k_base+t*K+offs_k, mask=offs_k<K).to(tl.float32)
                S=S+tl.expand_dims(k_blk,1)*tl.expand_dims(delta,0); tl.store(sp,S,boundary_check=(0,1))
        for bv in range(NUM_V):
            offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
            acc=tl.zeros((BLOCK_V,),dtype=tl.float32)
            for bk in range(NUM_K):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                q_blk=tl.load(q_base+t*K+offs_k, mask=offs_k<K).to(tl.float32)
                acc+=tl.sum(S*tl.expand_dims(q_blk,1),axis=0)
            tl.store(o_base+t*V+offs_v, acc*scale, mask=offs_v<V)

@triton.jit
def generic_bwd(q_ptr,k_ptr,v_ptr,g_ptr,beta_ptr,do_ptr,s_ptr,dq_ptr,dk_ptr,dv_ptr,db_ptr,dg_ptr,h_ptr,BH,T,K,V,scale,BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,NUM_K:tl.constexpr,NUM_V:tl.constexpr):
    pid=tl.program_id(0)
    q_base=q_ptr+pid*T*K; k_base=k_ptr+pid*T*K; v_base=v_ptr+pid*T*V; g_base=g_ptr+pid*T; b_base=beta_ptr+pid*T; do_base=do_ptr+pid*T*V
    for idx in range(T):
        t=T-1-idx
        if idx!=0:
            t_next=t+1; d_n=tl.exp(tl.load(g_base+t_next).to(tl.float32)); b2_n=tl.load(b_base+t_next).to(tl.float32)*2.0
            for bv in range(NUM_V):
                u_n=tl.zeros((BLOCK_V,),dtype=tl.float32)
                for bk in range(NUM_K):
                    hb=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                    offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K); k_blk=tl.load(k_base+t_next*K+offs_k, mask=offs_k<K).to(tl.float32)
                    u_n+=tl.sum(hb*tl.expand_dims(k_blk,1),axis=0)
                for bk in range(NUM_K):
                    hb=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                    offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K); k_blk=tl.load(k_base+t_next*K+offs_k, mask=offs_k<K).to(tl.float32)
                    hn=d_n*(hb - tl.expand_dims(k_blk,1)*tl.expand_dims(u_n,0)*b2_n)
                    tl.store(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),hn,boundary_check=(0,1))
        for bk in range(NUM_K):
            for bv in range(NUM_V):
                hb=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K); offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
                q_blk=tl.load(q_base+t*K+offs_k, mask=offs_k<K).to(tl.float32); do_blk=tl.load(do_base+t*V+offs_v, mask=offs_v<V).to(tl.float32)
                hb=hb+tl.expand_dims(q_blk,1)*tl.expand_dims(do_blk,0)*scale
                tl.store(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),hb,boundary_check=(0,1))
        d_t=tl.exp(tl.load(g_base+t).to(tl.float32)); b2_t=tl.load(b_base+t).to(tl.float32)*2.0
        fdot=0.0
        for bk in range(NUM_K):
            for bv in range(NUM_V):
                hb=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                sp=tl.load(tl.make_block_ptr(base=s_ptr+pid*(T+1)*K*V+t*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                fdot+=tl.sum(hb*sp)
        updot=0.0; db_acc=0.0
        for bv in range(NUM_V):
            u_b=tl.zeros((BLOCK_V,),dtype=tl.float32)
            for bk in range(NUM_K):
                hb=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K); k_blk=tl.load(k_base+t*K+offs_k, mask=offs_k<K).to(tl.float32)
                u_b+=tl.sum(hb*tl.expand_dims(k_blk,1),axis=0)
            p_b=tl.zeros((BLOCK_V,),dtype=tl.float32)
            for bk in range(NUM_K):
                sp=tl.load(tl.make_block_ptr(base=s_ptr+pid*(T+1)*K*V+t*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K); k_blk=tl.load(k_base+t*K+offs_k, mask=offs_k<K).to(tl.float32)
                p_b+=tl.sum(sp*tl.expand_dims(k_blk,1),axis=0)
            updot+=tl.sum(u_b*p_b)
            offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
            v_blk=tl.load(v_base+t*V+offs_v, mask=offs_v<V).to(tl.float32)
            a_b=v_blk - p_b*d_t
            tl.store(dv_ptr+pid*T*V+t*V+offs_v, u_b*b2_t, mask=offs_v<V)
            db_acc+=tl.sum(u_b*a_b)
        tl.store(db_ptr+pid*T+t, db_acc*2.0)
        tl.store(dg_ptr+pid*T+t, d_t*(fdot - b2_t*updot))
        for bk in range(NUM_K):
            dq_bk=tl.zeros((BLOCK_K,),dtype=tl.float32); hv_bk=tl.zeros((BLOCK_K,),dtype=tl.float32); hp_bk=tl.zeros((BLOCK_K,),dtype=tl.float32); su_bk=tl.zeros((BLOCK_K,),dtype=tl.float32)
            for bv in range(NUM_V):
                sc=tl.load(tl.make_block_ptr(base=s_ptr+pid*(T+1)*K*V+(t+1)*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V); do_blk=tl.load(do_base+t*V+offs_v, mask=offs_v<V).to(tl.float32)
                dq_bk+=tl.sum(sc*tl.expand_dims(do_blk,0),axis=1)*scale
                hb=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                v_blk=tl.load(v_base+t*V+offs_v, mask=offs_v<V).to(tl.float32)
                hv_bk+=tl.sum(hb*tl.expand_dims(v_blk,0),axis=1)
                p_b=tl.zeros((BLOCK_V,),dtype=tl.float32)
                for bk2 in range(NUM_K):
                    sp2=tl.load(tl.make_block_ptr(base=s_ptr+pid*(T+1)*K*V+t*K*V,shape=(K,V),strides=(V,1),offsets=(bk2*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                    offs_k2=bk2*BLOCK_K+tl.arange(0,BLOCK_K); k2=tl.load(k_base+t*K+offs_k2, mask=offs_k2<K).to(tl.float32)
                    p_b+=tl.sum(sp2*tl.expand_dims(k2,1),axis=0)
                hp_bk+=tl.sum(hb*tl.expand_dims(p_b,0),axis=1)
                u_b=tl.zeros((BLOCK_V,),dtype=tl.float32)
                for bk2 in range(NUM_K):
                    hb2=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk2*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                    offs_k2=bk2*BLOCK_K+tl.arange(0,BLOCK_K); k2=tl.load(k_base+t*K+offs_k2, mask=offs_k2<K).to(tl.float32)
                    u_b+=tl.sum(hb2*tl.expand_dims(k2,1),axis=0)
                sp=tl.load(tl.make_block_ptr(base=s_ptr+pid*(T+1)*K*V+t*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                su_bk+=tl.sum(sp*tl.expand_dims(u_b,0),axis=1)
            offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
            tl.store(dq_ptr+pid*T*K+t*K+offs_k, dq_bk, mask=offs_k<K)
            tl.store(dk_ptr+pid*T*K+t*K+offs_k, (hv_bk - d_t*hp_bk - d_t*su_bk)*b2_t, mask=offs_k<K)

class F(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,e,g,beta,gamma,scale):
        B,H,T,K=q.shape; Vd=v.shape[-1]; BH=B*H; scl=float(scale.item()) if isinstance(scale,torch.Tensor) else float(scale)
        qf=q.reshape(BH,T,K).contiguous(); kf=k.reshape(BH,T,K).contiguous(); vf=v.reshape(BH,T,Vd).contiguous()
        gf=g.reshape(BH,T).contiguous(); betaf=beta.reshape(BH,T).contiguous()
        out=torch.empty(BH,T,Vd,device=q.device,dtype=torch.float32)
        is128=(K==128 and Vd==128)
        s_buf=torch.empty(BH*(T+1)*K*Vd,device=q.device,dtype=torch.float32)
        if is128:
            fwd128_store[(BH*4,)](qf,kf,vf,gf,betaf,out,s_buf,BH,T,K,Vd,scl,32,32,4,num_warps=4,num_stages=2)
        else:
            Bk=32; Bv=32; Nk=(K+Bk-1)//Bk; Nv=(Vd+Bv-1)//Bv; scratch=torch.zeros(BH*K*Vd,device=q.device,dtype=torch.float32)
            generic_fwd[(BH,)](qf,kf,vf,gf,betaf,out,s_buf,scratch,BH,T,K,Vd,scl,Bk,Bv,Nk,Nv)
        ctx.save_for_backward(qf,kf,vf,gf,betaf,s_buf); ctx.shapes=(B,H,T,K,Vd,BH,scl); ctx.is128=is128
        ctx.qd=q.dtype; ctx.kd=k.dtype; ctx.vd=v.dtype; ctx.ed=e.dtype; ctx.gd=g.dtype; ctx.bd=beta.dtype; ctx.gamd=gamma.dtype
        return out.reshape(B,H,T,Vd).to(q.dtype)
    @staticmethod
    def backward(ctx,go):
        qf,kf,vf,gf,betaf,s_buf=ctx.saved_tensors; B,H,T,K,Vd,BH,scl=ctx.shapes; gof=go.reshape(BH,T,Vd).contiguous().float()
        dq=torch.zeros(BH,T,K,device=qf.device,dtype=torch.float32); dk=torch.zeros(BH,T,K,device=qf.device,dtype=torch.float32)
        dv=torch.empty(BH,T,Vd,device=qf.device,dtype=torch.float32); db=torch.zeros(BH,T,device=qf.device,dtype=torch.float32); dg=torch.zeros(BH,T,device=qf.device,dtype=torch.float32)
        if ctx.is128:
            bwd128[(BH*4,)](qf,kf,vf,gf,betaf,gof,s_buf,dq,dk,dv,db,dg,BH,T,K,Vd,scl,32,32,4,num_warps=4,num_stages=2)
        else:
            Bk=32; Bv=32; Nk=(K+Bk-1)//Bk; Nv=(Vd+Bv-1)//Bv; h=torch.zeros(BH*K*Vd,device=qf.device,dtype=torch.float32)
            generic_bwd[(BH,)](qf,kf,vf,gf,betaf,gof,s_buf,dq,dk,dv,db,dg,h,BH,T,K,Vd,scl,Bk,Bv,Nk,Nv)
        return dq.view(B,H,T,K).to(ctx.qd), dk.view(B,H,T,K).to(ctx.kd), dv.view(B,H,T,Vd).to(ctx.vd), torch.zeros(B,H,T,K,device=qf.device,dtype=ctx.ed), dg.view(B,H,T).to(ctx.gd), db.view(B,H,T).to(ctx.bd), torch.zeros(B,H,T,device=qf.device,dtype=ctx.gamd), None

def kernel_function(q,k,v,e,g,beta,gamma,scale):
    if torch.is_grad_enabled() and any(getattr(x,'requires_grad',False) for x in (q,k,v,e,g,beta,gamma)):
        return F.apply(q,k,v,e,g,beta,gamma,scale)
    B,H,T,K=q.shape; Vd=v.shape[-1]; BH=B*H; scl=float(scale.item()) if isinstance(scale,torch.Tensor) else float(scale)
    qf=q.reshape(BH,T,K).contiguous(); kf=k.reshape(BH,T,K).contiguous(); vf=v.reshape(BH,T,Vd).contiguous(); gf=g.reshape(BH,T).contiguous(); betaf=beta.reshape(BH,T).contiguous()
    out=torch.empty(BH,T,Vd,device=q.device,dtype=torch.float32)
    if K==128 and Vd==128:
        infer128[(BH*4,)](qf,kf,vf,gf,betaf,out,BH,T,K,Vd,scl,32,32,4,num_warps=4,num_stages=2)
    else:
        Bk=32; Bv=32; Nk=(K+Bk-1)//Bk; Nv=(Vd+Bv-1)//Bv; scratch=torch.zeros(BH*K*Vd,device=q.device,dtype=torch.float32)
        generic_infer[(BH,)](qf,kf,vf,gf,betaf,out,scratch,BH,T,K,Vd,scl,Bk,Bv,Nk,Nv)
    return out.reshape(B,H,T,Vd).to(q.dtype)
```
