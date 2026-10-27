```python
import torch
import triton
import triton.language as tl

# ---------- fast forward 128 : BH*4 programs, S in registers, store S ----------
@triton.jit
def fast_fwd_128_store(q_ptr,k_ptr,v_ptr,d_ptr,b2_ptr,out_ptr,s_ptr,
    BH,T,K,V,
    sqbh,sqt,sqk, skbh,skt,skk, svbh,svt,svk, sdbh,sdt, sbbh,sbt, sobh,sot,sov,
    sbuf_bh,sbuf_t, scale,
    BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,NUM_V:tl.constexpr):
    pid=tl.program_id(0)
    bh=pid//NUM_V
    bv=pid%NUM_V
    off_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
    off_k0=tl.arange(0,BLOCK_K)
    off_k1=tl.arange(0,BLOCK_K)+BLOCK_K
    off_k2=tl.arange(0,BLOCK_K)+2*BLOCK_K
    off_k3=tl.arange(0,BLOCK_K)+3*BLOCK_K
    S0=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32)
    S1=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32)
    S2=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32)
    S3=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32)
    for t in range(T):
        d_t=tl.load(d_ptr+bh*sdbh+t*sdt)
        b2_t=tl.load(b2_ptr+bh*sbbh+t*sbt)
        k0=tl.load(k_ptr+bh*skbh+t*skt+off_k0*skk).to(tl.float32)
        k1=tl.load(k_ptr+bh*skbh+t*skt+off_k1*skk).to(tl.float32)
        k2=tl.load(k_ptr+bh*skbh+t*skt+off_k2*skk).to(tl.float32)
        k3=tl.load(k_ptr+bh*skbh+t*skt+off_k3*skk).to(tl.float32)
        q0=tl.load(q_ptr+bh*sqbh+t*sqt+off_k0*sqk).to(tl.float32)
        q1=tl.load(q_ptr+bh*sqbh+t*sqt+off_k1*sqk).to(tl.float32)
        q2=tl.load(q_ptr+bh*sqbh+t*sqt+off_k2*sqk).to(tl.float32)
        q3=tl.load(q_ptr+bh*sqbh+t*sqt+off_k3*sqk).to(tl.float32)
        v_s=tl.load(v_ptr+bh*svbh+t*svt+off_v*svk).to(tl.float32)
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
        base=s_ptr+bh*sbuf_bh+(t+1)*sbuf_t
        tl.store(tl.make_block_ptr(base=base,shape=(K,V),strides=(V,1),offsets=(0,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),S0,boundary_check=(0,1))
        tl.store(tl.make_block_ptr(base=base,shape=(K,V),strides=(V,1),offsets=(BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),S1,boundary_check=(0,1))
        tl.store(tl.make_block_ptr(base=base,shape=(K,V),strides=(V,1),offsets=(2*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),S2,boundary_check=(0,1))
        tl.store(tl.make_block_ptr(base=base,shape=(K,V),strides=(V,1),offsets=(3*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),S3,boundary_check=(0,1))
        out_acc=tl.zeros((BLOCK_V,),dtype=tl.float32)
        out_acc+=tl.sum(S0*tl.expand_dims(q0,1),axis=0)
        out_acc+=tl.sum(S1*tl.expand_dims(q1,1),axis=0)
        out_acc+=tl.sum(S2*tl.expand_dims(q2,1),axis=0)
        out_acc+=tl.sum(S3*tl.expand_dims(q3,1),axis=0)
        tl.store(out_ptr+bh*sobh+t*sot+off_v*sov,out_acc*scale)

@triton.jit
def fast_infer_128(q_ptr,k_ptr,v_ptr,d_ptr,b2_ptr,out_ptr,
    BH,T,K,V,
    sqbh,sqt,sqk, skbh,skt,skk, svbh,svt,svk, sdbh,sdt, sbbh,sbt, sobh,sot,sov, scale,
    BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,NUM_V:tl.constexpr):
    pid=tl.program_id(0); bh=pid//NUM_V; bv=pid%NUM_V
    off_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
    off_k0=tl.arange(0,BLOCK_K); off_k1=tl.arange(0,BLOCK_K)+BLOCK_K; off_k2=tl.arange(0,BLOCK_K)+2*BLOCK_K; off_k3=tl.arange(0,BLOCK_K)+3*BLOCK_K
    S0=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32); S1=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32); S2=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32); S3=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32)
    for t in range(T):
        d_t=tl.load(d_ptr+bh*sdbh+t*sdt); b2_t=tl.load(b2_ptr+bh*sbbh+t*sbt)
        k0=tl.load(k_ptr+bh*skbh+t*skt+off_k0*skk).to(tl.float32); k1=tl.load(k_ptr+bh*skbh+t*skt+off_k1*skk).to(tl.float32); k2=tl.load(k_ptr+bh*skbh+t*skt+off_k2*skk).to(tl.float32); k3=tl.load(k_ptr+bh*skbh+t*skt+off_k3*skk).to(tl.float32)
        q0=tl.load(q_ptr+bh*sqbh+t*sqt+off_k0*sqk).to(tl.float32); q1=tl.load(q_ptr+bh*sqbh+t*sqt+off_k1*sqk).to(tl.float32); q2=tl.load(q_ptr+bh*sqbh+t*sqt+off_k2*sqk).to(tl.float32); q3=tl.load(q_ptr+bh*sqbh+t*sqt+off_k3*sqk).to(tl.float32)
        v_s=tl.load(v_ptr+bh*svbh+t*svt+off_v*svk).to(tl.float32)
        S0=S0*d_t; S1=S1*d_t; S2=S2*d_t; S3=S3*d_t
        pred=tl.zeros((BLOCK_V,),dtype=tl.float32)
        pred+=tl.sum(S0*tl.expand_dims(k0,1),axis=0); pred+=tl.sum(S1*tl.expand_dims(k1,1),axis=0); pred+=tl.sum(S2*tl.expand_dims(k2,1),axis=0); pred+=tl.sum(S3*tl.expand_dims(k3,1),axis=0)
        delta=(v_s-pred)*b2_t; drow=tl.expand_dims(delta,0)
        S0=S0+tl.expand_dims(k0,1)*drow; S1=S1+tl.expand_dims(k1,1)*drow; S2=S2+tl.expand_dims(k2,1)*drow; S3=S3+tl.expand_dims(k3,1)*drow
        out_acc=tl.zeros((BLOCK_V,),dtype=tl.float32)
        out_acc+=tl.sum(S0*tl.expand_dims(q0,1),axis=0); out_acc+=tl.sum(S1*tl.expand_dims(q1,1),axis=0); out_acc+=tl.sum(S2*tl.expand_dims(q2,1),axis=0); out_acc+=tl.sum(S3*tl.expand_dims(q3,1),axis=0)
        tl.store(out_ptr+bh*sobh+t*sot+off_v*sov,out_acc*scale)

@triton.jit
def fast_bwd_128(q_ptr,k_ptr,v_ptr,d_ptr,b2_ptr,do_ptr,s_ptr,dq_part_ptr,dk_part_ptr,dv_ptr,db_part_ptr,dg_part_ptr,
    BH,T,K,V,
    sqbh,sqt,sqk, skbh,skt,skk, svbh,svt,svk, sdbh,sdt, sbbh,sbt, sdobh,sdot,sdov, sbuf_bh,sbuf_t, scale,
    BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,NUM_V:tl.constexpr):
    pid=tl.program_id(0); bh=pid//NUM_V; bv=pid%NUM_V
    off_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
    off_k0=tl.arange(0,BLOCK_K); off_k1=tl.arange(0,BLOCK_K)+BLOCK_K; off_k2=tl.arange(0,BLOCK_K)+2*BLOCK_K; off_k3=tl.arange(0,BLOCK_K)+3*BLOCK_K
    H0=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32); H1=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32); H2=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32); H3=tl.zeros((BLOCK_K,BLOCK_V),dtype=tl.float32)
    k0_n=tl.zeros((BLOCK_K,),dtype=tl.float32); k1_n=tl.zeros((BLOCK_K,),dtype=tl.float32); k2_n=tl.zeros((BLOCK_K,),dtype=tl.float32); k3_n=tl.zeros((BLOCK_K,),dtype=tl.float32); d_n=0.0; b2_n=0.0
    first=True
    for idx in range(T):
        t=T-1-idx
        if first==False:
            u_n=tl.zeros((BLOCK_V,),dtype=tl.float32)
            u_n+=tl.sum(H0*tl.expand_dims(k0_n,1),axis=0); u_n+=tl.sum(H1*tl.expand_dims(k1_n,1),axis=0); u_n+=tl.sum(H2*tl.expand_dims(k2_n,1),axis=0); u_n+=tl.sum(H3*tl.expand_dims(k3_n,1),axis=0)
            H0=d_n*(H0 - tl.expand_dims(k0_n,1)*tl.expand_dims(u_n,0)*b2_n)
            H1=d_n*(H1 - tl.expand_dims(k1_n,1)*tl.expand_dims(u_n,0)*b2_n)
            H2=d_n*(H2 - tl.expand_dims(k2_n,1)*tl.expand_dims(u_n,0)*b2_n)
            H3=d_n*(H3 - tl.expand_dims(k3_n,1)*tl.expand_dims(u_n,0)*b2_n)
        k0=tl.load(k_ptr+bh*skbh+t*skt+off_k0*skk).to(tl.float32); k1=tl.load(k_ptr+bh*skbh+t*skt+off_k1*skk).to(tl.float32); k2=tl.load(k_ptr+bh*skbh+t*skt+off_k2*skk).to(tl.float32); k3=tl.load(k_ptr+bh*skbh+t*skt+off_k3*skk).to(tl.float32)
        q0=tl.load(q_ptr+bh*sqbh+t*sqt+off_k0*sqk).to(tl.float32); q1=tl.load(q_ptr+bh*sqbh+t*sqt+off_k1*sqk).to(tl.float32); q2=tl.load(q_ptr+bh*sqbh+t*sqt+off_k2*sqk).to(tl.float32); q3=tl.load(q_ptr+bh*sqbh+t*sqt+off_k3*sqk).to(tl.float32)
        v_s=tl.load(v_ptr+bh*svbh+t*svt+off_v*svk).to(tl.float32); do_s=tl.load(do_ptr+bh*sdobh+t*sdot+off_v*sdov).to(tl.float32)
        d_t=tl.load(d_ptr+bh*sdbh+t*sdt); b2_t=tl.load(b2_ptr+bh*sbbh+t*sbt)
        H0=H0+tl.expand_dims(q0,1)*tl.expand_dims(do_s,0)*scale; H1=H1+tl.expand_dims(q1,1)*tl.expand_dims(do_s,0)*scale; H2=H2+tl.expand_dims(q2,1)*tl.expand_dims(do_s,0)*scale; H3=H3+tl.expand_dims(q3,1)*tl.expand_dims(do_s,0)*scale
        u=tl.zeros((BLOCK_V,),dtype=tl.float32)
        u+=tl.sum(H0*tl.expand_dims(k0,1),axis=0); u+=tl.sum(H1*tl.expand_dims(k1,1),axis=0); u+=tl.sum(H2*tl.expand_dims(k2,1),axis=0); u+=tl.sum(H3*tl.expand_dims(k3,1),axis=0)
        dv_s=u*b2_t; tl.store(dv_ptr+bh*svbh+t*svt+off_v*svk,dv_s)
        base_s=s_ptr+bh*sbuf_bh+(t+1)*sbuf_t; base_sp=s_ptr+bh*sbuf_bh+t*sbuf_t
        S0c=tl.load(tl.make_block_ptr(base=base_s,shape=(K,V),strides=(V,1),offsets=(0,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        S1c=tl.load(tl.make_block_ptr(base=base_s,shape=(K,V),strides=(V,1),offsets=(BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        S2c=tl.load(tl.make_block_ptr(base=base_s,shape=(K,V),strides=(V,1),offsets=(2*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        S3c=tl.load(tl.make_block_ptr(base=base_s,shape=(K,V),strides=(V,1),offsets=(3*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        Sp0=tl.load(tl.make_block_ptr(base=base_sp,shape=(K,V),strides=(V,1),offsets=(0,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        Sp1=tl.load(tl.make_block_ptr(base=base_sp,shape=(K,V),strides=(V,1),offsets=(BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        Sp2=tl.load(tl.make_block_ptr(base=base_sp,shape=(K,V),strides=(V,1),offsets=(2*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        Sp3=tl.load(tl.make_block_ptr(base=base_sp,shape=(K,V),strides=(V,1),offsets=(3*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
        p=tl.zeros((BLOCK_V,),dtype=tl.float32); p+=tl.sum(Sp0*tl.expand_dims(k0,1),axis=0); p+=tl.sum(Sp1*tl.expand_dims(k1,1),axis=0); p+=tl.sum(Sp2*tl.expand_dims(k2,1),axis=0); p+=tl.sum(Sp3*tl.expand_dims(k3,1),axis=0)
        a=v_s - p*d_t; db_s=tl.sum(u*a)*2.0; tl.store(db_part_ptr+bh*(T*NUM_V)+t*NUM_V+bv,db_s)
        fdot=tl.sum(H0*Sp0)+tl.sum(H1*Sp1)+tl.sum(H2*Sp2)+tl.sum(H3*Sp3); updot=tl.sum(u*p); dg_s=d_t*(fdot - b2_t*updot); tl.store(dg_part_ptr+bh*(T*NUM_V)+t*NUM_V+bv,dg_s)
        base_dq=dq_part_ptr+bh*(T*NUM_V*K)+t*(NUM_V*K)+bv*K; base_dk=dk_part_ptr+bh*(T*NUM_V*K)+t*(NUM_V*K)+bv*K
        dq0=tl.sum(S0c*tl.expand_dims(do_s,0),axis=1)*scale; dq1=tl.sum(S1c*tl.expand_dims(do_s,0),axis=1)*scale; dq2=tl.sum(S2c*tl.expand_dims(do_s,0),axis=1)*scale; dq3=tl.sum(S3c*tl.expand_dims(do_s,0),axis=1)*scale
        tl.store(base_dq+off_k0,dq0); tl.store(base_dq+off_k1,dq1); tl.store(base_dq+off_k2,dq2); tl.store(base_dq+off_k3,dq3)
        hv0=tl.sum(H0*tl.expand_dims(v_s,0),axis=1); hv1=tl.sum(H1*tl.expand_dims(v_s,0),axis=1); hv2=tl.sum(H2*tl.expand_dims(v_s,0),axis=1); hv3=tl.sum(H3*tl.expand_dims(v_s,0),axis=1)
        hp0=tl.sum(H0*tl.expand_dims(p,0),axis=1); hp1=tl.sum(H1*tl.expand_dims(p,0),axis=1); hp2=tl.sum(H2*tl.expand_dims(p,0),axis=1); hp3=tl.sum(H3*tl.expand_dims(p,0),axis=1)
        su0=tl.sum(Sp0*tl.expand_dims(u,0),axis=1); su1=tl.sum(Sp1*tl.expand_dims(u,0),axis=1); su2=tl.sum(Sp2*tl.expand_dims(u,0),axis=1); su3=tl.sum(Sp3*tl.expand_dims(u,0),axis=1)
        dk0=(hv0 - d_t*hp0 - d_t*su0)*b2_t; dk1=(hv1 - d_t*hp1 - d_t*su1)*b2_t; dk2=(hv2 - d_t*hp2 - d_t*su2)*b2_t; dk3=(hv3 - d_t*hp3 - d_t*su3)*b2_t
        tl.store(base_dk+off_k0,dk0); tl.store(base_dk+off_k1,dk1); tl.store(base_dk+off_k2,dk2); tl.store(base_dk+off_k3,dk3)
        k0_n=k0; k1_n=k1; k2_n=k2; k3_n=k3; d_n=d_t; b2_n=b2_t; first=False

# ---------- generic kernels ----------
@triton.jit
def generic_train_fwd(q_ptr,k_ptr,v_ptr,d_ptr,b2_ptr,out_ptr,s_buf_ptr,scratch_ptr,
    BH,T,K,V,
    sqbh,sqt,sqk, skbh,skt,skk, svbh,svt,svk, sdbh,sdt, sbbh,sbt, sobh,sot,sov, sbuf_bh,sbuf_t, scale,
    BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,NUM_K:tl.constexpr,NUM_V:tl.constexpr):
    pid=tl.program_id(0)
    for t in range(T):
        d_t=tl.load(d_ptr+pid*sdbh+t*sdt); b2_t=tl.load(b2_ptr+pid*sbbh+t*sbt)
        for bk in range(NUM_K):
            for bv in range(NUM_V):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)*d_t
                tl.store(sp,S,boundary_check=(0,1))
        for bv in range(NUM_V):
            pred=tl.zeros((BLOCK_V,),dtype=tl.float32)
            for bk in range(NUM_K):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                k_blk=tl.load(k_ptr+pid*skbh+t*skt+offs_k*skk,mask=offs_k<K).to(tl.float32)
                pred+=tl.sum(S*tl.expand_dims(k_blk,1),axis=0)
            offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
            v_blk=tl.load(v_ptr+pid*svbh+t*svt+offs_v*svk,mask=offs_v<V).to(tl.float32)
            delta=(v_blk-pred)*b2_t
            for bk in range(NUM_K):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                k_blk=tl.load(k_ptr+pid*skbh+t*skt+offs_k*skk,mask=offs_k<K).to(tl.float32)
                S_new=S+tl.expand_dims(k_blk,1)*tl.expand_dims(delta,0)
                tl.store(sp,S_new,boundary_check=(0,1))
        for bv in range(NUM_V):
            offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
            out_acc=tl.zeros((BLOCK_V,),dtype=tl.float32)
            for bk in range(NUM_K):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                q_blk=tl.load(q_ptr+pid*sqbh+t*sqt+offs_k*sqk,mask=offs_k<K).to(tl.float32)
                out_acc+=tl.sum(S*tl.expand_dims(q_blk,1),axis=0)
            tl.store(out_ptr+pid*sobh+t*sot+offs_v*sov,out_acc*scale,mask=offs_v<V)
        for bk in range(NUM_K):
            for bv in range(NUM_V):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1))
                dst=tl.make_block_ptr(base=s_buf_ptr+pid*sbuf_bh+(t+1)*sbuf_t,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                tl.store(dst,S,boundary_check=(0,1))

@triton.jit
def generic_infer_fwd(q_ptr,k_ptr,v_ptr,d_ptr,b2_ptr,out_ptr,scratch_ptr,
    BH,T,K,V,
    sqbh,sqt,sqk, skbh,skt,skk, svbh,svt,svk, sdbh,sdt, sbbh,sbt, sobh,sot,sov, scale,
    BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,NUM_K:tl.constexpr,NUM_V:tl.constexpr):
    pid=tl.program_id(0)
    for t in range(T):
        d_t=tl.load(d_ptr+pid*sdbh+t*sdt); b2_t=tl.load(b2_ptr+pid*sbbh+t*sbt)
        for bk in range(NUM_K):
            for bv in range(NUM_V):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)*d_t
                tl.store(sp,S,boundary_check=(0,1))
        for bv in range(NUM_V):
            pred=tl.zeros((BLOCK_V,),dtype=tl.float32)
            for bk in range(NUM_K):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                k_blk=tl.load(k_ptr+pid*skbh+t*skt+offs_k*skk,mask=offs_k<K).to(tl.float32)
                pred+=tl.sum(S*tl.expand_dims(k_blk,1),axis=0)
            offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
            v_blk=tl.load(v_ptr+pid*svbh+t*svt+offs_v*svk,mask=offs_v<V).to(tl.float32)
            delta=(v_blk-pred)*b2_t
            for bk in range(NUM_K):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                k_blk=tl.load(k_ptr+pid*skbh+t*skt+offs_k*skk,mask=offs_k<K).to(tl.float32)
                S_new=S+tl.expand_dims(k_blk,1)*tl.expand_dims(delta,0)
                tl.store(sp,S_new,boundary_check=(0,1))
        for bv in range(NUM_V):
            offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
            out_acc=tl.zeros((BLOCK_V,),dtype=tl.float32)
            for bk in range(NUM_K):
                sp=tl.make_block_ptr(base=scratch_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0))
                S=tl.load(sp,boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                q_blk=tl.load(q_ptr+pid*sqbh+t*sqt+offs_k*sqk,mask=offs_k<K).to(tl.float32)
                out_acc+=tl.sum(S*tl.expand_dims(q_blk,1),axis=0)
            tl.store(out_ptr+pid*sobh+t*sot+offs_v*sov,out_acc*scale,mask=offs_v<V)

@triton.jit
def generic_bwd_kernel(q_ptr,k_ptr,v_ptr,d_ptr,b2_ptr,do_ptr,s_buf_ptr,dq_ptr,dk_ptr,dv_ptr,db_ptr,dg_ptr,h_ptr,
    BH,T,K,V,
    sqbh,sqt,sqk, skbh,skt,skk, svbh,svt,svk, sdbh,sdt, sbbh,sbt, sdobh,sdot,sdov, sbuf_bh,sbuf_t, sdqbh,sdqt,sdqk, sdkbh,sdkt,sdkk, sdvbh,sdvt,sdvk, sdbbh,sdbt, sdgbh,sdgt, scale,
    BLOCK_K:tl.constexpr,BLOCK_V:tl.constexpr,NUM_K:tl.constexpr,NUM_V:tl.constexpr):
    pid=tl.program_id(0)
    for idx in range(T):
        t=T-1-idx
        if idx!=0:
            t_next=t+1
            d_next=tl.load(d_ptr+pid*sdbh+t_next*sdt); b2_next=tl.load(b2_ptr+pid*sbbh+t_next*sbt)
            for bv in range(NUM_V):
                u_next=tl.zeros((BLOCK_V,),dtype=tl.float32)
                for bk in range(NUM_K):
                    hb=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                    offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                    k_blk=tl.load(k_ptr+pid*skbh+t_next*skt+offs_k*skk,mask=offs_k<K).to(tl.float32)
                    u_next+=tl.sum(hb*tl.expand_dims(k_blk,1),axis=0)
                for bk in range(NUM_K):
                    hb=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                    offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                    k_blk=tl.load(k_ptr+pid*skbh+t_next*skt+offs_k*skk,mask=offs_k<K).to(tl.float32)
                    hn=d_next*(hb - tl.expand_dims(k_blk,1)*tl.expand_dims(u_next,0)*b2_next)
                    tl.store(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),hn,boundary_check=(0,1))
        for bk in range(NUM_K):
            for bv in range(NUM_V):
                hb=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
                q_blk=tl.load(q_ptr+pid*sqbh+t*sqt+offs_k*sqk,mask=offs_k<K).to(tl.float32)
                do_blk=tl.load(do_ptr+pid*sdobh+t*sdot+offs_v*sdov,mask=offs_v<V).to(tl.float32)
                hb=hb+tl.expand_dims(q_blk,1)*tl.expand_dims(do_blk,0)*scale
                tl.store(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),hb,boundary_check=(0,1))
        d_t=tl.load(d_ptr+pid*sdbh+t*sdt); b2_t=tl.load(b2_ptr+pid*sbbh+t*sbt)
        fdot=0.0
        for bk in range(NUM_K):
            for bv in range(NUM_V):
                hb=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                sp=tl.load(tl.make_block_ptr(base=s_buf_ptr+pid*sbuf_bh+t*sbuf_t,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                fdot+=tl.sum(hb*sp)
        updot=0.0; db_acc=0.0
        for bv in range(NUM_V):
            u_b=tl.zeros((BLOCK_V,),dtype=tl.float32)
            for bk in range(NUM_K):
                hb=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                k_blk=tl.load(k_ptr+pid*skbh+t*skt+offs_k*skk,mask=offs_k<K).to(tl.float32)
                u_b+=tl.sum(hb*tl.expand_dims(k_blk,1),axis=0)
            p_b=tl.zeros((BLOCK_V,),dtype=tl.float32)
            for bk in range(NUM_K):
                sp=tl.load(tl.make_block_ptr(base=s_buf_ptr+pid*sbuf_bh+t*sbuf_t,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
                k_blk=tl.load(k_ptr+pid*skbh+t*skt+offs_k*skk,mask=offs_k<K).to(tl.float32)
                p_b+=tl.sum(sp*tl.expand_dims(k_blk,1),axis=0)
            updot+=tl.sum(u_b*p_b)
            offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
            v_blk=tl.load(v_ptr+pid*svbh+t*svt+offs_v*svk,mask=offs_v<V).to(tl.float32)
            a_b=v_blk - p_b*d_t
            tl.store(dv_ptr+pid*sdvbh+t*sdvt+offs_v*sdvk,u_b*b2_t,mask=offs_v<V)
            db_acc+=tl.sum(u_b*a_b)
        tl.store(db_ptr+pid*sdbbh+t*sdbt,db_acc*2.0)
        tl.store(dg_ptr+pid*sdgbh+t*sdgt,d_t*(fdot - b2_t*updot))
        for bk in range(NUM_K):
            dq_bk=tl.zeros((BLOCK_K,),dtype=tl.float32)
            hv_bk=tl.zeros((BLOCK_K,),dtype=tl.float32)
            hp_bk=tl.zeros((BLOCK_K,),dtype=tl.float32)
            su_bk=tl.zeros((BLOCK_K,),dtype=tl.float32)
            for bv in range(NUM_V):
                sc=tl.load(tl.make_block_ptr(base=s_buf_ptr+pid*sbuf_bh+(t+1)*sbuf_t,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                offs_v=bv*BLOCK_V+tl.arange(0,BLOCK_V)
                do_blk=tl.load(do_ptr+pid*sdobh+t*sdot+offs_v*sdov,mask=offs_v<V).to(tl.float32)
                dq_bk+=tl.sum(sc*tl.expand_dims(do_blk,0),axis=1)*scale
                hb=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                v_blk=tl.load(v_ptr+pid*svbh+t*svt+offs_v*svk,mask=offs_v<V).to(tl.float32)
                hv_bk+=tl.sum(hb*tl.expand_dims(v_blk,0),axis=1)
                p_b=tl.zeros((BLOCK_V,),dtype=tl.float32)
                for bk2 in range(NUM_K):
                    sp2=tl.load(tl.make_block_ptr(base=s_buf_ptr+pid*sbuf_bh+t*sbuf_t,shape=(K,V),strides=(V,1),offsets=(bk2*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                    offs_k2=bk2*BLOCK_K+tl.arange(0,BLOCK_K)
                    k2=tl.load(k_ptr+pid*skbh+t*skt+offs_k2*skk,mask=offs_k2<K).to(tl.float32)
                    p_b+=tl.sum(sp2*tl.expand_dims(k2,1),axis=0)
                hp_bk+=tl.sum(hb*tl.expand_dims(p_b,0),axis=1)
                u_b=tl.zeros((BLOCK_V,),dtype=tl.float32)
                for bk2 in range(NUM_K):
                    hb2=tl.load(tl.make_block_ptr(base=h_ptr+pid*K*V,shape=(K,V),strides=(V,1),offsets=(bk2*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                    offs_k2=bk2*BLOCK_K+tl.arange(0,BLOCK_K)
                    k2=tl.load(k_ptr+pid*skbh+t*skt+offs_k2*skk,mask=offs_k2<K).to(tl.float32)
                    u_b+=tl.sum(hb2*tl.expand_dims(k2,1),axis=0)
                sp=tl.load(tl.make_block_ptr(base=s_buf_ptr+pid*sbuf_bh+t*sbuf_t,shape=(K,V),strides=(V,1),offsets=(bk*BLOCK_K,bv*BLOCK_V),block_shape=(BLOCK_K,BLOCK_V),order=(1,0)),boundary_check=(0,1)).to(tl.float32)
                su_bk+=tl.sum(sp*tl.expand_dims(u_b,0),axis=1)
            offs_k=bk*BLOCK_K+tl.arange(0,BLOCK_K)
            tl.store(dq_ptr+pid*sdqbh+t*sdqt+offs_k*sdqk,dq_bk,mask=offs_k<K)
            dk_bk=(hv_bk - d_t*hp_bk - d_t*su_bk)*b2_t
            tl.store(dk_ptr+pid*sdkbh+t*sdkt+offs_k*sdkk,dk_bk,mask=offs_k<K)

class RecurrenceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,e,g,beta,gamma,scale):
        B,H,T,K=q.shape; Vdim=v.shape[-1]; BH=B*H
        scl=float(scale.item()) if isinstance(scale,torch.Tensor) else float(scale)
        q_f=q.reshape(BH,T,K).contiguous(); k_f=k.reshape(BH,T,K).contiguous(); v_f=v.reshape(BH,T,Vdim).contiguous()
        g_f=g.reshape(BH,T).contiguous(); beta_f=beta.reshape(BH,T).contiguous()
        d_f=torch.exp(g_f.float()).contiguous(); b2_f=(beta_f.float()*2.0).contiguous()
        out_f=torch.empty(BH,T,Vdim,device=q.device,dtype=torch.float32)
        sbuf_bh=(T+1)*K*Vdim; sbuf_t=K*Vdim
        s_buf=torch.zeros(BH*sbuf_bh,device=q.device,dtype=torch.float32)
        is_fast=(K==128 and Vdim==128)
        if is_fast:
            fast_fwd_128_store[(BH*4,)](q_f,k_f,v_f,d_f,b2_f,out_f,s_buf, BH,T,K,Vdim, q_f.stride(0),q_f.stride(1),q_f.stride(2), k_f.stride(0),k_f.stride(1),k_f.stride(2), v_f.stride(0),v_f.stride(1),v_f.stride(2), d_f.stride(0),d_f.stride(1), b2_f.stride(0),b2_f.stride(1), out_f.stride(0),out_f.stride(1),out_f.stride(2), sbuf_bh,sbuf_t, scl, 32,32,4, num_warps=4)
        else:
            BLOCK_K=32; BLOCK_V=32; NUM_K=(K+31)//32; NUM_V=(Vdim+31)//32
            scratch=torch.zeros(BH*K*Vdim,device=q.device,dtype=torch.float32)
            generic_train_fwd[(BH,)](q_f,k_f,v_f,d_f,b2_f,out_f,s_buf,scratch, BH,T,K,Vdim, q_f.stride(0),q_f.stride(1),q_f.stride(2), k_f.stride(0),k_f.stride(1),k_f.stride(2), v_f.stride(0),v_f.stride(1),v_f.stride(2), d_f.stride(0),d_f.stride(1), b2_f.stride(0),b2_f.stride(1), out_f.stride(0),out_f.stride(1),out_f.stride(2), sbuf_bh,sbuf_t, scl, BLOCK_K,BLOCK_V,NUM_K,NUM_V)
        ctx.save_for_backward(q_f,k_f,v_f,d_f,b2_f,s_buf)
        ctx.shapes=(B,H,T,K,Vdim,BH,scl,sbuf_bh,sbuf_t)
        ctx.is_fast=is_fast
        ctx.q_dtype=q.dtype; ctx.k_dtype=k.dtype; ctx.v_dtype=v.dtype; ctx.e_dtype=e.dtype; ctx.g_dtype=g.dtype; ctx.beta_dtype=beta.dtype; ctx.gamma_dtype=gamma.dtype
        return out_f.reshape(B,H,T,Vdim).to(q.dtype)
    @staticmethod
    def backward(ctx, grad_out):
        q_f,k_f,v_f,d_f,b2_f,s_buf=ctx.saved_tensors
        B,H,T,K,Vdim,BH,scl,sbuf_bh,sbuf_t=ctx.shapes; is_fast=ctx.is_fast
        go=grad_out.reshape(BH,T,Vdim).contiguous().float()
        if is_fast:
            dv=torch.empty(BH,T,Vdim,device=q_f.device,dtype=torch.float32)
            dq_part=torch.zeros(BH*T*4*K,device=q_f.device,dtype=torch.float32)
            dk_part=torch.zeros(BH*T*4*K,device=q_f.device,dtype=torch.float32)
            db_part=torch.zeros(BH*T*4,device=q_f.device,dtype=torch.float32)
            dg_part=torch.zeros(BH*T*4,device=q_f.device,dtype=torch.float32)
            fast_bwd_128[(BH*4,)](q_f,k_f,v_f,d_f,b2_f,go,s_buf,dq_part,dk_part,dv,db_part,dg_part, BH,T,K,Vdim, q_f.stride(0),q_f.stride(1),q_f.stride(2), k_f.stride(0),k_f.stride(1),k_f.stride(2), v_f.stride(0),v_f.stride(1),v_f.stride(2), d_f.stride(0),d_f.stride(1), b2_f.stride(0),b2_f.stride(1), go.stride(0),go.stride(1),go.stride(2), sbuf_bh,sbuf_t, scl, 32,32,4, num_warps=4)
            dq=dq_part.view(BH,T,4,K).sum(dim=2).view(B,H,T,K)
            dk=dk_part.view(BH,T,4,K).sum(dim=2).view(B,H,T,K)
            db=db_part.view(BH,T,4).sum(dim=2).view(B,H,T)
            dg=dg_part.view(BH,T,4).sum(dim=2).view(B,H,T)
            dv=dv.view(B,H,T,Vdim)
        else:
            BLOCK_K=32; BLOCK_V=32; NUM_K=(K+31)//32; NUM_V=(Vdim+31)//32
            dq=torch.zeros(BH,T,K,device=q_f.device,dtype=torch.float32); dk=torch.zeros(BH,T,K,device=q_f.device,dtype=torch.float32)
            dv=torch.zeros(BH,T,Vdim,device=q_f.device,dtype=torch.float32); db=torch.zeros(BH,T,device=q_f.device,dtype=torch.float32); dg=torch.zeros(BH,T,device=q_f.device,dtype=torch.float32)
            h_scratch=torch.zeros(BH*K*Vdim,device=q_f.device,dtype=torch.float32)
            generic_bwd_kernel[(BH,)](q_f,k_f,v_f,d_f,b2_f,go,s_buf,dq,dk,dv,db,dg,h_scratch, BH,T,K,Vdim, q_f.stride(0),q_f.stride(1),q_f.stride(2), k_f.stride(0),k_f.stride(1),k_f.stride(2), v_f.stride(0),v_f.stride(1),v_f.stride(2), d_f.stride(0),d_f.stride(1), b2_f.stride(0),b2_f.stride(1), go.stride(0),go.stride(1),go.stride(2), sbuf_bh,sbuf_t, dq.stride(0),dq.stride(1),dq.stride(2), dk.stride(0),dk.stride(1),dk.stride(2), dv.stride(0),dv.stride(1),dv.stride(2), db.stride(0),db.stride(1), dg.stride(0),dg.stride(1), scl, BLOCK_K,BLOCK_V,NUM_K,NUM_V)
            dq=dq.view(B,H,T,K); dk=dk.view(B,H,T,K); dv=dv.view(B,H,T,Vdim); db=db.view(B,H,T); dg=dg.view(B,H,T)
        grad_q=dq.to(ctx.q_dtype); grad_k=dk.to(ctx.k_dtype); grad_v=dv.to(ctx.v_dtype); grad_e=torch.zeros((B,H,T,K),device=q_f.device,dtype=ctx.e_dtype); grad_g=dg.to(ctx.g_dtype); grad_beta=db.to(ctx.beta_dtype); grad_gamma=torch.zeros((B,H,T),device=q_f.device,dtype=ctx.gamma_dtype)
        return grad_q, grad_k, grad_v, grad_e, grad_g, grad_beta, grad_gamma, None

def kernel_function(q,k,v,e,g,beta,gamma,scale):
    if torch.is_grad_enabled() and any(getattr(x,'requires_grad',False) for x in (q,k,v,e,g,beta,gamma)):
        return RecurrenceFunc.apply(q,k,v,e,g,beta,gamma,scale)
    B,H,T,K=q.shape; Vdim=v.shape[-1]; BH=B*H
    scl=float(scale.item()) if isinstance(scale,torch.Tensor) else float(scale)
    q_f=q.reshape(BH,T,K).contiguous(); k_f=k.reshape(BH,T,K).contiguous(); v_f=v.reshape(BH,T,Vdim).contiguous()
    g_f=g.reshape(BH,T).contiguous(); beta_f=beta.reshape(BH,T).contiguous()
    d_f=torch.exp(g_f.float()).contiguous(); b2_f=(beta_f.float()*2.0).contiguous()
    out_f=torch.empty(BH,T,Vdim,device=q.device,dtype=torch.float32)
    if K==128 and Vdim==128:
        fast_infer_128[(BH*4,)](q_f,k_f,v_f,d_f,b2_f,out_f, BH,T,K,Vdim, q_f.stride(0),q_f.stride(1),q_f.stride(2), k_f.stride(0),k_f.stride(1),k_f.stride(2), v_f.stride(0),v_f.stride(1),v_f.stride(2), d_f.stride(0),d_f.stride(1), b2_f.stride(0),b2_f.stride(1), out_f.stride(0),out_f.stride(1),out_f.stride(2), scl, 32,32,4, num_warps=4)
    else:
        BLOCK_K=32; BLOCK_V=32; NUM_K=(K+31)//32; NUM_V=(Vdim+31)//32
        scratch=torch.zeros(BH*K*Vdim,device=q.device,dtype=torch.float32)
        generic_infer_fwd[(BH,)](q_f,k_f,v_f,d_f,b2_f,out_f,scratch, BH,T,K,Vdim, q_f.stride(0),q_f.stride(1),q_f.stride(2), k_f.stride(0),k_f.stride(1),k_f.stride(2), v_f.stride(0),v_f.stride(1),v_f.stride(2), d_f.stride(0),d_f.stride(1), b2_f.stride(0),b2_f.stride(1), out_f.stride(0),out_f.stride(1),out_f.stride(2), scl, BLOCK_K,BLOCK_V,NUM_K,NUM_V)
    return out_f.reshape(B,H,T,Vdim).to(q.dtype)
```
