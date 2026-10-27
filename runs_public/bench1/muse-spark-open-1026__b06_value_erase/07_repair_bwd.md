```python
import torch
import triton
import triton.language as tl

@triton.jit
def fwd_kernel(q_ptr,k_ptr,v_ptr,e_ptr,g_ptr,beta_ptr,gamma_ptr,out_ptr,state_ptr,save_ptr,T:tl.constexpr,K:tl.constexpr,V:tl.constexpr,SCALE:tl.constexpr,CHUNK:tl.constexpr):
    pid=tl.program_id(0)
    offs=tl.arange(0,32)
    q_head=q_ptr+pid*T*K; k_head=k_ptr+pid*T*K; e_head=e_ptr+pid*T*K; g_head=g_ptr+pid*T*K
    v_head=v_ptr+pid*T*V; beta_head=beta_ptr+pid*T; gamma_head=gamma_ptr+pid*T
    out_head=out_ptr+pid*T*V; state_head=state_ptr+pid*K*V; save_head=save_ptr+pid*T*K*V
    r0=offs; r1=offs+32; r2=offs+64; r3=offs+96; c0=offs; c1=offs+32; c2=offs+64; c3=offs+96
    s00=tl.load(state_head+r0[:,None]*V+c0[None,:]); s01=tl.load(state_head+r0[:,None]*V+c1[None,:]); s02=tl.load(state_head+r0[:,None]*V+c2[None,:]); s03=tl.load(state_head+r0[:,None]*V+c3[None,:])
    s10=tl.load(state_head+r1[:,None]*V+c0[None,:]); s11=tl.load(state_head+r1[:,None]*V+c1[None,:]); s12=tl.load(state_head+r1[:,None]*V+c2[None,:]); s13=tl.load(state_head+r1[:,None]*V+c3[None,:])
    s20=tl.load(state_head+r2[:,None]*V+c0[None,:]); s21=tl.load(state_head+r2[:,None]*V+c1[None,:]); s22=tl.load(state_head+r2[:,None]*V+c2[None,:]); s23=tl.load(state_head+r2[:,None]*V+c3[None,:])
    s30=tl.load(state_head+r3[:,None]*V+c0[None,:]); s31=tl.load(state_head+r3[:,None]*V+c1[None,:]); s32=tl.load(state_head+r3[:,None]*V+c2[None,:]); s33=tl.load(state_head+r3[:,None]*V+c3[None,:])
    for step in range(32):
        cur_t=CHUNK+step
        q0=tl.load(q_head+cur_t*K+offs).to(tl.float32); q1=tl.load(q_head+cur_t*K+32+offs).to(tl.float32); q2=tl.load(q_head+cur_t*K+64+offs).to(tl.float32); q3=tl.load(q_head+cur_t*K+96+offs).to(tl.float32)
        k0=tl.load(k_head+cur_t*K+offs).to(tl.float32); k1=tl.load(k_head+cur_t*K+32+offs).to(tl.float32); k2=tl.load(k_head+cur_t*K+64+offs).to(tl.float32); k3=tl.load(k_head+cur_t*K+96+offs).to(tl.float32)
        e0=tl.load(e_head+cur_t*K+offs).to(tl.float32); e1=tl.load(e_head+cur_t*K+32+offs).to(tl.float32); e2=tl.load(e_head+cur_t*K+64+offs).to(tl.float32); e3=tl.load(e_head+cur_t*K+96+offs).to(tl.float32)
        g0=tl.load(g_head+cur_t*K+offs); g1=tl.load(g_head+cur_t*K+32+offs); g2=tl.load(g_head+cur_t*K+64+offs); g3=tl.load(g_head+cur_t*K+96+offs)
        v0=tl.load(v_head+cur_t*V+offs).to(tl.float32); v1=tl.load(v_head+cur_t*V+32+offs).to(tl.float32); v2=tl.load(v_head+cur_t*V+64+offs).to(tl.float32); v3=tl.load(v_head+cur_t*V+96+offs).to(tl.float32)
        beta_t=tl.load(beta_head+cur_t); gamma_t=tl.load(gamma_head+cur_t)
        d0=tl.exp(g0); d1=tl.exp(g1); d2=tl.exp(g2); d3=tl.exp(g3)
        s00=s00*d0[:,None]; s01=s01*d0[:,None]; s02=s02*d0[:,None]; s03=s03*d0[:,None]
        s10=s10*d1[:,None]; s11=s11*d1[:,None]; s12=s12*d1[:,None]; s13=s13*d1[:,None]
        s20=s20*d2[:,None]; s21=s21*d2[:,None]; s22=s22*d2[:,None]; s23=s23*d2[:,None]
        s30=s30*d3[:,None]; s31=s31*d3[:,None]; s32=s32*d3[:,None]; s33=s33*d3[:,None]
        h0=tl.sum(s00*e0[None,:],axis=1)+tl.sum(s01*e1[None,:],axis=1)+tl.sum(s02*e2[None,:],axis=1)+tl.sum(s03*e3[None,:],axis=1)
        h1=tl.sum(s10*e0[None,:],axis=1)+tl.sum(s11*e1[None,:],axis=1)+tl.sum(s12*e2[None,:],axis=1)+tl.sum(s13*e3[None,:],axis=1)
        h2=tl.sum(s20*e0[None,:],axis=1)+tl.sum(s21*e1[None,:],axis=1)+tl.sum(s22*e2[None,:],axis=1)+tl.sum(s23*e3[None,:],axis=1)
        h3=tl.sum(s30*e0[None,:],axis=1)+tl.sum(s31*e1[None,:],axis=1)+tl.sum(s32*e2[None,:],axis=1)+tl.sum(s33*e3[None,:],axis=1)
        s00=s00-gamma_t*h0[:,None]*e0[None,:]; s01=s01-gamma_t*h0[:,None]*e1[None,:]; s02=s02-gamma_t*h0[:,None]*e2[None,:]; s03=s03-gamma_t*h0[:,None]*e3[None,:]
        s10=s10-gamma_t*h1[:,None]*e0[None,:]; s11=s11-gamma_t*h1[:,None]*e1[None,:]; s12=s12-gamma_t*h1[:,None]*e2[None,:]; s13=s13-gamma_t*h1[:,None]*e3[None,:]
        s20=s20-gamma_t*h2[:,None]*e0[None,:]; s21=s21-gamma_t*h2[:,None]*e1[None,:]; s22=s22-gamma_t*h2[:,None]*e2[None,:]; s23=s23-gamma_t*h2[:,None]*e3[None,:]
        s30=s30-gamma_t*h3[:,None]*e0[None,:]; s31=s31-gamma_t*h3[:,None]*e1[None,:]; s32=s32-gamma_t*h3[:,None]*e2[None,:]; s33=s33-gamma_t*h3[:,None]*e3[None,:]
        pred0=tl.sum(s00*k0[:,None],axis=0)+tl.sum(s10*k1[:,None],axis=0)+tl.sum(s20*k2[:,None],axis=0)+tl.sum(s30*k3[:,None],axis=0)
        pred1=tl.sum(s01*k0[:,None],axis=0)+tl.sum(s11*k1[:,None],axis=0)+tl.sum(s21*k2[:,None],axis=0)+tl.sum(s31*k3[:,None],axis=0)
        pred2=tl.sum(s02*k0[:,None],axis=0)+tl.sum(s12*k1[:,None],axis=0)+tl.sum(s22*k2[:,None],axis=0)+tl.sum(s32*k3[:,None],axis=0)
        pred3=tl.sum(s03*k0[:,None],axis=0)+tl.sum(s13*k1[:,None],axis=0)+tl.sum(s23*k2[:,None],axis=0)+tl.sum(s33*k3[:,None],axis=0)
        delt0=v0-pred0; delt1=v1-pred1; delt2=v2-pred2; delt3=v3-pred3
        s00=s00+beta_t*k0[:,None]*delt0[None,:]; s01=s01+beta_t*k0[:,None]*delt1[None,:]; s02=s02+beta_t*k0[:,None]*delt2[None,:]; s03=s03+beta_t*k0[:,None]*delt3[None,:]
        s10=s10+beta_t*k1[:,None]*delt0[None,:]; s11=s11+beta_t*k1[:,None]*delt1[None,:]; s12=s12+beta_t*k1[:,None]*delt2[None,:]; s13=s13+beta_t*k1[:,None]*delt3[None,:]
        s20=s20+beta_t*k2[:,None]*delt0[None,:]; s21=s21+beta_t*k2[:,None]*delt1[None,:]; s22=s22+beta_t*k2[:,None]*delt2[None,:]; s23=s23+beta_t*k2[:,None]*delt3[None,:]
        s30=s30+beta_t*k3[:,None]*delt0[None,:]; s31=s31+beta_t*k3[:,None]*delt1[None,:]; s32=s32+beta_t*k3[:,None]*delt2[None,:]; s33=s33+beta_t*k3[:,None]*delt3[None,:]
        o0=tl.sum(s00*q0[:,None],axis=0)+tl.sum(s10*q1[:,None],axis=0)+tl.sum(s20*q2[:,None],axis=0)+tl.sum(s30*q3[:,None],axis=0)
        o1=tl.sum(s01*q0[:,None],axis=0)+tl.sum(s11*q1[:,None],axis=0)+tl.sum(s21*q2[:,None],axis=0)+tl.sum(s31*q3[:,None],axis=0)
        o2=tl.sum(s02*q0[:,None],axis=0)+tl.sum(s12*q1[:,None],axis=0)+tl.sum(s22*q2[:,None],axis=0)+tl.sum(s32*q3[:,None],axis=0)
        o3=tl.sum(s03*q0[:,None],axis=0)+tl.sum(s13*q1[:,None],axis=0)+tl.sum(s23*q2[:,None],axis=0)+tl.sum(s33*q3[:,None],axis=0)
        o0=o0*SCALE; o1=o1*SCALE; o2=o2*SCALE; o3=o3*SCALE
        tl.store(out_head+cur_t*V+offs,o0); tl.store(out_head+cur_t*V+32+offs,o1); tl.store(out_head+cur_t*V+64+offs,o2); tl.store(out_head+cur_t*V+96+offs,o3)
        base=save_head+cur_t*K*V
        tl.store(base+r0[:,None]*V+c0[None,:],s00); tl.store(base+r0[:,None]*V+c1[None,:],s01); tl.store(base+r0[:,None]*V+c2[None,:],s02); tl.store(base+r0[:,None]*V+c3[None,:],s03)
        tl.store(base+r1[:,None]*V+c0[None,:],s10); tl.store(base+r1[:,None]*V+c1[None,:],s11); tl.store(base+r1[:,None]*V+c2[None,:],s12); tl.store(base+r1[:,None]*V+c3[None,:],s13)
        tl.store(base+r2[:,None]*V+c0[None,:],s20); tl.store(base+r2[:,None]*V+c1[None,:],s21); tl.store(base+r2[:,None]*V+c2[None,:],s22); tl.store(base+r2[:,None]*V+c3[None,:],s23)
        tl.store(base+r3[:,None]*V+c0[None,:],s30); tl.store(base+r3[:,None]*V+c1[None,:],s31); tl.store(base+r3[:,None]*V+c2[None,:],s32); tl.store(base+r3[:,None]*V+c3[None,:],s33)
    tl.store(state_head+r0[:,None]*V+c0[None,:],s00); tl.store(state_head+r0[:,None]*V+c1[None,:],s01); tl.store(state_head+r0[:,None]*V+c2[None,:],s02); tl.store(state_head+r0[:,None]*V+c3[None,:],s03)
    tl.store(state_head+r1[:,None]*V+c0[None,:],s10); tl.store(state_head+r1[:,None]*V+c1[None,:],s11); tl.store(state_head+r1[:,None]*V+c2[None,:],s12); tl.store(state_head+r1[:,None]*V+c3[None,:],s13)
    tl.store(state_head+r2[:,None]*V+c0[None,:],s20); tl.store(state_head+r2[:,None]*V+c1[None,:],s21); tl.store(state_head+r2[:,None]*V+c2[None,:],s22); tl.store(state_head+r2[:,None]*V+c3[None,:],s23)
    tl.store(state_head+r3[:,None]*V+c0[None,:],s30); tl.store(state_head+r3[:,None]*V+c1[None,:],s31); tl.store(state_head+r3[:,None]*V+c2[None,:],s32); tl.store(state_head+r3[:,None]*V+c3[None,:],s33)

# backward 16 steps per launch to keep IR small
@triton.jit
def bwd_kernel(q_ptr,k_ptr,v_ptr,e_ptr,g_ptr,beta_ptr,gamma_ptr,go_ptr,save_ptr,dq_ptr,dk_ptr,dv_ptr,de_ptr,dg_ptr,dbeta_ptr,dgamma_ptr,gstate_ptr,T:tl.constexpr,K:tl.constexpr,V:tl.constexpr,SCALE:tl.constexpr,CHUNK:tl.constexpr):
    pid=tl.program_id(0)
    offs=tl.arange(0,32)
    q_head=q_ptr+pid*T*K; k_head=k_ptr+pid*T*K; e_head=e_ptr+pid*T*K; g_head=g_ptr+pid*T*K; v_head=v_ptr+pid*T*V; beta_head=beta_ptr+pid*T; gamma_head=gamma_ptr+pid*T
    go_head=go_ptr+pid*T*V; save_head=save_ptr+pid*T*K*V; dq_head=dq_ptr+pid*T*K; dk_head=dk_ptr+pid*T*K; dv_head=dv_ptr+pid*T*V; de_head=de_ptr+pid*T*K; dg_head=dg_ptr+pid*T*K; dbeta_head=dbeta_ptr+pid*T; dgamma_head=dgamma_ptr+pid*T; gstate_head=gstate_ptr+pid*K*V
    r0=offs; r1=offs+32; r2=offs+64; r3=offs+96; c0=offs; c1=offs+32; c2=offs+64; c3=offs+96
    g00=tl.load(gstate_head+r0[:,None]*V+c0[None,:]); g01=tl.load(gstate_head+r0[:,None]*V+c1[None,:]); g02=tl.load(gstate_head+r0[:,None]*V+c2[None,:]); g03=tl.load(gstate_head+r0[:,None]*V+c3[None,:])
    g10=tl.load(gstate_head+r1[:,None]*V+c0[None,:]); g11=tl.load(gstate_head+r1[:,None]*V+c1[None,:]); g12=tl.load(gstate_head+r1[:,None]*V+c2[None,:]); g13=tl.load(gstate_head+r1[:,None]*V+c3[None,:])
    g20=tl.load(gstate_head+r2[:,None]*V+c0[None,:]); g21=tl.load(gstate_head+r2[:,None]*V+c1[None,:]); g22=tl.load(gstate_head+r2[:,None]*V+c2[None,:]); g23=tl.load(gstate_head+r2[:,None]*V+c3[None,:])
    g30=tl.load(gstate_head+r3[:,None]*V+c0[None,:]); g31=tl.load(gstate_head+r3[:,None]*V+c1[None,:]); g32=tl.load(gstate_head+r3[:,None]*V+c2[None,:]); g33=tl.load(gstate_head+r3[:,None]*V+c3[None,:])
    for step in range(16):
        cur_t=CHUNK+15-step
        q0=tl.load(q_head+cur_t*K+offs).to(tl.float32); q1=tl.load(q_head+cur_t*K+32+offs).to(tl.float32); q2=tl.load(q_head+cur_t*K+64+offs).to(tl.float32); q3=tl.load(q_head+cur_t*K+96+offs).to(tl.float32)
        k0=tl.load(k_head+cur_t*K+offs).to(tl.float32); k1=tl.load(k_head+cur_t*K+32+offs).to(tl.float32); k2=tl.load(k_head+cur_t*K+64+offs).to(tl.float32); k3=tl.load(k_head+cur_t*K+96+offs).to(tl.float32)
        e0=tl.load(e_head+cur_t*K+offs).to(tl.float32); e1=tl.load(e_head+cur_t*K+32+offs).to(tl.float32); e2=tl.load(e_head+cur_t*K+64+offs).to(tl.float32); e3=tl.load(e_head+cur_t*K+96+offs).to(tl.float32)
        gv0=tl.load(g_head+cur_t*K+offs); gv1=tl.load(g_head+cur_t*K+32+offs); gv2=tl.load(g_head+cur_t*K+64+offs); gv3=tl.load(g_head+cur_t*K+96+offs)
        v0=tl.load(v_head+cur_t*V+offs).to(tl.float32); v1=tl.load(v_head+cur_t*V+32+offs).to(tl.float32); v2=tl.load(v_head+cur_t*V+64+offs).to(tl.float32); v3=tl.load(v_head+cur_t*V+96+offs).to(tl.float32)
        beta_t=tl.load(beta_head+cur_t); gamma_t=tl.load(gamma_head+cur_t)
        go0=tl.load(go_head+cur_t*V+offs); go1=tl.load(go_head+cur_t*V+32+offs); go2=tl.load(go_head+cur_t*V+64+offs); go3=tl.load(go_head+cur_t*V+96+offs)
        d0=tl.exp(gv0); d1=tl.exp(gv1); d2=tl.exp(gv2); d3=tl.exp(gv3)
        if cur_t==0:
            s00=tl.zeros((32,32),dtype=tl.float32); s01=tl.zeros((32,32),dtype=tl.float32); s02=tl.zeros((32,32),dtype=tl.float32); s03=tl.zeros((32,32),dtype=tl.float32)
            s10=tl.zeros((32,32),dtype=tl.float32); s11=tl.zeros((32,32),dtype=tl.float32); s12=tl.zeros((32,32),dtype=tl.float32); s13=tl.zeros((32,32),dtype=tl.float32)
            s20=tl.zeros((32,32),dtype=tl.float32); s21=tl.zeros((32,32),dtype=tl.float32); s22=tl.zeros((32,32),dtype=tl.float32); s23=tl.zeros((32,32),dtype=tl.float32)
            s30=tl.zeros((32,32),dtype=tl.float32); s31=tl.zeros((32,32),dtype=tl.float32); s32=tl.zeros((32,32),dtype=tl.float32); s33=tl.zeros((32,32),dtype=tl.float32)
        else:
            prev=save_head+(cur_t-1)*K*V
            s00=tl.load(prev+r0[:,None]*V+c0[None,:]); s01=tl.load(prev+r0[:,None]*V+c1[None,:]); s02=tl.load(prev+r0[:,None]*V+c2[None,:]); s03=tl.load(prev+r0[:,None]*V+c3[None,:])
            s10=tl.load(prev+r1[:,None]*V+c0[None,:]); s11=tl.load(prev+r1[:,None]*V+c1[None,:]); s12=tl.load(prev+r1[:,None]*V+c2[None,:]); s13=tl.load(prev+r1[:,None]*V+c3[None,:])
            s20=tl.load(prev+r2[:,None]*V+c0[None,:]); s21=tl.load(prev+r2[:,None]*V+c1[None,:]); s22=tl.load(prev+r2[:,None]*V+c2[None,:]); s23=tl.load(prev+r2[:,None]*V+c3[None,:])
            s30=tl.load(prev+r3[:,None]*V+c0[None,:]); s31=tl.load(prev+r3[:,None]*V+c1[None,:]); s32=tl.load(prev+r3[:,None]*V+c2[None,:]); s33=tl.load(prev+r3[:,None]*V+c3[None,:])
        s00=s00*d0[:,None]; s01=s01*d0[:,None]; s02=s02*d0[:,None]; s03=s03*d0[:,None]
        s10=s10*d1[:,None]; s11=s11*d1[:,None]; s12=s12*d1[:,None]; s13=s13*d1[:,None]
        s20=s20*d2[:,None]; s21=s21*d2[:,None]; s22=s22*d2[:,None]; s23=s23*d2[:,None]
        s30=s30*d3[:,None]; s31=s31*d3[:,None]; s32=s32*d3[:,None]; s33=s33*d3[:,None]
        h0=tl.sum(s00*e0[None,:],axis=1)+tl.sum(s01*e1[None,:],axis=1)+tl.sum(s02*e2[None,:],axis=1)+tl.sum(s03*e3[None,:],axis=1)
        h1=tl.sum(s10*e0[None,:],axis=1)+tl.sum(s11*e1[None,:],axis=1)+tl.sum(s12*e2[None,:],axis=1)+tl.sum(s13*e3[None,:],axis=1)
        h2=tl.sum(s20*e0[None,:],axis=1)+tl.sum(s21*e1[None,:],axis=1)+tl.sum(s22*e2[None,:],axis=1)+tl.sum(s23*e3[None,:],axis=1)
        h3=tl.sum(s30*e0[None,:],axis=1)+tl.sum(s31*e1[None,:],axis=1)+tl.sum(s32*e2[None,:],axis=1)+tl.sum(s33*e3[None,:],axis=1)
        s00=s00-gamma_t*h0[:,None]*e0[None,:]; s01=s01-gamma_t*h0[:,None]*e1[None,:]; s02=s02-gamma_t*h0[:,None]*e2[None,:]; s03=s03-gamma_t*h0[:,None]*e3[None,:]
        s10=s10-gamma_t*h1[:,None]*e0[None,:]; s11=s11-gamma_t*h1[:,None]*e1[None,:]; s12=s12-gamma_t*h1[:,None]*e2[None,:]; s13=s13-gamma_t*h1[:,None]*e3[None,:]
        s20=s20-gamma_t*h2[:,None]*e0[None,:]; s21=s21-gamma_t*h2[:,None]*e1[None,:]; s22=s22-gamma_t*h2[:,None]*e2[None,:]; s23=s23-gamma_t*h2[:,None]*e3[None,:]
        s30=s30-gamma_t*h3[:,None]*e0[None,:]; s31=s31-gamma_t*h3[:,None]*e1[None,:]; s32=s32-gamma_t*h3[:,None]*e2[None,:]; s33=s33-gamma_t*h3[:,None]*e3[None,:]
        pred0=tl.sum(s00*k0[:,None],axis=0)+tl.sum(s10*k1[:,None],axis=0)+tl.sum(s20*k2[:,None],axis=0)+tl.sum(s30*k3[:,None],axis=0)
        pred1=tl.sum(s01*k0[:,None],axis=0)+tl.sum(s11*k1[:,None],axis=0)+tl.sum(s21*k2[:,None],axis=0)+tl.sum(s31*k3[:,None],axis=0)
        pred2=tl.sum(s02*k0[:,None],axis=0)+tl.sum(s12*k1[:,None],axis=0)+tl.sum(s22*k2[:,None],axis=0)+tl.sum(s32*k3[:,None],axis=0)
        pred3=tl.sum(s03*k0[:,None],axis=0)+tl.sum(s13*k1[:,None],axis=0)+tl.sum(s23*k2[:,None],axis=0)+tl.sum(s33*k3[:,None],axis=0)
        w0=v0-pred0; w1=v1-pred1; w2=v2-pred2; w3=v3-pred3
        g00=g00+SCALE*q0[:,None]*go0[None,:]; g01=g01+SCALE*q0[:,None]*go1[None,:]; g02=g02+SCALE*q0[:,None]*go2[None,:]; g03=g03+SCALE*q0[:,None]*go3[None,:]
        g10=g10+SCALE*q1[:,None]*go0[None,:]; g11=g11+SCALE*q1[:,None]*go1[None,:]; g12=g12+SCALE*q1[:,None]*go2[None,:]; g13=g13+SCALE*q1[:,None]*go3[None,:]
        g20=g20+SCALE*q2[:,None]*go0[None,:]; g21=g21+SCALE*q2[:,None]*go1[None,:]; g22=g22+SCALE*q2[:,None]*go2[None,:]; g23=g23+SCALE*q2[:,None]*go3[None,:]
        g30=g30+SCALE*q3[:,None]*go0[None,:]; g31=g31+SCALE*q3[:,None]*go1[None,:]; g32=g32+SCALE*q3[:,None]*go2[None,:]; g33=g33+SCALE*q3[:,None]*go3[None,:]
        u0=tl.sum(g00*k0[:,None],axis=0)+tl.sum(g10*k1[:,None],axis=0)+tl.sum(g20*k2[:,None],axis=0)+tl.sum(g30*k3[:,None],axis=0)
        u1=tl.sum(g01*k0[:,None],axis=0)+tl.sum(g11*k1[:,None],axis=0)+tl.sum(g21*k2[:,None],axis=0)+tl.sum(g31*k3[:,None],axis=0)
        u2=tl.sum(g02*k0[:,None],axis=0)+tl.sum(g12*k1[:,None],axis=0)+tl.sum(g22*k2[:,None],axis=0)+tl.sum(g32*k3[:,None],axis=0)
        u3=tl.sum(g03*k0[:,None],axis=0)+tl.sum(g13*k1[:,None],axis=0)+tl.sum(g23*k2[:,None],axis=0)+tl.sum(g33*k3[:,None],axis=0)
        dv0=beta_t*u0; dv1=beta_t*u1; dv2=beta_t*u2; dv3=beta_t*u3
        dbeta=tl.sum(u0*w0)+tl.sum(u1*w1)+tl.sum(u2*w2)+tl.sum(u3*w3)
        se_go0=tl.sum(s00*go0[None,:],axis=1)+tl.sum(s01*go1[None,:],axis=1)+tl.sum(s02*go2[None,:],axis=1)+tl.sum(s03*go3[None,:],axis=1)
        se_go1=tl.sum(s10*go0[None,:],axis=1)+tl.sum(s11*go1[None,:],axis=1)+tl.sum(s12*go2[None,:],axis=1)+tl.sum(s13*go3[None,:],axis=1)
        se_go2=tl.sum(s20*go0[None,:],axis=1)+tl.sum(s21*go1[None,:],axis=1)+tl.sum(s22*go2[None,:],axis=1)+tl.sum(s23*go3[None,:],axis=1)
        se_go3=tl.sum(s30*go0[None,:],axis=1)+tl.sum(s31*go1[None,:],axis=1)+tl.sum(s32*go2[None,:],axis=1)+tl.sum(s33*go3[None,:],axis=1)
        dot_w_go=tl.sum(w0*go0)+tl.sum(w1*go1)+tl.sum(w2*go2)+tl.sum(w3*go3)
        dq0=SCALE*(se_go0+beta_t*k0*dot_w_go); dq1=SCALE*(se_go1+beta_t*k1*dot_w_go); dq2=SCALE*(se_go2+beta_t*k2*dot_w_go); dq3=SCALE*(se_go3+beta_t*k3*dot_w_go)
        Gw0=tl.sum(g00*w0[None,:],axis=1)+tl.sum(g01*w1[None,:],axis=1)+tl.sum(g02*w2[None,:],axis=1)+tl.sum(g03*w3[None,:],axis=1)
        Gw1=tl.sum(g10*w0[None,:],axis=1)+tl.sum(g11*w1[None,:],axis=1)+tl.sum(g12*w2[None,:],axis=1)+tl.sum(g13*w3[None,:],axis=1)
        Gw2=tl.sum(g20*w0[None,:],axis=1)+tl.sum(g21*w1[None,:],axis=1)+tl.sum(g22*w2[None,:],axis=1)+tl.sum(g23*w3[None,:],axis=1)
        Gw3=tl.sum(g30*w0[None,:],axis=1)+tl.sum(g31*w1[None,:],axis=1)+tl.sum(g32*w2[None,:],axis=1)+tl.sum(g33*w3[None,:],axis=1)
        Seu0=tl.sum(s00*u0[None,:],axis=1)+tl.sum(s01*u1[None,:],axis=1)+tl.sum(s02*u2[None,:],axis=1)+tl.sum(s03*u3[None,:],axis=1)
        Seu1=tl.sum(s10*u0[None,:],axis=1)+tl.sum(s11*u1[None,:],axis=1)+tl.sum(s12*u2[None,:],axis=1)+tl.sum(s13*u3[None,:],axis=1)
        Seu2=tl.sum(s20*u0[None,:],axis=1)+tl.sum(s21*u1[None,:],axis=1)+tl.sum(s22*u2[None,:],axis=1)+tl.sum(s23*u3[None,:],axis=1)
        Seu3=tl.sum(s30*u0[None,:],axis=1)+tl.sum(s31*u1[None,:],axis=1)+tl.sum(s32*u2[None,:],axis=1)+tl.sum(s33*u3[None,:],axis=1)
        dk0=beta_t*(Gw0-Seu0); dk1=beta_t*(Gw1-Seu1); dk2=beta_t*(Gw2-Seu2); dk3=beta_t*(Gw3-Seu3)
        g00=g00-beta_t*k0[:,None]*u0[None,:]; g01=g01-beta_t*k0[:,None]*u1[None,:]; g02=g02-beta_t*k0[:,None]*u2[None,:]; g03=g03-beta_t*k0[:,None]*u3[None,:]
        g10=g10-beta_t*k1[:,None]*u0[None,:]; g11=g11-beta_t*k1[:,None]*u1[None,:]; g12=g12-beta_t*k1[:,None]*u2[None,:]; g13=g13-beta_t*k1[:,None]*u3[None,:]
        g20=g20-beta_t*k2[:,None]*u0[None,:]; g21=g21-beta_t*k2[:,None]*u1[None,:]; g22=g22-beta_t*k2[:,None]*u2[None,:]; g23=g23-beta_t*k2[:,None]*u3[None,:]
        g30=g30-beta_t*k3[:,None]*u0[None,:]; g31=g31-beta_t*k3[:,None]*u1[None,:]; g32=g32-beta_t*k3[:,None]*u2[None,:]; g33=g33-beta_t*k3[:,None]*u3[None,:]
        ge0=tl.sum(g00*e0[None,:],axis=1)+tl.sum(g01*e1[None,:],axis=1)+tl.sum(g02*e2[None,:],axis=1)+tl.sum(g03*e3[None,:],axis=1)
        ge1=tl.sum(g10*e0[None,:],axis=1)+tl.sum(g11*e1[None,:],axis=1)+tl.sum(g12*e2[None,:],axis=1)+tl.sum(g13*e3[None,:],axis=1)
        ge2=tl.sum(g20*e0[None,:],axis=1)+tl.sum(g21*e1[None,:],axis=1)+tl.sum(g22*e2[None,:],axis=1)+tl.sum(g23*e3[None,:],axis=1)
        ge3=tl.sum(g30*e0[None,:],axis=1)+tl.sum(g31*e1[None,:],axis=1)+tl.sum(g32*e2[None,:],axis=1)+tl.sum(g33*e3[None,:],axis=1)
        dgamma=-(tl.sum(h0*ge0)+tl.sum(h1*ge1)+tl.sum(h2*ge2)+tl.sum(h3*ge3))
        a0=tl.sum(g00*h0[:,None],axis=0)+tl.sum(g10*h1[:,None],axis=0)+tl.sum(g20*h2[:,None],axis=0)+tl.sum(g30*h3[:,None],axis=0)
        a1=tl.sum(g01*h0[:,None],axis=0)+tl.sum(g11*h1[:,None],axis=0)+tl.sum(g21*h2[:,None],axis=0)+tl.sum(g31*h3[:,None],axis=0)
        a2=tl.sum(g02*h0[:,None],axis=0)+tl.sum(g12*h1[:,None],axis=0)+tl.sum(g22*h2[:,None],axis=0)+tl.sum(g32*h3[:,None],axis=0)
        a3=tl.sum(g03*h0[:,None],axis=0)+tl.sum(g13*h1[:,None],axis=0)+tl.sum(g23*h2[:,None],axis=0)+tl.sum(g33*h3[:,None],axis=0)
        c0a=tl.sum(s00*ge0[:,None],axis=0)+tl.sum(s10*ge1[:,None],axis=0)+tl.sum(s20*ge2[:,None],axis=0)+tl.sum(s30*ge3[:,None],axis=0)
        c1a=tl.sum(s01*ge0[:,None],axis=0)+tl.sum(s11*ge1[:,None],axis=0)+tl.sum(s21*ge2[:,None],axis=0)+tl.sum(s31*ge3[:,None],axis=0)
        c2a=tl.sum(s02*ge0[:,None],axis=0)+tl.sum(s12*ge1[:,None],axis=0)+tl.sum(s22*ge2[:,None],axis=0)+tl.sum(s32*ge3[:,None],axis=0)
        c3a=tl.sum(s03*ge0[:,None],axis=0)+tl.sum(s13*ge1[:,None],axis=0)+tl.sum(s23*ge2[:,None],axis=0)+tl.sum(s33*ge3[:,None],axis=0)
        de0=-gamma_t*(a0+c0a); de1=-gamma_t*(a1+c1a); de2=-gamma_t*(a2+c2a); de3=-gamma_t*(a3+c3a)
        g00=g00-gamma_t*ge0[:,None]*e0[None,:]; g01=g01-gamma_t*ge0[:,None]*e1[None,:]; g02=g02-gamma_t*ge0[:,None]*e2[None,:]; g03=g03-gamma_t*ge0[:,None]*e3[None,:]
        g10=g10-gamma_t*ge1[:,None]*e0[None,:]; g11=g11-gamma_t*ge1[:,None]*e1[None,:]; g12=g12-gamma_t*ge1[:,None]*e2[None,:]; g13=g13-gamma_t*ge1[:,None]*e3[None,:]
        g20=g20-gamma_t*ge2[:,None]*e0[None,:]; g21=g21-gamma_t*ge2[:,None]*e1[None,:]; g22=g22-gamma_t*ge2[:,None]*e2[None,:]; g23=g23-gamma_t*ge2[:,None]*e3[None,:]
        g30=g30-gamma_t*ge3[:,None]*e0[None,:]; g31=g31-gamma_t*ge3[:,None]*e1[None,:]; g32=g32-gamma_t*ge3[:,None]*e2[None,:]; g33=g33-gamma_t*ge3[:,None]*e3[None,:]
        gd0=tl.sum(g00*e0[None,:],axis=1)+tl.sum(g01*e1[None,:],axis=1)+tl.sum(g02*e2[None,:],axis=1)+tl.sum(g03*e3[None,:],axis=1)
        gd1=tl.sum(g10*e0[None,:],axis=1)+tl.sum(g11*e1[None,:],axis=1)+tl.sum(g12*e2[None,:],axis=1)+tl.sum(g13*e3[None,:],axis=1)
        gd2=tl.sum(g20*e0[None,:],axis=1)+tl.sum(g21*e1[None,:],axis=1)+tl.sum(g22*e2[None,:],axis=1)+tl.sum(g23*e3[None,:],axis=1)
        gd3=tl.sum(g30*e0[None,:],axis=1)+tl.sum(g31*e1[None,:],axis=1)+tl.sum(g32*e2[None,:],axis=1)+tl.sum(g33*e3[None,:],axis=1)
        dg0=tl.sum(g00*s00,axis=1)+tl.sum(g01*s01,axis=1)+tl.sum(g02*s02,axis=1)+tl.sum(g03*s03,axis=1)+gamma_t*h0*gd0
        dg1=tl.sum(g10*s10,axis=1)+tl.sum(g11*s11,axis=1)+tl.sum(g12*s12,axis=1)+tl.sum(g13*s13,axis=1)+gamma_t*h1*gd1
        dg2=tl.sum(g20*s20,axis=1)+tl.sum(g21*s21,axis=1)+tl.sum(g22*s22,axis=1)+tl.sum(g23*s23,axis=1)+gamma_t*h2*gd2
        dg3=tl.sum(g30*s30,axis=1)+tl.sum(g31*s31,axis=1)+tl.sum(g32*s32,axis=1)+tl.sum(g33*s33,axis=1)+gamma_t*h3*gd3
        tl.store(dq_head+cur_t*K+offs,dq0); tl.store(dq_head+cur_t*K+32+offs,dq1); tl.store(dq_head+cur_t*K+64+offs,dq2); tl.store(dq_head+cur_t*K+96+offs,dq3)
        tl.store(dk_head+cur_t*K+offs,dk0); tl.store(dk_head+cur_t*K+32+offs,dk1); tl.store(dk_head+cur_t*K+64+offs,dk2); tl.store(dk_head+cur_t*K+96+offs,dk3)
        tl.store(dv_head+cur_t*V+offs,dv0); tl.store(dv_head+cur_t*V+32+offs,dv1); tl.store(dv_head+cur_t*V+64+offs,dv2); tl.store(dv_head+cur_t*V+96+offs,dv3)
        tl.store(de_head+cur_t*K+offs,de0); tl.store(de_head+cur_t*K+32+offs,de1); tl.store(de_head+cur_t*K+64+offs,de2); tl.store(de_head+cur_t*K+96+offs,de3)
        tl.store(dg_head+cur_t*K+offs,dg0); tl.store(dg_head+cur_t*K+32+offs,dg1); tl.store(dg_head+cur_t*K+64+offs,dg2); tl.store(dg_head+cur_t*K+96+offs,dg3)
        tl.store(dbeta_head+cur_t,dbeta); tl.store(dgamma_head+cur_t,dgamma)
        g00=g00*d0[:,None]; g01=g01*d0[:,None]; g02=g02*d0[:,None]; g03=g03*d0[:,None]
        g10=g10*d1[:,None]; g11=g11*d1[:,None]; g12=g12*d1[:,None]; g13=g13*d1[:,None]
        g20=g20*d2[:,None]; g21=g21*d2[:,None]; g22=g22*d2[:,None]; g23=g23*d2[:,None]
        g30=g30*d3[:,None]; g31=g31*d3[:,None]; g32=g32*d3[:,None]; g33=g33*d3[:,None]
    tl.store(gstate_head+r0[:,None]*V+c0[None,:],g00); tl.store(gstate_head+r0[:,None]*V+c1[None,:],g01); tl.store(gstate_head+r0[:,None]*V+c2[None,:],g02); tl.store(gstate_head+r0[:,None]*V+c3[None,:],g03)
    tl.store(gstate_head+r1[:,None]*V+c0[None,:],g10); tl.store(gstate_head+r1[:,None]*V+c1[None,:],g11); tl.store(gstate_head+r1[:,None]*V+c2[None,:],g12); tl.store(gstate_head+r1[:,None]*V+c3[None,:],g13)
    tl.store(gstate_head+r2[:,None]*V+c0[None,:],g20); tl.store(gstate_head+r2[:,None]*V+c1[None,:],g21); tl.store(gstate_head+r2[:,None]*V+c2[None,:],g22); tl.store(gstate_head+r2[:,None]*V+c3[None,:],g23)
    tl.store(gstate_head+r3[:,None]*V+c0[None,:],g30); tl.store(gstate_head+r3[:,None]*V+c1[None,:],g31); tl.store(gstate_head+r3[:,None]*V+c2[None,:],g32); tl.store(gstate_head+r3[:,None]*V+c3[None,:],g33)

def kernel_function(q,k,v,e,g,beta,gamma,scale):
    B,H,T,K=q.shape; V=v.shape[-1]
    if isinstance(scale,torch.Tensor): scale_f=float(scale.item() if scale.numel()==1 else float(scale))
    else: scale_f=float(scale)
    device=q.device
    if K!=128 or V!=128 or T%32!=0:
        qf=q.to(torch.float32); kf=k.to(torch.float32); vf=v.to(torch.float32); ef=e.to(torch.float32); gf=g.to(torch.float32); bf=beta.to(torch.float32); gf2=gamma.to(torch.float32)
        S=torch.zeros(B,H,K,V,device=device,dtype=torch.float32); outs=[]
        for t in range(T):
            S=S*torch.exp(gf[:,:,:,t] if False else gf[:,:,t])[...,None] if False else S*torch.exp(gf[:,:,t])[...,None]
            et=ef[:,:,t]; ct=gf2[:,:,t]; S=S-ct[...,None,None]*(S*et[...,None,:]).sum(-1)[...,None]*et[...,None,:]
            kt=kf[:,:,t]; vt=vf[:,:,t]; bt=bf[:,:,t]; pred=(S*kt[...,None]).sum(-2); S=S+(bt[...,None]*(vt-pred))[...,None,:]*kt[...,None]
            outs.append((S*(qf[:,:,t]*scale_f)[...,None]).sum(-2))
        return torch.stack(outs,dim=2)
    BH=B*H
    def prep(x,D):
        y=x.reshape(BH,T,D)
        if not y.is_contiguous(): y=y.contiguous()
        return y
    q_c=prep(q,K); k_c=prep(k,K); v_c=prep(v,V); e_c=prep(e,K); g_c=prep(g,K)
    beta_c=beta.reshape(BH,T); 
    if not beta_c.is_contiguous(): beta_c=beta_c.contiguous()
    gamma_c=gamma.reshape(BH,T)
    if not gamma_c.is_contiguous(): gamma_c=gamma_c.contiguous()
    out_c=torch.empty(BH,T,V,device=device,dtype=torch.float32); state=torch.zeros(BH,K,V,device=device,dtype=torch.float32); save=torch.empty(BH,T,K,V,device=device,dtype=torch.float32)
    n=T//32
    for ck in range(n): fwd_kernel[(BH,)](q_c,k_c,v_c,e_c,g_c,beta_c,gamma_c,out_c,state,save,T,K,V,scale_f,ck*32)
    tail=T-n*32
    if tail:
        S=state
        for t in range(n*32,T):
            gt=g_c[:,t,:].float(); et=e_c[:,t,:].float(); kt=k_c[:,t,:].float(); vt=v_c[:,t,:].float(); qt=q_c[:,t,:].float(); ct=gamma_c[:,t]; bt=beta_c[:,t]
            S=S*torch.exp(gt).unsqueeze(-1); h=(S*et.unsqueeze(1)).sum(-1); S=S-ct[:,None,None]*h.unsqueeze(-1)*et.unsqueeze(1); pred=(S*kt.unsqueeze(-1)).sum(1); S=S+(bt[:,None]*(vt-pred)).unsqueeze(1)*kt.unsqueeze(-1)
            out_c[:,t,:]=(S*(qt*scale_f).unsqueeze(-1)).sum(1); base=save[:,t,:] if False else None; save[:,t]=S
    out=out_c.reshape(B,H,T,V)
    class Func(torch.autograd.Function):
        @staticmethod
        def forward(ctx,q_,k_,v_,e_,g_,beta_,gamma_):
            ctx.save_for_backward(q_c,k_c,v_c,e_c,g_c,beta_c,gamma_c,save); ctx.scale_f=scale_f; ctx.BH=BH; ctx.T=T; ctx.K=K; ctx.V=V; ctx.B=B; ctx.H=H; ctx.qd=q.dtype; ctx.kd=k.dtype; ctx.vd=v.dtype; ctx.ed=e.dtype
            return out
        @staticmethod
        def backward(ctx,grad_out):
            q_c_,k_c_,v_c_,e_c_,g_c_,beta_c_,gamma_c_,save_=ctx.saved_tensors
            go_c=grad_out.reshape(ctx.BH,ctx.T,ctx.V).contiguous()
            dq=torch.empty(ctx.BH,ctx.T,ctx.K,device=device,dtype=torch.float32); dk=torch.empty(ctx.BH,ctx.T,ctx.K,device=device,dtype=torch.float32); dv=torch.empty(ctx.BH,ctx.T,ctx.V,device=device,dtype=torch.float32)
            de=torch.empty(ctx.BH,ctx.T,ctx.K,device=device,dtype=torch.float32); dg=torch.empty(ctx.BH,ctx.T,ctx.K,device=device,dtype=torch.float32); dbeta=torch.empty(ctx.BH,ctx.T,device=device,dtype=torch.float32); dgamma=torch.empty(ctx.BH,ctx.T,device=device,dtype=torch.float32)
            gstate=torch.zeros(ctx.BH,ctx.K,ctx.V,device=device,dtype=torch.float32)
            n2=ctx.T//16
            for ck in range(n2-1,-1,-1): bwd_kernel[(ctx.BH,)](q_c_,k_c_,v_c_,e_c_,g_c_,beta_c_,gamma_c_,go_c,save_,dq,dk,dv,de,dg,dbeta,dgamma,gstate,ctx.T,ctx.K,ctx.V,ctx.scale_f,ck*16)
            tail2=ctx.T-n2*16
            if tail2:
                pass
            return dq.reshape(ctx.B,ctx.H,ctx.T,ctx.K).to(ctx.qd),dk.reshape(ctx.B,ctx.H,ctx.T,ctx.K).to(ctx.kd),dv.reshape(ctx.B,ctx.H,ctx.T,ctx.V).to(ctx.vd),de.reshape(ctx.B,ctx.H,ctx.T,ctx.K).to(ctx.ed),dg.reshape(ctx.B,ctx.H,ctx.T,ctx.K),dbeta.reshape(ctx.B,ctx.H,ctx.T),dgamma.reshape(ctx.B,ctx.H,ctx.T),None
    return Func.apply(q,k,v,e,g,beta,gamma)
```

My approach keeps the proven 32-step forward that tiles the 128×128 state in 16 32×32 registers and saves every S_t for reverse. The backward derivation propagates the adjoint G_t = dL/dS_t via the chain rule for decay, right-erase and delta update; halving the backward chunk to 16 steps per launch cuts the unrolled IR in half and avoids the CompilationError while staying fully in Triton kernels for the backward. Generic shapes fall back to a differentiable PyTorch loop.