```python
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_save_kernel(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, o_ptr, st_ptr,
                     T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     scale, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    k_mask = idx < K
    v_mask = idx < V
    r2 = tl.arange(0, BLOCK)
    c2 = tl.arange(0, BLOCK)
    mask2d = (r2[:, None] < K) & (c2[None, :] < V)
    S = tl.zeros((BLOCK, BLOCK), tl.float32)
    q_base = pid * T * K
    k_base = pid * T * K
    e_base = pid * T * K
    d_base = pid * T * K
    v_base = pid * T * V
    o_base = pid * T * V
    b_base = pid * T
    st_base = pid * T * K * V
    for t in tl.range(T):
        if EVEN:
            kt = tl.load(k_ptr + k_base + t * K + idx).to(tl.float32)
            et = tl.load(e_ptr + e_base + t * K + idx).to(tl.float32)
            dt = tl.load(d_ptr + d_base + t * K + idx)
            qt = tl.load(q_ptr + q_base + t * K + idx).to(tl.float32)
            vt = tl.load(v_ptr + v_base + t * V + idx).to(tl.float32)
        else:
            kt = tl.load(k_ptr + k_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
            et = tl.load(e_ptr + e_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
            dt = tl.load(d_ptr + d_base + t * K + idx, mask=k_mask, other=1.0)
            qt = tl.load(q_ptr + q_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
            vt = tl.load(v_ptr + v_base + t * V + idx, mask=v_mask, other=0.0).to(tl.float32)
        bt = tl.load(beta_ptr + b_base + t).to(tl.float32)
        ct = tl.load(gamma_ptr + b_base + t).to(tl.float32)
        S = S * dt[:, None]
        a = tl.sum(S * et[:, None], axis=0)
        S = S - ct * et[:, None] * a[None, :]
        b = tl.sum(S * et[None, :], axis=1)
        S = S - ct * b[:, None] * et[None, :]
        pred = tl.sum(S * kt[:, None], axis=0)
        diff = (vt - pred) * bt
        S = S + kt[:, None] * diff[None, :]
        qs = qt * scale
        out = tl.sum(S * qs[:, None], axis=0)
        if EVEN:
            tl.store(o_ptr + o_base + t * V + idx, out)
        else:
            tl.store(o_ptr + o_base + t * V + idx, out, mask=v_mask)
        s_off = st_base + t * K * V + r2[:, None] * V + c2[None, :]
        if EVEN:
            tl.store(st_ptr + s_off, S)
        else:
            tl.store(st_ptr + s_off, S, mask=mask2d)


@triton.jit
def _bwd_carry_kernel(q_ptr, k_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, do_ptr, gstate_ptr,
                      T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                      scale, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    k_mask = idx < K
    v_mask = idx < V
    r2 = tl.arange(0, BLOCK)
    c2 = tl.arange(0, BLOCK)
    mask2d = (r2[:, None] < K) & (c2[None, :] < V)
    Hc = tl.zeros((BLOCK, BLOCK), tl.float32)
    q_base = pid * T * K
    k_base = pid * T * K
    e_base = pid * T * K
    d_base = pid * T * K
    o_base = pid * T * V
    b_base = pid * T
    gs_base = pid * T * K * V
    for i in tl.range(T):
        t = T - 1 - i
        if EVEN:
            kt = tl.load(k_ptr + k_base + t * K + idx).to(tl.float32)
            et = tl.load(e_ptr + e_base + t * K + idx).to(tl.float32)
            dt = tl.load(d_ptr + d_base + t * K + idx)
            qt = tl.load(q_ptr + q_base + t * K + idx).to(tl.float32)
            do = tl.load(do_ptr + o_base + t * V + idx).to(tl.float32)
        else:
            kt = tl.load(k_ptr + k_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
            et = tl.load(e_ptr + e_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
            dt = tl.load(d_ptr + d_base + t * K + idx, mask=k_mask, other=1.0)
            qt = tl.load(q_ptr + q_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
            do = tl.load(do_ptr + o_base + t * V + idx, mask=v_mask, other=0.0).to(tl.float32)
        bt = tl.load(beta_ptr + b_base + t).to(tl.float32)
        ct = tl.load(gamma_ptr + b_base + t).to(tl.float32)
        qs = qt * scale
        G = Hc + qs[:, None] * do[None, :]
        s_off = gs_base + t * K * V + r2[:, None] * V + c2[None, :]
        if EVEN:
            tl.store(gstate_ptr + s_off, G)
        else:
            tl.store(gstate_ptr + s_off, G, mask=mask2d)
        uvec = tl.sum(G * kt[:, None], axis=0)
        egvec = tl.sum(G * et[:, None], axis=0)
        ke = tl.sum(kt * et)
        hk = egvec - bt * ke * uvec
        G = G - bt * kt[:, None] * uvec[None, :]
        G = G - ct * et[:, None] * hk[None, :]
        G = G * dt[:, None]
        fvec = tl.sum(G * et[None, :], axis=1)
        G = G - ct * fvec[:, None] * et[None, :]
        Hc = G


@triton.jit
def _bwd_grad_kernel(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, do_ptr, st_ptr, gs_ptr,
                     dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, db_ptr, dc_ptr,
                     T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     scale, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    bh = tl.program_id(0)
    t = tl.program_id(1)
    idx = tl.arange(0, BLOCK)
    k_mask = idx < K
    v_mask = idx < V
    r2 = tl.arange(0, BLOCK)
    c2 = tl.arange(0, BLOCK)
    mask2d = (r2[:, None] < K) & (c2[None, :] < V)
    q_base = bh * T * K
    k_base = bh * T * K
    e_base = bh * T * K
    d_base = bh * T * K
    v_base = bh * T * V
    o_base = bh * T * V
    b_base = bh * T
    st_base = bh * T * K * V
    if EVEN:
        kt = tl.load(k_ptr + k_base + t * K + idx).to(tl.float32)
        et = tl.load(e_ptr + e_base + t * K + idx).to(tl.float32)
        dt = tl.load(d_ptr + d_base + t * K + idx)
        qt = tl.load(q_ptr + q_base + t * K + idx).to(tl.float32)
        vt = tl.load(v_ptr + v_base + t * V + idx).to(tl.float32)
        do = tl.load(do_ptr + o_base + t * V + idx).to(tl.float32)
    else:
        kt = tl.load(k_ptr + k_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        et = tl.load(e_ptr + e_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        dt = tl.load(d_ptr + d_base + t * K + idx, mask=k_mask, other=1.0)
        qt = tl.load(q_ptr + q_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        vt = tl.load(v_ptr + v_base + t * V + idx, mask=v_mask, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + o_base + t * V + idx, mask=v_mask, other=0.0).to(tl.float32)
    bt = tl.load(beta_ptr + b_base + t).to(tl.float32)
    ct = tl.load(gamma_ptr + b_base + t).to(tl.float32)
    g_off = st_base + t * K * V + r2[:, None] * V + c2[None, :]
    if EVEN:
        G = tl.load(gs_ptr + g_off).to(tl.float32)
    else:
        G = tl.load(gs_ptr + g_off, mask=mask2d, other=0.0).to(tl.float32)
    if t == 0:
        S0 = tl.zeros((BLOCK, BLOCK), tl.float32)
    else:
        s_off = st_base + (t - 1) * K * V + r2[:, None] * V + c2[None, :]
        if EVEN:
            S0 = tl.load(st_ptr + s_off).to(tl.float32)
        else:
            S0 = tl.load(st_ptr + s_off, mask=mask2d, other=0.0).to(tl.float32)
    s0e = tl.sum(S0 * et[None, :], axis=1)
    ded = dt * et
    avec = tl.sum(S0 * ded[:, None], axis=0)
    ae = tl.sum(avec * et)
    s1e = dt * s0e
    bvec = s1e - ct * et * ae
    kd = kt * dt
    kS1 = tl.sum(S0 * kd[:, None], axis=0)
    ke = tl.sum(kt * et)
    kS2 = kS1 - ct * ke * avec
    kb = tl.sum(kt * bvec)
    pvec = kS2 - ct * kb * et
    vpvec = vt - pvec
    uvec = tl.sum(G * kt[:, None], axis=0)
    gevec = tl.sum(G * et[None, :], axis=1)
    egvec = tl.sum(G * et[:, None], axis=0)
    ue = tl.sum(uvec * et)
    fvec = gevec - bt * kt * ue
    ef = tl.sum(et * fvec)
    hvec = egvec - bt * ke * uvec - ct * ef * et
    ah = tl.sum(avec * hvec)
    bf = tl.sum(bvec * fvec)
    dc = -(ah + bf)
    Gavec = tl.sum(G * avec[None, :], axis=1)
    ua = tl.sum(uvec * avec)
    G2avec = Gavec - bt * kt * ua - ct * fvec * ae
    S0hvec = tl.sum(S0 * hvec[None, :], axis=1)
    S1hvec = dt * S0hvec
    de_left = -ct * (G2avec + S1hvec)
    dfvec = dt * fvec
    S1Tfvec = tl.sum(S0 * dfvec[:, None], axis=0)
    S2Tfvec = S1Tfvec - ct * avec * ef
    Gtbvec = tl.sum(G * bvec[:, None], axis=0)
    G3Tbvec = Gtbvec - bt * uvec * kb
    de_right = -ct * (S2Tfvec + G3Tbvec)
    devec = de_left + de_right
    S0uvec = tl.sum(S0 * uvec[None, :], axis=1)
    S1uvec = dt * S0uvec
    S2uvec = S1uvec - ct * et * ua
    S3uvec = S2uvec - ct * bvec * ue
    Gvpvec = tl.sum(G * vpvec[None, :], axis=1)
    dkvec = bt * (Gvpvec - S3uvec)
    S0dovec = tl.sum(S0 * do[None, :], axis=1)
    S1dovec = dt * S0dovec
    ado = tl.sum(avec * do)
    S2dovec = S1dovec - ct * et * ado
    edo = tl.sum(et * do)
    S3dovec = S2dovec - ct * bvec * edo
    diffvec = bt * vpvec
    diffdo = tl.sum(diffvec * do)
    dqvec = scale * (S3dovec + kt * diffdo)
    dvvec = bt * uvec
    dbt = tl.sum(uvec * vpvec)
    rGvec = tl.sum(G * S0, axis=1)
    rowdot = rGvec - bt * kt * S0uvec - ct * fvec * s0e - ct * et * S0hvec
    dgvec = dt * rowdot
    if EVEN:
        tl.store(dq_ptr + q_base + t * K + idx, dqvec)
        tl.store(dk_ptr + k_base + t * K + idx, dkvec)
        tl.store(dv_ptr + v_base + t * V + idx, dvvec)
        tl.store(de_ptr + e_base + t * K + idx, devec)
        tl.store(dg_ptr + d_base + t * K + idx, dgvec)
    else:
        tl.store(dq_ptr + q_base + t * K + idx, dqvec, mask=k_mask)
        tl.store(dk_ptr + k_base + t * K + idx, dkvec, mask=k_mask)
        tl.store(dv_ptr + v_base + t * V + idx, dvvec, mask=v_mask)
        tl.store(de_ptr + e_base + t * K + idx, devec, mask=k_mask)
        tl.store(dg_ptr + d_base + t * K + idx, dgvec, mask=k_mask)
    tl.store(db_ptr + b_base + t, dbt)
    tl.store(dc_ptr + b_base + t, dc)


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale_f):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        m = K if K > V else V
        BLOCK = 1 << (m - 1).bit_length()
        if BLOCK < 16:
            BLOCK = 16
        EVEN = (K == BLOCK and V == BLOCK)
        q0 = q.contiguous()
        k0 = k.contiguous()
        v0 = v.contiguous()
        e0 = e.contiguous()
        g0 = g.contiguous()
        b0 = beta.contiguous()
        c0 = gamma.contiguous()
        d0 = torch.exp(g0)
        o = torch.empty((B, H, T, V), dtype=torch.float32, device=q.device)
        states = torch.empty((BH, T, K, V), dtype=torch.float32, device=q.device)
        _fwd_save_kernel[(BH,)](q0, k0, v0, e0, d0, b0, c0, o, states,
                                T, K, V, scale_f, BLOCK, EVEN,
                                num_warps=8, num_stages=1)
        ctx.save_for_backward(q0, k0, v0, e0, d0, b0, c0, states)
        ctx.meta = (B, H, T, K, V, BLOCK, EVEN, scale_f,
                    q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
        return o

    @staticmethod
    def backward(ctx, grad_out):
        B, H, T, K, V, BLOCK, EVEN, scale_f, qd, kd, vd, ed, gd, bd, cd = ctx.meta
        q0, k0, v0, e0, d0, b0, c0, states = ctx.saved_tensors
        go = grad_out.contiguous()
        BH = B * H
        gstates = torch.empty_like(states)
        _bwd_carry_kernel[(BH,)](q0, k0, e0, d0, b0, c0, go, gstates,
                                 T, K, V, scale_f, BLOCK, EVEN,
                                 num_warps=8, num_stages=1)
        dq = torch.empty((BH, T, K), dtype=torch.float32, device=q0.device)
        dk = torch.empty((BH, T, K), dtype=torch.float32, device=q0.device)
        dv = torch.empty((BH, T, V), dtype=torch.float32, device=q0.device)
        de = torch.empty((BH, T, K), dtype=torch.float32, device=q0.device)
        dg = torch.empty((BH, T, K), dtype=torch.float32, device=q0.device)
        db = torch.empty((BH, T), dtype=torch.float32, device=q0.device)
        dc = torch.empty((BH, T), dtype=torch.float32, device=q0.device)
        _bwd_grad_kernel[(BH, T)](q0, k0, v0, e0, d0, b0, c0, go, states, gstates,
                                  dq, dk, dv, de, dg, db, dc,
                                  T, K, V, scale_f, BLOCK, EVEN,
                                  num_warps=8, num_stages=1)
        dq = dq.reshape(B, H, T, K).to(qd)
        dk = dk.reshape(B, H, T, K).to(kd)
        dv = dv.reshape(B, H, T, V).to(vd)
        de = de.reshape(B, H, T, K).to(ed)
        dg = dg.reshape(B, H, T, K).to(gd)
        db = db.reshape(B, H, T).to(bd)
        dc = dc.reshape(B, H, T).to(cd)
        return (dq, dk, dv, de, dg, db, dc, None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    scale_f = float(scale.item()) if torch.is_tensor(scale) else float(scale)
    return _Func.apply(q, k, v, e, g, beta, gamma, scale_f)
```