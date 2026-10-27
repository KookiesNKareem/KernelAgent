import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_fast_kernel(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, o_ptr,
                     T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     scale, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    k_mask = idx < K
    v_mask = idx < V
    S = tl.zeros((BLOCK, BLOCK), tl.float32)
    q_base = pid * T * K
    k_base = pid * T * K
    e_base = pid * T * K
    d_base = pid * T * K
    v_base = pid * T * V
    o_base = pid * T * V
    b_base = pid * T
    for t in tl.range(T):
        bt = tl.load(beta_ptr + b_base + t)
        ct = tl.load(gamma_ptr + b_base + t)
        if EVEN:
            kt = tl.load(k_ptr + k_base + t * K + idx, eviction_policy='evict_first').to(tl.float32)
            et = tl.load(e_ptr + e_base + t * K + idx, eviction_policy='evict_first').to(tl.float32)
            dt = tl.load(d_ptr + d_base + t * K + idx, eviction_policy='evict_first')
            qt = tl.load(q_ptr + q_base + t * K + idx, eviction_policy='evict_first').to(tl.float32)
            vt = tl.load(v_ptr + v_base + t * V + idx, eviction_policy='evict_first').to(tl.float32)
        else:
            kt = tl.load(k_ptr + k_base + t * K + idx, mask=k_mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
            et = tl.load(e_ptr + e_base + t * K + idx, mask=k_mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
            dt = tl.load(d_ptr + d_base + t * K + idx, mask=k_mask, other=1.0, eviction_policy='evict_first')
            qt = tl.load(q_ptr + q_base + t * K + idx, mask=k_mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
            vt = tl.load(v_ptr + v_base + t * V + idx, mask=v_mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
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


@triton.jit
def _fwd_save_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr, st_ptr,
                     T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    k_mask = idx < K
    v_mask = idx < V
    S = tl.zeros((BLOCK, BLOCK), tl.float32)
    q_base = pid * T * K
    k_base = pid * T * K
    e_base = pid * T * K
    g_base = pid * T * K
    v_base = pid * T * V
    o_base = pid * T * V
    b_base = pid * T
    for t in tl.range(T):
        kt = tl.load(k_ptr + k_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        et = tl.load(e_ptr + e_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        gt = tl.load(g_ptr + g_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        qt = tl.load(q_ptr + q_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        vt = tl.load(v_ptr + v_base + t * V + idx, mask=v_mask, other=0.0).to(tl.float32)
        bt = tl.load(beta_ptr + b_base + t).to(tl.float32)
        ct = tl.load(gamma_ptr + b_base + t).to(tl.float32)
        dt = tl.exp(gt)
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
        tl.store(o_ptr + o_base + t * V + idx, out, mask=v_mask)
        s_base = st_ptr + (pid * T + t) * K * V
        s_ptr = tl.make_block_ptr(base=s_base, shape=(K, V), strides=(V, 1), offsets=(0, 0), block_shape=(BLOCK, BLOCK), order=(1, 0))
        tl.store(s_ptr, S, boundary_check=(0, 1))


@triton.jit
def _bwd_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, do_ptr, st_ptr,
                dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, db_ptr, dc_ptr,
                T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    k_mask = idx < K
    v_mask = idx < V
    Hc = tl.zeros((BLOCK, BLOCK), tl.float32)
    q_base = pid * T * K
    k_base = pid * T * K
    e_base = pid * T * K
    g_base = pid * T * K
    v_base = pid * T * V
    o_base = pid * T * V
    b_base = pid * T
    for i in tl.range(T):
        t = T - 1 - i
        kt = tl.load(k_ptr + k_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        et = tl.load(e_ptr + e_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        gt = tl.load(g_ptr + g_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        qt = tl.load(q_ptr + q_base + t * K + idx, mask=k_mask, other=0.0).to(tl.float32)
        vt = tl.load(v_ptr + v_base + t * V + idx, mask=v_mask, other=0.0).to(tl.float32)
        bt = tl.load(beta_ptr + b_base + t).to(tl.float32)
        ct = tl.load(gamma_ptr + b_base + t).to(tl.float32)
        do = tl.load(do_ptr + o_base + t * V + idx, mask=v_mask, other=0.0).to(tl.float32)
        dt = tl.exp(gt)
        if t == 0:
            S0 = tl.zeros((BLOCK, BLOCK), tl.float32)
        else:
            s_base = st_ptr + (pid * T + t - 1) * K * V
            s_ptr = tl.make_block_ptr(base=s_base, shape=(K, V), strides=(V, 1), offsets=(0, 0), block_shape=(BLOCK, BLOCK), order=(1, 0))
            S0 = tl.load(s_ptr, boundary_check=(0, 1), other=0.0).to(tl.float32)
        qs = qt * scale
        G = Hc + qs[:, None] * do[None, :]
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
        tl.store(dq_ptr + q_base + t * K + idx, dqvec, mask=k_mask)
        tl.store(dk_ptr + k_base + t * K + idx, dkvec, mask=k_mask)
        tl.store(dv_ptr + v_base + t * V + idx, dvvec, mask=v_mask)
        tl.store(de_ptr + e_base + t * K + idx, devec, mask=k_mask)
        tl.store(dg_ptr + g_base + t * K + idx, dgvec, mask=k_mask)
        tl.store(db_ptr + b_base + t, dbt)
        tl.store(dc_ptr + b_base + t, dc)
        hkvec = egvec - bt * ke * uvec
        G = G - bt * kt[:, None] * uvec[None, :]
        G = G - ct * et[:, None] * hkvec[None, :]
        G = G * dt[:, None]
        fGdvec = tl.sum(G * et[None, :], axis=1)
        G = G - ct * fGdvec[:, None] * et[None, :]
        Hc = G


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
        q0 = q.contiguous()
        k0 = k.contiguous()
        v0 = v.contiguous()
        e0 = e.contiguous()
        g0 = g.contiguous()
        b0 = beta.contiguous()
        c0 = gamma.contiguous()
        # NOTE: inside Function.forward the inputs arrive detached, so
        # q.requires_grad (etc.) is not reliable. Gate only on grad mode.
        if torch.is_grad_enabled():
            o = torch.empty((B, H, T, V), dtype=torch.float32, device=q.device)
            states = torch.empty((BH, T, K, V), dtype=torch.float32, device=q.device)
            _fwd_save_kernel[(BH,)](q0, k0, v0, e0, g0, b0, c0, o, states, T, K, V, scale_f, BLOCK, num_warps=8, num_stages=1)
            ctx.save_for_backward(q0, k0, v0, e0, g0, b0, c0, states)
            ctx.meta = (B, H, T, K, V, BLOCK, scale_f, q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
        else:
            d = torch.exp(g0)
            o = torch.empty((B, H, T, V), dtype=torch.float32, device=q.device)
            EVEN = (K == BLOCK and V == BLOCK)
            _fwd_fast_kernel[(BH,)](q0, k0, v0, e0, d, b0, c0, o, T, K, V, scale_f, BLOCK, EVEN, num_warps=8, num_stages=3)
            ctx.save_for_backward()
            ctx.meta = None
        return o

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.meta is None:
            return (None,) * 8
        B, H, T, K, V, BLOCK, scale_f, qd, kd, vd, ed, gd, bd, cd = ctx.meta
        q0, k0, v0, e0, g0, b0, c0, states = ctx.saved_tensors
        go = grad_out.contiguous()
        BH = B * H
        dq = torch.empty((BH, T, K), dtype=torch.float32, device=q0.device)
        dk = torch.empty((BH, T, K), dtype=torch.float32, device=q0.device)
        dv = torch.empty((BH, T, V), dtype=torch.float32, device=q0.device)
        de = torch.empty((BH, T, K), dtype=torch.float32, device=q0.device)
        dg = torch.empty((BH, T, K), dtype=torch.float32, device=q0.device)
        db = torch.empty((BH, T), dtype=torch.float32, device=q0.device)
        dc = torch.empty((BH, T), dtype=torch.float32, device=q0.device)
        _bwd_kernel[(BH,)](q0, k0, v0, e0, g0, b0, c0, go, states, dq, dk, dv, de, dg, db, dc, T, K, V, scale_f, BLOCK, num_warps=8, num_stages=1)
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
