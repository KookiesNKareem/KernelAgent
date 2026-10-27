import torch
import triton
import triton.language as tl

CHUNK = 32

# ---------------- forward ----------------
@triton.jit
def _fwd_fast_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, o_ptr,
    T: tl.constexpr, scale,
):
    pid = tl.program_id(0)
    pid_bh = pid // 4
    pid_vb = pid % 4
    offs_k = tl.arange(0, 128)
    offs_v = pid_vb * 32 + tl.arange(0, 32)
    q_base = q_ptr + pid_bh * T * 128
    k_base = k_ptr + pid_bh * T * 128
    e_base = e_ptr + pid_bh * T * 128
    d_base = d_ptr + pid_bh * T * 128
    v_base = v_ptr + pid_bh * T * 128
    o_base = o_ptr + pid_bh * T * 128
    b_base = beta_ptr + pid_bh * T
    c_base = gamma_ptr + pid_bh * T
    S = tl.zeros((128, 32), tl.float32)
    for t in tl.range(T, num_stages=3):
        k_vec = tl.load(k_base + t * 128 + offs_k).to(tl.float32)
        e_vec = tl.load(e_base + t * 128 + offs_k).to(tl.float32)
        d_vec = tl.load(d_base + t * 128 + offs_k).to(tl.float32)
        q_vec = tl.load(q_base + t * 128 + offs_k).to(tl.float32)
        v_vec = tl.load(v_base + t * 128 + offs_v).to(tl.float32)
        bt = tl.load(b_base + t).to(tl.float32)
        ct = tl.load(c_base + t).to(tl.float32)
        S = S * d_vec[:, None]
        pred = tl.sum(S * k_vec[:, None], 0)
        S = S + k_vec[:, None] * (bt * (v_vec - pred))[None, :]
        dot_e = tl.sum(S * e_vec[:, None], 0)
        S = S - (ct * e_vec)[:, None] * dot_e[None, :]
        o_vec = tl.sum(S * (q_vec * scale)[:, None], 0)
        tl.store(o_base + t * 128 + offs_v, o_vec)

@triton.jit
def _fwd_generic_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, o_ptr,
    T: tl.constexpr, K, VD,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, N_VB: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // N_VB
    pid_vb = pid % N_VB
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < VD
    q_seq = q_ptr + pid_bh * T * K
    k_seq = k_ptr + pid_bh * T * K
    e_seq = e_ptr + pid_bh * T * K
    d_seq = d_ptr + pid_bh * T * K
    v_seq = v_ptr + pid_bh * T * VD
    o_seq = o_ptr + pid_bh * T * VD
    beta_seq = beta_ptr + pid_bh * T
    gamma_seq = gamma_ptr + pid_bh * T
    S = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)
    for t in tl.range(T, num_stages=2):
        k_vec = tl.load(k_seq + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        e_vec = tl.load(e_seq + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        d_vec = tl.load(d_seq + t * K + offs_k, mask=mask_k, other=1.0).to(tl.float32)
        q_vec = tl.load(q_seq + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_seq + t * VD + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        bt = tl.load(beta_seq + t).to(tl.float32)
        ct = tl.load(gamma_seq + t).to(tl.float32)
        S = S * d_vec[:, None]
        pred = tl.sum(S * k_vec[:, None], 0)
        S = S + k_vec[:, None] * (bt * (v_vec - pred))[None, :]
        dot_e = tl.sum(S * e_vec[:, None], 0)
        S = S - (ct * e_vec)[:, None] * dot_e[None, :]
        o_vec = tl.sum(S * (q_vec * scale)[:, None], 0)
        tl.store(o_seq + t * VD + offs_v, o_vec, mask=mask_v)

# ---------------- checkpoint / recompute ----------------
@triton.jit
def _ckpt_kernel(
    k_ptr, e_ptr, v_ptr, d_ptr, beta_ptr, gamma_ptr, ckpt_ptr,
    BH, K, VD,
    T: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, N_VB: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // N_VB
    pid_vb = pid % N_VB
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < VD
    k_seq = k_ptr + pid_bh * T * K
    e_seq = e_ptr + pid_bh * T * K
    d_seq = d_ptr + pid_bh * T * K
    v_seq = v_ptr + pid_bh * T * VD
    beta_seq = beta_ptr + pid_bh * T
    gamma_seq = gamma_ptr + pid_bh * T
    S = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)
    for t in tl.range(T, num_stages=2):
        k_vec = tl.load(k_seq + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        e_vec = tl.load(e_seq + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        d_vec = tl.load(d_seq + t * K + offs_k, mask=mask_k, other=1.0).to(tl.float32)
        v_vec = tl.load(v_seq + t * VD + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        bt = tl.load(beta_seq + t).to(tl.float32)
        ct = tl.load(gamma_seq + t).to(tl.float32)
        S = S * d_vec[:, None]
        pred = tl.sum(S * k_vec[:, None], 0)
        S = S + k_vec[:, None] * (bt * (v_vec - pred))[None, :]
        dot_e = tl.sum(S * e_vec[:, None], 0)
        S = S - (ct * e_vec)[:, None] * dot_e[None, :]
        is_bd = (t + 1) % CHUNK == 0
        if is_bd:
            c = (t + 1) // CHUNK
            base = ckpt_ptr + (c * BH + pid_bh) * K * VD
            ptrs = base + offs_k[:, None] * VD + offs_v[None, :]
            m = mask_k[:, None] & mask_v[None, :]
            tl.store(ptrs, S, mask=m)
    # final state -> ckpt[N]
    # N computed on host; pass via extra arg? recompute: N = cdiv(T,CHUNK)
    # store unconditionally to last slot using host-provided index Nlast
    # handled by separate store below via tl.device? we do it in python for remainder
    # instead store here if T%CHUNK !=0 is compile-time; do it:
    # last index:
    # (we pass Nlast as runtime below through ckpt_ptr offset trick: host launches
    #  a tiny kernel? To keep simple, always store final to slot computed as cdiv)
    # compute Nlast at compile time:
    # use python-level constant: TL cannot easily cdiv constexpr? do manual:
    # we just store final again to slot (T+CHUNK-1)//CHUNK
    Nlast = (T + CHUNK - 1) // CHUNK
    # avoid duplicate when divisible (duplicate store is fine)
    base = ckpt_ptr + (Nlast * BH + pid_bh) * K * VD
    ptrs = base + offs_k[:, None] * VD + offs_v[None, :]
    m = mask_k[:, None] & mask_v[None, :]
    # only store if last step wasn't already on boundary OR always (duplicate ok)
    # guard to avoid overwriting when T==0 (S==0, fine)
    tl.store(ptrs, S, mask=m)

@triton.jit
def _chunk_fwd_kernel(
    k_ptr, e_ptr, v_ptr, d_ptr, beta_ptr, gamma_ptr, ckpt_ptr, sbuf_ptr,
    c_start, Lc, BH, K, VD,
    T: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, N_VB: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // N_VB
    pid_vb = pid % N_VB
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < VD
    k_seq = k_ptr + pid_bh * T * K
    e_seq = e_ptr + pid_bh * T * K
    d_seq = d_ptr + pid_bh * T * K
    v_seq = v_ptr + pid_bh * T * VD
    beta_seq = beta_ptr + pid_bh * T
    gamma_seq = gamma_ptr + pid_bh * T
    ckpt_base = ckpt_ptr + (c_start * BH + pid_bh) * K * VD
    ckpt_ptrs = ckpt_base + offs_k[:, None] * VD + offs_v[None, :]
    m2 = mask_k[:, None] & mask_v[None, :]
    S = tl.load(ckpt_ptrs, mask=m2, other=0.0).to(tl.float32)
    for t in range(CHUNK):
        if t < Lc:
            gt = c_start * CHUNK + t
            k_vec = tl.load(k_seq + gt * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            e_vec = tl.load(e_seq + gt * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            d_vec = tl.load(d_seq + gt * K + offs_k, mask=mask_k, other=1.0).to(tl.float32)
            v_vec = tl.load(v_seq + gt * VD + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            bt = tl.load(beta_seq + gt).to(tl.float32)
            ct = tl.load(gamma_seq + gt).to(tl.float32)
            S = S * d_vec[:, None]
            pred = tl.sum(S * k_vec[:, None], 0)
            S = S + k_vec[:, None] * (bt * (v_vec - pred))[None, :]
            dot_e = tl.sum(S * e_vec[:, None], 0)
            S = S - (ct * e_vec)[:, None] * dot_e[None, :]
            sbase = sbuf_ptr + (pid_bh * CHUNK + t) * K * VD
            sptrs = sbase + offs_k[:, None] * VD + offs_v[None, :]
            tl.store(sptrs, S, mask=m2)

@triton.jit
def _chunk_bwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, do_ptr,
    sbuf_ptr, ckpt_ptr, carry_ptr,
    dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
    c_start, Lc, BH, K, VD, scale,
    T: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, N_VB: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // N_VB
    pid_vb = pid % N_VB
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < VD
    m2 = mask_k[:, None] & mask_v[None, :]
    q_seq = q_ptr + pid_bh * T * K
    k_seq = k_ptr + pid_bh * T * K
    e_seq = e_ptr + pid_bh * T * K
    d_seq = d_ptr + pid_bh * T * K
    v_seq = v_ptr + pid_bh * T * VD
    do_seq = do_ptr + pid_bh * T * VD
    beta_seq = beta_ptr + pid_bh * T
    gamma_seq = gamma_ptr + pid_bh * T
    dq_seq = dq_ptr + pid_bh * T * K
    dk_seq = dk_ptr + pid_bh * T * K
    de_seq = de_ptr + pid_bh * T * K
    dg_seq = dg_ptr + pid_bh * T * K
    dv_seq = dv_ptr + pid_bh * T * VD
    db_seq = dbeta_ptr + pid_bh * T
    dc_seq = dgamma_ptr + pid_bh * T
    carry_base = carry_ptr + pid_bh * K * VD
    carry_ptrs = carry_base + offs_k[:, None] * VD + offs_v[None, :]
    carry = tl.load(carry_ptrs, mask=m2, other=0.0).to(tl.float32)
    ckpt_base = ckpt_ptr + (c_start * BH + pid_bh) * K * VD
    ckpt_ptrs = ckpt_base + offs_k[:, None] * VD + offs_v[None, :]
    for t_rev in range(CHUNK):
        t = CHUNK - 1 - t_rev
        if t < Lc:
            gt = c_start * CHUNK + t
            q_vec = tl.load(q_seq + gt * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            k_vec = tl.load(k_seq + gt * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            e_vec = tl.load(e_seq + gt * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            d_vec = tl.load(d_seq + gt * K + offs_k, mask=mask_k, other=1.0).to(tl.float32)
            v_vec = tl.load(v_seq + gt * VD + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            do_vec = tl.load(do_seq + gt * VD + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            bt = tl.load(beta_seq + gt).to(tl.float32)
            ct = tl.load(gamma_seq + gt).to(tl.float32)
            sbase = sbuf_ptr + (pid_bh * CHUNK + t) * K * VD
            sptrs = sbase + offs_k[:, None] * VD + offs_v[None, :]
            St = tl.load(sptrs, mask=m2, other=0.0).to(tl.float32)
            if t == 0:
                Sp = tl.load(ckpt_ptrs, mask=m2, other=0.0).to(tl.float32)
            else:
                pbase = sbuf_ptr + (pid_bh * CHUNK + t - 1) * K * VD
                pptrs = pbase + offs_k[:, None] * VD + offs_v[None, :]
                Sp = tl.load(pptrs, mask=m2, other=0.0).to(tl.float32)
            Y = Sp * d_vec[:, None]
            pred = tl.sum(Y * k_vec[:, None], 0)
            delta = v_vec - pred
            Smid = Y + k_vec[:, None] * (bt * delta)[None, :]
            s_vec = tl.sum(Smid * e_vec[:, None], 0)
            qs = q_vec * scale
            qo = qs[:, None] * do_vec[None, :]
            lam = qo + carry
            a_vec = tl.sum(lam * e_vec[:, None], 0)
            mu = lam - (ct * e_vec)[:, None] * a_vec[None, :]
            b_vec = tl.sum(mu * k_vec[:, None], 0)
            nu = mu - (bt * k_vec)[:, None] * b_vec[None, :]
            # grads
            pdq = scale * tl.sum(St * do_vec[None, :], 1)
            dq_ptrs = dq_seq + gt * K + offs_k
            tl.atomic_add(dq_ptrs, pdq, mask=mask_k)
            dv_vec = bt * b_vec
            tl.store(dv_seq + gt * VD + offs_v, dv_vec, mask=mask_v)
            p_dbeta = tl.sum(b_vec * delta, 0)
            tl.atomic_add(db_seq + gt, p_dbeta)
            p_dgamma = -tl.sum(a_vec * s_vec, 0)
            tl.atomic_add(dc_seq + gt, p_dgamma)
            sum_mu = tl.sum(mu * delta[None, :], 1)
            sum_Y = tl.sum(Y * b_vec[None, :], 1)
            pdk = bt * (sum_mu - sum_Y)
            dk_ptrs = dk_seq + gt * K + offs_k
            tl.atomic_add(dk_ptrs, pdk, mask=mask_k)
            sum_lam = tl.sum(lam * s_vec[None, :], 1)
            sum_mid = tl.sum(Smid * a_vec[None, :], 1)
            pde = -ct * (sum_lam + sum_mid)
            de_ptrs = de_seq + gt * K + offs_k
            tl.atomic_add(de_ptrs, pde, mask=mask_k)
            sum_nu = tl.sum(nu * Sp, 1)
            pdg = d_vec * sum_nu
            dg_ptrs = dg_seq + gt * K + offs_k
            tl.atomic_add(dg_ptrs, pdg, mask=mask_k)
            carry = nu * d_vec[:, None]
    tl.store(carry_ptrs, carry, mask=m2)


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        VD = v.shape[-1]
        scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        if T == 0 or K == 0 or VD == 0:
            out = torch.empty(B, H, T, VD, device=q.device, dtype=torch.float32)
            ctx.empty_case = True
            ctx.shapes = (B, H, T, K, VD)
            ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
            ctx.device = str(q.device)
            return out
        ctx.empty_case = False
        q_c = q.contiguous() if q.is_contiguous() else q.contiguous()
        k_c = k.contiguous() if k.is_contiguous() else k.contiguous()
        v_c = v.contiguous() if v.is_contiguous() else v.contiguous()
        e_c = e.contiguous() if e.is_contiguous() else e.contiguous()
        g_c = g.contiguous() if g.is_contiguous() else g.contiguous()
        beta_c = beta.contiguous() if beta.is_contiguous() else beta.contiguous()
        gamma_c = gamma.contiguous() if gamma.is_contiguous() else gamma.contiguous()
        d_c = torch.exp(g_c).contiguous()
        out = torch.empty(B, H, T, VD, device=q.device, dtype=torch.float32)
        BH = B * H
        if K == 128 and VD == 128:
            grid = (BH * 4,)
            _fwd_fast_kernel[grid](q_c, k_c, v_c, e_c, d_c, beta_c, gamma_c, out, T, scale_f, num_warps=4, num_stages=3)
            BLOCK_K, BLOCK_V, N_VB = 128, 32, 4
        else:
            BLOCK_K = triton.next_power_of_2(K)
            if BLOCK_K < 16:
                BLOCK_K = 16
            tgt = 4096 // BLOCK_K
            if tgt > 64:
                tgt = 64
            if tgt < 16:
                tgt = 16
            BLOCK_V = tgt
            if VD < BLOCK_V:
                BLOCK_V = triton.next_power_of_2(VD)
                if BLOCK_V < 16:
                    BLOCK_V = 16
            N_VB = (VD + BLOCK_V - 1) // BLOCK_V
            grid = (BH * N_VB,)
            _fwd_generic_kernel[grid](q_c, k_c, v_c, e_c, d_c, beta_c, gamma_c, out, T, K, VD, scale_f, BLOCK_K, BLOCK_V, N_VB, num_warps=4, num_stages=2)
        N = (T + CHUNK - 1) // CHUNK
        ckpt = torch.zeros(N + 1, BH, K, VD, device=q.device, dtype=torch.float32)
        if N > 0:
            grid = (BH * N_VB,)
            _ckpt_kernel[grid](k_c, e_c, v_c, d_c, beta_c, gamma_c, ckpt, BH, K, VD, T, CHUNK, BLOCK_K, BLOCK_V, N_VB, num_warps=4, num_stages=2)
        ctx.save_for_backward(q_c, k_c, v_c, e_c, d_c, beta_c, gamma_c, ckpt)
        ctx.shapes = (B, H, T, K, VD, BH, N)
        ctx.blocks = (BLOCK_K, BLOCK_V, N_VB)
        ctx.scale_f = scale_f
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.empty_case:
            B, H, T, K, VD = ctx.shapes
            dev = torch.device(ctx.device)
            qd, kd, vd, ed, gd, bd, cd = ctx.dtypes
            return (torch.empty(B, H, T, K, device=dev, dtype=qd),
                    torch.empty(B, H, T, K, device=dev, dtype=kd),
                    torch.empty(B, H, T, VD, device=dev, dtype=vd),
                    torch.empty(B, H, T, K, device=dev, dtype=ed),
                    torch.empty(B, H, T, K, device=dev, dtype=gd),
                    torch.empty(B, H, T, device=dev, dtype=bd),
                    torch.empty(B, H, T, device=dev, dtype=cd), None)
        q_c, k_c, v_c, e_c, d_c, beta_c, gamma_c, ckpt = ctx.saved_tensors
        B, H, T, K, VD, BH, N = ctx.shapes
        BLOCK_K, BLOCK_V, N_VB = ctx.blocks
        scale_f = ctx.scale_f
        do_c = grad_output.contiguous().to(torch.float32)
        # flatten views: saved are [B,H,T,..], kernels use BH flattening with same layout
        dq = torch.zeros(BH, T, K, device=q_c.device, dtype=torch.float32)
        dk = torch.zeros(BH, T, K, device=q_c.device, dtype=torch.float32)
        dv = torch.zeros(BH, T, VD, device=q_c.device, dtype=torch.float32)
        de = torch.zeros(BH, T, K, device=q_c.device, dtype=torch.float32)
        dg = torch.zeros(BH, T, K, device=q_c.device, dtype=torch.float32)
        db = torch.zeros(BH, T, device=q_c.device, dtype=torch.float32)
        dc = torch.zeros(BH, T, device=q_c.device, dtype=torch.float32)
        carry = torch.zeros(BH, K, VD, device=q_c.device, dtype=torch.float32)
        sbuf = torch.empty(BH, CHUNK, K, VD, device=q_c.device, dtype=torch.float32)
        grid = (BH * N_VB,)
        for c in range(N - 1, -1, -1):
            Lc = T - c * CHUNK
            if Lc > CHUNK:
                Lc = CHUNK
            _chunk_fwd_kernel[grid](k_c, e_c, v_c, d_c, beta_c, gamma_c, ckpt, sbuf, c, Lc, BH, K, VD, T, CHUNK, BLOCK_K, BLOCK_V, N_VB, num_warps=4, num_stages=2)
            _chunk_bwd_kernel[grid](q_c, k_c, v_c, e_c, d_c, beta_c, gamma_c, do_c, sbuf, ckpt, carry, dq, dk, dv, de, dg, db, dc, c, Lc, BH, K, VD, scale_f, T, CHUNK, BLOCK_K, BLOCK_V, N_VB, num_warps=4, num_stages=2)
        qd, kd, vd, ed, gd, bd, cd = ctx.dtypes
        dq_o = dq.view(B, H, T, K).to(qd)
        dk_o = dk.view(B, H, T, K).to(kd)
        dv_o = dv.view(B, H, T, VD).to(vd)
        de_o = de.view(B, H, T, K).to(ed)
        dg_o = dg.view(B, H, T, K).to(gd)
        db_o = db.view(B, H, T).to(bd)
        dc_o = dc.view(B, H, T).to(cd)
        return dq_o, dk_o, dv_o, de_o, dg_o, db_o, dc_o, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _Func.apply(q, k, v, e, g, beta, gamma, scale)
