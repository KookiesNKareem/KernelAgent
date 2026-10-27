import torch
import triton
import triton.language as tl

CHUNK_FAST = 16

# ---------- fast fused forward (K=V=128, V-sharded x4) ----------
@triton.jit
def _fwd_ckpt_fast(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, o_ptr, ckpt_ptr,
                   T: tl.constexpr, CHUNK: tl.constexpr, BH, scale):
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
        delta = v_vec - pred
        S = S + k_vec[:, None] * (bt * delta)[None, :]
        s_vec = tl.sum(S * e_vec[:, None], 0)
        S = S - (ct * e_vec)[:, None] * s_vec[None, :]
        o_vec = tl.sum(S * (q_vec * scale)[:, None], 0)
        tl.store(o_base + t * 128 + offs_v, o_vec)
        if (t + 1) % CHUNK == 0:
            c = (t + 1) // CHUNK
            ckpt_base = ckpt_ptr + (c * BH + pid_bh) * 16384
            ptrs = ckpt_base + offs_k[:, None] * 128 + offs_v[None, :]
            tl.store(ptrs, S)

# ---------- fast backward: single launch, recompute Sp from ckpt ----------
@triton.jit
def _bwd_fast(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, do_ptr, ckpt_ptr,
              dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, db_ptr, dc_ptr,
              T: tl.constexpr, CHUNK: tl.constexpr, N: tl.constexpr, BH, scale):
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
    do_base = do_ptr + pid_bh * T * 128
    b_base = beta_ptr + pid_bh * T
    c_base = gamma_ptr + pid_bh * T
    dq_base = dq_ptr + pid_bh * T * 128
    dk_base = dk_ptr + pid_bh * T * 128
    de_base = de_ptr + pid_bh * T * 128
    dg_base = dg_ptr + pid_bh * T * 128
    dv_base = dv_ptr + pid_bh * T * 128
    db_base = db_ptr + pid_bh * T
    dc_base = dc_ptr + pid_bh * T
    carry = tl.zeros((128, 32), tl.float32)
    for cc in tl.range(N, num_stages=1):
        c = N - 1 - cc
        ckpt_base0 = ckpt_ptr + (c * BH + pid_bh) * 16384
        S_start = tl.load(ckpt_base0 + offs_k[:, None] * 128 + offs_v[None, :]).to(tl.float32)
        for t_rev in tl.range(CHUNK, num_stages=1):
            t = CHUNK - 1 - t_rev
            gt = c * CHUNK + t
            S = S_start
            for jj in tl.range(CHUNK, num_stages=1):
                if jj < t:
                    gj = c * CHUNK + jj
                    k_j = tl.load(k_base + gj * 128 + offs_k).to(tl.float32)
                    e_j = tl.load(e_base + gj * 128 + offs_k).to(tl.float32)
                    d_j = tl.load(d_base + gj * 128 + offs_k).to(tl.float32)
                    v_j = tl.load(v_base + gj * 128 + offs_v).to(tl.float32)
                    b_j = tl.load(b_base + gj).to(tl.float32)
                    c_j = tl.load(c_base + gj).to(tl.float32)
                    S = S * d_j[:, None]
                    pj = tl.sum(S * k_j[:, None], 0)
                    dj = v_j - pj
                    S = S + k_j[:, None] * (b_j * dj)[None, :]
                    sj = tl.sum(S * e_j[:, None], 0)
                    S = S - (c_j * e_j)[:, None] * sj[None, :]
            Sp = S
            k_vec = tl.load(k_base + gt * 128 + offs_k).to(tl.float32)
            e_vec = tl.load(e_base + gt * 128 + offs_k).to(tl.float32)
            d_vec = tl.load(d_base + gt * 128 + offs_k).to(tl.float32)
            v_vec = tl.load(v_base + gt * 128 + offs_v).to(tl.float32)
            q_vec = tl.load(q_base + gt * 128 + offs_k).to(tl.float32)
            do_vec = tl.load(do_base + gt * 128 + offs_v).to(tl.float32)
            bt = tl.load(b_base + gt).to(tl.float32)
            ct = tl.load(c_base + gt).to(tl.float32)
            Y = Sp * d_vec[:, None]
            pred = tl.sum(Y * k_vec[:, None], 0)
            delta = v_vec - pred
            Smid = Y + k_vec[:, None] * (bt * delta)[None, :]
            s_vec = tl.sum(Smid * e_vec[:, None], 0)
            St = Smid - (ct * e_vec)[:, None] * s_vec[None, :]
            qs = q_vec * scale
            lam = qs[:, None] * do_vec[None, :] + carry
            a_vec = tl.sum(lam * e_vec[:, None], 0)
            mu = lam - (ct * e_vec)[:, None] * a_vec[None, :]
            b_vec = tl.sum(mu * k_vec[:, None], 0)
            nu = mu - (bt * k_vec)[:, None] * b_vec[None, :]
            pdq = scale * tl.sum(St * do_vec[None, :], 1)
            tl.atomic_add(dq_base + gt * 128 + offs_k, pdq)
            tl.store(dv_base + gt * 128 + offs_v, bt * b_vec)
            tl.atomic_add(db_base + gt, tl.sum(b_vec * delta, 0))
            tl.atomic_add(dc_base + gt, -tl.sum(a_vec * s_vec, 0))
            pdk = bt * (tl.sum(mu * delta[None, :], 1) - tl.sum(Y * b_vec[None, :], 1))
            tl.atomic_add(dk_base + gt * 128 + offs_k, pdk)
            pde = -ct * (tl.sum(lam * s_vec[None, :], 1) + tl.sum(Smid * a_vec[None, :], 1))
            tl.atomic_add(de_base + gt * 128 + offs_k, pde)
            tl.atomic_add(dg_base + gt * 128 + offs_k, d_vec * tl.sum(nu * Sp, 1))
            carry = nu * d_vec[:, None]

# ---------- generic fallback (correctness for other shapes) ----------
@triton.jit
def _fwd_generic_kernel(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, o_ptr,
                        T: tl.constexpr, K, VD, scale,
                        BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, N_VB: tl.constexpr):
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

@triton.jit
def _ckpt_kernel(k_ptr, e_ptr, v_ptr, d_ptr, beta_ptr, gamma_ptr, ckpt_ptr,
                 BH, K, VD, T: tl.constexpr, CHUNK: tl.constexpr,
                 BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, N_VB: tl.constexpr):
    pid = tl.program_id(0)
    pid_bh = pid // N_VB
    pid_vb = pid % N_VB
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < VD
    m2 = mask_k[:, None] & mask_v[None, :]
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
        if (t + 1) % CHUNK == 0:
            c = (t + 1) // CHUNK
            base = ckpt_ptr + (c * BH + pid_bh) * K * VD
            tl.store(base + offs_k[:, None] * VD + offs_v[None, :], S, mask=m2)
    Nlast = (T + CHUNK - 1) // CHUNK
    base = ckpt_ptr + (Nlast * BH + pid_bh) * K * VD
    tl.store(base + offs_k[:, None] * VD + offs_v[None, :], S, mask=m2)

@triton.jit
def _chunk_fwd_kernel(k_ptr, e_ptr, v_ptr, d_ptr, beta_ptr, gamma_ptr, ckpt_ptr, sbuf_ptr,
                      c_start, Lc, BH, K, VD, T: tl.constexpr, CHUNK: tl.constexpr,
                      BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, N_VB: tl.constexpr):
    pid = tl.program_id(0)
    pid_bh = pid // N_VB
    pid_vb = pid % N_VB
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < VD
    m2 = mask_k[:, None] & mask_v[None, :]
    k_seq = k_ptr + pid_bh * T * K
    e_seq = e_ptr + pid_bh * T * K
    d_seq = d_ptr + pid_bh * T * K
    v_seq = v_ptr + pid_bh * T * VD
    beta_seq = beta_ptr + pid_bh * T
    gamma_seq = gamma_ptr + pid_bh * T
    ckpt_base = ckpt_ptr + (c_start * BH + pid_bh) * K * VD
    S = tl.load(ckpt_base + offs_k[:, None] * VD + offs_v[None, :], mask=m2, other=0.0).to(tl.float32)
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
            tl.store(sbase + offs_k[:, None] * VD + offs_v[None, :], S, mask=m2)

@triton.jit
def _chunk_bwd_kernel(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, do_ptr,
                      sbuf_ptr, ckpt_ptr, carry_ptr,
                      dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, db_ptr, dc_ptr,
                      c_start, Lc, BH, K, VD, scale,
                      T: tl.constexpr, CHUNK: tl.constexpr,
                      BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, N_VB: tl.constexpr):
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
    db_seq = db_ptr + pid_bh * T
    dc_seq = dc_ptr + pid_bh * T
    carry_base = carry_ptr + pid_bh * K * VD
    carry = tl.load(carry_base + offs_k[:, None] * VD + offs_v[None, :], mask=m2, other=0.0).to(tl.float32)
    ckpt_base = ckpt_ptr + (c_start * BH + pid_bh) * K * VD
    ckpt_S = tl.load(ckpt_base + offs_k[:, None] * VD + offs_v[None, :], mask=m2, other=0.0).to(tl.float32)
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
            St = tl.load(sbase + offs_k[:, None] * VD + offs_v[None, :], mask=m2, other=0.0).to(tl.float32)
            if t == 0:
                Sp = ckpt_S
            else:
                pbase = sbuf_ptr + (pid_bh * CHUNK + t - 1) * K * VD
                Sp = tl.load(pbase + offs_k[:, None] * VD + offs_v[None, :], mask=m2, other=0.0).to(tl.float32)
            Y = Sp * d_vec[:, None]
            pred = tl.sum(Y * k_vec[:, None], 0)
            delta = v_vec - pred
            Smid = Y + k_vec[:, None] * (bt * delta)[None, :]
            s_vec = tl.sum(Smid * e_vec[:, None], 0)
            qs = q_vec * scale
            lam = qs[:, None] * do_vec[None, :] + carry
            a_vec = tl.sum(lam * e_vec[:, None], 0)
            mu = lam - (ct * e_vec)[:, None] * a_vec[None, :]
            b_vec = tl.sum(mu * k_vec[:, None], 0)
            nu = mu - (bt * k_vec)[:, None] * b_vec[None, :]
            pdq = scale * tl.sum(St * do_vec[None, :], 1)
            tl.atomic_add(dq_seq + gt * K + offs_k, pdq, mask=mask_k)
            tl.store(dv_seq + gt * VD + offs_v, (bt * b_vec).to(tl.float32), mask=mask_v)
            tl.atomic_add(db_seq + gt, tl.sum(b_vec * delta, 0))
            tl.atomic_add(dc_seq + gt, -tl.sum(a_vec * s_vec, 0))
            tl.atomic_add(dk_seq + gt * K + offs_k, bt * (tl.sum(mu * delta[None, :], 1) - tl.sum(Y * b_vec[None, :], 1)), mask=mask_k)
            tl.atomic_add(de_seq + gt * K + offs_k, -ct * (tl.sum(lam * s_vec[None, :], 1) + tl.sum(Smid * a_vec[None, :], 1)), mask=mask_k)
            tl.atomic_add(dg_seq + gt * K + offs_k, d_vec * tl.sum(nu * Sp, 1), mask=mask_k)
            carry = nu * d_vec[:, None]
    tl.store(carry_base + offs_k[:, None] * VD + offs_v[None, :], carry, mask=m2)


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
        q_c = q.contiguous() if not q.is_contiguous() else q
        k_c = k.contiguous() if not k.is_contiguous() else k
        v_c = v.contiguous() if not v.is_contiguous() else v
        e_c = e.contiguous() if not e.is_contiguous() else e
        g_c = g.contiguous() if not g.is_contiguous() else g
        beta_c = beta.contiguous() if not beta.is_contiguous() else beta
        gamma_c = gamma.contiguous() if not gamma.is_contiguous() else gamma
        d_c = torch.exp(g_c).contiguous()
        BH = B * H
        fast = (K == 128 and VD == 128 and T % CHUNK_FAST == 0)
        out = torch.empty(B, H, T, VD, device=q.device, dtype=torch.float32)
        if fast:
            N = T // CHUNK_FAST
            ckpt = torch.zeros(N + 1, BH, K, VD, device=q.device, dtype=torch.float32)
            grid = (BH * 4,)
            _fwd_ckpt_fast[grid](q_c, k_c, v_c, e_c, d_c, beta_c, gamma_c, out, ckpt,
                                 T, CHUNK_FAST, BH, scale_f, num_warps=4, num_stages=3)
            ctx.fast = True
            ctx.blocks = (128, 32, 4)
            ctx.save_for_backward(q_c, k_c, v_c, e_c, d_c, beta_c, gamma_c, ckpt)
            ctx.shapes = (B, H, T, K, VD, BH, N)
            ctx.scale_f = scale_f
            ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
            return out
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
        _fwd_generic_kernel[grid](q_c, k_c, v_c, e_c, d_c, beta_c, gamma_c, out,
                                  T, K, VD, scale_f, BLOCK_K, BLOCK_V, N_VB,
                                  num_warps=4, num_stages=2)
        CH = 32
        N = (T + CH - 1) // CH
        ckpt = torch.zeros(N + 1, BH, K, VD, device=q.device, dtype=torch.float32)
        _ckpt_kernel[grid](k_c, e_c, v_c, d_c, beta_c, gamma_c, ckpt,
                           BH, K, VD, T, CH, BLOCK_K, BLOCK_V, N_VB,
                           num_warps=4, num_stages=2)
        ctx.fast = False
        ctx.blocks = (BLOCK_K, BLOCK_V, N_VB)
        ctx.chunk = CH
        ctx.save_for_backward(q_c, k_c, v_c, e_c, d_c, beta_c, gamma_c, ckpt)
        ctx.shapes = (B, H, T, K, VD, BH, N)
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
        scale_f = ctx.scale_f
        do_c = grad_output.contiguous().to(torch.float32)
        if ctx.fast:
            dq = torch.zeros(BH, T, K, device=q_c.device, dtype=torch.float32)
            dk = torch.zeros(BH, T, K, device=q_c.device, dtype=torch.float32)
            dv = torch.zeros(BH, T, VD, device=q_c.device, dtype=torch.float32)
            de = torch.zeros(BH, T, K, device=q_c.device, dtype=torch.float32)
            dg = torch.zeros(BH, T, K, device=q_c.device, dtype=torch.float32)
            db = torch.zeros(BH, T, device=q_c.device, dtype=torch.float32)
            dc = torch.zeros(BH, T, device=q_c.device, dtype=torch.float32)
            grid = (BH * 4,)
            _bwd_fast[grid](q_c, k_c, v_c, e_c, d_c, beta_c, gamma_c, do_c, ckpt,
                            dq, dk, dv, de, dg, db, dc,
                            T, CHUNK_FAST, N, BH, scale_f, num_warps=4, num_stages=1)
            qd, kd, vd, ed, gd, bd, cd = ctx.dtypes
            return (dq.view(B, H, T, K).to(qd), dk.view(B, H, T, K).to(kd),
                    dv.view(B, H, T, VD).to(vd), de.view(B, H, T, K).to(ed),
                    dg.view(B, H, T, K).to(gd), db.view(B, H, T).to(bd),
                    dc.view(B, H, T).to(cd), None)
        BLOCK_K, BLOCK_V, N_VB = ctx.blocks
        CH = ctx.chunk
        dq = torch.zeros(BH, T, K, device=q_c.device, dtype=torch.float32)
        dk = torch.zeros(BH, T, K, device=q_c.device, dtype=torch.float32)
        dv = torch.zeros(BH, T, VD, device=q_c.device, dtype=torch.float32)
        de = torch.zeros(BH, T, K, device=q_c.device, dtype=torch.float32)
        dg = torch.zeros(BH, T, K, device=q_c.device, dtype=torch.float32)
        db = torch.zeros(BH, T, device=q_c.device, dtype=torch.float32)
        dc = torch.zeros(BH, T, device=q_c.device, dtype=torch.float32)
        carry = torch.zeros(BH, K, VD, device=q_c.device, dtype=torch.float32)
        sbuf = torch.empty(BH, CH, K, VD, device=q_c.device, dtype=torch.float32)
        grid = (BH * N_VB,)
        for c in range(N - 1, -1, -1):
            Lc = T - c * CH
            if Lc > CH:
                Lc = CH
            _chunk_fwd_kernel[grid](k_c, e_c, v_c, d_c, beta_c, gamma_c, ckpt, sbuf,
                                    c, Lc, BH, K, VD, T, CH, BLOCK_K, BLOCK_V, N_VB,
                                    num_warps=4, num_stages=2)
            _chunk_bwd_kernel[grid](q_c, k_c, v_c, e_c, d_c, beta_c, gamma_c, do_c, sbuf, ckpt, carry,
                                    dq, dk, dv, de, dg, db, dc,
                                    c, Lc, BH, K, VD, scale_f, T, CH, BLOCK_K, BLOCK_V, N_VB,
                                    num_warps=4, num_stages=2)
        qd, kd, vd, ed, gd, bd, cd = ctx.dtypes
        return (dq.view(B, H, T, K).to(qd), dk.view(B, H, T, K).to(kd),
                dv.view(B, H, T, VD).to(vd), de.view(B, H, T, K).to(ed),
                dg.view(B, H, T, K).to(gd), db.view(B, H, T).to(bd),
                dc.view(B, H, T).to(cd), None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _Func.apply(q, k, v, e, g, beta, gamma, scale)
