"""Coupled erase Triton kernels (sequential V-split).

Forward recurrence per step (K = key dim, V = value dim):
  d = exp(g_t)                      # [K]
  S' = diag(d) S_{t-1}              # [K,V]
  c = gamma_t * (1 - beta_t)
  s_e = S'^T e_t                    # [V]
  S'' = S' - c e_t s_e^T
  pred = S''^T k_t                  # [V]
  r = v_t - pred
  S_t = S'' + beta_t k_t r^T
  o_t = scale * S_t^T q_t           # [V]

The state is split over the V dimension: one program handles K x BV,
so every reduction is over K (local) and every output over V (local).
Chunk-boundary states are saved for the backward recompute.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _coupled_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    o_ptr,
    sb_ptr,
    scale_ptr,
    BH,
    T,
    Kdim,
    Vdim,
    num_chunks,
    num_v_blocks,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    USE_F64: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // num_v_blocks
    vb = pid % num_v_blocks

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < Kdim
    offs_bv = tl.arange(0, BLOCK_V)
    offs_v = vb * BLOCK_V + offs_bv
    mask_v = offs_v < Vdim

    if USE_F64:
        S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float64)
        scale_val = tl.load(scale_ptr).to(tl.float64)
    else:
        S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        scale_val = tl.load(scale_ptr).to(tl.float32)

    q_base = bh * T * Kdim
    k_base = bh * T * Kdim
    e_base = bh * T * Kdim
    g_base = bh * T * Kdim
    v_base = bh * T * Vdim
    o_base = bh * T * Vdim
    b_base = bh * T

    for t in range(T):
        chunk = t // CHUNK
        if t % CHUNK == 0:
            sb_ptrs = (
                sb_ptr
                + ((bh * num_chunks + chunk) * Kdim + offs_k[:, None]) * Vdim
                + offs_v[None, :]
            )
            tl.store(sb_ptrs, S, mask=mask_k[:, None] & mask_v[None, :])

        q_ptrs = q_ptr + q_base + t * Kdim + offs_k
        k_ptrs = k_ptr + k_base + t * Kdim + offs_k
        e_ptrs = e_ptr + e_base + t * Kdim + offs_k
        g_ptrs = g_ptr + g_base + t * Kdim + offs_k
        v_ptrs = v_ptr + v_base + t * Vdim + offs_v
        if USE_F64:
            qv = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float64)
            kv = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float64)
            ev = tl.load(e_ptrs, mask=mask_k, other=0.0).to(tl.float64)
            gv = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float64)
            vv = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float64)
            bv = tl.load(beta_ptr + b_base + t).to(tl.float64)
            gv_s = tl.load(gamma_ptr + b_base + t).to(tl.float64)
        else:
            qv = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
            kv = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
            ev = tl.load(e_ptrs, mask=mask_k, other=0.0).to(tl.float32)
            gv = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float32)
            vv = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
            bv = tl.load(beta_ptr + b_base + t).to(tl.float32)
            gv_s = tl.load(gamma_ptr + b_base + t).to(tl.float32)

        d = tl.exp(gv)
        S = S * d[:, None]
        s_e = tl.sum(S * ev[:, None], axis=0)
        c = gv_s * (1.0 - bv)
        S = S - c * ev[:, None] * s_e[None, :]
        pred = tl.sum(S * kv[:, None], axis=0)
        r = vv - pred
        S = S + bv * kv[:, None] * r[None, :]
        qs = qv * scale_val
        out = tl.sum(S * qs[:, None], axis=0)
        o_ptrs = o_ptr + o_base + t * Vdim + offs_v
        tl.store(o_ptrs, out, mask=mask_v)


@triton.jit
def _coupled_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    go_ptr,
    dq_part_ptr,
    dk_part_ptr,
    de_part_ptr,
    dg_part_ptr,
    dbeta_part_ptr,
    dgamma_part_ptr,
    dv_ptr,
    sb_ptr,
    scratch_ptr,
    scale_ptr,
    BH,
    T,
    Kdim,
    Vdim,
    num_chunks,
    num_v_blocks,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    USE_F64: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // num_v_blocks
    vb = pid % num_v_blocks

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < Kdim
    offs_bv = tl.arange(0, BLOCK_V)
    offs_v = vb * BLOCK_V + offs_bv
    mask_v = offs_v < Vdim

    if USE_F64:
        dS = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float64)
        scale_val = tl.load(scale_ptr).to(tl.float64)
    else:
        dS = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        scale_val = tl.load(scale_ptr).to(tl.float32)

    q_base = bh * T * Kdim
    k_base = bh * T * Kdim
    e_base = bh * T * Kdim
    g_base = bh * T * Kdim
    v_base = bh * T * Vdim
    go_base = bh * T * Vdim
    b_base = bh * T

    scratch_base = pid * CHUNK * BLOCK_K * BLOCK_V

    for rev in range(num_chunks):
        chunk = num_chunks - 1 - rev
        start = chunk * CHUNK

        sb_ptrs0 = (
            sb_ptr
            + ((bh * num_chunks + chunk) * Kdim + offs_k[:, None]) * Vdim
            + offs_v[None, :]
        )
        S0 = tl.load(sb_ptrs0, mask=mask_k[:, None] & mask_v[None, :], other=0.0)
        if USE_F64:
            S0 = S0.to(tl.float64)
        else:
            S0 = S0.to(tl.float32)

        # Forward recompute through the chunk, saving S_t per step.
        S = S0
        for i in range(CHUNK):
            t = start + i
            if t < T:
                k_ptrs = k_ptr + k_base + t * Kdim + offs_k
                e_ptrs = e_ptr + e_base + t * Kdim + offs_k
                g_ptrs = g_ptr + g_base + t * Kdim + offs_k
                v_ptrs = v_ptr + v_base + t * Vdim + offs_v
                if USE_F64:
                    kv = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float64)
                    ev = tl.load(e_ptrs, mask=mask_k, other=0.0).to(tl.float64)
                    gvv = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float64)
                    vv = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float64)
                    bv = tl.load(beta_ptr + b_base + t).to(tl.float64)
                    gms = tl.load(gamma_ptr + b_base + t).to(tl.float64)
                else:
                    kv = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
                    ev = tl.load(e_ptrs, mask=mask_k, other=0.0).to(tl.float32)
                    gvv = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float32)
                    vv = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
                    bv = tl.load(beta_ptr + b_base + t).to(tl.float32)
                    gms = tl.load(gamma_ptr + b_base + t).to(tl.float32)
                d = tl.exp(gvv)
                S = S * d[:, None]
                s_e = tl.sum(S * ev[:, None], axis=0)
                c = gms * (1.0 - bv)
                S = S - c * ev[:, None] * s_e[None, :]
                pred = tl.sum(S * kv[:, None], axis=0)
                r = vv - pred
                S = S + bv * kv[:, None] * r[None, :]
                s_ptrs = (
                    scratch_ptr
                    + scratch_base
                    + i * BLOCK_K * BLOCK_V
                    + offs_k[:, None] * BLOCK_V
                    + offs_bv[None, :]
                )
                tl.store(s_ptrs, S)

        # Reverse pass through the chunk.
        for ii in range(CHUNK):
            i = CHUNK - 1 - ii
            t = start + i
            if t < T:
                if i == 0:
                    S_prev = S0
                else:
                    prev_ptrs = (
                        scratch_ptr
                        + scratch_base
                        + (i - 1) * BLOCK_K * BLOCK_V
                        + offs_k[:, None] * BLOCK_V
                        + offs_bv[None, :]
                    )
                    S_prev = tl.load(prev_ptrs)
                cur_ptrs = (
                    scratch_ptr
                    + scratch_base
                    + i * BLOCK_K * BLOCK_V
                    + offs_k[:, None] * BLOCK_V
                    + offs_bv[None, :]
                )
                S_cur = tl.load(cur_ptrs)

                q_ptrs = q_ptr + q_base + t * Kdim + offs_k
                k_ptrs = k_ptr + k_base + t * Kdim + offs_k
                e_ptrs = e_ptr + e_base + t * Kdim + offs_k
                g_ptrs = g_ptr + g_base + t * Kdim + offs_k
                v_ptrs = v_ptr + v_base + t * Vdim + offs_v
                go_ptrs = go_ptr + go_base + t * Vdim + offs_v
                if USE_F64:
                    qv = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float64)
                    kv = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float64)
                    ev = tl.load(e_ptrs, mask=mask_k, other=0.0).to(tl.float64)
                    gvv = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float64)
                    vv = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float64)
                    dov = tl.load(go_ptrs, mask=mask_v, other=0.0).to(tl.float64)
                    bv = tl.load(beta_ptr + b_base + t).to(tl.float64)
                    gms = tl.load(gamma_ptr + b_base + t).to(tl.float64)
                else:
                    qv = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
                    kv = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
                    ev = tl.load(e_ptrs, mask=mask_k, other=0.0).to(tl.float32)
                    gvv = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float32)
                    vv = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
                    dov = tl.load(go_ptrs, mask=mask_v, other=0.0).to(tl.float32)
                    bv = tl.load(beta_ptr + b_base + t).to(tl.float32)
                    gms = tl.load(gamma_ptr + b_base + t).to(tl.float32)

                d = tl.exp(gvv)
                S_prime = S_prev * d[:, None]
                s_e = tl.sum(S_prime * ev[:, None], axis=0)
                c = gms * (1.0 - bv)
                S_dprime = S_prime - c * ev[:, None] * s_e[None, :]
                pred = tl.sum(S_dprime * kv[:, None], axis=0)
                r = vv - pred

                qs = qv * scale_val
                out_term = qs[:, None] * dov[None, :]
                G = out_term + dS
                a = tl.sum(G * kv[:, None], axis=0)

                dv_slice = bv * a
                dv_ptrs = dv_ptr + v_base + t * Vdim + offs_v
                tl.store(dv_ptrs, dv_slice, mask=mask_v)

                dq_p = scale_val * tl.sum(S_cur * dov[None, :], axis=1)
                t1 = tl.sum(G * r[None, :], axis=1)
                t2 = tl.sum(S_dprime * a[None, :], axis=1)
                dk_p = bv * (t1 - t2)

                H = G - bv * kv[:, None] * a[None, :]
                t_e = tl.sum(H * ev[:, None], axis=0)
                tA = tl.sum(H * s_e[None, :], axis=1)
                tB = tl.sum(S_prime * t_e[None, :], axis=1)
                de_p = -c * (tA + tB)

                J = H - c * ev[:, None] * t_e[None, :]
                dg_p = d * tl.sum(J * S_prev, axis=1)

                sum_ra = tl.sum(r * a, axis=0)
                sum_st = tl.sum(s_e * t_e, axis=0)
                db_p = sum_ra + gms * sum_st
                dgm_p = -(1.0 - bv) * sum_st

                base_k = ((vb * BH + bh) * T + t) * Kdim
                tl.store(dq_part_ptr + base_k + offs_k, dq_p, mask=mask_k)
                tl.store(dk_part_ptr + base_k + offs_k, dk_p, mask=mask_k)
                tl.store(de_part_ptr + base_k + offs_k, de_p, mask=mask_k)
                tl.store(dg_part_ptr + base_k + offs_k, dg_p, mask=mask_k)
                base_1 = (vb * BH + bh) * T + t
                tl.store(dbeta_part_ptr + base_1, db_p)
                tl.store(dgamma_part_ptr + base_1, dgm_p)

                dS = d[:, None] * J


class _CoupledEraseFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        compute_f64 = (
            q.dtype == torch.float64
            or k.dtype == torch.float64
            or v.dtype == torch.float64
            or e.dtype == torch.float64
            or g.dtype == torch.float64
        )
        comp_dtype = torch.float64 if compute_f64 else torch.float32
        use_f64 = compute_f64

        q2 = q.reshape(BH, T, K).contiguous()
        k2 = k.reshape(BH, T, K).contiguous()
        e2 = e.reshape(BH, T, K).contiguous()
        g2 = g.reshape(BH, T, K).contiguous()
        v2 = v.reshape(BH, T, V).contiguous()
        beta2 = beta.reshape(BH, T).contiguous()
        gamma2 = gamma.reshape(BH, T).contiguous()

        BLOCK_K = triton.next_power_of_2(K)
        if BLOCK_K < 16:
            BLOCK_K = 16
        if V >= 32:
            BLOCK_V = 32
        else:
            BLOCK_V = triton.next_power_of_2(V)
            if BLOCK_V < 16:
                BLOCK_V = 16
        NV = triton.cdiv(V, BLOCK_V)
        CHUNK = 64
        num_chunks = triton.cdiv(T, CHUNK)

        out = torch.empty((BH, T, V), device=q.device, dtype=comp_dtype)
        sb = torch.empty((BH * num_chunks * K * V,), device=q.device, dtype=comp_dtype)
        scale_t = torch.tensor([float(scale)], device=q.device, dtype=comp_dtype)

        _coupled_fwd_kernel[(BH * NV,)](
            q2, k2, v2, e2, g2, beta2, gamma2, out, sb, scale_t,
            BH, T, K, V, num_chunks, NV,
            CHUNK=CHUNK, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, USE_F64=use_f64,
            num_warps=4, num_stages=2,
        )

        ctx.save_for_backward(q2, k2, v2, e2, g2, beta2, gamma2, sb, scale_t)
        ctx.dims = (B, H, T, K, V, BH, NV, num_chunks, BLOCK_K, BLOCK_V, CHUNK, use_f64)
        ctx.comp_dtype = comp_dtype
        ctx.orig_dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
        return out.reshape(B, H, T, V)

    @staticmethod
    def backward(ctx, grad_out):
        q2, k2, v2, e2, g2, beta2, gamma2, sb, scale_t = ctx.saved_tensors
        B, H, T, K, V, BH, NV, num_chunks, BLOCK_K, BLOCK_V, CHUNK, use_f64 = ctx.dims
        comp_dtype = ctx.comp_dtype
        q_dt, k_dt, v_dt, e_dt, g_dt, beta_dt, gamma_dt = ctx.orig_dtypes

        go2 = grad_out.reshape(BH, T, V).contiguous().to(comp_dtype)

        dq_buf = torch.empty((NV * BH * T * K,), device=q2.device, dtype=comp_dtype)
        dk_buf = torch.empty((NV * BH * T * K,), device=q2.device, dtype=comp_dtype)
        de_buf = torch.empty((NV * BH * T * K,), device=q2.device, dtype=comp_dtype)
        dg_buf = torch.empty((NV * BH * T * K,), device=q2.device, dtype=comp_dtype)
        dbeta_buf = torch.empty((NV * BH * T,), device=q2.device, dtype=comp_dtype)
        dgamma_buf = torch.empty((NV * BH * T,), device=q2.device, dtype=comp_dtype)
        dv = torch.empty((BH, T, V), device=q2.device, dtype=comp_dtype)
        scratch = torch.empty(
            (BH * NV * CHUNK * BLOCK_K * BLOCK_V,), device=q2.device, dtype=comp_dtype
        )

        _coupled_bwd_kernel[(BH * NV,)](
            q2, k2, v2, e2, g2, beta2, gamma2, go2,
            dq_buf, dk_buf, de_buf, dg_buf, dbeta_buf, dgamma_buf,
            dv, sb, scratch, scale_t,
            BH, T, K, V, num_chunks, NV,
            CHUNK=CHUNK, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, USE_F64=use_f64,
            num_warps=4, num_stages=2,
        )

        dq_p = dq_buf.reshape(NV, BH, T, K).sum(0).reshape(B, H, T, K).to(q_dt)
        dk_p = dk_buf.reshape(NV, BH, T, K).sum(0).reshape(B, H, T, K).to(k_dt)
        dv_p = dv.reshape(B, H, T, V).to(v_dt)
        de_p = de_buf.reshape(NV, BH, T, K).sum(0).reshape(B, H, T, K).to(e_dt)
        dg_p = dg_buf.reshape(NV, BH, T, K).sum(0).reshape(B, H, T, K).to(g_dt)
        dbeta_p = dbeta_buf.reshape(NV, BH, T).sum(0).reshape(B, H, T).to(beta_dt)
        dgamma_p = dgamma_buf.reshape(NV, BH, T).sum(0).reshape(B, H, T).to(gamma_dt)
        return dq_p, dk_p, dv_p, de_p, dg_p, dbeta_p, dgamma_p, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _CoupledEraseFunc.apply(q, k, v, e, g, beta, gamma, float(scale))
