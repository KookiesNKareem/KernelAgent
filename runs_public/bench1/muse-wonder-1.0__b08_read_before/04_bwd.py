import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward: exact sequential recurrence, state slice [BV, BK] in registers.
# Also writes the (original-frame) state at every _CP-step boundary so the
# backward can recompute each 32-step chunk independently.
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr, cp_ptr,
    T, H, K, V, NVB, NCP, scale,
    sq_b, sq_h, sq_t, sq_k,
    sk_b, sk_h, sk_t, sk_k,
    sv_b, sv_h, sv_t, sv_v,
    sg_b, sg_h, sg_t, sg_k,
    sb_b, sb_h, sb_t,
    BK: tl.constexpr, BV: tl.constexpr,
    K_FULL: tl.constexpr, V_FULL: tl.constexpr,
    CP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    bh = pid // NVB
    vb = pid % NVB
    b_idx = bh // H
    h_idx = bh % H

    offs_k = tl.arange(0, BK)
    offs_v = vb * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_kv = mask_k[:, None] & mask_v[None, :]

    q_ptrs = q_ptr + b_idx * sq_b + h_idx * sq_h + offs_k * sq_k
    k_ptrs = k_ptr + b_idx * sk_b + h_idx * sk_h + offs_k * sk_k
    g_ptrs = g_ptr + b_idx * sg_b + h_idx * sg_h + offs_k * sg_k
    v_ptrs = v_ptr + b_idx * sv_b + h_idx * sv_h + offs_v * sv_v
    beta_ptr_cur = beta_ptr + b_idx * sb_b + h_idx * sb_h
    out_ptrs = out_ptr + bh * T * V + offs_v
    cp_base = cp_ptr + bh * NCP * K * V

    S = tl.zeros([BV, BK], dtype=tl.float32)  # transposed state slice

    for t in range(0, T):
        if t % CP == 0:
            tl.store(
                cp_base + (t // CP) * K * V + offs_k[:, None] * V + offs_v[None, :],
                tl.trans(S),
                mask=mask_kv,
            )
        if K_FULL:
            qv = tl.load(q_ptrs).to(tl.float32)
            kv = tl.load(k_ptrs).to(tl.float32)
            gv = tl.load(g_ptrs).to(tl.float32)
        else:
            qv = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
            kv = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
            gv = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        if V_FULL:
            vv = tl.load(v_ptrs).to(tl.float32)
        else:
            vv = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_ptr_cur).to(tl.float32)

        S = S * tl.exp(gv)[None, :]
        o = tl.sum(S * qv[None, :], axis=1)
        if V_FULL:
            tl.store(out_ptrs, o * scale)
        else:
            tl.store(out_ptrs, o * scale, mask=mask_v)
        pred = tl.sum(S * kv[None, :], axis=1)
        delta = beta_t * (vv - pred)
        S = S + delta[:, None] * kv[None, :]

        q_ptrs += sq_t
        k_ptrs += sk_t
        g_ptrs += sg_t
        v_ptrs += sv_t
        beta_ptr_cur += sb_t
        out_ptrs += V

    tl.store(
        cp_base + (NCP - 1) * K * V + offs_k[:, None] * V + offs_v[None, :],
        tl.trans(S),
        mask=mask_kv,
    )


# ---------------------------------------------------------------------------
# Backward: chunks are processed in reverse, carrying the state adjoint
# D [BK, BV] for this value slice.  Inside a chunk the decay is factored
# out (pivot = half the chunk's total log-decay), turning the chunk into
# an undamped asymmetric delta rule:
#   Shat0 = exp(p) S0,  qhat = q e^{c-p}, khat = k e^{c-p}, ktilde = k e^{p-c}
#   (I + A) Delta = beta (V - Khat Shat0),  A_ij = beta_i <khat_i, ktilde_j>
#   O/scale = Qhat Shat0 + M Delta,   M_ij = <qhat_i, ktilde_j>   (j < i)
# All gradients below are the exact adjoints of these relations, plus the
# chain rule through c (reverse cumsum), p = total/2 and the two state
# rescalings.  dq/dk/dg/dbeta are partial sums over the value slice and
# are combined with atomic adds; dv belongs exclusively to this slice.
# Chunks whose cumulative-decay range is too wide for factored
# exponentials are replayed step-by-step instead.
# ---------------------------------------------------------------------------
@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, do_ptr, cp_ptr,
    dq_ptr, dk_ptr, dv_ptr, dg_ptr, dbeta_ptr,
    T, H, K, V, NVB, NCH, scale,
    sq_b, sq_h, sq_t, sq_k,
    sk_b, sk_h, sk_t, sk_k,
    sv_b, sv_h, sv_t, sv_v,
    sg_b, sg_h, sg_t, sg_k,
    sb_b, sb_h, sb_t,
    so_b, so_h, so_t, so_v,
    BK: tl.constexpr, BV: tl.constexpr, CHUNK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    bh = pid // NVB
    vb = pid % NVB
    b_idx = bh // H
    h_idx = bh % H

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = vb * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_kv = mask_k[:, None] & mask_v[None, :]

    base_q = b_idx * sq_b + h_idx * sq_h
    base_k = b_idx * sk_b + h_idx * sk_h
    base_v = b_idx * sv_b + h_idx * sv_h
    base_g = b_idx * sg_b + h_idx * sg_h
    base_beta = b_idx * sb_b + h_idx * sb_h
    base_o = b_idx * so_b + h_idx * so_h
    cp_base = bh * (NCH + 1) * K * V
    grad_base_k = bh * T * K
    grad_base_v = bh * T * V

    strict = offs_c[:, None] > offs_c[None, :]
    eye = tl.where(offs_c[:, None] == offs_c[None, :], 1.0, 0.0)
    revcum = tl.where(offs_c[None, :] >= offs_c[:, None], 1.0, 0.0)

    D = tl.zeros([BK, BV], dtype=tl.float32)  # dL/dS at current chunk end

    for chunk_rev in range(0, NCH):
        chunk = NCH - 1 - chunk_rev
        start = chunk * CHUNK
        t_idx = start + offs_c
        mask_t = t_idx < T
        mask_ck = mask_t[:, None] & mask_k[None, :]
        mask_cv = mask_t[:, None] & mask_v[None, :]

        g = tl.load(
            g_ptr + base_g + t_idx[:, None] * sg_t + offs_k[None, :] * sg_k,
            mask=mask_ck, other=0.0,
        ).to(tl.float32)
        c = tl.cumsum(g, axis=0)
        total = tl.sum(g, axis=0)
        c_range = tl.max(tl.max(c, axis=0) - tl.min(c, axis=0))

        S0 = tl.load(
            cp_ptr + cp_base + chunk * K * V
            + offs_k[:, None] * V + offs_v[None, :],
            mask=mask_kv, other=0.0,
        ).to(tl.float32)

        if c_range <= 80.0:
            pivot = total * 0.5
            q = tl.load(
                q_ptr + base_q + t_idx[:, None] * sq_t + offs_k[None, :] * sq_k,
                mask=mask_ck, other=0.0,
            ).to(tl.float32)
            k = tl.load(
                k_ptr + base_k + t_idx[:, None] * sk_t + offs_k[None, :] * sk_k,
                mask=mask_ck, other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptr + base_v + t_idx[:, None] * sv_t + offs_v[None, :] * sv_v,
                mask=mask_cv, other=0.0,
            ).to(tl.float32)
            do = tl.load(
                do_ptr + base_o + t_idx[:, None] * so_t + offs_v[None, :] * so_v,
                mask=mask_cv, other=0.0,
            ).to(tl.float32)
            beta = tl.load(
                beta_ptr + base_beta + t_idx * sb_t, mask=mask_t, other=0.0
            ).to(tl.float32)

            e_cp = tl.exp(c - pivot[None, :])
            e_pc = tl.exp(pivot[None, :] - c)
            q_hat = q * e_cp
            k_hat = k * e_cp
            k_tilde = k * e_pc
            s_hat0 = S0 * tl.exp(pivot)[:, None]

            kk = tl.dot(k_hat, tl.trans(k_tilde))
            a_mat = tl.where(strict, kk * beta[:, None], 0.0)
            b_mat = a_mat + eye
            inv = eye
            for _ in range(5):
                inv = inv + tl.dot(
                    inv, eye - tl.dot(b_mat, inv, input_precision="ieee"),
                    input_precision="ieee",
                )

            resid = v - tl.dot(k_hat, s_hat0)
            delta = tl.dot(inv, beta[:, None] * resid)
            s_last = s_hat0 + tl.dot(tl.trans(k_tilde), delta)

            a_last = D * tl.exp(total - pivot)[:, None]
            g_prime = do * scale

            m_mat = tl.where(strict, tl.dot(q_hat, tl.trans(k_tilde)), 0.0)
            d_delta = tl.dot(tl.trans(m_mat), g_prime) + tl.dot(k_tilde, a_last)
            d_rhs = tl.dot(tl.trans(inv), d_delta)
            d_a = tl.where(strict, -tl.dot(d_rhs, tl.trans(delta)), 0.0)
            grad_kk = d_a * beta[:, None]
            d_khat = tl.dot(grad_kk, k_tilde) - tl.dot(
                beta[:, None] * d_rhs, tl.trans(s_hat0)
            )
            d_m = tl.where(strict, tl.dot(g_prime, tl.trans(delta)), 0.0)
            d_qhat = tl.dot(g_prime, tl.trans(s_hat0)) + tl.dot(d_m, k_tilde)
            d_ktilde = (
                tl.dot(tl.trans(grad_kk), k_hat)
                + tl.dot(tl.trans(d_m), q_hat)
                + tl.dot(delta, tl.trans(a_last))
            )
            d_shat = (
                a_last
                + tl.dot(tl.trans(q_hat), g_prime)
                - tl.dot(tl.trans(k_hat), beta[:, None] * d_rhs)
            )

            dbeta = tl.sum(d_a * kk, axis=1) + tl.sum(d_rhs * resid, axis=1)
            tl.store(
                dv_ptr + grad_base_v + t_idx[:, None] * V + offs_v[None, :],
                beta[:, None] * d_rhs,
                mask=mask_cv,
            )
            tl.atomic_add(
                dbeta_ptr + bh * T + t_idx, dbeta, mask=mask_t
            )
            tl.atomic_add(
                dq_ptr + grad_base_k + t_idx[:, None] * K + offs_k[None, :],
                d_qhat * e_cp,
                mask=mask_ck,
            )
            tl.atomic_add(
                dk_ptr + grad_base_k + t_idx[:, None] * K + offs_k[None, :],
                d_khat * e_cp + d_ktilde * e_pc,
                mask=mask_ck,
            )

            dc = d_qhat * q_hat + d_khat * k_hat - d_ktilde * k_tilde
            send_term = tl.sum(a_last * s_last, axis=1)
            dp = (
                tl.sum(
                    -d_qhat * q_hat - d_khat * k_hat + d_ktilde * k_tilde,
                    axis=0,
                )
                + tl.sum(d_shat * s_hat0, axis=1)
                - send_term
            )
            dc = dc + tl.where(
                offs_c[:, None] == CHUNK - 1,
                (0.5 * dp + send_term)[None, :],
                0.0,
            )
            dg_mat = tl.dot(revcum, dc, input_precision="ieee")
            tl.atomic_add(
                dg_ptr + grad_base_k + t_idx[:, None] * K + offs_k[None, :],
                dg_mat,
                mask=mask_ck,
            )

            D = d_shat * tl.exp(pivot)[:, None]
        else:
            # Wide decay range: replay this chunk forwards for each step
            # (to recover its decayed state), then apply the exact
            # one-step adjoint.  Rare; correctness over speed here.
            for rev_i in tl.static_range(0, CHUNK):
                i = CHUNK - 1 - rev_i
                t = start + i
                if t < T:
                    s_cur = S0
                    s_minus = tl.zeros([BK, BV], dtype=tl.float32)
                    q_i = tl.zeros([BK], dtype=tl.float32)
                    k_i = tl.zeros([BK], dtype=tl.float32)
                    g_i = tl.zeros([BK], dtype=tl.float32)
                    delta_i = tl.zeros([BV], dtype=tl.float32)
                    resid_i = tl.zeros([BV], dtype=tl.float32)
                    beta_i = tl.zeros([], dtype=tl.float32)
                    for j in range(0, CHUNK):
                        tj = start + j
                        if j <= i:
                            qj = tl.load(
                                q_ptr + base_q + tj * sq_t + offs_k * sq_k,
                                mask=mask_k, other=0.0,
                            ).to(tl.float32)
                            kj = tl.load(
                                k_ptr + base_k + tj * sk_t + offs_k * sk_k,
                                mask=mask_k, other=0.0,
                            ).to(tl.float32)
                            gj = tl.load(
                                g_ptr + base_g + tj * sg_t + offs_k * sg_k,
                                mask=mask_k, other=0.0,
                            ).to(tl.float32)
                            vj = tl.load(
                                v_ptr + base_v + tj * sv_t + offs_v * sv_v,
                                mask=mask_v, other=0.0,
                            ).to(tl.float32)
                            bj = tl.load(
                                beta_ptr + base_beta + tj * sb_t
                            ).to(tl.float32)
                            s_dec = s_cur * tl.exp(gj)[:, None]
                            pred = tl.sum(s_dec * kj[:, None], axis=0)
                            rj = vj - pred
                            dj = bj * rj
                            if j == i:
                                s_minus = s_dec
                                q_i = qj
                                k_i = kj
                                g_i = gj
                                delta_i = dj
                                resid_i = rj
                                beta_i = bj
                            s_cur = s_dec + kj[:, None] * dj[None, :]
                    do_i = tl.load(
                        do_ptr + base_o + t * so_t + offs_v * so_v,
                        mask=mask_v, other=0.0,
                    ).to(tl.float32)
                    ak = tl.sum(D * k_i[:, None], axis=0)
                    bmat = (
                        D
                        - beta_i * k_i[:, None] * ak[None, :]
                        + scale * q_i[:, None] * do_i[None, :]
                    )
                    tl.store(
                        dv_ptr + grad_base_v + t * V + offs_v,
                        beta_i * ak,
                        mask=mask_v,
                    )
                    tl.atomic_add(
                        dbeta_ptr + bh * T + t,
                        tl.sum(ak * resid_i),
                    )
                    tl.atomic_add(
                        dq_ptr + grad_base_k + t * K + offs_k,
                        scale * tl.sum(s_minus * do_i[None, :], axis=1),
                        mask=mask_k,
                    )
                    tl.atomic_add(
                        dk_ptr + grad_base_k + t * K + offs_k,
                        tl.sum(D * delta_i[None, :], axis=1)
                        - beta_i * tl.sum(s_minus * ak[None, :], axis=1),
                        mask=mask_k,
                    )
                    tl.atomic_add(
                        dg_ptr + grad_base_k + t * K + offs_k,
                        tl.sum(bmat * s_minus, axis=1),
                        mask=mask_k,
                    )
                    D = bmat * tl.exp(g_i)[:, None]


def _next_pow2(x):
    n = 1
    while n < x:
        n *= 2
    return n


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        ctx.shape = (B, H, T, K, V)
        ctx.scale = float(scale)
        ctx.save_for_backward(q, k, v, e, g, beta, gamma)
        ctx.cp = None
        ctx.trivial = B == 0 or H == 0 or T == 0 or V == 0 or K == 0

        out = torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)
        if ctx.trivial:
            return out

        CHUNK = 32
        nch = (T + CHUNK - 1) // CHUNK
        cp = torch.empty(
            (B * H * (nch + 1) * K * V,), device=q.device, dtype=torch.float32
        )
        ctx.cp = cp
        ctx.nch = nch

        BK = max(16, _next_pow2(K))
        if BK <= 64:
            BV = 64
        elif BK <= 128:
            BV = 32
        else:
            BV = 16
        nvb = (V + BV - 1) // BV
        _fwd_kernel[(B * H * nvb,)](
            q, k, v, g, beta, out, cp,
            T, H, K, V, nvb, nch + 1, float(scale),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            BK=BK, BV=BV,
            K_FULL=(K == BK), V_FULL=(V % BV == 0),
            CP=CHUNK,
            num_warps=4, num_stages=2,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma = ctx.saved_tensors
        B, H, T, K, V = ctx.shape
        if ctx.trivial or grad_out is None:
            return (
                torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v),
                torch.zeros_like(e), torch.zeros_like(g),
                torch.zeros_like(beta), torch.zeros_like(gamma), None,
            )

        go = grad_out.contiguous() if not grad_out.is_contiguous() else grad_out
        dq = torch.zeros((B * H * T * K,), device=q.device, dtype=torch.float32)
        dk = torch.zeros((B * H * T * K,), device=q.device, dtype=torch.float32)
        dg = torch.zeros((B * H * T * K,), device=q.device, dtype=torch.float32)
        dv = torch.empty((B * H * T * V,), device=q.device, dtype=torch.float32)
        dbeta = torch.zeros((B * H * T,), device=q.device, dtype=torch.float32)

        BK = max(16, _next_pow2(K))
        BV = 32 if BK <= 128 else 16
        nvb = (V + BV - 1) // BV
        _bwd_kernel[(B * H * nvb,)](
            q, k, v, g, beta, go, ctx.cp,
            dq, dk, dv, dg, dbeta,
            T, H, K, V, nvb, ctx.nch, ctx.scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            go.stride(0), go.stride(1), go.stride(2), go.stride(3),
            BK=BK, BV=BV, CHUNK=32,
            num_warps=4, num_stages=1,
        )

        return (
            dq.view(B, H, T, K).to(q.dtype),
            dk.view(B, H, T, K).to(k.dtype),
            dv.view(B, H, T, V).to(v.dtype),
            torch.zeros_like(e),
            dg.view(B, H, T, K).to(g.dtype),
            dbeta.view(B, H, T).to(beta.dtype),
            torch.zeros_like(gamma),
            None,
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _GatedDelta.apply(q, k, v, e, g, beta, gamma, float(scale))
