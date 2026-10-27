import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    sq_b, sq_h, sq_t, sq_k,
    sk_b, sk_h, sk_t, sk_k,
    sv_b, sv_h, sv_t, sv_v,
    se_b, se_h, se_t, se_k,
    sg_b, sg_h, sg_t, sg_k,
    sb_b, sb_h, sb_t,
    sgm_b, sgm_h, sgm_t,
    so_b, so_h, so_t, so_v,
    H, T, K, V,
    scale,
    BC: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, BC)
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V

    q_base = q_ptr + b * sq_b + h * sq_h
    k_base = k_ptr + b * sk_b + h * sk_h
    v_base = v_ptr + b * sv_b + h * sv_h
    e_base = e_ptr + b * se_b + h * se_h
    g_base = g_ptr + b * sg_b + h * sg_h
    beta_base = beta_ptr + b * sb_b + h * sb_h
    gamma_base = gamma_ptr + b * sgm_b + h * sgm_h
    out_base = out_ptr + b * so_b + h * so_h

    # Persistent actual state for this (batch, head, V-block): [K, BV].
    S = tl.zeros((BK, BV), dtype=tl.float32)

    offs_s = offs_m // 2
    is_e = (offs_m % 2) == 1

    for start in range(0, T, BC):
        t_elem = start + offs_s
        mask_elem = t_elem < T
        mask_mk = mask_elem[:, None] & mask_k[None, :]
        mask_mv = mask_elem[:, None] & mask_v[None, :]

        k_m = tl.load(
            k_base + t_elem[:, None] * sk_t + offs_k[None, :] * sk_k,
            mask=mask_mk, other=0.0,
        ).to(tl.float32)
        e_m = tl.load(
            e_base + t_elem[:, None] * se_t + offs_k[None, :] * se_k,
            mask=mask_mk, other=0.0,
        ).to(tl.float32)
        g_m_load = tl.load(
            g_base + t_elem[:, None] * sg_t + offs_k[None, :] * sg_k,
            mask=mask_mk, other=0.0,
        ).to(tl.float32)
        beta_m = tl.load(
            beta_base + t_elem * sb_t, mask=mask_elem, other=0.0
        ).to(tl.float32)
        gamma_m = tl.load(
            gamma_base + t_elem * sgm_t, mask=mask_elem, other=0.0
        ).to(tl.float32)

        # Interleaved elementary sequence: write(k), erase(e), write, erase...
        # In cumulative-decay coordinates both are generalized delta updates.
        g_elem = tl.where(is_e[:, None], 0.0, g_m_load)
        G_m = tl.cumsum(g_elem, axis=0)
        G_last = tl.sum(
            tl.where((offs_m == BM - 1)[:, None], G_m, 0.0), axis=0
        )
        # Referencing decay to the chunk midpoint keeps exp() arguments
        # centred. This is only finite while the chunk's cumulative decay
        # range is moderate: exp(+/-40) is safely inside fp32, beyond that
        # the scaled read/write keys overflow (exp(+400) = inf) even though
        # every true pairwise factor exp(G_j - G_i) <= 1.
        log_mid = 0.5 * G_last
        dg = G_m - log_mid[None, :]
        need_seq = tl.max(tl.abs(dg)) > 40.0

        if need_seq:
            # Strong-decay chunk: run the plain recurrence for these BC
            # steps. It only ever evaluates exp(g_t) <= 1, so it is stable
            # for arbitrarily negative g. S is already the actual state.
            for i in range(BC):
                t = start + i
                mt = t < T
                gv = tl.load(
                    g_base + t * sg_t + offs_k * sg_k,
                    mask=mask_k & mt, other=0.0,
                ).to(tl.float32)
                S = S * tl.exp(gv)[:, None]

                kv = tl.load(
                    k_base + t * sk_t + offs_k * sk_k,
                    mask=mask_k & mt, other=0.0,
                ).to(tl.float32)
                vv = tl.load(
                    v_base + t * sv_t + offs_v * sv_v,
                    mask=mask_v & mt, other=0.0,
                ).to(tl.float32)
                beta_t = tl.load(
                    beta_base + t * sb_t, mask=mt, other=0.0
                ).to(tl.float32)
                pred = tl.sum(S * kv[:, None], axis=0)
                S = S + kv[:, None] * (beta_t * (vv - pred))[None, :]

                ev = tl.load(
                    e_base + t * se_t + offs_k * se_k,
                    mask=mask_k & mt, other=0.0,
                ).to(tl.float32)
                gamma_t = tl.load(
                    gamma_base + t * sgm_t, mask=mt, other=0.0
                ).to(tl.float32)
                proj = tl.sum(S * ev[:, None], axis=0)
                S = S - ev[:, None] * (gamma_t * proj)[None, :]

                qv = tl.load(
                    q_base + t * sq_t + offs_k * sq_k,
                    mask=mask_k & mt, other=0.0,
                ).to(tl.float32)
                out_t = tl.sum(S * (qv * scale)[:, None], axis=0)
                tl.store(
                    out_base + t * so_t + offs_v * so_v,
                    out_t, mask=mask_v & mt,
                )
        else:
            sc_m = tl.exp(dg)          # c'_t, per elementary row
            inv_sc_m = tl.exp(-dg)     # 1 / c'_t
            d_vec = tl.exp(log_mid)    # also equals c' at the chunk end

            orig_m = tl.where(is_e[:, None], e_m, k_m)
            coeff_m = tl.where(is_e, gamma_m, beta_m)
            W = orig_m * sc_m          # read keys  w = orig * c'
            Akey = orig_m * inv_sc_m   # write keys a = orig / c'

            # State expressed in this chunk's scaled coordinates.
            S_hat = S * d_vec[:, None]

            # Strictly-lower interaction matrix between elementary updates.
            A_mat = tl.dot(W, tl.trans(Akey))
            A_mat = tl.where(
                offs_m[:, None] > offs_m[None, :],
                coeff_m[:, None] * A_mat,
                0.0,
            )

            v_m = tl.load(
                v_base + t_elem[:, None] * sv_t + offs_v[None, :] * sv_v,
                mask=mask_mv, other=0.0,
            ).to(tl.float32)
            val_m = tl.where(is_e[:, None], 0.0, v_m)
            Y = coeff_m[:, None] * (val_m - tl.dot(W, S_hat))

            # Solve (I + A) R = Y for strictly-lower A (BM = 32, so A is
            # nilpotent of index 32) by Neumann doubling, unrolled exactly:
            # R successively covers sum_{p<2}, <4, <8, <16, <32 of P^p Y
            # with P = -A. The final, useless squaring is omitted.
            P = -A_mat
            R = Y + tl.dot(P, Y)
            Qpow = tl.dot(P, P)
            R = R + tl.dot(Qpow, R)
            Qpow = tl.dot(Qpow, Qpow)
            R = R + tl.dot(Qpow, R)
            Qpow = tl.dot(Qpow, Qpow)
            R = R + tl.dot(Qpow, R)
            Qpow = tl.dot(Qpow, Qpow)
            R = R + tl.dot(Qpow, R)

            # ---- outputs for this chunk ---------------------------------
            t_c = start + offs_c
            mask_c = t_c < T
            q_c = tl.load(
                q_base + t_c[:, None] * sq_t + offs_k[None, :] * sq_k,
                mask=mask_c[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
            g_c = tl.load(
                g_base + t_c[:, None] * sg_t + offs_k[None, :] * sg_k,
                mask=mask_c[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
            G_c = tl.cumsum(g_c, axis=0)
            q_hat = q_c * tl.exp(G_c - log_mid[None, :])

            out_c = tl.dot(q_hat, S_hat)
            Aq = tl.dot(Akey, tl.trans(q_hat))
            Aq = tl.where(
                offs_m[:, None] <= (2 * offs_c[None, :] + 1), Aq, 0.0
            )
            out_c = out_c + tl.dot(tl.trans(Aq), R)
            tl.store(
                out_base + t_c[:, None] * so_t + offs_v[None, :] * so_v,
                out_c * scale,
                mask=mask_c[:, None] & mask_v[None, :],
            )

            # ---- advance the persistent state ---------------------------
            S_end_hat = S_hat + tl.dot(tl.trans(Akey), R)
            S = S_end_hat * d_vec[:, None]


@triton.jit
def _recurrent_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    sq_b, sq_h, sq_t, sq_k,
    sk_b, sk_h, sk_t, sk_k,
    sv_b, sv_h, sv_t, sv_v,
    se_b, se_h, se_t, se_k,
    sg_b, sg_h, sg_t, sg_k,
    sb_b, sb_h, sb_t,
    sgm_b, sgm_h, sgm_t,
    so_b, so_h, so_t, so_v,
    H, T, K, V,
    scale,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V

    q_ptrs = q_ptr + b * sq_b + h * sq_h + offs_k * sq_k
    k_ptrs = k_ptr + b * sk_b + h * sk_h + offs_k * sk_k
    e_ptrs = e_ptr + b * se_b + h * se_h + offs_k * se_k
    g_ptrs = g_ptr + b * sg_b + h * sg_h + offs_k * sg_k
    v_ptrs = v_ptr + b * sv_b + h * sv_h + offs_v * sv_v
    out_ptrs = out_ptr + b * so_b + h * so_h + offs_v * so_v
    beta_pt = beta_ptr + b * sb_b + h * sb_h
    gamma_pt = gamma_ptr + b * sgm_b + h * sgm_h

    S = tl.zeros((BV, BK), dtype=tl.float32)
    for _t in range(T):
        g_vec = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        S = S * tl.exp(g_vec)[None, :]
        k_vec = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_pt).to(tl.float32)
        pred = tl.sum(S * k_vec[None, :], axis=1)
        S = S + (beta_t * (v_vec - pred))[:, None] * k_vec[None, :]
        e_vec = tl.load(e_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        gamma_t = tl.load(gamma_pt).to(tl.float32)
        proj = tl.sum(S * e_vec[None, :], axis=1)
        S = S - (gamma_t * proj)[:, None] * e_vec[None, :]
        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        out_vec = tl.sum(S * (q_vec * scale)[None, :], axis=1)
        tl.store(out_ptrs, out_vec, mask=mask_v)
        q_ptrs += sq_t
        k_ptrs += sk_t
        v_ptrs += sv_t
        e_ptrs += se_t
        g_ptrs += sg_t
        out_ptrs += so_t
        beta_pt += sb_t
        gamma_pt += sgm_t


def _run_recurrent(q, k, v, e, g, beta, gamma, out, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    BK = max(1, triton.next_power_of_2(max(1, K)))
    if BK <= 128:
        BV = 32
    elif BK <= 256:
        BV = 16
    elif BK <= 512:
        BV = 8
    else:
        BV = 4
    BV = min(BV, max(1, triton.next_power_of_2(V)))
    bh = B * H
    while BV > 4 and bh * ((V + BV - 1) // BV) < 64:
        BV //= 2
    num_warps = 8 if BV * BK >= 4096 else (4 if BV * BK >= 1024 else 2)
    grid = (bh, (V + BV - 1) // BV)
    _recurrent_kernel[grid](
        q, k, v, e, g, beta, gamma, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        e.stride(0), e.stride(1), e.stride(2), e.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H, T, K, V,
        float(scale),
        BK=BK,
        BV=BV,
        num_warps=num_warps,
    )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0 or V == 0:
        return out

    if K <= 128 and T >= 32:
        BC = 16
        BM = 32
        BK = max(16, triton.next_power_of_2(K))
        BV = 64 if V >= 64 else max(16, triton.next_power_of_2(V))
        grid = (B * H, (V + BV - 1) // BV)
        _chunk_kernel[grid](
            q, k, v, e, g, beta, gamma, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            e.stride(0), e.stride(1), e.stride(2), e.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            H, T, K, V,
            float(scale),
            BC=BC,
            BM=BM,
            BK=BK,
            BV=BV,
            num_warps=8,
        )
    else:
        _run_recurrent(q, k, v, e, g, beta, gamma, out, scale)
    return out
