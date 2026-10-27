```python
import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr, bnd_ptr,
    sq_b, sq_h, sq_t, sq_k,
    sk_b, sk_h, sk_t, sk_k,
    sv_b, sv_h, sv_t, sv_v,
    se_b, se_h, se_t, se_k,
    sg_b, sg_h, sg_t, sg_k,
    sb_b, sb_h, sb_t,
    sgm_b, sgm_h, sgm_t,
    so_b, so_h, so_t, so_v,
    H, T, K, V, NC,
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
    bnd_base = bnd_ptr + pid_bh * NC * K * V

    # Persistent actual state for this (batch, head, V-block): [K, BV].
    S = tl.zeros((BK, BV), dtype=tl.float32)

    offs_s = offs_m // 2
    is_e = (offs_m % 2) == 1

    chunk_idx = 0
    for start in range(0, T, BC):
        # Save chunk-boundary state for the backward pass.
        tl.store(
            bnd_base + chunk_idx * K * V + offs_k[:, None] * V + offs_v[None, :],
            S, mask=mask_k[:, None] & mask_v[None, :],
        )
        chunk_idx += 1

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
        g_elem = tl.where(is_e[:, None], 0.0, g_m_load)
        G_m = tl.cumsum(g_elem, axis=0)
        G_last = tl.sum(
            tl.where((offs_m == BM - 1)[:, None], G_m, 0.0), axis=0
        )
        log_mid = 0.5 * G_last
        dg = G_m - log_mid[None, :]
        need_seq = tl.max(tl.abs(dg)) > 40.0

        if need_seq:
            # Strong-decay chunk: plain recurrence (only exp(g_t) <= 1).
            for i in range(BC):
                t = start + i
                if t < T:
                    gv = tl.load(
                        g_base + t * sg_t + offs_k * sg_k, mask=mask_k, other=0.0
                    ).to(tl.float32)
                    S = S * tl.exp(gv)[:, None]
                    kv = tl.load(
                        k_base + t * sk_t + offs_k * sk_k, mask=mask_k, other=0.0
                    ).to(tl.float32)
                    vv = tl.load(
                        v_base + t * sv_t + offs_v * sv_v, mask=mask_v, other=0.0
                    ).to(tl.float32)
                    beta_t = tl.load(beta_base + t * sb_t).to(tl.float32)
                    pred = tl.sum(S * kv[:, None], axis=0)
                    S = S + kv[:, None] * (beta_t * (vv - pred))[None, :]

                    ev = tl.load(
                        e_base + t * se_t + offs_k * se_k, mask=mask_k, other=0.0
                    ).to(tl.float32)
                    gamma_t = tl.load(gamma_base + t * sgm_t).to(tl.float32)
                    proj = tl.sum(S * ev[:, None], axis=0)
                    S = S - ev[:, None] * (gamma_t * proj)[None, :]

                    qv = tl.load(
                        q_base + t * sq_t + offs_k * sq_k, mask=mask_k, other=0.0
                    ).to(tl.float32)
                    out_t = tl.sum(S * (qv * scale)[:, None], axis=0)
                    tl.store(out_base + t * so_t + offs_v * so_v, out_t, mask=mask_v)
        else:
            sc_m = tl.exp(dg)
            inv_sc_m = tl.exp(-dg)
            d_vec = tl.exp(log_mid)

            orig_m = tl.where(is_e[:, None], e_m, k_m)
            coeff_m = tl.where(is_e, gamma_m, beta_m)
            W = orig_m * sc_m
            Akey = orig_m * inv_sc_m

            S_hat = S * d_vec[:, None]

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

            # Neumann doubling for (I + A) R = Y, BM = 32 (nilpotent).
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

            S_end_hat = S_hat + tl.dot(tl.trans(Akey), R)
            S = S_end_hat * d_vec[:, None]


@triton.jit
def _recurrent_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr, bnd_ptr,
    sq_b, sq_h, sq_t, sq_k,
    sk_b, sk_h, sk_t, sk_k,
    sv_b, sv_h, sv_t, sv_v,
    se_b, se_h, se_t, se_k,
    sg_b, sg_h, sg_t, sg_k,
    sb_b, sb_h, sb_t,
    sgm_b, sgm_h, sgm_t,
    so_b, so_h, so_t, so_v,
    H, T, K, V, NC,
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
    bnd_base = bnd_ptr + pid_bh * NC * K * V

    S = tl.zeros((BV, BK), dtype=tl.float32)
    for t in range(T):
        if t % 16 == 0:
            tl.store(
                bnd_base + (t // 16) * K * V + offs_k[:, None] * V + offs_v[None, :],
                tl.trans(S), mask=mask_k[:, None] & mask_v[None, :],
            )
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


@triton.jit
def _backward_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr,
    go_ptr, bnd_ptr, scr_ptr,
    dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, db_ptr, dgm_ptr,
    sq_b, sq_h, sq_t, sq_k,
    sk_b, sk_h, sk_t, sk_k,
    sv_b, sv_h, sv_t, sv_v,
    se_b, se_h, se_t, se_k,
    sg_b, sg_h, sg_t, sg_k,
    sb_b, sb_h, sb_t,
    sgm_b, sgm_h, sgm_t,
    sgo_b, sgo_h, sgo_t, sgo_v,
    H, T, K, V, NC, NV,
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
    offs_vl = tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_kv = mask_k[:, None] & mask_v[None, :]

    q_base = q_ptr + b * sq_b + h * sq_h
    k_base = k_ptr + b * sk_b + h * sk_h
    v_base = v_ptr + b * sv_b + h * sv_h
    e_base = e_ptr + b * se_b + h * se_h
    g_base = g_ptr + b * sg_b + h * sg_h
    beta_base = beta_ptr + b * sb_b + h * sb_h
    gamma_base = gamma_ptr + b * sgm_b + h * sgm_h
    go_base = go_ptr + b * sgo_b + h * sgo_h

    bnd_base = bnd_ptr + pid_bh * NC * K * V
    scr_base = scr_ptr + (pid_bh * NV + pid_v) * BC * BK * BV
    kt_base = pid_bh * T * K
    vt_base = pid_bh * T * V
    tt_base = pid_bh * T

    offs_s = offs_m // 2
    is_e = (offs_m % 2) == 1
    is_w = (offs_m % 2) == 0
    strict_lower = offs_m[:, None] > offs_m[None, :]
    out_mask = offs_m[:, None] <= (2 * offs_c[None, :] + 1)
    pair_mat = tl.where(
        offs_c[:, None] == offs_s[None, :], 1.0, 0.0
    ).to(tl.float32)
    suffix_mat = tl.where(
        offs_c[None, :] >= offs_c[:, None], 1.0, 0.0
    ).to(tl.float32)

    # Adjoint dL/dS_t for the [K, BV] state block, carried backwards
    # chunk by chunk.  Unlike the previous version, a normal chunk is
    # differentiated with transposed tensor-core matmuls; only a chunk
    # whose cumulative decay would overflow the scaled coordinates
    # falls back to a per-step adjoint (using the scratch buffer).
    Adj = tl.zeros((BK, BV), dtype=tl.float32)

    for crev in range(NC):
        chunk = NC - 1 - crev
        start = chunk * BC

        S_in = tl.load(
            bnd_base + chunk * K * V + offs_k[:, None] * V + offs_v[None, :],
            mask=mask_kv, other=0.0,
        ).to(tl.float32)

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

        g_elem = tl.where(is_e[:, None], 0.0, g_m_load)
        G_m = tl.cumsum(g_elem, axis=0)
        G_last = tl.sum(
            tl.where((offs_m == BM - 1)[:, None], G_m, 0.0), axis=0
        )
        log_mid = 0.5 * G_last
        dg_m = G_m - log_mid[None, :]
        need_seq = tl.max(tl.abs(dg_m)) > 40.0

        if need_seq:
            # ---- per-step adjoint for this chunk ---------------------
            Sseq = S_in
            for i in range(BC):
                t = start + i
                tl.store(
                    scr_base + i * BK * BV + offs_k[:, None] * BV + offs_vl[None, :],
                    Sseq, mask=mask_kv,
                )
                if t < T:
                    gv = tl.load(g_base + t * sg_t + offs_k * sg_k,
                                 mask=mask_k, other=0.0).to(tl.float32)
                    Sseq = Sseq * tl.exp(gv)[:, None]
                    kv = tl.load(k_base + t * sk_t + offs_k * sk_k,
                                 mask=mask_k, other=0.0).to(tl.float32)
                    vv = tl.load(v_base + t * sv_t + offs_v * sv_v,
                                 mask=mask_v, other=0.0).to(tl.float32)
                    beta_t = tl.load(beta_base + t * sb_t).to(tl.float32)
                    pred = tl.sum(Sseq * kv[:, None], axis=0)
                    Sseq = Sseq + kv[:, None] * (beta_t * (vv - pred))[None, :]
                    ev = tl.load(e_base + t * se_t + offs_k * se_k,
                                 mask=mask_k, other=0.0).to(tl.float32)
                    gamma_t = tl.load(gamma_base + t * sgm_t).to(tl.float32)
                    proj = tl.sum(Sseq * ev[:, None], axis=0)
                    Sseq = Sseq - ev[:, None] * (gamma_t * proj)[None, :]

            for ii in range(BC):
                i = BC - 1 - ii
                t = start + i
                if t < T:
                    S_prev = tl.load(
                        scr_base + i * BK * BV + offs_k[:, None] * BV + offs_vl[None, :],
                        mask=mask_kv, other=0.0,
                    ).to(tl.float32)
                    gv = tl.load(g_base + t * sg_t + offs_k * sg_k,
                                 mask=mask_k, other=0.0).to(tl.float32)
                    decay = tl.exp(gv)
                    Sp = S_prev * decay[:, None]

                    kv = tl.load(k_base + t * sk_t + offs_k * sk_k,
                                 mask=mask_k, other=0.0).to(tl.float32)
                    vv = tl.load(v_base + t * sv_t + offs_v * sv_v,
                                 mask=mask_v, other=0.0).to(tl.float32)
                    beta_t = tl.load(beta_base + t * sb_t).to(tl.float32)
                    pred = tl.sum(Sp * kv[:, None], axis=0)
                    delta = vv - pred
                    Spp = Sp + kv[:, None] * (beta_t * delta)[None, :]

                    ev = tl.load(e_base + t * se_t + offs_k * se_k,
                                 mask=mask_k, other=0.0).to(tl.float32)
                    gamma_t = tl.load(gamma_base + t * sgm_t).to(tl.float32)
                    proj = tl.sum(Spp * ev[:, None], axis=0)
                    S_t = Spp - ev[:, None] * (gamma_t * proj)[None, :]

                    qv = tl.load(q_base + t * sq_t + offs_k * sq_k,
                                 mask=mask_k, other=0.0).to(tl.float32)
                    go = tl.load(go_base + t * sgo_t + offs_v * sgo_v,
                                 mask=mask_v, other=0.0).to(tl.float32)

                    GS = Adj + (scale * qv)[:, None] * go[None, :]
                    dq_vec = scale * tl.sum(S_t * go[None, :], axis=1)
                    tl.atomic_add(dq_ptr + kt_base + t * K + offs_k, dq_vec, mask=mask_k)

                    eGS = tl.sum(GS * ev[:, None], axis=0)
                    Hm = GS - ev[:, None] * (gamma_t * eGS)[None, :]
                    tl.atomic_add(dgm_ptr + tt_base + t, -tl.sum(eGS * proj))
                    de_vec = -gamma_t * (
                        tl.sum(GS * proj[None, :], axis=1)
                        + tl.sum(Spp * eGS[None, :], axis=1)
                    )
                    tl.atomic_add(de_ptr + kt_base + t * K + offs_k, de_vec, mask=mask_k)

                    kTH = tl.sum(Hm * kv[:, None], axis=0)
                    Pm = Hm - kv[:, None] * (beta_t * kTH)[None, :]
                    tl.store(dv_ptr + vt_base + t * V + offs_v, beta_t * kTH, mask=mask_v)
                    tl.atomic_add(db_ptr + tt_base + t, tl.sum(kTH * delta))
                    dk_vec = beta_t * (
                        tl.sum(Hm * delta[None, :], axis=1)
                        - tl.sum(Sp * kTH[None, :], axis=1)
                    )
                    tl.atomic_add(dk_ptr + kt_base + t * K + offs_k, dk_vec, mask=mask_k)

                    dg_vec = tl.sum(Pm * Sp, axis=1)
                    tl.atomic_add(dg_ptr + kt_base + t * K + offs_k, dg_vec, mask=mask_k)

                    Adj = Pm * decay[:, None]
        else:
            # ---- chunkwise transposed adjoint ------------------------
            sc_m = tl.exp(dg_m)
            inv_sc_m = tl.exp(-dg_m)
            d_vec = tl.exp(log_mid)

            orig_m = tl.where(is_e[:, None], e_m, k_m)
            coeff_m = tl.where(is_e, gamma_m, beta_m)
            W = orig_m * sc_m
            U = orig_m * inv_sc_m

            S_hat = S_in * d_vec[:, None]

            Araw = tl.dot(W, tl.trans(U))
            A_mat = tl.where(
                strict_lower, coeff_m[:, None] * Araw, 0.0
            )

            v_m = tl.load(
                v_base + t_elem[:, None] * sv_t + offs_v[None, :] * sv_v,
                mask=mask_mv, other=0.0,
            ).to(tl.float32)
            val_m = tl.where(is_e[:, None], 0.0, v_m)
            Bm = val_m - tl.dot(W, S_hat)
            Y = coeff_m[:, None] * Bm

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

            S_end_hat = S_hat + tl.dot(tl.trans(U), R)

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
            E_q = tl.exp(G_c - log_mid[None, :])
            Qh = q_c * E_q
            go_c = tl.load(
                go_base + t_c[:, None] * sgo_t + offs_v[None, :] * sgo_v,
                mask=mask_c[:, None] & mask_v[None, :], other=0.0,
            ).to(tl.float32)
            Gpre = go_c * scale

            # Output + final-state adjoints.
            GSeh = Adj * d_vec[:, None]
            gd_sout = tl.sum(Adj * S_end_hat, axis=1)
            dR = tl.dot(U, GSeh)
            dShat = GSeh

            Mq = tl.where(out_mask, tl.dot(U, tl.trans(Qh)), 0.0)
            dMq = tl.where(out_mask, tl.dot(R, tl.trans(Gpre)), 0.0)
            dU = tl.dot(R, tl.trans(GSeh)) + tl.dot(dMq, Qh)
            dQh = tl.dot(Gpre, tl.trans(S_hat)) + tl.dot(tl.trans(dMq), U)
            dShat = dShat + tl.dot(tl.trans(Qh), Gpre)
            dR = dR + tl.dot(Mq, Gpre)

            # Z = (I + A)^{-T} dR by Neumann doubling on P^T.
            Pt = tl.trans(P)
            Z = dR + tl.dot(Pt, dR)
            Qt = tl.dot(Pt, Pt)
            Z = Z + tl.dot(Qt, Z)
            Qt = tl.dot(Qt, Qt)
            Z = Z + tl.dot(Qt, Z)
            Qt = tl.dot(Qt, Qt)
            Z = Z + tl.dot(Qt, Z)
            Qt = tl.dot(Qt, Qt)
            Z = Z + tl.dot(Qt, Z)

            dA_adj = -tl.dot(Z, tl.trans(R))
            dA_masked = tl.where(strict_lower, dA_adj, 0.0)
            dAraw = coeff_m[:, None] * dA_masked
            dcoeff = tl.sum(dA_masked * Araw, axis=1)

            dBm = coeff_m[:, None] * Z
            dcoeff = dcoeff + tl.sum(Z * Bm, axis=1)
            dW = tl.dot(dAraw, U) - tl.dot(dBm, tl.trans(S_hat))
            dU = dU + tl.dot(tl.trans(dAraw), W)
            dShat = dShat - tl.dot(tl.trans(W), dBm)

            Adj = dShat * d_vec[:, None]
            gd_vec = gd_sout + tl.sum(dShat * S_in, axis=1)

            # Decay-scaling adjoints -> g, and orig/coeff/value grads.
            dorig = dW * sc_m + dU * inv_sc_m
            ddg = (dW * orig_m) * sc_m - (dU * orig_m) * inv_sc_m
            dq_mat = dQh * E_q
            dArg_q = dQh * Qh
            dlogmid = gd_vec * d_vec - tl.sum(ddg, axis=0) - tl.sum(dArg_q, axis=0)
            dGc = dArg_q + tl.dot(pair_mat, ddg)
            dg_mat = tl.dot(suffix_mat, dGc) + 0.5 * dlogmid[None, :]

            tl.atomic_add(
                dq_ptr + kt_base + t_c[:, None] * K + offs_k[None, :],
                dq_mat, mask=mask_c[:, None] & mask_k[None, :],
            )
            tl.atomic_add(
                dg_ptr + kt_base + t_c[:, None] * K + offs_k[None, :],
                dg_mat, mask=mask_c[:, None] & mask_k[None, :],
            )
            tl.atomic_add(
                dk_ptr + kt_base + t_elem[:, None] * K + offs_k[None, :],
                dorig, mask=is_w[:, None] & mask_mk,
            )
            tl.atomic_add(
                de_ptr + kt_base + t_elem[:, None] * K + offs_k[None, :],
                dorig, mask=is_e[:, None] & mask_mk,
            )
            tl.store(
                dv_ptr + vt_base + t_elem[:, None] * V + offs_v[None, :],
                dBm, mask=is_w[:, None] & mask_mv,
            )
            tl.atomic_add(
                db_ptr + tt_base + t_elem, dcoeff, mask=is_w & mask_elem
            )
            tl.atomic_add(
                dgm_ptr + tt_base + t_elem, dcoeff, mask=is_e & mask_elem
            )


def _run_forward(q, k, v, e, g, beta, gamma, out, bnd, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    NC = bnd.shape[2]
    strides = (
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        e.stride(0), e.stride(1), e.stride(2), e.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    if K <= 128 and T >= 32:
        BC = 16
        BM = 32
        BK = max(16, triton.next_power_of_2(K))
        BV = 64 if V >= 64 else max(16, triton.next_power_of_2(V))
        grid = (B * H, (V + BV - 1) // BV)
        _chunk_kernel[grid](
            q, k, v, e, g, beta, gamma, out, bnd,
            *strides, H, T, K, V, NC, float(scale),
            BC=BC, BM=BM, BK=BK, BV=BV, num_warps=8,
        )
    else:
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
            q, k, v, e, g, beta, gamma, out, bnd,
            *strides, H, T, K, V, NC, float(scale),
            BK=BK, BV=BV, num_warps=num_warps,
        )


def _run_backward(q, k, v, e, g, beta, gamma, go, bnd, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    dev = q.device
    dq = torch.zeros(q.shape, device=dev, dtype=torch.float32)
    dk = torch.zeros(k.shape, device=dev, dtype=torch.float32)
    de = torch.zeros(e.shape, device=dev, dtype=torch.float32)
    dg = torch.zeros(g.shape, device=dev, dtype=torch.float32)
    dv = torch.zeros(v.shape, device=dev, dtype=torch.float32)
    db = torch.zeros(beta.shape, device=dev, dtype=torch.float32)
    dgm = torch.zeros(gamma.shape, device=dev, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0 or V == 0 or K == 0:
        return dq, dk, dv, de, dg, db, dgm

    BK = max(16, triton.next_power_of_2(max(1, K)))
    if BK <= 128:
        BV = 64
    elif BK <= 256:
        BV = 32
    else:
        BV = 16
    BV = min(BV, max(16, triton.next_power_of_2(V)))
    NV = (V + BV - 1) // BV
    NC = bnd.shape[2]
    BC = 16
    BM = 32
    scr = torch.empty((B * H * NV * BC * BK * BV,), device=dev, dtype=torch.float32)

    grid = (B * H, NV)
    _backward_kernel[grid](
        q, k, v, e, g, beta, gamma,
        go, bnd, scr,
        dq, dk, dv, de, dg, db, dgm,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        e.stride(0), e.stride(1), e.stride(2), e.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        go.stride(0), go.stride(1), go.stride(2), go.stride(3),
        H, T, K, V, NC, NV, float(scale),
        BC=BC, BM=BM, BK=BK, BV=BV,
        num_warps=8,
    )
    return dq, dk, dv, de, dg, db, dgm


class _LayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        NC = max(1, (T + 15) // 16)
        bnd = torch.empty((B, H, NC, K, V), device=q.device, dtype=torch.float32)
        if B > 0 and H > 0 and T > 0 and V > 0 and K > 0:
            _run_forward(q, k, v, e, g, beta, gamma, out, bnd, scale)
        ctx.save_for_backward(q, k, v, e, g, beta, gamma, bnd)
        ctx.scale_val = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma, bnd = ctx.saved_tensors
        dq, dk, dv, de, dg, db, dgm = _run_backward(
            q, k, v, e, g, beta, gamma, grad_out, bnd, ctx.scale_val
        )
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            de.to(e.dtype),
            dg.to(g.dtype),
            db.to(beta.dtype),
            dgm.to(gamma.dtype),
            None,
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if isinstance(scale, torch.Tensor):
        scale = float(scale.detach().item())
    return _LayerFn.apply(q, k, v, e, g, beta, gamma, float(scale))
```

The forward is unchanged, including its chunk-boundary state saves, strong-decay sequential chunks, and general fallback. The backward is now chunkwise rather than step-by-step: it walks chunks in reverse and differentiates the chunk's scaled delta-rule formulation with transposed tensor-core matmuls, including a Neumann-doubling solve for `(I+A)^{-T}`, the output/state adjoints, and the decay-scaling adjoints propagated back to `g` via pair-sum and suffix-sum matmuls. Only chunks whose cumulative decay would overflow the scaled coordinates (the `decay_strong` case) use the previous exact per-step adjoint, staged through a small per-program scratch buffer. All accumulation remains fp32, with casts to each input's dtype only at the autograd boundary.