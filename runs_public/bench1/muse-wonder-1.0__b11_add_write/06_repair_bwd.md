```python
import torch
import triton
import triton.language as tl

_CB = 16


def _cfg(K, V):
    BLOCK_K = triton.next_power_of_2(K)
    v_pow2 = triton.next_power_of_2(V)
    BLOCK_V = 16
    while BLOCK_V < v_pow2 and BLOCK_K * BLOCK_V < 4096:
        BLOCK_V *= 2
    tile = BLOCK_K * BLOCK_V
    if tile >= 8192:
        warps = 8
    elif tile >= 4096:
        warps = 4
    elif tile >= 2048:
        warps = 2
    else:
        warps = 1
    return BLOCK_K, BLOCK_V, warps


@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    bnd_ptr,
    H, T, NUM_CHUNKS, BND_STRIDE_BH,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    seb, seh, set_, sek,
    sgb, sgh, sgt, sgk,
    sbb, sbh, sbt,
    scb, sch, sct,
    sob, soh, sot, sov,
    scale,
    KDIM, VDIM,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    CB: tl.constexpr, SAVE_BWD: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < KDIM
    mask_v = offs_v < VDIM

    q_ptrs = q_ptr + b * sqb + h * sqh + offs_k * sqk
    k_ptrs = k_ptr + b * skb + h * skh + offs_k * skk
    v_ptrs = v_ptr + b * svb + h * svh + offs_v * svv
    e_ptrs = e_ptr + b * seb + h * seh + offs_k * sek
    g_ptrs = g_ptr + b * sgb + h * sgh + offs_k * sgk
    out_ptrs = out_ptr + b * sob + h * soh + offs_v * sov
    beta_pos = beta_ptr + b * sbb + h * sbh
    gamma_pos = gamma_ptr + b * scb + h * sch

    bnd_ptrs = bnd_ptr + pid_bh.to(tl.int64) * BND_STRIDE_BH \
        + offs_k[:, None] * VDIM + offs_v[None, :]
    mask_kv = mask_k[:, None] & mask_v[None, :]

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in range(T):
        if SAVE_BWD:
            if t % CB == 0:
                tl.store(bnd_ptrs + (t // CB) * KDIM * VDIM, S, mask=mask_kv)

        kt = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        gt = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        vt = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
        et = tl.load(e_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        qt = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        bt = tl.load(beta_pos).to(tl.float32)
        ct = tl.load(gamma_pos).to(tl.float32)

        S = S * tl.exp(gt)[:, None]
        pred = tl.sum(S * kt[:, None], axis=0)
        S = S + kt[:, None] * (bt * (vt - pred))[None, :]
        S = S + et[:, None] * (ct * vt)[None, :]

        ot = tl.sum(S * (qt * scale)[:, None], axis=0)
        tl.store(out_ptrs, ot, mask=mask_v)

        q_ptrs += sqt
        k_ptrs += skt
        v_ptrs += svt
        e_ptrs += set_
        g_ptrs += sgt
        out_ptrs += sot
        beta_pos += sbt
        gamma_pos += sct

    if SAVE_BWD:
        tl.store(bnd_ptrs + NUM_CHUNKS * KDIM * VDIM, S, mask=mask_kv)


@triton.jit
def _bwd_chunk_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, go_ptr,
    bnd_ptr, scr_ptr,
    dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
    H, NUM_CHUNKS, PART_K_SIZE, PART_S_SIZE,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    seb, seh, set_, sek,
    sgb, sgh, sgt, sgk,
    sbb, sbh, sbt,
    scb, sch, sct,
    sgob, sgoh, sgot, sgov,
    scale,
    T, KDIM, VDIM,
    BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, BLOCK_C)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < KDIM
    mask_v = offs_v < VDIM
    mask_kv = mask_k[:, None] & mask_v[None, :]

    q_b = q_ptr + b * sqb + h * sqh
    k_b = k_ptr + b * skb + h * skh
    v_b = v_ptr + b * svb + h * svh
    e_b = e_ptr + b * seb + h * seh
    g_b = g_ptr + b * sgb + h * sgh
    go_b = go_ptr + b * sgob + h * sgoh
    beta_b = beta_ptr + b * sbb + h * sbh
    gamma_b = gamma_ptr + b * scb + h * sch

    KV: tl.constexpr = KDIM * VDIM
    bnd_b = bnd_ptr + pid_bh.to(tl.int64) * (NUM_CHUNKS + 1) * KV \
        + offs_k[:, None] * VDIM + offs_v[None, :]
    scr_b = scr_ptr + pid_bh.to(tl.int64) * BLOCK_C * KV \
        + offs_k[:, None] * VDIM + offs_v[None, :]

    # Per-value-block gradient partials: no atomics.
    part_k = pid_v.to(tl.int64) * PART_K_SIZE + pid_bh.to(tl.int64) * T * KDIM
    part_s = pid_v.to(tl.int64) * PART_S_SIZE + pid_bh.to(tl.int64) * T
    dv_b = dv_ptr + pid_bh.to(tl.int64) * T * VDIM

    row_i = offs_c[:, None]
    col_i = offs_c[None, :]
    strict_lower = row_i > col_i
    lower_incl = row_i >= col_i
    strict_upper = row_i < col_i
    upper_incl = row_i <= col_i
    is_mid = offs_c == (BLOCK_C // 2)
    is_last = offs_c == (BLOCK_C - 1)

    # Adjoint w.r.t. the original (untransformed) state at chunk end.
    lam = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for chunk_rev in range(NUM_CHUNKS):
        chunk = NUM_CHUNKS - 1 - chunk_rev
        t_start = chunk * BLOCK_C
        mask_c = (t_start + offs_c) < T
        mask_ck = mask_c[:, None] & mask_k[None, :]
        mask_cv = mask_c[:, None] & mask_v[None, :]

        g_mat = tl.load(
            g_b + (t_start + offs_c[:, None]) * sgt + offs_k[None, :] * sgk,
            mask=mask_ck, other=0.0).to(tl.float32)
        Gc = tl.cumsum(g_mat, axis=0)
        gc_hi = tl.max(Gc, axis=0)
        gc_lo = tl.min(Gc, axis=0)
        range_all = tl.max(
            tl.maximum(gc_hi, 0.0) - tl.minimum(gc_lo, 0.0), axis=0)

        if range_all > 50.0:
            # Exact per-step adjoint for extreme-decay chunks.  Y_t is
            # recomputed from the checkpoint into private scratch.
            S_cur = tl.load(bnd_b + chunk * KV,
                            mask=mask_kv, other=0.0).to(tl.float32)
            for step in range(BLOCK_C):
                t = t_start + step
                valid = t < T
                kt = tl.load(k_b + t * skt + offs_k * skk,
                             mask=mask_k & valid, other=0.0).to(tl.float32)
                vt = tl.load(v_b + t * svt + offs_v * svv,
                             mask=mask_v & valid, other=0.0).to(tl.float32)
                et = tl.load(e_b + t * set_ + offs_k * sek,
                             mask=mask_k & valid, other=0.0).to(tl.float32)
                gt = tl.load(g_b + t * sgt + offs_k * sgk,
                             mask=mask_k & valid, other=0.0).to(tl.float32)
                bt = tl.load(beta_b + t * sbt,
                             mask=valid, other=0.0).to(tl.float32)
                ct = tl.load(gamma_b + t * sct,
                             mask=valid, other=0.0).to(tl.float32)
                Y = S_cur * tl.exp(gt)[:, None]
                tl.store(scr_b + step * KV, Y, mask=mask_kv)
                pred = tl.sum(Y * kt[:, None], axis=0)
                S_cur = Y + kt[:, None] * (bt * (vt - pred))[None, :] \
                    + et[:, None] * (ct * vt)[None, :]

            tl.debug_barrier()

            for rev_step in range(BLOCK_C):
                step = BLOCK_C - 1 - rev_step
                t = t_start + step
                valid = t < T
                qt = tl.load(q_b + t * sqt + offs_k * sqk,
                             mask=mask_k & valid, other=0.0).to(tl.float32)
                kt = tl.load(k_b + t * skt + offs_k * skk,
                             mask=mask_k & valid, other=0.0).to(tl.float32)
                vt = tl.load(v_b + t * svt + offs_v * svv,
                             mask=mask_v & valid, other=0.0).to(tl.float32)
                et = tl.load(e_b + t * set_ + offs_k * sek,
                             mask=mask_k & valid, other=0.0).to(tl.float32)
                gt = tl.load(g_b + t * sgt + offs_k * sgk,
                             mask=mask_k & valid, other=0.0).to(tl.float32)
                got = tl.load(go_b + t * sgot + offs_v * sgov,
                              mask=mask_v & valid, other=0.0).to(tl.float32)
                bt = tl.load(beta_b + t * sbt,
                             mask=valid, other=0.0).to(tl.float32)
                ct = tl.load(gamma_b + t * sct,
                             mask=valid, other=0.0).to(tl.float32)
                Y = tl.load(scr_b + step * KV,
                            mask=mask_kv, other=0.0).to(tl.float32)

                lam = lam + (qt * scale)[:, None] * got[None, :]
                lam_k = tl.sum(lam * kt[:, None], axis=0)
                lam_e = tl.sum(lam * et[:, None], axis=0)
                lam_v = tl.sum(lam * vt[None, :], axis=1)
                pred = tl.sum(Y * kt[:, None], axis=0)
                delta = vt - pred
                lam_delta = tl.sum(lam * delta[None, :], axis=1)
                y_lamk = tl.sum(Y * lam_k[None, :], axis=1)

                gk_vec = bt * (lam_delta - y_lamk)
                grad_y = lam - (bt * kt)[:, None] * lam_k[None, :]
                gg_vec = tl.sum(grad_y * Y, axis=1)
                ge_vec = ct * lam_v
                gv_vec = bt * lam_k + ct * lam_e
                gbeta_val = tl.sum(lam_k * delta)
                ggamma_val = tl.sum(lam_v * et)
                s_go = tl.sum(Y * got[None, :], axis=1) \
                    + bt * kt * tl.sum(delta * got) \
                    + ct * et * tl.sum(vt * got)
                gq_vec = scale * s_go

                lam = tl.exp(gt)[:, None] * grad_y

                tl.store(dv_b + t * VDIM + offs_v, gv_vec,
                         mask=mask_v & valid)
                tl.store(dq_ptr + part_k + t * KDIM + offs_k, gq_vec,
                         mask=mask_k & valid)
                tl.store(dk_ptr + part_k + t * KDIM + offs_k, gk_vec,
                         mask=mask_k & valid)
                tl.store(de_ptr + part_k + t * KDIM + offs_k, ge_vec,
                         mask=mask_k & valid)
                tl.store(dg_ptr + part_k + t * KDIM + offs_k, gg_vec,
                         mask=mask_k & valid)
                tl.store(dbeta_ptr + part_s + t, gbeta_val, mask=valid)
                tl.store(dgamma_ptr + part_s + t, ggamma_val, mask=valid)

            tl.debug_barrier()
        else:
            q_mat = tl.load(
                q_b + (t_start + offs_c[:, None]) * sqt + offs_k[None, :] * sqk,
                mask=mask_ck, other=0.0).to(tl.float32)
            k_mat = tl.load(
                k_b + (t_start + offs_c[:, None]) * skt + offs_k[None, :] * skk,
                mask=mask_ck, other=0.0).to(tl.float32)
            v_mat = tl.load(
                v_b + (t_start + offs_c[:, None]) * svt + offs_v[None, :] * svv,
                mask=mask_cv, other=0.0).to(tl.float32)
            e_mat = tl.load(
                e_b + (t_start + offs_c[:, None]) * set_ + offs_k[None, :] * sek,
                mask=mask_ck, other=0.0).to(tl.float32)
            go_mat = tl.load(
                go_b + (t_start + offs_c[:, None]) * sgot + offs_v[None, :] * sgov,
                mask=mask_cv, other=0.0).to(tl.float32)
            beta = tl.load(beta_b + (t_start + offs_c) * sbt,
                           mask=mask_c, other=0.0).to(tl.float32)
            gamma = tl.load(gamma_b + (t_start + offs_c) * sct,
                            mask=mask_c, other=0.0).to(tl.float32)

            r_vec = tl.sum(tl.where(is_mid[:, None], Gc, 0.0), axis=0)
            diff = Gc - r_vec[None, :]
            exp_diff = tl.exp(diff)
            exp_ndiff = tl.exp(-diff)

            u_mat = k_mat * exp_diff            # prediction side
            w_mat = k_mat * exp_ndiff           # write side
            ew_mat = e_mat * exp_ndiff
            qp_mat = q_mat * exp_diff * scale

            S0c = tl.load(bnd_b + chunk * KV,
                          mask=mask_kv, other=0.0).to(tl.float32) \
                * tl.exp(r_vec)[:, None]

            # ---- forward solve inside the chunk (transformed) ----
            a_mat = tl.dot(u_mat, tl.trans(w_mat))
            a_mat = tl.where(strict_lower, a_mat, 0.0)
            b_mat = tl.dot(u_mat, tl.trans(ew_mat))
            b_mat = tl.where(strict_lower, b_mat, 0.0)
            z_mat = gamma[:, None] * v_mat
            inter_pred = tl.dot(u_mat, S0c)
            r_mat = v_mat - inter_pred - tl.dot(b_mat, z_mat)
            rhs = beta[:, None] * r_mat

            nf = -beta[:, None] * a_mat
            d_mat = rhs
            d_mat = d_mat + tl.dot(nf, d_mat)
            if BLOCK_C > 2:
                nf2 = tl.dot(nf, nf)
                d_mat = d_mat + tl.dot(nf2, d_mat)
                if BLOCK_C > 4:
                    nf4 = tl.dot(nf2, nf2)
                    d_mat = d_mat + tl.dot(nf4, d_mat)
                    if BLOCK_C > 8:
                        nf8 = tl.dot(nf4, nf4)
                        d_mat = d_mat + tl.dot(nf8, d_mat)

            delta_mat = r_mat - tl.dot(a_mat, d_mat)
            s_end_c = S0c + tl.dot(tl.trans(w_mat), d_mat) \
                + tl.dot(tl.trans(ew_mat), z_mat)

            # ---- adjoint in transformed coordinates ----
            exp_last = tl.sum(
                tl.where(is_last[:, None], exp_diff, 0.0), axis=0)
            lam_c = lam * exp_last[:, None]     # adjoint w.r.t. S_end_c
            lam_t = tl.trans(lam_c)

            cqp_w = tl.dot(w_mat, tl.trans(qp_mat))
            cqp_w = tl.where(upper_incl, cqp_w, 0.0)
            cqp_e = tl.dot(ew_mat, tl.trans(qp_mat))
            cqp_e = tl.where(upper_incl, cqp_e, 0.0)

            # aw_i = w_i^T L_i solves an upper-triangular system.
            rhs_aw = tl.dot(w_mat, lam_c) + tl.dot(cqp_w, go_mat)
            na = -tl.trans(a_mat) * beta[None, :]
            na = tl.where(strict_upper, na, 0.0)
            aw = rhs_aw
            aw = aw + tl.dot(na, aw)
            if BLOCK_C > 2:
                na2 = tl.dot(na, na)
                aw = aw + tl.dot(na2, aw)
                if BLOCK_C > 4:
                    na4 = tl.dot(na2, na2)
                    aw = aw + tl.dot(na4, aw)
                    if BLOCK_C > 8:
                        na8 = tl.dot(na4, na4)
                        aw = aw + tl.dot(na8, aw)

            coef_e = tl.trans(b_mat) * beta[None, :]
            coef_e = tl.where(strict_upper, coef_e, 0.0)
            ae = tl.dot(ew_mat, lam_c) + tl.dot(cqp_e, go_mat) \
                - tl.dot(coef_e, aw)

            # gw_i = L_i d_i ; ge_i = L_i z_i
            gw = tl.dot(d_mat, lam_t)
            m_ = tl.trans(tl.dot(go_mat, tl.trans(d_mat)))
            m_ = tl.where(upper_incl, m_, 0.0)
            gw = gw + tl.dot(m_, qp_mat)
            m_ = tl.trans(tl.dot(aw, tl.trans(d_mat)) * beta[:, None])
            m_ = tl.where(strict_upper, m_, 0.0)
            gw = gw - tl.dot(m_, u_mat)

            ge = tl.dot(z_mat, lam_t)
            m_ = tl.trans(tl.dot(go_mat, tl.trans(z_mat)))
            m_ = tl.where(upper_incl, m_, 0.0)
            ge = ge + tl.dot(m_, qp_mat)
            m_ = tl.trans(tl.dot(aw, tl.trans(z_mat)) * beta[:, None])
            m_ = tl.where(strict_upper, m_, 0.0)
            ge = ge - tl.dot(m_, u_mat)

            # gu_i = -beta_i * S_{i-1} aw_i
            gu = tl.dot(aw, tl.trans(S0c))
            m_ = tl.trans(tl.dot(d_mat, tl.trans(aw)))
            m_ = tl.where(strict_lower, m_, 0.0)
            gu = gu + tl.dot(m_, w_mat)
            m_ = tl.trans(tl.dot(z_mat, tl.trans(aw)))
            m_ = tl.where(strict_lower, m_, 0.0)
            gu = gu + tl.dot(m_, ew_mat)
            gu = -beta[:, None] * gu

            # gqp_i = S_i go_i
            gqp = tl.dot(go_mat, tl.trans(S0c))
            m_ = tl.trans(tl.dot(d_mat, tl.trans(go_mat)))
            m_ = tl.where(lower_incl, m_, 0.0)
            gqp = gqp + tl.dot(m_, w_mat)
            m_ = tl.trans(tl.dot(z_mat, tl.trans(go_mat)))
            m_ = tl.where(lower_incl, m_, 0.0)
            gqp = gqp + tl.dot(m_, ew_mat)

            # Adjoint to the (original) state at the chunk start.
            lam_start_c = lam_c + tl.dot(tl.trans(qp_mat), go_mat) \
                - tl.dot(tl.trans(u_mat), beta[:, None] * aw)
            lam = lam_start_c * tl.exp(r_vec)[:, None]

            # ---- map gradients back to the original inputs ----
            gq_mat = gqp * exp_diff * scale
            gk_mat = gu * exp_diff + gw * exp_ndiff
            ge_mat = ge * exp_ndiff
            gv_mat = beta[:, None] * aw + gamma[:, None] * ae
            gbeta_vec = tl.sum(aw * delta_mat, axis=1)
            ggamma_vec = tl.sum(ae * v_mat, axis=1)

            gdif = gu * u_mat - gw * w_mat - ge * ew_mat + gqp * qp_mat
            h_vec = tl.sum(lam_c * s_end_c, axis=1)
            gg_mat = tl.cumsum(gdif, axis=0, reverse=True) + h_vec[None, :]

            tl.store(dv_b + (t_start + offs_c[:, None]) * VDIM
                     + offs_v[None, :], gv_mat, mask=mask_cv)
            tl.store(dq_ptr + part_k + (t_start + offs_c[:, None]) * KDIM
                     + offs_k[None, :], gq_mat, mask=mask_ck)
            tl.store(dk_ptr + part_k + (t_start + offs_c[:, None]) * KDIM
                     + offs_k[None, :], gk_mat, mask=mask_ck)
            tl.store(de_ptr + part_k + (t_start + offs_c[:, None]) * KDIM
                     + offs_k[None, :], ge_mat, mask=mask_ck)
            tl.store(dg_ptr + part_k + (t_start + offs_c[:, None]) * KDIM
                     + offs_k[None, :], gg_mat, mask=mask_ck)
            tl.store(dbeta_ptr + part_s + t_start + offs_c, gbeta_vec,
                     mask=mask_c)
            tl.store(dgamma_ptr + part_s + t_start + offs_c, ggamma_vec,
                     mask=mask_c)


@triton.jit
def _bwd_recurrent_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, go_ptr,
    bnd_ptr, scr_ptr,
    dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
    H, NUM_CHUNKS,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    seb, seh, set_, sek,
    sgb, sgh, sgt, sgk,
    sbb, sbh, sbt,
    scb, sch, sct,
    sgob, sgoh, sgot, sgov,
    sdqb, sdqh, sdqt, sdqk,
    sdvb, sdvh, sdvt, sdvv,
    sdbb, sdbh, sdbt,
    scale,
    T, KDIM, VDIM,
    BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    # Exact per-step adjoint; used only for KDIM > 128.
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < KDIM
    mask_v = offs_v < VDIM
    mask_kv = mask_k[:, None] & mask_v[None, :]

    q_b = q_ptr + b * sqb + h * sqh
    k_b = k_ptr + b * skb + h * skh
    v_b = v_ptr + b * svb + h * svh
    e_b = e_ptr + b * seb + h * seh
    g_b = g_ptr + b * sgb + h * sgh
    go_b = go_ptr + b * sgob + h * sgoh
    beta_b = beta_ptr + b * sbb + h * sbh
    gamma_b = gamma_ptr + b * scb + h * sch

    dq_b = dq_ptr + b * sdqb + h * sdqh
    dk_b = dk_ptr + b * sdqb + h * sdqh
    de_b = de_ptr + b * sdqb + h * sdqh
    dg_b = dg_ptr + b * sdqb + h * sdqh
    dv_b = dv_ptr + b * sdvb + h * sdvh
    dbeta_b = dbeta_ptr + b * sdbb + h * sdbh
    dgamma_b = dgamma_ptr + b * sdbb + h * sdbh

    KV = KDIM * VDIM
    bnd_b = bnd_ptr + pid_bh.to(tl.int64) * (NUM_CHUNKS + 1) * KV \
        + offs_k[:, None] * VDIM + offs_v[None, :]
    scr_b = scr_ptr + pid_bh.to(tl.int64) * BLOCK_C * KV \
        + offs_k[:, None] * VDIM + offs_v[None, :]

    lam = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for chunk_rev in range(NUM_CHUNKS):
        chunk = NUM_CHUNKS - 1 - chunk_rev
        t_start = chunk * BLOCK_C

        S_cur = tl.load(bnd_b + chunk * KV,
                        mask=mask_kv, other=0.0).to(tl.float32)
        for step in range(BLOCK_C):
            t = t_start + step
            valid = t < T
            kt = tl.load(k_b + t * skt + offs_k * skk,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            vt = tl.load(v_b + t * svt + offs_v * svv,
                         mask=mask_v & valid, other=0.0).to(tl.float32)
            et = tl.load(e_b + t * set_ + offs_k * sek,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            gt = tl.load(g_b + t * sgt + offs_k * sgk,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            bt = tl.load(beta_b + t * sbt, mask=valid, other=0.0).to(tl.float32)
            ct = tl.load(gamma_b + t * sct, mask=valid, other=0.0).to(tl.float32)
            Y = S_cur * tl.exp(gt)[:, None]
            tl.store(scr_b + step * KV, Y, mask=mask_kv)
            pred = tl.sum(Y * kt[:, None], axis=0)
            S_cur = Y + kt[:, None] * (bt * (vt - pred))[None, :] \
                + et[:, None] * (ct * vt)[None, :]

        tl.debug_barrier()

        for rev_step in range(BLOCK_C):
            step = BLOCK_C - 1 - rev_step
            t = t_start + step
            valid = t < T
            qt = tl.load(q_b + t * sqt + offs_k * sqk,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            kt = tl.load(k_b + t * skt + offs_k * skk,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            vt = tl.load(v_b + t * svt + offs_v * svv,
                         mask=mask_v & valid, other=0.0).to(tl.float32)
            et = tl.load(e_b + t * set_ + offs_k * sek,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            gt = tl.load(g_b + t * sgt + offs_k * sgk,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            got = tl.load(go_b + t * sgot + offs_v * sgov,
                          mask=mask_v & valid, other=0.0).to(tl.float32)
            bt = tl.load(beta_b + t * sbt, mask=valid, other=0.0).to(tl.float32)
            ct = tl.load(gamma_b + t * sct, mask=valid, other=0.0).to(tl.float32)
            Y = tl.load(scr_b + step * KV, mask=mask_kv, other=0.0).to(tl.float32)

            lam = lam + (qt * scale)[:, None] * got[None, :]
            lam_k = tl.sum(lam * kt[:, None], axis=0)
            lam_e = tl.sum(lam * et[:, None], axis=0)
            lam_v = tl.sum(lam * vt[None, :], axis=1)
            pred = tl.sum(Y * kt[:, None], axis=0)
            delta = vt - pred
            lam_delta = tl.sum(lam * delta[None, :], axis=1)
            y_lamk = tl.sum(Y * lam_k[None, :], axis=1)

            gk_vec = bt * (lam_delta - y_lamk)
            grad_y = lam - (bt * kt)[:, None] * lam_k[None, :]
            gg_vec = tl.sum(grad_y * Y, axis=1)
            ge_vec = ct * lam_v
            gv_vec = bt * lam_k + ct * lam_e
            gbeta_val = tl.sum(lam_k * delta)
            ggamma_val = tl.sum(lam_v * et)
            s_go = tl.sum(Y * got[None, :], axis=1) \
                + bt * kt * tl.sum(delta * got) \
                + ct * et * tl.sum(vt * got)
            gq_vec = scale * s_go

            lam = tl.exp(gt)[:, None] * grad_y

            tl.store(dv_b + t * sdvt + offs_v * sdvv, gv_vec,
                     mask=mask_v & valid)
            tl.atomic_add(dq_b + t * sdqt + offs_k * sdqk, gq_vec,
                          mask=mask_k & valid)
            tl.atomic_add(dk_b + t * sdqt + offs_k * sdqk, gk_vec,
                          mask=mask_k & valid)
            tl.atomic_add(de_b + t * sdqt + offs_k * sdqk, ge_vec,
                          mask=mask_k & valid)
            tl.atomic_add(dg_b + t * sdqt + offs_k * sdqk, gg_vec,
                          mask=mask_k & valid)
            tl.atomic_add(dbeta_b + t * sdbt, gbeta_val, mask=valid)
            tl.atomic_add(dgamma_b + t * sdbt, ggamma_val, mask=valid)

        tl.debug_barrier()


def _forward_launch(q, k, v, e, g, beta, gamma, scale, want_bwd):
    B, H, T, K = q.shape
    V = v.shape[-1]
    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    BLOCK_K, BLOCK_V, warps = _cfg(K, V)
    num_chunks = (T + _CB - 1) // _CB
    if want_bwd:
        bnd = torch.empty((B, H, num_chunks + 1, K, V),
                          device=q.device, dtype=torch.float32)
    else:
        bnd = torch.empty(0, device=q.device, dtype=torch.float32)

    grid = (B * H, (V + BLOCK_V - 1) // BLOCK_V)
    _fwd_kernel[grid](
        q, k, v, e, g, beta, gamma, out, bnd,
        H, T, num_chunks, (num_chunks + 1) * K * V,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        e.stride(0), e.stride(1), e.stride(2), e.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        float(scale), K, V,
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, CB=_CB, SAVE_BWD=want_bwd,
        num_warps=warps, num_stages=2,
    )
    return out, bnd


class _Layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        ctx.scale = float(scale)
        ctx.empty_case = (B == 0 or H == 0 or T == 0 or K == 0 or V == 0)
        if ctx.empty_case:
            ctx.save_for_backward(q, k, v, e, g, beta, gamma)
            return torch.zeros((B, H, T, V), device=q.device,
                               dtype=torch.float32)

        want_bwd = any(x.requires_grad
                       for x in (q, k, v, e, g, beta, gamma))
        ctx.need_bwd = want_bwd
        out, bnd = _forward_launch(q, k, v, e, g, beta, gamma,
                                   scale, want_bwd)
        ctx.save_for_backward(q, k, v, e, g, beta, gamma, bnd)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.empty_case:
            q, k, v, e, g, beta, gamma = ctx.saved_tensors
            return (torch.zeros(q.shape, dtype=q.dtype, device=q.device),
                    torch.zeros(k.shape, dtype=k.dtype, device=k.device),
                    torch.zeros(v.shape, dtype=v.dtype, device=v.device),
                    torch.zeros(e.shape, dtype=e.dtype, device=e.device),
                    torch.zeros(g.shape, dtype=g.dtype, device=g.device),
                    torch.zeros(beta.shape, dtype=beta.dtype,
                                device=beta.device),
                    torch.zeros(gamma.shape, dtype=gamma.dtype,
                                device=gamma.device),
                    None)

        q, k, v, e, g, beta, gamma, bnd = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        dev = q.device
        fp = torch.float32

        if not ctx.need_bwd:
            return (torch.zeros(q.shape, dtype=q.dtype, device=dev),
                    torch.zeros(k.shape, dtype=k.dtype, device=dev),
                    torch.zeros(v.shape, dtype=v.dtype, device=dev),
                    torch.zeros(e.shape, dtype=e.dtype, device=dev),
                    torch.zeros(g.shape, dtype=g.dtype, device=dev),
                    torch.zeros(beta.shape, dtype=beta.dtype, device=dev),
                    torch.zeros(gamma.shape, dtype=gamma.dtype, device=dev),
                    None)

        num_chunks = (T + _CB - 1) // _CB

        if K > 128:
            dq = torch.zeros((B, H, T, K), device=dev, dtype=fp)
            dk = torch.zeros((B, H, T, K), device=dev, dtype=fp)
            de = torch.zeros((B, H, T, K), device=dev, dtype=fp)
            dg = torch.zeros((B, H, T, K), device=dev, dtype=fp)
            dv = torch.zeros((B, H, T, V), device=dev, dtype=fp)
            dbeta = torch.zeros((B, H, T), device=dev, dtype=fp)
            dgamma = torch.zeros((B, H, T), device=dev, dtype=fp)
            scratch = torch.empty(B * H * _CB * K * V, device=dev, dtype=fp)

            BLOCK_K, BLOCK_V, warps = _cfg(K, V)
            grid = (B * H, (V + BLOCK_V - 1) // BLOCK_V)
            _bwd_recurrent_kernel[grid](
                q, k, v, e, g, beta, gamma, grad_out, bnd, scratch,
                dq, dk, dv, de, dg, dbeta, dgamma,
                H, num_chunks,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                e.stride(0), e.stride(1), e.stride(2), e.stride(3),
                g.stride(0), g.stride(1), g.stride(2), g.stride(3),
                beta.stride(0), beta.stride(1), beta.stride(2),
                gamma.stride(0), gamma.stride(1), gamma.stride(2),
                grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
                grad_out.stride(3),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                dbeta.stride(0), dbeta.stride(1), dbeta.stride(2),
                ctx.scale, T, K, V,
                BLOCK_C=_CB, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
                num_warps=warps, num_stages=1,
            )
            return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                    de.to(e.dtype), dg.to(g.dtype), dbeta.to(beta.dtype),
                    dgamma.to(gamma.dtype), None)

        # Chunked tensor-core backward.
        BLOCK_K = max(16, triton.next_power_of_2(K))
        BLOCK_V = 32 if V >= 32 else 16
        num_v_blocks = (V + BLOCK_V - 1) // BLOCK_V
        grid = (B * H, num_v_blocks)

        dq_part = torch.zeros((num_v_blocks, B, H, T, K), device=dev, dtype=fp)
        dk_part = torch.zeros((num_v_blocks, B, H, T, K), device=dev, dtype=fp)
        de_part = torch.zeros((num_v_blocks, B, H, T, K), device=dev, dtype=fp)
        dg_part = torch.zeros((num_v_blocks, B, H, T, K), device=dev, dtype=fp)
        dv = torch.empty((B, H, T, V), device=dev, dtype=fp)
        dbeta_part = torch.zeros((num_v_blocks, B, H, T), device=dev, dtype=fp)
        dgamma_part = torch.zeros((num_v_blocks, B, H, T), device=dev, dtype=fp)
        scratch = torch.empty(B * H * _CB * K * V, device=dev, dtype=fp)

        _bwd_chunk_kernel[grid](
            q, k, v, e, g, beta, gamma, grad_out, bnd, scratch,
            dq_part, dk_part, dv, de_part, dg_part, dbeta_part, dgamma_part,
            H, num_chunks, B * H * T * K, B * H * T,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            e.stride(0), e.stride(1), e.stride(2), e.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            grad_out.stride(3),
            ctx.scale, T, K, V,
            BLOCK_C=_CB, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
            num_warps=8, num_stages=1,
        )

        dq = dq_part.sum(dim=0)
        dk = dk_part.sum(dim=0)
        de = de_part.sum(dim=0)
        dg = dg_part.sum(dim=0)
        dbeta = dbeta_part.sum(dim=0)
        dgamma = dgamma_part.sum(dim=0)

        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                de.to(e.dtype), dg.to(g.dtype), dbeta.to(beta.dtype),
                dgamma.to(gamma.dtype), None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if torch.is_grad_enabled() and any(
            x.requires_grad for x in (q, k, v, e, g, beta, gamma)):
        return _Layer.apply(q, k, v, e, g, beta, gamma, float(scale))

    B, H, T, K = q.shape
    V = v.shape[-1]
    if B == 0 or H == 0 or T == 0 or V == 0 or K == 0:
        return torch.zeros((B, H, T, V), device=q.device,
                           dtype=torch.float32)
    out, _ = _forward_launch(q, k, v, e, g, beta, gamma, scale, False)
    return out
```

The previous failure came from a PyTorch-loop backward with broadcasting errors and incorrect adjoint terms; that code is removed entirely. The restored forward remains the exact recurrent Triton kernel, while the new backward processes 16-step chunks with tensor-core triangular solves for both the delta values and the adjoint coupling, avoiding the old backward’s per-step global scratch traffic and atomic-heavy reduction (value-block partials are summed once at the end). Chunks whose cumulative decay range exceeds 50 nats, and shapes with $K>128$, fall back to the previously validated exact per-step adjoint so strong-decay and large-$K$ cases stay correct. I could not run this on hardware, so Triton compile details of the large chunk kernel are the main uncertainty.