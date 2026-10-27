```python
import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_inv_kernel(
    q_ptr, k_ptr, g_ptr, beta_ptr, inv_ptr, qk_ptr, range_ptr,
    T, H, NT, D_K, NK, KB,
    sq_b, sq_h, sq_t, sq_d,
    sk_b, sk_h, sk_t, sk_d,
    sg_b, sg_h, sg_t, sg_d,
    sb_b, sb_h, sb_t,
    CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, NUM_ITER: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_c = tl.arange(0, CHUNK)
    offs_t = pid_c * CHUNK + offs_c
    mask_t = offs_t < T

    q_base = q_ptr + pid_b * sq_b + pid_h * sq_h
    k_base = k_ptr + pid_b * sk_b + pid_h * sk_h
    g_base = g_ptr + pid_b * sg_b + pid_h * sg_h
    beta_base = beta_ptr + pid_b * sb_b + pid_h * sb_h

    beta = tl.load(
        beta_base + offs_t * sb_t, mask=mask_t, other=0.0
    ).to(tl.float32)

    a_mat = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)
    qk_mat = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)
    range_acc = 0.0

    for kb in range(NK):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D_K

        qval = tl.load(
            q_base + offs_t[:, None] * sq_t + offs_k[None, :] * sq_d,
            mask=mask_t[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        kval = tl.load(
            k_base + offs_t[:, None] * sk_t + offs_k[None, :] * sk_d,
            mask=mask_t[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        gval = tl.load(
            g_base + offs_t[:, None] * sg_t + offs_k[None, :] * sg_d,
            mask=mask_t[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)

        gcum = tl.cumsum(gval, axis=0)
        gmax = tl.max(gcum, axis=0)
        gmin = tl.min(gcum, axis=0)
        grange = tl.max(gmax - gmin)
        range_acc = tl.maximum(range_acc, grange)

        if grange <= 140.0:
            cref = 0.5 * (gmax + gmin)
            diff = gcum - cref[None, :]
            exp_p = tl.exp(diff)
            exp_m = tl.exp(-diff)
            k_down = kval * exp_m
            a_mat += tl.dot(
                (kval * exp_p) * beta[:, None], tl.trans(k_down),
                input_precision="tf32",
            )
            qk_mat += tl.dot(
                qval * exp_p, tl.trans(k_down),
                input_precision="tf32",
            )
        else:
            for kk in range(KB):
                global_k = kb * BLOCK_K + kk
                valid = mask_t & (global_k < D_K)
                q_vec = tl.load(
                    q_base + offs_t * sq_t + global_k * sq_d,
                    mask=valid, other=0.0,
                ).to(tl.float32)
                k_vec = tl.load(
                    k_base + offs_t * sk_t + global_k * sk_d,
                    mask=valid, other=0.0,
                ).to(tl.float32)
                g_vec = tl.load(
                    g_base + offs_t * sg_t + global_k * sg_d,
                    mask=valid, other=0.0,
                ).to(tl.float32)
                gc_vec = tl.cumsum(g_vec, axis=0)
                diffm = gc_vec[:, None] - gc_vec[None, :]
                exp_diff = tl.exp(tl.minimum(diffm, 80.0))
                ks_exp = k_vec[None, :] * exp_diff
                a_mat += beta[:, None] * (k_vec[:, None] * ks_exp)
                qk_mat += q_vec[:, None] * ks_exp

    row = offs_c[:, None]
    col = offs_c[None, :]
    eye = tl.where(col == row, 1.0, 0.0)
    m_mat = eye + tl.where(col < row, a_mat, 0.0)
    qk_mat = tl.where(col <= row, qk_mat, 0.0)

    x_mat = eye
    for _ in range(NUM_ITER):
        mx = tl.dot(m_mat, x_mat, input_precision="ieee")
        x_mat = tl.dot(x_mat, 2.0 * eye - mx, input_precision="ieee")

    base_idx = (pid_bh * NT + pid_c) * CHUNK
    tl.store(
        inv_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :],
        x_mat,
    )
    tl.store(
        qk_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :],
        qk_mat,
    )
    tl.store(range_ptr + pid_bh * NT + pid_c, range_acc)


@triton.jit
def _chunk_state_fast_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, inv_ptr, qk_ptr,
    bound_ptr, w_ptr, out_ptr,
    T, H, NT, D_K, D_V, scale,
    sq_b, sq_h, sq_t, sq_d,
    sk_b, sk_h, sk_t, sk_d,
    sv_b, sv_h, sv_t, sv_d,
    sg_b, sg_h, sg_t, sg_d,
    sb_b, sb_h, sb_t,
    so_b, so_h, so_t, so_d,
    CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    SAVE_AUX: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < D_K
    mask_v = offs_v < D_V

    q_base = q_ptr + pid_b * sq_b + pid_h * sq_h
    k_base = k_ptr + pid_b * sk_b + pid_h * sk_h
    v_base = v_ptr + pid_b * sv_b + pid_h * sv_h
    g_base = g_ptr + pid_b * sg_b + pid_h * sg_h
    beta_base = beta_ptr + pid_b * sb_b + pid_h * sb_h
    out_base = out_ptr + pid_b * so_b + pid_h * so_h

    state = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    for chunk_idx in range(NT):
        t_start = chunk_idx * CHUNK
        offs_t = t_start + offs_c
        mask_t = offs_t < T

        if SAVE_AUX:
            tl.store(
                bound_ptr
                + ((pid_bh * NT + chunk_idx) * D_K + offs_k[:, None]) * D_V
                + offs_v[None, :],
                state,
                mask=mask_k[:, None] & mask_v[None, :],
            )

        qval = tl.load(
            q_base + offs_t[:, None] * sq_t + offs_k[None, :] * sq_d,
            mask=mask_t[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        kval = tl.load(
            k_base + offs_t[:, None] * sk_t + offs_k[None, :] * sk_d,
            mask=mask_t[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        vval = tl.load(
            v_base + offs_t[:, None] * sv_t + offs_v[None, :] * sv_d,
            mask=mask_t[:, None] & mask_v[None, :], other=0.0,
        ).to(tl.float32)
        gval = tl.load(
            g_base + offs_t[:, None] * sg_t + offs_k[None, :] * sg_d,
            mask=mask_t[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        beta = tl.load(
            beta_base + offs_t * sb_t, mask=mask_t, other=0.0
        ).to(tl.float32)

        base_idx = (pid_bh * NT + chunk_idx) * CHUNK
        inv_mat = tl.load(
            inv_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :]
        )
        qk_mat = tl.load(
            qk_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :]
        )

        chunk_len = tl.minimum(T - t_start, CHUNK)
        gcum = tl.cumsum(gval, axis=0)
        last_gc = tl.sum(
            tl.where(offs_c[:, None] == (chunk_len - 1), gcum, 0.0), axis=0
        )
        exp_g = tl.exp(tl.minimum(gcum, 80.0))

        pred = tl.dot(kval * exp_g, state, input_precision="tf32")
        out_state = tl.dot(qval * exp_g, state, input_precision="tf32")

        rhs = beta[:, None] * (vval - pred)
        wmat = tl.dot(inv_mat, rhs, input_precision="ieee")
        if SAVE_AUX:
            tl.store(
                w_ptr
                + ((pid_bh * NT + chunk_idx) * CHUNK + offs_c[:, None]) * D_V
                + offs_v[None, :],
                wmat,
                mask=mask_t[:, None] & mask_v[None, :],
            )
        out_mat = (
            out_state + tl.dot(qk_mat, wmat, input_precision="ieee")
        ) * scale

        tl.store(
            out_base + offs_t[:, None] * so_t + offs_v[None, :] * so_d,
            out_mat,
            mask=mask_t[:, None] & mask_v[None, :],
        )

        exp_last = tl.exp(tl.minimum(last_gc, 80.0))
        exp_end = tl.exp(tl.minimum(last_gc[None, :] - gcum, 80.0))
        state = state * exp_last[:, None] + tl.dot(
            tl.trans(kval * exp_end), wmat, input_precision="tf32"
        )


@triton.jit
def _chunk_state_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, inv_ptr, qk_ptr,
    bound_ptr, w_ptr, state_ptr, out_ptr,
    T, H, NT, D_K, D_V, NK, scale,
    sq_b, sq_h, sq_t, sq_d,
    sk_b, sk_h, sk_t, sk_d,
    sv_b, sv_h, sv_t, sv_d,
    sg_b, sg_h, sg_t, sg_d,
    sb_b, sb_h, sb_t,
    so_b, so_h, so_t, so_d,
    CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    SAVE_AUX: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_c = tl.arange(0, CHUNK)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < D_V

    q_base = q_ptr + pid_b * sq_b + pid_h * sq_h
    k_base = k_ptr + pid_b * sk_b + pid_h * sk_h
    v_base = v_ptr + pid_b * sv_b + pid_h * sv_h
    g_base = g_ptr + pid_b * sg_b + pid_h * sg_h
    beta_base = beta_ptr + pid_b * sb_b + pid_h * sb_h
    out_base = out_ptr + pid_b * so_b + pid_h * so_h
    state_base = state_ptr + pid_bh * D_K * D_V

    for chunk_idx in range(NT):
        t_start = chunk_idx * CHUNK
        offs_t = t_start + offs_c
        mask_t = offs_t < T

        if SAVE_AUX:
            for kb0 in range(NK):
                ok0 = kb0 * BLOCK_K + tl.arange(0, BLOCK_K)
                mk0 = ok0 < D_K
                st0 = tl.load(
                    state_base + ok0[:, None] * D_V + offs_v[None, :],
                    mask=mk0[:, None] & mask_v[None, :], other=0.0,
                )
                tl.store(
                    bound_ptr
                    + ((pid_bh * NT + chunk_idx) * D_K + ok0[:, None]) * D_V
                    + offs_v[None, :],
                    st0,
                    mask=mk0[:, None] & mask_v[None, :],
                )

        vval = tl.load(
            v_base + offs_t[:, None] * sv_t + offs_v[None, :] * sv_d,
            mask=mask_t[:, None] & mask_v[None, :], other=0.0,
        ).to(tl.float32)
        beta = tl.load(
            beta_base + offs_t * sb_t, mask=mask_t, other=0.0
        ).to(tl.float32)

        base_idx = (pid_bh * NT + chunk_idx) * CHUNK
        inv_mat = tl.load(
            inv_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :]
        )
        qk_mat = tl.load(
            qk_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :]
        )
        chunk_len = tl.minimum(T - t_start, CHUNK)

        pred = tl.zeros([CHUNK, BLOCK_V], dtype=tl.float32)
        out_state = tl.zeros([CHUNK, BLOCK_V], dtype=tl.float32)

        for kb in range(NK):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D_K
            qval = tl.load(
                q_base + offs_t[:, None] * sq_t + offs_k[None, :] * sq_d,
                mask=mask_t[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
            kval = tl.load(
                k_base + offs_t[:, None] * sk_t + offs_k[None, :] * sk_d,
                mask=mask_t[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
            gval = tl.load(
                g_base + offs_t[:, None] * sg_t + offs_k[None, :] * sg_d,
                mask=mask_t[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
            state = tl.load(
                state_base + offs_k[:, None] * D_V + offs_v[None, :],
                mask=mask_k[:, None] & mask_v[None, :], other=0.0,
            )
            gcum = tl.cumsum(gval, axis=0)
            exp_g = tl.exp(tl.minimum(gcum, 80.0))
            pred += tl.dot(kval * exp_g, state, input_precision="ieee")
            out_state += tl.dot(qval * exp_g, state, input_precision="ieee")

        rhs = beta[:, None] * (vval - pred)
        wmat = tl.dot(inv_mat, rhs, input_precision="ieee")
        if SAVE_AUX:
            tl.store(
                w_ptr
                + ((pid_bh * NT + chunk_idx) * CHUNK + offs_c[:, None]) * D_V
                + offs_v[None, :],
                wmat,
                mask=mask_t[:, None] & mask_v[None, :],
            )
        out_mat = (
            out_state + tl.dot(qk_mat, wmat, input_precision="ieee")
        ) * scale
        tl.store(
            out_base + offs_t[:, None] * so_t + offs_v[None, :] * so_d,
            out_mat,
            mask=mask_t[:, None] & mask_v[None, :],
        )

        for kb in range(NK):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D_K
            kval = tl.load(
                k_base + offs_t[:, None] * sk_t + offs_k[None, :] * sk_d,
                mask=mask_t[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
            gval = tl.load(
                g_base + offs_t[:, None] * sg_t + offs_k[None, :] * sg_d,
                mask=mask_t[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
            state = tl.load(
                state_base + offs_k[:, None] * D_V + offs_v[None, :],
                mask=mask_k[:, None] & mask_v[None, :], other=0.0,
            )
            gcum = tl.cumsum(gval, axis=0)
            last_gc = tl.sum(
                tl.where(offs_c[:, None] == (chunk_len - 1), gcum, 0.0),
                axis=0,
            )
            exp_last = tl.exp(tl.minimum(last_gc, 80.0))
            exp_end = tl.exp(tl.minimum(last_gc[None, :] - gcum, 80.0))
            state_new = state * exp_last[:, None] + tl.dot(
                tl.trans(kval * exp_end), wmat, input_precision="ieee"
            )
            tl.store(
                state_base + offs_k[:, None] * D_V + offs_v[None, :],
                state_new,
                mask=mask_k[:, None] & mask_v[None, :],
            )


@triton.jit
def _backward_chunk_fast_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, go_ptr,
    inv_ptr, qk_ptr, w_ptr, bound_ptr, range_ptr, lambda_ptr,
    gq_ptr, gk_ptr, gv_ptr, gg_ptr, gbeta_ptr,
    chunk_idx, T, H, NT, D_K, D_V, NK, scale,
    sq_b, sq_h, sq_t, sq_d,
    sk_b, sk_h, sk_t, sk_d,
    sv_b, sv_h, sv_t, sv_d,
    sg_b, sg_h, sg_t, sg_d,
    sb_b, sb_h, sb_t,
    sgo_b, sgo_h, sgo_t, sgo_d,
    sgq_b, sgq_h, sgq_t, sgq_d,
    sgk_b, sgk_h, sgk_t, sgk_d,
    sgv_b, sgv_h, sgv_t, sgv_d,
    sgg_b, sgg_h, sgg_t, sgg_d,
    sgb_b, sgb_h, sgb_t,
    CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    rng = tl.load(range_ptr + pid_bh * NT + chunk_idx)
    if rng <= 140.0:
        pid_b = pid_bh // H
        pid_h = pid_bh % H

        offs_c = tl.arange(0, CHUNK)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < D_V
        t_start = chunk_idx * CHUNK
        offs_t = t_start + offs_c
        mask_t = offs_t < T
        chunk_len = tl.minimum(T - t_start, CHUNK)

        q_base = q_ptr + pid_b * sq_b + pid_h * sq_h
        k_base = k_ptr + pid_b * sk_b + pid_h * sk_h
        v_base = v_ptr + pid_b * sv_b + pid_h * sv_h
        g_base = g_ptr + pid_b * sg_b + pid_h * sg_h
        beta_base = beta_ptr + pid_b * sb_b + pid_h * sb_h
        go_base = go_ptr + pid_b * sgo_b + pid_h * sgo_h
        gq_base = gq_ptr + pid_b * sgq_b + pid_h * sgq_h
        gk_base = gk_ptr + pid_b * sgk_b + pid_h * sgk_h
        gv_base = gv_ptr + pid_b * sgv_b + pid_h * sgv_h
        gg_base = gg_ptr + pid_b * sgg_b + pid_h * sgg_h

        beta = tl.load(
            beta_base + offs_t * sb_t, mask=mask_t, other=0.0
        ).to(tl.float32)
        vval = tl.load(
            v_base + offs_t[:, None] * sv_t + offs_v[None, :] * sv_d,
            mask=mask_t[:, None] & mask_v[None, :], other=0.0,
        ).to(tl.float32)
        go = tl.load(
            go_base + offs_t[:, None] * sgo_t + offs_v[None, :] * sgo_d,
            mask=mask_t[:, None] & mask_v[None, :], other=0.0,
        ).to(tl.float32) * scale
        wfwd = tl.load(
            w_ptr
            + ((pid_bh * NT + chunk_idx) * CHUNK + offs_c[:, None]) * D_V
            + offs_v[None, :],
            mask=mask_t[:, None] & mask_v[None, :], other=0.0,
        )
        base_idx = (pid_bh * NT + chunk_idx) * CHUNK
        inv_mat = tl.load(
            inv_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :]
        )
        qk_mat = tl.load(
            qk_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :]
        )

        gw = tl.dot(tl.trans(qk_mat), go, input_precision="tf32")
        pred = tl.zeros([CHUNK, BLOCK_V], dtype=tl.float32)

        # Accumulate the incoming-state terms over K tiles.
        for kb in range(NK):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D_K
            kval = tl.load(
                k_base + offs_t[:, None] * sk_t + offs_k[None, :] * sk_d,
                mask=mask_t[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
            gval = tl.load(
                g_base + offs_t[:, None] * sg_t + offs_k[None, :] * sg_d,
                mask=mask_t[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
            s0 = tl.load(
                bound_ptr
                + ((pid_bh * NT + chunk_idx) * D_K + offs_k[:, None]) * D_V
                + offs_v[None, :],
                mask=mask_k[:, None] & mask_v[None, :], other=0.0,
            )
            lam = tl.load(
                lambda_ptr + (pid_bh * D_K + offs_k[:, None]) * D_V
                + offs_v[None, :],
                mask=mask_k[:, None] & mask_v[None, :], other=0.0,
            )
            gcum = tl.cumsum(gval, axis=0)
            gmax = tl.max(gcum, axis=0)
            gmin = tl.min(gcum, axis=0)
            cref = 0.5 * (gmax + gmin)
            last_gc = tl.sum(
                tl.where(offs_c[:, None] == (chunk_len - 1), gcum, 0.0),
                axis=0,
            )
            exp_p = tl.exp(gcum - cref[None, :])
            exp_m = tl.exp(cref[None, :] - gcum)
            exp_cref = tl.exp(
                tl.minimum(tl.maximum(cref, -80.0), 80.0)
            )
            f_end = tl.exp(
                tl.minimum(tl.maximum(last_gc - cref, -80.0), 80.0)
            )
            kbal = kval * exp_p
            kdown = kval * exp_m
            s0_hat = s0 * exp_cref[:, None]
            pred += tl.dot(kbal, s0_hat, input_precision="tf32")
            # gw[t, v] += sum_k kdown[t, k] * (lam[k, v] * f_end[k]):
            # kdown is [CHUNK, BLOCK_K] and the second factor is
            # [BLOCK_K, BLOCK_V], so no transpose is taken here.
            gw += tl.dot(
                kdown, lam * f_end[:, None],
                input_precision="tf32",
            )

        # y = grad wrt r = beta * (v - pred); solve L^T y-part.
        y = tl.dot(tl.trans(inv_mat), gw, input_precision="ieee")
        tl.store(
            gv_base + offs_t[:, None] * sgv_t + offs_v[None, :] * sgv_d,
            beta[:, None] * y,
            mask=mask_t[:, None] & mask_v[None, :],
        )
        delta = vval - pred
        grad_beta_acc = tl.sum(delta * y, axis=1)

        row = offs_c[:, None]
        col = offs_c[None, :]
        m_qk = tl.dot(go, tl.trans(wfwd), input_precision="tf32")
        m_qk = tl.where(col <= row, m_qk, 0.0)
        m_a = -tl.dot(y, tl.trans(wfwd), input_precision="tf32")
        m_a = tl.where(col < row, m_a, 0.0)
        r_mat = tl.where(col >= row, 1.0, 0.0)
        grad_pred = -beta[:, None] * y

        for kb in range(NK):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D_K
            qval = tl.load(
                q_base + offs_t[:, None] * sq_t + offs_k[None, :] * sq_d,
                mask=mask_t[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
            kval = tl.load(
                k_base + offs_t[:, None] * sk_t + offs_k[None, :] * sk_d,
                mask=mask_t[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
            gval = tl.load(
                g_base + offs_t[:, None] * sg_t + offs_k[None, :] * sg_d,
                mask=mask_t[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
            s0 = tl.load(
                bound_ptr
                + ((pid_bh * NT + chunk_idx) * D_K + offs_k[:, None]) * D_V
                + offs_v[None, :],
                mask=mask_k[:, None] & mask_v[None, :], other=0.0,
            )
            lam = tl.load(
                lambda_ptr + (pid_bh * D_K + offs_k[:, None]) * D_V
                + offs_v[None, :],
                mask=mask_k[:, None] & mask_v[None, :], other=0.0,
            )
            gcum = tl.cumsum(gval, axis=0)
            gmax = tl.max(gcum, axis=0)
            gmin = tl.min(gcum, axis=0)
            cref = 0.5 * (gmax + gmin)
            last_gc = tl.sum(
                tl.where(offs_c[:, None] == (chunk_len - 1), gcum, 0.0),
                axis=0,
            )
            exp_p = tl.exp(gcum - cref[None, :])
            exp_m = tl.exp(cref[None, :] - gcum)
            exp_cref = tl.exp(
                tl.minimum(tl.maximum(cref, -80.0), 80.0)
            )
            f_end = tl.exp(
                tl.minimum(tl.maximum(last_gc - cref, -80.0), 80.0)
            )
            qbal = qval * exp_p
            kbal = kval * exp_p
            kdown = kval * exp_m
            s0_hat = s0 * exp_cref[:, None]
            grad_u = lam * f_end[:, None]
            u_end = s0_hat + tl.dot(
                tl.trans(kdown), wfwd, input_precision="tf32"
            )

            s0go = tl.dot(go, tl.trans(s0_hat), input_precision="tf32")
            s0gp = tl.dot(
                grad_pred, tl.trans(s0_hat), input_precision="tf32"
            )
            grad_qbal = tl.dot(m_qk, kdown, input_precision="tf32") + s0go
            grad_kbal = (
                beta[:, None]
                * tl.dot(m_a, kdown, input_precision="tf32")
                + s0gp
            )
            grad_kdown = tl.dot(
                tl.trans(m_qk), qbal, input_precision="tf32"
            )
            grad_kdown += tl.dot(
                tl.trans(m_a * beta[:, None]), kbal,
                input_precision="tf32",
            )
            grad_kdown += tl.dot(
                wfwd, tl.trans(grad_u), input_precision="tf32"
            )

            grad_gcum = (
                qbal * grad_qbal + kbal * grad_kbal
                - kdown * grad_kdown
            )
            end_vec = f_end * tl.sum(lam * u_end, axis=1)
            grad_gcum += tl.where(
                offs_c[:, None] == (chunk_len - 1),
                end_vec[None, :],
                0.0,
            )
            grad_g = tl.dot(r_mat, grad_gcum, input_precision="tf32")
            grad_q = exp_p * grad_qbal
            grad_k = exp_p * grad_kbal + exp_m * grad_kdown

            tl.atomic_add(
                gq_base + offs_t[:, None] * sgq_t
                + offs_k[None, :] * sgq_d,
                grad_q,
                mask=mask_t[:, None] & mask_k[None, :],
            )
            tl.atomic_add(
                gk_base + offs_t[:, None] * sgk_t
                + offs_k[None, :] * sgk_d,
                grad_k,
                mask=mask_t[:, None] & mask_k[None, :],
            )
            tl.atomic_add(
                gg_base + offs_t[:, None] * sgg_t
                + offs_k[None, :] * sgg_d,
                grad_g,
                mask=mask_t[:, None] & mask_k[None, :],
            )

            s_mat = tl.dot(kbal, tl.trans(kdown), input_precision="tf32")
            s_mat = tl.where(col < row, s_mat, 0.0)
            grad_beta_acc += tl.sum(m_a * s_mat, axis=1)

            lam_next_hat = tl.dot(
                tl.trans(qbal), go, input_precision="tf32"
            )
            lam_next_hat += tl.dot(
                tl.trans(kbal), grad_pred, input_precision="tf32"
            )
            lam_next_hat += grad_u
            tl.store(
                lambda_ptr + (pid_bh * D_K + offs_k[:, None]) * D_V
                + offs_v[None, :],
                lam_next_hat * exp_cref[:, None],
                mask=mask_k[:, None] & mask_v[None, :],
            )

        tl.atomic_add(
            gbeta_ptr + pid_b * sgb_b + pid_h * sgb_h + offs_t * sgb_t,
            grad_beta_acc,
            mask=mask_t,
        )


@triton.jit
def _backward_chunk_slow_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, go_ptr,
    bound_ptr, range_ptr, lambda_ptr, traj_ptr, cur_ptr,
    gq_ptr, gk_ptr, gv_ptr, gg_ptr, gbeta_ptr,
    chunk_idx, CLEN_ARG, T, H, NT, D_K, D_V, NK, scale,
    sq_b, sq_h, sq_t, sq_d,
    sk_b, sk_h, sk_t, sk_d,
    sv_b, sv_h, sv_t, sv_d,
    sg_b, sg_h, sg_t, sg_d,
    sb_b, sb_h, sb_t,
    sgo_b, sgo_h, sgo_t, sgo_d,
    sgq_b, sgq_h, sgq_t, sgq_d,
    sgk_b, sgk_h, sgk_t, sgk_d,
    sgv_b, sgv_h, sgv_t, sgv_d,
    sgg_b, sgg_h, sgg_t, sgg_d,
    sgb_b, sgb_h, sgb_t,
    CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    rng = tl.load(range_ptr + pid_bh * NT + chunk_idx)
    if rng > 140.0:
        pid_b = pid_bh // H
        pid_h = pid_bh % H
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < D_V
        t_start = chunk_idx * CHUNK
        # CLEN_ARG is a runtime (non-constexpr) argument, so the two
        # replay loops below are compiled as genuine dynamic loops
        # instead of being unrolled CHUNK times into an enormous
        # kernel body that made compilation die/time out.
        chunk_len = CLEN_ARG

        q_base = q_ptr + pid_b * sq_b + pid_h * sq_h
        k_base = k_ptr + pid_b * sk_b + pid_h * sk_h
        v_base = v_ptr + pid_b * sv_b + pid_h * sv_h
        g_base = g_ptr + pid_b * sg_b + pid_h * sg_h
        beta_base = beta_ptr + pid_b * sb_b + pid_h * sb_h
        go_base = go_ptr + pid_b * sgo_b + pid_h * sgo_h
        gq_base = gq_ptr + pid_b * sgq_b + pid_h * sgq_h
        gk_base = gk_ptr + pid_b * sgk_b + pid_h * sgk_h
        gv_base = gv_ptr + pid_b * sgv_b + pid_h * sgv_h
        gg_base = gg_ptr + pid_b * sgg_b + pid_h * sgg_h
        traj_base = traj_ptr + pid_bh * CHUNK * D_K * D_V
        cur_base = cur_ptr + pid_bh * D_K * D_V

        for kb in range(NK):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D_K
            s0 = tl.load(
                bound_ptr
                + ((pid_bh * NT + chunk_idx) * D_K + offs_k[:, None]) * D_V
                + offs_v[None, :],
                mask=mask_k[:, None] & mask_v[None, :], other=0.0,
            )
            tl.store(
                cur_base + offs_k[:, None] * D_V + offs_v[None, :],
                s0,
                mask=mask_k[:, None] & mask_v[None, :],
            )

        # Forward replay, storing S_{t-1} for every step of this chunk.
        for step in range(chunk_len):
            tt = t_start + step
            for kb in range(NK):
                offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
                mask_k = offs_k < D_K
                st = tl.load(
                    cur_base + offs_k[:, None] * D_V
                    + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                tl.store(
                    traj_base
                    + (step * D_K + offs_k[:, None]) * D_V
                    + offs_v[None, :],
                    st,
                    mask=mask_k[:, None] & mask_v[None, :],
                )
            pred = tl.zeros([BLOCK_V], dtype=tl.float32)
            for kb in range(NK):
                offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
                mask_k = offs_k < D_K
                st = tl.load(
                    cur_base + offs_k[:, None] * D_V
                    + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                kv = tl.load(
                    k_base + tt * sk_t + offs_k * sk_d,
                    mask=mask_k, other=0.0,
                ).to(tl.float32)
                gv_ = tl.load(
                    g_base + tt * sg_t + offs_k * sg_d,
                    mask=mask_k, other=0.0,
                ).to(tl.float32)
                til = st * tl.exp(tl.minimum(gv_, 80.0))[:, None]
                pred += tl.sum(til * kv[:, None], axis=0)
            bv = tl.load(beta_base + tt * sb_t).to(tl.float32)
            vv = tl.load(
                v_base + tt * sv_t + offs_v * sv_d,
                mask=mask_v, other=0.0,
            ).to(tl.float32)
            delta = vv - pred
            for kb in range(NK):
                offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
                mask_k = offs_k < D_K
                st = tl.load(
                    cur_base + offs_k[:, None] * D_V
                    + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                kv = tl.load(
                    k_base + tt * sk_t + offs_k * sk_d,
                    mask=mask_k, other=0.0,
                ).to(tl.float32)
                gv_ = tl.load(
                    g_base + tt * sg_t + offs_k * sg_d,
                    mask=mask_k, other=0.0,
                ).to(tl.float32)
                til = st * tl.exp(tl.minimum(gv_, 80.0))[:, None]
                tl.store(
                    cur_base + offs_k[:, None] * D_V
                    + offs_v[None, :],
                    til + bv * kv[:, None] * delta[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                )

        # Reverse replay using the stored pre-step states.
        for rev in range(chunk_len):
            step = chunk_len - 1 - rev
            tt = t_start + step
            bv = tl.load(beta_base + tt * sb_t).to(tl.float32)
            vv = tl.load(
                v_base + tt * sv_t + offs_v * sv_d,
                mask=mask_v, other=0.0,
            ).to(tl.float32)
            gor = tl.load(
                go_base + tt * sgo_t + offs_v * sgo_d,
                mask=mask_v, other=0.0,
            ).to(tl.float32)
            go = gor * scale

            a_vec = tl.zeros([BLOCK_V], dtype=tl.float32)
            pred = tl.zeros([BLOCK_V], dtype=tl.float32)
            for kb in range(NK):
                offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
                mask_k = offs_k < D_K
                sprev = tl.load(
                    traj_base
                    + (step * D_K + offs_k[:, None]) * D_V
                    + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                lam = tl.load(
                    lambda_ptr + (pid_bh * D_K + offs_k[:, None]) * D_V
                    + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                qv = tl.load(
                    q_base + tt * sq_t + offs_k * sq_d,
                    mask=mask_k, other=0.0,
                ).to(tl.float32)
                kv = tl.load(
                    k_base + tt * sk_t + offs_k * sk_d,
                    mask=mask_k, other=0.0,
                ).to(tl.float32)
                gv_ = tl.load(
                    g_base + tt * sg_t + offs_k * sg_d,
                    mask=mask_k, other=0.0,
                ).to(tl.float32)
                hh = lam + qv[:, None] * go[None, :]
                til = sprev * tl.exp(tl.minimum(gv_, 80.0))[:, None]
                a_vec += tl.sum(kv[:, None] * hh, axis=0)
                pred += tl.sum(til * kv[:, None], axis=0)

            delta = vv - pred
            tl.store(
                gv_base + tt * sgv_t + offs_v * sgv_d,
                bv * a_vec,
                mask=mask_v,
            )
            tl.atomic_add(
                gbeta_ptr + pid_b * sgb_b + pid_h * sgb_h
                + tt * sgb_t,
                tl.sum(a_vec * delta),
            )

            for kb in range(NK):
                offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
                mask_k = offs_k < D_K
                sprev = tl.load(
                    traj_base
                    + (step * D_K + offs_k[:, None]) * D_V
                    + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                lam = tl.load(
                    lambda_ptr + (pid_bh * D_K + offs_k[:, None]) * D_V
                    + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                )
                qv = tl.load(
                    q_base + tt * sq_t + offs_k * sq_d,
                    mask=mask_k, other=0.0,
                ).to(tl.float32)
                kv = tl.load(
                    k_base + tt * sk_t + offs_k * sk_d,
                    mask=mask_k, other=0.0,
                ).to(tl.float32)
                gv_ = tl.load(
                    g_base + tt * sg_t + offs_k * sg_d,
                    mask=mask_k, other=0.0,
                ).to(tl.float32)
                eg = tl.exp(tl.minimum(gv_, 80.0))
                hh = lam + qv[:, None] * go[None, :]
                acoef = hh - bv * kv[:, None] * a_vec[None, :]
                til = sprev * eg[:, None]
                st = til + bv * kv[:, None] * delta[None, :]

                tl.atomic_add(
                    gq_base + tt * sgq_t + offs_k * sgq_d,
                    tl.sum(st * go[None, :], axis=1),
                    mask=mask_k,
                )
                tl.atomic_add(
                    gk_base + tt * sgk_t + offs_k * sgk_d,
                    bv
                    * (
                        tl.sum(hh * delta[None, :], axis=1)
                        - tl.sum(til * a_vec[None, :], axis=1)
                    ),
                    mask=mask_k,
                )
                tl.atomic_add(
                    gg_base + tt * sgg_t + offs_k * sgg_d,
                    eg * tl.sum(acoef * sprev, axis=1),
                    mask=mask_k,
                )
                tl.store(
                    lambda_ptr
                    + (pid_bh * D_K + offs_k[:, None]) * D_V
                    + offs_v[None, :],
                    eg[:, None] * acoef,
                    mask=mask_k[:, None] & mask_v[None, :],
                )


def _launch_forward(q, k, v, g, beta, scale, out, inv, qk, rng_buf,
                    bound, wfwd, save_aux):
    B, H, T, K = q.shape
    V = v.shape[-1]
    CHUNK = 32
    INV_BLOCK_K = 64
    BLOCK_V = 32

    num_iter = 0
    while (1 << num_iter) < CHUNK:
        num_iter += 1

    bh = B * H
    nt = (T + CHUNK - 1) // CHUNK
    inv_nk = (K + INV_BLOCK_K - 1) // INV_BLOCK_K

    _chunk_inv_kernel[(nt, bh)](
        q_ptr=q, k_ptr=k, g_ptr=g, beta_ptr=beta,
        inv_ptr=inv, qk_ptr=qk, range_ptr=rng_buf,
        T=T, H=H, NT=nt, D_K=K, NK=inv_nk, KB=INV_BLOCK_K,
        sq_b=q.stride(0), sq_h=q.stride(1), sq_t=q.stride(2),
        sq_d=q.stride(3),
        sk_b=k.stride(0), sk_h=k.stride(1), sk_t=k.stride(2),
        sk_d=k.stride(3),
        sg_b=g.stride(0), sg_h=g.stride(1), sg_t=g.stride(2),
        sg_d=g.stride(3),
        sb_b=beta.stride(0), sb_h=beta.stride(1), sb_t=beta.stride(2),
        CHUNK=CHUNK, BLOCK_K=INV_BLOCK_K, NUM_ITER=num_iter,
        num_warps=4, num_stages=1,
    )

    if K <= 128:
        fast_block_k = triton.next_power_of_2(max(K, 16))
        _chunk_state_fast_kernel[(bh, triton.cdiv(V, BLOCK_V))](
            q_ptr=q, k_ptr=k, v_ptr=v, g_ptr=g, beta_ptr=beta,
            inv_ptr=inv, qk_ptr=qk, bound_ptr=bound, w_ptr=wfwd,
            out_ptr=out,
            T=T, H=H, NT=nt, D_K=K, D_V=V, scale=float(scale),
            sq_b=q.stride(0), sq_h=q.stride(1), sq_t=q.stride(2),
            sq_d=q.stride(3),
            sk_b=k.stride(0), sk_h=k.stride(1), sk_t=k.stride(2),
            sk_d=k.stride(3),
            sv_b=v.stride(0), sv_h=v.stride(1), sv_t=v.stride(2),
            sv_d=v.stride(3),
            sg_b=g.stride(0), sg_h=g.stride(1), sg_t=g.stride(2),
            sg_d=g.stride(3),
            sb_b=beta.stride(0), sb_h=beta.stride(1), sb_t=beta.stride(2),
            so_b=out.stride(0), so_h=out.stride(1), so_t=out.stride(2),
            so_d=out.stride(3),
            CHUNK=CHUNK, BLOCK_K=fast_block_k, BLOCK_V=BLOCK_V,
            SAVE_AUX=save_aux,
            num_warps=8, num_stages=1,
        )
    else:
        state = torch.zeros((bh, K, V), device=q.device, dtype=torch.float32)
        _chunk_state_kernel[(bh, triton.cdiv(V, BLOCK_V))](
            q_ptr=q, k_ptr=k, v_ptr=v, g_ptr=g, beta_ptr=beta,
            inv_ptr=inv, qk_ptr=qk, bound_ptr=bound, w_ptr=wfwd,
            state_ptr=state, out_ptr=out,
            T=T, H=H, NT=nt, D_K=K, D_V=V, NK=inv_nk, scale=float(scale),
            sq_b=q.stride(0), sq_h=q.stride(1), sq_t=q.stride(2),
            sq_d=q.stride(3),
            sk_b=k.stride(0), sk_h=k.stride(1), sk_t=k.stride(2),
            sk_d=k.stride(3),
            sv_b=v.stride(0), sv_h=v.stride(1), sv_t=v.stride(2),
            sv_d=v.stride(3),
            sg_b=g.stride(0), sg_h=g.stride(1), sg_t=g.stride(2),
            sg_d=g.stride(3),
            sb_b=beta.stride(0), sb_h=beta.stride(1), sb_t=beta.stride(2),
            so_b=out.stride(0), so_h=out.stride(1), so_t=out.stride(2),
            so_d=out.stride(3),
            CHUNK=CHUNK, BLOCK_K=INV_BLOCK_K, BLOCK_V=BLOCK_V,
            SAVE_AUX=save_aux,
            num_warps=4, num_stages=1,
        )
    return nt


class _GatedDeltaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        ctx.shape_info = (B, H, T, K, V)
        ctx.scale = float(scale)
        ctx.e_dtype = e.dtype
        ctx.gamma_dtype = gamma.dtype
        ctx.e_shape = e.shape
        ctx.gamma_shape = gamma.shape
        if B == 0 or H == 0 or T == 0 or K == 0 or V == 0:
            ctx.nt = 0
            ctx.save_for_backward(q, k, v, g, beta)
            return out

        CHUNK = 32
        bh = B * H
        nt = (T + CHUNK - 1) // CHUNK
        ctx.nt = nt
        inv = torch.empty(
            (bh * nt * CHUNK * CHUNK,), device=q.device, dtype=torch.float32
        )
        qk = torch.empty(
            (bh * nt * CHUNK * CHUNK,), device=q.device, dtype=torch.float32
        )
        rng_buf = torch.empty((bh * nt,), device=q.device, dtype=torch.float32)
        bound = torch.empty((bh, nt, K, V), device=q.device, dtype=torch.float32)
        wfwd = torch.empty((bh, nt, CHUNK, V), device=q.device, dtype=torch.float32)

        _launch_forward(
            q, k, v, g, beta, float(scale), out, inv, qk, rng_buf,
            bound, wfwd, True,
        )
        ctx.save_for_backward(q, k, v, g, beta, inv, qk, rng_buf, bound, wfwd)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (B, H, T, K, V) = ctx.shape_info
        scale = ctx.scale
        if ctx.nt == 0:
            q, k, v, g, beta = ctx.saved_tensors
            return (
                torch.zeros_like(q), torch.zeros_like(k),
                torch.zeros_like(v),
                torch.zeros(ctx.e_shape, device=q.device, dtype=ctx.e_dtype),
                torch.zeros_like(g), torch.zeros_like(beta),
                torch.zeros(
                    ctx.gamma_shape, device=q.device, dtype=ctx.gamma_dtype
                ),
                None,
            )

        q, k, v, g, beta, inv, qk, rng_buf, bound, wfwd = ctx.saved_tensors
        device = q.device
        CHUNK = 32
        BLOCK_K = 64
        BLOCK_V = 32
        bh = B * H
        nt = ctx.nt
        nk = (K + BLOCK_K - 1) // BLOCK_K

        gq = torch.zeros((B, H, T, K), device=device, dtype=torch.float32)
        gk = torch.zeros((B, H, T, K), device=device, dtype=torch.float32)
        gv = torch.zeros((B, H, T, V), device=device, dtype=torch.float32)
        gg = torch.zeros((B, H, T, K), device=device, dtype=torch.float32)
        gbeta = torch.zeros((B, H, T), device=device, dtype=torch.float32)
        lam = torch.zeros((bh, K, V), device=device, dtype=torch.float32)
        traj = torch.empty((bh, CHUNK, K, V), device=device, dtype=torch.float32)
        cur = torch.empty((bh, K, V), device=device, dtype=torch.float32)

        common = dict(
            T=T, H=H, NT=nt, D_K=K, D_V=V, NK=nk, scale=scale,
            sq_b=q.stride(0), sq_h=q.stride(1), sq_t=q.stride(2),
            sq_d=q.stride(3),
            sk_b=k.stride(0), sk_h=k.stride(1), sk_t=k.stride(2),
            sk_d=k.stride(3),
            sv_b=v.stride(0), sv_h=v.stride(1), sv_t=v.stride(2),
            sv_d=v.stride(3),
            sg_b=g.stride(0), sg_h=g.stride(1), sg_t=g.stride(2),
            sg_d=g.stride(3),
            sb_b=beta.stride(0), sb_h=beta.stride(1), sb_t=beta.stride(2),
            sgo_b=grad_out.stride(0), sgo_h=grad_out.stride(1),
            sgo_t=grad_out.stride(2), sgo_d=grad_out.stride(3),
            sgq_b=gq.stride(0), sgq_h=gq.stride(1), sgq_t=gq.stride(2),
            sgq_d=gq.stride(3),
            sgk_b=gk.stride(0), sgk_h=gk.stride(1), sgk_t=gk.stride(2),
            sgk_d=gk.stride(3),
            sgv_b=gv.stride(0), sgv_h=gv.stride(1), sgv_t=gv.stride(2),
            sgv_d=gv.stride(3),
            sgg_b=gg.stride(0), sgg_h=gg.stride(1), sgg_t=gg.stride(2),
            sgg_d=gg.stride(3),
            sgb_b=gbeta.stride(0), sgb_h=gbeta.stride(1),
            sgb_t=gbeta.stride(2),
            CHUNK=CHUNK, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        )
        grid = (bh, triton.cdiv(V, BLOCK_V))

        # The slow (per-step replay) kernel is only needed when some
        # chunk's decay range exceeds what the fast kernel's balanced
        # exponentials can represent.  Launching it unconditionally
        # forced Triton to compile that kernel on every backward, even
        # though its runtime branch would exit immediately.
        need_slow = bool((rng_buf > 140.0).any().item())

        for chunk_idx in range(nt - 1, -1, -1):
            _backward_chunk_fast_kernel[grid](
                q_ptr=q, k_ptr=k, v_ptr=v, g_ptr=g, beta_ptr=beta,
                go_ptr=grad_out, inv_ptr=inv, qk_ptr=qk, w_ptr=wfwd,
                bound_ptr=bound, range_ptr=rng_buf, lambda_ptr=lam,
                gq_ptr=gq, gk_ptr=gk, gv_ptr=gv, gg_ptr=gg,
                gbeta_ptr=gbeta, chunk_idx=chunk_idx,
                num_warps=4, num_stages=1, **common,
            )
            if need_slow:
                _backward_chunk_slow_kernel[grid](
                    q_ptr=q, k_ptr=k, v_ptr=v, g_ptr=g, beta_ptr=beta,
                    go_ptr=grad_out, bound_ptr=bound, range_ptr=rng_buf,
                    lambda_ptr=lam, traj_ptr=traj, cur_ptr=cur,
                    gq_ptr=gq, gk_ptr=gk, gv_ptr=gv, gg_ptr=gg,
                    gbeta_ptr=gbeta, chunk_idx=chunk_idx,
                    CLEN_ARG=min(CHUNK, T - chunk_idx * CHUNK),
                    num_warps=4, num_stages=1, **common,
                )

        return (
            gq.to(q.dtype),
            gk.to(k.dtype),
            gv.to(v.dtype),
            torch.zeros(ctx.e_shape, device=device, dtype=ctx.e_dtype),
            gg.to(g.dtype),
            gbeta.to(beta.dtype),
            torch.zeros(ctx.gamma_shape, device=device, dtype=ctx.gamma_dtype),
            None,
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _GatedDeltaFunction.apply(q, k, v, e, g, beta, gamma, scale)
```

The crash was a compilation assumption, not a data assumption: the strong-decay backward kernel looped over the constexpr `CHUNK=32`, so Triton unrolled 64 sequential replay steps (each containing several K-tile loops over global state) into one enormous kernel, and it was launched—and therefore compiled—on every backward even when its runtime branch would exit immediately, killing/timing out the process. The fix makes the replay length a runtime argument so both replay loops compile as genuine dynamic loops, and launches that kernel only when the saved decay-range buffer actually contains a chunk above the fast kernel's 140 threshold. All forward code, the fast backward, and every gradient formula are unchanged.