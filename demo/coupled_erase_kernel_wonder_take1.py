import torch
import triton
import triton.language as tl

CHUNK = 16
LCHUNK = 32
BK = 32


@triton.jit
def _inv_kernel(k_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, inv_ptr, T, Kdim,
                num_chunks, BLOCK_K: tl.constexpr, CHUNK: tl.constexpr,
                LCHUNK: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // num_chunks
    c = pid % num_chunks
    t0 = c * CHUNK
    offs_l = tl.arange(0, LCHUNK)
    row_t = offs_l // 2
    is_write = (offs_l % 2) == 1
    t_l = t0 + row_t
    mask_l = t_l < T
    # A gram
    A = tl.zeros((LCHUNK, LCHUNK), dtype=tl.float32)
    for kb in range(0, BLOCK_K, BK):
        offk = kb + tl.arange(0, BK)
        mask_k = offk < Kdim
        e_b = tl.load(e_ptr + (bh * T + t_l)[:, None] * Kdim + offk[None, :],
                      mask=mask_l[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        k_b = tl.load(k_ptr + (bh * T + t_l)[:, None] * Kdim + offk[None, :],
                      mask=mask_l[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        g_b = tl.load(g_ptr + (bh * T + t_l)[:, None] * Kdim + offk[None, :],
                      mask=mask_l[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        U_b = tl.where(is_write[:, None], k_b, e_b)
        g_step = tl.where(is_write[:, None], 0.0, g_b)
        G_b = tl.cumsum(g_step, axis=0)
        diff = G_b[:, None, :] - G_b[None, :, :]
        diff = tl.minimum(tl.maximum(diff, -60.0), 60.0)
        wgt = tl.exp(diff)
        A += tl.sum(U_b[:, None, :] * U_b[None, :, :] * wgt, axis=2)
    beta_l = tl.load(beta_ptr + bh * T + t_l, mask=mask_l, other=0.0).to(tl.float32)
    gamma_l = tl.load(gamma_ptr + bh * T + t_l, mask=mask_l, other=0.0).to(tl.float32)
    alpha_l = gamma_l * (1.0 - beta_l)
    b_l = tl.where(is_write, beta_l, alpha_l)
    row = offs_l[:, None]
    col = offs_l[None, :]
    strict = col < row
    N = tl.where(strict, b_l[:, None] * A, 0.0)
    I = tl.where(row == col, 1.0, 0.0)
    X = -N
    Sacc = I
    Ppow = X
    for _ in range(5):
        Sacc = Sacc + tl.dot(Ppow, Sacc, input_precision="ieee")
        Ppow = tl.dot(Ppow, Ppow, input_precision="ieee")
    out_ptr = inv_ptr + (bh * num_chunks + c) * LCHUNK * LCHUNK
    tl.store(out_ptr + offs_l[:, None] * LCHUNK + offs_l[None, :], Sacc)


@triton.jit
def _fwd_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr,
                inv_ptr, sb_ptr, o_ptr, T, Kdim, Vdim, scale, num_chunks,
                num_v_blocks, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
                CHUNK: tl.constexpr, LCHUNK: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // num_v_blocks
    vb = pid % num_v_blocks
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_c = tl.arange(0, CHUNK)
    offs_l = tl.arange(0, LCHUNK)
    row_t = offs_l // 2
    is_write = (offs_l % 2) == 1
    mask_k = offs_k < Kdim
    mask_v = offs_v < Vdim
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    for c in range(num_chunks):
        t0 = c * CHUNK
        t_c = t0 + offs_c
        t_l = t0 + row_t
        mask_c = t_c < T
        mask_l = t_l < T
        sb_cur = sb_ptr + (bh * num_chunks + c) * Kdim * Vdim
        tl.store(sb_cur + offs_k[:, None] * Vdim + offs_v[None, :], S,
                 mask=mask_k[:, None] & mask_v[None, :])
        # load chunk inputs
        q_t = tl.load(q_ptr + (bh * T + t_c)[:, None] * Kdim + offs_k[None, :],
                      mask=mask_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        k_c = tl.load(k_ptr + (bh * T + t_c)[:, None] * Kdim + offs_k[None, :],
                      mask=mask_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        e_c = tl.load(e_ptr + (bh * T + t_c)[:, None] * Kdim + offs_k[None, :],
                      mask=mask_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        g_c = tl.load(g_ptr + (bh * T + t_c)[:, None] * Kdim + offs_k[None, :],
                      mask=mask_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        v_c = tl.load(v_ptr + (bh * T + t_c)[:, None] * Vdim + offs_v[None, :],
                      mask=mask_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
        beta_c = tl.load(beta_ptr + bh * T + t_c, mask=mask_c, other=0.0).to(tl.float32)
        gamma_c = tl.load(gamma_ptr + bh * T + t_c, mask=mask_c, other=0.0).to(tl.float32)
        # interleaved
        e_l = tl.load(e_ptr + (bh * T + t_l)[:, None] * Kdim + offs_k[None, :],
                      mask=mask_l[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        k_l = tl.load(k_ptr + (bh * T + t_l)[:, None] * Kdim + offs_k[None, :],
                      mask=mask_l[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        g_l = tl.load(g_ptr + (bh * T + t_l)[:, None] * Kdim + offs_k[None, :],
                      mask=mask_l[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        v_l = tl.load(v_ptr + (bh * T + t_l)[:, None] * Vdim + offs_v[None, :],
                      mask=mask_l[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
        beta_l = tl.load(beta_ptr + bh * T + t_l, mask=mask_l, other=0.0).to(tl.float32)
        gamma_l = tl.load(gamma_ptr + bh * T + t_l, mask=mask_l, other=0.0).to(tl.float32)
        U = tl.where(is_write[:, None], k_l, e_l)
        alpha_l = gamma_l * (1.0 - beta_l)
        b_l = tl.where(is_write, beta_l, alpha_l)
        target = tl.where(is_write[:, None], v_l, 0.0)
        g_step = tl.where(is_write[:, None], 0.0, g_l)
        Gmat = tl.cumsum(g_step, axis=0)
        cg = tl.cumsum(g_c, axis=0)
        total = tl.sum(g_c, axis=0)
        exp_G = tl.exp(Gmat)
        exp_cg = tl.exp(cg)
        Up = U * exp_G
        Qe = q_t * exp_cg
        P = tl.load(inv_ptr + (bh * num_chunks + c) * LCHUNK * LCHUNK +
                    offs_l[:, None] * LCHUNK + offs_l[None, :]).to(tl.float32)
        E = tl.dot(Up, S)
        rhs = b_l[:, None] * (target - E)
        W = tl.dot(P, rhs)
        # Mx gram
        Mx = tl.zeros((CHUNK, LCHUNK), dtype=tl.float32)
        for kb in range(0, BLOCK_K, BK):
            offk = kb + tl.arange(0, BK)
            mk = offk < Kdim
            q_b = tl.load(q_ptr + (bh * T + t_c)[:, None] * Kdim + offk[None, :],
                          mask=mask_c[:, None] & mk[None, :], other=0.0).to(tl.float32)
            e_b = tl.load(e_ptr + (bh * T + t_l)[:, None] * Kdim + offk[None, :],
                          mask=mask_l[:, None] & mk[None, :], other=0.0).to(tl.float32)
            kk_b = tl.load(k_ptr + (bh * T + t_l)[:, None] * Kdim + offk[None, :],
                           mask=mask_l[:, None] & mk[None, :], other=0.0).to(tl.float32)
            g_cb = tl.load(g_ptr + (bh * T + t_c)[:, None] * Kdim + offk[None, :],
                           mask=mask_c[:, None] & mk[None, :], other=0.0).to(tl.float32)
            g_lb = tl.load(g_ptr + (bh * T + t_l)[:, None] * Kdim + offk[None, :],
                           mask=mask_l[:, None] & mk[None, :], other=0.0).to(tl.float32)
            U_b = tl.where(is_write[:, None], kk_b, e_b)
            cg_b = tl.cumsum(g_cb, axis=0)
            G_b = tl.cumsum(tl.where(is_write[:, None], 0.0, g_lb), axis=0)
            diff = cg_b[:, None, :] - G_b[None, :, :]
            diff = tl.minimum(tl.maximum(diff, -60.0), 60.0)
            Mx += tl.sum(q_b[:, None, :] * U_b[None, :, :] * tl.exp(diff), axis=2)
        ready = offs_l[None, :] <= (2 * offs_c[:, None] + 1)
        Mx = tl.where(ready, Mx, 0.0)
        QS = tl.dot(Qe, S)
        out = (QS + tl.dot(Mx, W)) * scale
        tl.store(o_ptr + (bh * T + t_c)[:, None] * Vdim + offs_v[None, :], out,
                 mask=mask_c[:, None] & mask_v[None, :])
        Ue = U * tl.exp(total[None, :] - Gmat)
        S = S * tl.exp(total)[:, None] + tl.dot(tl.trans(Ue), W)


@triton.jit
def _bwd_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr,
                inv_ptr, sb_ptr, go_ptr, dq_ptr, dk_ptr, de_ptr, dv_ptr,
                dg_ptr, dbeta_ptr, dgamma_ptr, ds_ptr, BH, T, Kdim, Vdim,
                scale, num_chunks, num_v_blocks,
                BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
                CHUNK: tl.constexpr, LCHUNK: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // num_v_blocks
    vb = pid % num_v_blocks
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_c = tl.arange(0, CHUNK)
    offs_l = tl.arange(0, LCHUNK)
    row_t = offs_l // 2
    is_write = (offs_l % 2) == 1
    mask_k = offs_k < Kdim
    mask_v = offs_v < Vdim
    # grouping / suffix matrices (constant)
    grp = (row_t[None, :] == offs_c[:, None]).to(tl.float32)
    suf = (offs_c[None, :] >= offs_c[:, None]).to(tl.float32)
    row = offs_l[:, None]
    col = offs_l[None, :]
    strict = col < row
    for rev in range(num_chunks):
        c = num_chunks - 1 - rev
        t0 = c * CHUNK
        t_c = t0 + offs_c
        t_l = t0 + row_t
        mask_c = t_c < T
        mask_l = t_l < T
        # loads
        q_t = tl.load(q_ptr + (bh * T + t_c)[:, None] * Kdim + offs_k[None, :],
                      mask=mask_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        g_c = tl.load(g_ptr + (bh * T + t_c)[:, None] * Kdim + offs_k[None, :],
                      mask=mask_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        e_l = tl.load(e_ptr + (bh * T + t_l)[:, None] * Kdim + offs_k[None, :],
                      mask=mask_l[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        k_l = tl.load(k_ptr + (bh * T + t_l)[:, None] * Kdim + offs_k[None, :],
                      mask=mask_l[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        g_l = tl.load(g_ptr + (bh * T + t_l)[:, None] * Kdim + offs_k[None, :],
                      mask=mask_l[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        v_l = tl.load(v_ptr + (bh * T + t_l)[:, None] * Vdim + offs_v[None, :],
                      mask=mask_l[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
        beta_l = tl.load(beta_ptr + bh * T + t_l, mask=mask_l, other=0.0).to(tl.float32)
        gamma_l = tl.load(gamma_ptr + bh * T + t_l, mask=mask_l, other=0.0).to(tl.float32)
        beta_c = tl.load(beta_ptr + bh * T + t_c, mask=mask_c, other=0.0).to(tl.float32)
        gamma_c = tl.load(gamma_ptr + bh * T + t_c, mask=mask_c, other=0.0).to(tl.float32)
        U = tl.where(is_write[:, None], k_l, e_l)
        alpha_l = gamma_l * (1.0 - beta_l)
        b_l = tl.where(is_write, beta_l, alpha_l)
        target = tl.where(is_write[:, None], v_l, 0.0)
        Gmat = tl.cumsum(tl.where(is_write[:, None], 0.0, g_l), axis=0)
        cg = tl.cumsum(g_c, axis=0)
        total = tl.sum(g_c, axis=0)
        exp_G = tl.exp(Gmat)
        exp_cg = tl.exp(cg)
        exp_total = tl.exp(total)
        Up = U * exp_G
        Qe = q_t * exp_cg
        Ue = U * tl.exp(total[None, :] - Gmat)
        S0 = tl.load(sb_ptr + (bh * num_chunks + c) * Kdim * Vdim +
                     offs_k[:, None] * Vdim + offs_v[None, :],
                     mask=mask_k[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
        P = tl.load(inv_ptr + (bh * num_chunks + c) * LCHUNK * LCHUNK +
                    offs_l[:, None] * LCHUNK + offs_l[None, :]).to(tl.float32)
        GO = tl.load(go_ptr + (bh * T + t_c)[:, None] * Vdim + offs_v[None, :],
                     mask=mask_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32) * scale
        ds_off = (bh * num_v_blocks + vb) * Kdim * Vdim
        dSend = tl.load(ds_ptr + ds_off + offs_k[:, None] * Vdim + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
        # recompute forward chunk quantities
        E = tl.dot(Up, S0, input_precision="ieee")
        rhs = b_l[:, None] * (target - E)
        W = tl.dot(P, rhs, input_precision="ieee")
        # A and Mx grams
        A = tl.zeros((LCHUNK, LCHUNK), dtype=tl.float32)
        Mx = tl.zeros((CHUNK, LCHUNK), dtype=tl.float32)
        for kb in range(0, BLOCK_K, BK):
            offk = kb + tl.arange(0, BK)
            mk = offk < Kdim
            e_b = tl.load(e_ptr + (bh * T + t_l)[:, None] * Kdim + offk[None, :],
                          mask=mask_l[:, None] & mk[None, :], other=0.0).to(tl.float32)
            kk_b = tl.load(k_ptr + (bh * T + t_l)[:, None] * Kdim + offk[None, :],
                           mask=mask_l[:, None] & mk[None, :], other=0.0).to(tl.float32)
            g_lb = tl.load(g_ptr + (bh * T + t_l)[:, None] * Kdim + offk[None, :],
                           mask=mask_l[:, None] & mk[None, :], other=0.0).to(tl.float32)
            q_b = tl.load(q_ptr + (bh * T + t_c)[:, None] * Kdim + offk[None, :],
                          mask=mask_c[:, None] & mk[None, :], other=0.0).to(tl.float32)
            g_cb = tl.load(g_ptr + (bh * T + t_c)[:, None] * Kdim + offk[None, :],
                           mask=mask_c[:, None] & mk[None, :], other=0.0).to(tl.float32)
            U_b = tl.where(is_write[:, None], kk_b, e_b)
            G_b = tl.cumsum(tl.where(is_write[:, None], 0.0, g_lb), axis=0)
            cg_b = tl.cumsum(g_cb, axis=0)
            dA_ = G_b[:, None, :] - G_b[None, :, :]
            dA_ = tl.minimum(tl.maximum(dA_, -60.0), 60.0)
            A += tl.sum(U_b[:, None, :] * U_b[None, :, :] * tl.exp(dA_), axis=2)
            dM_ = cg_b[:, None, :] - G_b[None, :, :]
            dM_ = tl.minimum(tl.maximum(dM_, -60.0), 60.0)
            Mx += tl.sum(q_b[:, None, :] * U_b[None, :, :] * tl.exp(dM_), axis=2)
        ready = offs_l[None, :] <= (2 * offs_c[:, None] + 1)
        Mx = tl.where(ready, Mx, 0.0)
        # adjoints through output and end state
        dMx = tl.dot(GO, tl.trans(W), input_precision="ieee")
        dMx = tl.where(ready, dMx, 0.0)
        dW = tl.dot(tl.trans(Mx), GO, input_precision="ieee") + tl.dot(Ue, dSend, input_precision="ieee")
        dP = tl.dot(dW, tl.trans(rhs), input_precision="ieee")
        drhs = tl.dot(tl.trans(P), dW, input_precision="ieee")
        diff_tgt = target - E
        ddiff = drhs * b_l[:, None]
        db_rhs = tl.sum(drhs * diff_tgt, axis=1)
        dE = -ddiff
        tmp = tl.dot(dP, tl.trans(P), input_precision="ieee")
        dMmat = -tl.dot(tl.trans(P), tmp, input_precision="ieee")
        dAmat = tl.where(strict, dMmat, 0.0) * b_l[:, None]
        db_N = tl.sum(tl.where(strict, dMmat, 0.0) * A, axis=1)
        db_tot = db_rhs + db_N
        # state adjoint propagation (stored after the per-K-block loop below)
        dS0 = dSend * exp_total[:, None] + tl.dot(tl.trans(Qe), GO, input_precision="ieee") + tl.dot(tl.trans(Up), dE, input_precision="ieee")
        # dv (write rows only)
        dv_off = (bh * T + t_l)[:, None] * Vdim + offs_v[None, :]
        tl.store(dv_ptr + dv_off, ddiff, mask=is_write[:, None] & mask_l[:, None] & mask_v[None, :])
        # beta / gamma (per timestep, combine erase+write rows via grp)
        beta_at = beta_l
        gamma_at = gamma_l
        contrib_beta = tl.where(is_write, db_tot, -db_tot * gamma_at)
        contrib_gamma = tl.where(is_write, 0.0, db_tot * (1.0 - beta_at))
        dbeta_c = tl.sum(grp * contrib_beta[None, :], axis=1)
        dgamma_c = tl.sum(grp * contrib_gamma[None, :], axis=1)
        part1 = (vb * BH + bh) * T + t_c
        tl.store(dbeta_ptr + part1, dbeta_c, mask=mask_c)
        tl.store(dgamma_ptr + part1, dgamma_c, mask=mask_c)
        # per-K-block input gradients
        for kb in range(0, BLOCK_K, BK):
            offk = kb + tl.arange(0, BK)
            mk = offk < Kdim
            e_b = tl.load(e_ptr + (bh * T + t_l)[:, None] * Kdim + offk[None, :],
                          mask=mask_l[:, None] & mk[None, :], other=0.0).to(tl.float32)
            kk_b = tl.load(k_ptr + (bh * T + t_l)[:, None] * Kdim + offk[None, :],
                           mask=mask_l[:, None] & mk[None, :], other=0.0).to(tl.float32)
            g_lb = tl.load(g_ptr + (bh * T + t_l)[:, None] * Kdim + offk[None, :],
                           mask=mask_l[:, None] & mk[None, :], other=0.0).to(tl.float32)
            q_b = tl.load(q_ptr + (bh * T + t_c)[:, None] * Kdim + offk[None, :],
                          mask=mask_c[:, None] & mk[None, :], other=0.0).to(tl.float32)
            g_cb = tl.load(g_ptr + (bh * T + t_c)[:, None] * Kdim + offk[None, :],
                           mask=mask_c[:, None] & mk[None, :], other=0.0).to(tl.float32)
            S0_b = tl.load(sb_ptr + (bh * num_chunks + c) * Kdim * Vdim +
                           offk[:, None] * Vdim + offs_v[None, :],
                           mask=mk[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            dS_b = tl.load(ds_ptr + ds_off + offk[:, None] * Vdim + offs_v[None, :],
                           mask=mk[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            U_b = tl.where(is_write[:, None], kk_b, e_b)
            G_b = tl.cumsum(tl.where(is_write[:, None], 0.0, g_lb), axis=0)
            cg_b = tl.cumsum(g_cb, axis=0)
            total_b = tl.sum(g_cb, axis=0)
            expG_b = tl.exp(G_b)
            expcg_b = tl.exp(cg_b)
            Up_b = U_b * expG_b
            Qe_b = q_b * expcg_b
            Re_b = tl.exp(total_b[None, :] - G_b)
            Ue_b = U_b * Re_b
            dQe_b = tl.dot(GO, tl.trans(S0_b), input_precision="ieee")
            dUp_b = tl.dot(dE, tl.trans(S0_b), input_precision="ieee")
            # Mx gram grads
            dMx_ = cg_b[:, None, :] - G_b[None, :, :]
            dMx_ = tl.minimum(tl.maximum(dMx_, -60.0), 60.0)
            eMx = tl.exp(dMx_)
            contribMx = q_b[:, None, :] * U_b[None, :, :] * eMx
            dq_mx = tl.sum(dMx[:, :, None] * U_b[None, :, :] * eMx, axis=1)
            dU_mx = tl.sum(dMx[:, :, None] * q_b[:, None, :] * eMx, axis=0)
            dcg_mx = tl.sum(dMx[:, :, None] * contribMx, axis=1)
            dG_mx = -tl.sum(dMx[:, :, None] * contribMx, axis=0)
            # A gram grads
            dA_ = G_b[:, None, :] - G_b[None, :, :]
            dA_ = tl.minimum(tl.maximum(dA_, -60.0), 60.0)
            eA = tl.exp(dA_)
            contribA = U_b[:, None, :] * U_b[None, :, :] * eA
            dU_A = tl.sum(dAmat[:, :, None] * U_b[None, :, :] * eA, axis=1) + tl.sum(dAmat[:, :, None] * U_b[:, None, :] * eA, axis=0)
            dG_A = tl.sum(dAmat[:, :, None] * contribA, axis=1) - tl.sum(dAmat[:, :, None] * contribA, axis=0)
            dUe_b = tl.dot(W, tl.trans(dS_b), input_precision="ieee")
            GU_b = dU_mx + dU_A + dUp_b * expG_b + dUe_b * Re_b
            dG_b = dG_mx + dG_A + dUp_b * Up_b - dUe_b * Ue_b
            dq_b = dq_mx + dQe_b * expcg_b
            dcg_b = dcg_mx + dQe_b * Qe_b + tl.dot(grp, dG_b, input_precision="ieee")
            exp_total_b = tl.exp(total_b)
            dtotal_b = tl.sum(dS_b * exp_total_b[:, None] * S0_b, axis=1) + tl.sum(dUe_b * Ue_b, axis=0)
            dcg_b = dcg_b + tl.where(offs_c[:, None] == (CHUNK - 1), dtotal_b[None, :], 0.0)
            part_k = (vb * BH + bh) * T * Kdim + t_c[:, None] * Kdim + offk[None, :]
            tl.store(dq_ptr + part_k, dq_b, mask=mask_c[:, None] & mk[None, :])
            lk_off = (vb * BH + bh) * T * Kdim + t_l[:, None] * Kdim + offk[None, :]
            tl.store(dk_ptr + lk_off, GU_b, mask=is_write[:, None] & mask_l[:, None] & mk[None, :])
            tl.store(de_ptr + lk_off, GU_b, mask=(~is_write)[:, None] & mask_l[:, None] & mk[None, :])
            dg_b = tl.dot(suf, dcg_b, input_precision="ieee")
            tl.store(dg_ptr + part_k, dg_b, mask=mask_c[:, None] & mk[None, :])
        tl.store(ds_ptr + ds_off + offs_k[:, None] * Vdim + offs_v[None, :], dS0,
                 mask=mask_k[:, None] & mask_v[None, :])


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        ctx.orig_dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype, beta.dtype, gamma.dtype)
        ctx.dims = (B, H, T, K, V)
        ctx.scale = float(scale)
        q2 = q.reshape(BH, T, K).detach().to(torch.float32).contiguous()
        k2 = k.reshape(BH, T, K).detach().to(torch.float32).contiguous()
        e2 = e.reshape(BH, T, K).detach().to(torch.float32).contiguous()
        v2 = v.reshape(BH, T, V).detach().to(torch.float32).contiguous()
        g2 = g.reshape(BH, T, K).detach().to(torch.float32).contiguous()
        beta2 = beta.reshape(BH, T).detach().to(torch.float32).contiguous()
        gamma2 = gamma.reshape(BH, T).detach().to(torch.float32).contiguous()
        num_chunks = (T + CHUNK - 1) // CHUNK
        BLOCK_K = max(16, triton.next_power_of_2(K))
        BLOCK_V = 64
        num_v_blocks = (V + BLOCK_V - 1) // BLOCK_V
        ctx.num_chunks = num_chunks
        ctx.num_v_blocks = num_v_blocks
        ctx.BLOCK_K = BLOCK_K
        ctx.BLOCK_V = BLOCK_V
        inv = torch.empty((BH * num_chunks * LCHUNK * LCHUNK,), device=q.device, dtype=torch.float32)
        sb = torch.empty((BH * num_chunks * K * V,), device=q.device, dtype=torch.float32)
        out = torch.empty((BH, T, V), device=q.device, dtype=torch.float32)
        _inv_kernel[(BH * num_chunks,)](k2, e2, g2, beta2, gamma2, inv, T, K,
                                         num_chunks, BLOCK_K=BLOCK_K, CHUNK=CHUNK,
                                         LCHUNK=LCHUNK, BK=BK, num_warps=4, num_stages=1)
        _fwd_kernel[(BH * num_v_blocks,)](q2, k2, v2, e2, g2, beta2, gamma2, inv, sb, out,
                                          T, K, V, float(scale), num_chunks, num_v_blocks,
                                          BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, CHUNK=CHUNK,
                                          LCHUNK=LCHUNK, BK=BK, num_warps=4, num_stages=1)
        ctx.save_for_backward(q2, k2, v2, e2, g2, beta2, gamma2, inv, sb)
        return out.reshape(B, H, T, V)

    @staticmethod
    def backward(ctx, grad_out):
        q2, k2, v2, e2, g2, beta2, gamma2, inv, sb = ctx.saved_tensors
        B, H, T, K, V = ctx.dims
        BH = B * H
        num_chunks = ctx.num_chunks
        BLOCK_K = ctx.BLOCK_K
        BLOCK_V = 32
        num_v_blocks = (V + BLOCK_V - 1) // BLOCK_V
        go2 = grad_out.reshape(BH, T, V).detach().to(torch.float32).contiguous()
        dq_buf = torch.zeros((num_v_blocks, BH, T, K), device=q2.device, dtype=torch.float32)
        dk_buf = torch.zeros((num_v_blocks, BH, T, K), device=q2.device, dtype=torch.float32)
        de_buf = torch.zeros((num_v_blocks, BH, T, K), device=q2.device, dtype=torch.float32)
        dv = torch.zeros((BH, T, V), device=q2.device, dtype=torch.float32)
        dg_buf = torch.zeros((num_v_blocks, BH, T, K), device=q2.device, dtype=torch.float32)
        dbeta_buf = torch.zeros((num_v_blocks, BH, T), device=q2.device, dtype=torch.float32)
        dgamma_buf = torch.zeros((num_v_blocks, BH, T), device=q2.device, dtype=torch.float32)
        ds = torch.zeros((BH * num_v_blocks * K * V,), device=q2.device, dtype=torch.float32)
        _bwd_kernel[(BH * num_v_blocks,)](q2, k2, v2, e2, g2, beta2, gamma2, inv, sb, go2,
                                          dq_buf, dk_buf, de_buf, dv, dg_buf, dbeta_buf, dgamma_buf, ds,
                                          BH, T, K, V, ctx.scale, num_chunks, num_v_blocks,
                                          BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, CHUNK=CHUNK,
                                          LCHUNK=LCHUNK, BK=BK, num_warps=4, num_stages=1)
        dq, dk, dvv, de, dg, db, dga = ctx.orig_dtypes
        return (dq_buf.sum(0).reshape(B, H, T, K).to(dq),
                dk_buf.sum(0).reshape(B, H, T, K).to(dk),
                dv.reshape(B, H, T, V).to(dvv),
                de_buf.sum(0).reshape(B, H, T, K).to(de),
                dg_buf.sum(0).reshape(B, H, T, K).to(dg),
                dbeta_buf.sum(0).reshape(B, H, T).to(db),
                dgamma_buf.sum(0).reshape(B, H, T).to(dga),
                None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _Fn.apply(q, k, v, e, g, beta, gamma, float(scale))

