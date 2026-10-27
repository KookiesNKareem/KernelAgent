```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# (I - N)^-1 = I + N + N^2 + ... for strictly triangular (nilpotent) N,
# evaluated by repeated squaring, and its adjoint
# dL/dN = Y^T (dL/dY) Y^T for Y = (I - N)^-1.
# ---------------------------------------------------------------------------
@triton.jit
def _series_sum(N, BT: tl.constexpr, STEPS: tl.constexpr):
    offs = tl.arange(0, BT)
    S = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0).to(tl.float32)
    P = N
    for _ in range(STEPS):
        S = S + tl.dot(P, S, input_precision="ieee")
        P = tl.dot(P, P, input_precision="ieee")
    return S


@triton.jit
def _series_grad(Op, GOp):
    tmp = tl.dot(tl.trans(Op), GOp, input_precision="tf32")
    return tl.dot(tmp, tl.trans(Op), input_precision="tf32")


# ---------------------------------------------------------------------------
# Forward, kernel 1 (parallel over batch*head*chunk): everything inside one
# chunk that does not depend on the incoming state.
#
# With F_i = diag(exp(cumsum g)) and P_i = R_0 ... R_i,
# R_j = I - gamma_j u_j u_j^T, the transformed state
# Shat_i = F_i^-1 S_i P_i^-1 obeys an ordinary delta-rule recurrence.
# Per chunk this computes A = k_i/F_i, Q = q_i*F_i, Pm = M @ (k_i*F_i),
# H0 = M @ vhat (M inverts the in-chunk delta coupling, vhat = P_i^-T v_i),
# the triangular operators OutOp/EndOp that undo P_i on the outputs and on
# the outgoing state, and the chunk's total per-channel log decay.
# ---------------------------------------------------------------------------
@triton.jit
def _pre_kernel(
    q_ptr, k_ptr, v_ptr, u_ptr, g_ptr, beta_ptr, gamma_ptr,
    a_ptr, p_ptr, qt_ptr, h0_ptr, outop_ptr, endop_ptr, tot_ptr,
    T, NC,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    STEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = (pid // NC).to(tl.int64)
    chunk = pid % NC
    start = chunk * BT
    rem = T - start
    off_t = tl.arange(0, BT)
    off_k = tl.arange(0, BK)
    off_v = tl.arange(0, BV)
    m_t = off_t < rem
    row = off_t[:, None]
    col = off_t[None, :]
    upper = row <= col
    strict_lower = col < row
    strict_upper = col > row

    base_k = bh * T * BK + (start + off_t)[:, None] * BK + off_k[None, :]
    base_v = bh * T * BV + (start + off_t)[:, None] * BV + off_v[None, :]

    qq = tl.load(q_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)
    kk = tl.load(k_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)
    gg = tl.load(g_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)

    cum_g = tl.cumsum(gg, axis=0)
    total = tl.sum(tl.where(off_t[:, None] == rem - 1, cum_g, 0.0), axis=0)
    shift = 0.5 * total
    rel = cum_g - shift[None, :]
    exp_rel = tl.exp(rel)
    a_key = kk * tl.exp(-rel)
    c_key = kk * exp_rel
    q_tilde = qq * exp_rel

    out_a = bh * T * BK + (start + off_t)[:, None] * BK + off_k[None, :]
    tl.store(a_ptr + out_a, a_key, mask=m_t[:, None])
    tl.store(qt_ptr + out_a, q_tilde, mask=m_t[:, None])
    tl.store(tot_ptr + (bh * NC + chunk) * BK + off_k, total)

    vv = tl.load(v_ptr + base_v, mask=m_t[:, None], other=0.0).to(tl.float32)
    uu = tl.load(u_ptr + base_v, mask=m_t[:, None], other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + bh * T + start + off_t, mask=m_t, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + bh * T + start + off_t, mask=m_t, other=0.0).to(tl.float32)

    gram_u = tl.dot(uu, tl.trans(uu), input_precision="tf32")

    # vhat_i = P_i^-T v_i = v_i + sum_{j<=i} alpha_{j,i} u_j
    delta = gamma / (1.0 - gamma)
    b_alpha = tl.where(
        upper, delta[:, None] * tl.dot(uu, tl.trans(vv), input_precision="tf32"), 0.0
    )
    alpha_op = _series_sum(delta[:, None] * tl.where(strict_upper, gram_u, 0.0), BT, STEPS)
    alpha = tl.dot(alpha_op, b_alpha, input_precision="tf32")
    v_hat = vv + tl.dot(tl.trans(alpha), uu, input_precision="tf32")

    # In-chunk delta-rule coupling inverse.  h = (I + diag(beta) K)^-1
    # diag(beta) rhs, so the folded operator is left_op @ diag(beta):
    # beta scales the columns of left_op.
    ka = tl.dot(c_key, tl.trans(a_key), input_precision="tf32")
    left_op = _series_sum(-(beta[:, None] * tl.where(strict_lower, ka, 0.0)), BT, STEPS)
    mb = left_op * beta[None, :]
    h0 = tl.dot(mb, v_hat, input_precision="tf32")
    pmat = tl.dot(mb, c_key, input_precision="tf32")
    tl.store(p_ptr + out_a, pmat, mask=m_t[:, None])
    tl.store(h0_ptr + base_v, h0, mask=m_t[:, None])

    op_base = (bh * NC + chunk) * BT * BT + off_t[:, None] * BT + off_t[None, :]
    out_op = _series_sum(-(gamma[:, None] * tl.where(strict_lower, gram_u, 0.0)), BT, STEPS)
    end_op = _series_sum(-(tl.where(strict_upper, gram_u, 0.0) * gamma[None, :]), BT, STEPS)
    tl.store(outop_ptr + op_base, out_op)
    tl.store(endop_ptr + op_base, end_op)


# ---------------------------------------------------------------------------
# Forward, kernel 2 (one program per batch*head): sequential scan over
# chunks.  The [K, V] state lives in a scratch buffer and is only touched
# in KB x BV blocks, so no [128, 128] dot operand is ever staged.  The
# state at every chunk boundary is checkpointed for the backward.
# ---------------------------------------------------------------------------
@triton.jit
def _scan_kernel(
    a_ptr, p_ptr, qt_ptr, h0_ptr, outop_ptr, endop_ptr, tot_ptr,
    u_ptr, gamma_ptr, o_ptr, state_ptr, ckpt_ptr,
    T, NC, scale,
    BT: tl.constexpr, BK: tl.constexpr, KB: tl.constexpr,
    BV: tl.constexpr, NUM_KB: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    off_t = tl.arange(0, BT)
    off_kb = tl.arange(0, KB)
    off_v = tl.arange(0, BV)
    row = off_t[:, None]
    col = off_t[None, :]
    causal = col <= row
    upper = row <= col

    for chunk in range(NC):
        start = chunk * BT
        rem = T - start
        m_t = off_t < rem
        base_v = bh * T * BV + (start + off_t)[:, None] * BV + off_v[None, :]

        for kb in range(NUM_KB):
            koff = kb * KB
            s_ptr = state_ptr + bh * BK * BV + (koff + off_kb)[:, None] * BV + off_v[None, :]
            tl.store(
                ckpt_ptr + ((bh * NC + chunk) * BK + (koff + off_kb)[:, None]) * BV + off_v[None, :],
                tl.load(s_ptr),
            )

        pred = tl.zeros([BT, BV], dtype=tl.float32)
        opred = tl.zeros([BT, BV], dtype=tl.float32)
        aq = tl.zeros([BT, BT], dtype=tl.float32)
        for kb in range(NUM_KB):
            koff = kb * KB
            base_k = bh * T * BK + (start + off_t)[:, None] * BK + (koff + off_kb)[None, :]
            total = tl.load(tot_ptr + (bh * NC + chunk) * BK + koff + off_kb)
            shift = 0.5 * total
            s_ptr = state_ptr + bh * BK * BV + (koff + off_kb)[:, None] * BV + off_v[None, :]
            s_hat = tl.load(s_ptr) * tl.exp(shift)[:, None]
            pmat = tl.load(p_ptr + base_k, mask=m_t[:, None], other=0.0)
            qt = tl.load(qt_ptr + base_k, mask=m_t[:, None], other=0.0)
            ak = tl.load(a_ptr + base_k, mask=m_t[:, None], other=0.0)
            pred += tl.dot(pmat, s_hat, input_precision="tf32")
            opred += tl.dot(qt, s_hat, input_precision="tf32")
            aq += tl.dot(qt, tl.trans(ak), input_precision="tf32")

        h = tl.load(h0_ptr + base_v, mask=m_t[:, None], other=0.0) - pred
        uu = tl.load(u_ptr + base_v, mask=m_t[:, None], other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + bh * T + start + off_t, mask=m_t, other=0.0).to(tl.float32)
        op_base = (bh * NC + chunk) * BT * BT + off_t[:, None] * BT + off_t[None, :]

        aq = tl.where(causal, aq, 0.0)
        o_hat = opred + tl.dot(aq, h, input_precision="tf32")
        b_out = tl.where(
            upper,
            gamma[:, None] * tl.dot(uu, tl.trans(o_hat), input_precision="tf32"),
            0.0,
        )
        out_op = tl.load(outop_ptr + op_base)
        coeff = tl.dot(out_op, b_out, input_precision="tf32")
        coeff = tl.where(upper, coeff, 0.0)
        out = (o_hat - tl.dot(tl.trans(coeff), uu, input_precision="tf32")) * scale
        tl.store(o_ptr + base_v, out, mask=m_t[:, None])

        end_op = tl.load(endop_ptr + op_base)
        for kb in range(NUM_KB):
            koff = kb * KB
            base_k = bh * T * BK + (start + off_t)[:, None] * BK + (koff + off_kb)[None, :]
            total = tl.load(tot_ptr + (bh * NC + chunk) * BK + koff + off_kb)
            shift = 0.5 * total
            s_ptr = state_ptr + bh * BK * BV + (koff + off_kb)[:, None] * BV + off_v[None, :]
            s_hat = tl.load(s_ptr) * tl.exp(shift)[:, None]
            ak = tl.load(a_ptr + base_k, mask=m_t[:, None], other=0.0)
            x_hat = s_hat + tl.dot(tl.trans(ak), h, input_precision="tf32")
            e_mat = tl.dot(x_hat, tl.trans(uu), input_precision="tf32")
            d_coeff = tl.dot(e_mat * gamma[None, :], end_op, input_precision="tf32")
            x_end = x_hat - tl.dot(d_coeff, uu, input_precision="tf32")
            tl.store(s_ptr, x_end * tl.exp(shift)[:, None])


# ---------------------------------------------------------------------------
# Backward, kernel 1 (one program per batch*head): reverse scan over
# chunks.  Carries the state adjoint dS in a scratch buffer and emits
# per-chunk gradients wrt the precomputed intermediates (A, Q, Pm, H0,
# U, gamma, OutOp, EndOp, total decay).  Each chunk first recomputes its
# forward tile quantities from the checkpointed incoming state, then
# applies the adjoint of the state-end transform (giving the total
# adjoint of h), then the adjoints of the output and write transforms.
# ---------------------------------------------------------------------------
@triton.jit
def _bscan_kernel(
    a_ptr, p_ptr, qt_ptr, h0_ptr, outop_ptr, endop_ptr, tot_ptr,
    u_ptr, gamma_ptr, do_ptr, ckpt_ptr,
    da_ptr, dqt_ptr, dp_ptr, dh0_ptr, du_ptr, dgamma_ptr,
    doutop_ptr, dendop_ptr, dtot_ptr, gstate_ptr, tmp_ptr,
    T, NC, scale,
    BT: tl.constexpr, BK: tl.constexpr, KB: tl.constexpr,
    BV: tl.constexpr, NUM_KB: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    off_t = tl.arange(0, BT)
    off_kb = tl.arange(0, KB)
    off_v = tl.arange(0, BV)
    row = off_t[:, None]
    col = off_t[None, :]
    causal = col <= row
    upper = row <= col

    for ci in range(NC):
        chunk = NC - 1 - ci
        start = chunk * BT
        rem = T - start
        m_t = off_t < rem
        base_v = bh * T * BV + (start + off_t)[:, None] * BV + off_v[None, :]
        op_base = (bh * NC + chunk) * BT * BT + off_t[:, None] * BT + off_t[None, :]

        uu = tl.load(u_ptr + base_v, mask=m_t[:, None], other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + bh * T + start + off_t, mask=m_t, other=0.0).to(tl.float32)
        out_op = tl.load(outop_ptr + op_base)
        end_op = tl.load(endop_ptr + op_base)
        gout = tl.load(do_ptr + base_v, mask=m_t[:, None], other=0.0) * scale

        # ---- recompute chunk forward (h, o_hat) from the checkpoint ----
        pred = tl.zeros([BT, BV], dtype=tl.float32)
        opred = tl.zeros([BT, BV], dtype=tl.float32)
        aq = tl.zeros([BT, BT], dtype=tl.float32)
        for kb in range(NUM_KB):
            koff = kb * KB
            base_k = bh * T * BK + (start + off_t)[:, None] * BK + (koff + off_kb)[None, :]
            total = tl.load(tot_ptr + (bh * NC + chunk) * BK + koff + off_kb)
            s_in = tl.load(
                ckpt_ptr + ((bh * NC + chunk) * BK + (koff + off_kb)[:, None]) * BV + off_v[None, :]
            )
            s_hat = s_in * tl.exp(0.5 * total)[:, None]
            pmat = tl.load(p_ptr + base_k, mask=m_t[:, None], other=0.0)
            qt = tl.load(qt_ptr + base_k, mask=m_t[:, None], other=0.0)
            ak = tl.load(a_ptr + base_k, mask=m_t[:, None], other=0.0)
            pred += tl.dot(pmat, s_hat, input_precision="tf32")
            opred += tl.dot(qt, s_hat, input_precision="tf32")
            aq += tl.dot(qt, tl.trans(ak), input_precision="tf32")
        h = tl.load(h0_ptr + base_v, mask=m_t[:, None], other=0.0) - pred
        aq_c = tl.where(causal, aq, 0.0)
        o_hat = opred + tl.dot(aq_c, h, input_precision="tf32")

        # ---- adjoint of the output transform o = o_hat - Cmask^T U ----
        bout_pre = tl.dot(uu, tl.trans(o_hat), input_precision="tf32")
        bout = tl.where(upper, gamma[:, None] * bout_pre, 0.0)
        coeff = tl.where(upper, tl.dot(out_op, bout, input_precision="tf32"), 0.0)
        gcoeff = tl.where(upper, -tl.dot(uu, tl.trans(gout), input_precision="tf32"), 0.0)
        tl.store(doutop_ptr + op_base, tl.dot(gcoeff, tl.trans(bout), input_precision="tf32"))
        gbout = tl.where(
            upper, tl.dot(tl.trans(out_op), gcoeff, input_precision="tf32"), 0.0
        )
        gb_gamma = gbout * gamma[:, None]
        go_hat = gout + tl.dot(tl.trans(gb_gamma), uu, input_precision="tf32")
        gu = tl.dot(gb_gamma, o_hat, input_precision="tf32") - tl.dot(
            coeff, gout, input_precision="tf32"
        )
        dgamma = tl.sum(gbout * bout_pre, axis=1)

        gh = tl.dot(tl.trans(aq_c), go_hat, input_precision="tf32")
        gaq = tl.where(causal, tl.dot(go_hat, tl.trans(h), input_precision="tf32"), 0.0)

        # ---- adjoint of the state-end transform, per K block ----
        gendop = tl.zeros([BT, BT], dtype=tl.float32)
        gh_state = tl.zeros([BT, BV], dtype=tl.float32)
        for kb in range(NUM_KB):
            koff = kb * KB
            base_k = bh * T * BK + (start + off_t)[:, None] * BK + (koff + off_kb)[None, :]
            total = tl.load(tot_ptr + (bh * NC + chunk) * BK + koff + off_kb)
            e_sh = tl.exp(0.5 * total)
            s_in = tl.load(
                ckpt_ptr + ((bh * NC + chunk) * BK + (koff + off_kb)[:, None]) * BV + off_v[None, :]
            )
            s_hat = s_in * e_sh[:, None]
            ak = tl.load(a_ptr + base_k, mask=m_t[:, None], other=0.0)
            qt = tl.load(qt_ptr + base_k, mask=m_t[:, None], other=0.0)
            gs_ptr = gstate_ptr + bh * BK * BV + (koff + off_kb)[:, None] * BV + off_v[None, :]
            gs_out = tl.load(gs_ptr)

            x_hat = s_hat + tl.dot(tl.trans(ak), h, input_precision="tf32")
            e_mat = tl.dot(x_hat, tl.trans(uu), input_precision="tf32")
            y_mat = e_mat * gamma[None, :]
            dcoef = tl.dot(y_mat, end_op, input_precision="tf32")

            gx_end = e_sh[:, None] * gs_out
            gdcoef = -tl.dot(gx_end, tl.trans(uu), input_precision="tf32")
            gy = tl.dot(gdcoef, tl.trans(end_op), input_precision="tf32")
            gendop += tl.dot(tl.trans(y_mat), gdcoef, input_precision="tf32")
            ge_mat = gy * gamma[None, :]
            dgamma += tl.sum(gy * e_mat, axis=0)
            gx_hat = gx_end + tl.dot(ge_mat, uu, input_precision="tf32")
            gu += tl.dot(tl.trans(ge_mat), x_hat, input_precision="tf32")
            gu -= tl.dot(tl.trans(dcoef), gx_end, input_precision="tf32")
            gh_state += tl.dot(ak, gx_hat, input_precision="tf32")

            ga = tl.dot(tl.trans(gaq), qt, input_precision="tf32") + tl.dot(
                h, tl.trans(gx_hat), input_precision="tf32"
            )
            tl.store(da_ptr + base_k, ga, mask=m_t[:, None])
            tl.store(
                tmp_ptr + bh * BK * BV + (koff + off_kb)[:, None] * BV + off_v[None, :],
                gx_hat,
            )
        tl.store(dendop_ptr + op_base, gendop)
        gh += gh_state
        tl.store(dh0_ptr + base_v, gh, mask=m_t[:, None])
        tl.store(du_ptr + base_v, gu, mask=m_t[:, None])
        tl.store(dgamma_ptr + bh * T + start + off_t, dgamma, mask=m_t)

        # ---- adjoint wrt the incoming (transformed) state and Pm/Q ----
        for kb in range(NUM_KB):
            koff = kb * KB
            base_k = bh * T * BK + (start + off_t)[:, None] * BK + (koff + off_kb)[None, :]
            total = tl.load(tot_ptr + (bh * NC + chunk) * BK + koff + off_kb)
            e_sh = tl.exp(0.5 * total)
            s_in = tl.load(
                ckpt_ptr + ((bh * NC + chunk) * BK + (koff + off_kb)[:, None]) * BV + off_v[None, :]
            )
            s_hat = s_in * e_sh[:, None]
            pmat = tl.load(p_ptr + base_k, mask=m_t[:, None], other=0.0)
            qt = tl.load(qt_ptr + base_k, mask=m_t[:, None], other=0.0)
            ak = tl.load(a_ptr + base_k, mask=m_t[:, None], other=0.0)
            gs_ptr = gstate_ptr + bh * BK * BV + (koff + off_kb)[:, None] * BV + off_v[None, :]
            gs_out = tl.load(gs_ptr)
            gx_hat = tl.load(
                tmp_ptr + bh * BK * BV + (koff + off_kb)[:, None] * BV + off_v[None, :]
            )

            gs_hat = (
                tl.dot(tl.trans(qt), go_hat, input_precision="tf32")
                - tl.dot(tl.trans(pmat), gh, input_precision="tf32")
                + gx_hat
            )
            tl.store(gs_ptr, e_sh[:, None] * gs_hat)

            x_hat = s_hat + tl.dot(tl.trans(ak), h, input_precision="tf32")
            e_mat = tl.dot(x_hat, tl.trans(uu), input_precision="tf32")
            dcoef = tl.dot(e_mat * gamma[None, :], end_op, input_precision="tf32")
            x_end = x_hat - tl.dot(dcoef, uu, input_precision="tf32")
            dtot = 0.5 * e_sh * tl.sum(gs_out * x_end + gs_hat * s_in, axis=1)
            tl.store(dtot_ptr + (bh * NC + chunk) * BK + koff + off_kb, dtot)

            gqt = tl.dot(go_hat, tl.trans(s_hat), input_precision="tf32") + tl.dot(
                gaq, ak, input_precision="tf32"
            )
            tl.store(dqt_ptr + base_k, gqt, mask=m_t[:, None])
            tl.store(
                dp_ptr + base_k,
                -tl.dot(gh, tl.trans(s_hat), input_precision="tf32"),
                mask=m_t[:, None],
            )


# ---------------------------------------------------------------------------
# Backward, kernel 2 (parallel over batch*head*chunk): differentiates the
# precomputation.  Recomputes the chunk's forward intermediates, then
# propagates the intermediate gradients through the folded delta-rule
# operator (beta on the columns), the four triangular series (using
# _series_grad), the U Gram matrix, alpha/vhat, and finally the decay
# rescaling (reverse cumulative sum) to the raw inputs.
# ---------------------------------------------------------------------------
@triton.jit
def _bpre_kernel(
    q_ptr, k_ptr, v_ptr, u_ptr, g_ptr, beta_ptr, gamma_ptr,
    da_ptr, dqt_ptr, dp_ptr, dh0_ptr, du_ptr, dgamma_ptr,
    doutop_ptr, dendop_ptr, dtot_ptr,
    dq_ptr, dk_ptr, dv_ptr, du_out_ptr, dg_ptr, dbeta_ptr, dgamma_out_ptr,
    T, NC,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    STEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = (pid // NC).to(tl.int64)
    chunk = pid % NC
    start = chunk * BT
    rem = T - start
    off_t = tl.arange(0, BT)
    off_k = tl.arange(0, BK)
    off_v = tl.arange(0, BV)
    m_t = off_t < rem
    row = off_t[:, None]
    col = off_t[None, :]
    upper = row <= col
    strict_lower = col < row
    strict_upper = col > row

    base_k = bh * T * BK + (start + off_t)[:, None] * BK + off_k[None, :]
    base_v = bh * T * BV + (start + off_t)[:, None] * BV + off_v[None, :]
    op_base = (bh * NC + chunk) * BT * BT + off_t[:, None] * BT + off_t[None, :]

    qq = tl.load(q_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)
    kk = tl.load(k_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)
    gg = tl.load(g_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)
    vv = tl.load(v_ptr + base_v, mask=m_t[:, None], other=0.0).to(tl.float32)
    uu = tl.load(u_ptr + base_v, mask=m_t[:, None], other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + bh * T + start + off_t, mask=m_t, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + bh * T + start + off_t, mask=m_t, other=0.0).to(tl.float32)

    gda = tl.load(da_ptr + base_k, mask=m_t[:, None], other=0.0)
    gdqt = tl.load(dqt_ptr + base_k, mask=m_t[:, None], other=0.0)
    gdp = tl.load(dp_ptr + base_k, mask=m_t[:, None], other=0.0)
    gdh0 = tl.load(dh0_ptr + base_v, mask=m_t[:, None], other=0.0)
    gu = tl.load(du_ptr + base_v, mask=m_t[:, None], other=0.0)
    dgamma = tl.load(dgamma_ptr + bh * T + start + off_t, mask=m_t, other=0.0)
    gdoutop = tl.load(doutop_ptr + op_base)
    gdendop = tl.load(dendop_ptr + op_base)
    gdtot = tl.load(dtot_ptr + (bh * NC + chunk) * BK + off_k)

    # ---- recompute forward intermediates ----
    cum_g = tl.cumsum(gg, axis=0)
    total = tl.sum(tl.where(off_t[:, None] == rem - 1, cum_g, 0.0), axis=0)
    shift = 0.5 * total
    rel = cum_g - shift[None, :]
    exp_rel = tl.exp(rel)
    exp_nrel = tl.exp(-rel)
    a_key = kk * exp_nrel
    c_key = kk * exp_rel
    q_tilde = qq * exp_rel

    gram_u = tl.dot(uu, tl.trans(uu), input_precision="tf32")
    delta = gamma / (1.0 - gamma)
    muv = tl.dot(uu, tl.trans(vv), input_precision="tf32")
    b_alpha = tl.where(upper, delta[:, None] * muv, 0.0)
    alpha_op = _series_sum(delta[:, None] * tl.where(strict_upper, gram_u, 0.0), BT, STEPS)
    alpha = tl.dot(alpha_op, b_alpha, input_precision="tf32")
    v_hat = vv + tl.dot(tl.trans(alpha), uu, input_precision="tf32")

    ka = tl.dot(c_key, tl.trans(a_key), input_precision="tf32")
    left_op = _series_sum(-(beta[:, None] * tl.where(strict_lower, ka, 0.0)), BT, STEPS)
    mb = left_op * beta[None, :]
    out_op = _series_sum(-(gamma[:, None] * tl.where(strict_lower, gram_u, 0.0)), BT, STEPS)
    end_op = _series_sum(-(tl.where(strict_upper, gram_u, 0.0) * gamma[None, :]), BT, STEPS)

    # ---- folded operator mb = left_op @ diag(beta) ----
    gmb = tl.dot(gdp, tl.trans(c_key), input_precision="tf32") + tl.dot(
        gdh0, tl.trans(v_hat), input_precision="tf32"
    )
    gc = tl.dot(tl.trans(mb), gdp, input_precision="tf32")
    gv_hat = tl.dot(tl.trans(mb), gdh0, input_precision="tf32")
    gleft = gmb * beta[None, :]
    dbeta = tl.sum(gmb * left_op, axis=0)

    gn = tl.where(strict_lower, _series_grad(left_op, gleft), 0.0)
    gka = -beta[:, None] * gn
    dbeta += tl.sum(gn * (-tl.where(strict_lower, ka, 0.0)), axis=1)
    gc += tl.dot(gka, a_key, input_precision="tf32")
    ga = gda + tl.dot(tl.trans(gka), c_key, input_precision="tf32")

    # ---- triangular operators on the Gram matrix ----
    ggram = tl.zeros([BT, BT], dtype=tl.float32)
    gn = tl.where(strict_lower, _series_grad(out_op, gdoutop), 0.0)
    ggram += -gamma[:, None] * gn
    dgamma += tl.sum(gn * (-tl.where(strict_lower, gram_u, 0.0)), axis=1)
    gn = tl.where(strict_upper, _series_grad(end_op, gdendop), 0.0)
    ggram += -gn * gamma[None, :]
    dgamma += tl.sum(gn * (-tl.where(strict_upper, gram_u, 0.0)), axis=0)

    # ---- vhat / alpha ----
    gv = gv_hat
    galpha = tl.dot(uu, tl.trans(gv_hat), input_precision="tf32")
    gu += tl.dot(alpha, gv_hat, input_precision="tf32")
    gb_alpha = tl.dot(tl.trans(alpha_op), galpha, input_precision="tf32")
    galpha_op = tl.dot(galpha, tl.trans(b_alpha), input_precision="tf32")
    gn = tl.where(strict_upper, _series_grad(alpha_op, galpha_op), 0.0)
    ggram += delta[:, None] * gn
    ddelta = tl.sum(gn * tl.where(strict_upper, gram_u, 0.0), axis=1)
    gb = tl.where(upper, gb_alpha, 0.0)
    ddelta += tl.sum(gb * muv, axis=1)
    gmuv = gb * delta[:, None]
    gu += tl.dot(gmuv, vv, input_precision="tf32")
    gv += tl.dot(tl.trans(gmuv), uu, input_precision="tf32")
    dgamma += ddelta / ((1.0 - gamma) * (1.0 - gamma))

    gu += tl.dot(ggram + tl.trans(ggram), uu, input_precision="tf32")

    # ---- decay rescaling and reverse cumulative sum ----
    grel = gc * c_key + gdqt * q_tilde - ga * a_key
    gcum = grel
    gshift = -tl.sum(grel, axis=0)
    gtotal = 0.5 * gshift + gdtot
    gcum += tl.where(off_t[:, None] == rem - 1, gtotal[None, :], 0.0)
    ones_upper = tl.where(upper, 1.0, 0.0).to(tl.float32)
    dg_tile = tl.dot(ones_upper, gcum, input_precision="tf32")

    tl.store(dq_ptr + base_k, gdqt * exp_rel, mask=m_t[:, None])
    tl.store(dk_ptr + base_k, ga * exp_nrel + gc * exp_rel, mask=m_t[:, None])
    tl.store(dg_ptr + base_k, dg_tile, mask=m_t[:, None])
    tl.store(dv_ptr + base_v, gv, mask=m_t[:, None])
    tl.store(du_out_ptr + base_v, gu, mask=m_t[:, None])
    tl.store(dbeta_ptr + bh * T + start + off_t, dbeta, mask=m_t)
    tl.store(dgamma_out_ptr + bh * T + start + off_t, dgamma, mask=m_t)


# ---------------------------------------------------------------------------
# Recurrent fallback kernels (shapes / regimes where the chunk transform
# is not applicable).  Forward checkpoints the state every BC steps;
# backward recomputes each block from its checkpoint and walks it in
# reverse with the exact adjoint recurrences.
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, u_ptr, g_ptr, beta_ptr, gamma_ptr,
    o_ptr, ckpt_ptr,
    T, scale, NC,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    off_k = tl.arange(0, BK)
    off_v = tl.arange(0, BV)
    S = tl.zeros([BK, BV], dtype=tl.float32)

    for t in range(T):
        if t % BC == 0:
            chunk = t // BC
            tl.store(
                ckpt_ptr
                + ((bh * NC + chunk) * BK + off_k[:, None]) * BV
                + off_v[None, :],
                S,
            )

        row = bh * T + t
        g = tl.load(g_ptr + row * BK + off_k).to(tl.float32)
        u = tl.load(u_ptr + row * BV + off_v).to(tl.float32)
        kk = tl.load(k_ptr + row * BK + off_k).to(tl.float32)
        vv = tl.load(v_ptr + row * BV + off_v).to(tl.float32)
        qq = tl.load(q_ptr + row * BK + off_k).to(tl.float32)
        beta = tl.load(beta_ptr + row).to(tl.float32)
        gamma = tl.load(gamma_ptr + row).to(tl.float32)

        S = S * tl.exp(g)[:, None]
        su = tl.sum(S * u[None, :], axis=1)
        S = S - gamma * su[:, None] * u[None, :]
        pred = tl.sum(S * kk[:, None], axis=0)
        S = S + kk[:, None] * (beta * (vv - pred))[None, :]
        out = scale * tl.sum(S * qq[:, None], axis=0)
        tl.store(o_ptr + row * BV + off_v, out)


@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, u_ptr, g_ptr, beta_ptr, gamma_ptr,
    do_ptr, ckpt_ptr, scratch_ptr,
    dq_ptr, dk_ptr, dv_ptr, du_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
    T, scale, NC,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    off_k = tl.arange(0, BK)
    off_v = tl.arange(0, BV)
    A = tl.zeros([BK, BV], dtype=tl.float32)

    for ci in range(NC):
        chunk = NC - 1 - ci
        start = chunk * BC
        chunk_len = BC
        if start + BC > T:
            chunk_len = T - start

        S = tl.load(
            ckpt_ptr
            + ((bh * NC + chunk) * BK + off_k[:, None]) * BV
            + off_v[None, :]
        )
        for i in range(chunk_len):
            row = bh * T + start + i
            g = tl.load(g_ptr + row * BK + off_k).to(tl.float32)
            u = tl.load(u_ptr + row * BV + off_v).to(tl.float32)
            kk = tl.load(k_ptr + row * BK + off_k).to(tl.float32)
            vv = tl.load(v_ptr + row * BV + off_v).to(tl.float32)
            beta = tl.load(beta_ptr + row).to(tl.float32)
            gamma = tl.load(gamma_ptr + row).to(tl.float32)

            S = S * tl.exp(g)[:, None]
            tl.store(
                scratch_ptr
                + ((bh * BC + i) * BK + off_k[:, None]) * BV
                + off_v[None, :],
                S,
            )
            su = tl.sum(S * u[None, :], axis=1)
            S = S - gamma * su[:, None] * u[None, :]
            pred = tl.sum(S * kk[:, None], axis=0)
            S = S + kk[:, None] * (beta * (vv - pred))[None, :]

        for jj in range(chunk_len):
            i = chunk_len - 1 - jj
            t = start + i
            row = bh * T + t

            g = tl.load(g_ptr + row * BK + off_k).to(tl.float32)
            u = tl.load(u_ptr + row * BV + off_v).to(tl.float32)
            kk = tl.load(k_ptr + row * BK + off_k).to(tl.float32)
            vv = tl.load(v_ptr + row * BV + off_v).to(tl.float32)
            qq = tl.load(q_ptr + row * BK + off_k).to(tl.float32)
            dov = tl.load(do_ptr + row * BV + off_v).to(tl.float32)
            beta = tl.load(beta_ptr + row).to(tl.float32)
            gamma = tl.load(gamma_ptr + row).to(tl.float32)

            Sa = tl.load(
                scratch_ptr
                + ((bh * BC + i) * BK + off_k[:, None]) * BV
                + off_v[None, :]
            )

            w = tl.sum(Sa * u[None, :], axis=1)
            pred = tl.sum(Sa * kk[:, None], axis=0) - gamma * tl.sum(w * kk) * u
            r = vv - pred

            Sa_do = tl.sum(Sa * dov[None, :], axis=1)
            dq_vec = scale * (
                Sa_do - gamma * w * tl.sum(u * dov)
                + beta * kk * tl.sum(r * dov)
            )
            tl.store(dq_ptr + row * BK + off_k, dq_vec)

            A = A + scale * qq[:, None] * dov[None, :]

            kA = tl.sum(kk[:, None] * A, axis=0)
            dr = beta * kA
            tl.store(dv_ptr + row * BV + off_v, dr)
            tl.store(dbeta_ptr + row, tl.sum(kA * r))

            Ar = tl.sum(A * r[None, :], axis=1)
            Sa_dr = tl.sum(Sa * dr[None, :], axis=1)
            dk_vec = beta * Ar - (Sa_dr - gamma * w * tl.sum(u * dr))
            tl.store(dk_ptr + row * BK + off_k, dk_vec)

            A = A - kk[:, None] * dr[None, :]
            z = tl.sum(A * u[None, :], axis=1)
            Bt_w = tl.sum(A * w[:, None], axis=0)
            SaT_z = tl.sum(Sa * z[:, None], axis=0)
            tl.store(du_ptr + row * BV + off_v, -gamma * (Bt_w + SaT_z))
            tl.store(dgamma_ptr + row, -tl.sum(w * z))

            A = A - gamma * z[:, None] * u[None, :]
            tl.store(dg_ptr + row * BK + off_k, tl.sum(A * Sa, axis=1))
            A = A * tl.exp(g)[:, None]


_BC = 32


def _pad_last_dim(x, padded_size):
    if x.shape[-1] == padded_size:
        return x.contiguous()
    result = torch.zeros((*x.shape[:-1], padded_size), device=x.device, dtype=x.dtype)
    result[..., : x.shape[-1]] = x
    return result


def _num_warps(state_elements):
    if state_elements <= 4096:
        return 4
    if state_elements <= 16384:
        return 16
    return 32


class _LayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, u, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        BK = max(16, triton.next_power_of_2(K))
        BV = max(16, triton.next_power_of_2(V))
        dev = q.device

        q_ = _pad_last_dim(q, BK).reshape(BH, T, BK)
        k_ = _pad_last_dim(k, BK).reshape(BH, T, BK)
        g_ = _pad_last_dim(g, BK).reshape(BH, T, BK)
        v_ = _pad_last_dim(v, BV).reshape(BH, T, BV)
        u_ = _pad_last_dim(u, BV).reshape(BH, T, BV)
        beta_ = beta.contiguous().reshape(BH, T)
        gamma_ = gamma.contiguous().reshape(BH, T)
        out_ = torch.empty((BH, T, BV), device=dev, dtype=torch.float32)

        # The chunk transform divides by (1 - gamma) and exponentiates the
        # cumulative decay, so use it only where it is well conditioned;
        # everything else goes to the recurrent kernels.
        use_chunk = False
        BT = 32
        if BK <= 128 and BV <= 128:
            b_lo, b_hi = (float(x) for x in beta.aminmax())
            ga_lo, ga_hi = (float(x) for x in gamma.aminmax())
            g_lo, g_hi = (float(x) for x in g.aminmax())
            if 0.0 <= b_lo and b_hi <= 1.0 and 0.0 <= ga_lo and g_hi <= 0.0 and g_lo >= -30.0:
                if ga_hi <= 0.90 and 32.0 * abs(g_lo) <= 40.0:
                    BT, use_chunk = 32, True
                elif ga_hi <= 0.99 and 16.0 * abs(g_lo) <= 40.0:
                    BT, use_chunk = 16, True

        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
        ctx.BK, ctx.BV = BK, BV
        ctx.scale = float(scale)
        ctx.use_chunk = use_chunk
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, u.dtype,
                      g.dtype, beta.dtype, gamma.dtype)

        if use_chunk:
            NC = (T + BT - 1) // BT
            steps = (BT - 1).bit_length()
            a_ = torch.empty((BH, T, BK), device=dev, dtype=torch.float32)
            p_ = torch.empty((BH, T, BK), device=dev, dtype=torch.float32)
            qt_ = torch.empty((BH, T, BK), device=dev, dtype=torch.float32)
            h0_ = torch.empty((BH, T, BV), device=dev, dtype=torch.float32)
            outop_ = torch.empty((BH, NC, BT, BT), device=dev, dtype=torch.float32)
            endop_ = torch.empty((BH, NC, BT, BT), device=dev, dtype=torch.float32)
            tot_ = torch.empty((BH, NC, BK), device=dev, dtype=torch.float32)
            state = torch.zeros((BH, BK, BV), device=dev, dtype=torch.float32)
            ckpt = torch.empty((BH, NC, BK, BV), device=dev, dtype=torch.float32)

            _pre_kernel[(BH * NC,)](
                q_, k_, v_, u_, g_, beta_, gamma_,
                a_, p_, qt_, h0_, outop_, endop_, tot_,
                T, NC,
                BT=BT, BK=BK, BV=BV, STEPS=steps,
                num_warps=8, num_stages=1,
            )
            KB = min(64, BK)
            _scan_kernel[(BH,)](
                a_, p_, qt_, h0_, outop_, endop_, tot_,
                u_, gamma_, out_, state, ckpt,
                T, NC, float(scale),
                BT=BT, BK=BK, KB=KB, BV=BV, NUM_KB=BK // KB,
                num_warps=8, num_stages=1,
            )
            ctx.BT, ctx.NC = BT, NC
            ctx.save_for_backward(q_, k_, v_, u_, g_, beta_, gamma_,
                                  a_, p_, qt_, h0_, outop_, endop_, tot_, ckpt)
        else:
            NC = (T + _BC - 1) // _BC
            ckpt = torch.zeros((BH, NC, BK, BV), device=dev, dtype=torch.float32)
            _fwd_kernel[(BH,)](
                q_, k_, v_, u_, g_, beta_, gamma_, out_, ckpt,
                T, float(scale), NC,
                BK=BK, BV=BV, BC=_BC,
                num_warps=_num_warps(BK * BV), num_stages=1,
            )
            ctx.NC = NC
            ctx.save_for_backward(q_, k_, v_, u_, g_, beta_, gamma_, ckpt)

        return out_[:, :, :V].reshape(B, H, T, V)

    @staticmethod
    def backward(ctx, grad_out):
        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        BK, BV = ctx.BK, ctx.BV
        BH = B * H
        dev = grad_out.device

        go_ = _pad_last_dim(grad_out, BV).reshape(BH, T, BV)
        if go_.dtype != torch.float32:
            go_ = go_.float()

        dq_ = torch.zeros((BH, T, BK), device=dev, dtype=torch.float32)
        dk_ = torch.zeros((BH, T, BK), device=dev, dtype=torch.float32)
        dg_ = torch.zeros((BH, T, BK), device=dev, dtype=torch.float32)
        dv_ = torch.zeros((BH, T, BV), device=dev, dtype=torch.float32)
        du_ = torch.zeros((BH, T, BV), device=dev, dtype=torch.float32)
        dbeta_ = torch.zeros((BH, T), device=dev, dtype=torch.float32)
        dgamma_ = torch.zeros((BH, T), device=dev, dtype=torch.float32)

        if ctx.use_chunk:
            (q_, k_, v_, u_, g_, beta_, gamma_,
             a_, p_, qt_, h0_, outop_, endop_, tot_, ckpt) = ctx.saved_tensors
            BT, NC = ctx.BT, ctx.NC
            steps = (BT - 1).bit_length()

            da_ = torch.empty((BH, T, BK), device=dev, dtype=torch.float32)
            dp_ = torch.empty((BH, T, BK), device=dev, dtype=torch.float32)
            dqt_ = torch.empty((BH, T, BK), device=dev, dtype=torch.float32)
            dh0_ = torch.empty((BH, T, BV), device=dev, dtype=torch.float32)
            du_s_ = torch.empty((BH, T, BV), device=dev, dtype=torch.float32)
            dgam_s_ = torch.empty((BH, T), device=dev, dtype=torch.float32)
            doutop_ = torch.empty((BH, NC, BT, BT), device=dev, dtype=torch.float32)
            dendop_ = torch.empty((BH, NC, BT, BT), device=dev, dtype=torch.float32)
            dtot_ = torch.empty((BH, NC, BK), device=dev, dtype=torch.float32)
            gstate = torch.zeros((BH, BK, BV), device=dev, dtype=torch.float32)
            tmp = torch.empty((BH, BK, BV), device=dev, dtype=torch.float32)

            KB = min(64, BK)
            _bscan_kernel[(BH,)](
                a_, p_, qt_, h0_, outop_, endop_, tot_,
                u_, gamma_, go_, ckpt,
                da_, dqt_, dp_, dh0_, du_s_, dgam_s_,
                doutop_, dendop_, dtot_, gstate, tmp,
                T, NC, ctx.scale,
                BT=BT, BK=BK, KB=KB, BV=BV, NUM_KB=BK // KB,
                num_warps=8, num_stages=1,
            )
            _bpre_kernel[(BH * NC,)](
                q_, k_, v_, u_, g_, beta_, gamma_,
                da_, dqt_, dp_, dh0_, du_s_, dgam_s_,
                doutop_, dendop_, dtot_,
                dq_, dk_, dv_, du_, dg_, dbeta_, dgamma_,
                T, NC,
                BT=BT, BK=BK, BV=BV, STEPS=steps,
                num_warps=8, num_stages=1,
            )
        else:
            q_, k_, v_, u_, g_, beta_, gamma_, ckpt = ctx.saved_tensors
            scratch = torch.empty((BH, _BC, BK, BV), device=dev, dtype=torch.float32)
            _bwd_kernel[(BH,)](
                q_, k_, v_, u_, g_, beta_, gamma_,
                go_, ckpt, scratch,
                dq_, dk_, dv_, du_, dg_, dbeta_, dgamma_,
                T, ctx.scale, ctx.NC,
                BK=BK, BV=BV, BC=_BC,
                num_warps=_num_warps(BK * BV), num_stages=1,
            )

        dq_d, dk_d, dv_d, du_d, dg_d, dbeta_d, dgamma_d = ctx.dtypes
        dq = dq_.reshape(B, H, T, BK)[:, :, :, :K].to(dq_d)
        dk = dk_.reshape(B, H, T, BK)[:, :, :, :K].to(dk_d)
        dg = dg_.reshape(B, H, T, BK)[:, :, :, :K].to(dg_d)
        dv = dv_.reshape(B, H, T, BV)[:, :, :, :V].to(dv_d)
        du = du_.reshape(B, H, T, BV)[:, :, :, :V].to(du_d)
        dbeta = dbeta_.reshape(B, H, T).to(dbeta_d)
        dgamma = dgamma_.reshape(B, H, T).to(dgamma_d)
        return dq, dk, dv, du, dg, dbeta, dgamma, None


def kernel_function(q, k, v, u, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    if B * H == 0 or T == 0 or K == 0 or V == 0:
        return torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)
    return _LayerFn.apply(q, k, v, u, g, beta, gamma, float(scale))
```

The slow baseline was the step-by-step recurrent backward (about 104 ms of the 124 ms total), so both directions now use the chunkwise transform: cumulative decay is folded into rescaled keys/queries and the right-hand erase is inverted away, turning each 32-step chunk into a standard delta-rule system whose couplings are small triangular Gram systems solved exactly by a nilpotent Neumann series. The forward splits into a fully parallel per-chunk precomputation plus a K-blocked sequential scan that checkpoints the state at chunk boundaries (never staging a 128×128 dot operand, which is what previously exceeded shared memory), and the backward mirrors this with a reverse state-adjoint scan followed by a parallel per-chunk differentiation of the precomputation, including the exact adjoint `Yᵀ(dL/dY)Yᵀ` of each triangular series. Correctness-critical state products use tf32 tensor cores with fp32 accumulation, and out-of-regime inputs (`gamma` near 1, strong or positive decay, out-of-range gates, oversized dims) fall back to the previously passing checkpointed recurrent forward/backward pair. My main uncertainty is register pressure in the per-chunk backward precomputation kernel, which carries many 32×128 intermediates, and whether the model's actual `gamma`/decay distribution lets the timed shapes take the 32-step chunk path rather than the 16-step one.