import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# (I - N)^-1 = I + N + N^2 + ... for strictly triangular (nilpotent) N,
# evaluated by repeated squaring (exact in exact arithmetic).  The
# 32x32 / 16x16 systems are small, so they are evaluated in full fp32
# ("ieee") rather than tf32: these triangular inverses can have large,
# cancelling entries, and tf32 rounding inside them was one source of
# the blow-ups seen previously.
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


# ---------------------------------------------------------------------------
# Kernel 1 (parallel over batch*head*chunk): everything inside one chunk
# that does not depend on the incoming state.
#
# With F_i = diag(exp(cumsum g)) and P_i = R_0 ... R_i,
# R_j = I - gamma_j u_j u_j^T, the transformed state
# Shat_i = F_i^-1 S_i P_i^-1 obeys an ordinary delta-rule recurrence.
# This kernel computes, per chunk:
#   A  = k_i / F_i,  Q = q_i * F_i           (decay rescaled, exp(shift))
#   Pm = M @ (k_i * F_i),  H0 = M @ vhat,    where M inverts the in-chunk
#        delta-rule coupling and vhat = P_i^-T v_i,
#   OutOp / EndOp: triangular Gram operators needed to undo P_i on the
#        outputs and on the outgoing state,
#   total_g: the chunk's total per-channel log decay.
#
# FIX vs. the rejected version: the in-chunk solve is
#     h = (I + diag(beta) K)^-1 diag(beta) (vhat - C S_in),
# i.e. diag(beta) multiplies on the RIGHT of the inverse (it scales the
# columns of the operator).  The rejected kernel scaled the rows
# instead.  For near-orthogonal random keys the coupling K is weak and
# the error was below tolerance, but for real, correlated keys the
# wrongly-scaled solve no longer cancels previous writes and the state
# grows without bound (the observed non-finite mismatches).
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

    # In-chunk delta-rule coupling inverse, folded with beta.
    # h = left_op @ (beta * rhs), so the folded operator is
    # left_op @ diag(beta): beta scales the COLUMNS of left_op.
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
# Kernel 2 (one program per batch*head): sequential scan over chunks.
# The [K, V] state lives in a scratch buffer and is only ever touched in
# KB x BV blocks, so no [128, 128] dot operand is ever staged and the
# shared-memory footprint stays far below the L4 limit.
# ---------------------------------------------------------------------------
@triton.jit
def _scan_kernel(
    a_ptr, p_ptr, qt_ptr, h0_ptr, outop_ptr, endop_ptr, tot_ptr,
    u_ptr, gamma_ptr, o_ptr, state_ptr,
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

        # Accumulate state contributions over K blocks.
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

        # Transformed outputs, then undo P_i on each row (fp32, not bf16:
        # these products carry the large cancelling erase terms).
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

        # Outgoing state: Shat_last @ P_last, then undo the decay scaling.
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
# Direct recurrent kernel: fallback for shapes / regimes where the chunk
# transform is not applicable (K or V > 128, gamma near 1, extreme decay,
# out-of-range gates).  One program per (batch, head), state in registers.
# ---------------------------------------------------------------------------
@triton.jit
def _recurrent_kernel(
    q_ptr, k_ptr, v_ptr, u_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr,
    T, scale,
    BK: tl.constexpr, BV: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    off_k = tl.arange(0, BK)
    off_v = tl.arange(0, BV)
    S = tl.zeros([BK, BV], dtype=tl.float32)

    for t in range(T):
        r = bh * T + t
        g = tl.load(g_ptr + r * BK + off_k).to(tl.float32)
        u = tl.load(u_ptr + r * BV + off_v).to(tl.float32)
        kk = tl.load(k_ptr + r * BK + off_k).to(tl.float32)
        vv = tl.load(v_ptr + r * BV + off_v).to(tl.float32)
        qq = tl.load(q_ptr + r * BK + off_k).to(tl.float32)
        beta = tl.load(beta_ptr + r).to(tl.float32)
        gamma = tl.load(gamma_ptr + r).to(tl.float32)

        S = S * tl.exp(g)[:, None]
        su = tl.sum(S * u[None, :], axis=1)
        S = S - gamma * su[:, None] * u[None, :]
        pred = tl.sum(S * kk[:, None], axis=0)
        S = S + kk[:, None] * (beta * (vv - pred))[None, :]
        out = scale * tl.sum(S * qq[:, None], axis=0)
        tl.store(o_ptr + r * BV + off_v, out)


def _pad_last_dim(x, padded_size):
    if x.shape[-1] == padded_size:
        return x.contiguous()
    result = torch.zeros((*x.shape[:-1], padded_size), device=x.device, dtype=x.dtype)
    result[..., : x.shape[-1]] = x
    return result


def kernel_function(q, k, v, u, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    if B * H == 0 or T == 0 or K == 0 or V == 0:
        return torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)

    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))

    q_ = _pad_last_dim(q, BK).reshape(B * H, T, BK)
    k_ = _pad_last_dim(k, BK).reshape(B * H, T, BK)
    g_ = _pad_last_dim(g, BK).reshape(B * H, T, BK)
    v_ = _pad_last_dim(v, BV).reshape(B * H, T, BV)
    u_ = _pad_last_dim(u, BV).reshape(B * H, T, BV)
    beta_ = beta.contiguous().reshape(B * H, T)
    gamma_ = gamma.contiguous().reshape(B * H, T)
    out_ = torch.empty((B * H, T, BV), device=q.device, dtype=torch.float32)

    # The chunk transform divides by (1 - gamma) and exponentiates the
    # cumulative decay, so use it only where it is well conditioned.
    # The bounds are tighter than in the rejected version (which allowed
    # gamma up to 0.9999 at chunk 32 and a decay range of 80, letting the
    # transformed values overflow on the real model inputs); anything
    # outside the safe region goes to the direct recurrent kernel.
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

    if use_chunk:
        BH = B * H
        NC = (T + BT - 1) // BT
        steps = (BT - 1).bit_length()
        a_ = torch.empty((BH, T, BK), device=q.device, dtype=torch.float32)
        p_ = torch.empty((BH, T, BK), device=q.device, dtype=torch.float32)
        qt_ = torch.empty((BH, T, BK), device=q.device, dtype=torch.float32)
        h0_ = torch.empty((BH, T, BV), device=q.device, dtype=torch.float32)
        outop_ = torch.empty((BH, NC, BT, BT), device=q.device, dtype=torch.float32)
        endop_ = torch.empty((BH, NC, BT, BT), device=q.device, dtype=torch.float32)
        tot_ = torch.empty((BH, NC, BK), device=q.device, dtype=torch.float32)
        state = torch.zeros((BH, BK, BV), device=q.device, dtype=torch.float32)

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
            u_, gamma_, out_, state,
            T, NC, float(scale),
            BT=BT, BK=BK, KB=KB, BV=BV, NUM_KB=BK // KB,
            num_warps=8, num_stages=1,
        )
    else:
        state_elements = BK * BV
        if state_elements <= 4096:
            num_warps = 4
        elif state_elements <= 16384:
            num_warps = 16
        else:
            num_warps = 32
        _recurrent_kernel[(B * H,)](
            q_, k_, v_, u_, g_, beta_, gamma_, out_,
            T, float(scale),
            BK=BK, BV=BV,
            num_warps=num_warps, num_stages=1,
        )

    return out_[:, :, :V].reshape(B, H, T, V)
