import torch
import triton
import triton.language as tl

CHUNK = 64


@triton.jit
def _prep_kernel(k_ptr, v_ptr, g_ptr, beta_ptr, w_ptr, u_ptr,
                 H, T, NC,
                 ksb, ksh, kst, ksd,
                 vsb, vsh, vst, vsd,
                 gsb, gsh, gst,
                 bsb, bsh, bst,
                 K: tl.constexpr, V: tl.constexpr, C: tl.constexpr, LOGC: tl.constexpr, BB: tl.constexpr):
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    b = i_bh // H
    h = i_bh % H
    rows = tl.arange(0, C)
    offs_t = i_c * C + rows
    mt = offs_t < T
    ok = tl.arange(0, K)
    ov = tl.arange(0, V)

    g = tl.load(g_ptr + b * gsb + h * gsh + offs_t * gst, mask=mt, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + b * bsb + h * bsh + offs_t * bst, mask=mt, other=0.0).to(tl.float32)
    bc = tl.cumsum(g, axis=0)

    kk = tl.zeros([C, C], dtype=tl.float32)
    for kb in tl.static_range(K // BB):
        okb = kb * BB + tl.arange(0, BB)
        kt = tl.load(k_ptr + b * ksb + h * ksh + offs_t[:, None] * kst + okb[None, :] * ksd,
                     mask=mt[:, None], other=0.0).to(tl.float32)
        kk += tl.dot(kt, tl.trans(kt), input_precision="tf32")
    row = rows[:, None]
    col = rows[None, :]
    lower = row > col
    diff = tl.where(lower, bc[:, None] - bc[None, :], 0.0)
    A = tl.where(lower, beta[:, None] * tl.exp(diff) * kk, 0.0)

    X = (row == col).to(tl.float32)
    P = -A
    for _ in tl.static_range(LOGC):
        X = X + tl.dot(X, P, input_precision="tf32x3")
        P = tl.dot(P, P, input_precision="tf32x3")

    base = (i_bh * NC * C + i_c * C + rows)
    bg = beta * tl.exp(bc)
    for kb in tl.static_range(K // BB):
        okb = kb * BB + tl.arange(0, BB)
        kt = tl.load(k_ptr + b * ksb + h * ksh + offs_t[:, None] * kst + okb[None, :] * ksd,
                     mask=mt[:, None], other=0.0).to(tl.float32)
        Wb = tl.dot(X, kt * bg[:, None], input_precision="tf32")
        tl.store(w_ptr + base[:, None] * K + okb[None, :], Wb)
    for vb in tl.static_range(V // BB):
        ovb = vb * BB + tl.arange(0, BB)
        vt = tl.load(v_ptr + b * vsb + h * vsh + offs_t[:, None] * vst + ovb[None, :] * vsd,
                     mask=mt[:, None], other=0.0).to(tl.float32)
        Ub = tl.dot(X, vt * beta[:, None], input_precision="tf32")
        tl.store(u_ptr + base[:, None] * V + ovb[None, :], Ub)


@triton.jit
def _state_kernel(k_ptr, g_ptr, w_ptr, u_ptr, s_ptr,
                  H, T, NC,
                  ksb, ksh, kst, ksd,
                  gsb, gsh, gst,
                  K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr, C: tl.constexpr):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    b = i_bh // H
    h = i_bh % H
    rows = tl.arange(0, C)
    ok = tl.arange(0, K)
    ov = i_v * BV + tl.arange(0, BV)

    S = tl.zeros([K, BV], dtype=tl.float32)
    for c in range(NC):
        # S at chunk start (state entering chunk c)
        s_off = ((i_bh * NC + c) * K + ok[:, None]) * V + ov[None, :]
        tl.store(s_ptr + s_off, S)

        base = i_bh * NC * C + c * C + rows
        W = tl.load(w_ptr + base[:, None] * K + ok[None, :])
        u_off = base[:, None] * V + ov[None, :]
        U = tl.load(u_ptr + u_off) - tl.dot(W, S, input_precision="tf32")
        tl.store(u_ptr + u_off, U)

        offs_t = c * C + rows
        mt = offs_t < T
        g = tl.load(g_ptr + b * gsb + h * gsh + offs_t * gst, mask=mt, other=0.0).to(tl.float32)
        bc = tl.cumsum(g, axis=0)
        total = tl.sum(g, axis=0)
        kT = tl.load(k_ptr + b * ksb + h * ksh + offs_t[None, :] * kst + ok[:, None] * ksd,
                     mask=mt[None, :], other=0.0).to(tl.float32)
        kd = kT * tl.exp(total - bc)[None, :]
        S = S * tl.exp(total) + tl.dot(kd, U, input_precision="tf32")


@triton.jit
def _out_kernel(q_ptr, k_ptr, g_ptr, u_ptr, s_ptr, o_ptr, scale,
                H, T, NC,
                qsb, qsh, qst, qsd,
                ksb, ksh, kst, ksd,
                gsb, gsh, gst,
                osb, osh, ost, osd,
                K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr, C: tl.constexpr, BB: tl.constexpr):
    i_c = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    b = i_bh // H
    h = i_bh % H
    rows = tl.arange(0, C)
    offs_t = i_c * C + rows
    mt = offs_t < T
    ok = tl.arange(0, K)
    ov = i_v * BV + tl.arange(0, BV)

    g = tl.load(g_ptr + b * gsb + h * gsh + offs_t * gst, mask=mt, other=0.0).to(tl.float32)
    bc = tl.cumsum(g, axis=0)
    eb = tl.exp(bc)

    qk = tl.zeros([C, C], dtype=tl.float32)
    acc = tl.zeros([C, BV], dtype=tl.float32)
    for kb in tl.static_range(K // BB):
        okb = kb * BB + tl.arange(0, BB)
        q = tl.load(q_ptr + b * qsb + h * qsh + offs_t[:, None] * qst + okb[None, :] * qsd,
                    mask=mt[:, None], other=0.0).to(tl.float32)
        k = tl.load(k_ptr + b * ksb + h * ksh + offs_t[:, None] * kst + okb[None, :] * ksd,
                    mask=mt[:, None], other=0.0).to(tl.float32)
        qk += tl.dot(q, tl.trans(k), input_precision="tf32")
        S0 = tl.load(s_ptr + ((i_bh * NC + i_c) * K + okb[:, None]) * V + ov[None, :])
        acc += tl.dot(q * eb[:, None], S0, input_precision="tf32")

    base = i_bh * NC * C + i_c * C + rows
    U = tl.load(u_ptr + base[:, None] * V + ov[None, :])
    le = rows[:, None] >= rows[None, :]
    diff = tl.where(le, bc[:, None] - bc[None, :], 0.0)
    D = tl.where(le, qk * tl.exp(diff), 0.0)
    O = (acc + tl.dot(D, U, input_precision="tf32")) * scale
    tl.store(o_ptr + b * osb + h * osh + offs_t[:, None] * ost + ov[None, :] * osd,
             O, mask=mt[:, None])


def _forward(q, k, v, g, beta, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    assert K in (64, 128) and V in (64, 128)
    C = CHUNK
    LOGC = 6
    NC = triton.cdiv(T, C)
    dev = q.device
    W = torch.empty(B * H, NC * C, K, device=dev, dtype=torch.float32)
    U = torch.empty(B * H, NC * C, V, device=dev, dtype=torch.float32)
    Sw = torch.empty(B * H, NC, K, V, device=dev, dtype=torch.float32)
    out = torch.empty(B, H, T, V, device=dev, dtype=torch.float32)

    _prep_kernel[(NC, B * H)](
        k, v, g, beta, W, U, H, T, NC,
        *k.stride(), *v.stride(), *g.stride(), *beta.stride(),
        K=K, V=V, C=C, LOGC=LOGC, BB=64, num_warps=4, num_stages=1)

    BV2 = 16
    _state_kernel[(V // BV2, B * H)](
        k, g, W, U, Sw, H, T, NC,
        *k.stride(), *g.stride(),
        K=K, V=V, BV=BV2, C=C, num_warps=4, num_stages=1)

    BV3 = 32
    _out_kernel[(NC, V // BV3, B * H)](
        q, k, g, U, Sw, out, float(scale), H, T, NC,
        *q.stride(), *k.stride(), *g.stride(), *out.stride(),
        K=K, V=V, BV=BV3, C=C, BB=64, num_warps=4, num_stages=1)
    return out, U, Sw

@triton.jit
def _bwd_prep_kernel(q_ptr, k_ptr, g_ptr, beta_ptr, m_ptr, d_ptr,
                     H, T, NC,
                     qsb, qsh, qst, qsd,
                     ksb, ksh, kst, ksd,
                     gsb, gsh, gst,
                     bsb, bsh, bst,
                     K: tl.constexpr, C: tl.constexpr, LOGC: tl.constexpr, BB: tl.constexpr):
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    b = i_bh // H
    h = i_bh % H
    rows = tl.arange(0, C)
    offs_t = i_c * C + rows
    mt = offs_t < T
    g = tl.load(g_ptr + b * gsb + h * gsh + offs_t * gst, mask=mt, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + b * bsb + h * bsh + offs_t * bst, mask=mt, other=0.0).to(tl.float32)
    bc = tl.cumsum(g, axis=0)
    kk = tl.zeros([C, C], dtype=tl.float32)
    qk = tl.zeros([C, C], dtype=tl.float32)
    for kb in tl.static_range(K // BB):
        okb = kb * BB + tl.arange(0, BB)
        kt = tl.load(k_ptr + b * ksb + h * ksh + offs_t[:, None] * kst + okb[None, :] * ksd,
                     mask=mt[:, None], other=0.0).to(tl.float32)
        qt = tl.load(q_ptr + b * qsb + h * qsh + offs_t[:, None] * qst + okb[None, :] * qsd,
                     mask=mt[:, None], other=0.0).to(tl.float32)
        kk += tl.dot(kt, tl.trans(kt), input_precision="tf32")
        qk += tl.dot(qt, tl.trans(kt), input_precision="tf32")
    row = rows[:, None]
    col = rows[None, :]
    lower = row > col
    le = row >= col
    diff = tl.where(le, bc[:, None] - bc[None, :], 0.0)
    E = tl.exp(diff)
    A = tl.where(lower, beta[:, None] * E * kk, 0.0)
    X = (row == col).to(tl.float32)
    P = -A
    for _ in tl.static_range(LOGC):
        X = X + tl.dot(X, P, input_precision="tf32x3")
        P = tl.dot(P, P, input_precision="tf32x3")
    D = tl.where(le, qk * E, 0.0)
    off = (i_bh * NC + i_c) * C * C + row * C + col
    tl.store(m_ptr + off, X)
    tl.store(d_ptr + off, D)


@triton.jit
def _bwd_state_kernel(q_ptr, k_ptr, g_ptr, beta_ptr, do_ptr, m_ptr, d_ptr, dr_ptr, dsw_ptr, scale,
                      H, T, NC,
                      qsb, qsh, qst, qsd,
                      ksb, ksh, kst, ksd,
                      gsb, gsh, gst,
                      bsb, bsh, bst,
                      osb, osh, ost, osd,
                      K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr, C: tl.constexpr):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    b = i_bh // H
    h = i_bh % H
    rows = tl.arange(0, C)
    ok = tl.arange(0, K)
    ov = i_v * BV + tl.arange(0, BV)
    row = rows[:, None]
    col = rows[None, :]

    dS = tl.zeros([K, BV], dtype=tl.float32)
    for ci in range(NC):
        c = NC - 1 - ci
        tl.store(dsw_ptr + ((i_bh * NC + c) * K + ok[:, None]) * V + ov[None, :], dS)

        offs_t = c * C + rows
        mt = offs_t < T
        g = tl.load(g_ptr + b * gsb + h * gsh + offs_t * gst, mask=mt, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + b * bsb + h * bsh + offs_t * bst, mask=mt, other=0.0).to(tl.float32)
        bc = tl.cumsum(g, axis=0)
        total = tl.sum(g, axis=0)
        G = tl.exp(bc)
        w = tl.exp(total - bc)

        dOs = tl.load(do_ptr + b * osb + h * osh + offs_t[:, None] * ost + ov[None, :] * osd,
                      mask=mt[:, None], other=0.0).to(tl.float32) * scale
        off = (i_bh * NC + c) * C * C + row * C + col
        D = tl.load(d_ptr + off)
        dU = tl.dot(tl.trans(D), dOs, input_precision="tf32")
        kN = tl.load(k_ptr + b * ksb + h * ksh + offs_t[:, None] * kst + ok[None, :] * ksd,
                     mask=mt[:, None], other=0.0).to(tl.float32)
        dU += w[:, None] * tl.dot(kN, dS, input_precision="tf32")
        Mm = tl.load(m_ptr + off)
        dR = tl.dot(tl.trans(Mm), dU, input_precision="tf32")
        base = i_bh * NC * C + c * C + rows
        tl.store(dr_ptr + base[:, None] * V + ov[None, :], dR)
        dRhat = beta[:, None] * dR

        qT = tl.load(q_ptr + b * qsb + h * qsh + offs_t[None, :] * qst + ok[:, None] * qsd,
                     mask=mt[None, :], other=0.0).to(tl.float32)
        t1 = tl.dot(qT * G[None, :], dOs, input_precision="tf32")
        kT = tl.load(k_ptr + b * ksb + h * ksh + offs_t[None, :] * kst + ok[:, None] * ksd,
                     mask=mt[None, :], other=0.0).to(tl.float32)
        t2 = tl.dot(kT * G[None, :], dRhat, input_precision="tf32")
        dS = dS * tl.exp(total) + t1 - t2


@triton.jit
def _bwd_grad_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, do_ptr, u_ptr, s_ptr, dr_ptr, dsw_ptr,
                     dq_ptr, dk_ptr, dv_ptr, dg_ptr, db_ptr, scale,
                     H, T, NC,
                     qsb, qsh, qst, qsd,
                     ksb, ksh, kst, ksd,
                     vsb, vsh, vst, vsd,
                     gsb, gsh, gst,
                     bsb, bsh, bst,
                     osb, osh, ost, osd,
                     dqsb, dqsh, dqst, dqsd,
                     dksb, dksh, dkst, dksd,
                     dvsb, dvsh, dvst, dvsd,
                     dgsb, dgsh, dgst,
                     dbsb, dbsh, dbst,
                     K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr, C: tl.constexpr):
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    b = i_bh // H
    h = i_bh % H
    rows = tl.arange(0, C)
    offs_t = i_c * C + rows
    mt = offs_t < T
    ok = tl.arange(0, K)
    row = rows[:, None]
    col = rows[None, :]

    g = tl.load(g_ptr + b * gsb + h * gsh + offs_t * gst, mask=mt, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + b * bsb + h * bsh + offs_t * bst, mask=mt, other=0.0).to(tl.float32)
    bc = tl.cumsum(g, axis=0)
    total = tl.sum(g, axis=0)
    G = tl.exp(bc)
    w = tl.exp(total - bc)
    etot = tl.exp(total)
    base = i_bh * NC * C + i_c * C + rows

    dQ = tl.zeros([C, K], dtype=tl.float32)
    dK = tl.zeros([C, K], dtype=tl.float32)
    dDacc = tl.zeros([C, C], dtype=tl.float32)
    dRU = tl.zeros([C, C], dtype=tl.float32)
    dbeta = tl.zeros([C], dtype=tl.float32)
    dbacc = tl.zeros([C], dtype=tl.float32)
    dgT = 0.0

    for vb in tl.static_range(V // BV):
        ov = vb * BV + tl.arange(0, BV)
        Ub = tl.load(u_ptr + base[:, None] * V + ov[None, :])
        dRb = tl.load(dr_ptr + base[:, None] * V + ov[None, :])
        sidx = ((i_bh * NC + i_c) * K + ok[:, None]) * V + ov[None, :]
        S0b = tl.load(s_ptr + sidx)
        dSb = tl.load(dsw_ptr + sidx)
        dOs = tl.load(do_ptr + b * osb + h * osh + offs_t[:, None] * ost + ov[None, :] * osd,
                      mask=mt[:, None], other=0.0).to(tl.float32) * scale
        vt = tl.load(v_ptr + b * vsb + h * vsh + offs_t[:, None] * vst + ov[None, :] * vsd,
                     mask=mt[:, None], other=0.0).to(tl.float32)
        k = tl.load(k_ptr + b * ksb + h * ksh + offs_t[:, None] * kst + ok[None, :] * ksd,
                    mask=mt[:, None], other=0.0).to(tl.float32)
        KS0 = tl.dot(k, S0b, input_precision="tf32")
        Rhat = vt - G[:, None] * KS0
        dbeta += tl.sum(dRb * Rhat, axis=1)
        dRhat = beta[:, None] * dRb
        tl.store(dv_ptr + b * dvsb + h * dvsh + offs_t[:, None] * dvst + ov[None, :] * dvsd,
                 dRhat, mask=mt[:, None])
        dbacc -= G * tl.sum(dRhat * KS0, axis=1)
        dK -= G[:, None] * tl.dot(dRhat, tl.trans(S0b), input_precision="tf32")
        dK += w[:, None] * tl.dot(Ub, tl.trans(dSb), input_precision="tf32")
        Y2 = tl.dot(k, dSb, input_precision="tf32")
        sj = w * tl.sum(Ub * Y2, axis=1)
        dbacc -= sj
        dgT += tl.sum(sj, axis=0) + etot * tl.sum(tl.sum(S0b * dSb, axis=1), axis=0)
        q = tl.load(q_ptr + b * qsb + h * qsh + offs_t[:, None] * qst + ok[None, :] * qsd,
                    mask=mt[:, None], other=0.0).to(tl.float32)
        QS0 = tl.dot(q, S0b, input_precision="tf32")
        dbacc += G * tl.sum(QS0 * dOs, axis=1)
        dQ += G[:, None] * tl.dot(dOs, tl.trans(S0b), input_precision="tf32")
        dDacc += tl.dot(dOs, tl.trans(Ub), input_precision="tf32")
        dRU += tl.dot(dRb, tl.trans(Ub), input_precision="tf32")

    q = tl.load(q_ptr + b * qsb + h * qsh + offs_t[:, None] * qst + ok[None, :] * qsd,
                mask=mt[:, None], other=0.0).to(tl.float32)
    k = tl.load(k_ptr + b * ksb + h * ksh + offs_t[:, None] * kst + ok[None, :] * ksd,
                mask=mt[:, None], other=0.0).to(tl.float32)
    qk = tl.dot(q, tl.trans(k), input_precision="tf32")
    kk = tl.dot(k, tl.trans(k), input_precision="tf32")
    lower = row > col
    le = row >= col
    diff = tl.where(le, bc[:, None] - bc[None, :], 0.0)
    E = tl.exp(diff)
    DE = tl.where(le, dDacc, 0.0) * E
    P1 = DE * qk
    dA = tl.where(lower, -dRU, 0.0)
    Eh = tl.where(lower, E * kk, 0.0)
    PA = dA * (beta[:, None] * Eh)
    dbeta += tl.sum(dA * Eh, axis=1)
    dbacc += tl.sum(P1, axis=1) - tl.sum(P1, axis=0) + tl.sum(PA, axis=1) - tl.sum(PA, axis=0)
    dKK = dA * (beta[:, None] * E)
    dQ += tl.dot(DE, k, input_precision="tf32")
    dK += tl.dot(tl.trans(DE), q, input_precision="tf32")
    dK += tl.dot(dKK + tl.trans(dKK), k, input_precision="tf32")
    dg = tl.cumsum(dbacc, axis=0, reverse=True) + dgT

    tl.store(dq_ptr + b * dqsb + h * dqsh + offs_t[:, None] * dqst + ok[None, :] * dqsd,
             dQ, mask=mt[:, None])
    tl.store(dk_ptr + b * dksb + h * dksh + offs_t[:, None] * dkst + ok[None, :] * dksd,
             dK, mask=mt[:, None])
    tl.store(dg_ptr + b * dgsb + h * dgsh + offs_t * dgst, dg, mask=mt)
    tl.store(db_ptr + b * dbsb + h * dbsh + offs_t * dbst, dbeta, mask=mt)


def _backward(q, k, v, g, beta, scale, U, Sw, dout):
    B, H, T, K = q.shape
    V = v.shape[-1]
    C = CHUNK
    LOGC = 6
    NC = triton.cdiv(T, C)
    dev = q.device
    dout = dout.contiguous().float()
    Mw = torch.empty(B * H, NC, C, C, device=dev, dtype=torch.float32)
    Dw = torch.empty(B * H, NC, C, C, device=dev, dtype=torch.float32)
    dR = torch.empty(B * H, NC * C, V, device=dev, dtype=torch.float32)
    dSw = torch.empty(B * H, NC, K, V, device=dev, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    dbeta = torch.empty_like(beta)

    _bwd_prep_kernel[(NC, B * H)](
        q, k, g, beta, Mw, Dw, H, T, NC,
        *q.stride(), *k.stride(), *g.stride(), *beta.stride(),
        K=K, C=C, LOGC=LOGC, BB=64, num_warps=4, num_stages=1)

    BV2 = 16
    _bwd_state_kernel[(V // BV2, B * H)](
        q, k, g, beta, dout, Mw, Dw, dR, dSw, float(scale), H, T, NC,
        *q.stride(), *k.stride(), *g.stride(), *beta.stride(), *dout.stride(),
        K=K, V=V, BV=BV2, C=C, num_warps=4, num_stages=1)

    BV3 = 32
    _bwd_grad_kernel[(NC, B * H)](
        q, k, v, g, beta, dout, U, Sw, dR, dSw,
        dq, dk, dv, dg, dbeta, float(scale), H, T, NC,
        *q.stride(), *k.stride(), *v.stride(), *g.stride(), *beta.stride(), *dout.stride(),
        *dq.stride(), *dk.stride(), *dv.stride(), *dg.stride(), *dbeta.stride(),
        K=K, V=V, BV=BV3, C=C, num_warps=8, num_stages=1)
    return dq, dk, dv, dg, dbeta


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale):
        out, U, Sw = _forward(q, k, v, g, beta, scale)
        ctx.save_for_backward(q, k, v, g, beta, U, Sw)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, U, Sw = ctx.saved_tensors
        dq, dk, dv, dg, db = _backward(q, k, v, g, beta, ctx.scale, U, Sw, dout)
        return dq, dk, dv, dg, db, None


def kernel_function(q, k, v, g, beta, scale):
    return _GatedDeltaRule.apply(q, k, v, g, beta, float(scale))
