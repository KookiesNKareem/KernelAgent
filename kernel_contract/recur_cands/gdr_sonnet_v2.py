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
        kk += tl.dot(kt, tl.trans(kt), input_precision="ieee")
    row = rows[:, None]
    col = rows[None, :]
    lower = row > col
    diff = tl.where(lower, bc[:, None] - bc[None, :], 0.0)
    A = tl.where(lower, beta[:, None] * tl.exp(diff) * kk, 0.0)

    X = (row == col).to(tl.float32)
    P = -A
    for _ in tl.static_range(LOGC):
        X = X + tl.dot(X, P, input_precision="ieee")
        P = tl.dot(P, P, input_precision="ieee")

    base = (i_bh * NC * C + i_c * C + rows)
    bg = beta * tl.exp(bc)
    for kb in tl.static_range(K // BB):
        okb = kb * BB + tl.arange(0, BB)
        kt = tl.load(k_ptr + b * ksb + h * ksh + offs_t[:, None] * kst + okb[None, :] * ksd,
                     mask=mt[:, None], other=0.0).to(tl.float32)
        Wb = tl.dot(X, kt * bg[:, None], input_precision="ieee")
        tl.store(w_ptr + base[:, None] * K + okb[None, :], Wb)
    for vb in tl.static_range(V // BB):
        ovb = vb * BB + tl.arange(0, BB)
        vt = tl.load(v_ptr + b * vsb + h * vsh + offs_t[:, None] * vst + ovb[None, :] * vsd,
                     mask=mt[:, None], other=0.0).to(tl.float32)
        Ub = tl.dot(X, vt * beta[:, None], input_precision="ieee")
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
        U = tl.load(u_ptr + u_off) - tl.dot(W, S, input_precision="ieee")
        tl.store(u_ptr + u_off, U)

        offs_t = c * C + rows
        mt = offs_t < T
        g = tl.load(g_ptr + b * gsb + h * gsh + offs_t * gst, mask=mt, other=0.0).to(tl.float32)
        bc = tl.cumsum(g, axis=0)
        total = tl.sum(g, axis=0)
        kT = tl.load(k_ptr + b * ksb + h * ksh + offs_t[None, :] * kst + ok[:, None] * ksd,
                     mask=mt[None, :], other=0.0).to(tl.float32)
        kd = kT * tl.exp(total - bc)[None, :]
        S = S * tl.exp(total) + tl.dot(kd, U, input_precision="ieee")


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
        qk += tl.dot(q, tl.trans(k), input_precision="ieee")
        S0 = tl.load(s_ptr + ((i_bh * NC + i_c) * K + okb[:, None]) * V + ov[None, :])
        acc += tl.dot(q * eb[:, None], S0, input_precision="ieee")

    base = i_bh * NC * C + i_c * C + rows
    U = tl.load(u_ptr + base[:, None] * V + ov[None, :])
    le = rows[:, None] >= rows[None, :]
    diff = tl.where(le, bc[:, None] - bc[None, :], 0.0)
    D = tl.where(le, qk * tl.exp(diff), 0.0)
    O = (acc + tl.dot(D, U, input_precision="ieee")) * scale
    tl.store(o_ptr + b * osb + h * osh + offs_t[:, None] * ost + ov[None, :] * osd,
             O, mask=mt[:, None])


def kernel_function(q, k, v, g, beta, scale):
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
        K=K, V=V, C=C, LOGC=LOGC, BB=32, num_warps=4, num_stages=1)

    BV2 = 16
    _state_kernel[(V // BV2, B * H)](
        k, g, W, U, Sw, H, T, NC,
        *k.stride(), *g.stride(),
        K=K, V=V, BV=BV2, C=C, num_warps=4, num_stages=1)

    BV3 = 32
    _out_kernel[(NC, V // BV3, B * H)](
        q, k, g, U, Sw, out, float(scale), H, T, NC,
        *q.stride(), *k.stride(), *g.stride(), *out.stride(),
        K=K, V=V, BV=BV3, C=C, BB=32, num_warps=4, num_stages=1)
    return out
