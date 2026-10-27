```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward chunk preparation: one program per (batch, head, chunk).
# Builds Kg, Qeff, U and the intra-chunk output, independently of state.
# ---------------------------------------------------------------------------
@triton.jit
def _prep_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
    kg_ptr, qeff_ptr, u_ptr, ointra_ptr, cum_ptr,
    H, T, K_DIM, V_DIM, NUM_CHUNKS,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    sgb, sgh, sgt,
    sbb, sbh, sbt,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BVFULL: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // NUM_CHUNKS
    chunk_idx = pid % NUM_CHUNKS
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    q_base = q_ptr + b_idx * sqb + h_idx * sqh
    k_base = k_ptr + b_idx * skb + h_idx * skh
    v_base = v_ptr + b_idx * svb + h_idx * svh
    g_base = g_ptr + b_idx * sgb + h_idx * sgh
    beta_base = beta_ptr + b_idx * sbb + h_idx * sbh

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BVFULL)
    offs_t = chunk_idx * CHUNK + offs_c
    mask_t = offs_t < T
    mask_k = offs_k < K_DIM
    mask_v = offs_v < V_DIM

    g_vec = tl.load(g_base + offs_t * sgt, mask=mask_t, other=0.0).to(tl.float32)
    beta_vec = tl.load(beta_base + offs_t * sbt, mask=mask_t, other=0.0).to(tl.float32)
    cum = tl.cumsum(g_vec, axis=0)
    a_vec = tl.exp(cum)

    slice_idx = pid_bh * NUM_CHUNKS + chunk_idx
    tl.store(cum_ptr + slice_idx * CHUNK + offs_c, cum)

    k_tile = tl.load(
        k_base + offs_t[:, None] * skt + offs_k[None, :] * skk,
        mask=mask_t[:, None] & mask_k[None, :], other=0.0)
    q_tile = tl.load(
        q_base + offs_t[:, None] * sqt + offs_k[None, :] * sqk,
        mask=mask_t[:, None] & mask_k[None, :], other=0.0)

    row = offs_c[:, None]
    col = offs_c[None, :]
    eye = (row == col).to(tl.float32)
    lt_mask = col < row
    le_mask = col <= row
    cum_diff = cum[:, None] - cum[None, :]

    kk = tl.dot(k_tile, tl.trans(k_tile))
    qk = tl.dot(q_tile, tl.trans(k_tile))
    decay_lt = tl.exp(tl.where(lt_mask, cum_diff, 0.0))
    l_mat = tl.where(lt_mask, kk * beta_vec[:, None] * decay_lt, 0.0)
    decay_le = tl.exp(tl.where(le_mask, cum_diff, 0.0))
    m_mat = tl.where(le_mask, qk * decay_le, 0.0)

    # (I+L)^-1 = (I-L)(I+L^2)(I+L^4)... for nilpotent strictly-lower L.
    inv = eye - l_mat
    power = l_mat
    if CHUNK > 2:
        power = tl.dot(power, power, input_precision="tf32")
        inv = tl.dot(inv, eye + power, input_precision="tf32")
    if CHUNK > 4:
        power = tl.dot(power, power, input_precision="tf32")
        inv = tl.dot(inv, eye + power, input_precision="tf32")
    if CHUNK > 8:
        power = tl.dot(power, power, input_precision="tf32")
        inv = tl.dot(inv, eye + power, input_precision="tf32")
    if CHUNK > 16:
        power = tl.dot(power, power, input_precision="tf32")
        inv = tl.dot(inv, eye + power, input_precision="tf32")
    if CHUNK > 32:
        power = tl.dot(power, power, input_precision="tf32")
        inv = tl.dot(inv, eye + power, input_precision="tf32")

    k_f32 = k_tile.to(tl.float32)
    q_f32 = q_tile.to(tl.float32)

    kg = tl.dot(inv, k_f32 * (beta_vec * a_vec)[:, None], input_precision="tf32")
    tl.store(kg_ptr + slice_idx * CHUNK * K_DIM + offs_c[:, None] * K_DIM + offs_k[None, :],
             kg, mask=mask_k[None, :])

    qeff = q_f32 * a_vec[:, None] - tl.dot(m_mat, kg, input_precision="tf32")
    tl.store(qeff_ptr + slice_idx * CHUNK * K_DIM + offs_c[:, None] * K_DIM + offs_k[None, :],
             qeff, mask=mask_k[None, :])

    v_tile = tl.load(
        v_base + offs_t[:, None] * svt + offs_v[None, :] * svv,
        mask=mask_t[:, None] & mask_v[None, :], other=0.0)
    u_mat = tl.dot(inv, v_tile.to(tl.float32) * beta_vec[:, None], input_precision="tf32")
    tl.store(u_ptr + slice_idx * CHUNK * V_DIM + offs_c[:, None] * V_DIM + offs_v[None, :],
             u_mat, mask=mask_v[None, :])

    ointra = tl.dot(m_mat, u_mat, input_precision="tf32")
    tl.store(ointra_ptr + slice_idx * CHUNK * V_DIM + offs_c[:, None] * V_DIM + offs_v[None, :],
             ointra, mask=mask_v[None, :])


# ---------------------------------------------------------------------------
# Forward scan: one program per (batch, head, V-block), state in registers.
# Optionally stores per-chunk W and boundary states S0 for the backward.
# ---------------------------------------------------------------------------
@triton.jit
def _scan_kernel(
    k_ptr, kg_ptr, qeff_ptr, u_ptr, ointra_ptr, cum_ptr, o_ptr,
    w_ptr, s_ptr, scale,
    H, T, K_DIM, V_DIM, NUM_CHUNKS,
    skb, skh, skt, skk,
    sob, soh, sot, sov,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_BWD: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    k_base = k_ptr + b_idx * skb + h_idx * skh
    o_base = o_ptr + b_idx * sob + h_idx * soh

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    offs_v_global = pid_v * BV + offs_v
    mask_k = offs_k < K_DIM
    mask_v = offs_v_global < V_DIM

    state = tl.zeros([BK, BV], dtype=tl.float32)

    for chunk_idx in range(NUM_CHUNKS):
        offs_t = chunk_idx * CHUNK + offs_c
        mask_t = offs_t < T
        slice_idx = pid_bh * NUM_CHUNKS + chunk_idx

        if STORE_BWD:
            tl.store(s_ptr + slice_idx * K_DIM * V_DIM + offs_k[:, None] * V_DIM + offs_v_global[None, :],
                     state, mask=mask_k[:, None] & mask_v[None, :])

        kg = tl.load(kg_ptr + slice_idx * CHUNK * K_DIM + offs_c[:, None] * K_DIM + offs_k[None, :],
                     mask=mask_k[None, :], other=0.0)
        u_mat = tl.load(u_ptr + slice_idx * CHUNK * V_DIM + offs_c[:, None] * V_DIM + offs_v_global[None, :],
                        mask=mask_v[None, :], other=0.0)
        w_tile = u_mat - tl.dot(kg, state, input_precision="tf32")
        if STORE_BWD:
            tl.store(w_ptr + slice_idx * CHUNK * V_DIM + offs_c[:, None] * V_DIM + offs_v_global[None, :],
                     w_tile, mask=mask_t[:, None] & mask_v[None, :])

        qeff = tl.load(qeff_ptr + slice_idx * CHUNK * K_DIM + offs_c[:, None] * K_DIM + offs_k[None, :],
                       mask=mask_k[None, :], other=0.0)
        ointra = tl.load(ointra_ptr + slice_idx * CHUNK * V_DIM + offs_c[:, None] * V_DIM + offs_v_global[None, :],
                         mask=mask_v[None, :], other=0.0)
        o_tile = scale * (tl.dot(qeff, state, input_precision="tf32") + ointra)
        tl.store(o_base + offs_t[:, None] * sot + offs_v_global[None, :] * sov,
                 o_tile, mask=mask_t[:, None] & mask_v[None, :])

        cum = tl.load(cum_ptr + slice_idx * CHUNK + offs_c)
        total_cum = tl.sum(tl.where(offs_c == CHUNK - 1, cum, 0.0), axis=0)
        k_tile = tl.load(k_base + offs_t[:, None] * skt + offs_k[None, :] * skk,
                         mask=mask_t[:, None] & mask_k[None, :], other=0.0)
        k_suffix = k_tile.to(tl.float32) * tl.exp(total_cum - cum)[:, None]
        state = tl.exp(total_cum) * state + tl.dot(tl.trans(k_suffix), w_tile, input_precision="tf32")


# ---------------------------------------------------------------------------
# Fallback forward (odd shapes), optionally storing W / boundary states.
# ---------------------------------------------------------------------------
@triton.jit
def _fallback_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr, w_ptr, s_ptr,
    scale, H, T, K_DIM, V_DIM, NUM_CHUNKS,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    sgb, sgh, sgt,
    sbb, sbh, sbt,
    sob, soh, sot, sov,
    CHUNK: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    STORE_BWD: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    q_base = q_ptr + b_idx * sqb + h_idx * sqh
    k_base = k_ptr + b_idx * skb + h_idx * skh
    v_base = v_ptr + b_idx * svb + h_idx * svh
    g_base = g_ptr + b_idx * sgb + h_idx * sgh
    beta_base = beta_ptr + b_idx * sbb + h_idx * sbh
    o_base = o_ptr + b_idx * sob + h_idx * soh

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    offs_v_global = pid_v * BV + offs_v
    mask_k = offs_k < K_DIM
    mask_v = offs_v_global < V_DIM
    row = offs_c[:, None]
    col = offs_c[None, :]
    eye = (row == col).to(tl.float32)
    lt_mask = col < row
    le_mask = col <= row
    state = tl.zeros([BK, BV], dtype=tl.float32)

    for chunk_idx in range(NUM_CHUNKS):
        offs_t = chunk_idx * CHUNK + offs_c
        mask_t = offs_t < T
        slice_idx = pid_bh * NUM_CHUNKS + chunk_idx
        if STORE_BWD:
            tl.store(s_ptr + slice_idx * K_DIM * V_DIM + offs_k[:, None] * V_DIM + offs_v_global[None, :],
                     state, mask=mask_k[:, None] & mask_v[None, :])

        k_tile = tl.load(k_base + offs_t[:, None] * skt + offs_k[None, :] * skk,
                         mask=mask_t[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        q_tile = tl.load(q_base + offs_t[:, None] * sqt + offs_k[None, :] * sqk,
                         mask=mask_t[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        v_tile = tl.load(v_base + offs_t[:, None] * svt + offs_v_global[None, :] * svv,
                         mask=mask_t[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
        g_vec = tl.load(g_base + offs_t * sgt, mask=mask_t, other=0.0).to(tl.float32)
        beta_vec = tl.load(beta_base + offs_t * sbt, mask=mask_t, other=0.0).to(tl.float32)

        cum = tl.cumsum(g_vec, axis=0)
        a_vec = tl.exp(cum)
        total_log_decay = tl.sum(g_vec, axis=0)
        a_last = tl.exp(total_log_decay)
        cum_diff = cum[:, None] - cum[None, :]

        kk = tl.dot(k_tile, tl.trans(k_tile), input_precision="ieee")
        decay_lt = tl.exp(tl.where(lt_mask, cum_diff, 0.0))
        l_mat = tl.where(lt_mask, kk * beta_vec[:, None] * decay_lt, 0.0)

        inv = eye - l_mat
        power = tl.dot(l_mat, l_mat, input_precision="ieee")
        inv = tl.dot(inv, eye + power, input_precision="ieee")
        power = tl.dot(power, power, input_precision="ieee")
        inv = tl.dot(inv, eye + power, input_precision="ieee")
        power = tl.dot(power, power, input_precision="ieee")
        inv = tl.dot(inv, eye + power, input_precision="ieee")
        power = tl.dot(power, power, input_precision="ieee")
        inv = tl.dot(inv, eye + power, input_precision="ieee")

        ks = tl.dot(k_tile, state, input_precision="ieee")
        rhs = beta_vec[:, None] * (v_tile - a_vec[:, None] * ks)
        w_tile = tl.dot(inv, rhs, input_precision="ieee")
        if STORE_BWD:
            tl.store(w_ptr + slice_idx * CHUNK * V_DIM + offs_c[:, None] * V_DIM + offs_v_global[None, :],
                     w_tile, mask=mask_t[:, None] & mask_v[None, :])

        qk = tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee")
        decay_le = tl.exp(tl.where(le_mask, cum_diff, 0.0))
        m_mat = tl.where(le_mask, qk * decay_le, 0.0)
        qs = tl.dot(q_tile, state, input_precision="ieee")
        mw = tl.dot(m_mat, w_tile, input_precision="ieee")
        o_tile = scale * (a_vec[:, None] * qs + mw)
        tl.store(o_base + offs_t[:, None] * sot + offs_v_global[None, :] * sov,
                 o_tile, mask=mask_t[:, None] & mask_v[None, :])

        k_suffix = k_tile * tl.exp(total_log_decay - cum)[:, None]
        state = a_last * state + tl.dot(tl.trans(k_suffix), w_tile, input_precision="ieee")


# ---------------------------------------------------------------------------
# Backward preparation: one program per (batch, head, chunk) recomputes
# Inv = (I+L)^-1 and M (decay-masked QK^T) for the backward scan.
# ---------------------------------------------------------------------------
@triton.jit
def _bwd_prep_kernel(
    q_ptr, k_ptr, g_ptr, beta_ptr, inv_ptr, m_ptr,
    H, T, K_DIM, NUM_CHUNKS,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    sgb, sgh, sgt,
    sbb, sbh, sbt,
    CHUNK: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // NUM_CHUNKS
    chunk_idx = pid % NUM_CHUNKS
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    q_base = q_ptr + b_idx * sqb + h_idx * sqh
    k_base = k_ptr + b_idx * skb + h_idx * skh
    g_base = g_ptr + b_idx * sgb + h_idx * sgh
    beta_base = beta_ptr + b_idx * sbb + h_idx * sbh

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_t = chunk_idx * CHUNK + offs_c
    mask_t = offs_t < T
    mask_k = offs_k < K_DIM

    g_vec = tl.load(g_base + offs_t * sgt, mask=mask_t, other=0.0).to(tl.float32)
    beta_vec = tl.load(beta_base + offs_t * sbt, mask=mask_t, other=0.0).to(tl.float32)
    cum = tl.cumsum(g_vec, axis=0)

    k_tile = tl.load(k_base + offs_t[:, None] * skt + offs_k[None, :] * skk,
                     mask=mask_t[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
    q_tile = tl.load(q_base + offs_t[:, None] * sqt + offs_k[None, :] * sqk,
                     mask=mask_t[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

    row = offs_c[:, None]
    col = offs_c[None, :]
    eye = (row == col).to(tl.float32)
    lt_mask = col < row
    le_mask = col <= row
    cum_diff = cum[:, None] - cum[None, :]

    kk = tl.dot(k_tile, tl.trans(k_tile), input_precision="tf32")
    qk = tl.dot(q_tile, tl.trans(k_tile), input_precision="tf32")
    decay_lt = tl.exp(tl.where(lt_mask, cum_diff, 0.0))
    l_mat = tl.where(lt_mask, kk * beta_vec[:, None] * decay_lt, 0.0)
    decay_le = tl.exp(tl.where(le_mask, cum_diff, 0.0))
    m_mat = tl.where(le_mask, qk * decay_le, 0.0)

    inv = eye - l_mat
    power = l_mat
    if CHUNK > 2:
        power = tl.dot(power, power, input_precision="tf32")
        inv = tl.dot(inv, eye + power, input_precision="tf32")
    if CHUNK > 4:
        power = tl.dot(power, power, input_precision="tf32")
        inv = tl.dot(inv, eye + power, input_precision="tf32")
    if CHUNK > 8:
        power = tl.dot(power, power, input_precision="tf32")
        inv = tl.dot(inv, eye + power, input_precision="tf32")
    if CHUNK > 16:
        power = tl.dot(power, power, input_precision="tf32")
        inv = tl.dot(inv, eye + power, input_precision="tf32")
    if CHUNK > 32:
        power = tl.dot(power, power, input_precision="tf32")
        inv = tl.dot(inv, eye + power, input_precision="tf32")

    slice_idx = pid_bh * NUM_CHUNKS + chunk_idx
    tl.store(inv_ptr + slice_idx * CHUNK * CHUNK + offs_c[:, None] * CHUNK + offs_c[None, :], inv)
    tl.store(m_ptr + slice_idx * CHUNK * CHUNK + offs_c[:, None] * CHUNK + offs_c[None, :], m_mat)


# ---------------------------------------------------------------------------
# Backward scan: one program per (batch, head, V-block) walks chunks in
# reverse, carrying the state adjoint dS in registers.  Per chunk, with
# DY = scale*dO, S0 the forward boundary state, dS1 the incoming adjoint,
# W the stored forward update, Ksuf_j = exp(cum_last-cum_j) k_j:
#   G  = M^T DY + Ksuf dS1 ;  Z = (I+L)^-T G
#   dS0 = A_last dS1 + (A Q)^T DY - (beta A K)^T Z
# plus closed-form chunk-local gradients for Q, K, V, g, beta (the parts
# that reduce over V are accumulated with atomics over the V-blocks).
# ---------------------------------------------------------------------------
@triton.jit
def _bwd_scan_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, do_ptr,
    w_ptr, s_ptr, inv_ptr, m_ptr,
    dq_ptr, dk_ptr, dv_ptr, dg_ptr, dbeta_ptr,
    scale, H, T, K_DIM, V_DIM, NUM_CHUNKS,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    sgb, sgh, sgt,
    sbb, sbh, sbt,
    sdb, sdh, sdt, sdv,
    CHUNK: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    q_base = q_ptr + b_idx * sqb + h_idx * sqh
    k_base = k_ptr + b_idx * skb + h_idx * skh
    v_base = v_ptr + b_idx * svb + h_idx * svh
    g_base = g_ptr + b_idx * sgb + h_idx * sgh
    beta_base = beta_ptr + b_idx * sbb + h_idx * sbh
    do_base = do_ptr + b_idx * sdb + h_idx * sdh

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    offs_v_global = pid_v * BV + offs_v
    mask_k = offs_k < K_DIM
    mask_v = offs_v_global < V_DIM
    row = offs_c[:, None]
    col = offs_c[None, :]
    lt_mask = col < row
    le_mask = col <= row

    dq_base = dq_ptr + pid_bh * T * K_DIM
    dk_base = dk_ptr + pid_bh * T * K_DIM
    dv_base = dv_ptr + pid_bh * T * V_DIM
    dg_base = dg_ptr + pid_bh * T
    dbeta_base = dbeta_ptr + pid_bh * T

    dstate = tl.zeros([BK, BV], dtype=tl.float32)

    for rev_idx in range(NUM_CHUNKS):
        chunk_idx = NUM_CHUNKS - 1 - rev_idx
        offs_t = chunk_idx * CHUNK + offs_c
        mask_t = offs_t < T
        slice_idx = pid_bh * NUM_CHUNKS + chunk_idx

        g_vec = tl.load(g_base + offs_t * sgt, mask=mask_t, other=0.0).to(tl.float32)
        beta_vec = tl.load(beta_base + offs_t * sbt, mask=mask_t, other=0.0).to(tl.float32)
        cum = tl.cumsum(g_vec, axis=0)
        a_vec = tl.exp(cum)
        cum_last = tl.sum(tl.where(offs_c == CHUNK - 1, cum, 0.0), axis=0)
        a_last = tl.exp(cum_last)
        f_suffix = tl.exp(cum_last - cum)
        cum_diff = cum[:, None] - cum[None, :]

        k_tile = tl.load(k_base + offs_t[:, None] * skt + offs_k[None, :] * skk,
                         mask=mask_t[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        q_tile = tl.load(q_base + offs_t[:, None] * sqt + offs_k[None, :] * sqk,
                         mask=mask_t[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        v_tile = tl.load(v_base + offs_t[:, None] * svt + offs_v_global[None, :] * svv,
                         mask=mask_t[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
        dy = scale * tl.load(do_base + offs_t[:, None] * sdt + offs_v_global[None, :] * sdv,
                             mask=mask_t[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
        w_tile = tl.load(w_ptr + slice_idx * CHUNK * V_DIM + offs_c[:, None] * V_DIM + offs_v_global[None, :],
                         mask=mask_v[None, :], other=0.0)
        s0_tile = tl.load(s_ptr + slice_idx * K_DIM * V_DIM + offs_k[:, None] * V_DIM + offs_v_global[None, :],
                          mask=mask_k[:, None] & mask_v[None, :], other=0.0)
        m_mat = tl.load(m_ptr + slice_idx * CHUNK * CHUNK + offs_c[:, None] * CHUNK + offs_c[None, :])
        inv_mat = tl.load(inv_ptr + slice_idx * CHUNK * CHUNK + offs_c[:, None] * CHUNK + offs_c[None, :])

        ksuf = k_tile * f_suffix[:, None]

        g_w = tl.dot(tl.trans(m_mat), dy, input_precision="tf32") + tl.dot(ksuf, dstate, input_precision="tf32")
        d_m = tl.where(le_mask, tl.dot(dy, tl.trans(w_tile), input_precision="tf32"), 0.0)
        d_ksuf = tl.dot(w_tile, tl.trans(dstate), input_precision="tf32")
        d_qa = tl.dot(dy, tl.trans(s0_tile), input_precision="tf32")

        z_mat = tl.dot(tl.trans(inv_mat), g_w, input_precision="tf32")
        d_l = tl.where(lt_mask, -tl.dot(z_mat, tl.trans(w_tile), input_precision="tf32"), 0.0)
        d_d = -tl.dot(z_mat, tl.trans(s0_tile), input_precision="tf32")

        # Adjoint carried to the previous chunk (uses the old dstate).
        dstate_prev = a_last * dstate \
            + tl.dot(tl.trans(q_tile * a_vec[:, None]), dy, input_precision="tf32") \
            - tl.dot(tl.trans(k_tile * (beta_vec * a_vec)[:, None]), z_mat, input_precision="tf32")

        kk = tl.dot(k_tile, tl.trans(k_tile), input_precision="tf32")
        e_le = tl.exp(tl.where(le_mask, cum_diff, 0.0))
        e_lt = tl.exp(tl.where(lt_mask, cum_diff, 0.0))
        l_mat = tl.where(lt_mask, kk * beta_vec[:, None] * e_lt, 0.0)

        d_qk = d_m * e_le
        dq_tile = tl.dot(d_qk, k_tile, input_precision="tf32") + a_vec[:, None] * d_qa

        d_kk = d_l * e_lt
        dk_tile = tl.dot(tl.trans(d_qk), q_tile, input_precision="tf32") \
            + tl.dot(d_kk + tl.trans(d_kk), k_tile, input_precision="tf32") \
            + d_ksuf * f_suffix[:, None] \
            + d_d * (beta_vec * a_vec)[:, None]

        tl.store(dv_base + offs_t[:, None] * V_DIM + offs_v_global[None, :],
                 beta_vec[:, None] * z_mat, mask=mask_t[:, None] & mask_v[None, :])

        dbeta_vec = tl.sum(d_l * e_lt * kk, axis=1) \
            + tl.sum(z_mat * v_tile, axis=1) \
            + tl.sum(d_d * (a_vec[:, None] * k_tile), axis=1)

        dcum = tl.sum(d_qa * q_tile, axis=1) * a_vec \
            + tl.sum(d_d * (beta_vec[:, None] * k_tile), axis=1) * a_vec
        s_ksuf = tl.sum(d_ksuf * ksuf, axis=1)
        dcum = dcum - s_ksuf
        dcum = dcum + tl.where(offs_c == CHUNK - 1, tl.sum(s_ksuf), 0.0)
        dcum = dcum + tl.where(offs_c == CHUNK - 1,
                                a_last * tl.sum(dstate * s0_tile), 0.0)
        h_m = d_m * m_mat
        h_l = d_l * l_mat
        dcum = dcum + tl.sum(h_m, axis=1) - tl.sum(h_m, axis=0) \
            + tl.sum(h_l, axis=1) - tl.sum(h_l, axis=0)

        # dg_j = sum_{i >= j} dcum_i  (reverse cumsum over the chunk).
        dg_vec = tl.sum(tl.where(col >= row, dcum[None, :], 0.0), axis=1)

        tl.atomic_add(dq_base + offs_t[:, None] * K_DIM + offs_k[None, :],
                      dq_tile, mask=mask_t[:, None] & mask_k[None, :])
        tl.atomic_add(dk_base + offs_t[:, None] * K_DIM + offs_k[None, :],
                      dk_tile, mask=mask_t[:, None] & mask_k[None, :])
        tl.atomic_add(dg_base + offs_t, dg_vec, mask=mask_t)
        tl.atomic_add(dbeta_base + offs_t, dbeta_vec, mask=mask_t)

        dstate = dstate_prev


def _next_power_of_2(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _forward_impl(q, k, v, g, beta, scale, need_bwd):
    B, H, T, K_DIM = q.shape
    V_DIM = v.shape[-1]
    out = torch.empty((B, H, T, V_DIM), device=q.device, dtype=torch.float32)
    if out.numel() == 0:
        dummy = torch.empty((0,), device=q.device, dtype=torch.float32)
        return out, dummy, dummy, 64, (T + 63) // 64

    BK = max(16, _next_power_of_2(K_DIM))
    BVFULL = max(16, _next_power_of_2(V_DIM))
    use_fast = (
        q.is_cuda and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16 and g.dtype == torch.float32
        and beta.dtype == torch.float32 and BK <= 128 and BVFULL <= 128
    )

    bh = B * H
    if use_fast:
        # Smaller chunks when saving backward intermediates keeps the
        # backward tiles (and the per-chunk boundary states) manageable.
        CHUNK = 32 if need_bwd else 64
        num_chunks = (T + CHUNK - 1) // CHUNK
        prep_elems = bh * num_chunks * CHUNK * (2 * K_DIM + 2 * V_DIM + 1)
        use_fast = prep_elems <= 200_000_000 and bh * num_chunks <= 1_000_000

    if need_bwd:
        num_chunks_tmp = (T + (CHUNK if use_fast else (16 if K_DIM > 128 else 32)) - 1) // \
            (CHUNK if use_fast else (16 if K_DIM > 128 else 32))
        chunk_tmp = CHUNK if use_fast else (16 if K_DIM > 128 else 32)
        w_buf = torch.empty((bh, num_chunks_tmp, chunk_tmp, V_DIM),
                            device=q.device, dtype=torch.float32)
        s_buf = torch.empty((bh, num_chunks_tmp, K_DIM, V_DIM),
                            device=q.device, dtype=torch.float32)
    else:
        w_buf = torch.empty((0,), device=q.device, dtype=torch.float32)
        s_buf = torch.empty((0,), device=q.device, dtype=torch.float32)
        chunk_tmp = CHUNK if use_fast else (16 if K_DIM > 128 else 32)
        num_chunks_tmp = (T + chunk_tmp - 1) // chunk_tmp

    if use_fast:
        BV = 64 if V_DIM > 64 else BVFULL
        kg_buf = torch.empty((bh, num_chunks_tmp, CHUNK, K_DIM), device=q.device, dtype=torch.float32)
        qeff_buf = torch.empty((bh, num_chunks_tmp, CHUNK, K_DIM), device=q.device, dtype=torch.float32)
        u_buf = torch.empty((bh, num_chunks_tmp, CHUNK, V_DIM), device=q.device, dtype=torch.float32)
        ointra_buf = torch.empty((bh, num_chunks_tmp, CHUNK, V_DIM), device=q.device, dtype=torch.float32)
        cum_buf = torch.empty((bh, num_chunks_tmp, CHUNK), device=q.device, dtype=torch.float32)

        _prep_kernel[(bh * num_chunks_tmp,)](
            q, k, v, g, beta, kg_buf, qeff_buf, u_buf, ointra_buf, cum_buf,
            H, T, K_DIM, V_DIM, num_chunks_tmp,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            g.stride(0), g.stride(1), g.stride(2),
            beta.stride(0), beta.stride(1), beta.stride(2),
            CHUNK=CHUNK, BK=BK, BVFULL=BVFULL, num_warps=8, num_stages=1)

        _scan_kernel[(bh, (V_DIM + BV - 1) // BV)](
            k, kg_buf, qeff_buf, u_buf, ointra_buf, cum_buf, out, w_buf, s_buf,
            float(scale), H, T, K_DIM, V_DIM, num_chunks_tmp,
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            CHUNK=CHUNK, BK=BK, BV=BV, STORE_BWD=bool(need_bwd),
            num_warps=8, num_stages=1)
        return out, w_buf, s_buf, CHUNK, num_chunks_tmp

    if K_DIM > 128:
        CHUNK = 16
        BV = 16
        num_warps = 8
    else:
        CHUNK = 32
        BV = 32 if V_DIM > 32 else max(16, _next_power_of_2(V_DIM))
        num_warps = 4
    num_chunks = (T + CHUNK - 1) // CHUNK
    _fallback_kernel[(B * H, (V_DIM + BV - 1) // BV)](
        q, k, v, g, beta, out, w_buf, s_buf,
        float(scale), H, T, K_DIM, V_DIM, num_chunks,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        beta.stride(0), beta.stride(1), beta.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        CHUNK=CHUNK, BK=BK, BV=BV, STORE_BWD=bool(need_bwd),
        num_warps=num_warps, num_stages=1)
    return out, w_buf, s_buf, CHUNK, num_chunks


class _GatedDeltaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        need_bwd = any(
            torch.is_tensor(x) and x.requires_grad for x in (q, k, v, g, beta))
        out, w_buf, s_buf, chunk, num_chunks = _forward_impl(
            q, k, v, g, beta, scale, need_bwd)
        ctx.save_for_backward(q, k, v, g, beta, w_buf, s_buf)
        ctx.scale = float(scale)
        ctx.chunk = chunk
        ctx.num_chunks = num_chunks
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, g, beta, w_buf, s_buf = ctx.saved_tensors
        B, H, T, K_DIM = q.shape
        V_DIM = v.shape[-1]
        CHUNK = ctx.chunk
        num_chunks = ctx.num_chunks
        bh = B * H
        BK = max(16, _next_power_of_2(K_DIM))
        dev = q.device

        dq_buf = torch.zeros((B, H, T, K_DIM), device=dev, dtype=torch.float32)
        dk_buf = torch.zeros((B, H, T, K_DIM), device=dev, dtype=torch.float32)
        dv_buf = torch.empty((B, H, T, V_DIM), device=dev, dtype=torch.float32)
        dg_buf = torch.zeros((B, H, T), device=dev, dtype=torch.float32)
        dbeta_buf = torch.zeros((B, H, T), device=dev, dtype=torch.float32)

        if T > 0 and K_DIM > 0 and V_DIM > 0:
            inv_buf = torch.empty((bh, num_chunks, CHUNK, CHUNK), device=dev, dtype=torch.float32)
            m_buf = torch.empty((bh, num_chunks, CHUNK, CHUNK), device=dev, dtype=torch.float32)
            _bwd_prep_kernel[(bh * num_chunks,)](
                q, k, g, beta, inv_buf, m_buf,
                H, T, K_DIM, num_chunks,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                g.stride(0), g.stride(1), g.stride(2),
                beta.stride(0), beta.stride(1), beta.stride(2),
                CHUNK=CHUNK, BK=BK,
                num_warps=4 if CHUNK <= 32 else 8, num_stages=1)

            BV = 32 if V_DIM > 32 else max(16, _next_power_of_2(V_DIM))
            go = grad_out.contiguous() if not grad_out.is_contiguous() else grad_out
            _bwd_scan_kernel[(bh, (V_DIM + BV - 1) // BV)](
                q, k, v, g, beta, go, w_buf, s_buf, inv_buf, m_buf,
                dq_buf, dk_buf, dv_buf, dg_buf, dbeta_buf,
                ctx.scale, H, T, K_DIM, V_DIM, num_chunks,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                g.stride(0), g.stride(1), g.stride(2),
                beta.stride(0), beta.stride(1), beta.stride(2),
                go.stride(0), go.stride(1), go.stride(2), go.stride(3),
                CHUNK=CHUNK, BK=BK, BV=BV, num_warps=4, num_stages=1)
        else:
            dv_buf.zero_()

        # e and gamma are unused by the reference, hence exactly zero grads.
        return (dq_buf.to(q.dtype), dk_buf.to(k.dtype), dv_buf.to(v.dtype),
                torch.zeros_like(ctx_saved_e(ctx, q)), dg_buf.to(g.dtype),
                dbeta_buf.to(beta.dtype), torch.zeros_like(ctx_saved_gamma(ctx, q)), None)


# Small helpers so backward can build zero grads for e / gamma without
# having saved them (they never influence the output).
def ctx_saved_e(ctx, ref):
    return ctx._e_shape_ref


def ctx_saved_gamma(ctx, ref):
    return ctx._gamma_shape_ref


_old_forward = _GatedDeltaFunction.forward
_old_apply = _GatedDeltaFunction.apply


def _forward_with_refs(ctx, q, k, v, e, g, beta, gamma, scale):
    ctx._e_shape_ref = e
    ctx._gamma_shape_ref = gamma
    return _old_forward(ctx, q, k, v, e, g, beta, gamma, scale)


_GatedDeltaFunction.forward = staticmethod(_forward_with_refs)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _GatedDeltaFunction.apply(q, k, v, e, g, beta, gamma, scale)
```

The forward reuses the chunkwise gated-delta kernels (parallel per-chunk preparation plus a register-state scan), and when gradients are needed it additionally stores each chunk's update `W` and boundary state `S0`. The backward recomputes `(I+L)⁻¹` and the decay-masked `QKᵀ` per chunk in a Triton kernel, then a reverse-time Triton scan carries the state adjoint in registers and evaluates the closed-form chunk-local derivatives for `q, k, v, g, beta`, using atomics only for reductions across V-blocks and returning `e`/`gamma` gradients as exact zeros since the reference ignores them. I was unable to test on hardware, so the main uncertainties are register/shared-memory pressure in the backward scan for the widest shapes and TF32-level accuracy of the gradient matmuls relative to the checker's tolerance.