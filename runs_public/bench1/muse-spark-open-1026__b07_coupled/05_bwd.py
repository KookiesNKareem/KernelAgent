import torch
import triton
import triton.language as tl

BLOCK_K = 32
BLOCK_V = 32
CHUNK = 64

# ---------- fast forward for inference (no autograd) ----------
@triton.jit
def _fwd_col_NK4(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    s0_q, s1_q, s2_q, s3_q, s0_k, s1_k, s2_k, s3_k, s0_v, s1_v, s2_v, s3_v,
    s0_e, s1_e, s2_e, s3_e, s0_g, s1_g, s2_g, s3_g, s0_b, s1_b, s2_b, s0_gm, s1_gm, s2_gm, s0_o, s1_o, s2_o, s3_o,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NV: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, scale,
):
    pid = tl.program_id(0)
    iv = pid % NV
    rem = pid // NV
    h = rem % H
    b = rem // H
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_col = iv * BLOCK_V
    s0 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    s1 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    s2 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    s3 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    for t in range(T):
        bt = tl.load(beta_ptr + b * s0_b + h * s1_b + t * s2_b).to(tl.float32)
        gt = tl.load(gamma_ptr + b * s0_gm + h * s1_gm + t * s2_gm).to(tl.float32)
        ct = gt * (1.0 - bt)
        v_vals = tl.load(v_ptr + b * s0_v + h * s1_v + t * s2_v + (v_col + offs_v) * s3_v, mask=(v_col + offs_v) < V, other=0.0).to(tl.float32)
        e0 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + offs_k * s3_e, mask=offs_k < K, other=0.0).to(tl.float32)
        k0 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + offs_k * s3_k, mask=offs_k < K, other=0.0).to(tl.float32)
        g0 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + offs_k * s3_g, mask=offs_k < K, other=0.0).to(tl.float32)
        d0 = tl.exp(g0)
        q0 = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + offs_k * s3_q, mask=offs_k < K, other=0.0).to(tl.float32)
        e1 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + (32 + offs_k) * s3_e, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        k1 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + (32 + offs_k) * s3_k, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        g1 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + (32 + offs_k) * s3_g, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        d1 = tl.exp(g1)
        q1 = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + (32 + offs_k) * s3_q, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        e2 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + (64 + offs_k) * s3_e, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        k2 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + (64 + offs_k) * s3_k, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        g2 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + (64 + offs_k) * s3_g, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        d2 = tl.exp(g2)
        q2 = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + (64 + offs_k) * s3_q, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        e3 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + (96 + offs_k) * s3_e, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        k3 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + (96 + offs_k) * s3_k, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        g3 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + (96 + offs_k) * s3_g, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        d3 = tl.exp(g3)
        q3 = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + (96 + offs_k) * s3_q, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        Bdot = tl.sum(k0 * e0) + tl.sum(k1 * e1) + tl.sum(k2 * e2) + tl.sum(k3 * e3)
        eTS = tl.sum((e0 * d0)[:, None] * s0, axis=0) + tl.sum((e1 * d1)[:, None] * s1, axis=0) + tl.sum((e2 * d2)[:, None] * s2, axis=0) + tl.sum((e3 * d3)[:, None] * s3, axis=0)
        A = tl.sum((k0 * d0)[:, None] * s0, axis=0) + tl.sum((k1 * d1)[:, None] * s1, axis=0) + tl.sum((k2 * d2)[:, None] * s2, axis=0) + tl.sum((k3 * d3)[:, None] * s3, axis=0)
        pred = A - ct * Bdot * eTS
        delta = v_vals - pred
        s0 = s0 * d0[:, None] - ct * e0[:, None] * eTS[None, :] + bt * k0[:, None] * delta[None, :]
        s1 = s1 * d1[:, None] - ct * e1[:, None] * eTS[None, :] + bt * k1[:, None] * delta[None, :]
        s2 = s2 * d2[:, None] - ct * e2[:, None] * eTS[None, :] + bt * k2[:, None] * delta[None, :]
        s3 = s3 * d3[:, None] - ct * e3[:, None] * eTS[None, :] + bt * k3[:, None] * delta[None, :]
        out_acc = tl.sum(q0[:, None] * s0, axis=0) + tl.sum(q1[:, None] * s1, axis=0) + tl.sum(q2[:, None] * s2, axis=0) + tl.sum(q3[:, None] * s3, axis=0)
        out_acc = out_acc * scale
        tl.store(out_ptr + b * s0_o + h * s1_o + t * s2_o + (v_col + offs_v) * s3_o, out_acc, mask=(v_col + offs_v) < V)

# HBM fallback for forward (any K)
@triton.jit
def _fwd_hbm(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, s_ptr, out_ptr,
    s0_q, s1_q, s2_q, s3_q, s0_k, s1_k, s2_k, s3_k, s0_v, s1_v, s2_v, s3_v, s0_e, s1_e, s2_e, s3_e, s0_g, s1_g, s2_g, s3_g, s0_b, s1_b, s2_b, s0_gm, s1_gm, s2_gm, s0_s, s1_s, s2_s, s3_s, s0_o, s1_o, s2_o, s3_o,
    B, H, T, K, V, NV: tl.constexpr, NK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, scale,
):
    pid = tl.program_id(0)
    iv = pid % NV
    rem = pid // NV
    h = rem % H
    b = rem // H
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_col = iv * BLOCK_V
    q_bh = q_ptr + b * s0_q + h * s1_q
    k_bh = k_ptr + b * s0_k + h * s1_k
    v_bh = v_ptr + b * s0_v + h * s1_v
    e_bh = e_ptr + b * s0_e + h * s1_e
    g_bh = g_ptr + b * s0_g + h * s1_g
    b_bh = beta_ptr + b * s0_b + h * s1_b
    gm_bh = gamma_ptr + b * s0_gm + h * s1_gm
    s_bh = s_ptr + b * s0_s + h * s1_s
    o_bh = out_ptr + b * s0_o + h * s1_o
    for t in range(T):
        bt = tl.load(b_bh + t * s2_b).to(tl.float32)
        gt = tl.load(gm_bh + t * s2_gm).to(tl.float32)
        ct = gt * (1.0 - bt)
        v_vals = tl.load(v_bh + t * s2_v + (v_col + offs_v) * s3_v, mask=(v_col + offs_v) < V, other=0.0).to(tl.float32)
        eTS = tl.zeros((BLOCK_V,), dtype=tl.float32)
        A = tl.zeros((BLOCK_V,), dtype=tl.float32)
        Bdot = 0.0
        for ik in range(NK):
            k_row = ik * BLOCK_K
            e_blk = tl.load(e_bh + t * s2_e + (k_row + offs_k) * s3_e, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            k_blk = tl.load(k_bh + t * s2_k + (k_row + offs_k) * s3_k, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            g_blk = tl.load(g_bh + t * s2_g + (k_row + offs_k) * s3_g, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            d_blk = tl.exp(g_blk)
            Bdot += tl.sum(k_blk * e_blk)
            s_tile = tl.load(s_bh + k_row * s2_s + v_col * s3_s + offs_k[:, None] * s2_s + offs_v[None, :] * s3_s, mask=((k_row + offs_k)[:, None] < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
            eTS += tl.sum((e_blk * d_blk)[:, None] * s_tile, axis=0)
            A += tl.sum((k_blk * d_blk)[:, None] * s_tile, axis=0)
        pred = A - ct * Bdot * eTS
        delta = v_vals - pred
        out_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for ik in range(NK):
            k_row = ik * BLOCK_K
            e_blk = tl.load(e_bh + t * s2_e + (k_row + offs_k) * s3_e, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            k_blk = tl.load(k_bh + t * s2_k + (k_row + offs_k) * s3_k, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            g_blk = tl.load(g_bh + t * s2_g + (k_row + offs_k) * s3_g, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            q_blk = tl.load(q_bh + t * s2_q + (k_row + offs_k) * s3_q, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            d_blk = tl.exp(g_blk)
            s_tile = tl.load(s_bh + k_row * s2_s + v_col * s3_s + offs_k[:, None] * s2_s + offs_v[None, :] * s3_s, mask=((k_row + offs_k)[:, None] < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
            s_new = s_tile * d_blk[:, None] - ct * e_blk[:, None] * eTS[None, :] + bt * k_blk[:, None] * delta[None, :]
            tl.store(s_bh + k_row * s2_s + v_col * s3_s + offs_k[:, None] * s2_s + offs_v[None, :] * s3_s, s_new, mask=((k_row + offs_k)[:, None] < K) & ((v_col + offs_v)[None, :] < V))
            out_acc += tl.sum(q_blk[:, None] * s_new, axis=0)
        out_acc = out_acc * scale
        tl.store(o_bh + t * s2_o + (v_col + offs_v) * s3_o, out_acc, mask=(v_col + offs_v) < V)

# ---------- chunked forward for training ----------
@triton.jit
def _fwd_chunk_NK4(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, S_init_ptr, S_final_ptr, out_ptr,
    s0_q, s1_q, s2_q, s3_q, s0_k, s1_k, s2_k, s3_k, s0_v, s1_v, s2_v, s3_v, s0_e, s1_e, s2_e, s3_e, s0_g, s1_g, s2_g, s3_g, s0_b, s1_b, s2_b, s0_gm, s1_gm, s2_gm, s0_si, s1_si, s2_si, s3_si, s0_sf, s1_sf, s2_sf, s3_sf, s0_o, s1_o, s2_o, s3_o,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NV: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, scale, t_start: tl.constexpr, Tcur: tl.constexpr,
):
    pid = tl.program_id(0)
    iv = pid % NV
    rem = pid // NV
    h = rem % H
    b = rem // H
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_col = iv * BLOCK_V
    # load S_init
    s0 = tl.load(S_init_ptr + b * s0_si + h * s1_si + 0 * s2_si + v_col * s3_si + offs_k[:, None] * s2_si + offs_v[None, :] * s3_si, mask=(offs_k[:, None] < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
    s1 = tl.load(S_init_ptr + b * s0_si + h * s1_si + 32 * s2_si + v_col * s3_si + offs_k[:, None] * s2_si + offs_v[None, :] * s3_si, mask=((32 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
    s2 = tl.load(S_init_ptr + b * s0_si + h * s1_si + 64 * s2_si + v_col * s3_si + offs_k[:, None] * s2_si + offs_v[None, :] * s3_si, mask=((64 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
    s3 = tl.load(S_init_ptr + b * s0_si + h * s1_si + 96 * s2_si + v_col * s3_si + offs_k[:, None] * s2_si + offs_v[None, :] * s3_si, mask=((96 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
    for tc in range(Tcur):
        t = t_start + tc
        bt = tl.load(beta_ptr + b * s0_b + h * s1_b + t * s2_b).to(tl.float32)
        gt = tl.load(gamma_ptr + b * s0_gm + h * s1_gm + t * s2_gm).to(tl.float32)
        ct = gt * (1.0 - bt)
        v_vals = tl.load(v_ptr + b * s0_v + h * s1_v + t * s2_v + (v_col + offs_v) * s3_v, mask=(v_col + offs_v) < V, other=0.0).to(tl.float32)
        e0 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + offs_k * s3_e, mask=offs_k < K, other=0.0).to(tl.float32)
        k0 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + offs_k * s3_k, mask=offs_k < K, other=0.0).to(tl.float32)
        g0 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + offs_k * s3_g, mask=offs_k < K, other=0.0).to(tl.float32)
        d0 = tl.exp(g0)
        q0 = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + offs_k * s3_q, mask=offs_k < K, other=0.0).to(tl.float32)
        e1 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + (32 + offs_k) * s3_e, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        k1 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + (32 + offs_k) * s3_k, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        g1 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + (32 + offs_k) * s3_g, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        d1 = tl.exp(g1)
        q1 = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + (32 + offs_k) * s3_q, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        e2 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + (64 + offs_k) * s3_e, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        k2 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + (64 + offs_k) * s3_k, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        g2 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + (64 + offs_k) * s3_g, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        d2 = tl.exp(g2)
        q2 = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + (64 + offs_k) * s3_q, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        e3 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + (96 + offs_k) * s3_e, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        k3 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + (96 + offs_k) * s3_k, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        g3 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + (96 + offs_k) * s3_g, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        d3 = tl.exp(g3)
        q3 = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + (96 + offs_k) * s3_q, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        Bdot = tl.sum(k0 * e0) + tl.sum(k1 * e1) + tl.sum(k2 * e2) + tl.sum(k3 * e3)
        eTS = tl.sum((e0 * d0)[:, None] * s0, axis=0) + tl.sum((e1 * d1)[:, None] * s1, axis=0) + tl.sum((e2 * d2)[:, None] * s2, axis=0) + tl.sum((e3 * d3)[:, None] * s3, axis=0)
        A = tl.sum((k0 * d0)[:, None] * s0, axis=0) + tl.sum((k1 * d1)[:, None] * s1, axis=0) + tl.sum((k2 * d2)[:, None] * s2, axis=0) + tl.sum((k3 * d3)[:, None] * s3, axis=0)
        pred = A - ct * Bdot * eTS
        delta = v_vals - pred
        s0 = s0 * d0[:, None] - ct * e0[:, None] * eTS[None, :] + bt * k0[:, None] * delta[None, :]
        s1 = s1 * d1[:, None] - ct * e1[:, None] * eTS[None, :] + bt * k1[:, None] * delta[None, :]
        s2 = s2 * d2[:, None] - ct * e2[:, None] * eTS[None, :] + bt * k2[:, None] * delta[None, :]
        s3 = s3 * d3[:, None] - ct * e3[:, None] * eTS[None, :] + bt * k3[:, None] * delta[None, :]
        out_acc = tl.sum(q0[:, None] * s0, axis=0) + tl.sum(q1[:, None] * s1, axis=0) + tl.sum(q2[:, None] * s2, axis=0) + tl.sum(q3[:, None] * s3, axis=0)
        out_acc = out_acc * scale
        tl.store(out_ptr + b * s0_o + h * s1_o + t * s2_o + (v_col + offs_v) * s3_o, out_acc, mask=(v_col + offs_v) < V)
    tl.store(S_final_ptr + b * s0_sf + h * s1_sf + 0 * s2_sf + v_col * s3_sf + offs_k[:, None] * s2_sf + offs_v[None, :] * s3_sf, s0, mask=(offs_k[:, None] < K) & ((v_col + offs_v)[None, :] < V))
    tl.store(S_final_ptr + b * s0_sf + h * s1_sf + 32 * s2_sf + v_col * s3_sf + offs_k[:, None] * s2_sf + offs_v[None, :] * s3_sf, s1, mask=((32 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V))
    tl.store(S_final_ptr + b * s0_sf + h * s1_sf + 64 * s2_sf + v_col * s3_sf + offs_k[:, None] * s2_sf + offs_v[None, :] * s3_sf, s2, mask=((64 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V))
    tl.store(S_final_ptr + b * s0_sf + h * s1_sf + 96 * s2_sf + v_col * s3_sf + offs_k[:, None] * s2_sf + offs_v[None, :] * s3_sf, s3, mask=((96 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V))

# ---------- backward chunk for NK=4 ----------
@triton.jit
def _bwd_chunk_NK4(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, grad_out_ptr,
    S_init_ptr, S_tmp_ptr, lambda_ptr,
    gq_stage_ptr, gk_stage_ptr, ge_stage_ptr, gg_stage_ptr, gb_stage_ptr, ggm_stage_ptr, gv_ptr,
    s0_q, s1_q, s2_q, s3_q, s0_k, s1_k, s2_k, s3_k, s0_v, s1_v, s2_v, s3_v, s0_e, s1_e, s2_e, s3_e, s0_g, s1_g, s2_g, s3_g, s0_b, s1_b, s2_b, s0_gm, s1_gm, s2_gm, s0_go, s1_go, s2_go, s3_go,
    s0_si, s1_si, s2_si, s3_si, s0_tmp, s1_tmp, s2_tmp, s3_tmp, s4_tmp, s0_lam, s1_lam, s2_lam, s3_lam,
    s0_gq, s1_gq, s2_gq, s3_gq, s4_gq, s0_gk, s1_gk, s2_gk, s3_gk, s4_gk, s0_ge, s1_ge, s2_ge, s3_ge, s4_ge, s0_gg, s1_gg, s2_gg, s3_gg, s4_gg, s0_gb, s1_gb, s2_gb, s3_gb, s0_ggm, s1_ggm, s2_ggm, s3_ggm, s0_gv, s1_gv, s2_gv, s3_gv,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NV: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, scale, t_start: tl.constexpr, Tcur: tl.constexpr,
):
    pid = tl.program_id(0)
    iv = pid % NV
    rem = pid // NV
    h = rem % H
    b = rem // H
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_col = iv * BLOCK_V
    # load S_init
    s0 = tl.load(S_init_ptr + b * s0_si + h * s1_si + 0 * s2_si + v_col * s3_si + offs_k[:, None] * s2_si + offs_v[None, :] * s3_si, mask=(offs_k[:, None] < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
    s1 = tl.load(S_init_ptr + b * s0_si + h * s1_si + 32 * s2_si + v_col * s3_si + offs_k[:, None] * s2_si + offs_v[None, :] * s3_si, mask=((32 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
    s2 = tl.load(S_init_ptr + b * s0_si + h * s1_si + 64 * s2_si + v_col * s3_si + offs_k[:, None] * s2_si + offs_v[None, :] * s3_si, mask=((64 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
    s3 = tl.load(S_init_ptr + b * s0_si + h * s1_si + 96 * s2_si + v_col * s3_si + offs_k[:, None] * s2_si + offs_v[None, :] * s3_si, mask=((96 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
    # forward pass to fill S_tmp
    for tc in range(Tcur):
        t = t_start + tc
        bt = tl.load(beta_ptr + b * s0_b + h * s1_b + t * s2_b).to(tl.float32)
        gt = tl.load(gamma_ptr + b * s0_gm + h * s1_gm + t * s2_gm).to(tl.float32)
        ct = gt * (1.0 - bt)
        v_vals = tl.load(v_ptr + b * s0_v + h * s1_v + t * s2_v + (v_col + offs_v) * s3_v, mask=(v_col + offs_v) < V, other=0.0).to(tl.float32)
        e0 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + offs_k * s3_e, mask=offs_k < K, other=0.0).to(tl.float32)
        k0 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + offs_k * s3_k, mask=offs_k < K, other=0.0).to(tl.float32)
        g0 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + offs_k * s3_g, mask=offs_k < K, other=0.0).to(tl.float32)
        d0 = tl.exp(g0)
        e1 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + (32 + offs_k) * s3_e, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        k1 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + (32 + offs_k) * s3_k, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        g1 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + (32 + offs_k) * s3_g, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        d1 = tl.exp(g1)
        e2 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + (64 + offs_k) * s3_e, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        k2 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + (64 + offs_k) * s3_k, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        g2 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + (64 + offs_k) * s3_g, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        d2 = tl.exp(g2)
        e3 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + (96 + offs_k) * s3_e, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        k3 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + (96 + offs_k) * s3_k, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        g3 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + (96 + offs_k) * s3_g, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        d3 = tl.exp(g3)
        Bdot = tl.sum(k0 * e0) + tl.sum(k1 * e1) + tl.sum(k2 * e2) + tl.sum(k3 * e3)
        eTS = tl.sum((e0 * d0)[:, None] * s0, axis=0) + tl.sum((e1 * d1)[:, None] * s1, axis=0) + tl.sum((e2 * d2)[:, None] * s2, axis=0) + tl.sum((e3 * d3)[:, None] * s3, axis=0)
        A = tl.sum((k0 * d0)[:, None] * s0, axis=0) + tl.sum((k1 * d1)[:, None] * s1, axis=0) + tl.sum((k2 * d2)[:, None] * s2, axis=0) + tl.sum((k3 * d3)[:, None] * s3, axis=0)
        pred = A - ct * Bdot * eTS
        delta = v_vals - pred
        s0 = s0 * d0[:, None] - ct * e0[:, None] * eTS[None, :] + bt * k0[:, None] * delta[None, :]
        s1 = s1 * d1[:, None] - ct * e1[:, None] * eTS[None, :] + bt * k1[:, None] * delta[None, :]
        s2 = s2 * d2[:, None] - ct * e2[:, None] * eTS[None, :] + bt * k2[:, None] * delta[None, :]
        s3 = s3 * d3[:, None] - ct * e3[:, None] * eTS[None, :] + bt * k3[:, None] * delta[None, :]
        tl.store(S_tmp_ptr + b * s0_tmp + h * s1_tmp + tc * s2_tmp + 0 * s3_tmp + v_col * s4_tmp + offs_k[:, None] * s3_tmp + offs_v[None, :] * s4_tmp, s0, mask=(offs_k[:, None] < K) & ((v_col + offs_v)[None, :] < V))
        tl.store(S_tmp_ptr + b * s0_tmp + h * s1_tmp + tc * s2_tmp + 32 * s3_tmp + v_col * s4_tmp + offs_k[:, None] * s3_tmp + offs_v[None, :] * s4_tmp, s1, mask=((32 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V))
        tl.store(S_tmp_ptr + b * s0_tmp + h * s1_tmp + tc * s2_tmp + 64 * s3_tmp + v_col * s4_tmp + offs_k[:, None] * s3_tmp + offs_v[None, :] * s4_tmp, s2, mask=((64 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V))
        tl.store(S_tmp_ptr + b * s0_tmp + h * s1_tmp + tc * s2_tmp + 96 * s3_tmp + v_col * s4_tmp + offs_k[:, None] * s3_tmp + offs_v[None, :] * s4_tmp, s3, mask=((96 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V))
    # load lambda
    lam0 = tl.load(lambda_ptr + b * s0_lam + h * s1_lam + 0 * s2_lam + v_col * s3_lam + offs_k[:, None] * s2_lam + offs_v[None, :] * s3_lam, mask=(offs_k[:, None] < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
    lam1 = tl.load(lambda_ptr + b * s0_lam + h * s1_lam + 32 * s2_lam + v_col * s3_lam + offs_k[:, None] * s2_lam + offs_v[None, :] * s3_lam, mask=((32 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
    lam2 = tl.load(lambda_ptr + b * s0_lam + h * s1_lam + 64 * s2_lam + v_col * s3_lam + offs_k[:, None] * s2_lam + offs_v[None, :] * s3_lam, mask=((64 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
    lam3 = tl.load(lambda_ptr + b * s0_lam + h * s1_lam + 96 * s2_lam + v_col * s3_lam + offs_k[:, None] * s2_lam + offs_v[None, :] * s3_lam, mask=((96 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
    for tc in range(Tcur):
        t = t_start + Tcur - 1 - tc
        # transform lambda with t+1 if needed
        if t + 1 < T:
            bt_n = tl.load(beta_ptr + b * s0_b + h * s1_b + (t + 1) * s2_b).to(tl.float32)
            gt_n = tl.load(gamma_ptr + b * s0_gm + h * s1_gm + (t + 1) * s2_gm).to(tl.float32)
            ct_n = gt_n * (1.0 - bt_n)
            e0n = tl.load(e_ptr + b * s0_e + h * s1_e + (t + 1) * s2_e + offs_k * s3_e, mask=offs_k < K, other=0.0).to(tl.float32)
            k0n = tl.load(k_ptr + b * s0_k + h * s1_k + (t + 1) * s2_k + offs_k * s3_k, mask=offs_k < K, other=0.0).to(tl.float32)
            g0n = tl.load(g_ptr + b * s0_g + h * s1_g + (t + 1) * s2_g + offs_k * s3_g, mask=offs_k < K, other=0.0).to(tl.float32)
            d0n = tl.exp(g0n)
            e1n = tl.load(e_ptr + b * s0_e + h * s1_e + (t + 1) * s2_e + (32 + offs_k) * s3_e, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
            k1n = tl.load(k_ptr + b * s0_k + h * s1_k + (t + 1) * s2_k + (32 + offs_k) * s3_k, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
            g1n = tl.load(g_ptr + b * s0_g + h * s1_g + (t + 1) * s2_g + (32 + offs_k) * s3_g, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
            d1n = tl.exp(g1n)
            e2n = tl.load(e_ptr + b * s0_e + h * s1_e + (t + 1) * s2_e + (64 + offs_k) * s3_e, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
            k2n = tl.load(k_ptr + b * s0_k + h * s1_k + (t + 1) * s2_k + (64 + offs_k) * s3_k, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
            g2n = tl.load(g_ptr + b * s0_g + h * s1_g + (t + 1) * s2_g + (64 + offs_k) * s3_g, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
            d2n = tl.exp(g2n)
            e3n = tl.load(e_ptr + b * s0_e + h * s1_e + (t + 1) * s2_e + (96 + offs_k) * s3_e, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
            k3n = tl.load(k_ptr + b * s0_k + h * s1_k + (t + 1) * s2_k + (96 + offs_k) * s3_k, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
            g3n = tl.load(g_ptr + b * s0_g + h * s1_g + (t + 1) * s2_g + (96 + offs_k) * s3_g, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
            d3n = tl.exp(g3n)
            kTlam = tl.sum(k0n[:, None] * lam0, axis=0) + tl.sum(k1n[:, None] * lam1, axis=0) + tl.sum(k2n[:, None] * lam2, axis=0) + tl.sum(k3n[:, None] * lam3, axis=0)
            y0 = lam0 - bt_n * k0n[:, None] * kTlam[None, :]
            y1 = lam1 - bt_n * k1n[:, None] * kTlam[None, :]
            y2 = lam2 - bt_n * k2n[:, None] * kTlam[None, :]
            y3 = lam3 - bt_n * k3n[:, None] * kTlam[None, :]
            eTY = tl.sum(e0n[:, None] * y0, axis=0) + tl.sum(e1n[:, None] * y1, axis=0) + tl.sum(e2n[:, None] * y2, axis=0) + tl.sum(e3n[:, None] * y3, axis=0)
            z0 = y0 - ct_n * e0n[:, None] * eTY[None, :]
            z1 = y1 - ct_n * e1n[:, None] * eTY[None, :]
            z2 = y2 - ct_n * e2n[:, None] * eTY[None, :]
            z3 = y3 - ct_n * e3n[:, None] * eTY[None, :]
            lam0 = z0 * d0n[:, None]
            lam1 = z1 * d1n[:, None]
            lam2 = z2 * d2n[:, None]
            lam3 = z3 * d3n[:, None]
        # add grad from output
        grad_o = tl.load(grad_out_ptr + b * s0_go + h * s1_go + t * s2_go + (v_col + offs_v) * s3_go, mask=(v_col + offs_v) < V, other=0.0).to(tl.float32)
        q0 = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + offs_k * s3_q, mask=offs_k < K, other=0.0).to(tl.float32)
        q1 = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + (32 + offs_k) * s3_q, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        q2 = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + (64 + offs_k) * s3_q, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        q3 = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + (96 + offs_k) * s3_q, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        lam0 = lam0 + scale * q0[:, None] * grad_o[None, :]
        lam1 = lam1 + scale * q1[:, None] * grad_o[None, :]
        lam2 = lam2 + scale * q2[:, None] * grad_o[None, :]
        lam3 = lam3 + scale * q3[:, None] * grad_o[None, :]
        # load S_cur and S_prev
        cur0 = tl.load(S_tmp_ptr + b * s0_tmp + h * s1_tmp + (t - t_start) * s2_tmp + 0 * s3_tmp + v_col * s4_tmp + offs_k[:, None] * s3_tmp + offs_v[None, :] * s4_tmp, mask=(offs_k[:, None] < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
        cur1 = tl.load(S_tmp_ptr + b * s0_tmp + h * s1_tmp + (t - t_start) * s2_tmp + 32 * s3_tmp + v_col * s4_tmp + offs_k[:, None] * s3_tmp + offs_v[None, :] * s4_tmp, mask=((32 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
        cur2 = tl.load(S_tmp_ptr + b * s0_tmp + h * s1_tmp + (t - t_start) * s2_tmp + 64 * s3_tmp + v_col * s4_tmp + offs_k[:, None] * s3_tmp + offs_v[None, :] * s4_tmp, mask=((64 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
        cur3 = tl.load(S_tmp_ptr + b * s0_tmp + h * s1_tmp + (t - t_start) * s2_tmp + 96 * s3_tmp + v_col * s4_tmp + offs_k[:, None] * s3_tmp + offs_v[None, :] * s4_tmp, mask=((96 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
        if t - t_start == 0:
            prev0 = tl.load(S_init_ptr + b * s0_si + h * s1_si + 0 * s2_si + v_col * s3_si + offs_k[:, None] * s2_si + offs_v[None, :] * s3_si, mask=(offs_k[:, None] < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
            prev1 = tl.load(S_init_ptr + b * s0_si + h * s1_si + 32 * s2_si + v_col * s3_si + offs_k[:, None] * s2_si + offs_v[None, :] * s3_si, mask=((32 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
            prev2 = tl.load(S_init_ptr + b * s0_si + h * s1_si + 64 * s2_si + v_col * s3_si + offs_k[:, None] * s2_si + offs_v[None, :] * s3_si, mask=((64 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
            prev3 = tl.load(S_init_ptr + b * s0_si + h * s1_si + 96 * s2_si + v_col * s3_si + offs_k[:, None] * s2_si + offs_v[None, :] * s3_si, mask=((96 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
        else:
            prev0 = tl.load(S_tmp_ptr + b * s0_tmp + h * s1_tmp + (t - t_start - 1) * s2_tmp + 0 * s3_tmp + v_col * s4_tmp + offs_k[:, None] * s3_tmp + offs_v[None, :] * s4_tmp, mask=(offs_k[:, None] < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
            prev1 = tl.load(S_tmp_ptr + b * s0_tmp + h * s1_tmp + (t - t_start - 1) * s2_tmp + 32 * s3_tmp + v_col * s4_tmp + offs_k[:, None] * s3_tmp + offs_v[None, :] * s4_tmp, mask=((32 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
            prev2 = tl.load(S_tmp_ptr + b * s0_tmp + h * s1_tmp + (t - t_start - 1) * s2_tmp + 64 * s3_tmp + v_col * s4_tmp + offs_k[:, None] * s3_tmp + offs_v[None, :] * s4_tmp, mask=((64 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
            prev3 = tl.load(S_tmp_ptr + b * s0_tmp + h * s1_tmp + (t - t_start - 1) * s2_tmp + 96 * s3_tmp + v_col * s4_tmp + offs_k[:, None] * s3_tmp + offs_v[None, :] * s4_tmp, mask=((96 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
        # params for this t
        bt = tl.load(beta_ptr + b * s0_b + h * s1_b + t * s2_b).to(tl.float32)
        gt = tl.load(gamma_ptr + b * s0_gm + h * s1_gm + t * s2_gm).to(tl.float32)
        ct = gt * (1.0 - bt)
        e0 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + offs_k * s3_e, mask=offs_k < K, other=0.0).to(tl.float32)
        k0 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + offs_k * s3_k, mask=offs_k < K, other=0.0).to(tl.float32)
        g0 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + offs_k * s3_g, mask=offs_k < K, other=0.0).to(tl.float32)
        d0 = tl.exp(g0)
        e1 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + (32 + offs_k) * s3_e, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        k1 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + (32 + offs_k) * s3_k, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        g1 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + (32 + offs_k) * s3_g, mask=(32 + offs_k) < K, other=0.0).to(tl.float32)
        d1 = tl.exp(g1)
        e2 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + (64 + offs_k) * s3_e, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        k2 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + (64 + offs_k) * s3_k, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        g2 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + (64 + offs_k) * s3_g, mask=(64 + offs_k) < K, other=0.0).to(tl.float32)
        d2 = tl.exp(g2)
        e3 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + (96 + offs_k) * s3_e, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        k3 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + (96 + offs_k) * s3_k, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        g3 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + (96 + offs_k) * s3_g, mask=(96 + offs_k) < K, other=0.0).to(tl.float32)
        d3 = tl.exp(g3)
        # vectors
        U0 = prev0 * d0[:, None]
        U1 = prev1 * d1[:, None]
        U2 = prev2 * d2[:, None]
        U3 = prev3 * d3[:, None]
        eTU = tl.sum(e0[:, None] * U0, axis=0) + tl.sum(e1[:, None] * U1, axis=0) + tl.sum(e2[:, None] * U2, axis=0) + tl.sum(e3[:, None] * U3, axis=0)
        kTU = tl.sum(k0[:, None] * U0, axis=0) + tl.sum(k1[:, None] * U1, axis=0) + tl.sum(k2[:, None] * U2, axis=0) + tl.sum(k3[:, None] * U3, axis=0)
        Bdot = tl.sum(k0 * e0) + tl.sum(k1 * e1) + tl.sum(k2 * e2) + tl.sum(k3 * e3)
        kTW = kTU - ct * Bdot * eTU
        kTlam = tl.sum(k0[:, None] * lam0, axis=0) + tl.sum(k1[:, None] * lam1, axis=0) + tl.sum(k2[:, None] * lam2, axis=0) + tl.sum(k3[:, None] * lam3, axis=0)
        eTlam = tl.sum(e0[:, None] * lam0, axis=0) + tl.sum(e1[:, None] * lam1, axis=0) + tl.sum(e2[:, None] * lam2, axis=0) + tl.sum(e3[:, None] * lam3, axis=0)
        mu0 = lam0 - bt * k0[:, None] * kTlam[None, :]
        mu1 = lam1 - bt * k1[:, None] * kTlam[None, :]
        mu2 = lam2 - bt * k2[:, None] * kTlam[None, :]
        mu3 = lam3 - bt * k3[:, None] * kTlam[None, :]
        eTmu = tl.sum(e0[:, None] * mu0, axis=0) + tl.sum(e1[:, None] * mu1, axis=0) + tl.sum(e2[:, None] * mu2, axis=0) + tl.sum(e3[:, None] * mu3, axis=0)
        W0 = U0 - ct * e0[:, None] * eTU[None, :]
        W1 = U1 - ct * e1[:, None] * eTU[None, :]
        W2 = U2 - ct * e2[:, None] * eTU[None, :]
        W3 = U3 - ct * e3[:, None] * eTU[None, :]
        v_vals = tl.load(v_ptr + b * s0_v + h * s1_v + t * s2_v + (v_col + offs_v) * s3_v, mask=(v_col + offs_v) < V, other=0.0).to(tl.float32)
        # gv
        gv_vals = bt * kTlam
        tl.store(gv_ptr + b * s0_gv + h * s1_gv + t * s2_gv + (v_col + offs_v) * s3_gv, gv_vals, mask=(v_col + offs_v) < V)
        # stage grads for K dims
        # gq
        gq0 = tl.sum(cur0 * grad_o[None, :], axis=1) * scale
        gq1 = tl.sum(cur1 * grad_o[None, :], axis=1) * scale
        gq2 = tl.sum(cur2 * grad_o[None, :], axis=1) * scale
        gq3 = tl.sum(cur3 * grad_o[None, :], axis=1) * scale
        tl.store(gq_stage_ptr + b * s0_gq + h * s1_gq + iv * s2_gq + (t - t_start) * s3_gq + 0 * s4_gq + offs_k * s4_gq, gq0, mask=offs_k < K)
        tl.store(gq_stage_ptr + b * s0_gq + h * s1_gq + iv * s2_gq + (t - t_start) * s3_gq + 32 * s4_gq + offs_k * s4_gq, gq1, mask=(32 + offs_k) < K)
        tl.store(gq_stage_ptr + b * s0_gq + h * s1_gq + iv * s2_gq + (t - t_start) * s3_gq + 64 * s4_gq + offs_k * s4_gq, gq2, mask=(64 + offs_k) < K)
        tl.store(gq_stage_ptr + b * s0_gq + h * s1_gq + iv * s2_gq + (t - t_start) * s3_gq + 96 * s4_gq + offs_k * s4_gq, gq3, mask=(96 + offs_k) < K)
        # gk
        gk0 = bt * tl.sum(lam0 * (v_vals - kTW)[None, :] - W0 * kTlam[None, :], axis=1)
        gk1 = bt * tl.sum(lam1 * (v_vals - kTW)[None, :] - W1 * kTlam[None, :], axis=1)
        gk2 = bt * tl.sum(lam2 * (v_vals - kTW)[None, :] - W2 * kTlam[None, :], axis=1)
        gk3 = bt * tl.sum(lam3 * (v_vals - kTW)[None, :] - W3 * kTlam[None, :], axis=1)
        tl.store(gk_stage_ptr + b * s0_gk + h * s1_gk + iv * s2_gk + (t - t_start) * s3_gk + 0 * s4_gk + offs_k * s4_gk, gk0, mask=offs_k < K)
        tl.store(gk_stage_ptr + b * s0_gk + h * s1_gk + iv * s2_gk + (t - t_start) * s3_gk + 32 * s4_gk + offs_k * s4_gk, gk1, mask=(32 + offs_k) < K)
        tl.store(gk_stage_ptr + b * s0_gk + h * s1_gk + iv * s2_gk + (t - t_start) * s3_gk + 64 * s4_gk + offs_k * s4_gk, gk2, mask=(64 + offs_k) < K)
        tl.store(gk_stage_ptr + b * s0_gk + h * s1_gk + iv * s2_gk + (t - t_start) * s3_gk + 96 * s4_gk + offs_k * s4_gk, gk3, mask=(96 + offs_k) < K)
        # ge
        ge0 = -ct * tl.sum(mu0 * eTU[None, :] + U0 * eTmu[None, :], axis=1)
        ge1 = -ct * tl.sum(mu1 * eTU[None, :] + U1 * eTmu[None, :], axis=1)
        ge2 = -ct * tl.sum(mu2 * eTU[None, :] + U2 * eTmu[None, :], axis=1)
        ge3 = -ct * tl.sum(mu3 * eTU[None, :] + U3 * eTmu[None, :], axis=1)
        tl.store(ge_stage_ptr + b * s0_ge + h * s1_ge + iv * s2_ge + (t - t_start) * s3_ge + 0 * s4_ge + offs_k * s4_ge, ge0, mask=offs_k < K)
        tl.store(ge_stage_ptr + b * s0_ge + h * s1_ge + iv * s2_ge + (t - t_start) * s3_ge + 32 * s4_ge + offs_k * s4_ge, ge1, mask=(32 + offs_k) < K)
        tl.store(ge_stage_ptr + b * s0_ge + h * s1_ge + iv * s2_ge + (t - t_start) * s3_ge + 64 * s4_ge + offs_k * s4_ge, ge2, mask=(64 + offs_k) < K)
        tl.store(ge_stage_ptr + b * s0_ge + h * s1_ge + iv * s2_ge + (t - t_start) * s3_ge + 96 * s4_ge + offs_k * s4_ge, ge3, mask=(96 + offs_k) < K)
        # gg
        nu0 = mu0 - ct * e0[:, None] * eTmu[None, :]
        nu1 = mu1 - ct * e1[:, None] * eTmu[None, :]
        nu2 = mu2 - ct * e2[:, None] * eTmu[None, :]
        nu3 = mu3 - ct * e3[:, None] * eTmu[None, :]
        gg0 = d0 * tl.sum(nu0 * prev0, axis=1)
        gg1 = d1 * tl.sum(nu1 * prev1, axis=1)
        gg2 = d2 * tl.sum(nu2 * prev2, axis=1)
        gg3 = d3 * tl.sum(nu3 * prev3, axis=1)
        tl.store(gg_stage_ptr + b * s0_gg + h * s1_gg + iv * s2_gg + (t - t_start) * s3_gg + 0 * s4_gg + offs_k * s4_gg, gg0, mask=offs_k < K)
        tl.store(gg_stage_ptr + b * s0_gg + h * s1_gg + iv * s2_gg + (t - t_start) * s3_gg + 32 * s4_gg + offs_k * s4_gg, gg1, mask=(32 + offs_k) < K)
        tl.store(gg_stage_ptr + b * s0_gg + h * s1_gg + iv * s2_gg + (t - t_start) * s3_gg + 64 * s4_gg + offs_k * s4_gg, gg2, mask=(64 + offs_k) < K)
        tl.store(gg_stage_ptr + b * s0_gg + h * s1_gg + iv * s2_gg + (t - t_start) * s3_gg + 96 * s4_gg + offs_k * s4_gg, gg3, mask=(96 + offs_k) < K)
        # scalar stages
        gb_part = gt * tl.sum(eTU * eTlam) + tl.sum(kTlam * (v_vals - kTW)) - bt * gt * Bdot * tl.sum(eTU * kTlam)
        ggm_part = -(1.0 - bt) * tl.sum(eTmu * eTU)
        tl.store(gb_stage_ptr + b * s0_gb + h * s1_gb + iv * s2_gb + (t - t_start) * s3_gb, gb_part)
        tl.store(ggm_stage_ptr + b * s0_ggm + h * s1_ggm + iv * s2_ggm + (t - t_start) * s3_ggm, ggm_part)
    tl.store(lambda_ptr + b * s0_lam + h * s1_lam + 0 * s2_lam + v_col * s3_lam + offs_k[:, None] * s2_lam + offs_v[None, :] * s3_lam, lam0, mask=(offs_k[:, None] < K) & ((v_col + offs_v)[None, :] < V))
    tl.store(lambda_ptr + b * s0_lam + h * s1_lam + 32 * s2_lam + v_col * s3_lam + offs_k[:, None] * s2_lam + offs_v[None, :] * s3_lam, lam1, mask=((32 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V))
    tl.store(lambda_ptr + b * s0_lam + h * s1_lam + 64 * s2_lam + v_col * s3_lam + offs_k[:, None] * s2_lam + offs_v[None, :] * s3_lam, lam2, mask=((64 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V))
    tl.store(lambda_ptr + b * s0_lam + h * s1_lam + 96 * s2_lam + v_col * s3_lam + offs_k[:, None] * s2_lam + offs_v[None, :] * s3_lam, lam3, mask=((96 + offs_k[:, None]) < K) & ((v_col + offs_v)[None, :] < V))

def _reference_pytorch_fallback(q,k,v,e,g,beta,gamma,scale,grad_out=None):
    # used for small non-128 shapes fallback (kept simple)
    B,H,T,K = q.shape
    V=v.shape[-1]
    qf=q.float(); kf=k.float(); vf=v.float(); ef=e.float(); gf=g.float(); bf=beta.float(); gf2=gamma.float()
    if grad_out is None:
        # forward
        S=torch.zeros(B,H,K,V,device=q.device,dtype=torch.float32)
        outs=[]
        for t in range(T):
            kt=kf[:,:,t]; vt=vf[:,:,t]; et=ef[:,:,t]; bt=bf[:,:,t]; gam=gf2[:,:,t]
            ct=gam*(1-bt)
            a=torch.exp(gf[:,:,t])
            S = S * a[...,None]
            S = S - ct[...,None,None]*et[...,None]*( (S*et[...,None]).sum(-2)[...,None,:])
            pred=(S*kt[...,None]).sum(-2)
            S = S + (bt[...,None]*(vt-pred))[...,None,:]*kt[...,None]
            outs.append((S*(qf[:,:,t]*scale)[...,None]).sum(-2))
        return torch.stack(outs,dim=2)
    else:
        qf.requires_grad_(True); kf.requires_grad_(True); vf.requires_grad_(True); ef.requires_grad_(True); gf.requires_grad_(True); bf.requires_grad_(True); gf2.requires_grad_(True)
        out=_reference_pytorch_fallback(qf,kf,vf,ef,gf,bf,gf2,scale,None)
        out.backward(grad_out)
        return qf.grad,kf.grad,vf.grad,ef.grad,gf.grad,bf.grad,gf2.grad

class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,e,g,beta,gamma,scale):
        B,H,T,K=q.shape; V=v.shape[-1]
        # save for backward
        ctx.save_for_backward(q,k,v,e,g,beta,gamma)
        ctx.scale=scale; ctx.B=B; ctx.H=H; ctx.T=T; ctx.K=K; ctx.V=V
        if K==128 and V==128:
            # chunked forward with bounds
            NB=(T+CHUNK-1)//CHUNK
            out=torch.empty(B,H,T,V,device=q.device,dtype=torch.float32)
            bounds=torch.zeros(B,H,NB+1,K,V,device=q.device,dtype=torch.float32)
            # bounds 0 already zero
            NV_b=(V+BLOCK_V-1)//BLOCK_V
            for idx in range(NB):
                t0=idx*CHUNK
                Tc=min(CHUNK,T-t0)
                S_init=bounds[:,:,idx]  # [B,H,K,V]
                S_final=bounds[:,:,idx+1]
                # slice out for this chunk is just whole out, kernel uses t_start
                grid=(B*H*NV_b,)
                _fwd_chunk_NK4[grid](
                    q,k,v,e,g,beta,gamma,S_init,S_final,out,
                    q.stride(0),q.stride(1),q.stride(2),q.stride(3),
                    k.stride(0),k.stride(1),k.stride(2),k.stride(3),
                    v.stride(0),v.stride(1),v.stride(2),v.stride(3),
                    e.stride(0),e.stride(1),e.stride(2),e.stride(3),
                    g.stride(0),g.stride(1),g.stride(2),g.stride(3),
                    beta.stride(0),beta.stride(1),beta.stride(2),
                    gamma.stride(0),gamma.stride(1),gamma.stride(2),
                    S_init.stride(0),S_init.stride(1),S_init.stride(2),S_init.stride(3),
                    S_final.stride(0),S_final.stride(1),S_final.stride(2),S_final.stride(3),
                    out.stride(0),out.stride(1),out.stride(2),out.stride(3),
                    B,H,T,K,V,NV_b,BLOCK_K,BLOCK_V,float(scale),t0,Tc,
                    num_warps=4,num_stages=2)
            ctx.bounds=bounds
            return out
        else:
            # fallback torch forward (still need bounds for uniform API, but we can store None and use torch backward)
            out=_reference_pytorch_fallback(q,k,v,e,g,beta,gamma,scale,None)
            ctx.bounds=None
            ctx.fallback=True
            return out

    @staticmethod
    def backward(ctx,grad_out):
        q,k,v,e,g,beta,gamma=ctx.saved_tensors
        scale=ctx.scale; B=ctx.B; H=ctx.H; T=ctx.T; K=ctx.K; V=ctx.V
        if getattr(ctx,'fallback',False):
            gq,gk,gv,ge,gg,gb,ggm=_reference_pytorch_fallback(q,k,v,e,g,beta,gamma,scale,grad_out)
            # cast to input dtypes
            return gq.to(q.dtype) if gq is not None else None, gk.to(k.dtype) if gk is not None else None, gv.to(v.dtype) if gv is not None else None, ge.to(e.dtype) if ge is not None else None, gg.to(g.dtype) if gg is not None else None, gb.to(beta.dtype) if gb is not None else None, ggm.to(gamma.dtype) if ggm is not None else None, None
        # fast path NK4
        NB=(T+CHUNK-1)//CHUNK
        bounds=ctx.bounds
        NV_b=(V+BLOCK_V-1)//BLOCK_V
        # grad buffers
        grad_q=torch.zeros(B,H,T,K,device=q.device,dtype=torch.float32)
        grad_k=torch.zeros(B,H,T,K,device=q.device,dtype=torch.float32)
        grad_e=torch.zeros(B,H,T,K,device=q.device,dtype=torch.float32)
        grad_g=torch.zeros(B,H,T,K,device=q.device,dtype=torch.float32)
        grad_v=torch.zeros(B,H,T,V,device=q.device,dtype=torch.float32)
        grad_beta=torch.zeros(B,H,T,device=q.device,dtype=torch.float32)
        grad_gamma=torch.zeros(B,H,T,device=q.device,dtype=torch.float32)
        lam=torch.zeros(B,H,K,V,device=q.device,dtype=torch.float32)
        grad_out_c=grad_out.contiguous().float()
        # tmp buffers
        S_tmp=torch.empty(B,H,CHUNK,K,V,device=q.device,dtype=torch.float32)
        # stage buffers per chunk
        stage_q=torch.empty(B,H,NV_b,CHUNK,K,device=q.device,dtype=torch.float32)
        stage_k=torch.empty(B,H,NV_b,CHUNK,K,device=q.device,dtype=torch.float32)
        stage_e=torch.empty(B,H,NV_b,CHUNK,K,device=q.device,dtype=torch.float32)
        stage_g=torch.empty(B,H,NV_b,CHUNK,K,device=q.device,dtype=torch.float32)
        stage_b=torch.empty(B,H,NV_b,CHUNK,device=q.device,dtype=torch.float32)
        stage_gm=torch.empty(B,H,NV_b,CHUNK,device=q.device,dtype=torch.float32)
        for idx in range(NB-1,-1,-1):
            t0=idx*CHUNK
            Tc=min(CHUNK,T-t0)
            S_init=bounds[:,:,idx]
            grid=(B*H*NV_b,)
            _bwd_chunk_NK4[grid](
                q,k,v,e,g,beta,gamma,grad_out_c,
                S_init,S_tmp,lam,
                stage_q,stage_k,stage_e,stage_g,stage_b,stage_gm,grad_v,
                q.stride(0),q.stride(1),q.stride(2),q.stride(3),
                k.stride(0),k.stride(1),k.stride(2),k.stride(3),
                v.stride(0),v.stride(1),v.stride(2),v.stride(3),
                e.stride(0),e.stride(1),e.stride(2),e.stride(3),
                g.stride(0),g.stride(1),g.stride(2),g.stride(3),
                beta.stride(0),beta.stride(1),beta.stride(2),
                gamma.stride(0),gamma.stride(1),gamma.stride(2),
                grad_out_c.stride(0),grad_out_c.stride(1),grad_out_c.stride(2),grad_out_c.stride(3),
                S_init.stride(0),S_init.stride(1),S_init.stride(2),S_init.stride(3),
                S_tmp.stride(0),S_tmp.stride(1),S_tmp.stride(2),S_tmp.stride(3),S_tmp.stride(4),
                lam.stride(0),lam.stride(1),lam.stride(2),lam.stride(3),
                stage_q.stride(0),stage_q.stride(1),stage_q.stride(2),stage_q.stride(3),stage_q.stride(4),
                stage_k.stride(0),stage_k.stride(1),stage_k.stride(2),stage_k.stride(3),stage_k.stride(4),
                stage_e.stride(0),stage_e.stride(1),stage_e.stride(2),stage_e.stride(3),stage_e.stride(4),
                stage_g.stride(0),stage_g.stride(1),stage_g.stride(2),stage_g.stride(3),stage_g.stride(4),
                stage_b.stride(0),stage_b.stride(1),stage_b.stride(2),stage_b.stride(3),
                stage_gm.stride(0),stage_gm.stride(1),stage_gm.stride(2),stage_gm.stride(3),
                grad_v.stride(0),grad_v.stride(1),grad_v.stride(2),grad_v.stride(3),
                B,H,T,K,V,NV_b,BLOCK_K,BLOCK_V,float(scale),t0,Tc,
                num_warps=4,num_stages=2)
            # reduce stages for this chunk
            # stage shape [B,H,NV,CHUNK,K] -> sum over NV
            # only first Tc valid
            grad_q[:,:,t0:t0+Tc] += stage_q[:,:,:NV_b,:Tc,:].sum(dim=2)
            grad_k[:,:,t0:t0+Tc] += stage_k[:,:,:NV_b,:Tc,:].sum(dim=2)
            grad_e[:,:,t0:t0+Tc] += stage_e[:,:,:NV_b,:Tc,:].sum(dim=2)
            grad_g[:,:,t0:t0+Tc] += stage_g[:,:,:NV_b,:Tc,:].sum(dim=2)
            grad_beta[:,:,t0:t0+Tc] += stage_b[:,:,:NV_b,:Tc].sum(dim=2)
            grad_gamma[:,:,t0:t0+Tc] += stage_gm[:,:,:NV_b,:Tc].sum(dim=2)
        # cast to input dtypes
        return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype), grad_e.to(e.dtype), grad_g.to(g.dtype), grad_beta.to(beta.dtype), grad_gamma.to(gamma.dtype), None

def kernel_function(q,k,v,e,g,beta,gamma,scale):
    # fast inference path when not in autograd context, else training path
    # we always go through autograd Function so backward is available
    return _Func.apply(q,k,v,e,g,beta,gamma,scale)

