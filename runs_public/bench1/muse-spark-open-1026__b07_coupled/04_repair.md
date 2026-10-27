```python
import torch
import triton
import triton.language as tl

BLOCK_K = 32
BLOCK_V = 32

# HBM fallback for very large K (K>256)
@triton.jit
def _fwd_hbm(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, s_ptr, out_ptr,
    s0_q, s1_q, s2_q, s3_q,
    s0_k, s1_k, s2_k, s3_k,
    s0_v, s1_v, s2_v, s3_v,
    s0_e, s1_e, s2_e, s3_e,
    s0_g, s1_g, s2_g, s3_g,
    s0_b, s1_b, s2_b,
    s0_gm, s1_gm, s2_gm,
    s0_s, s1_s, s2_s, s3_s,
    s0_o, s1_o, s2_o, s3_o,
    B, H, T, K, V,
    NV: tl.constexpr, NK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, scale,
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


@triton.jit
def _fwd_col_NK4(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    s0_q, s1_q, s2_q, s3_q,
    s0_k, s1_k, s2_k, s3_k,
    s0_v, s1_v, s2_v, s3_v,
    s0_e, s1_e, s2_e, s3_e,
    s0_g, s1_g, s2_g, s3_g,
    s0_b, s1_b, s2_b,
    s0_gm, s1_gm, s2_gm,
    s0_o, s1_o, s2_o, s3_o,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NV: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, scale,
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

@triton.jit
def _fwd_col_NK3(
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
        Bdot = tl.sum(k0 * e0) + tl.sum(k1 * e1) + tl.sum(k2 * e2)
        eTS = tl.sum((e0 * d0)[:, None] * s0, axis=0) + tl.sum((e1 * d1)[:, None] * s1, axis=0) + tl.sum((e2 * d2)[:, None] * s2, axis=0)
        A = tl.sum((k0 * d0)[:, None] * s0, axis=0) + tl.sum((k1 * d1)[:, None] * s1, axis=0) + tl.sum((k2 * d2)[:, None] * s2, axis=0)
        pred = A - ct * Bdot * eTS
        delta = v_vals - pred
        s0 = s0 * d0[:, None] - ct * e0[:, None] * eTS[None, :] + bt * k0[:, None] * delta[None, :]
        s1 = s1 * d1[:, None] - ct * e1[:, None] * eTS[None, :] + bt * k1[:, None] * delta[None, :]
        s2 = s2 * d2[:, None] - ct * e2[:, None] * eTS[None, :] + bt * k2[:, None] * delta[None, :]
        out_acc = tl.sum(q0[:, None] * s0, axis=0) + tl.sum(q1[:, None] * s1, axis=0) + tl.sum(q2[:, None] * s2, axis=0)
        out_acc = out_acc * scale
        tl.store(out_ptr + b * s0_o + h * s1_o + t * s2_o + (v_col + offs_v) * s3_o, out_acc, mask=(v_col + offs_v) < V)

@triton.jit
def _fwd_col_NK2(
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
        Bdot = tl.sum(k0 * e0) + tl.sum(k1 * e1)
        eTS = tl.sum((e0 * d0)[:, None] * s0, axis=0) + tl.sum((e1 * d1)[:, None] * s1, axis=0)
        A = tl.sum((k0 * d0)[:, None] * s0, axis=0) + tl.sum((k1 * d1)[:, None] * s1, axis=0)
        pred = A - ct * Bdot * eTS
        delta = v_vals - pred
        s0 = s0 * d0[:, None] - ct * e0[:, None] * eTS[None, :] + bt * k0[:, None] * delta[None, :]
        s1 = s1 * d1[:, None] - ct * e1[:, None] * eTS[None, :] + bt * k1[:, None] * delta[None, :]
        out_acc = tl.sum(q0[:, None] * s0, axis=0) + tl.sum(q1[:, None] * s1, axis=0)
        out_acc = out_acc * scale
        tl.store(out_ptr + b * s0_o + h * s1_o + t * s2_o + (v_col + offs_v) * s3_o, out_acc, mask=(v_col + offs_v) < V)

@triton.jit
def _fwd_col_NK5(
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
    s4 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
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
        e4 = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + (128 + offs_k) * s3_e, mask=(128 + offs_k) < K, other=0.0).to(tl.float32)
        k4 = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + (128 + offs_k) * s3_k, mask=(128 + offs_k) < K, other=0.0).to(tl.float32)
        g4 = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + (128 + offs_k) * s3_g, mask=(128 + offs_k) < K, other=0.0).to(tl.float32)
        d4 = tl.exp(g4)
        q4 = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + (128 + offs_k) * s3_q, mask=(128 + offs_k) < K, other=0.0).to(tl.float32)
        Bdot = tl.sum(k0 * e0) + tl.sum(k1 * e1) + tl.sum(k2 * e2) + tl.sum(k3 * e3) + tl.sum(k4 * e4)
        eTS = tl.sum((e0 * d0)[:, None] * s0, axis=0) + tl.sum((e1 * d1)[:, None] * s1, axis=0) + tl.sum((e2 * d2)[:, None] * s2, axis=0) + tl.sum((e3 * d3)[:, None] * s3, axis=0) + tl.sum((e4 * d4)[:, None] * s4, axis=0)
        A = tl.sum((k0 * d0)[:, None] * s0, axis=0) + tl.sum((k1 * d1)[:, None] * s1, axis=0) + tl.sum((k2 * d2)[:, None] * s2, axis=0) + tl.sum((k3 * d3)[:, None] * s3, axis=0) + tl.sum((k4 * d4)[:, None] * s4, axis=0)
        pred = A - ct * Bdot * eTS
        delta = v_vals - pred
        s0 = s0 * d0[:, None] - ct * e0[:, None] * eTS[None, :] + bt * k0[:, None] * delta[None, :]
        s1 = s1 * d1[:, None] - ct * e1[:, None] * eTS[None, :] + bt * k1[:, None] * delta[None, :]
        s2 = s2 * d2[:, None] - ct * e2[:, None] * eTS[None, :] + bt * k2[:, None] * delta[None, :]
        s3 = s3 * d3[:, None] - ct * e3[:, None] * eTS[None, :] + bt * k3[:, None] * delta[None, :]
        s4 = s4 * d4[:, None] - ct * e4[:, None] * eTS[None, :] + bt * k4[:, None] * delta[None, :]
        out_acc = tl.sum(q0[:, None] * s0, axis=0) + tl.sum(q1[:, None] * s1, axis=0) + tl.sum(q2[:, None] * s2, axis=0) + tl.sum(q3[:, None] * s3, axis=0) + tl.sum(q4[:, None] * s4, axis=0)
        out_acc = out_acc * scale
        tl.store(out_ptr + b * s0_o + h * s1_o + t * s2_o + (v_col + offs_v) * s3_o, out_acc, mask=(v_col + offs_v) < V)

@triton.jit
def _fwd_col_NK1(
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
        Bdot = tl.sum(k0 * e0)
        eTS = tl.sum((e0 * d0)[:, None] * s0, axis=0)
        A = tl.sum((k0 * d0)[:, None] * s0, axis=0)
        pred = A - ct * Bdot * eTS
        delta = v_vals - pred
        s0 = s0 * d0[:, None] - ct * e0[:, None] * eTS[None, :] + bt * k0[:, None] * delta[None, :]
        out_acc = tl.sum(q0[:, None] * s0, axis=0)
        out_acc = out_acc * scale
        tl.store(out_ptr + b * s0_o + h * s1_o + t * s2_o + (v_col + offs_v) * s3_o, out_acc, mask=(v_col + offs_v) < V)

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)
    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    NK = (K + BLOCK_K - 1) // BLOCK_K
    NV = (V + BLOCK_V - 1) // BLOCK_V
    grid = (B * H * NV,)
    common = (q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3), e.stride(0), e.stride(1), e.stride(2), e.stride(3), g.stride(0), g.stride(1), g.stride(2), g.stride(3), beta.stride(0), beta.stride(1), beta.stride(2), gamma.stride(0), gamma.stride(1), gamma.stride(2), out.stride(0), out.stride(1), out.stride(2), out.stride(3), B, H, T, K, V, NV, BLOCK_K, BLOCK_V, scale_f)
    if NK == 4:
        _fwd_col_NK4[grid](q, k, v, e, g, beta, gamma, out, *common, num_warps=4, num_stages=2)
    elif NK == 3:
        _fwd_col_NK3[grid](q, k, v, e, g, beta, gamma, out, *common, num_warps=4, num_stages=2)
    elif NK == 2:
        _fwd_col_NK2[grid](q, k, v, e, g, beta, gamma, out, *common, num_warps=4, num_stages=2)
    elif NK == 5:
        _fwd_col_NK5[grid](q, k, v, e, g, beta, gamma, out, *common, num_warps=4, num_stages=2)
    elif NK == 1:
        _fwd_col_NK1[grid](q, k, v, e, g, beta, gamma, out, *common, num_warps=4, num_stages=2)
    else:
        # fallback HBM for NK>5 (e.g. 6+)
        scratch = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32)
        NKc = NK
        _fwd_hbm[(B * H * NV,)](q, k, v, e, g, beta, gamma, scratch, out, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3), e.stride(0), e.stride(1), e.stride(2), e.stride(3), g.stride(0), g.stride(1), g.stride(2), g.stride(3), beta.stride(0), beta.stride(1), beta.stride(2), gamma.stride(0), gamma.stride(1), gamma.stride(2), scratch.stride(0), scratch.stride(1), scratch.stride(2), scratch.stride(3), out.stride(0), out.stride(1), out.stride(2), out.stride(3), B, H, T, K, V, NV, NKc, BLOCK_K, BLOCK_V, scale_f, num_warps=4, num_stages=2)
    return out
```