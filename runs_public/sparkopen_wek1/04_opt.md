```python
import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# generic small-K/V (register tiled 32x32) – fallback for K,V <=128
# ------------------------------------------------------------------
@triton.jit
def fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr,
    stride_q_b, stride_q_h, stride_q_t,
    stride_k_b, stride_k_h, stride_k_t,
    stride_v_b, stride_v_h, stride_v_t,
    stride_e_b, stride_e_h, stride_e_t,
    stride_g_b, stride_g_h, stride_g_t,
    stride_beta_b, stride_beta_h, stride_beta_t,
    stride_gamma_b, stride_gamma_h, stride_gamma_t,
    stride_o_b, stride_o_h, stride_o_t,
    H, T, K, V, scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % H
    b = pid // H
    off_q_bh = b * stride_q_b + h * stride_q_h
    off_k_bh = b * stride_k_b + h * stride_k_h
    off_v_bh = b * stride_v_b + h * stride_v_h
    off_e_bh = b * stride_e_b + h * stride_e_h
    off_g_bh = b * stride_g_b + h * stride_g_h
    off_beta_bh = b * stride_beta_b + h * stride_beta_h
    off_gamma_bh = b * stride_gamma_b + h * stride_gamma_h
    off_o_bh = b * stride_o_b + h * stride_o_h
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    S00 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S01 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S02 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S03 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S10 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S11 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S12 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S13 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S20 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S21 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S22 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S23 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S30 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S31 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S32 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    S33 = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    for t in range(T):
        k0 = tl.load(k_ptr + off_k_bh + t * stride_k_t + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
        k1 = tl.load(k_ptr + off_k_bh + t * stride_k_t + BLOCK_K + offs_k, mask=(BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
        k2 = tl.load(k_ptr + off_k_bh + t * stride_k_t + 2 * BLOCK_K + offs_k, mask=(2 * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
        k3 = tl.load(k_ptr + off_k_bh + t * stride_k_t + 3 * BLOCK_K + offs_k, mask=(3 * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
        q0 = tl.load(q_ptr + off_q_bh + t * stride_q_t + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
        q1 = tl.load(q_ptr + off_q_bh + t * stride_q_t + BLOCK_K + offs_k, mask=(BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
        q2 = tl.load(q_ptr + off_q_bh + t * stride_q_t + 2 * BLOCK_K + offs_k, mask=(2 * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
        q3 = tl.load(q_ptr + off_q_bh + t * stride_q_t + 3 * BLOCK_K + offs_k, mask=(3 * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
        e0 = tl.load(e_ptr + off_e_bh + t * stride_e_t + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
        e1 = tl.load(e_ptr + off_e_bh + t * stride_e_t + BLOCK_K + offs_k, mask=(BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
        e2 = tl.load(e_ptr + off_e_bh + t * stride_e_t + 2 * BLOCK_K + offs_k, mask=(2 * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
        e3 = tl.load(e_ptr + off_e_bh + t * stride_e_t + 3 * BLOCK_K + offs_k, mask=(3 * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
        g0 = tl.load(g_ptr + off_g_bh + t * stride_g_t + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
        g1 = tl.load(g_ptr + off_g_bh + t * stride_g_t + BLOCK_K + offs_k, mask=(BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
        g2 = tl.load(g_ptr + off_g_bh + t * stride_g_t + 2 * BLOCK_K + offs_k, mask=(2 * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
        g3 = tl.load(g_ptr + off_g_bh + t * stride_g_t + 3 * BLOCK_K + offs_k, mask=(3 * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + off_beta_bh + t * stride_beta_t).to(tl.float32)
        gamma_t = tl.load(gamma_ptr + off_gamma_bh + t * stride_gamma_t).to(tl.float32)
        eg0 = tl.exp(g0)
        eg1 = tl.exp(g1)
        eg2 = tl.exp(g2)
        eg3 = tl.exp(g3)
        S00 = S00 * eg0[:, None]
        S01 = S01 * eg0[:, None]
        S02 = S02 * eg0[:, None]
        S03 = S03 * eg0[:, None]
        S10 = S10 * eg1[:, None]
        S11 = S11 * eg1[:, None]
        S12 = S12 * eg1[:, None]
        S13 = S13 * eg1[:, None]
        S20 = S20 * eg2[:, None]
        S21 = S21 * eg2[:, None]
        S22 = S22 * eg2[:, None]
        S23 = S23 * eg2[:, None]
        S30 = S30 * eg3[:, None]
        S31 = S31 * eg3[:, None]
        S32 = S32 * eg3[:, None]
        S33 = S33 * eg3[:, None]
        pred0 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        pred1 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        pred2 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        pred3 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        pred0 += tl.sum(S00 * k0[:, None], axis=0)
        pred0 += tl.sum(S10 * k1[:, None], axis=0)
        pred0 += tl.sum(S20 * k2[:, None], axis=0)
        pred0 += tl.sum(S30 * k3[:, None], axis=0)
        pred1 += tl.sum(S01 * k0[:, None], axis=0)
        pred1 += tl.sum(S11 * k1[:, None], axis=0)
        pred1 += tl.sum(S21 * k2[:, None], axis=0)
        pred1 += tl.sum(S31 * k3[:, None], axis=0)
        pred2 += tl.sum(S02 * k0[:, None], axis=0)
        pred2 += tl.sum(S12 * k1[:, None], axis=0)
        pred2 += tl.sum(S22 * k2[:, None], axis=0)
        pred2 += tl.sum(S32 * k3[:, None], axis=0)
        pred3 += tl.sum(S03 * k0[:, None], axis=0)
        pred3 += tl.sum(S13 * k1[:, None], axis=0)
        pred3 += tl.sum(S23 * k2[:, None], axis=0)
        pred3 += tl.sum(S33 * k3[:, None], axis=0)
        v0 = tl.load(v_ptr + off_v_bh + t * stride_v_t + offs_v, mask=offs_v < V, other=0.0).to(tl.float32)
        v1 = tl.load(v_ptr + off_v_bh + t * stride_v_t + BLOCK_V + offs_v, mask=(BLOCK_V + offs_v) < V, other=0.0).to(tl.float32)
        v2 = tl.load(v_ptr + off_v_bh + t * stride_v_t + 2 * BLOCK_V + offs_v, mask=(2 * BLOCK_V + offs_v) < V, other=0.0).to(tl.float32)
        v3 = tl.load(v_ptr + off_v_bh + t * stride_v_t + 3 * BLOCK_V + offs_v, mask=(3 * BLOCK_V + offs_v) < V, other=0.0).to(tl.float32)
        delta0 = beta_t * (v0 - pred0)
        delta1 = beta_t * (v1 - pred1)
        delta2 = beta_t * (v2 - pred2)
        delta3 = beta_t * (v3 - pred3)
        S00 += k0[:, None] * delta0[None, :]
        S01 += k0[:, None] * delta1[None, :]
        S02 += k0[:, None] * delta2[None, :]
        S03 += k0[:, None] * delta3[None, :]
        S10 += k1[:, None] * delta0[None, :]
        S11 += k1[:, None] * delta1[None, :]
        S12 += k1[:, None] * delta2[None, :]
        S13 += k1[:, None] * delta3[None, :]
        S20 += k2[:, None] * delta0[None, :]
        S21 += k2[:, None] * delta1[None, :]
        S22 += k2[:, None] * delta2[None, :]
        S23 += k2[:, None] * delta3[None, :]
        S30 += k3[:, None] * delta0[None, :]
        S31 += k3[:, None] * delta1[None, :]
        S32 += k3[:, None] * delta2[None, :]
        S33 += k3[:, None] * delta3[None, :]
        tmp0 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        tmp1 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        tmp2 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        tmp3 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        tmp0 += tl.sum(S00 * e0[:, None], axis=0)
        tmp0 += tl.sum(S10 * e1[:, None], axis=0)
        tmp0 += tl.sum(S20 * e2[:, None], axis=0)
        tmp0 += tl.sum(S30 * e3[:, None], axis=0)
        tmp1 += tl.sum(S01 * e0[:, None], axis=0)
        tmp1 += tl.sum(S11 * e1[:, None], axis=0)
        tmp1 += tl.sum(S21 * e2[:, None], axis=0)
        tmp1 += tl.sum(S31 * e3[:, None], axis=0)
        tmp2 += tl.sum(S02 * e0[:, None], axis=0)
        tmp2 += tl.sum(S12 * e1[:, None], axis=0)
        tmp2 += tl.sum(S22 * e2[:, None], axis=0)
        tmp2 += tl.sum(S32 * e3[:, None], axis=0)
        tmp3 += tl.sum(S03 * e0[:, None], axis=0)
        tmp3 += tl.sum(S13 * e1[:, None], axis=0)
        tmp3 += tl.sum(S23 * e2[:, None], axis=0)
        tmp3 += tl.sum(S33 * e3[:, None], axis=0)
        gtmp0 = gamma_t * tmp0
        gtmp1 = gamma_t * tmp1
        gtmp2 = gamma_t * tmp2
        gtmp3 = gamma_t * tmp3
        S00 -= e0[:, None] * gtmp0[None, :]
        S01 -= e0[:, None] * gtmp1[None, :]
        S02 -= e0[:, None] * gtmp2[None, :]
        S03 -= e0[:, None] * gtmp3[None, :]
        S10 -= e1[:, None] * gtmp0[None, :]
        S11 -= e1[:, None] * gtmp1[None, :]
        S12 -= e1[:, None] * gtmp2[None, :]
        S13 -= e1[:, None] * gtmp3[None, :]
        S20 -= e2[:, None] * gtmp0[None, :]
        S21 -= e2[:, None] * gtmp1[None, :]
        S22 -= e2[:, None] * gtmp2[None, :]
        S23 -= e2[:, None] * gtmp3[None, :]
        S30 -= e3[:, None] * gtmp0[None, :]
        S31 -= e3[:, None] * gtmp1[None, :]
        S32 -= e3[:, None] * gtmp2[None, :]
        S33 -= e3[:, None] * gtmp3[None, :]
        o0 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        o1 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        o2 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        o3 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        o0 += tl.sum(S00 * q0[:, None], axis=0)
        o0 += tl.sum(S10 * q1[:, None], axis=0)
        o0 += tl.sum(S20 * q2[:, None], axis=0)
        o0 += tl.sum(S30 * q3[:, None], axis=0)
        o1 += tl.sum(S01 * q0[:, None], axis=0)
        o1 += tl.sum(S11 * q1[:, None], axis=0)
        o1 += tl.sum(S21 * q2[:, None], axis=0)
        o1 += tl.sum(S31 * q3[:, None], axis=0)
        o2 += tl.sum(S02 * q0[:, None], axis=0)
        o2 += tl.sum(S12 * q1[:, None], axis=0)
        o2 += tl.sum(S22 * q2[:, None], axis=0)
        o2 += tl.sum(S32 * q3[:, None], axis=0)
        o3 += tl.sum(S03 * q0[:, None], axis=0)
        o3 += tl.sum(S13 * q1[:, None], axis=0)
        o3 += tl.sum(S23 * q2[:, None], axis=0)
        o3 += tl.sum(S33 * q3[:, None], axis=0)
        o0 = o0 * scale
        o1 = o1 * scale
        o2 = o2 * scale
        o3 = o3 * scale
        tl.store(o_ptr + off_o_bh + t * stride_o_t + offs_v, o0, mask=offs_v < V)
        tl.store(o_ptr + off_o_bh + t * stride_o_t + BLOCK_V + offs_v, o1, mask=(BLOCK_V + offs_v) < V)
        tl.store(o_ptr + off_o_bh + t * stride_o_t + 2 * BLOCK_V + offs_v, o2, mask=(2 * BLOCK_V + offs_v) < V)
        tl.store(o_ptr + off_o_bh + t * stride_o_t + 3 * BLOCK_V + offs_v, o3, mask=(3 * BLOCK_V + offs_v) < V)

# ------------------------------------------------------------------
# HOT PATH: K==128 V==128 – 4×64×64 tiles, exp(g) pre-computed in Python
# ------------------------------------------------------------------
@triton.jit
def fwd_128_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, eg_ptr, beta_ptr, gamma_ptr, o_ptr,
    stride_q_b, stride_q_h, stride_q_t,
    stride_k_b, stride_k_h, stride_k_t,
    stride_v_b, stride_v_h, stride_v_t,
    stride_e_b, stride_e_h, stride_e_t,
    stride_eg_b, stride_eg_h, stride_eg_t,
    stride_beta_b, stride_beta_h, stride_beta_t,
    stride_gamma_b, stride_gamma_h, stride_gamma_t,
    stride_o_b, stride_o_h, stride_o_t,
    H, T, scale,
):
    pid = tl.program_id(0)
    h = pid % H
    b = pid // H
    off_q_bh = b * stride_q_b + h * stride_q_h
    off_k_bh = b * stride_k_b + h * stride_k_h
    off_v_bh = b * stride_v_b + h * stride_v_h
    off_e_bh = b * stride_e_b + h * stride_e_h
    off_eg_bh = b * stride_eg_b + h * stride_eg_h
    off_beta_bh = b * stride_beta_b + h * stride_beta_h
    off_gamma_bh = b * stride_gamma_b + h * stride_gamma_h
    off_o_bh = b * stride_o_b + h * stride_o_h
    offs = tl.arange(0, 64)
    S00 = tl.zeros((64, 64), dtype=tl.float32)
    S01 = tl.zeros((64, 64), dtype=tl.float32)
    S10 = tl.zeros((64, 64), dtype=tl.float32)
    S11 = tl.zeros((64, 64), dtype=tl.float32)
    off64 = 64
    for t in range(T):
        k0 = tl.load(k_ptr + off_k_bh + t * stride_k_t + offs).to(tl.float32)
        k1 = tl.load(k_ptr + off_k_bh + t * stride_k_t + off64 + offs).to(tl.float32)
        q0 = tl.load(q_ptr + off_q_bh + t * stride_q_t + offs).to(tl.float32)
        q1 = tl.load(q_ptr + off_q_bh + t * stride_q_t + off64 + offs).to(tl.float32)
        e0 = tl.load(e_ptr + off_e_bh + t * stride_e_t + offs).to(tl.float32)
        e1 = tl.load(e_ptr + off_e_bh + t * stride_e_t + off64 + offs).to(tl.float32)
        eg0 = tl.load(eg_ptr + off_eg_bh + t * stride_eg_t + offs).to(tl.float32)
        eg1 = tl.load(eg_ptr + off_eg_bh + t * stride_eg_t + off64 + offs).to(tl.float32)
        beta_t = tl.load(beta_ptr + off_beta_bh + t * stride_beta_t).to(tl.float32)
        gamma_t = tl.load(gamma_ptr + off_gamma_bh + t * stride_gamma_t).to(tl.float32)

        # decay – no exp inside hot loop
        S00 = S00 * eg0[:, None]
        S01 = S01 * eg0[:, None]
        S10 = S10 * eg1[:, None]
        S11 = S11 * eg1[:, None]

        # pred = S^T k
        pred0 = tl.sum(S00 * k0[:, None], axis=0) + tl.sum(S10 * k1[:, None], axis=0)
        pred1 = tl.sum(S01 * k0[:, None], axis=0) + tl.sum(S11 * k1[:, None], axis=0)

        v0 = tl.load(v_ptr + off_v_bh + t * stride_v_t + offs).to(tl.float32)
        v1 = tl.load(v_ptr + off_v_bh + t * stride_v_t + off64 + offs).to(tl.float32)
        delta0 = beta_t * (v0 - pred0)
        delta1 = beta_t * (v1 - pred1)

        # write
        S00 += k0[:, None] * delta0[None, :]
        S01 += k0[:, None] * delta1[None, :]
        S10 += k1[:, None] * delta0[None, :]
        S11 += k1[:, None] * delta1[None, :]

        # erase
        tmp0 = tl.sum(S00 * e0[:, None], axis=0) + tl.sum(S10 * e1[:, None], axis=0)
        tmp1 = tl.sum(S01 * e0[:, None], axis=0) + tl.sum(S11 * e1[:, None], axis=0)
        gtmp0 = gamma_t * tmp0
        gtmp1 = gamma_t * tmp1
        S00 -= e0[:, None] * gtmp0[None, :]
        S01 -= e0[:, None] * gtmp1[None, :]
        S10 -= e1[:, None] * gtmp0[None, :]
        S11 -= e1[:, None] * gtmp1[None, :]

        # output
        o0 = tl.sum(S00 * q0[:, None], axis=0) + tl.sum(S10 * q1[:, None], axis=0)
        o1 = tl.sum(S01 * q0[:, None], axis=0) + tl.sum(S11 * q1[:, None], axis=0)
        o0 = o0 * scale
        o1 = o1 * scale
        tl.store(o_ptr + off_o_bh + t * stride_o_t + offs, o0)
        tl.store(o_ptr + off_o_bh + t * stride_o_t + off64 + offs, o1)


@triton.jit
def fwd_generic_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr, s_ptr,
    stride_q_b, stride_q_h, stride_q_t,
    stride_k_b, stride_k_h, stride_k_t,
    stride_v_b, stride_v_h, stride_v_t,
    stride_e_b, stride_e_h, stride_e_t,
    stride_g_b, stride_g_h, stride_g_t,
    stride_beta_b, stride_beta_h, stride_beta_t,
    stride_gamma_b, stride_gamma_h, stride_gamma_t,
    stride_o_b, stride_o_h, stride_o_t,
    H, T, K, V, scale,
    NK: tl.constexpr, NV: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % H
    b = pid // H
    off_q_bh = b * stride_q_b + h * stride_q_h
    off_k_bh = b * stride_k_b + h * stride_k_h
    off_v_bh = b * stride_v_b + h * stride_v_h
    off_e_bh = b * stride_e_b + h * stride_e_h
    off_g_bh = b * stride_g_b + h * stride_g_h
    off_beta_bh = b * stride_beta_b + h * stride_beta_h
    off_gamma_bh = b * stride_gamma_b + h * stride_gamma_h
    off_o_bh = b * stride_o_b + h * stride_o_h
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    s_base = s_ptr + pid * K * V
    for t in range(T):
        beta_t = tl.load(beta_ptr + off_beta_bh + t * stride_beta_t).to(tl.float32)
        gamma_t = tl.load(gamma_ptr + off_gamma_bh + t * stride_gamma_t).to(tl.float32)
        for kb in range(NK):
            g_tile = tl.load(g_ptr + off_g_bh + t * stride_g_t + kb * BLOCK_K + offs_k, mask=(kb * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
            eg = tl.exp(g_tile)
            for vb in range(NV):
                s_blk = tl.make_block_ptr(base=s_base, shape=(K, V), strides=(V, 1), offsets=(kb * BLOCK_K, vb * BLOCK_V), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_blk, boundary_check=(0, 1))
                s = s * eg[:, None]
                tl.store(s_blk, s, boundary_check=(0, 1))
        for vb in range(NV):
            pred = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for kb in range(NK):
                k_tile = tl.load(k_ptr + off_k_bh + t * stride_k_t + kb * BLOCK_K + offs_k, mask=(kb * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
                s_blk = tl.make_block_ptr(base=s_base, shape=(K, V), strides=(V, 1), offsets=(kb * BLOCK_K, vb * BLOCK_V), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_blk, boundary_check=(0, 1))
                pred += tl.sum(s * k_tile[:, None], axis=0)
            v_tile = tl.load(v_ptr + off_v_bh + t * stride_v_t + vb * BLOCK_V + offs_v, mask=(vb * BLOCK_V + offs_v) < V, other=0.0).to(tl.float32)
            delta = beta_t * (v_tile - pred)
            for kb in range(NK):
                k_tile = tl.load(k_ptr + off_k_bh + t * stride_k_t + kb * BLOCK_K + offs_k, mask=(kb * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
                s_blk = tl.make_block_ptr(base=s_base, shape=(K, V), strides=(V, 1), offsets=(kb * BLOCK_K, vb * BLOCK_V), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_blk, boundary_check=(0, 1))
                s = s + k_tile[:, None] * delta[None, :]
                tl.store(s_blk, s, boundary_check=(0, 1))
        for vb in range(NV):
            tmp = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for kb in range(NK):
                e_tile = tl.load(e_ptr + off_e_bh + t * stride_e_t + kb * BLOCK_K + offs_k, mask=(kb * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
                s_blk = tl.make_block_ptr(base=s_base, shape=(K, V), strides=(V, 1), offsets=(kb * BLOCK_K, vb * BLOCK_V), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_blk, boundary_check=(0, 1))
                tmp += tl.sum(s * e_tile[:, None], axis=0)
            gtmp = gamma_t * tmp
            for kb in range(NK):
                e_tile = tl.load(e_ptr + off_e_bh + t * stride_e_t + kb * BLOCK_K + offs_k, mask=(kb * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
                s_blk = tl.make_block_ptr(base=s_base, shape=(K, V), strides=(V, 1), offsets=(kb * BLOCK_K, vb * BLOCK_V), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_blk, boundary_check=(0, 1))
                s = s - e_tile[:, None] * gtmp[None, :]
                tl.store(s_blk, s, boundary_check=(0, 1))
        for vb in range(NV):
            out_vec = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for kb in range(NK):
                q_tile = tl.load(q_ptr + off_q_bh + t * stride_q_t + kb * BLOCK_K + offs_k, mask=(kb * BLOCK_K + offs_k) < K, other=0.0).to(tl.float32)
                s_blk = tl.make_block_ptr(base=s_base, shape=(K, V), strides=(V, 1), offsets=(kb * BLOCK_K, vb * BLOCK_V), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_blk, boundary_check=(0, 1))
                out_vec += tl.sum(s * q_tile[:, None], axis=0)
            out_vec = out_vec * scale
            tl.store(o_ptr + off_o_bh + t * stride_o_t + vb * BLOCK_V + offs_v, out_vec, mask=(vb * BLOCK_V + offs_v) < V)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    e = e.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    gamma = gamma.contiguous()
    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    grid = (B * H,)
    if K == 128 and V == 128:
        # hoist exp(g) out of the Triton loop – cheap elementwise in PyTorch,
        # saves ~128*2048 exps per head inside the sequential loop.
        eg = torch.exp(g)
        eg = eg.contiguous()
        fwd_128_kernel[grid](
            q_ptr=q, k_ptr=k, v_ptr=v, e_ptr=e, eg_ptr=eg, beta_ptr=beta, gamma_ptr=gamma, o_ptr=out,
            stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_t=q.stride(2),
            stride_k_b=k.stride(0), stride_k_h=k.stride(1), stride_k_t=k.stride(2),
            stride_v_b=v.stride(0), stride_v_h=v.stride(1), stride_v_t=v.stride(2),
            stride_e_b=e.stride(0), stride_e_h=e.stride(1), stride_e_t=e.stride(2),
            stride_eg_b=eg.stride(0), stride_eg_h=eg.stride(1), stride_eg_t=eg.stride(2),
            stride_beta_b=beta.stride(0), stride_beta_h=beta.stride(1), stride_beta_t=beta.stride(2),
            stride_gamma_b=gamma.stride(0), stride_gamma_h=gamma.stride(1), stride_gamma_t=gamma.stride(2),
            stride_o_b=out.stride(0), stride_o_h=out.stride(1), stride_o_t=out.stride(2),
            H=H, T=T, scale=float(scale),
            num_warps=4, num_stages=3,
        )
        return out
    BLOCK_K = 32
    BLOCK_V = 32
    if K <= 128 and V <= 128:
        fwd_kernel[grid](
            q_ptr=q, k_ptr=k, v_ptr=v, e_ptr=e, g_ptr=g, beta_ptr=beta, gamma_ptr=gamma, o_ptr=out,
            stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_t=q.stride(2),
            stride_k_b=k.stride(0), stride_k_h=k.stride(1), stride_k_t=k.stride(2),
            stride_v_b=v.stride(0), stride_v_h=v.stride(1), stride_v_t=v.stride(2),
            stride_e_b=e.stride(0), stride_e_h=e.stride(1), stride_e_t=e.stride(2),
            stride_g_b=g.stride(0), stride_g_h=g.stride(1), stride_g_t=g.stride(2),
            stride_beta_b=beta.stride(0), stride_beta_h=beta.stride(1), stride_beta_t=beta.stride(2),
            stride_gamma_b=gamma.stride(0), stride_gamma_h=gamma.stride(1), stride_gamma_t=gamma.stride(2),
            stride_o_b=out.stride(0), stride_o_h=out.stride(1), stride_o_t=out.stride(2),
            H=H, T=T, K=K, V=V, scale=float(scale),
            BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
            num_warps=4, num_stages=2,
        )
    else:
        NK = (K + BLOCK_K - 1) // BLOCK_K
        NV = (V + BLOCK_V - 1) // BLOCK_V
        scratch = torch.zeros((B * H, K, V), device=q.device, dtype=torch.float32)
        fwd_generic_kernel[grid](
            q_ptr=q, k_ptr=k, v_ptr=v, e_ptr=e, g_ptr=g, beta_ptr=beta, gamma_ptr=gamma, o_ptr=out, s_ptr=scratch,
            stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_t=q.stride(2),
            stride_k_b=k.stride(0), stride_k_h=k.stride(1), stride_k_t=k.stride(2),
            stride_v_b=v.stride(0), stride_v_h=v.stride(1), stride_v_t=v.stride(2),
            stride_e_b=e.stride(0), stride_e_h=e.stride(1), stride_e_t=e.stride(2),
            stride_g_b=g.stride(0), stride_g_h=g.stride(1), stride_g_t=g.stride(2),
            stride_beta_b=beta.stride(0), stride_beta_h=beta.stride(1), stride_beta_t=beta.stride(2),
            stride_gamma_b=gamma.stride(0), stride_gamma_h=gamma.stride(1), stride_gamma_t=gamma.stride(2),
            stride_o_b=out.stride(0), stride_o_h=out.stride(1), stride_o_t=out.stride(2),
            H=H, T=T, K=K, V=V, scale=float(scale),
            NK=NK, NV=NV, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
            num_warps=4, num_stages=2,
        )
    return out
```

Hot path `K=V=128` keeps the `128x128` state in registers as 4×`64x64` tiles (≈4× fewer reductions than 32×32) and hoists `exp(g)` to a single cheap PyTorch `torch.exp` outside the 2048-step sequential loop, removing per-element `tl.exp` from the critical path; the recurrence order `D → pred → write → erase → read` is exactly the reference in FP32, with full correctness for arbitrary `K,V` via the generic tiled kernel.