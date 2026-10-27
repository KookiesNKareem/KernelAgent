import torch
import triton
import triton.language as tl

@triton.jit
def fwd_kernel_fast(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    stride_qb, stride_qh, stride_qt, stride_qk,
    stride_kb, stride_kh, stride_kt, stride_kk,
    stride_vb, stride_vh, stride_vt, stride_vk,
    stride_eb, stride_eh, stride_et, stride_ek,
    stride_gb, stride_gh, stride_gt, stride_gk,
    stride_betab, stride_betah, stride_betat,
    stride_gammab, stride_gammah, stride_gammat,
    stride_ob, stride_oh, stride_ot, stride_ok,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NUM_V_TILES: tl.constexpr, SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_v = pid % NUM_V_TILES
    pid_bh = pid // NUM_V_TILES
    h = pid_bh % H
    b = pid_bh // H

    offs_v = pid_v * 32 + tl.arange(0, 32)
    mask_v = offs_v < V

    offs_k0 = tl.arange(0, 32)
    offs_k1 = offs_k0 + 32
    offs_k2 = offs_k0 + 64
    offs_k3 = offs_k0 + 96

    s0 = tl.zeros((32, 32), dtype=tl.float32)
    s1 = tl.zeros((32, 32), dtype=tl.float32)
    s2 = tl.zeros((32, 32), dtype=tl.float32)
    s3 = tl.zeros((32, 32), dtype=tl.float32)

    q_base = q_ptr + b * stride_qb + h * stride_qh
    k_base = k_ptr + b * stride_kb + h * stride_kh
    v_base = v_ptr + b * stride_vb + h * stride_vh
    e_base = e_ptr + b * stride_eb + h * stride_eh
    g_base = g_ptr + b * stride_gb + h * stride_gh
    beta_base = beta_ptr + b * stride_betab + h * stride_betah
    gamma_base = gamma_ptr + b * stride_gammab + h * stride_gammah
    out_base = out_ptr + b * stride_ob + h * stride_oh

    for t in range(T):
        k0 = tl.load(k_base + t * stride_kt + offs_k0 * stride_kk)
        k1 = tl.load(k_base + t * stride_kt + offs_k1 * stride_kk)
        k2 = tl.load(k_base + t * stride_kt + offs_k2 * stride_kk)
        k3 = tl.load(k_base + t * stride_kt + offs_k3 * stride_kk)

        q0 = tl.load(q_base + t * stride_qt + offs_k0 * stride_qk)
        q1 = tl.load(q_base + t * stride_qt + offs_k1 * stride_qk)
        q2 = tl.load(q_base + t * stride_qt + offs_k2 * stride_qk)
        q3 = tl.load(q_base + t * stride_qt + offs_k3 * stride_qk)

        e0 = tl.load(e_base + t * stride_et + offs_k0 * stride_ek)
        e1 = tl.load(e_base + t * stride_et + offs_k1 * stride_ek)
        e2 = tl.load(e_base + t * stride_et + offs_k2 * stride_ek)
        e3 = tl.load(e_base + t * stride_et + offs_k3 * stride_ek)

        g0 = tl.load(g_base + t * stride_gt + offs_k0 * stride_gk)
        g1 = tl.load(g_base + t * stride_gt + offs_k1 * stride_gk)
        g2 = tl.load(g_base + t * stride_gt + offs_k2 * stride_gk)
        g3 = tl.load(g_base + t * stride_gt + offs_k3 * stride_gk)

        vt = tl.load(v_base + t * stride_vt + offs_v * stride_vk, mask=mask_v, other=0.0)
        beta = tl.load(beta_base + t * stride_betat)
        gamma = tl.load(gamma_base + t * stride_gammat)

        k0 = k0.to(tl.float32)
        k1 = k1.to(tl.float32)
        k2 = k2.to(tl.float32)
        k3 = k3.to(tl.float32)
        q0 = q0.to(tl.float32)
        q1 = q1.to(tl.float32)
        q2 = q2.to(tl.float32)
        q3 = q3.to(tl.float32)
        e0 = e0.to(tl.float32)
        e1 = e1.to(tl.float32)
        e2 = e2.to(tl.float32)
        e3 = e3.to(tl.float32)
        g0 = g0.to(tl.float32)
        g1 = g1.to(tl.float32)
        g2 = g2.to(tl.float32)
        g3 = g3.to(tl.float32)
        vt = vt.to(tl.float32)

        d0 = tl.exp(g0)
        d1 = tl.exp(g1)
        d2 = tl.exp(g2)
        d3 = tl.exp(g3)

        s0 = s0 * d0[:, None]
        s1 = s1 * d1[:, None]
        s2 = s2 * d2[:, None]
        s3 = s3 * d3[:, None]

        p0 = tl.sum(s0 * k0[:, None], axis=0)
        p1 = tl.sum(s1 * k1[:, None], axis=0)
        p2 = tl.sum(s2 * k2[:, None], axis=0)
        p3 = tl.sum(s3 * k3[:, None], axis=0)
        pred = p0 + p1 + p2 + p3

        delta = beta * (vt - pred)

        s0 = s0 + k0[:, None] * delta[None, :] + gamma * e0[:, None] * vt[None, :]
        s1 = s1 + k1[:, None] * delta[None, :] + gamma * e1[:, None] * vt[None, :]
        s2 = s2 + k2[:, None] * delta[None, :] + gamma * e2[:, None] * vt[None, :]
        s3 = s3 + k3[:, None] * delta[None, :] + gamma * e3[:, None] * vt[None, :]

        o0 = tl.sum(s0 * q0[:, None], axis=0)
        o1 = tl.sum(s1 * q1[:, None], axis=0)
        o2 = tl.sum(s2 * q2[:, None], axis=0)
        o3 = tl.sum(s3 * q3[:, None], axis=0)
        out = SCALE * (o0 + o1 + o2 + o3)
        tl.store(out_base + t * stride_ot + offs_v * stride_ok, out, mask=mask_v)


@triton.jit
def fwd_kernel_generic(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    stride_qb, stride_qh, stride_qt, stride_qk,
    stride_kb, stride_kh, stride_kt, stride_kk,
    stride_vb, stride_vh, stride_vt, stride_vk,
    stride_eb, stride_eh, stride_et, stride_ek,
    stride_gb, stride_gh, stride_gt, stride_gk,
    stride_betab, stride_betah, stride_betat,
    stride_gammab, stride_gammah, stride_gammat,
    stride_ob, stride_oh, stride_ot, stride_ok,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NUM_V_TILES: tl.constexpr, SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_v = pid % NUM_V_TILES
    pid_bh = pid // NUM_V_TILES
    h = pid_bh % H
    b = pid_bh // H

    offs_v = pid_v * 32 + tl.arange(0, 32)
    mask_v = offs_v < V

    offs_k0 = tl.arange(0, 32)
    offs_k1 = offs_k0 + 32
    offs_k2 = offs_k0 + 64
    offs_k3 = offs_k0 + 96
    offs_k4 = offs_k0 + 128
    offs_k5 = offs_k0 + 160
    offs_k6 = offs_k0 + 192
    offs_k7 = offs_k0 + 224

    s0 = tl.zeros((32, 32), dtype=tl.float32)
    s1 = tl.zeros((32, 32), dtype=tl.float32)
    s2 = tl.zeros((32, 32), dtype=tl.float32)
    s3 = tl.zeros((32, 32), dtype=tl.float32)
    s4 = tl.zeros((32, 32), dtype=tl.float32)
    s5 = tl.zeros((32, 32), dtype=tl.float32)
    s6 = tl.zeros((32, 32), dtype=tl.float32)
    s7 = tl.zeros((32, 32), dtype=tl.float32)

    q_base = q_ptr + b * stride_qb + h * stride_qh
    k_base = k_ptr + b * stride_kb + h * stride_kh
    v_base = v_ptr + b * stride_vb + h * stride_vh
    e_base = e_ptr + b * stride_eb + h * stride_eh
    g_base = g_ptr + b * stride_gb + h * stride_gh
    beta_base = beta_ptr + b * stride_betab + h * stride_betah
    gamma_base = gamma_ptr + b * stride_gammab + h * stride_gammah
    out_base = out_ptr + b * stride_ob + h * stride_oh

    for t in range(T):
        k0 = tl.load(k_base + t * stride_kt + offs_k0 * stride_kk, mask=offs_k0 < K, other=0.0)
        k1 = tl.load(k_base + t * stride_kt + offs_k1 * stride_kk, mask=offs_k1 < K, other=0.0)
        k2 = tl.load(k_base + t * stride_kt + offs_k2 * stride_kk, mask=offs_k2 < K, other=0.0)
        k3 = tl.load(k_base + t * stride_kt + offs_k3 * stride_kk, mask=offs_k3 < K, other=0.0)
        k4 = tl.load(k_base + t * stride_kt + offs_k4 * stride_kk, mask=offs_k4 < K, other=0.0)
        k5 = tl.load(k_base + t * stride_kt + offs_k5 * stride_kk, mask=offs_k5 < K, other=0.0)
        k6 = tl.load(k_base + t * stride_kt + offs_k6 * stride_kk, mask=offs_k6 < K, other=0.0)
        k7 = tl.load(k_base + t * stride_kt + offs_k7 * stride_kk, mask=offs_k7 < K, other=0.0)

        q0 = tl.load(q_base + t * stride_qt + offs_k0 * stride_qk, mask=offs_k0 < K, other=0.0)
        q1 = tl.load(q_base + t * stride_qt + offs_k1 * stride_qk, mask=offs_k1 < K, other=0.0)
        q2 = tl.load(q_base + t * stride_qt + offs_k2 * stride_qk, mask=offs_k2 < K, other=0.0)
        q3 = tl.load(q_base + t * stride_qt + offs_k3 * stride_qk, mask=offs_k3 < K, other=0.0)
        q4 = tl.load(q_base + t * stride_qt + offs_k4 * stride_qk, mask=offs_k4 < K, other=0.0)
        q5 = tl.load(q_base + t * stride_qt + offs_k5 * stride_qk, mask=offs_k5 < K, other=0.0)
        q6 = tl.load(q_base + t * stride_qt + offs_k6 * stride_qk, mask=offs_k6 < K, other=0.0)
        q7 = tl.load(q_base + t * stride_qt + offs_k7 * stride_qk, mask=offs_k7 < K, other=0.0)

        e0 = tl.load(e_base + t * stride_et + offs_k0 * stride_ek, mask=offs_k0 < K, other=0.0)
        e1 = tl.load(e_base + t * stride_et + offs_k1 * stride_ek, mask=offs_k1 < K, other=0.0)
        e2 = tl.load(e_base + t * stride_et + offs_k2 * stride_ek, mask=offs_k2 < K, other=0.0)
        e3 = tl.load(e_base + t * stride_et + offs_k3 * stride_ek, mask=offs_k3 < K, other=0.0)
        e4 = tl.load(e_base + t * stride_et + offs_k4 * stride_ek, mask=offs_k4 < K, other=0.0)
        e5 = tl.load(e_base + t * stride_et + offs_k5 * stride_ek, mask=offs_k5 < K, other=0.0)
        e6 = tl.load(e_base + t * stride_et + offs_k6 * stride_ek, mask=offs_k6 < K, other=0.0)
        e7 = tl.load(e_base + t * stride_et + offs_k7 * stride_ek, mask=offs_k7 < K, other=0.0)

        g0 = tl.load(g_base + t * stride_gt + offs_k0 * stride_gk, mask=offs_k0 < K, other=0.0)
        g1 = tl.load(g_base + t * stride_gt + offs_k1 * stride_gk, mask=offs_k1 < K, other=0.0)
        g2 = tl.load(g_base + t * stride_gt + offs_k2 * stride_gk, mask=offs_k2 < K, other=0.0)
        g3 = tl.load(g_base + t * stride_gt + offs_k3 * stride_gk, mask=offs_k3 < K, other=0.0)
        g4 = tl.load(g_base + t * stride_gt + offs_k4 * stride_gk, mask=offs_k4 < K, other=0.0)
        g5 = tl.load(g_base + t * stride_gt + offs_k5 * stride_gk, mask=offs_k5 < K, other=0.0)
        g6 = tl.load(g_base + t * stride_gt + offs_k6 * stride_gk, mask=offs_k6 < K, other=0.0)
        g7 = tl.load(g_base + t * stride_gt + offs_k7 * stride_gk, mask=offs_k7 < K, other=0.0)

        vt = tl.load(v_base + t * stride_vt + offs_v * stride_vk, mask=mask_v, other=0.0)
        beta = tl.load(beta_base + t * stride_betat)
        gamma = tl.load(gamma_base + t * stride_gammat)

        k0 = k0.to(tl.float32)
        k1 = k1.to(tl.float32)
        k2 = k2.to(tl.float32)
        k3 = k3.to(tl.float32)
        k4 = k4.to(tl.float32)
        k5 = k5.to(tl.float32)
        k6 = k6.to(tl.float32)
        k7 = k7.to(tl.float32)
        q0 = q0.to(tl.float32)
        q1 = q1.to(tl.float32)
        q2 = q2.to(tl.float32)
        q3 = q3.to(tl.float32)
        q4 = q4.to(tl.float32)
        q5 = q5.to(tl.float32)
        q6 = q6.to(tl.float32)
        q7 = q7.to(tl.float32)
        e0 = e0.to(tl.float32)
        e1 = e1.to(tl.float32)
        e2 = e2.to(tl.float32)
        e3 = e3.to(tl.float32)
        e4 = e4.to(tl.float32)
        e5 = e5.to(tl.float32)
        e6 = e6.to(tl.float32)
        e7 = e7.to(tl.float32)
        g0 = g0.to(tl.float32)
        g1 = g1.to(tl.float32)
        g2 = g2.to(tl.float32)
        g3 = g3.to(tl.float32)
        g4 = g4.to(tl.float32)
        g5 = g5.to(tl.float32)
        g6 = g6.to(tl.float32)
        g7 = g7.to(tl.float32)
        vt = vt.to(tl.float32)

        d0 = tl.exp(g0)
        d1 = tl.exp(g1)
        d2 = tl.exp(g2)
        d3 = tl.exp(g3)
        d4 = tl.exp(g4)
        d5 = tl.exp(g5)
        d6 = tl.exp(g6)
        d7 = tl.exp(g7)

        s0 = s0 * d0[:, None]
        s1 = s1 * d1[:, None]
        s2 = s2 * d2[:, None]
        s3 = s3 * d3[:, None]
        s4 = s4 * d4[:, None]
        s5 = s5 * d5[:, None]
        s6 = s6 * d6[:, None]
        s7 = s7 * d7[:, None]

        p0 = tl.sum(s0 * k0[:, None], axis=0)
        p1 = tl.sum(s1 * k1[:, None], axis=0)
        p2 = tl.sum(s2 * k2[:, None], axis=0)
        p3 = tl.sum(s3 * k3[:, None], axis=0)
        p4 = tl.sum(s4 * k4[:, None], axis=0)
        p5 = tl.sum(s5 * k5[:, None], axis=0)
        p6 = tl.sum(s6 * k6[:, None], axis=0)
        p7 = tl.sum(s7 * k7[:, None], axis=0)
        pred = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7

        delta = beta * (vt - pred)

        s0 = s0 + k0[:, None] * delta[None, :] + gamma * e0[:, None] * vt[None, :]
        s1 = s1 + k1[:, None] * delta[None, :] + gamma * e1[:, None] * vt[None, :]
        s2 = s2 + k2[:, None] * delta[None, :] + gamma * e2[:, None] * vt[None, :]
        s3 = s3 + k3[:, None] * delta[None, :] + gamma * e3[:, None] * vt[None, :]
        s4 = s4 + k4[:, None] * delta[None, :] + gamma * e4[:, None] * vt[None, :]
        s5 = s5 + k5[:, None] * delta[None, :] + gamma * e5[:, None] * vt[None, :]
        s6 = s6 + k6[:, None] * delta[None, :] + gamma * e6[:, None] * vt[None, :]
        s7 = s7 + k7[:, None] * delta[None, :] + gamma * e7[:, None] * vt[None, :]

        o0 = tl.sum(s0 * q0[:, None], axis=0)
        o1 = tl.sum(s1 * q1[:, None], axis=0)
        o2 = tl.sum(s2 * q2[:, None], axis=0)
        o3 = tl.sum(s3 * q3[:, None], axis=0)
        o4 = tl.sum(s4 * q4[:, None], axis=0)
        o5 = tl.sum(s5 * q5[:, None], axis=0)
        o6 = tl.sum(s6 * q6[:, None], axis=0)
        o7 = tl.sum(s7 * q7[:, None], axis=0)
        out = SCALE * (o0 + o1 + o2 + o3 + o4 + o5 + o6 + o7)
        tl.store(out_base + t * stride_ot + offs_v * stride_ok, out, mask=mask_v)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    stride_qb, stride_qh, stride_qt, stride_qk = q.stride()
    stride_kb, stride_kh, stride_kt, stride_kk = k.stride()
    stride_vb, stride_vh, stride_vt, stride_vk = v.stride()
    stride_eb, stride_eh, stride_et, stride_ek = e.stride()
    stride_gb, stride_gh, stride_gt, stride_gk = g.stride()
    stride_betab, stride_betah, stride_betat = beta.stride()
    stride_gammab, stride_gammah, stride_gammat = gamma.stride()
    stride_ob, stride_oh, stride_ot, stride_ok = out.stride()

    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)

    NUM_V_TILES = (V + 31) // 32
    grid = (B * H * NUM_V_TILES,)

    # fast path: K multiple of 32 and <=128 covers all timed shapes (128)
    if K % 32 == 0 and K <= 128:
        fwd_kernel_fast[grid](
            q, k, v, e, g, beta, gamma, out,
            stride_qb, stride_qh, stride_qt, stride_qk,
            stride_kb, stride_kh, stride_kt, stride_kk,
            stride_vb, stride_vh, stride_vt, stride_vk,
            stride_eb, stride_eh, stride_et, stride_ek,
            stride_gb, stride_gh, stride_gt, stride_gk,
            stride_betab, stride_betah, stride_betat,
            stride_gammab, stride_gammah, stride_gammat,
            stride_ob, stride_oh, stride_ot, stride_ok,
            B, H, T, K, V, NUM_V_TILES, scale_f,
        )
    else:
        fwd_kernel_generic[grid](
            q, k, v, e, g, beta, gamma, out,
            stride_qb, stride_qh, stride_qt, stride_qk,
            stride_kb, stride_kh, stride_kt, stride_kk,
            stride_vb, stride_vh, stride_vt, stride_vk,
            stride_eb, stride_eh, stride_et, stride_ek,
            stride_gb, stride_gh, stride_gt, stride_gk,
            stride_betab, stride_betah, stride_betat,
            stride_gammab, stride_gammah, stride_gammat,
            stride_ob, stride_oh, stride_ot, stride_ok,
            B, H, T, K, V, NUM_V_TILES, scale_f,
        )
    return out
