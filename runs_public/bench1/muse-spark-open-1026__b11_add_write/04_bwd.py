import torch
import triton
import triton.language as tl

# ---------- forward with final state ----------
@triton.jit
def fwd_kernel_final(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr, final_ptr,
    stride_qb, stride_qh, stride_qt, stride_qk,
    stride_kb, stride_kh, stride_kt, stride_kk,
    stride_vb, stride_vh, stride_vt, stride_vk,
    stride_eb, stride_eh, stride_et, stride_ek,
    stride_gb, stride_gh, stride_gt, stride_gk,
    stride_betab, stride_betah, stride_betat,
    stride_gammab, stride_gammah, stride_gammat,
    stride_ob, stride_oh, stride_ot, stride_ok,
    stride_fb, stride_fh, stride_fk, stride_fv,
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
        k0 = tl.load(k_base + t * stride_kt + offs_k0 * stride_kk, mask=offs_k0 < K, other=0.0)
        k1 = tl.load(k_base + t * stride_kt + offs_k1 * stride_kk, mask=offs_k1 < K, other=0.0)
        k2 = tl.load(k_base + t * stride_kt + offs_k2 * stride_kk, mask=offs_k2 < K, other=0.0)
        k3 = tl.load(k_base + t * stride_kt + offs_k3 * stride_kk, mask=offs_k3 < K, other=0.0)
        q0 = tl.load(q_base + t * stride_qt + offs_k0 * stride_qk, mask=offs_k0 < K, other=0.0)
        q1 = tl.load(q_base + t * stride_qt + offs_k1 * stride_qk, mask=offs_k1 < K, other=0.0)
        q2 = tl.load(q_base + t * stride_qt + offs_k2 * stride_qk, mask=offs_k2 < K, other=0.0)
        q3 = tl.load(q_base + t * stride_qt + offs_k3 * stride_qk, mask=offs_k3 < K, other=0.0)
        e0 = tl.load(e_base + t * stride_et + offs_k0 * stride_ek, mask=offs_k0 < K, other=0.0)
        e1 = tl.load(e_base + t * stride_et + offs_k1 * stride_ek, mask=offs_k1 < K, other=0.0)
        e2 = tl.load(e_base + t * stride_et + offs_k2 * stride_ek, mask=offs_k2 < K, other=0.0)
        e3 = tl.load(e_base + t * stride_et + offs_k3 * stride_ek, mask=offs_k3 < K, other=0.0)
        g0 = tl.load(g_base + t * stride_gt + offs_k0 * stride_gk, mask=offs_k0 < K, other=0.0)
        g1 = tl.load(g_base + t * stride_gt + offs_k1 * stride_gk, mask=offs_k1 < K, other=0.0)
        g2 = tl.load(g_base + t * stride_gt + offs_k2 * stride_gk, mask=offs_k2 < K, other=0.0)
        g3 = tl.load(g_base + t * stride_gt + offs_k3 * stride_gk, mask=offs_k3 < K, other=0.0)
        vt = tl.load(v_base + t * stride_vt + offs_v * stride_vk, mask=mask_v, other=0.0)
        beta = tl.load(beta_base + t * stride_betat)
        gamma = tl.load(gamma_base + t * stride_gammat)
        k0 = k0.to(tl.float32); k1 = k1.to(tl.float32); k2 = k2.to(tl.float32); k3 = k3.to(tl.float32)
        q0 = q0.to(tl.float32); q1 = q1.to(tl.float32); q2 = q2.to(tl.float32); q3 = q3.to(tl.float32)
        e0 = e0.to(tl.float32); e1 = e1.to(tl.float32); e2 = e2.to(tl.float32); e3 = e3.to(tl.float32)
        g0 = g0.to(tl.float32); g1 = g1.to(tl.float32); g2 = g2.to(tl.float32); g3 = g3.to(tl.float32)
        vt = vt.to(tl.float32)
        d0 = tl.exp(g0); d1 = tl.exp(g1); d2 = tl.exp(g2); d3 = tl.exp(g3)
        s0 = s0 * d0[:, None]; s1 = s1 * d1[:, None]; s2 = s2 * d2[:, None]; s3 = s3 * d3[:, None]
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

    # store final S
    final_base = final_ptr + b * stride_fb + h * stride_fh
    # need 2D tile store
    offs_k0_2d = offs_k0[:, None]
    offs_k1_2d = offs_k1[:, None]
    offs_k2_2d = offs_k2[:, None]
    offs_k3_2d = offs_k3[:, None]
    offs_v_2d = offs_v[None, :]
    mask0 = (offs_k0[:, None] < K) & (offs_v[None, :] < V)
    mask1 = (offs_k1[:, None] < K) & (offs_v[None, :] < V)
    mask2 = (offs_k2[:, None] < K) & (offs_v[None, :] < V)
    mask3 = (offs_k3[:, None] < K) & (offs_v[None, :] < V)
    ptr0 = final_base + offs_k0_2d * stride_fk + offs_v_2d * stride_fv
    ptr1 = final_base + offs_k1_2d * stride_fk + offs_v_2d * stride_fv
    ptr2 = final_base + offs_k2_2d * stride_fk + offs_v_2d * stride_fv
    ptr3 = final_base + offs_k3_2d * stride_fk + offs_v_2d * stride_fv
    tl.store(ptr0, s0, mask=mask0)
    tl.store(ptr1, s1, mask=mask1)
    tl.store(ptr2, s2, mask=mask2)
    tl.store(ptr3, s3, mask=mask3)


@triton.jit
def bwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, go_ptr, final_ptr,
    dq_ptr, dk_ptr, de_ptr, dg_ptr, dbeta_ptr, dgamma_ptr, dv_ptr,
    stride_qb, stride_qh, stride_qt, stride_qk,
    stride_kb, stride_kh, stride_kt, stride_kk,
    stride_vb, stride_vh, stride_vt, stride_vk,
    stride_eb, stride_eh, stride_et, stride_ek,
    stride_gb, stride_gh, stride_gt, stride_gk,
    stride_betab, stride_betah, stride_betat,
    stride_gammab, stride_gammah, stride_gammat,
    stride_gob, stride_goh, stride_got, stride_gov,
    stride_fb, stride_fh, stride_fk, stride_fv,
    stride_dqb, stride_dqh, stride_dqt, stride_dqk,
    stride_dkb, stride_dkh, stride_dkt, stride_dkk,
    stride_deb, stride_deh, stride_det, stride_dek,
    stride_dgb, stride_dgh, stride_dgt, stride_dgk,
    stride_dbetab, stride_dbetah, stride_dbetat,
    stride_dgammab, stride_dgammah, stride_dgammat,
    stride_dvb, stride_dvh, stride_dvt, stride_dvv,
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

    # bases
    q_base = q_ptr + b * stride_qb + h * stride_qh
    k_base = k_ptr + b * stride_kb + h * stride_kh
    v_base = v_ptr + b * stride_vb + h * stride_vh
    e_base = e_ptr + b * stride_eb + h * stride_eh
    g_base = g_ptr + b * stride_gb + h * stride_gh
    beta_base = beta_ptr + b * stride_betab + h * stride_betah
    gamma_base = gamma_ptr + b * stride_gammab + h * stride_gammah
    go_base = go_ptr + b * stride_gob + h * stride_goh
    dq_base = dq_ptr + b * stride_dqb + h * stride_dqh
    dk_base = dk_ptr + b * stride_dkb + h * stride_dkh
    de_base = de_ptr + b * stride_deb + h * stride_deh
    dg_base = dg_ptr + b * stride_dgb + h * stride_dgh
    dv_base = dv_ptr + b * stride_dvb + h * stride_dvh
    dbeta_base = dbeta_ptr + b * stride_dbetab + h * stride_dbetah
    dgamma_base = dgamma_ptr + b * stride_dgammab + h * stride_dgammah
    final_base = final_ptr + b * stride_fb + h * stride_fh

    # init S from final
    offs_k0_2d = offs_k0[:, None]
    offs_k1_2d = offs_k1[:, None]
    offs_k2_2d = offs_k2[:, None]
    offs_k3_2d = offs_k3[:, None]
    offs_v_2d = offs_v[None, :]
    ptr0 = final_base + offs_k0_2d * stride_fk + offs_v_2d * stride_fv
    ptr1 = final_base + offs_k1_2d * stride_fk + offs_v_2d * stride_fv
    ptr2 = final_base + offs_k2_2d * stride_fk + offs_v_2d * stride_fv
    ptr3 = final_base + offs_k3_2d * stride_fk + offs_v_2d * stride_fv
    s0 = tl.load(ptr0, mask=(offs_k0[:, None] < K) & (offs_v[None, :] < V), other=0.0)
    s1 = tl.load(ptr1, mask=(offs_k1[:, None] < K) & (offs_v[None, :] < V), other=0.0)
    s2 = tl.load(ptr2, mask=(offs_k2[:, None] < K) & (offs_v[None, :] < V), other=0.0)
    s3 = tl.load(ptr3, mask=(offs_k3[:, None] < K) & (offs_v[None, :] < V), other=0.0)

    cur0 = tl.zeros((32, 32), dtype=tl.float32)
    cur1 = tl.zeros((32, 32), dtype=tl.float32)
    cur2 = tl.zeros((32, 32), dtype=tl.float32)
    cur3 = tl.zeros((32, 32), dtype=tl.float32)

    for t_inc in range(T):
        t = T - 1 - t_inc
        k0 = tl.load(k_base + t * stride_kt + offs_k0 * stride_kk, mask=offs_k0 < K, other=0.0)
        k1 = tl.load(k_base + t * stride_kt + offs_k1 * stride_kk, mask=offs_k1 < K, other=0.0)
        k2 = tl.load(k_base + t * stride_kt + offs_k2 * stride_kk, mask=offs_k2 < K, other=0.0)
        k3 = tl.load(k_base + t * stride_kt + offs_k3 * stride_kk, mask=offs_k3 < K, other=0.0)
        q0 = tl.load(q_base + t * stride_qt + offs_k0 * stride_qk, mask=offs_k0 < K, other=0.0)
        q1 = tl.load(q_base + t * stride_qt + offs_k1 * stride_qk, mask=offs_k1 < K, other=0.0)
        q2 = tl.load(q_base + t * stride_qt + offs_k2 * stride_qk, mask=offs_k2 < K, other=0.0)
        q3 = tl.load(q_base + t * stride_qt + offs_k3 * stride_qk, mask=offs_k3 < K, other=0.0)
        e0 = tl.load(e_base + t * stride_et + offs_k0 * stride_ek, mask=offs_k0 < K, other=0.0)
        e1 = tl.load(e_base + t * stride_et + offs_k1 * stride_ek, mask=offs_k1 < K, other=0.0)
        e2 = tl.load(e_base + t * stride_et + offs_k2 * stride_ek, mask=offs_k2 < K, other=0.0)
        e3 = tl.load(e_base + t * stride_et + offs_k3 * stride_ek, mask=offs_k3 < K, other=0.0)
        g0 = tl.load(g_base + t * stride_gt + offs_k0 * stride_gk, mask=offs_k0 < K, other=0.0)
        g1 = tl.load(g_base + t * stride_gt + offs_k1 * stride_gk, mask=offs_k1 < K, other=0.0)
        g2 = tl.load(g_base + t * stride_gt + offs_k2 * stride_gk, mask=offs_k2 < K, other=0.0)
        g3 = tl.load(g_base + t * stride_gt + offs_k3 * stride_gk, mask=offs_k3 < K, other=0.0)
        vt = tl.load(v_base + t * stride_vt + offs_v * stride_vk, mask=mask_v, other=0.0)
        got = tl.load(go_base + t * stride_got + offs_v * stride_gov, mask=mask_v, other=0.0)
        beta = tl.load(beta_base + t * stride_betat)
        gamma = tl.load(gamma_base + t * stride_gammat)

        k0 = k0.to(tl.float32); k1 = k1.to(tl.float32); k2 = k2.to(tl.float32); k3 = k3.to(tl.float32)
        q0 = q0.to(tl.float32); q1 = q1.to(tl.float32); q2 = q2.to(tl.float32); q3 = q3.to(tl.float32)
        e0 = e0.to(tl.float32); e1 = e1.to(tl.float32); e2 = e2.to(tl.float32); e3 = e3.to(tl.float32)
        g0 = g0.to(tl.float32); g1 = g1.to(tl.float32); g2 = g2.to(tl.float32); g3 = g3.to(tl.float32)
        vt = vt.to(tl.float32); got = got.to(tl.float32)

        # G = QGO + cur
        qgo0 = SCALE * q0[:, None] * got[None, :]
        qgo1 = SCALE * q1[:, None] * got[None, :]
        qgo2 = SCALE * q2[:, None] * got[None, :]
        qgo3 = SCALE * q3[:, None] * got[None, :]
        g0t = qgo0 + cur0
        g1t = qgo1 + cur1
        g2t = qgo2 + cur2
        g3t = qgo3 + cur3

        # for inverse
        k2norm = tl.sum(k0 * k0) + tl.sum(k1 * k1) + tl.sum(k2 * k2) + tl.sum(k3 * k3)
        denom = 1.0 - beta * k2norm
        alpha = beta / denom
        # if denom near 0, avoid inf (clamp)
        # alpha = tl.where(tl.abs(denom) < 1e-6, 0.0, alpha)

        w0 = beta * k0 + gamma * e0
        w1 = beta * k1 + gamma * e1
        w2 = beta * k2 + gamma * e2
        w3 = beta * k3 + gamma * e3

        # X = S - w v^T
        x0 = s0 - w0[:, None] * vt[None, :]
        x1 = s1 - w1[:, None] * vt[None, :]
        x2 = s2 - w2[:, None] * vt[None, :]
        x3 = s3 - w3[:, None] * vt[None, :]

        kX = tl.sum(x0 * k0[:, None], axis=0) + tl.sum(x1 * k1[:, None], axis=0) + tl.sum(x2 * k2[:, None], axis=0) + tl.sum(x3 * k3[:, None], axis=0)

        y0 = x0 + alpha * k0[:, None] * kX[None, :]
        y1 = x1 + alpha * k1[:, None] * kX[None, :]
        y2 = x2 + alpha * k2[:, None] * kX[None, :]
        y3 = x3 + alpha * k3[:, None] * kX[None, :]

        pred = tl.sum(y0 * k0[:, None], axis=0) + tl.sum(y1 * k1[:, None], axis=0) + tl.sum(y2 * k2[:, None], axis=0) + tl.sum(y3 * k3[:, None], axis=0)
        delta = vt - pred

        # a = G^T k
        a = tl.sum(g0t * k0[:, None], axis=0) + tl.sum(g1t * k1[:, None], axis=0) + tl.sum(g2t * k2[:, None], axis=0) + tl.sum(g3t * k3[:, None], axis=0)

        # AG
        ag0 = g0t - beta * k0[:, None] * a[None, :]
        ag1 = g1t - beta * k1[:, None] * a[None, :]
        ag2 = g2t - beta * k2[:, None] * a[None, :]
        ag3 = g3t - beta * k3[:, None] * a[None, :]

        # dv = G^T w (per V)
        dv_slice = tl.sum(g0t * w0[:, None], axis=0) + tl.sum(g1t * w1[:, None], axis=0) + tl.sum(g2t * w2[:, None], axis=0) + tl.sum(g3t * w3[:, None], axis=0)
        tl.store(dv_base + t * stride_dvt + offs_v * stride_dvv, dv_slice, mask=mask_v)

        # dq partial = SCALE * S @ go
        dq0 = tl.sum(s0 * got[None, :], axis=1) * SCALE
        dq1 = tl.sum(s1 * got[None, :], axis=1) * SCALE
        dq2 = tl.sum(s2 * got[None, :], axis=1) * SCALE
        dq3 = tl.sum(s3 * got[None, :], axis=1) * SCALE
        # de partial = gamma * G v
        de0 = tl.sum(g0t * vt[None, :], axis=1) * gamma
        de1 = tl.sum(g1t * vt[None, :], axis=1) * gamma
        de2 = tl.sum(g2t * vt[None, :], axis=1) * gamma
        de3 = tl.sum(g3t * vt[None, :], axis=1) * gamma
        # dk partials: beta*( G*delta - M*a )
        # G*delta
        gdk0 = tl.sum(g0t * delta[None, :], axis=1)
        gdk1 = tl.sum(g1t * delta[None, :], axis=1)
        gdk2 = tl.sum(g2t * delta[None, :], axis=1)
        gdk3 = tl.sum(g3t * delta[None, :], axis=1)
        # M*a
        mak0 = tl.sum(y0 * a[None, :], axis=1)
        mak1 = tl.sum(y1 * a[None, :], axis=1)
        mak2 = tl.sum(y2 * a[None, :], axis=1)
        mak3 = tl.sum(y3 * a[None, :], axis=1)
        dk0 = beta * (gdk0 - mak0)
        dk1 = beta * (gdk1 - mak1)
        dk2 = beta * (gdk2 - mak2)
        dk3 = beta * (gdk3 - mak3)

        # dg partial: d * sum(Sprev * AG)
        invd0 = tl.exp(-g0); invd1 = tl.exp(-g1); invd2 = tl.exp(-g2); invd3 = tl.exp(-g3)
        sp0 = y0 * invd0[:, None]
        sp1 = y1 * invd1[:, None]
        sp2 = y2 * invd2[:, None]
        sp3 = y3 * invd3[:, None]
        d0e = tl.exp(g0); d1e = tl.exp(g1); d2e = tl.exp(g2); d3e = tl.exp(g3)
        dg0 = d0e * tl.sum(sp0 * ag0, axis=1)
        dg1 = d1e * tl.sum(sp1 * ag1, axis=1)
        dg2 = d2e * tl.sum(sp2 * ag2, axis=1)
        dg3 = d3e * tl.sum(sp3 * ag3, axis=1)

        # atomic adds for K vectors
        tl.atomic_add(dq_base + t * stride_dqt + offs_k0 * stride_dqk, dq0, mask=offs_k0 < K)
        tl.atomic_add(dq_base + t * stride_dqt + offs_k1 * stride_dqk, dq1, mask=offs_k1 < K)
        tl.atomic_add(dq_base + t * stride_dqt + offs_k2 * stride_dqk, dq2, mask=offs_k2 < K)
        tl.atomic_add(dq_base + t * stride_dqt + offs_k3 * stride_dqk, dq3, mask=offs_k3 < K)

        tl.atomic_add(dk_base + t * stride_dkt + offs_k0 * stride_dkk, dk0, mask=offs_k0 < K)
        tl.atomic_add(dk_base + t * stride_dkt + offs_k1 * stride_dkk, dk1, mask=offs_k1 < K)
        tl.atomic_add(dk_base + t * stride_dkt + offs_k2 * stride_dkk, dk2, mask=offs_k2 < K)
        tl.atomic_add(dk_base + t * stride_dkt + offs_k3 * stride_dkk, dk3, mask=offs_k3 < K)

        tl.atomic_add(de_base + t * stride_det + offs_k0 * stride_dek, de0, mask=offs_k0 < K)
        tl.atomic_add(de_base + t * stride_det + offs_k1 * stride_dek, de1, mask=offs_k1 < K)
        tl.atomic_add(de_base + t * stride_det + offs_k2 * stride_dek, de2, mask=offs_k2 < K)
        tl.atomic_add(de_base + t * stride_det + offs_k3 * stride_dek, de3, mask=offs_k3 < K)

        tl.atomic_add(dg_base + t * stride_dgt + offs_k0 * stride_dgk, dg0, mask=offs_k0 < K)
        tl.atomic_add(dg_base + t * stride_dgt + offs_k1 * stride_dgk, dg1, mask=offs_k1 < K)
        tl.atomic_add(dg_base + t * stride_dgt + offs_k2 * stride_dgk, dg2, mask=offs_k2 < K)
        tl.atomic_add(dg_base + t * stride_dgt + offs_k3 * stride_dgk, dg3, mask=offs_k3 < K)

        # beta / gamma scalars
        beta_part = tl.sum(a * delta)
        # G^T e
        gTe = tl.sum(g0t * e0[:, None], axis=0) + tl.sum(g1t * e1[:, None], axis=0) + tl.sum(g2t * e2[:, None], axis=0) + tl.sum(g3t * e3[:, None], axis=0)
        gamma_part = tl.sum(gTe * vt)
        tl.atomic_add(dbeta_base + t * stride_dbetat, beta_part)
        tl.atomic_add(dgamma_base + t * stride_dgammat, gamma_part)

        # prepare next iter
        s0 = sp0; s1 = sp1; s2 = sp2; s3 = sp3
        cur0 = ag0 * d0e[:, None]
        cur1 = ag1 * d1e[:, None]
        cur2 = ag2 * d2e[:, None]
        cur3 = ag3 * d3e[:, None]


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        final = torch.empty((B, H, K, V), device=q.device, dtype=torch.float32)
        BLOCK_V = 32
        NUM_V_TILES = (V + BLOCK_V - 1)//BLOCK_V
        grid = (B*H*NUM_V_TILES,)
        scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        fwd_kernel_final[grid](
            q, k, v, e, g, beta, gamma, out, final,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            e.stride(0), e.stride(1), e.stride(2), e.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            final.stride(0), final.stride(1), final.stride(2), final.stride(3),
            B, H, T, K, V, NUM_V_TILES, scale_f
        )
        ctx.save_for_backward(q,k,v,e,g,beta,gamma,final)
        ctx.scale = scale_f
        ctx.BHW = (B,H,T,K,V,NUM_V_TILES)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q,k,v,e,g,beta,gamma,final = ctx.saved_tensors
        scale_f = ctx.scale
        B,H,T,K,V,NUM_V_TILES = ctx.BHW
        # grad_out is [B,H,T,V] float32
        dq = torch.zeros((B,H,T,K), device=q.device, dtype=torch.float32)
        dk = torch.zeros((B,H,T,K), device=q.device, dtype=torch.float32)
        de = torch.zeros((B,H,T,K), device=q.device, dtype=torch.float32)
        dg = torch.zeros((B,H,T,K), device=q.device, dtype=torch.float32)
        dv = torch.zeros((B,H,T,V), device=q.device, dtype=torch.float32)
        dbeta = torch.zeros((B,H,T), device=q.device, dtype=torch.float32)
        dgamma = torch.zeros((B,H,T), device=q.device, dtype=torch.float32)
        grad_out_c = grad_out.contiguous()
        grid = (B*H*NUM_V_TILES,)
        bwd_kernel[grid](
            q,k,v,e,g,beta,gamma, grad_out_c, final,
            dq, dk, de, dg, dbeta, dgamma, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            e.stride(0), e.stride(1), e.stride(2), e.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            grad_out_c.stride(0), grad_out_c.stride(1), grad_out_c.stride(2), grad_out_c.stride(3),
            final.stride(0), final.stride(1), final.stride(2), final.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            de.stride(0), de.stride(1), de.stride(2), de.stride(3),
            dg.stride(0), dg.stride(1), dg.stride(2), dg.stride(3),
            dbeta.stride(0), dbeta.stride(1), dbeta.stride(2),
            dgamma.stride(0), dgamma.stride(1), dgamma.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            B,H,T,K,V,NUM_V_TILES, scale_f
        )
        # cast to input dtypes
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), de.to(e.dtype), dg.to(g.dtype), dbeta.to(beta.dtype), dgamma.to(gamma.dtype), None

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _Func.apply(q, k, v, e, g, beta, gamma, scale)
