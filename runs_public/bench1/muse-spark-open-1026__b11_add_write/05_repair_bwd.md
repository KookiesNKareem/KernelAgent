```python
import torch
import triton
import triton.language as tl

CHUNK = 32

@triton.jit
def fwd_ckpt_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr, ckpt_ptr,
    stride_qb, stride_qh, stride_qt, stride_qk,
    stride_kb, stride_kh, stride_kt, stride_kk,
    stride_vb, stride_vh, stride_vt, stride_vk,
    stride_eb, stride_eh, stride_et, stride_ek,
    stride_gb, stride_gh, stride_gt, stride_gk,
    stride_betab, stride_betah, stride_betat,
    stride_gammab, stride_gammah, stride_gammat,
    stride_ob, stride_oh, stride_ot, stride_ok,
    stride_cb, stride_ch, stride_cn, stride_ck, stride_cv,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NUM_V_TILES: tl.constexpr, SCALE: tl.constexpr, CHUNK_C: tl.constexpr, NUM_CKPTS: tl.constexpr,
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
    ckpt_base = ckpt_ptr + b * stride_cb + h * stride_ch

    # ckpt 0 = zeros
    if pid_v == 0:
        pass
    # store zero ckpt if needed - done by python zero init

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
        # checkpoint after CHUNK steps
        is_ckpt = (t + 1) % CHUNK_C == 0
        if is_ckpt:
            cidx = (t + 1) // CHUNK_C
            ckpt_c = ckpt_base + cidx * stride_cn
            tl.store(ckpt_c + offs_k0[:, None] * stride_ck + offs_v[None, :] * stride_cv, s0, mask=(offs_k0[:, None] < K) & (offs_v[None, :] < V))
            tl.store(ckpt_c + offs_k1[:, None] * stride_ck + offs_v[None, :] * stride_cv, s1, mask=(offs_k1[:, None] < K) & (offs_v[None, :] < V))
            tl.store(ckpt_c + offs_k2[:, None] * stride_ck + offs_v[None, :] * stride_cv, s2, mask=(offs_k2[:, None] < K) & (offs_v[None, :] < V))
            tl.store(ckpt_c + offs_k3[:, None] * stride_ck + offs_v[None, :] * stride_cv, s3, mask=(offs_k3[:, None] < K) & (offs_v[None, :] < V))
    # final ckpt if T not multiple
    rem = T % CHUNK_C
    if rem != 0:
        cidx = NUM_CKPTS - 1
        ckpt_c = ckpt_base + cidx * stride_cn
        tl.store(ckpt_c + offs_k0[:, None] * stride_ck + offs_v[None, :] * stride_cv, s0, mask=(offs_k0[:, None] < K) & (offs_v[None, :] < V))
        tl.store(ckpt_c + offs_k1[:, None] * stride_ck + offs_v[None, :] * stride_cv, s1, mask=(offs_k1[:, None] < K) & (offs_v[None, :] < V))
        tl.store(ckpt_c + offs_k2[:, None] * stride_ck + offs_v[None, :] * stride_cv, s2, mask=(offs_k2[:, None] < K) & (offs_v[None, :] < V))
        tl.store(ckpt_c + offs_k3[:, None] * stride_ck + offs_v[None, :] * stride_cv, s3, mask=(offs_k3[:, None] < K) & (offs_v[None, :] < V))


@triton.jit
def bwd_chunk_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, go_ptr,
    ckpt_ptr, scratch_ptr, gnext_ptr,
    dq_ptr, dk_ptr, de_ptr, dg_ptr, dbeta_ptr, dgamma_ptr, dv_ptr,
    stride_qb, stride_qh, stride_qt, stride_qk,
    stride_kb, stride_kh, stride_kt, stride_kk,
    stride_vb, stride_vh, stride_vt, stride_vk,
    stride_eb, stride_eh, stride_et, stride_ek,
    stride_gb, stride_gh, stride_gt, stride_gk,
    stride_betab, stride_betah, stride_betat,
    stride_gammab, stride_gammah, stride_gammat,
    stride_gob, stride_goh, stride_got, stride_gov,
    stride_cb, stride_ch, stride_cn, stride_ck, stride_cv,
    stride_sb, stride_sh, stride_sn, stride_sk, stride_sv,
    stride_gnb, stride_gnh, stride_gnk, stride_gnv,
    stride_dqb, stride_dqh, stride_dqt, stride_dqk,
    stride_dkb, stride_dkh, stride_dkt, stride_dkk,
    stride_deb, stride_deh, stride_det, stride_dek,
    stride_dgb, stride_dgh, stride_dgt, stride_dgk,
    stride_dbetab, stride_dbetah, stride_dbetat,
    stride_dgammab, stride_dgammah, stride_dgammat,
    stride_dvb, stride_dvh, stride_dvt, stride_dvv,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NUM_V_TILES: tl.constexpr, SCALE: tl.constexpr, CHUNK_C: tl.constexpr,
    L: tl.constexpr, CUR_LEN: tl.constexpr, CKPT_IDX: tl.constexpr,
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

    q_base = q_ptr + b * stride_qb + h * stride_qh
    k_base = k_ptr + b * stride_kb + h * stride_kh
    v_base = v_ptr + b * stride_vb + h * stride_vh
    e_base = e_ptr + b * stride_eb + h * stride_eh
    g_base = g_ptr + b * stride_gb + h * stride_gh
    beta_base = beta_ptr + b * stride_betab + h * stride_betah
    gamma_base = gamma_ptr + b * stride_gammab + h * stride_gammah
    go_base = go_ptr + b * stride_gob + h * stride_goh
    ckpt_base = ckpt_ptr + b * stride_cb + h * stride_ch + CKPT_IDX * stride_cn
    scratch_base = scratch_ptr + b * stride_sb + h * stride_sh
    gnext_base = gnext_ptr + b * stride_gnb + h * stride_gnh
    dq_base = dq_ptr + b * stride_dqb + h * stride_dqh
    dk_base = dk_ptr + b * stride_dkb + h * stride_dkh
    de_base = de_ptr + b * stride_deb + h * stride_deh
    dg_base = dg_ptr + b * stride_dgb + h * stride_dgh
    dv_base = dv_ptr + b * stride_dvb + h * stride_dvh
    dbeta_base = dbeta_ptr + b * stride_dbetab + h * stride_dbetah
    dgamma_base = dgamma_ptr + b * stride_dgammab + h * stride_dgammah

    # load start state
    s0 = tl.load(ckpt_base + offs_k0[:, None] * stride_ck + offs_v[None, :] * stride_cv, mask=(offs_k0[:, None] < K) & (offs_v[None, :] < V), other=0.0)
    s1 = tl.load(ckpt_base + offs_k1[:, None] * stride_ck + offs_v[None, :] * stride_cv, mask=(offs_k1[:, None] < K) & (offs_v[None, :] < V), other=0.0)
    s2 = tl.load(ckpt_base + offs_k2[:, None] * stride_ck + offs_v[None, :] * stride_cv, mask=(offs_k2[:, None] < K) & (offs_v[None, :] < V), other=0.0)
    s3 = tl.load(ckpt_base + offs_k3[:, None] * stride_ck + offs_v[None, :] * stride_cv, mask=(offs_k3[:, None] < K) & (offs_v[None, :] < V), other=0.0)
    # load gnext as cur
    cur0 = tl.load(gnext_base + offs_k0[:, None] * stride_gnk + offs_v[None, :] * stride_gnv, mask=(offs_k0[:, None] < K) & (offs_v[None, :] < V), other=0.0)
    cur1 = tl.load(gnext_base + offs_k1[:, None] * stride_gnk + offs_v[None, :] * stride_gnv, mask=(offs_k1[:, None] < K) & (offs_v[None, :] < V), other=0.0)
    cur2 = tl.load(gnext_base + offs_k2[:, None] * stride_gnk + offs_v[None, :] * stride_gnv, mask=(offs_k2[:, None] < K) & (offs_v[None, :] < V), other=0.0)
    cur3 = tl.load(gnext_base + offs_k3[:, None] * stride_gnk + offs_v[None, :] * stride_gnv, mask=(offs_k3[:, None] < K) & (offs_v[None, :] < V), other=0.0)

    # forward fill scratch
    sf0 = s0
    sf1 = s1
    sf2 = s2
    sf3 = s3
    for idx in range(CHUNK_C):
        if idx < CUR_LEN:
            t = L + idx
            k0 = tl.load(k_base + t * stride_kt + offs_k0 * stride_kk, mask=offs_k0 < K, other=0.0).to(tl.float32)
            k1 = tl.load(k_base + t * stride_kt + offs_k1 * stride_kk, mask=offs_k1 < K, other=0.0).to(tl.float32)
            k2 = tl.load(k_base + t * stride_kt + offs_k2 * stride_kk, mask=offs_k2 < K, other=0.0).to(tl.float32)
            k3 = tl.load(k_base + t * stride_kt + offs_k3 * stride_kk, mask=offs_k3 < K, other=0.0).to(tl.float32)
            e0 = tl.load(e_base + t * stride_et + offs_k0 * stride_ek, mask=offs_k0 < K, other=0.0).to(tl.float32)
            e1 = tl.load(e_base + t * stride_et + offs_k1 * stride_ek, mask=offs_k1 < K, other=0.0).to(tl.float32)
            e2 = tl.load(e_base + t * stride_et + offs_k2 * stride_ek, mask=offs_k2 < K, other=0.0).to(tl.float32)
            e3 = tl.load(e_base + t * stride_et + offs_k3 * stride_ek, mask=offs_k3 < K, other=0.0).to(tl.float32)
            g0 = tl.load(g_base + t * stride_gt + offs_k0 * stride_gk, mask=offs_k0 < K, other=0.0).to(tl.float32)
            g1 = tl.load(g_base + t * stride_gt + offs_k1 * stride_gk, mask=offs_k1 < K, other=0.0).to(tl.float32)
            g2 = tl.load(g_base + t * stride_gt + offs_k2 * stride_gk, mask=offs_k2 < K, other=0.0).to(tl.float32)
            g3 = tl.load(g_base + t * stride_gt + offs_k3 * stride_gk, mask=offs_k3 < K, other=0.0).to(tl.float32)
            vt = tl.load(v_base + t * stride_vt + offs_v * stride_vk, mask=mask_v, other=0.0).to(tl.float32)
            beta = tl.load(beta_base + t * stride_betat).to(tl.float32)
            gamma = tl.load(gamma_base + t * stride_gammat).to(tl.float32)
            d0 = tl.exp(g0); d1 = tl.exp(g1); d2 = tl.exp(g2); d3 = tl.exp(g3)
            sf0 = sf0 * d0[:, None]; sf1 = sf1 * d1[:, None]; sf2 = sf2 * d2[:, None]; sf3 = sf3 * d3[:, None]
            p0 = tl.sum(sf0 * k0[:, None], axis=0); p1 = tl.sum(sf1 * k1[:, None], axis=0); p2 = tl.sum(sf2 * k2[:, None], axis=0); p3 = tl.sum(sf3 * k3[:, None], axis=0)
            pred = p0 + p1 + p2 + p3
            delta = beta * (vt - pred)
            sf0 = sf0 + k0[:, None] * delta[None, :] + gamma * e0[:, None] * vt[None, :]
            sf1 = sf1 + k1[:, None] * delta[None, :] + gamma * e1[:, None] * vt[None, :]
            sf2 = sf2 + k2[:, None] * delta[None, :] + gamma * e2[:, None] * vt[None, :]
            sf3 = sf3 + k3[:, None] * delta[None, :] + gamma * e3[:, None] * vt[None, :]
            sp = scratch_base + idx * stride_sn
            tl.store(sp + offs_k0[:, None] * stride_sk + offs_v[None, :] * stride_sv, sf0, mask=(offs_k0[:, None] < K) & (offs_v[None, :] < V))
            tl.store(sp + offs_k1[:, None] * stride_sk + offs_v[None, :] * stride_sv, sf1, mask=(offs_k1[:, None] < K) & (offs_v[None, :] < V))
            tl.store(sp + offs_k2[:, None] * stride_sk + offs_v[None, :] * stride_sv, sf2, mask=(offs_k2[:, None] < K) & (offs_v[None, :] < V))
            tl.store(sp + offs_k3[:, None] * stride_sk + offs_v[None, :] * stride_sv, sf3, mask=(offs_k3[:, None] < K) & (offs_v[None, :] < V))

    # backward sweep
    for idx in range(CHUNK_C):
        rev = CUR_LEN - 1 - idx
        if rev >= 0:
            t = L + rev
            k0 = tl.load(k_base + t * stride_kt + offs_k0 * stride_kk, mask=offs_k0 < K, other=0.0).to(tl.float32)
            k1 = tl.load(k_base + t * stride_kt + offs_k1 * stride_kk, mask=offs_k1 < K, other=0.0).to(tl.float32)
            k2 = tl.load(k_base + t * stride_kt + offs_k2 * stride_kk, mask=offs_k2 < K, other=0.0).to(tl.float32)
            k3 = tl.load(k_base + t * stride_kt + offs_k3 * stride_kk, mask=offs_k3 < K, other=0.0).to(tl.float32)
            q0 = tl.load(q_base + t * stride_qt + offs_k0 * stride_qk, mask=offs_k0 < K, other=0.0).to(tl.float32)
            q1 = tl.load(q_base + t * stride_qt + offs_k1 * stride_qk, mask=offs_k1 < K, other=0.0).to(tl.float32)
            q2 = tl.load(q_base + t * stride_qt + offs_k2 * stride_qk, mask=offs_k2 < K, other=0.0).to(tl.float32)
            q3 = tl.load(q_base + t * stride_qt + offs_k3 * stride_qk, mask=offs_k3 < K, other=0.0).to(tl.float32)
            e0 = tl.load(e_base + t * stride_et + offs_k0 * stride_ek, mask=offs_k0 < K, other=0.0).to(tl.float32)
            e1 = tl.load(e_base + t * stride_et + offs_k1 * stride_ek, mask=offs_k1 < K, other=0.0).to(tl.float32)
            e2 = tl.load(e_base + t * stride_et + offs_k2 * stride_ek, mask=offs_k2 < K, other=0.0).to(tl.float32)
            e3 = tl.load(e_base + t * stride_et + offs_k3 * stride_ek, mask=offs_k3 < K, other=0.0).to(tl.float32)
            g0 = tl.load(g_base + t * stride_gt + offs_k0 * stride_gk, mask=offs_k0 < K, other=0.0).to(tl.float32)
            g1 = tl.load(g_base + t * stride_gt + offs_k1 * stride_gk, mask=offs_k1 < K, other=0.0).to(tl.float32)
            g2 = tl.load(g_base + t * stride_gt + offs_k2 * stride_gk, mask=offs_k2 < K, other=0.0).to(tl.float32)
            g3 = tl.load(g_base + t * stride_gt + offs_k3 * stride_gk, mask=offs_k3 < K, other=0.0).to(tl.float32)
            vt = tl.load(v_base + t * stride_vt + offs_v * stride_vk, mask=mask_v, other=0.0).to(tl.float32)
            got = tl.load(go_base + t * stride_got + offs_v * stride_gov, mask=mask_v, other=0.0).to(tl.float32)
            beta = tl.load(beta_base + t * stride_betat).to(tl.float32)
            gamma = tl.load(gamma_base + t * stride_gammat).to(tl.float32)

            sp = scratch_base + rev * stride_sn
            st0 = tl.load(sp + offs_k0[:, None] * stride_sk + offs_v[None, :] * stride_sv, mask=(offs_k0[:, None] < K) & (offs_v[None, :] < V), other=0.0)
            st1 = tl.load(sp + offs_k1[:, None] * stride_sk + offs_v[None, :] * stride_sv, mask=(offs_k1[:, None] < K) & (offs_v[None, :] < V), other=0.0)
            st2 = tl.load(sp + offs_k2[:, None] * stride_sk + offs_v[None, :] * stride_sv, mask=(offs_k2[:, None] < K) & (offs_v[None, :] < V), other=0.0)
            st3 = tl.load(sp + offs_k3[:, None] * stride_sk + offs_v[None, :] * stride_sv, mask=(offs_k3[:, None] < K) & (offs_v[None, :] < V), other=0.0)
            # S_prev
            if rev == 0:
                sprev0 = s0; sprev1 = s1; sprev2 = s2; sprev3 = s3
            else:
                sp2 = scratch_base + (rev - 1) * stride_sn
                sprev0 = tl.load(sp2 + offs_k0[:, None] * stride_sk + offs_v[None, :] * stride_sv, mask=(offs_k0[:, None] < K) & (offs_v[None, :] < V), other=0.0)
                sprev1 = tl.load(sp2 + offs_k1[:, None] * stride_sk + offs_v[None, :] * stride_sv, mask=(offs_k1[:, None] < K) & (offs_v[None, :] < V), other=0.0)
                sprev2 = tl.load(sp2 + offs_k2[:, None] * stride_sk + offs_v[None, :] * stride_sv, mask=(offs_k2[:, None] < K) & (offs_v[None, :] < V), other=0.0)
                sprev3 = tl.load(sp2 + offs_k3[:, None] * stride_sk + offs_v[None, :] * stride_sv, mask=(offs_k3[:, None] < K) & (offs_v[None, :] < V), other=0.0)

            # G_t = QGO + cur
            gt0 = SCALE * q0[:, None] * got[None, :] + cur0
            gt1 = SCALE * q1[:, None] * got[None, :] + cur1
            gt2 = SCALE * q2[:, None] * got[None, :] + cur2
            gt3 = SCALE * q3[:, None] * got[None, :] + cur3

            # a = G^T k, per V
            a0 = tl.sum(gt0 * k0[:, None], axis=0); a1 = tl.sum(gt1 * k1[:, None], axis=0); a2 = tl.sum(gt2 * k2[:, None], axis=0); a3 = tl.sum(gt3 * k3[:, None], axis=0)
            a = a0 + a1 + a2 + a3

            # y = D * Sprev for pred and other
            d0 = tl.exp(g0); d1 = tl.exp(g1); d2 = tl.exp(g2); d3 = tl.exp(g3)
            y0 = sprev0 * d0[:, None]; y1 = sprev1 * d1[:, None]; y2 = sprev2 * d2[:, None]; y3 = sprev3 * d3[:, None]
            p0 = tl.sum(y0 * k0[:, None], axis=0); p1 = tl.sum(y1 * k1[:, None], axis=0); p2 = tl.sum(y2 * k2[:, None], axis=0); p3 = tl.sum(y3 * k3[:, None], axis=0)
            pred = p0 + p1 + p2 + p3
            delta = vt - pred

            # dk, dq etc use G_t
            # dq = SCALE * S_t * go  (sum over V)
            dq0 = tl.sum(st0 * got[None, :], axis=1) * SCALE
            dq1 = tl.sum(st1 * got[None, :], axis=1) * SCALE
            dq2 = tl.sum(st2 * got[None, :], axis=1) * SCALE
            dq3 = tl.sum(st3 * got[None, :], axis=1) * SCALE
            # de = gamma * G v
            de0 = tl.sum(gt0 * vt[None, :], axis=1) * gamma
            de1 = tl.sum(gt1 * vt[None, :], axis=1) * gamma
            de2 = tl.sum(gt2 * vt[None, :], axis=1) * gamma
            de3 = tl.sum(gt3 * vt[None, :], axis=1) * gamma
            # Gv for dk parts
            gv0 = tl.sum(gt0 * vt[None, :], axis=1)
            gv1 = tl.sum(gt1 * vt[None, :], axis=1)
            gv2 = tl.sum(gt2 * vt[None, :], axis=1)
            gv3 = tl.sum(gt3 * vt[None, :], axis=1)
            # G pred
            gp0 = tl.sum(gt0 * pred[None, :], axis=1)
            gp1 = tl.sum(gt1 * pred[None, :], axis=1)
            gp2 = tl.sum(gt2 * pred[None, :], axis=1)
            gp3 = tl.sum(gt3 * pred[None, :], axis=1)
            # y a
            ya0 = tl.sum(y0 * a[None, :], axis=1)
            ya1 = tl.sum(y1 * a[None, :], axis=1)
            ya2 = tl.sum(y2 * a[None, :], axis=1)
            ya3 = tl.sum(y3 * a[None, :], axis=1)
            dk0 = beta * (gv0 - gp0 - ya0)
            dk1 = beta * (gv1 - gp1 - ya1)
            dk2 = beta * (gv2 - gp2 - ya2)
            dk3 = beta * (gv3 - gp3 - ya3)

            # dg = D * sum( AG * Sprev ) ; AG = G - beta k a^T
            ag0 = gt0 - beta * k0[:, None] * a[None, :]
            ag1 = gt1 - beta * k1[:, None] * a[None, :]
            ag2 = gt2 - beta * k2[:, None] * a[None, :]
            ag3 = gt3 - beta * k3[:, None] * a[None, :]
            dg0 = d0 * tl.sum(ag0 * sprev0, axis=1)
            dg1 = d1 * tl.sum(ag1 * sprev1, axis=1)
            dg2 = d2 * tl.sum(ag2 * sprev2, axis=1)
            dg3 = d3 * tl.sum(ag3 * sprev3, axis=1)

            # dv = G^T w ; w = beta k + gamma e
            w0 = beta * k0 + gamma * e0
            w1 = beta * k1 + gamma * e1
            w2 = beta * k2 + gamma * e2
            w3 = beta * k3 + gamma * e3
            dv_part = tl.sum(gt0 * w0[:, None], axis=0) + tl.sum(gt1 * w1[:, None], axis=0) + tl.sum(gt2 * w2[:, None], axis=0) + tl.sum(gt3 * w3[:, None], axis=0)
            tl.store(dv_base + t * stride_dvt + offs_v * stride_dvv, dv_part, mask=mask_v)

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

            beta_part = tl.sum(a * delta)
            # for dgamma: G^T e dot v
            gte0 = tl.sum(gt0 * e0[:, None], axis=0); gte1 = tl.sum(gt1 * e1[:, None], axis=0); gte2 = tl.sum(gt2 * e2[:, None], axis=0); gte3 = tl.sum(gt3 * e3[:, None], axis=0)
            gte = gte0 + gte1 + gte2 + gte3
            gamma_part = tl.sum(gte * vt)
            tl.atomic_add(dbeta_base + t * stride_dbetat, beta_part)
            tl.atomic_add(dgamma_base + t * stride_dgammat, gamma_part)

            # propagate cur = D * (G - beta k a)
            cur0 = d0[:, None] * ag0
            cur1 = d1[:, None] * ag1
            cur2 = d2[:, None] * ag2
            cur3 = d3[:, None] * ag3

    # write back cur for next chunk
    tl.store(gnext_base + offs_k0[:, None] * stride_gnk + offs_v[None, :] * stride_gnv, cur0, mask=(offs_k0[:, None] < K) & (offs_v[None, :] < V))
    tl.store(gnext_base + offs_k1[:, None] * stride_gnk + offs_v[None, :] * stride_gnv, cur1, mask=(offs_k1[:, None] < K) & (offs_v[None, :] < V))
    tl.store(gnext_base + offs_k2[:, None] * stride_gnk + offs_v[None, :] * stride_gnv, cur2, mask=(offs_k2[:, None] < K) & (offs_v[None, :] < V))
    tl.store(gnext_base + offs_k3[:, None] * stride_gnk + offs_v[None, :] * stride_gnv, cur3, mask=(offs_k3[:, None] < K) & (offs_v[None, :] < V))


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        Nc = (T + CHUNK - 1)//CHUNK
        ckpt = torch.zeros((B, H, Nc+1, K, V), device=q.device, dtype=torch.float32)
        BLOCK_V = 32
        NUM_V_TILES = (V + BLOCK_V - 1)//BLOCK_V
        grid = (B*H*NUM_V_TILES,)
        scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        fwd_ckpt_kernel[grid](
            q, k, v, e, g, beta, gamma, out, ckpt,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            e.stride(0), e.stride(1), e.stride(2), e.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            ckpt.stride(0), ckpt.stride(1), ckpt.stride(2), ckpt.stride(3), ckpt.stride(4),
            B, H, T, K, V, NUM_V_TILES, scale_f, CHUNK, Nc+1
        )
        ctx.save_for_backward(q,k,v,e,g,beta,gamma,ckpt)
        ctx.scale_f = scale_f
        ctx.dims = (B,H,T,K,V,NUM_V_TILES,Nc)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q,k,v,e,g,beta,gamma,ckpt = ctx.saved_tensors
        scale_f = ctx.scale_f
        B,H,T,K,V,NUM_V_TILES,Nc = ctx.dims
        dq = torch.zeros((B,H,T,K), device=q.device, dtype=torch.float32)
        dk = torch.zeros((B,H,T,K), device=q.device, dtype=torch.float32)
        de = torch.zeros((B,H,T,K), device=q.device, dtype=torch.float32)
        dg = torch.zeros((B,H,T,K), device=q.device, dtype=torch.float32)
        dv = torch.zeros((B,H,T,V), device=q.device, dtype=torch.float32)
        dbeta = torch.zeros((B,H,T), device=q.device, dtype=torch.float32)
        dgamma = torch.zeros((B,H,T), device=q.device, dtype=torch.float32)
        gnext = torch.zeros((B,H,K,V), device=q.device, dtype=torch.float32)
        scratch = torch.empty((B,H,CHUNK,K,V), device=q.device, dtype=torch.float32)
        grad_out_c = grad_out.contiguous()
        grid = (B*H*NUM_V_TILES,)
        for c in range(Nc-1, -1, -1):
            L = c * CHUNK
            cur_len = T - L if c == Nc-1 else CHUNK
            if cur_len <= 0:
                continue
            bwd_chunk_kernel[grid](
                q,k,v,e,g,beta,gamma, grad_out_c,
                ckpt, scratch, gnext,
                dq, dk, de, dg, dbeta, dgamma, dv,
                q.stride(0),q.stride(1),q.stride(2),q.stride(3),
                k.stride(0),k.stride(1),k.stride(2),k.stride(3),
                v.stride(0),v.stride(1),v.stride(2),v.stride(3),
                e.stride(0),e.stride(1),e.stride(2),e.stride(3),
                g.stride(0),g.stride(1),g.stride(2),g.stride(3),
                beta.stride(0),beta.stride(1),beta.stride(2),
                gamma.stride(0),gamma.stride(1),gamma.stride(2),
                grad_out_c.stride(0),grad_out_c.stride(1),grad_out_c.stride(2),grad_out_c.stride(3),
                ckpt.stride(0),ckpt.stride(1),ckpt.stride(2),ckpt.stride(3),ckpt.stride(4),
                scratch.stride(0),scratch.stride(1),scratch.stride(2),scratch.stride(3),scratch.stride(4),
                gnext.stride(0),gnext.stride(1),gnext.stride(2),gnext.stride(3),
                dq.stride(0),dq.stride(1),dq.stride(2),dq.stride(3),
                dk.stride(0),dk.stride(1),dk.stride(2),dk.stride(3),
                de.stride(0),de.stride(1),de.stride(2),de.stride(3),
                dg.stride(0),dg.stride(1),dg.stride(2),dg.stride(3),
                dbeta.stride(0),dbeta.stride(1),dbeta.stride(2),
                dgamma.stride(0),dgamma.stride(1),dgamma.stride(2),
                dv.stride(0),dv.stride(1),dv.stride(2),dv.stride(3),
                B,H,T,K,V,NUM_V_TILES, scale_f, CHUNK, L, cur_len, c
            )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), de.to(e.dtype), dg.to(g.dtype), dbeta.to(beta.dtype), dgamma.to(gamma.dtype), None

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _Func.apply(q, k, v, e, g, beta, gamma, scale)
```
Previous failure used inverse `exp(-g)` and `1/(1-beta||k||²)` giving huge `exp(25)` overflow and division by zero for non-finite grads. Fix is stable checkpointed adjoint: forward saves `S` every 32 steps, backward recomputes a chunk forward without inverses and propagates `G_{t-1}=D_t(G_t-beta k_t(G_tᵀk_t)ᵀ)` (`D=diag(exp g)≤1`). `V` is tiled 32-wide, `K=128` as 4×32 blocks in registers; per-tile partials for `dq,dk,de,dg,dbeta,dgamma` that sum over `V` are reduced via `atomic_add`.

Unsure of scratch `CHUNK*K*V` memory and loop-carried `tl` control flow compilation on Triton3.5 sm89.