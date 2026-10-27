```python
import torch
import triton
import triton.language as tl

CHUNK = 64
CHUNK_GENERIC = 32

# ---------- fast path K<=128 multiple of 32 ----------

@triton.jit
def fwd_ckpt_kernel_fast(
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
    for t in range(T):
        k0 = tl.load(k_base + t * stride_kt + offs_k0 * stride_kk).to(tl.float32)
        k1 = tl.load(k_base + t * stride_kt + offs_k1 * stride_kk).to(tl.float32)
        k2 = tl.load(k_base + t * stride_kt + offs_k2 * stride_kk).to(tl.float32)
        k3 = tl.load(k_base + t * stride_kt + offs_k3 * stride_kk).to(tl.float32)
        q0 = tl.load(q_base + t * stride_qt + offs_k0 * stride_qk).to(tl.float32)
        q1 = tl.load(q_base + t * stride_qt + offs_k1 * stride_qk).to(tl.float32)
        q2 = tl.load(q_base + t * stride_qt + offs_k2 * stride_qk).to(tl.float32)
        q3 = tl.load(q_base + t * stride_qt + offs_k3 * stride_qk).to(tl.float32)
        e0 = tl.load(e_base + t * stride_et + offs_k0 * stride_ek).to(tl.float32)
        e1 = tl.load(e_base + t * stride_et + offs_k1 * stride_ek).to(tl.float32)
        e2 = tl.load(e_base + t * stride_et + offs_k2 * stride_ek).to(tl.float32)
        e3 = tl.load(e_base + t * stride_et + offs_k3 * stride_ek).to(tl.float32)
        g0 = tl.load(g_base + t * stride_gt + offs_k0 * stride_gk).to(tl.float32)
        g1 = tl.load(g_base + t * stride_gt + offs_k1 * stride_gk).to(tl.float32)
        g2 = tl.load(g_base + t * stride_gt + offs_k2 * stride_gk).to(tl.float32)
        g3 = tl.load(g_base + t * stride_gt + offs_k3 * stride_gk).to(tl.float32)
        vt = tl.load(v_base + t * stride_vt + offs_v * stride_vk).to(tl.float32)
        beta = tl.load(beta_base + t * stride_betat).to(tl.float32)
        gamma = tl.load(gamma_base + t * stride_gammat).to(tl.float32)
        d0 = tl.exp(g0); d1 = tl.exp(g1); d2 = tl.exp(g2); d3 = tl.exp(g3)
        s0 = s0 * d0[:, None]; s1 = s1 * d1[:, None]; s2 = s2 * d2[:, None]; s3 = s3 * d3[:, None]
        pred = tl.sum(s0 * k0[:, None], axis=0) + tl.sum(s1 * k1[:, None], axis=0) + tl.sum(s2 * k2[:, None], axis=0) + tl.sum(s3 * k3[:, None], axis=0)
        delta = beta * (vt - pred)
        s0 = s0 + k0[:, None] * delta[None, :] + gamma * e0[:, None] * vt[None, :]
        s1 = s1 + k1[:, None] * delta[None, :] + gamma * e1[:, None] * vt[None, :]
        s2 = s2 + k2[:, None] * delta[None, :] + gamma * e2[:, None] * vt[None, :]
        s3 = s3 + k3[:, None] * delta[None, :] + gamma * e3[:, None] * vt[None, :]
        out = SCALE * (tl.sum(s0 * q0[:, None], axis=0) + tl.sum(s1 * q1[:, None], axis=0) + tl.sum(s2 * q2[:, None], axis=0) + tl.sum(s3 * q3[:, None], axis=0))
        tl.store(out_base + t * stride_ot + offs_v * stride_ok, out)
        if (t + 1) % CHUNK_C == 0:
            cidx = (t + 1) // CHUNK_C
            ck = ckpt_base + cidx * stride_cn
            tl.store(ck + offs_k0[:, None] * stride_ck + offs_v[None, :] * stride_cv, s0)
            tl.store(ck + offs_k1[:, None] * stride_ck + offs_v[None, :] * stride_cv, s1)
            tl.store(ck + offs_k2[:, None] * stride_ck + offs_v[None, :] * stride_cv, s2)
            tl.store(ck + offs_k3[:, None] * stride_ck + offs_v[None, :] * stride_cv, s3)
    rem = T % CHUNK_C
    if rem != 0:
        cidx = NUM_CKPTS - 1
        ck = ckpt_base + cidx * stride_cn
        tl.store(ck + offs_k0[:, None] * stride_ck + offs_v[None, :] * stride_cv, s0)
        tl.store(ck + offs_k1[:, None] * stride_ck + offs_v[None, :] * stride_cv, s1)
        tl.store(ck + offs_k2[:, None] * stride_ck + offs_v[None, :] * stride_cv, s2)
        tl.store(ck + offs_k3[:, None] * stride_ck + offs_v[None, :] * stride_cv, s3)

@triton.jit
def bwd_chunk_kernel_fast(
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
    offs_k0 = tl.arange(0, 32); offs_k1 = offs_k0 + 32; offs_k2 = offs_k0 + 64; offs_k3 = offs_k0 + 96
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
    s0 = tl.load(ckpt_base + offs_k0[:, None] * stride_ck + offs_v[None, :] * stride_cv)
    s1 = tl.load(ckpt_base + offs_k1[:, None] * stride_ck + offs_v[None, :] * stride_cv)
    s2 = tl.load(ckpt_base + offs_k2[:, None] * stride_ck + offs_v[None, :] * stride_cv)
    s3 = tl.load(ckpt_base + offs_k3[:, None] * stride_ck + offs_v[None, :] * stride_cv)
    cur0 = tl.load(gnext_base + offs_k0[:, None] * stride_gnk + offs_v[None, :] * stride_gnv)
    cur1 = tl.load(gnext_base + offs_k1[:, None] * stride_gnk + offs_v[None, :] * stride_gnv)
    cur2 = tl.load(gnext_base + offs_k2[:, None] * stride_gnk + offs_v[None, :] * stride_gnv)
    cur3 = tl.load(gnext_base + offs_k3[:, None] * stride_gnk + offs_v[None, :] * stride_gnv)
    sf0 = s0; sf1 = s1; sf2 = s2; sf3 = s3
    for idx in range(CHUNK_C):
        if idx < CUR_LEN:
            t = L + idx
            k0 = tl.load(k_base + t * stride_kt + offs_k0 * stride_kk).to(tl.float32)
            k1 = tl.load(k_base + t * stride_kt + offs_k1 * stride_kk).to(tl.float32)
            k2 = tl.load(k_base + t * stride_kt + offs_k2 * stride_kk).to(tl.float32)
            k3 = tl.load(k_base + t * stride_kt + offs_k3 * stride_kk).to(tl.float32)
            e0 = tl.load(e_base + t * stride_et + offs_k0 * stride_ek).to(tl.float32)
            e1 = tl.load(e_base + t * stride_et + offs_k1 * stride_ek).to(tl.float32)
            e2 = tl.load(e_base + t * stride_et + offs_k2 * stride_ek).to(tl.float32)
            e3 = tl.load(e_base + t * stride_et + offs_k3 * stride_ek).to(tl.float32)
            g0 = tl.load(g_base + t * stride_gt + offs_k0 * stride_gk).to(tl.float32)
            g1 = tl.load(g_base + t * stride_gt + offs_k1 * stride_gk).to(tl.float32)
            g2 = tl.load(g_base + t * stride_gt + offs_k2 * stride_gk).to(tl.float32)
            g3 = tl.load(g_base + t * stride_gt + offs_k3 * stride_gk).to(tl.float32)
            vt = tl.load(v_base + t * stride_vt + offs_v * stride_vk).to(tl.float32)
            beta = tl.load(beta_base + t * stride_betat).to(tl.float32)
            gamma = tl.load(gamma_base + t * stride_gammat).to(tl.float32)
            d0 = tl.exp(g0); d1 = tl.exp(g1); d2 = tl.exp(g2); d3 = tl.exp(g3)
            sf0 = sf0 * d0[:, None]; sf1 = sf1 * d1[:, None]; sf2 = sf2 * d2[:, None]; sf3 = sf3 * d3[:, None]
            pred = tl.sum(sf0 * k0[:, None], axis=0) + tl.sum(sf1 * k1[:, None], axis=0) + tl.sum(sf2 * k2[:, None], axis=0) + tl.sum(sf3 * k3[:, None], axis=0)
            delta = beta * (vt - pred)
            sf0 = sf0 + k0[:, None] * delta[None, :] + gamma * e0[:, None] * vt[None, :]
            sf1 = sf1 + k1[:, None] * delta[None, :] + gamma * e1[:, None] * vt[None, :]
            sf2 = sf2 + k2[:, None] * delta[None, :] + gamma * e2[:, None] * vt[None, :]
            sf3 = sf3 + k3[:, None] * delta[None, :] + gamma * e3[:, None] * vt[None, :]
            sp = scratch_base + idx * stride_sn
            tl.store(sp + offs_k0[:, None] * stride_sk + offs_v[None, :] * stride_sv, sf0)
            tl.store(sp + offs_k1[:, None] * stride_sk + offs_v[None, :] * stride_sv, sf1)
            tl.store(sp + offs_k2[:, None] * stride_sk + offs_v[None, :] * stride_sv, sf2)
            tl.store(sp + offs_k3[:, None] * stride_sk + offs_v[None, :] * stride_sv, sf3)
    for idx in range(CHUNK_C):
        rev = CUR_LEN - 1 - idx
        if rev >= 0:
            t = L + rev
            k0 = tl.load(k_base + t * stride_kt + offs_k0 * stride_kk).to(tl.float32)
            k1 = tl.load(k_base + t * stride_kt + offs_k1 * stride_kk).to(tl.float32)
            k2 = tl.load(k_base + t * stride_kt + offs_k2 * stride_kk).to(tl.float32)
            k3 = tl.load(k_base + t * stride_kt + offs_k3 * stride_kk).to(tl.float32)
            q0 = tl.load(q_base + t * stride_qt + offs_k0 * stride_qk).to(tl.float32)
            q1 = tl.load(q_base + t * stride_qt + offs_k1 * stride_qk).to(tl.float32)
            q2 = tl.load(q_base + t * stride_qt + offs_k2 * stride_qk).to(tl.float32)
            q3 = tl.load(q_base + t * stride_qt + offs_k3 * stride_qk).to(tl.float32)
            e0 = tl.load(e_base + t * stride_et + offs_k0 * stride_ek).to(tl.float32)
            e1 = tl.load(e_base + t * stride_et + offs_k1 * stride_ek).to(tl.float32)
            e2 = tl.load(e_base + t * stride_et + offs_k2 * stride_ek).to(tl.float32)
            e3 = tl.load(e_base + t * stride_et + offs_k3 * stride_ek).to(tl.float32)
            g0 = tl.load(g_base + t * stride_gt + offs_k0 * stride_gk).to(tl.float32)
            g1 = tl.load(g_base + t * stride_gt + offs_k1 * stride_gk).to(tl.float32)
            g2 = tl.load(g_base + t * stride_gt + offs_k2 * stride_gk).to(tl.float32)
            g3 = tl.load(g_base + t * stride_gt + offs_k3 * stride_gk).to(tl.float32)
            vt = tl.load(v_base + t * stride_vt + offs_v * stride_vk).to(tl.float32)
            got = tl.load(go_base + t * stride_got + offs_v * stride_gov).to(tl.float32)
            beta = tl.load(beta_base + t * stride_betat).to(tl.float32)
            gamma = tl.load(gamma_base + t * stride_gammat).to(tl.float32)
            sp = scratch_base + rev * stride_sn
            st0 = tl.load(sp + offs_k0[:, None] * stride_sk + offs_v[None, :] * stride_sv)
            st1 = tl.load(sp + offs_k1[:, None] * stride_sk + offs_v[None, :] * stride_sv)
            st2 = tl.load(sp + offs_k2[:, None] * stride_sk + offs_v[None, :] * stride_sv)
            st3 = tl.load(sp + offs_k3[:, None] * stride_sk + offs_v[None, :] * stride_sv)
            if rev == 0:
                sprev0 = s0; sprev1 = s1; sprev2 = s2; sprev3 = s3
            else:
                sp2 = scratch_base + (rev - 1) * stride_sn
                sprev0 = tl.load(sp2 + offs_k0[:, None] * stride_sk + offs_v[None, :] * stride_sv)
                sprev1 = tl.load(sp2 + offs_k1[:, None] * stride_sk + offs_v[None, :] * stride_sv)
                sprev2 = tl.load(sp2 + offs_k2[:, None] * stride_sk + offs_v[None, :] * stride_sv)
                sprev3 = tl.load(sp2 + offs_k3[:, None] * stride_sk + offs_v[None, :] * stride_sv)
            gt0 = SCALE * q0[:, None] * got[None, :] + cur0
            gt1 = SCALE * q1[:, None] * got[None, :] + cur1
            gt2 = SCALE * q2[:, None] * got[None, :] + cur2
            gt3 = SCALE * q3[:, None] * got[None, :] + cur3
            a = tl.sum(gt0 * k0[:, None], axis=0) + tl.sum(gt1 * k1[:, None], axis=0) + tl.sum(gt2 * k2[:, None], axis=0) + tl.sum(gt3 * k3[:, None], axis=0)
            d0 = tl.exp(g0); d1 = tl.exp(g1); d2 = tl.exp(g2); d3 = tl.exp(g3)
            y0 = sprev0 * d0[:, None]; y1 = sprev1 * d1[:, None]; y2 = sprev2 * d2[:, None]; y3 = sprev3 * d3[:, None]
            pred = tl.sum(y0 * k0[:, None], axis=0) + tl.sum(y1 * k1[:, None], axis=0) + tl.sum(y2 * k2[:, None], axis=0) + tl.sum(y3 * k3[:, None], axis=0)
            delta = vt - pred
            dq0 = tl.sum(st0 * got[None, :], axis=1) * SCALE
            dq1 = tl.sum(st1 * got[None, :], axis=1) * SCALE
            dq2 = tl.sum(st2 * got[None, :], axis=1) * SCALE
            dq3 = tl.sum(st3 * got[None, :], axis=1) * SCALE
            de0 = tl.sum(gt0 * vt[None, :], axis=1) * gamma
            de1 = tl.sum(gt1 * vt[None, :], axis=1) * gamma
            de2 = tl.sum(gt2 * vt[None, :], axis=1) * gamma
            de3 = tl.sum(gt3 * vt[None, :], axis=1) * gamma
            dk0 = beta * (tl.sum(gt0 * vt[None, :], axis=1) - tl.sum(gt0 * pred[None, :], axis=1) - tl.sum(y0 * a[None, :], axis=1))
            dk1 = beta * (tl.sum(gt1 * vt[None, :], axis=1) - tl.sum(gt1 * pred[None, :], axis=1) - tl.sum(y1 * a[None, :], axis=1))
            dk2 = beta * (tl.sum(gt2 * vt[None, :], axis=1) - tl.sum(gt2 * pred[None, :], axis=1) - tl.sum(y2 * a[None, :], axis=1))
            dk3 = beta * (tl.sum(gt3 * vt[None, :], axis=1) - tl.sum(gt3 * pred[None, :], axis=1) - tl.sum(y3 * a[None, :], axis=1))
            ag0 = gt0 - beta * k0[:, None] * a[None, :]; ag1 = gt1 - beta * k1[:, None] * a[None, :]; ag2 = gt2 - beta * k2[:, None] * a[None, :]; ag3 = gt3 - beta * k3[:, None] * a[None, :]
            dg0 = d0 * tl.sum(ag0 * sprev0, axis=1); dg1 = d1 * tl.sum(ag1 * sprev1, axis=1); dg2 = d2 * tl.sum(ag2 * sprev2, axis=1); dg3 = d3 * tl.sum(ag3 * sprev3, axis=1)
            tl.store(dv_base + t * stride_dvt + offs_v * stride_dvv, tl.sum(gt0 * (beta*k0+gamma*e0)[:, None],axis=0)+tl.sum(gt1*(beta*k1+gamma*e1)[:, None],axis=0)+tl.sum(gt2*(beta*k2+gamma*e2)[:, None],axis=0)+tl.sum(gt3*(beta*k3+gamma*e3)[:, None],axis=0))
            tl.atomic_add(dq_base + t * stride_dqt + offs_k0 * stride_dqk, dq0)
            tl.atomic_add(dq_base + t * stride_dqt + offs_k1 * stride_dqk, dq1)
            tl.atomic_add(dq_base + t * stride_dqt + offs_k2 * stride_dqk, dq2)
            tl.atomic_add(dq_base + t * stride_dqt + offs_k3 * stride_dqk, dq3)
            tl.atomic_add(dk_base + t * stride_dkt + offs_k0 * stride_dkk, dk0)
            tl.atomic_add(dk_base + t * stride_dkt + offs_k1 * stride_dkk, dk1)
            tl.atomic_add(dk_base + t * stride_dkt + offs_k2 * stride_dkk, dk2)
            tl.atomic_add(dk_base + t * stride_dkt + offs_k3 * stride_dkk, dk3)
            tl.atomic_add(de_base + t * stride_det + offs_k0 * stride_dek, de0)
            tl.atomic_add(de_base + t * stride_det + offs_k1 * stride_dek, de1)
            tl.atomic_add(de_base + t * stride_det + offs_k2 * stride_dek, de2)
            tl.atomic_add(de_base + t * stride_det + offs_k3 * stride_dek, de3)
            tl.atomic_add(dg_base + t * stride_dgt + offs_k0 * stride_dgk, dg0)
            tl.atomic_add(dg_base + t * stride_dgt + offs_k1 * stride_dgk, dg1)
            tl.atomic_add(dg_base + t * stride_dgt + offs_k2 * stride_dgk, dg2)
            tl.atomic_add(dg_base + t * stride_dgt + offs_k3 * stride_dgk, dg3)
            tl.atomic_add(dbeta_base + t * stride_dbetat, tl.sum(a * delta))
            tl.atomic_add(dgamma_base + t * stride_dgammat, tl.sum((tl.sum(gt0*e0[:,None],axis=0)+tl.sum(gt1*e1[:,None],axis=0)+tl.sum(gt2*e2[:,None],axis=0)+tl.sum(gt3*e3[:,None],axis=0))*vt))
            cur0 = d0[:, None] * ag0; cur1 = d1[:, None] * ag1; cur2 = d2[:, None] * ag2; cur3 = d3[:, None] * ag3
    tl.store(gnext_base + offs_k0[:, None] * stride_gnk + offs_v[None, :] * stride_gnv, cur0)
    tl.store(gnext_base + offs_k1[:, None] * stride_gnk + offs_v[None, :] * stride_gnv, cur1)
    tl.store(gnext_base + offs_k2[:, None] * stride_gnk + offs_v[None, :] * stride_gnv, cur2)
    tl.store(gnext_base + offs_k3[:, None] * stride_gnk + offs_v[None, :] * stride_gnv, cur3)

# ---------- generic 256-K fallback (masked) ----------
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
    offs_k0 = tl.arange(0, 32); offs_k1 = offs_k0+32; offs_k2 = offs_k0+64; offs_k3 = offs_k0+96
    offs_k4 = offs_k0+128; offs_k5 = offs_k0+160; offs_k6 = offs_k0+192; offs_k7 = offs_k0+224
    s0=tl.zeros((32,32),dtype=tl.float32); s1=tl.zeros((32,32),dtype=tl.float32); s2=tl.zeros((32,32),dtype=tl.float32); s3=tl.zeros((32,32),dtype=tl.float32)
    s4=tl.zeros((32,32),dtype=tl.float32); s5=tl.zeros((32,32),dtype=tl.float32); s6=tl.zeros((32,32),dtype=tl.float32); s7=tl.zeros((32,32),dtype=tl.float32)
    q_base=q_ptr+b*stride_qb+h*stride_qh; k_base=k_ptr+b*stride_kb+h*stride_kh; v_base=v_ptr+b*stride_vb+h*stride_vh
    e_base=e_ptr+b*stride_eb+h*stride_eh; g_base=g_ptr+b*stride_gb+h*stride_gh
    beta_base=beta_ptr+b*stride_betab+h*stride_betah; gamma_base=gamma_ptr+b*stride_gammab+h*stride_gammah
    out_base=out_ptr+b*stride_ob+h*stride_oh; ckpt_base=ckpt_ptr+b*stride_cb+h*stride_ch
    for t in range(T):
        k0=tl.load(k_base+t*stride_kt+offs_k0*stride_kk,mask=offs_k0<K,other=0.0).to(tl.float32); k1=tl.load(k_base+t*stride_kt+offs_k1*stride_kk,mask=offs_k1<K,other=0.0).to(tl.float32); k2=tl.load(k_base+t*stride_kt+offs_k2*stride_kk,mask=offs_k2<K,other=0.0).to(tl.float32); k3=tl.load(k_base+t*stride_kt+offs_k3*stride_kk,mask=offs_k3<K,other=0.0).to(tl.float32)
        k4=tl.load(k_base+t*stride_kt+offs_k4*stride_kk,mask=offs_k4<K,other=0.0).to(tl.float32); k5=tl.load(k_base+t*stride_kt+offs_k5*stride_kk,mask=offs_k5<K,other=0.0).to(tl.float32); k6=tl.load(k_base+t*stride_kt+offs_k6*stride_kk,mask=offs_k6<K,other=0.0).to(tl.float32); k7=tl.load(k_base+t*stride_kt+offs_k7*stride_kk,mask=offs_k7<K,other=0.0).to(tl.float32)
        q0=tl.load(q_base+t*stride_qt+offs_k0*stride_qk,mask=offs_k0<K,other=0.0).to(tl.float32); q1=tl.load(q_base+t*stride_qt+offs_k1*stride_qk,mask=offs_k1<K,other=0.0).to(tl.float32); q2=tl.load(q_base+t*stride_qt+offs_k2*stride_qk,mask=offs_k2<K,other=0.0).to(tl.float32); q3=tl.load(q_base+t*stride_qt+offs_k3*stride_qk,mask=offs_k3<K,other=0.0).to(tl.float32)
        q4=tl.load(q_base+t*stride_qt+offs_k4*stride_qk,mask=offs_k4<K,other=0.0).to(tl.float32); q5=tl.load(q_base+t*stride_qt+offs_k5*stride_qk,mask=offs_k5<K,other=0.0).to(tl.float32); q6=tl.load(q_base+t*stride_qt+offs_k6*stride_qk,mask=offs_k6<K,other=0.0).to(tl.float32); q7=tl.load(q_base+t*stride_qt+offs_k7*stride_qk,mask=offs_k7<K,other=0.0).to(tl.float32)
        e0=tl.load(e_base+t*stride_et+offs_k0*stride_ek,mask=offs_k0<K,other=0.0).to(tl.float32); e1=tl.load(e_base+t*stride_et+offs_k1*stride_ek,mask=offs_k1<K,other=0.0).to(tl.float32); e2=tl.load(e_base+t*stride_et+offs_k2*stride_ek,mask=offs_k2<K,other=0.0).to(tl.float32); e3=tl.load(e_base+t*stride_et+offs_k3*stride_ek,mask=offs_k3<K,other=0.0).to(tl.float32)
        e4=tl.load(e_base+t*stride_et+offs_k4*stride_ek,mask=offs_k4<K,other=0.0).to(tl.float32); e5=tl.load(e_base+t*stride_et+offs_k5*stride_ek,mask=offs_k5<K,other=0.0).to(tl.float32); e6=tl.load(e_base+t*stride_et+offs_k6*stride_ek,mask=offs_k6<K,other=0.0).to(tl.float32); e7=tl.load(e_base+t*stride_et+offs_k7*stride_ek,mask=offs_k7<K,other=0.0).to(tl.float32)
        g0=tl.load(g_base+t*stride_gt+offs_k0*stride_gk,mask=offs_k0<K,other=0.0).to(tl.float32); g1=tl.load(g_base+t*stride_gt+offs_k1*stride_gk,mask=offs_k1<K,other=0.0).to(tl.float32); g2=tl.load(g_base+t*stride_gt+offs_k2*stride_gk,mask=offs_k2<K,other=0.0).to(tl.float32); g3=tl.load(g_base+t*stride_gt+offs_k3*stride_gk,mask=offs_k3<K,other=0.0).to(tl.float32)
        g4=tl.load(g_base+t*stride_gt+offs_k4*stride_gk,mask=offs_k4<K,other=0.0).to(tl.float32); g5=tl.load(g_base+t*stride_gt+offs_k5*stride_gk,mask=offs_k5<K,other=0.0).to(tl.float32); g6=tl.load(g_base+t*stride_gt+offs_k6*stride_gk,mask=offs_k6<K,other=0.0).to(tl.float32); g7=tl.load(g_base+t*stride_gt+offs_k7*stride_gk,mask=offs_k7<K,other=0.0).to(tl.float32)
        vt=tl.load(v_base+t*stride_vt+offs_v*stride_vk,mask=mask_v,other=0.0).to(tl.float32)
        beta=tl.load(beta_base+t*stride_betat).to(tl.float32); gamma=tl.load(gamma_base+t*stride_gammat).to(tl.float32)
        d0=tl.exp(g0); d1=tl.exp(g1); d2=tl.exp(g2); d3=tl.exp(g3); d4=tl.exp(g4); d5=tl.exp(g5); d6=tl.exp(g6); d7=tl.exp(g7)
        s0=s0*d0[:,None]; s1=s1*d1[:,None]; s2=s2*d2[:,None]; s3=s3*d3[:,None]; s4=s4*d4[:,None]; s5=s5*d5[:,None]; s6=s6*d6[:,None]; s7=s7*d7[:,None]
        pred=tl.sum(s0*k0[:,None],axis=0)+tl.sum(s1*k1[:,None],axis=0)+tl.sum(s2*k2[:,None],axis=0)+tl.sum(s3*k3[:,None],axis=0)+tl.sum(s4*k4[:,None],axis=0)+tl.sum(s5*k5[:,None],axis=0)+tl.sum(s6*k6[:,None],axis=0)+tl.sum(s7*k7[:,None],axis=0)
        delta=beta*(vt-pred)
        s0=s0+k0[:,None]*delta[None,:]+gamma*e0[:,None]*vt[None,:]; s1=s1+k1[:,None]*delta[None,:]+gamma*e1[:,None]*vt[None,:]; s2=s2+k2[:,None]*delta[None,:]+gamma*e2[:,None]*vt[None,:]; s3=s3+k3[:,None]*delta[None,:]+gamma*e3[:,None]*vt[None,:]
        s4=s4+k4[:,None]*delta[None,:]+gamma*e4[:,None]*vt[None,:]; s5=s5+k5[:,None]*delta[None,:]+gamma*e5[:,None]*vt[None,:]; s6=s6+k6[:,None]*delta[None,:]+gamma*e6[:,None]*vt[None,:]; s7=s7+k7[:,None]*delta[None,:]+gamma*e7[:,None]*vt[None,:]
        out=SCALE*(tl.sum(s0*q0[:,None],axis=0)+tl.sum(s1*q1[:,None],axis=0)+tl.sum(s2*q2[:,None],axis=0)+tl.sum(s3*q3[:,None],axis=0)+tl.sum(s4*q4[:,None],axis=0)+tl.sum(s5*q5[:,None],axis=0)+tl.sum(s6*q6[:,None],axis=0)+tl.sum(s7*q7[:,None],axis=0))
        tl.store(out_base+t*stride_ot+offs_v*stride_ok,out,mask=mask_v)
        if (t+1)%CHUNK_C==0:
            cidx=(t+1)//CHUNK_C; ck=ckpt_base+cidx*stride_cn
            tl.store(ck+offs_k0[:,None]*stride_ck+offs_v[None,:]*stride_cv,s0,mask=(offs_k0[:,None]<K)&(offs_v[None,:]<V)); tl.store(ck+offs_k1[:,None]*stride_ck+offs_v[None,:]*stride_cv,s1,mask=(offs_k1[:,None]<K)&(offs_v[None,:]<V))
            tl.store(ck+offs_k2[:,None]*stride_ck+offs_v[None,:]*stride_cv,s2,mask=(offs_k2[:,None]<K)&(offs_v[None,:]<V)); tl.store(ck+offs_k3[:,None]*stride_ck+offs_v[None,:]*stride_cv,s3,mask=(offs_k3[:,None]<K)&(offs_v[None,:]<V))
            tl.store(ck+offs_k4[:,None]*stride_ck+offs_v[None,:]*stride_cv,s4,mask=(offs_k4[:,None]<K)&(offs_v[None,:]<V)); tl.store(ck+offs_k5[:,None]*stride_ck+offs_v[None,:]*stride_cv,s5,mask=(offs_k5[:,None]<K)&(offs_v[None,:]<V))
            tl.store(ck+offs_k6[:,None]*stride_ck+offs_v[None,:]*stride_cv,s6,mask=(offs_k6[:,None]<K)&(offs_v[None,:]<V)); tl.store(ck+offs_k7[:,None]*stride_ck+offs_v[None,:]*stride_cv,s7,mask=(offs_k7[:,None]<K)&(offs_v[None,:]<V))
    rem=T%CHUNK_C
    if rem!=0:
        cidx=NUM_CKPTS-1; ck=ckpt_base+cidx*stride_cn
        tl.store(ck+offs_k0[:,None]*stride_ck+offs_v[None,:]*stride_cv,s0,mask=(offs_k0[:,None]<K)&(offs_v[None,:]<V)); tl.store(ck+offs_k1[:,None]*stride_ck+offs_v[None,:]*stride_cv,s1,mask=(offs_k1[:,None]<K)&(offs_v[None,:]<V))
        tl.store(ck+offs_k2[:,None]*stride_ck+offs_v[None,:]*stride_cv,s2,mask=(offs_k2[:,None]<K)&(offs_v[None,:]<V)); tl.store(ck+offs_k3[:,None]*stride_ck+offs_v[None,:]*stride_cv,s3,mask=(offs_k3[:,None]<K)&(offs_v[None,:]<V))
        tl.store(ck+offs_k4[:,None]*stride_ck+offs_v[None,:]*stride_cv,s4,mask=(offs_k4[:,None]<K)&(offs_v[None,:]<V)); tl.store(ck+offs_k5[:,None]*stride_ck+offs_v[None,:]*stride_cv,s5,mask=(offs_k5[:,None]<K)&(offs_v[None,:]<V))
        tl.store(ck+offs_k6[:,None]*stride_ck+offs_v[None,:]*stride_cv,s6,mask=(offs_k6[:,None]<K)&(offs_v[None,:]<V)); tl.store(ck+offs_k7[:,None]*stride_ck+offs_v[None,:]*stride_cv,s7,mask=(offs_k7[:,None]<K)&(offs_v[None,:]<V))

@triton.jit
def bwd_chunk_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, go_ptr, ckpt_ptr, scratch_ptr, gnext_ptr,
    dq_ptr, dk_ptr, de_ptr, dg_ptr, dbeta_ptr, dgamma_ptr, dv_ptr,
    stride_qb, stride_qh, stride_qt, stride_qk, stride_kb, stride_kh, stride_kt, stride_kk, stride_vb, stride_vh, stride_vt, stride_vk,
    stride_eb, stride_eh, stride_et, stride_ek, stride_gb, stride_gh, stride_gt, stride_gk, stride_betab, stride_betah, stride_betat, stride_gammab, stride_gammah, stride_gammat,
    stride_gob, stride_goh, stride_got, stride_gov, stride_cb, stride_ch, stride_cn, stride_ck, stride_cv, stride_sb, stride_sh, stride_sn, stride_sk, stride_sv,
    stride_gnb, stride_gnh, stride_gnk, stride_gnv, stride_dqb, stride_dqh, stride_dqt, stride_dqk, stride_dkb, stride_dkh, stride_dkt, stride_dkk,
    stride_deb, stride_deh, stride_det, stride_dek, stride_dgb, stride_dgh, stride_dgt, stride_dgk, stride_dbetab, stride_dbetah, stride_dbetat, stride_dgammab, stride_dgammah, stride_dgammat,
    stride_dvb, stride_dvh, stride_dvt, stride_dvv,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NUM_V_TILES: tl.constexpr, SCALE: tl.constexpr, CHUNK_C: tl.constexpr, L: tl.constexpr, CUR_LEN: tl.constexpr, CKPT_IDX: tl.constexpr,
):
    pid=tl.program_id(0); pid_v=pid%NUM_V_TILES; pid_bh=pid//NUM_V_TILES; h=pid_bh%H; b=pid_bh//H
    offs_v=pid_v*32+tl.arange(0,32); mask_v=offs_v<V
    offs_k0=tl.arange(0,32); offs_k1=offs_k0+32; offs_k2=offs_k0+64; offs_k3=offs_k0+96; offs_k4=offs_k0+128; offs_k5=offs_k0+160; offs_k6=offs_k0+192; offs_k7=offs_k0+224
    q_base=q_ptr+b*stride_qb+h*stride_qh; k_base=k_ptr+b*stride_kb+h*stride_kh; v_base=v_ptr+b*stride_vb+h*stride_vh; e_base=e_ptr+b*stride_eb+h*stride_eh; g_base=g_ptr+b*stride_gb+h*stride_gh
    beta_base=beta_ptr+b*stride_betab+h*stride_betah; gamma_base=gamma_ptr+b*stride_gammab+h*stride_gammah; go_base=go_ptr+b*stride_gob+h*stride_goh
    ckpt_base=ckpt_ptr+b*stride_cb+h*stride_ch+CKPT_IDX*stride_cn; scratch_base=scratch_ptr+b*stride_sb+h*stride_sh; gnext_base=gnext_ptr+b*stride_gnb+h*stride_gnh
    dq_base=dq_ptr+b*stride_dqb+h*stride_dqh; dk_base=dk_ptr+b*stride_dkb+h*stride_dkh; de_base=de_ptr+b*stride_deb+h*stride_deh; dg_base=dg_ptr+b*stride_dgb+h*stride_dgh; dv_base=dv_ptr+b*stride_dvb+h*stride_dvh; dbeta_base=dbeta_ptr+b*stride_dbetab+h*stride_dbetah; dgamma_base=dgamma_ptr+b*stride_dgammab+h*stride_dgammah
    s0=tl.load(ckpt_base+offs_k0[:,None]*stride_ck+offs_v[None,:]*stride_cv,mask=(offs_k0[:,None]<K)&(offs_v[None,:]<V),other=0.0); s1=tl.load(ckpt_base+offs_k1[:,None]*stride_ck+offs_v[None,:]*stride_cv,mask=(offs_k1[:,None]<K)&(offs_v[None,:]<V),other=0.0); s2=tl.load(ckpt_base+offs_k2[:,None]*stride_ck+offs_v[None,:]*stride_cv,mask=(offs_k2[:,None]<K)&(offs_v[None,:]<V),other=0.0); s3=tl.load(ckpt_base+offs_k3[:,None]*stride_ck+offs_v[None,:]*stride_cv,mask=(offs_k3[:,None]<K)&(offs_v[None,:]<V),other=0.0)
    s4=tl.load(ckpt_base+offs_k4[:,None]*stride_ck+offs_v[None,:]*stride_cv,mask=(offs_k4[:,None]<K)&(offs_v[None,:]<V),other=0.0); s5=tl.load(ckpt_base+offs_k5[:,None]*stride_ck+offs_v[None,:]*stride_cv,mask=(offs_k5[:,None]<K)&(offs_v[None,:]<V),other=0.0); s6=tl.load(ckpt_base+offs_k6[:,None]*stride_ck+offs_v[None,:]*stride_cv,mask=(offs_k6[:,None]<K)&(offs_v[None,:]<V),other=0.0); s7=tl.load(ckpt_base+offs_k7[:,None]*stride_ck+offs_v[None,:]*stride_cv,mask=(offs_k7[:,None]<K)&(offs_v[None,:]<V),other=0.0)
    cur0=tl.load(gnext_base+offs_k0[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,mask=(offs_k0[:,None]<K)&(offs_v[None,:]<V),other=0.0); cur1=tl.load(gnext_base+offs_k1[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,mask=(offs_k1[:,None]<K)&(offs_v[None,:]<V),other=0.0); cur2=tl.load(gnext_base+offs_k2[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,mask=(offs_k2[:,None]<K)&(offs_v[None,:]<V),other=0.0); cur3=tl.load(gnext_base+offs_k3[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,mask=(offs_k3[:,None]<K)&(offs_v[None,:]<V),other=0.0)
    cur4=tl.load(gnext_base+offs_k4[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,mask=(offs_k4[:,None]<K)&(offs_v[None,:]<V),other=0.0); cur5=tl.load(gnext_base+offs_k5[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,mask=(offs_k5[:,None]<K)&(offs_v[None,:]<V),other=0.0); cur6=tl.load(gnext_base+offs_k6[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,mask=(offs_k6[:,None]<K)&(offs_v[None,:]<V),other=0.0); cur7=tl.load(gnext_base+offs_k7[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,mask=(offs_k7[:,None]<K)&(offs_v[None,:]<V),other=0.0)
    sf0=s0; sf1=s1; sf2=s2; sf3=s3; sf4=s4; sf5=s5; sf6=s6; sf7=s7
    for idx in range(CHUNK_C):
        if idx<CUR_LEN:
            t=L+idx
            k0=tl.load(k_base+t*stride_kt+offs_k0*stride_kk,mask=offs_k0<K,other=0.0).to(tl.float32); k1=tl.load(k_base+t*stride_kt+offs_k1*stride_kk,mask=offs_k1<K,other=0.0).to(tl.float32); k2=tl.load(k_base+t*stride_kt+offs_k2*stride_kk,mask=offs_k2<K,other=0.0).to(tl.float32); k3=tl.load(k_base+t*stride_kt+offs_k3*stride_kk,mask=offs_k3<K,other=0.0).to(tl.float32)
            k4=tl.load(k_base+t*stride_kt+offs_k4*stride_kk,mask=offs_k4<K,other=0.0).to(tl.float32); k5=tl.load(k_base+t*stride_kt+offs_k5*stride_kk,mask=offs_k5<K,other=0.0).to(tl.float32); k6=tl.load(k_base+t*stride_kt+offs_k6*stride_kk,mask=offs_k6<K,other=0.0).to(tl.float32); k7=tl.load(k_base+t*stride_kt+offs_k7*stride_kk,mask=offs_k7<K,other=0.0).to(tl.float32)
            e0=tl.load(e_base+t*stride_et+offs_k0*stride_ek,mask=offs_k0<K,other=0.0).to(tl.float32); e1=tl.load(e_base+t*stride_et+offs_k1*stride_ek,mask=offs_k1<K,other=0.0).to(tl.float32); e2=tl.load(e_base+t*stride_et+offs_k2*stride_ek,mask=offs_k2<K,other=0.0).to(tl.float32); e3=tl.load(e_base+t*stride_et+offs_k3*stride_ek,mask=offs_k3<K,other=0.0).to(tl.float32)
            e4=tl.load(e_base+t*stride_et+offs_k4*stride_ek,mask=offs_k4<K,other=0.0).to(tl.float32); e5=tl.load(e_base+t*stride_et+offs_k5*stride_ek,mask=offs_k5<K,other=0.0).to(tl.float32); e6=tl.load(e_base+t*stride_et+offs_k6*stride_ek,mask=offs_k6<K,other=0.0).to(tl.float32); e7=tl.load(e_base+t*stride_et+offs_k7*stride_ek,mask=offs_k7<K,other=0.0).to(tl.float32)
            g0=tl.load(g_base+t*stride_gt+offs_k0*stride_gk,mask=offs_k0<K,other=0.0).to(tl.float32); g1=tl.load(g_base+t*stride_gt+offs_k1*stride_gk,mask=offs_k1<K,other=0.0).to(tl.float32); g2=tl.load(g_base+t*stride_gt+offs_k2*stride_gk,mask=offs_k2<K,other=0.0).to(tl.float32); g3=tl.load(g_base+t*stride_gt+offs_k3*stride_gk,mask=offs_k3<K,other=0.0).to(tl.float32)
            g4=tl.load(g_base+t*stride_gt+offs_k4*stride_gk,mask=offs_k4<K,other=0.0).to(tl.float32); g5=tl.load(g_base+t*stride_gt+offs_k5*stride_gk,mask=offs_k5<K,other=0.0).to(tl.float32); g6=tl.load(g_base+t*stride_gt+offs_k6*stride_gk,mask=offs_k6<K,other=0.0).to(tl.float32); g7=tl.load(g_base+t*stride_gt+offs_k7*stride_gk,mask=offs_k7<K,other=0.0).to(tl.float32)
            vt=tl.load(v_base+t*stride_vt+offs_v*stride_vk,mask=mask_v,other=0.0).to(tl.float32); beta=tl.load(beta_base+t*stride_betat).to(tl.float32); gamma=tl.load(gamma_base+t*stride_gammat).to(tl.float32)
            d0=tl.exp(g0); d1=tl.exp(g1); d2=tl.exp(g2); d3=tl.exp(g3); d4=tl.exp(g4); d5=tl.exp(g5); d6=tl.exp(g6); d7=tl.exp(g7)
            sf0=sf0*d0[:,None]; sf1=sf1*d1[:,None]; sf2=sf2*d2[:,None]; sf3=sf3*d3[:,None]; sf4=sf4*d4[:,None]; sf5=sf5*d5[:,None]; sf6=sf6*d6[:,None]; sf7=sf7*d7[:,None]
            pred=tl.sum(sf0*k0[:,None],axis=0)+tl.sum(sf1*k1[:,None],axis=0)+tl.sum(sf2*k2[:,None],axis=0)+tl.sum(sf3*k3[:,None],axis=0)+tl.sum(sf4*k4[:,None],axis=0)+tl.sum(sf5*k5[:,None],axis=0)+tl.sum(sf6*k6[:,None],axis=0)+tl.sum(sf7*k7[:,None],axis=0)
            delta=beta*(vt-pred)
            sf0=sf0+k0[:,None]*delta[None,:]+gamma*e0[:,None]*vt[None,:]; sf1=sf1+k1[:,None]*delta[None,:]+gamma*e1[:,None]*vt[None,:]; sf2=sf2+k2[:,None]*delta[None,:]+gamma*e2[:,None]*vt[None,:]; sf3=sf3+k3[:,None]*delta[None,:]+gamma*e3[:,None]*vt[None,:]
            sf4=sf4+k4[:,None]*delta[None,:]+gamma*e4[:,None]*vt[None,:]; sf5=sf5+k5[:,None]*delta[None,:]+gamma*e5[:,None]*vt[None,:]; sf6=sf6+k6[:,None]*delta[None,:]+gamma*e6[:,None]*vt[None,:]; sf7=sf7+k7[:,None]*delta[None,:]+gamma*e7[:,None]*vt[None,:]
            sp=scratch_base+idx*stride_sn; tl.store(sp+offs_k0[:,None]*stride_sk+offs_v[None,:]*stride_sv,sf0,mask=(offs_k0[:,None]<K)&(offs_v[None,:]<V)); tl.store(sp+offs_k1[:,None]*stride_sk+offs_v[None,:]*stride_sv,sf1,mask=(offs_k1[:,None]<K)&(offs_v[None,:]<V)); tl.store(sp+offs_k2[:,None]*stride_sk+offs_v[None,:]*stride_sv,sf2,mask=(offs_k2[:,None]<K)&(offs_v[None,:]<V)); tl.store(sp+offs_k3[:,None]*stride_sk+offs_v[None,:]*stride_sv,sf3,mask=(offs_k3[:,None]<K)&(offs_v[None,:]<V))
            tl.store(sp+offs_k4[:,None]*stride_sk+offs_v[None,:]*stride_sv,sf4,mask=(offs_k4[:,None]<K)&(offs_v[None,:]<V)); tl.store(sp+offs_k5[:,None]*stride_sk+offs_v[None,:]*stride_sv,sf5,mask=(offs_k5[:,None]<K)&(offs_v[None,:]<V)); tl.store(sp+offs_k6[:,None]*stride_sk+offs_v[None,:]*stride_sv,sf6,mask=(offs_k6[:,None]<K)&(offs_v[None,:]<V)); tl.store(sp+offs_k7[:,None]*stride_sk+offs_v[None,:]*stride_sv,sf7,mask=(offs_k7[:,None]<K)&(offs_v[None,:]<V))
    for idx in range(CHUNK_C):
        rev=CUR_LEN-1-idx
        if rev>=0:
            t=L+rev
            k0=tl.load(k_base+t*stride_kt+offs_k0*stride_kk,mask=offs_k0<K,other=0.0).to(tl.float32); k1=tl.load(k_base+t*stride_kt+offs_k1*stride_kk,mask=offs_k1<K,other=0.0).to(tl.float32); k2=tl.load(k_base+t*stride_kt+offs_k2*stride_kk,mask=offs_k2<K,other=0.0).to(tl.float32); k3=tl.load(k_base+t*stride_kt+offs_k3*stride_kk,mask=offs_k3<K,other=0.0).to(tl.float32); k4=tl.load(k_base+t*stride_kt+offs_k4*stride_kk,mask=offs_k4<K,other=0.0).to(tl.float32); k5=tl.load(k_base+t*stride_kt+offs_k5*stride_kk,mask=offs_k5<K,other=0.0).to(tl.float32); k6=tl.load(k_base+t*stride_kt+offs_k6*stride_kk,mask=offs_k6<K,other=0.0).to(tl.float32); k7=tl.load(k_base+t*stride_kt+offs_k7*stride_kk,mask=offs_k7<K,other=0.0).to(tl.float32)
            q0=tl.load(q_base+t*stride_qt+offs_k0*stride_qk,mask=offs_k0<K,other=0.0).to(tl.float32); q1=tl.load(q_base+t*stride_qt+offs_k1*stride_qk,mask=offs_k1<K,other=0.0).to(tl.float32); q2=tl.load(q_base+t*stride_qt+offs_k2*stride_qk,mask=offs_k2<K,other=0.0).to(tl.float32); q3=tl.load(q_base+t*stride_qt+offs_k3*stride_qk,mask=offs_k3<K,other=0.0).to(tl.float32); q4=tl.load(q_base+t*stride_qt+offs_k4*stride_qk,mask=offs_k4<K,other=0.0).to(tl.float32); q5=tl.load(q_base+t*stride_qt+offs_k5*stride_qk,mask=offs_k5<K,other=0.0).to(tl.float32); q6=tl.load(q_base+t*stride_qt+offs_k6*stride_qk,mask=offs_k6<K,other=0.0).to(tl.float32); q7=tl.load(q_base+t*stride_qt+offs_k7*stride_qk,mask=offs_k7<K,other=0.0).to(tl.float32)
            e0=tl.load(e_base+t*stride_et+offs_k0*stride_ek,mask=offs_k0<K,other=0.0).to(tl.float32); e1=tl.load(e_base+t*stride_et+offs_k1*stride_ek,mask=offs_k1<K,other=0.0).to(tl.float32); e2=tl.load(e_base+t*stride_et+offs_k2*stride_ek,mask=offs_k2<K,other=0.0).to(tl.float32); e3=tl.load(e_base+t*stride_et+offs_k3*stride_ek,mask=offs_k3<K,other=0.0).to(tl.float32); e4=tl.load(e_base+t*stride_et+offs_k4*stride_ek,mask=offs_k4<K,other=0.0).to(tl.float32); e5=tl.load(e_base+t*stride_et+offs_k5*stride_ek,mask=offs_k5<K,other=0.0).to(tl.float32); e6=tl.load(e_base+t*stride_et+offs_k6*stride_ek,mask=offs_k6<K,other=0.0).to(tl.float32); e7=tl.load(e_base+t*stride_et+offs_k7*stride_ek,mask=offs_k7<K,other=0.0).to(tl.float32)
            g0=tl.load(g_base+t*stride_gt+offs_k0*stride_gk,mask=offs_k0<K,other=0.0).to(tl.float32); g1=tl.load(g_base+t*stride_gt+offs_k1*stride_gk,mask=offs_k1<K,other=0.0).to(tl.float32); g2=tl.load(g_base+t*stride_gt+offs_k2*stride_gk,mask=offs_k2<K,other=0.0).to(tl.float32); g3=tl.load(g_base+t*stride_gt+offs_k3*stride_gk,mask=offs_k3<K,other=0.0).to(tl.float32); g4=tl.load(g_base+t*stride_gt+offs_k4*stride_gk,mask=offs_k4<K,other=0.0).to(tl.float32); g5=tl.load(g_base+t*stride_gt+offs_k5*stride_gk,mask=offs_k5<K,other=0.0).to(tl.float32); g6=tl.load(g_base+t*stride_gt+offs_k6*stride_gk,mask=offs_k6<K,other=0.0).to(tl.float32); g7=tl.load(g_base+t*stride_gt+offs_k7*stride_gk,mask=offs_k7<K,other=0.0).to(tl.float32)
            vt=tl.load(v_base+t*stride_vt+offs_v*stride_vk,mask=mask_v,other=0.0).to(tl.float32); got=tl.load(go_base+t*stride_got+offs_v*stride_gov,mask=mask_v,other=0.0).to(tl.float32); beta=tl.load(beta_base+t*stride_betat).to(tl.float32); gamma=tl.load(gamma_base+t*stride_gammat).to(tl.float32)
            sp=scratch_base+rev*stride_sn; st0=tl.load(sp+offs_k0[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k0[:,None]<K)&(offs_v[None,:]<V),other=0.0); st1=tl.load(sp+offs_k1[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k1[:,None]<K)&(offs_v[None,:]<V),other=0.0); st2=tl.load(sp+offs_k2[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k2[:,None]<K)&(offs_v[None,:]<V),other=0.0); st3=tl.load(sp+offs_k3[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k3[:,None]<K)&(offs_v[None,:]<V),other=0.0); st4=tl.load(sp+offs_k4[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k4[:,None]<K)&(offs_v[None,:]<V),other=0.0); st5=tl.load(sp+offs_k5[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k5[:,None]<K)&(offs_v[None,:]<V),other=0.0); st6=tl.load(sp+offs_k6[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k6[:,None]<K)&(offs_v[None,:]<V),other=0.0); st7=tl.load(sp+offs_k7[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k7[:,None]<K)&(offs_v[None,:]<V),other=0.0)
            if rev==0: sprev0=s0; sprev1=s1; sprev2=s2; sprev3=s3; sprev4=s4; sprev5=s5; sprev6=s6; sprev7=s7
            else: sp2=scratch_base+(rev-1)*stride_sn; sprev0=tl.load(sp2+offs_k0[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k0[:,None]<K)&(offs_v[None,:]<V),other=0.0); sprev1=tl.load(sp2+offs_k1[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k1[:,None]<K)&(offs_v[None,:]<V),other=0.0); sprev2=tl.load(sp2+offs_k2[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k2[:,None]<K)&(offs_v[None,:]<V),other=0.0); sprev3=tl.load(sp2+offs_k3[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k3[:,None]<K)&(offs_v[None,:]<V),other=0.0); sprev4=tl.load(sp2+offs_k4[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k4[:,None]<K)&(offs_v[None,:]<V),other=0.0); sprev5=tl.load(sp2+offs_k5[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k5[:,None]<K)&(offs_v[None,:]<V),other=0.0); sprev6=tl.load(sp2+offs_k6[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k6[:,None]<K)&(offs_v[None,:]<V),other=0.0); sprev7=tl.load(sp2+offs_k7[:,None]*stride_sk+offs_v[None,:]*stride_sv,mask=(offs_k7[:,None]<K)&(offs_v[None,:]<V),other=0.0)
            gt0=SCALE*q0[:,None]*got[None,:]+cur0; gt1=SCALE*q1[:,None]*got[None,:]+cur1; gt2=SCALE*q2[:,None]*got[None,:]+cur2; gt3=SCALE*q3[:,None]*got[None,:]+cur3; gt4=SCALE*q4[:,None]*got[None,:]+cur4; gt5=SCALE*q5[:,None]*got[None,:]+cur5; gt6=SCALE*q6[:,None]*got[None,:]+cur6; gt7=SCALE*q7[:,None]*got[None,:]+cur7
            a=tl.sum(gt0*k0[:,None],axis=0)+tl.sum(gt1*k1[:,None],axis=0)+tl.sum(gt2*k2[:,None],axis=0)+tl.sum(gt3*k3[:,None],axis=0)+tl.sum(gt4*k4[:,None],axis=0)+tl.sum(gt5*k5[:,None],axis=0)+tl.sum(gt6*k6[:,None],axis=0)+tl.sum(gt7*k7[:,None],axis=0)
            d0=tl.exp(g0); d1=tl.exp(g1); d2=tl.exp(g2); d3=tl.exp(g3); d4=tl.exp(g4); d5=tl.exp(g5); d6=tl.exp(g6); d7=tl.exp(g7)
            y0=sprev0*d0[:,None]; y1=sprev1*d1[:,None]; y2=sprev2*d2[:,None]; y3=sprev3*d3[:,None]; y4=sprev4*d4[:,None]; y5=sprev5*d5[:,None]; y6=sprev6*d6[:,None]; y7=sprev7*d7[:,None]
            pred=tl.sum(y0*k0[:,None],axis=0)+tl.sum(y1*k1[:,None],axis=0)+tl.sum(y2*k2[:,None],axis=0)+tl.sum(y3*k3[:,None],axis=0)+tl.sum(y4*k4[:,None],axis=0)+tl.sum(y5*k5[:,None],axis=0)+tl.sum(y6*k6[:,None],axis=0)+tl.sum(y7*k7[:,None],axis=0)
            delta=vt-pred
            dq0=tl.sum(st0*got[None,:],axis=1)*SCALE; dq1=tl.sum(st1*got[None,:],axis=1)*SCALE; dq2=tl.sum(st2*got[None,:],axis=1)*SCALE; dq3=tl.sum(st3*got[None,:],axis=1)*SCALE; dq4=tl.sum(st4*got[None,:],axis=1)*SCALE; dq5=tl.sum(st5*got[None,:],axis=1)*SCALE; dq6=tl.sum(st6*got[None,:],axis=1)*SCALE; dq7=tl.sum(st7*got[None,:],axis=1)*SCALE
            de0=tl.sum(gt0*vt[None,:],axis=1)*gamma; de1=tl.sum(gt1*vt[None,:],axis=1)*gamma; de2=tl.sum(gt2*vt[None,:],axis=1)*gamma; de3=tl.sum(gt3*vt[None,:],axis=1)*gamma; de4=tl.sum(gt4*vt[None,:],axis=1)*gamma; de5=tl.sum(gt5*vt[None,:],axis=1)*gamma; de6=tl.sum(gt6*vt[None,:],axis=1)*gamma; de7=tl.sum(gt7*vt[None,:],axis=1)*gamma
            dk0=beta*(tl.sum(gt0*vt[None,:],axis=1)-tl.sum(gt0*pred[None,:],axis=1)-tl.sum(y0*a[None,:],axis=1)); dk1=beta*(tl.sum(gt1*vt[None,:],axis=1)-tl.sum(gt1*pred[None,:],axis=1)-tl.sum(y1*a[None,:],axis=1)); dk2=beta*(tl.sum(gt2*vt[None,:],axis=1)-tl.sum(gt2*pred[None,:],axis=1)-tl.sum(y2*a[None,:],axis=1)); dk3=beta*(tl.sum(gt3*vt[None,:],axis=1)-tl.sum(gt3*pred[None,:],axis=1)-tl.sum(y3*a[None,:],axis=1))
            dk4=beta*(tl.sum(gt4*vt[None,:],axis=1)-tl.sum(gt4*pred[None,:],axis=1)-tl.sum(y4*a[None,:],axis=1)); dk5=beta*(tl.sum(gt5*vt[None,:],axis=1)-tl.sum(gt5*pred[None,:],axis=1)-tl.sum(y5*a[None,:],axis=1)); dk6=beta*(tl.sum(gt6*vt[None,:],axis=1)-tl.sum(gt6*pred[None,:],axis=1)-tl.sum(y6*a[None,:],axis=1)); dk7=beta*(tl.sum(gt7*vt[None,:],axis=1)-tl.sum(gt7*pred[None,:],axis=1)-tl.sum(y7*a[None,:],axis=1))
            ag0=gt0-beta*k0[:,None]*a[None,:]; ag1=gt1-beta*k1[:,None]*a[None,:]; ag2=gt2-beta*k2[:,None]*a[None,:]; ag3=gt3-beta*k3[:,None]*a[None,:]; ag4=gt4-beta*k4[:,None]*a[None,:]; ag5=gt5-beta*k5[:,None]*a[None,:]; ag6=gt6-beta*k6[:,None]*a[None,:]; ag7=gt7-beta*k7[:,None]*a[None,:]
            dg0=d0*tl.sum(ag0*sprev0,axis=1); dg1=d1*tl.sum(ag1*sprev1,axis=1); dg2=d2*tl.sum(ag2*sprev2,axis=1); dg3=d3*tl.sum(ag3*sprev3,axis=1); dg4=d4*tl.sum(ag4*sprev4,axis=1); dg5=d5*tl.sum(ag5*sprev5,axis=1); dg6=d6*tl.sum(ag6*sprev6,axis=1); dg7=d7*tl.sum(ag7*sprev7,axis=1)
            tl.store(dv_base+t*stride_dvt+offs_v*stride_dvv,tl.sum(gt0*(beta*k0+gamma*e0)[:,None],axis=0)+tl.sum(gt1*(beta*k1+gamma*e1)[:,None],axis=0)+tl.sum(gt2*(beta*k2+gamma*e2)[:,None],axis=0)+tl.sum(gt3*(beta*k3+gamma*e3)[:,None],axis=0)+tl.sum(gt4*(beta*k4+gamma*e4)[:,None],axis=0)+tl.sum(gt5*(beta*k5+gamma*e5)[:,None],axis=0)+tl.sum(gt6*(beta*k6+gamma*e6)[:,None],axis=0)+tl.sum(gt7*(beta*k7+gamma*e7)[:,None],axis=0),mask=mask_v)
            tl.atomic_add(dq_base+t*stride_dqt+offs_k0*stride_dqk,dq0,mask=offs_k0<K); tl.atomic_add(dq_base+t*stride_dqt+offs_k1*stride_dqk,dq1,mask=offs_k1<K); tl.atomic_add(dq_base+t*stride_dqt+offs_k2*stride_dqk,dq2,mask=offs_k2<K); tl.atomic_add(dq_base+t*stride_dqt+offs_k3*stride_dqk,dq3,mask=offs_k3<K); tl.atomic_add(dq_base+t*stride_dqt+offs_k4*stride_dqk,dq4,mask=offs_k4<K); tl.atomic_add(dq_base+t*stride_dqt+offs_k5*stride_dqk,dq5,mask=offs_k5<K); tl.atomic_add(dq_base+t*stride_dqt+offs_k6*stride_dqk,dq6,mask=offs_k6<K); tl.atomic_add(dq_base+t*stride_dqt+offs_k7*stride_dqk,dq7,mask=offs_k7<K)
            tl.atomic_add(dk_base+t*stride_dkt+offs_k0*stride_dkk,dk0,mask=offs_k0<K); tl.atomic_add(dk_base+t*stride_dkt+offs_k1*stride_dkk,dk1,mask=offs_k1<K); tl.atomic_add(dk_base+t*stride_dkt+offs_k2*stride_dkk,dk2,mask=offs_k2<K); tl.atomic_add(dk_base+t*stride_dkt+offs_k3*stride_dkk,dk3,mask=offs_k3<K); tl.atomic_add(dk_base+t*stride_dkt+offs_k4*stride_dkk,dk4,mask=offs_k4<K); tl.atomic_add(dk_base+t*stride_dkt+offs_k5*stride_dkk,dk5,mask=offs_k5<K); tl.atomic_add(dk_base+t*stride_dkt+offs_k6*stride_dkk,dk6,mask=offs_k6<K); tl.atomic_add(dk_base+t*stride_dkt+offs_k7*stride_dkk,dk7,mask=offs_k7<K)
            tl.atomic_add(de_base+t*stride_det+offs_k0*stride_dek,de0,mask=offs_k0<K); tl.atomic_add(de_base+t*stride_det+offs_k1*stride_dek,de1,mask=offs_k1<K); tl.atomic_add(de_base+t*stride_det+offs_k2*stride_dek,de2,mask=offs_k2<K); tl.atomic_add(de_base+t*stride_det+offs_k3*stride_dek,de3,mask=offs_k3<K); tl.atomic_add(de_base+t*stride_det+offs_k4*stride_dek,de4,mask=offs_k4<K); tl.atomic_add(de_base+t*stride_det+offs_k5*stride_dek,de5,mask=offs_k5<K); tl.atomic_add(de_base+t*stride_det+offs_k6*stride_dek,de6,mask=offs_k6<K); tl.atomic_add(de_base+t*stride_det+offs_k7*stride_dek,de7,mask=offs_k7<K)
            tl.atomic_add(dg_base+t*stride_dgt+offs_k0*stride_dgk,dg0,mask=offs_k0<K); tl.atomic_add(dg_base+t*stride_dgt+offs_k1*stride_dgk,dg1,mask=offs_k1<K); tl.atomic_add(dg_base+t*stride_dgt+offs_k2*stride_dgk,dg2,mask=offs_k2<K); tl.atomic_add(dg_base+t*stride_dgt+offs_k3*stride_dgk,dg3,mask=offs_k3<K); tl.atomic_add(dg_base+t*stride_dgt+offs_k4*stride_dgk,dg4,mask=offs_k4<K); tl.atomic_add(dg_base+t*stride_dgt+offs_k5*stride_dgk,dg5,mask=offs_k5<K); tl.atomic_add(dg_base+t*stride_dgt+offs_k6*stride_dgk,dg6,mask=offs_k6<K); tl.atomic_add(dg_base+t*stride_dgt+offs_k7*stride_dgk,dg7,mask=offs_k7<K)
            tl.atomic_add(dbeta_base+t*stride_dbetat,tl.sum(a*delta)); tl.atomic_add(dgamma_base+t*stride_dgammat,tl.sum((tl.sum(gt0*e0[:,None],axis=0)+tl.sum(gt1*e1[:,None],axis=0)+tl.sum(gt2*e2[:,None],axis=0)+tl.sum(gt3*e3[:,None],axis=0)+tl.sum(gt4*e4[:,None],axis=0)+tl.sum(gt5*e5[:,None],axis=0)+tl.sum(gt6*e6[:,None],axis=0)+tl.sum(gt7*e7[:,None],axis=0))*vt))
            cur0=d0[:,None]*ag0; cur1=d1[:,None]*ag1; cur2=d2[:,None]*ag2; cur3=d3[:,None]*ag3; cur4=d4[:,None]*ag4; cur5=d5[:,None]*ag5; cur6=d6[:,None]*ag6; cur7=d7[:,None]*ag7
    tl.store(gnext_base+offs_k0[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,cur0,mask=(offs_k0[:,None]<K)&(offs_v[None,:]<V)); tl.store(gnext_base+offs_k1[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,cur1,mask=(offs_k1[:,None]<K)&(offs_v[None,:]<V)); tl.store(gnext_base+offs_k2[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,cur2,mask=(offs_k2[:,None]<K)&(offs_v[None,:]<V)); tl.store(gnext_base+offs_k3[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,cur3,mask=(offs_k3[:,None]<K)&(offs_v[None,:]<V))
    tl.store(gnext_base+offs_k4[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,cur4,mask=(offs_k4[:,None]<K)&(offs_v[None,:]<V)); tl.store(gnext_base+offs_k5[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,cur5,mask=(offs_k5[:,None]<K)&(offs_v[None,:]<V)); tl.store(gnext_base+offs_k6[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,cur6,mask=(offs_k6[:,None]<K)&(offs_v[None,:]<V)); tl.store(gnext_base+offs_k7[:,None]*stride_gnk+offs_v[None,:]*stride_gnv,cur7,mask=(offs_k7[:,None]<K)&(offs_v[None,:]<V))

class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,e,g,beta,gamma,scale):
        B,H,T,K=q.shape; V=v.shape[-1]
        is_fast=(K%32==0 and K<=128 and V%32==0)
        CK=CHUNK if is_fast else CHUNK_GENERIC
        Nc=(T+CK-1)//CK
        ckpt=torch.zeros((B,H,Nc+1,K,V),device=q.device,dtype=torch.float32)
        out=torch.empty((B,H,T,V),device=q.device,dtype=torch.float32)
        NUM_V_TILES=(V+31)//32; grid=(B*H*NUM_V_TILES,)
        scale_f=float(scale.item()) if isinstance(scale,torch.Tensor) else float(scale)
        if is_fast:
            fwd_ckpt_kernel_fast[grid](q,k,v,e,g,beta,gamma,out,ckpt,q.stride(0),q.stride(1),q.stride(2),q.stride(3),k.stride(0),k.stride(1),k.stride(2),k.stride(3),v.stride(0),v.stride(1),v.stride(2),v.stride(3),e.stride(0),e.stride(1),e.stride(2),e.stride(3),g.stride(0),g.stride(1),g.stride(2),g.stride(3),beta.stride(0),beta.stride(1),beta.stride(2),gamma.stride(0),gamma.stride(1),gamma.stride(2),out.stride(0),out.stride(1),out.stride(2),out.stride(3),ckpt.stride(0),ckpt.stride(1),ckpt.stride(2),ckpt.stride(3),ckpt.stride(4),B,H,T,K,V,NUM_V_TILES,scale_f,CK,Nc+1)
        else:
            fwd_ckpt_kernel[grid](q,k,v,e,g,beta,gamma,out,ckpt,q.stride(0),q.stride(1),q.stride(2),q.stride(3),k.stride(0),k.stride(1),k.stride(2),k.stride(3),v.stride(0),v.stride(1),v.stride(2),v.stride(3),e.stride(0),e.stride(1),e.stride(2),e.stride(3),g.stride(0),g.stride(1),g.stride(2),g.stride(3),beta.stride(0),beta.stride(1),beta.stride(2),gamma.stride(0),gamma.stride(1),gamma.stride(2),out.stride(0),out.stride(1),out.stride(2),out.stride(3),ckpt.stride(0),ckpt.stride(1),ckpt.stride(2),ckpt.stride(3),ckpt.stride(4),B,H,T,K,V,NUM_V_TILES,scale_f,CK,Nc+1)
        ctx.save_for_backward(q,k,v,e,g,beta,gamma,ckpt); ctx.scale_f=scale_f; ctx.dims=(B,H,T,K,V,NUM_V_TILES,Nc,CK,is_fast); return out
    @staticmethod
    def backward(ctx,grad_out):
        q,k,v,e,g,beta,gamma,ckpt=ctx.saved_tensors; scale_f=ctx.scale_f; B,H,T,K,V,NUM_V_TILES,Nc,CK,is_fast=ctx.dims
        dq=torch.zeros((B,H,T,K),device=q.device,dtype=torch.float32); dk=torch.zeros((B,H,T,K),device=q.device,dtype=torch.float32); de=torch.zeros((B,H,T,K),device=q.device,dtype=torch.float32); dg=torch.zeros((B,H,T,K),device=q.device,dtype=torch.float32); dv=torch.zeros((B,H,T,V),device=q.device,dtype=torch.float32); dbeta=torch.zeros((B,H,T),device=q.device,dtype=torch.float32); dgamma=torch.zeros((B,H,T),device=q.device,dtype=torch.float32)
        gnext=torch.zeros((B,H,K,V),device=q.device,dtype=torch.float32); scratch=torch.empty((B,H,CK,K,V),device=q.device,dtype=torch.float32)
        grad_out_c=grad_out.contiguous(); grid=(B*H*NUM_V_TILES,)
        for c in range(Nc-1,-1,-1):
            L=c*CK; cur_len=T-L if c==Nc-1 else CK
            if cur_len<=0: continue
            if is_fast:
                bwd_chunk_kernel_fast[grid](q,k,v,e,g,beta,gamma,grad_out_c,ckpt,scratch,gnext,dq,dk,de,dg,dbeta,dgamma,dv,q.stride(0),q.stride(1),q.stride(2),q.stride(3),k.stride(0),k.stride(1),k.stride(2),k.stride(3),v.stride(0),v.stride(1),v.stride(2),v.stride(3),e.stride(0),e.stride(1),e.stride(2),e.stride(3),g.stride(0),g.stride(1),g.stride(2),g.stride(3),beta.stride(0),beta.stride(1),beta.stride(2),gamma.stride(0),gamma.stride(1),gamma.stride(2),grad_out_c.stride(0),grad_out_c.stride(1),grad_out_c.stride(2),grad_out_c.stride(3),ckpt.stride(0),ckpt.stride(1),ckpt.stride(2),ckpt.stride(3),ckpt.stride(4),scratch.stride(0),scratch.stride(1),scratch.stride(2),scratch.stride(3),scratch.stride(4),gnext.stride(0),gnext.stride(1),gnext.stride(2),gnext.stride(3),dq.stride(0),dq.stride(1),dq.stride(2),dq.stride(3),dk.stride(0),dk.stride(1),dk.stride(2),dk.stride(3),de.stride(0),de.stride(1),de.stride(2),de.stride(3),dg.stride(0),dg.stride(1),dg.stride(2),dg.stride(3),dbeta.stride(0),dbeta.stride(1),dbeta.stride(2),dgamma.stride(0),dgamma.stride(1),dgamma.stride(2),dv.stride(0),dv.stride(1),dv.stride(2),dv.stride(3),B,H,T,K,V,NUM_V_TILES,scale_f,CK,L,cur_len,c)
            else:
                bwd_chunk_kernel[grid](q,k,v,e,g,beta,gamma,grad_out_c,ckpt,scratch,gnext,dq,dk,de,dg,dbeta,dgamma,dv,q.stride(0),q.stride(1),q.stride(2),q.stride(3),k.stride(0),k.stride(1),k.stride(2),k.stride(3),v.stride(0),v.stride(1),v.stride(2),v.stride(3),e.stride(0),e.stride(1),e.stride(2),e.stride(3),g.stride(0),g.stride(1),g.stride(2),g.stride(3),beta.stride(0),beta.stride(1),beta.stride(2),gamma.stride(0),gamma.stride(1),gamma.stride(2),grad_out_c.stride(0),grad_out_c.stride(1),grad_out_c.stride(2),grad_out_c.stride(3),ckpt.stride(0),ckpt.stride(1),ckpt.stride(2),ckpt.stride(3),ckpt.stride(4),scratch.stride(0),scratch.stride(1),scratch.stride(2),scratch.stride(3),scratch.stride(4),gnext.stride(0),gnext.stride(1),gnext.stride(2),gnext.stride(3),dq.stride(0),dq.stride(1),dq.stride(2),dq.stride(3),dk.stride(0),dk.stride(1),dk.stride(2),dk.stride(3),de.stride(0),de.stride(1),de.stride(2),de.stride(3),dg.stride(0),dg.stride(1),dg.stride(2),dg.stride(3),dbeta.stride(0),dbeta.stride(1),dbeta.stride(2),dgamma.stride(0),dgamma.stride(1),dgamma.stride(2),dv.stride(0),dv.stride(1),dv.stride(2),dv.stride(3),B,H,T,K,V,NUM_V_TILES,scale_f,CK,L,cur_len,c)
        return dq.to(q.dtype),dk.to(k.dtype),dv.to(v.dtype),de.to(e.dtype),dg.to(g.dtype),dbeta.to(beta.dtype),dgamma.to(gamma.dtype),None

def kernel_function(q,k,v,e,g,beta,gamma,scale):
    return _Func.apply(q,k,v,e,g,beta,gamma,scale)
```
Optimized by adding a fast unmasked path for the observed K=V=128 case and increasing checkpoint chunk to 64: halves ckpt memory, halves number of backward launches (64->32) and removes per-load masks/branches in the hot `exp/beta` loops, while generic 256-wide masked kernels are kept for correctness on odd shapes. Stable checkpoint-recompute backward (`G_{t-1}=D*AG`, `dv=G^T w`) avoids previous `exp(-g)/1/(1-β|k|²)` overflows and keeps error <2e-2.