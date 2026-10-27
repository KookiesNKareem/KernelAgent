import torch
import triton
import triton.language as tl


def _np2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


# ---------------- fast D=128 E=4 ----------------
@triton.jit
def _fwd_ckpt_fast(
    q_ptr, k_ptr, v_ptr, q2_ptr, k2_ptr, dec_ptr, beta_ptr, out_ptr, ckpt_ptr,
    T, V, BH, NV, C, nC, K, scale,
    BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    kd = tl.arange(0, BLOCK_D)
    vd = vid * BLOCK_V + tl.arange(0, BLOCK_V)
    S0 = tl.zeros((BLOCK_D, BLOCK_V), dtype=tl.float32)
    S1 = tl.zeros((BLOCK_D, BLOCK_V), dtype=tl.float32)
    S2 = tl.zeros((BLOCK_D, BLOCK_V), dtype=tl.float32)
    S3 = tl.zeros((BLOCK_D, BLOCK_V), dtype=tl.float32)
    q_bh = bh * T * BLOCK_D
    k_bh = bh * T * BLOCK_D
    v_bh = bh * T * V
    q2_bh = bh * T * 4
    k2_bh = bh * T * 4
    dec_bh = bh * T * 4
    b_bh = bh * T
    o_bh = bh * T * V
    EV = 4 * V
    for t in range(T):
        k = tl.load(k_ptr + k_bh + t * BLOCK_D + kd).to(tl.float32)
        q = tl.load(q_ptr + q_bh + t * BLOCK_D + kd).to(tl.float32)
        vv = tl.load(v_ptr + v_bh + t * V + vd).to(tl.float32)
        beta_t = tl.load(beta_ptr + b_bh + t).to(tl.float32)
        k2b = k2_ptr + k2_bh + t * 4
        q2b = q2_ptr + q2_bh + t * 4
        db = dec_ptr + dec_bh + t * 4
        k20 = tl.load(k2b + 0).to(tl.float32)
        k21 = tl.load(k2b + 1).to(tl.float32)
        k22 = tl.load(k2b + 2).to(tl.float32)
        k23 = tl.load(k2b + 3).to(tl.float32)
        q20 = tl.load(q2b + 0).to(tl.float32)
        q21 = tl.load(q2b + 1).to(tl.float32)
        q22 = tl.load(q2b + 2).to(tl.float32)
        q23 = tl.load(q2b + 3).to(tl.float32)
        d0 = tl.load(db + 0).to(tl.float32)
        d1 = tl.load(db + 1).to(tl.float32)
        d2 = tl.load(db + 2).to(tl.float32)
        d3 = tl.load(db + 3).to(tl.float32)
        kq = tl.sum(k * q)
        k2q2 = k20 * q20 + k21 * q21 + k22 * q22 + k23 * q23
        c_raw = kq * k2q2
        ks0e = k * (k20 * d0)
        ks1e = k * (k21 * d1)
        ks2e = k * (k22 * d2)
        ks3e = k * (k23 * d3)
        qs0e = q * (q20 * d0)
        qs1e = q * (q21 * d1)
        qs2e = q * (q22 * d2)
        qs3e = q * (q23 * d3)
        p0 = tl.sum(S0 * tl.expand_dims(ks0e, 1), 0)
        p1 = tl.sum(S1 * tl.expand_dims(ks1e, 1), 0)
        p2 = tl.sum(S2 * tl.expand_dims(ks2e, 1), 0)
        p3 = tl.sum(S3 * tl.expand_dims(ks3e, 1), 0)
        pred = p0 + p1 + p2 + p3
        o0 = tl.sum(S0 * tl.expand_dims(qs0e, 1), 0)
        o1 = tl.sum(S1 * tl.expand_dims(qs1e, 1), 0)
        o2 = tl.sum(S2 * tl.expand_dims(qs2e, 1), 0)
        o3 = tl.sum(S3 * tl.expand_dims(qs3e, 1), 0)
        odec = o0 + o1 + o2 + o3
        delta = (vv - pred) * beta_t
        outv = (odec + c_raw * delta) * scale
        tl.store(out_ptr + o_bh + t * V + vd, outv)
        de = tl.expand_dims(delta, 0)
        S0 = S0 * d0 + tl.expand_dims(k * k20, 1) * de
        S1 = S1 * d1 + tl.expand_dims(k * k21, 1) * de
        S2 = S2 * d2 + tl.expand_dims(k * k22, 1) * de
        S3 = S3 * d3 + tl.expand_dims(k * k23, 1) * de
        do_ckpt = False
        cidx = 0
        if (t + 1) % C == 0:
            do_ckpt = True
            cidx = (t + 1) // C
        elif t == T - 1:
            do_ckpt = True
            cidx = nC
        if do_ckpt:
            base = (bh * (nC + 1) + cidx) * K * V
            off2d = tl.expand_dims(kd, 1) * EV + tl.expand_dims(vd, 0)
            tl.store(ckpt_ptr + base + 0 * V + off2d, S0)
            tl.store(ckpt_ptr + base + 1 * V + off2d, S1)
            tl.store(ckpt_ptr + base + 2 * V + off2d, S2)
            tl.store(ckpt_ptr + base + 3 * V + off2d, S3)


@triton.jit
def _recomp_fast(
    k_ptr, k2_ptr, v_ptr, dec_ptr, beta_ptr, ckpt_ptr, scratch_ptr,
    T, V, BH, NV, nC, K, t0, cur_C, chunk_idx,
    BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    kd = tl.arange(0, BLOCK_D)
    vd = vid * BLOCK_V + tl.arange(0, BLOCK_V)
    EV = 4 * V
    cbase = (bh * (nC + 1) + chunk_idx) * K * V
    off2d = tl.expand_dims(kd, 1) * EV + tl.expand_dims(vd, 0)
    S0 = tl.load(ckpt_ptr + cbase + 0 * V + off2d).to(tl.float32)
    S1 = tl.load(ckpt_ptr + cbase + 1 * V + off2d).to(tl.float32)
    S2 = tl.load(ckpt_ptr + cbase + 2 * V + off2d).to(tl.float32)
    S3 = tl.load(ckpt_ptr + cbase + 3 * V + off2d).to(tl.float32)
    k_bh = bh * T * BLOCK_D
    v_bh = bh * T * V
    k2_bh = bh * T * 4
    dec_bh = bh * T * 4
    b_bh = bh * T
    for ii in range(cur_C):
        t = t0 + ii
        k = tl.load(k_ptr + k_bh + t * BLOCK_D + kd).to(tl.float32)
        vv = tl.load(v_ptr + v_bh + t * V + vd).to(tl.float32)
        beta_t = tl.load(beta_ptr + b_bh + t).to(tl.float32)
        k2b = k2_ptr + k2_bh + t * 4
        db = dec_ptr + dec_bh + t * 4
        k20 = tl.load(k2b + 0).to(tl.float32)
        k21 = tl.load(k2b + 1).to(tl.float32)
        k22 = tl.load(k2b + 2).to(tl.float32)
        k23 = tl.load(k2b + 3).to(tl.float32)
        d0 = tl.load(db + 0).to(tl.float32)
        d1 = tl.load(db + 1).to(tl.float32)
        d2 = tl.load(db + 2).to(tl.float32)
        d3 = tl.load(db + 3).to(tl.float32)
        ks0e = k * (k20 * d0)
        ks1e = k * (k21 * d1)
        ks2e = k * (k22 * d2)
        ks3e = k * (k23 * d3)
        p0 = tl.sum(S0 * tl.expand_dims(ks0e, 1), 0)
        p1 = tl.sum(S1 * tl.expand_dims(ks1e, 1), 0)
        p2 = tl.sum(S2 * tl.expand_dims(ks2e, 1), 0)
        p3 = tl.sum(S3 * tl.expand_dims(ks3e, 1), 0)
        pred = p0 + p1 + p2 + p3
        delta = (vv - pred) * beta_t
        de = tl.expand_dims(delta, 0)
        S0 = S0 * d0 + tl.expand_dims(k * k20, 1) * de
        S1 = S1 * d1 + tl.expand_dims(k * k21, 1) * de
        S2 = S2 * d2 + tl.expand_dims(k * k22, 1) * de
        S3 = S3 * d3 + tl.expand_dims(k * k23, 1) * de
        sbase2 = (bh * 64 + ii) * K * V
        tl.store(scratch_ptr + sbase2 + 0 * V + off2d, S0)
        tl.store(scratch_ptr + sbase2 + 1 * V + off2d, S1)
        tl.store(scratch_ptr + sbase2 + 2 * V + off2d, S2)
        tl.store(scratch_ptr + sbase2 + 3 * V + off2d, S3)


@triton.jit
def _bwd_fast_partial(
    q_ptr, k_ptr, v_ptr, q2_ptr, k2_ptr, dec_ptr, beta_ptr, dO_ptr, scratch_ptr, ckpt_ptr, carry_ptr,
    dqp_ptr, dkp_ptr, dv_ptr, dq2p_ptr, dk2p_ptr, dgp_ptr, dbp_ptr,
    T, V, BH, NV, nC, K, t0, cur_C, chunk_idx, scale,
    BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    kd = tl.arange(0, BLOCK_D)
    vd = vid * BLOCK_V + tl.arange(0, BLOCK_V)
    EV = 4 * V
    off2d = tl.expand_dims(kd, 1) * EV + tl.expand_dims(vd, 0)
    cbase = bh * K * V
    C0 = tl.load(carry_ptr + cbase + 0 * V + off2d).to(tl.float32)
    C1 = tl.load(carry_ptr + cbase + 1 * V + off2d).to(tl.float32)
    C2 = tl.load(carry_ptr + cbase + 2 * V + off2d).to(tl.float32)
    C3 = tl.load(carry_ptr + cbase + 3 * V + off2d).to(tl.float32)
    s_last = (bh * 64 + cur_C - 1) * K * V
    St0 = tl.load(scratch_ptr + s_last + 0 * V + off2d).to(tl.float32)
    St1 = tl.load(scratch_ptr + s_last + 1 * V + off2d).to(tl.float32)
    St2 = tl.load(scratch_ptr + s_last + 2 * V + off2d).to(tl.float32)
    St3 = tl.load(scratch_ptr + s_last + 3 * V + off2d).to(tl.float32)
    q_bh = bh * T * BLOCK_D
    k_bh = bh * T * BLOCK_D
    v_bh = bh * T * V
    q2_bh = bh * T * 4
    k2_bh = bh * T * 4
    dec_bh = bh * T * 4
    b_bh = bh * T
    o_bh = bh * T * V
    ckbase = (bh * (nC + 1) + chunk_idx) * K * V
    Ddim = BLOCK_D
    for rev in range(cur_C):
        ii = cur_C - 1 - rev
        t = t0 + ii
        if ii == 0:
            Sp0 = tl.load(ckpt_ptr + ckbase + 0 * V + off2d).to(tl.float32)
            Sp1 = tl.load(ckpt_ptr + ckbase + 1 * V + off2d).to(tl.float32)
            Sp2 = tl.load(ckpt_ptr + ckbase + 2 * V + off2d).to(tl.float32)
            Sp3 = tl.load(ckpt_ptr + ckbase + 3 * V + off2d).to(tl.float32)
        else:
            spbase = (bh * 64 + ii - 1) * K * V
            Sp0 = tl.load(scratch_ptr + spbase + 0 * V + off2d).to(tl.float32)
            Sp1 = tl.load(scratch_ptr + spbase + 1 * V + off2d).to(tl.float32)
            Sp2 = tl.load(scratch_ptr + spbase + 2 * V + off2d).to(tl.float32)
            Sp3 = tl.load(scratch_ptr + spbase + 3 * V + off2d).to(tl.float32)
        k = tl.load(k_ptr + k_bh + t * BLOCK_D + kd).to(tl.float32)
        q = tl.load(q_ptr + q_bh + t * BLOCK_D + kd).to(tl.float32)
        vv = tl.load(v_ptr + v_bh + t * V + vd).to(tl.float32)
        dO = tl.load(dO_ptr + o_bh + t * V + vd).to(tl.float32)
        beta_t = tl.load(beta_ptr + b_bh + t).to(tl.float32)
        k2b = k2_ptr + k2_bh + t * 4
        q2b = q2_ptr + q2_bh + t * 4
        db = dec_ptr + dec_bh + t * 4
        k20 = tl.load(k2b + 0).to(tl.float32)
        k21 = tl.load(k2b + 1).to(tl.float32)
        k22 = tl.load(k2b + 2).to(tl.float32)
        k23 = tl.load(k2b + 3).to(tl.float32)
        q20 = tl.load(q2b + 0).to(tl.float32)
        q21 = tl.load(q2b + 1).to(tl.float32)
        q22 = tl.load(q2b + 2).to(tl.float32)
        q23 = tl.load(q2b + 3).to(tl.float32)
        d0 = tl.load(db + 0).to(tl.float32)
        d1 = tl.load(db + 1).to(tl.float32)
        d2 = tl.load(db + 2).to(tl.float32)
        d3 = tl.load(db + 3).to(tl.float32)
        kap0 = k * k20
        kap1 = k * k21
        kap2 = k * k22
        kap3 = k * k23
        qq0 = q * q20
        qq1 = q * q21
        qq2 = q * q22
        qq3 = q * q23
        Shat0 = Sp0 * d0
        Shat1 = Sp1 * d1
        Shat2 = Sp2 * d2
        Shat3 = Sp3 * d3
        p0 = tl.sum(Shat0 * tl.expand_dims(kap0, 1), 0)
        p1 = tl.sum(Shat1 * tl.expand_dims(kap1, 1), 0)
        p2 = tl.sum(Shat2 * tl.expand_dims(kap2, 1), 0)
        p3 = tl.sum(Shat3 * tl.expand_dims(kap3, 1), 0)
        pred = p0 + p1 + p2 + p3
        vmp = vv - pred
        dOs = dO * scale
        dOs2 = tl.expand_dims(dOs, 0)
        DS0 = tl.expand_dims(qq0, 1) * dOs2 + C0
        DS1 = tl.expand_dims(qq1, 1) * dOs2 + C1
        DS2 = tl.expand_dims(qq2, 1) * dOs2 + C2
        DS3 = tl.expand_dims(qq3, 1) * dOs2 + C3
        a0 = tl.sum(DS0 * tl.expand_dims(kap0, 1), 0)
        a1 = tl.sum(DS1 * tl.expand_dims(kap1, 1), 0)
        a2 = tl.sum(DS2 * tl.expand_dims(kap2, 1), 0)
        a3 = tl.sum(DS3 * tl.expand_dims(kap3, 1), 0)
        avec = a0 + a1 + a2 + a3
        ba = beta_t * avec
        tl.store(dv_ptr + v_bh + t * V + vd, ba)
        db_part = tl.sum(avec * vmp)
        vmp2 = tl.expand_dims(vmp, 0)
        avec2 = tl.expand_dims(avec, 0)
        dot1_0 = tl.sum(DS0 * vmp2, 1)
        dot1_1 = tl.sum(DS1 * vmp2, 1)
        dot1_2 = tl.sum(DS2 * vmp2, 1)
        dot1_3 = tl.sum(DS3 * vmp2, 1)
        dot2_0 = tl.sum(Shat0 * avec2, 1)
        dot2_1 = tl.sum(Shat1 * avec2, 1)
        dot2_2 = tl.sum(Shat2 * avec2, 1)
        dot2_3 = tl.sum(Shat3 * avec2, 1)
        dkap0 = (dot1_0 - dot2_0) * beta_t
        dkap1 = (dot1_1 - dot2_1) * beta_t
        dkap2 = (dot1_2 - dot2_2) * beta_t
        dkap3 = (dot1_3 - dot2_3) * beta_t
        dqq0 = tl.sum(St0 * dOs2, 1)
        dqq1 = tl.sum(St1 * dOs2, 1)
        dqq2 = tl.sum(St2 * dOs2, 1)
        dqq3 = tl.sum(St3 * dOs2, 1)
        dk_p = dkap0 * k20 + dkap1 * k21 + dkap2 * k22 + dkap3 * k23
        dq_p = dqq0 * q20 + dqq1 * q21 + dqq2 * q22 + dqq3 * q23
        pbase_q = ((bh * NV + vid) * T + t) * Ddim
        tl.store(dqp_ptr + pbase_q + kd, dq_p)
        tl.store(dkp_ptr + pbase_q + kd, dk_p)
        pbase_e = ((bh * NV + vid) * T + t) * 4
        tl.store(dq2p_ptr + pbase_e + 0, tl.sum(dqq0 * q))
        tl.store(dq2p_ptr + pbase_e + 1, tl.sum(dqq1 * q))
        tl.store(dq2p_ptr + pbase_e + 2, tl.sum(dqq2 * q))
        tl.store(dq2p_ptr + pbase_e + 3, tl.sum(dqq3 * q))
        tl.store(dk2p_ptr + pbase_e + 0, tl.sum(dkap0 * k))
        tl.store(dk2p_ptr + pbase_e + 1, tl.sum(dkap1 * k))
        tl.store(dk2p_ptr + pbase_e + 2, tl.sum(dkap2 * k))
        tl.store(dk2p_ptr + pbase_e + 3, tl.sum(dkap3 * k))
        tl.store(dbp_ptr + (bh * NV + vid) * T + t, db_part)
        E0 = DS0 - tl.expand_dims(kap0, 1) * tl.expand_dims(ba, 0)
        E1 = DS1 - tl.expand_dims(kap1, 1) * tl.expand_dims(ba, 0)
        E2 = DS2 - tl.expand_dims(kap2, 1) * tl.expand_dims(ba, 0)
        E3 = DS3 - tl.expand_dims(kap3, 1) * tl.expand_dims(ba, 0)
        dd0 = tl.sum(tl.sum(E0 * Sp0, 1)) * d0
        dd1 = tl.sum(tl.sum(E1 * Sp1, 1)) * d1
        dd2 = tl.sum(tl.sum(E2 * Sp2, 1)) * d2
        dd3 = tl.sum(tl.sum(E3 * Sp3, 1)) * d3
        tl.store(dgp_ptr + pbase_e + 0, dd0)
        tl.store(dgp_ptr + pbase_e + 1, dd1)
        tl.store(dgp_ptr + pbase_e + 2, dd2)
        tl.store(dgp_ptr + pbase_e + 3, dd3)
        C0 = E0 * d0
        C1 = E1 * d1
        C2 = E2 * d2
        C3 = E3 * d3
        St0 = Sp0
        St1 = Sp1
        St2 = Sp2
        St3 = Sp3
    tl.store(carry_ptr + cbase + 0 * V + off2d, C0)
    tl.store(carry_ptr + cbase + 1 * V + off2d, C1)
    tl.store(carry_ptr + cbase + 2 * V + off2d, C2)
    tl.store(carry_ptr + cbase + 3 * V + off2d, C3)


# ---------------- generic ----------------
@triton.jit
def _fwd_ckpt_gen(
    q_ptr, k_ptr, v_ptr, q2_ptr, k2_ptr, dec_ptr, beta_ptr, out_ptr, ckpt_ptr,
    T, D, E, V, BH, NV, C, nC, K, scale,
    BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    d_idx = tl.arange(0, BLOCK_D)
    e_idx = tl.arange(0, BLOCK_E)
    v_idx = vid * BLOCK_V + tl.arange(0, BLOCK_V)
    d_mask = d_idx < D
    e_mask = e_idx < E
    v_mask = v_idx < V
    d_idx_s = tl.where(d_mask, d_idx, 0)
    e_idx_s = tl.where(e_mask, e_idx, 0)
    v_idx_s = tl.where(v_mask, v_idx, 0)
    de_v_mask = tl.expand_dims(tl.expand_dims(d_mask, 1), 2) & tl.expand_dims(tl.expand_dims(e_mask, 0), 2) & tl.expand_dims(tl.expand_dims(v_mask, 0), 0)
    S = tl.zeros((BLOCK_D, BLOCK_E, BLOCK_V), dtype=tl.float32)
    for t in range(T):
        k_d = tl.load(k_ptr + bh * T * D + t * D + d_idx_s, mask=d_mask, other=0.0).to(tl.float32)
        q_d = tl.load(q_ptr + bh * T * D + t * D + d_idx_s, mask=d_mask, other=0.0).to(tl.float32)
        k2_e = tl.load(k2_ptr + bh * T * E + t * E + e_idx_s, mask=e_mask, other=0.0).to(tl.float32)
        q2_e = tl.load(q2_ptr + bh * T * E + t * E + e_idx_s, mask=e_mask, other=0.0).to(tl.float32)
        dec_e = tl.load(dec_ptr + bh * T * E + t * E + e_idx_s, mask=e_mask, other=1.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + bh * T + t).to(tl.float32)
        v_v = tl.load(v_ptr + bh * T * V + t * V + v_idx_s, mask=v_mask, other=0.0).to(tl.float32)
        k_3 = tl.expand_dims(tl.expand_dims(k_d, 1), 2)
        q_3 = tl.expand_dims(tl.expand_dims(q_d, 1), 2)
        k2_3 = tl.expand_dims(tl.expand_dims(k2_e, 0), 2)
        q2_3 = tl.expand_dims(tl.expand_dims(q2_e, 0), 2)
        dec_3 = tl.expand_dims(tl.expand_dims(dec_e, 0), 2)
        Shat = S * dec_3
        pred = tl.sum(tl.sum(Shat * k_3 * k2_3, 0), 0)
        delta = (v_v - pred) * beta_t
        S = Shat + k_3 * k2_3 * tl.expand_dims(tl.expand_dims(delta, 0), 0)
        out_v = tl.sum(tl.sum(S * q_3 * q2_3, 0), 0) * scale
        tl.store(out_ptr + bh * T * V + t * V + v_idx_s, out_v, mask=v_mask)
        do_ckpt = False
        cidx = 0
        if (t + 1) % C == 0:
            do_ckpt = True
            cidx = (t + 1) // C
        elif t == T - 1:
            do_ckpt = True
            cidx = nC
        if do_ckpt:
            d_e = tl.expand_dims(tl.expand_dims(d_idx, 1), 2) * E + tl.expand_dims(tl.expand_dims(e_idx, 0), 2)
            vv_e = tl.expand_dims(tl.expand_dims(v_idx, 0), 0)
            offs = d_e * V + vv_e
            offs_s = tl.where(de_v_mask, offs, 0)
            base = (bh * (nC + 1) + cidx) * K * V
            tl.store(ckpt_ptr + base + offs_s, S, mask=de_v_mask)


@triton.jit
def _recomp_gen(
    k_ptr, k2_ptr, v_ptr, dec_ptr, beta_ptr, ckpt_ptr, scratch_ptr,
    T, D, E, V, BH, NV, nC, K, Cmax, t0, cur_C, chunk_idx,
    BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    d_idx = tl.arange(0, BLOCK_D)
    e_idx = tl.arange(0, BLOCK_E)
    v_idx = vid * BLOCK_V + tl.arange(0, BLOCK_V)
    d_mask = d_idx < D
    e_mask = e_idx < E
    v_mask = v_idx < V
    d_idx_s = tl.where(d_mask, d_idx, 0)
    e_idx_s = tl.where(e_mask, e_idx, 0)
    v_idx_s = tl.where(v_mask, v_idx, 0)
    de_v_mask = tl.expand_dims(tl.expand_dims(d_mask, 1), 2) & tl.expand_dims(tl.expand_dims(e_mask, 0), 2) & tl.expand_dims(tl.expand_dims(v_mask, 0), 0)
    d_e = tl.expand_dims(tl.expand_dims(d_idx, 1), 2) * E + tl.expand_dims(tl.expand_dims(e_idx, 0), 2)
    vv_e = tl.expand_dims(tl.expand_dims(v_idx, 0), 0)
    offs = d_e * V + vv_e
    offs_s = tl.where(de_v_mask, offs, 0)
    sbase = (bh * (nC + 1) + chunk_idx) * K * V
    S = tl.load(ckpt_ptr + sbase + offs_s, mask=de_v_mask, other=0.0).to(tl.float32)
    for ii in range(cur_C):
        t = t0 + ii
        k_d = tl.load(k_ptr + bh * T * D + t * D + d_idx_s, mask=d_mask, other=0.0).to(tl.float32)
        k2_e = tl.load(k2_ptr + bh * T * E + t * E + e_idx_s, mask=e_mask, other=0.0).to(tl.float32)
        dec_e = tl.load(dec_ptr + bh * T * E + t * E + e_idx_s, mask=e_mask, other=1.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + bh * T + t).to(tl.float32)
        v_v = tl.load(v_ptr + bh * T * V + t * V + v_idx_s, mask=v_mask, other=0.0).to(tl.float32)
        k_3 = tl.expand_dims(tl.expand_dims(k_d, 1), 2)
        k2_3 = tl.expand_dims(tl.expand_dims(k2_e, 0), 2)
        dec_3 = tl.expand_dims(tl.expand_dims(dec_e, 0), 2)
        Shat = S * dec_3
        pred = tl.sum(tl.sum(Shat * k_3 * k2_3, 0), 0)
        delta = (v_v - pred) * beta_t
        S = Shat + k_3 * k2_3 * tl.expand_dims(tl.expand_dims(delta, 0), 0)
        sbase2 = (bh * Cmax + ii) * K * V
        tl.store(scratch_ptr + sbase2 + offs_s, S, mask=de_v_mask)


@triton.jit
def _bwd_gen_partial(
    q_ptr, k_ptr, v_ptr, q2_ptr, k2_ptr, dec_ptr, beta_ptr, dO_ptr, scratch_ptr, ckpt_ptr, carry_ptr,
    dqp_ptr, dkp_ptr, dv_ptr, dq2p_ptr, dk2p_ptr, dgp_ptr, dbp_ptr,
    T, D, E, V, BH, NV, nC, K, Cmax, t0, cur_C, chunk_idx, scale,
    BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    d_idx = tl.arange(0, BLOCK_D)
    e_idx = tl.arange(0, BLOCK_E)
    v_idx = vid * BLOCK_V + tl.arange(0, BLOCK_V)
    d_mask = d_idx < D
    e_mask = e_idx < E
    v_mask = v_idx < V
    d_idx_s = tl.where(d_mask, d_idx, 0)
    e_idx_s = tl.where(e_mask, e_idx, 0)
    v_idx_s = tl.where(v_mask, v_idx, 0)
    de_v_mask = tl.expand_dims(tl.expand_dims(d_mask, 1), 2) & tl.expand_dims(tl.expand_dims(e_mask, 0), 2) & tl.expand_dims(tl.expand_dims(v_mask, 0), 0)
    d_e = tl.expand_dims(tl.expand_dims(d_idx, 1), 2) * E + tl.expand_dims(tl.expand_dims(e_idx, 0), 2)
    vv_e = tl.expand_dims(tl.expand_dims(v_idx, 0), 0)
    offs = d_e * V + vv_e
    offs_s = tl.where(de_v_mask, offs, 0)
    cbase = bh * K * V
    carry = tl.load(carry_ptr + cbase + offs_s, mask=de_v_mask, other=0.0).to(tl.float32)
    sbase0 = (bh * (nC + 1) + chunk_idx) * K * V
    S_start = tl.load(ckpt_ptr + sbase0 + offs_s, mask=de_v_mask, other=0.0).to(tl.float32)
    for ii in range(cur_C):
        t_local = cur_C - 1 - ii
        t = t0 + t_local
        k_d = tl.load(k_ptr + bh * T * D + t * D + d_idx_s, mask=d_mask, other=0.0).to(tl.float32)
        q_d = tl.load(q_ptr + bh * T * D + t * D + d_idx_s, mask=d_mask, other=0.0).to(tl.float32)
        k2_e = tl.load(k2_ptr + bh * T * E + t * E + e_idx_s, mask=e_mask, other=0.0).to(tl.float32)
        q2_e = tl.load(q2_ptr + bh * T * E + t * E + e_idx_s, mask=e_mask, other=0.0).to(tl.float32)
        dec_e = tl.load(dec_ptr + bh * T * E + t * E + e_idx_s, mask=e_mask, other=1.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + bh * T + t).to(tl.float32)
        v_v = tl.load(v_ptr + bh * T * V + t * V + v_idx_s, mask=v_mask, other=0.0).to(tl.float32)
        dO_v = tl.load(dO_ptr + bh * T * V + t * V + v_idx_s, mask=v_mask, other=0.0).to(tl.float32)
        St = tl.load(scratch_ptr + (bh * Cmax + t_local) * K * V + offs_s, mask=de_v_mask, other=0.0).to(tl.float32)
        if t_local == 0:
            Sp = S_start
        else:
            Sp = tl.load(scratch_ptr + (bh * Cmax + t_local - 1) * K * V + offs_s, mask=de_v_mask, other=0.0).to(tl.float32)
        k_3 = tl.expand_dims(tl.expand_dims(k_d, 1), 2)
        q_3 = tl.expand_dims(tl.expand_dims(q_d, 1), 2)
        k2_3 = tl.expand_dims(tl.expand_dims(k2_e, 0), 2)
        q2_3 = tl.expand_dims(tl.expand_dims(q2_e, 0), 2)
        dec_3 = tl.expand_dims(tl.expand_dims(dec_e, 0), 2)
        kap = k_3 * k2_3
        qq = q_3 * q2_3
        dO_3 = tl.expand_dims(tl.expand_dims(dO_v, 0), 0)
        direct = qq * dO_3 * scale
        DS = direct + carry
        Shat = Sp * dec_3
        pred = tl.sum(tl.sum(Shat * kap, 0), 0)
        vmp = v_v - pred
        a = tl.sum(tl.sum(DS * kap, 0), 0)
        ba = beta_t * a
        ba_3 = tl.expand_dims(tl.expand_dims(ba, 0), 0)
        vmp_3 = tl.expand_dims(tl.expand_dims(vmp, 0), 0)
        a_3 = tl.expand_dims(tl.expand_dims(a, 0), 0)
        Emat = DS - kap * ba_3
        tl.store(dv_ptr + bh * T * V + t * V + v_idx_s, ba, mask=v_mask)
        db_part = tl.sum(a * vmp)
        tl.store(dbp_ptr + (bh * NV + vid) * T + t, db_part)
        dot1 = tl.sum(DS * vmp_3, 2)
        dot2 = tl.sum(Shat * a_3, 2)
        dkap = (dot1 - dot2) * beta_t
        dqq = tl.sum(St * dO_3, 2) * scale
        dDmat = tl.sum(Emat * Sp, 2)
        dk_p = tl.sum(dkap * tl.expand_dims(k2_e, 0), 1)
        dq_p = tl.sum(dqq * tl.expand_dims(q2_e, 0), 1)
        dk2_p = tl.sum(dkap * tl.expand_dims(k_d, 1), 0)
        dq2_p = tl.sum(dqq * tl.expand_dims(q_d, 1), 0)
        dg_p = tl.sum(dDmat, 0) * dec_e
        pbase_d = ((bh * NV + vid) * T + t) * D
        tl.store(dqp_ptr + pbase_d + d_idx_s, dq_p, mask=d_mask)
        tl.store(dkp_ptr + pbase_d + d_idx_s, dk_p, mask=d_mask)
        pbase_e = ((bh * NV + vid) * T + t) * E
        tl.store(dq2p_ptr + pbase_e + e_idx_s, dq2_p, mask=e_mask)
        tl.store(dk2p_ptr + pbase_e + e_idx_s, dk2_p, mask=e_mask)
        tl.store(dgp_ptr + pbase_e + e_idx_s, dg_p, mask=e_mask)
        carry = Emat * dec_3
    tl.store(carry_ptr + cbase + offs_s, carry, mask=de_v_mask)


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q2, k2, g, beta, scale):
        B, H, T, D = q.shape
        E = q2.shape[-1]
        Vd = v.shape[-1]
        BH = B * H
        K = D * E
        q3 = q.contiguous().view(BH, T, D)
        k3 = k.contiguous().view(BH, T, D)
        v3 = v.contiguous().view(BH, T, Vd)
        q23 = q2.contiguous().view(BH, T, E)
        k23 = k2.contiguous().view(BH, T, E)
        dec = torch.exp(g.to(torch.float32)).contiguous().view(BH, T, E)
        b3 = beta.contiguous().view(BH, T)
        out3 = torch.empty((BH, T, Vd), device=q.device, dtype=torch.float32)
        scale_f = float(scale.item()) if torch.is_tensor(scale) else float(scale)
        fast = (D == 128 and E == 4 and Vd % 32 == 0)
        C = 64
        nC = (T + C - 1) // C
        if fast:
            ckpt = torch.zeros((BH * (nC + 1) * K * Vd,), device=q.device, dtype=torch.float32)
            BLOCK_V = 32
            NV = Vd // BLOCK_V
            grid = (BH * NV,)
            _fwd_ckpt_fast[grid](
                q3, k3, v3, q23, k23, dec, b3, out3, ckpt,
                T, Vd, BH, NV, C, nC, K, scale_f,
                BLOCK_D=128, BLOCK_V=BLOCK_V, num_warps=8, num_stages=2,
            )
            ctx.fast = True
        else:
            if T <= 32:
                C = T
                nC = (T + C - 1) // C
            ckpt = torch.zeros((BH * (nC + 1) * K * Vd,), device=q.device, dtype=torch.float32)
            BLOCK_D = _np2(D)
            BLOCK_E = _np2(E)
            BLOCK_V = 16
            NV = (Vd + BLOCK_V - 1) // BLOCK_V
            grid = (BH * NV,)
            _fwd_ckpt_gen[grid](
                q3, k3, v3, q23, k23, dec, b3, out3, ckpt,
                T, D, E, Vd, BH, NV, C, nC, K, scale_f,
                BLOCK_D=BLOCK_D, BLOCK_E=BLOCK_E, BLOCK_V=BLOCK_V,
                num_warps=4, num_stages=1,
            )
            ctx.fast = False
        ctx.save_for_backward(q3, k3, v3, q23, k23, dec, b3, ckpt)
        ctx.B, ctx.H, ctx.T, ctx.D, ctx.E, ctx.Vd, ctx.BH, ctx.K = B, H, T, D, E, Vd, BH, K
        ctx.C, ctx.nC, ctx.scale_f = C, nC, scale_f
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, q2.dtype, k2.dtype, g.dtype, beta.dtype)
        return out3.view(B, H, T, Vd)

    @staticmethod
    def backward(ctx, grad_out):
        q3, k3, v3, q23, k23, dec, b3, ckpt = ctx.saved_tensors
        B, H, T, D, E, Vd, BH, K = ctx.B, ctx.H, ctx.T, ctx.D, ctx.E, ctx.Vd, ctx.BH, ctx.K
        C, nC, scale_f = ctx.C, ctx.nC, ctx.scale_f
        g_out = grad_out.contiguous().view(BH, T, Vd).to(torch.float32)
        dv = torch.zeros((BH, T, Vd), device=q3.device, dtype=torch.float32)
        if ctx.fast:
            BLOCK_V = 32
            NV = Vd // BLOCK_V
            grid = (BH * NV,)
            carry = torch.zeros((BH * K * Vd,), device=q3.device, dtype=torch.float32)
            scratch = torch.empty((BH * 64 * K * Vd,), device=q3.device, dtype=torch.float32)
            dqp = torch.zeros((BH * NV * T * D,), device=q3.device, dtype=torch.float32)
            dkp = torch.zeros((BH * NV * T * D,), device=q3.device, dtype=torch.float32)
            dq2p = torch.zeros((BH * NV * T * E,), device=q3.device, dtype=torch.float32)
            dk2p = torch.zeros((BH * NV * T * E,), device=q3.device, dtype=torch.float32)
            dgp = torch.zeros((BH * NV * T * E,), device=q3.device, dtype=torch.float32)
            dbp = torch.zeros((BH * NV * T,), device=q3.device, dtype=torch.float32)
            for ci in range(nC - 1, -1, -1):
                t0 = ci * C
                cur_C = min(C, T - t0)
                _recomp_fast[grid](
                    k3, k23, v3, dec, b3, ckpt, scratch,
                    T, Vd, BH, NV, nC, K, t0, cur_C, ci,
                    BLOCK_D=128, BLOCK_V=BLOCK_V, num_warps=8, num_stages=2,
                )
                _bwd_fast_partial[grid](
                    q3, k3, v3, q23, k23, dec, b3, g_out, scratch, ckpt, carry,
                    dqp, dkp, dv, dq2p, dk2p, dgp, dbp,
                    T, Vd, BH, NV, nC, K, t0, cur_C, ci, scale_f,
                    BLOCK_D=128, BLOCK_V=BLOCK_V, num_warps=8, num_stages=2,
                )
            dq = dqp.view(BH, NV, T, D).sum(1)
            dk = dkp.view(BH, NV, T, D).sum(1)
            dq2 = dq2p.view(BH, NV, T, E).sum(1)
            dk2 = dk2p.view(BH, NV, T, E).sum(1)
            dg = dgp.view(BH, NV, T, E).sum(1)
            db = dbp.view(BH, NV, T).sum(1)
        else:
            BLOCK_D = _np2(D)
            BLOCK_E = _np2(E)
            BLOCK_V = 16
            NV = (Vd + BLOCK_V - 1) // BLOCK_V
            grid = (BH * NV,)
            carry = torch.zeros((BH * K * Vd,), device=q3.device, dtype=torch.float32)
            scratch = torch.empty((BH * C * K * Vd,), device=q3.device, dtype=torch.float32)
            dqp = torch.zeros((BH * NV * T * D,), device=q3.device, dtype=torch.float32)
            dkp = torch.zeros((BH * NV * T * D,), device=q3.device, dtype=torch.float32)
            dq2p = torch.zeros((BH * NV * T * E,), device=q3.device, dtype=torch.float32)
            dk2p = torch.zeros((BH * NV * T * E,), device=q3.device, dtype=torch.float32)
            dgp = torch.zeros((BH * NV * T * E,), device=q3.device, dtype=torch.float32)
            dbp = torch.zeros((BH * NV * T,), device=q3.device, dtype=torch.float32)
            for ci in range(nC - 1, -1, -1):
                t0 = ci * C
                cur_C = min(C, T - t0)
                _recomp_gen[grid](
                    k3, k23, v3, dec, b3, ckpt, scratch,
                    T, D, E, Vd, BH, NV, nC, K, C, t0, cur_C, ci,
                    BLOCK_D=BLOCK_D, BLOCK_E=BLOCK_E, BLOCK_V=BLOCK_V,
                    num_warps=4, num_stages=1,
                )
                _bwd_gen_partial[grid](
                    q3, k3, v3, q23, k23, dec, b3, g_out, scratch, ckpt, carry,
                    dqp, dkp, dv, dq2p, dk2p, dgp, dbp,
                    T, D, E, Vd, BH, NV, nC, K, C, t0, cur_C, ci, scale_f,
                    BLOCK_D=BLOCK_D, BLOCK_E=BLOCK_E, BLOCK_V=BLOCK_V,
                    num_warps=4, num_stages=1,
                )
            dq = dqp.view(BH, NV, T, D).sum(1)
            dk = dkp.view(BH, NV, T, D).sum(1)
            dq2 = dq2p.view(BH, NV, T, E).sum(1)
            dk2 = dk2p.view(BH, NV, T, E).sum(1)
            dg = dgp.view(BH, NV, T, E).sum(1)
            db = dbp.view(BH, NV, T).sum(1)
        dt = ctx.dtypes
        dq = dq.view(B, H, T, D).to(dt[0])
        dk = dk.view(B, H, T, D).to(dt[1])
        dv = dv.view(B, H, T, Vd).to(dt[2])
        dq2 = dq2.view(B, H, T, E).to(dt[3])
        dk2 = dk2.view(B, H, T, E).to(dt[4])
        dg = dg.view(B, H, T, E).to(dt[5])
        db = db.view(B, H, T).to(dt[6])
        return dq, dk, dv, dq2, dk2, dg, db, None


def kernel_function(q, k, v, q2, k2, g, beta, scale):
    return _Func.apply(q, k, v, q2, k2, g, beta, scale)
