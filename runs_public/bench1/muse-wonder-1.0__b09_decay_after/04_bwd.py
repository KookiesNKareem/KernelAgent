import torch
import triton
import triton.language as tl


@triton.jit
def _prep_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    w_ptr,
    kwf_ptr,
    qeff_ptr,
    dec_ptr,
    u_ptr,
    o_ptr,
    nmat_ptr,
    invmat_ptr,
    T,
    K,
    V,
    NT,
    BC: tl.constexpr,
    BKB: tl.constexpr,
    BK2: tl.constexpr,
    BV2: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // NT
    pid_nt = pid % NT

    offs_c = tl.arange(0, BC)
    idx_t = pid_nt * BC + offs_c
    mask_t = idx_t < T

    base_k = pid_bh * T * K
    base_v = pid_bh * T * V
    base_b = pid_bh * T

    b_beta = tl.load(beta_ptr + base_b + idx_t, mask=mask_t, other=0.0).to(
        tl.float32
    )

    # ---- Intra-chunk Gram matrices M and N, accumulated over small K
    # blocks.  For the used (causal) entries every exponent is
    # cum_pre[t] - cum_pre[j] <= 0 because g <= 0, so clamping the rest is
    # harmless.
    m_mat = tl.zeros([BC, BC], dtype=tl.float32)
    n_mat = tl.zeros([BC, BC], dtype=tl.float32)

    for kb in range(0, K, BKB):
        offs_ks = kb + tl.arange(0, BKB)
        mask_ks = offs_ks < K
        mask_s = mask_t[:, None] & mask_ks[None, :]
        ptrs = base_k + idx_t[:, None] * K + offs_ks[None, :]

        ks = tl.load(k_ptr + ptrs, mask=mask_s, other=0.0).to(tl.float32)
        qs = tl.load(q_ptr + ptrs, mask=mask_s, other=0.0).to(tl.float32)
        gs = tl.load(g_ptr + ptrs, mask=mask_s, other=0.0).to(tl.float32)

        cum_s = tl.cumsum(gs, axis=0)
        cump_s = cum_s - gs

        diff = tl.expand_dims(cump_s, 1) - tl.expand_dims(cump_s, 0)
        diff = tl.minimum(diff, 0.0)
        e_mat = tl.exp(diff)

        m_mat += tl.sum(
            tl.expand_dims(ks, 1) * tl.expand_dims(ks, 0) * e_mat, axis=2
        )
        qg_s = qs * tl.exp(gs)
        n_mat += tl.sum(
            tl.expand_dims(qg_s, 1) * tl.expand_dims(ks, 0) * e_mat, axis=2
        )

    loc_row = offs_c[:, None]
    loc_col = offs_c[None, :]
    eye = tl.where(loc_row == loc_col, 1.0, 0.0)

    a_mat = tl.where(loc_row > loc_col, m_mat, 0.0) * b_beta[:, None]
    n_mat = tl.where(loc_row >= loc_col, n_mat, 0.0)

    # (I + A)^-1 for strictly-lower-triangular, hence nilpotent, A (BC=32):
    # (I - A)(I + A^2)(I + A^4)(I + A^8)(I + A^16).
    inv_mat = eye - a_mat
    pow_mat = tl.dot(a_mat, a_mat, input_precision="tf32")
    for _ in range(4):
        inv_mat = tl.dot(inv_mat, eye + pow_mat, input_precision="tf32")
        pow_mat = tl.dot(pow_mat, pow_mat, input_precision="tf32")

    # Saved for the backward pass.
    mat_ptrs = (
        pid_bh * NT * BC * BC
        + pid_nt * BC * BC
        + loc_row * BC
        + loc_col
    )
    tl.store(nmat_ptr + mat_ptrs, n_mat)
    tl.store(invmat_ptr + mat_ptrs, inv_mat)

    # ---- Per-K-block quantities: W, final-state keys, effective queries,
    # and the total chunk decay (one vector per chunk, reused later).
    for kb in range(0, K, BK2):
        offs_kb = kb + tl.arange(0, BK2)
        mask_kb = offs_kb < K
        mask_k2 = mask_t[:, None] & mask_kb[None, :]
        ptrs = base_k + idx_t[:, None] * K + offs_kb[None, :]

        b_k = tl.load(k_ptr + ptrs, mask=mask_k2, other=0.0).to(tl.float32)
        b_q = tl.load(q_ptr + ptrs, mask=mask_k2, other=0.0).to(tl.float32)
        b_g = tl.load(g_ptr + ptrs, mask=mask_k2, other=0.0).to(tl.float32)

        b_cum = tl.cumsum(b_g, axis=0)
        b_cum_prev = b_cum - b_g
        cum_last = tl.sum(b_g, axis=0)

        b_kp = b_k * tl.exp(b_cum_prev)
        b_qf = b_q * tl.exp(b_cum)
        b_kwf = b_k * tl.exp(cum_last[None, :] - b_cum_prev)

        b_w = tl.dot(inv_mat, b_beta[:, None] * b_kp, input_precision="tf32")
        b_qeff = b_qf - tl.dot(n_mat, b_w, input_precision="tf32")

        tl.store(w_ptr + ptrs, b_w, mask=mask_k2)
        tl.store(kwf_ptr + ptrs, b_kwf, mask=mask_k2)
        tl.store(qeff_ptr + ptrs, b_qeff, mask=mask_k2)
        tl.store(
            dec_ptr + pid_bh * NT * K + pid_nt * K + offs_kb,
            cum_last,
            mask=mask_kb,
        )

    # ---- Per-V-block state-independent values and outputs.
    for vb in range(0, V, BV2):
        offs_vb = vb + tl.arange(0, BV2)
        mask_vb = offs_vb < V
        mask_v2 = mask_t[:, None] & mask_vb[None, :]
        ptrs_v = base_v + idx_t[:, None] * V + offs_vb[None, :]

        b_v = tl.load(v_ptr + ptrs_v, mask=mask_v2, other=0.0).to(tl.float32)
        b_u0 = tl.dot(
            inv_mat, b_beta[:, None] * b_v, input_precision="tf32"
        )
        b_o0 = tl.dot(n_mat, b_u0, input_precision="tf32")

        tl.store(u_ptr + ptrs_v, b_u0, mask=mask_v2)
        tl.store(o_ptr + ptrs_v, b_o0, mask=mask_v2)


@triton.jit
def _state_kernel(
    w_ptr,
    kwf_ptr,
    qeff_ptr,
    dec_ptr,
    u_ptr,
    o_ptr,
    out_ptr,
    sbound_ptr,
    T,
    K,
    V,
    NV,
    NT,
    scale,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // NV
    pid_vb = pid % NV

    offs_c = tl.arange(0, BC)
    offs_k = tl.arange(0, BK)
    offs_v = pid_vb * BV + tl.arange(0, BV)

    mask_k = offs_k < K
    mask_v = offs_v < V

    base_k = pid_bh * T * K
    base_v = pid_bh * T * V

    s = tl.zeros([BK, BV], dtype=tl.float32)

    for chunk in range(NT):
        idx_t = chunk * BC + offs_c
        mask_t = idx_t < T
        mask_k2 = mask_t[:, None] & mask_k[None, :]
        mask_v2 = mask_t[:, None] & mask_v[None, :]

        ptrs_k = base_k + idx_t[:, None] * K + offs_k[None, :]
        ptrs_v = base_v + idx_t[:, None] * V + offs_v[None, :]

        # State entering this chunk, saved for the backward pass.
        tl.store(
            sbound_ptr
            + pid_bh * NT * K * V
            + chunk * K * V
            + offs_k[:, None] * V
            + offs_v[None, :],
            s,
            mask=mask_k[:, None] & mask_v[None, :],
        )

        b_w = tl.load(w_ptr + ptrs_k, mask=mask_k2, other=0.0).to(tl.float32)
        b_kwf = tl.load(kwf_ptr + ptrs_k, mask=mask_k2, other=0.0).to(tl.float32)
        b_qeff = tl.load(qeff_ptr + ptrs_k, mask=mask_k2, other=0.0).to(tl.float32)
        b_u = tl.load(u_ptr + ptrs_v, mask=mask_v2, other=0.0).to(tl.float32)
        b_o0 = tl.load(o_ptr + ptrs_v, mask=mask_v2, other=0.0).to(tl.float32)
        b_dec = tl.load(
            dec_ptr + pid_bh * NT * K + chunk * K + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

        u_corr = b_u - tl.dot(b_w, s, input_precision="tf32")
        b_o = b_o0 + tl.dot(b_qeff, s, input_precision="tf32")
        tl.store(out_ptr + ptrs_v, b_o * scale, mask=mask_v2)

        s = s * tl.exp(b_dec)[:, None] + tl.dot(
            tl.trans(b_kwf), u_corr, input_precision="tf32"
        )


@triton.jit
def _bwd_state_kernel(
    w_ptr,
    kwf_ptr,
    qeff_ptr,
    dec_ptr,
    u_ptr,
    nmat_ptr,
    sbound_ptr,
    do_ptr,
    du0_ptr,
    dw_part_ptr,
    dkwf_part_ptr,
    dqeff_part_ptr,
    dn_part_ptr,
    ddec_part_ptr,
    T,
    K,
    V,
    NT,
    scale,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    # NV is folded into the grid size; recover it from K/V-independent args
    # via program count is not available, so the wrapper uses BV s.t. the
    # caller passes grid = BH * NV and pid_vb = pid % NV with NV encoded
    # by the BV tiling of V: recompute below from pid and BH is ambiguous,
    # hence the wrapper launches with NV passed through `NT`-independent
    # trick: pid_bh is supplied by using grid ordering set in wrapper.
    # To keep this unambiguous, the wrapper sets BV so NV == gridDim/BH and
    # passes NV as an extra runtime argument through `NVB` below.
    tl.static_assert(True)


@triton.jit
def _bwd_state_kernel_impl(
    w_ptr,
    kwf_ptr,
    qeff_ptr,
    dec_ptr,
    u_ptr,
    nmat_ptr,
    sbound_ptr,
    do_ptr,
    du0_ptr,
    dw_part_ptr,
    dkwf_part_ptr,
    dqeff_part_ptr,
    dn_part_ptr,
    ddec_part_ptr,
    T,
    K,
    V,
    NT,
    NV,
    scale,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // NV
    pid_vb = pid % NV

    offs_c = tl.arange(0, BC)
    offs_k = tl.arange(0, BK)
    offs_v = pid_vb * BV + tl.arange(0, BV)

    mask_k = offs_k < K
    mask_v = offs_v < V

    base_k = pid_bh * T * K
    base_v = pid_bh * T * V
    part_k = pid_vb * (T * K) + base_k  # placeholder, corrected below

    # Gradient w.r.t. the state leaving a chunk, swept in reverse.
    gs = tl.zeros([BK, BV], dtype=tl.float32)

    for rev in range(NT):
        chunk = NT - 1 - rev
        idx_t = chunk * BC + offs_c
        mask_t = idx_t < T
        mask_k2 = mask_t[:, None] & mask_k[None, :]
        mask_v2 = mask_t[:, None] & mask_v[None, :]

        ptrs_k = base_k + idx_t[:, None] * K + offs_k[None, :]
        ptrs_v = base_v + idx_t[:, None] * V + offs_v[None, :]

        b_w = tl.load(w_ptr + ptrs_k, mask=mask_k2, other=0.0).to(tl.float32)
        b_kwf = tl.load(kwf_ptr + ptrs_k, mask=mask_k2, other=0.0).to(tl.float32)
        b_qeff = tl.load(qeff_ptr + ptrs_k, mask=mask_k2, other=0.0).to(tl.float32)
        b_u0 = tl.load(u_ptr + ptrs_v, mask=mask_v2, other=0.0).to(tl.float32)
        b_do = tl.load(do_ptr + ptrs_v, mask=mask_v2, other=0.0).to(tl.float32)
        b_do = b_do * scale

        n_mat = tl.load(
            nmat_ptr
            + pid_bh * NT * BC * BC
            + chunk * BC * BC
            + offs_c[:, None] * BC
            + offs_c[None, :]
        ).to(tl.float32)

        s_in = tl.load(
            sbound_ptr
            + pid_bh * NT * K * V
            + chunk * K * V
            + offs_k[:, None] * V
            + offs_v[None, :],
            mask=mask_k[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)

        b_dec = tl.load(
            dec_ptr + pid_bh * NT * K + chunk * K + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        dec_exp = tl.exp(b_dec)

        # Actual corrected values of this chunk and their gradients.
        act_u = b_u0 - tl.dot(b_w, s_in, input_precision="tf32")
        g_u = tl.dot(b_kwf, gs, input_precision="tf32")
        g_u0 = g_u + tl.dot(tl.trans(n_mat), b_do, input_precision="tf32")

        tl.store(du0_ptr + ptrs_v, g_u0, mask=mask_v2)

        # Partials over this V block (summed by the wrapper over V blocks).
        g_w = -tl.dot(g_u0, tl.trans(s_in), input_precision="tf32")
        g_kwf = tl.dot(act_u, tl.trans(gs), input_precision="tf32")
        g_qeff = tl.dot(b_do, tl.trans(s_in), input_precision="tf32")
        g_n = tl.dot(b_do, tl.trans(act_u), input_precision="tf32")
        g_dec = tl.sum(gs * (s_in * dec_exp[:, None]), axis=1)

        part_base_k = pid_vb * (NT * T * K // NT)  # unused, see below
        pk = (pid_vb * (T * K) * (NT // NT))  # unused placeholder

        # Store partials: layout [NV, BH, T, K] / [NV, BH, NT, BC, BC] /
        # [NV, BH, NT, K].  BH is recovered as (total BH) from base offsets:
        # pid_bh already indexes BH, and the per-block stride is BH*T*K,
        # which the wrapper guarantees by allocating with leading dim NV and
        # passing the base pointer; the BH stride is (T*K) * (num BH) --
        # encoded via `base_k` scaling below.
        # NOTE: strides are computed from pid_bh and the tensor extents
        # passed implicitly through the allocations; to avoid ambiguity we
        # use flat offsets built from (pid_vb, pid_bh) with BH stride
        # supplied as T*K*BH where BH = (grid / NV) is not visible here.
        # The wrapper therefore launches this kernel with one V block when
        # BH stride would be ambiguous in general; see wrapper (NV == 2
        # handled by explicit BH argument `BHV`).
        tl.store(
            dw_part_ptr + pid_vb * 0 + ptrs_k - base_k + base_k, g_w, mask=mask_k2
        )
        # The three stores above/below are rewritten by the wrapper's
        # specialised launch (see _run_bwd_state); this kernel body is not
        # used directly.
        tl.store(dkwf_part_ptr + ptrs_k, g_kwf, mask=mask_k2)
        tl.store(dqeff_part_ptr + ptrs_k, g_qeff, mask=mask_k2)
        tl.store(
            dn_part_ptr
            + chunk * BC * BC
            + offs_c[:, None] * BC
            + offs_c[None, :],
            g_n,
        )
        tl.store(ddec_part_ptr + chunk * K + offs_k, g_dec, mask=mask_k)

        gs = (
            gs * dec_exp[:, None]
            - tl.dot(tl.trans(b_w), g_u, input_precision="tf32")
            + tl.dot(tl.trans(b_qeff), b_do, input_precision="tf32")
        )


@triton.jit
def _bwd_state_kernel2(
    w_ptr,
    kwf_ptr,
    qeff_ptr,
    dec_ptr,
    u_ptr,
    nmat_ptr,
    sbound_ptr,
    do_ptr,
    du0_ptr,
    dw_part_ptr,
    dkwf_part_ptr,
    dqeff_part_ptr,
    dn_part_ptr,
    ddec_part_ptr,
    T,
    K,
    V,
    NT,
    NV,
    BH,
    scale,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // NV
    pid_vb = pid % NV

    offs_c = tl.arange(0, BC)
    offs_k = tl.arange(0, BK)
    offs_v = pid_vb * BV + tl.arange(0, BV)

    mask_k = offs_k < K
    mask_v = offs_v < V

    base_k = pid_bh * T * K
    base_v = pid_bh * T * V

    part_k_stride = BH * T * K
    part_n_stride = BH * NT * BC * BC
    part_d_stride = BH * NT * K

    # Gradient w.r.t. the state leaving a chunk, swept in reverse.
    gs = tl.zeros([BK, BV], dtype=tl.float32)

    for rev in range(NT):
        chunk = NT - 1 - rev
        idx_t = chunk * BC + offs_c
        mask_t = idx_t < T
        mask_k2 = mask_t[:, None] & mask_k[None, :]
        mask_v2 = mask_t[:, None] & mask_v[None, :]

        ptrs_k = base_k + idx_t[:, None] * K + offs_k[None, :]
        ptrs_v = base_v + idx_t[:, None] * V + offs_v[None, :]

        b_w = tl.load(w_ptr + ptrs_k, mask=mask_k2, other=0.0).to(tl.float32)
        b_kwf = tl.load(kwf_ptr + ptrs_k, mask=mask_k2, other=0.0).to(tl.float32)
        b_qeff = tl.load(qeff_ptr + ptrs_k, mask=mask_k2, other=0.0).to(tl.float32)
        b_u0 = tl.load(u_ptr + ptrs_v, mask=mask_v2, other=0.0).to(tl.float32)
        b_do = tl.load(do_ptr + ptrs_v, mask=mask_v2, other=0.0).to(tl.float32)
        b_do = b_do * scale

        n_mat = tl.load(
            nmat_ptr
            + pid_bh * NT * BC * BC
            + chunk * BC * BC
            + offs_c[:, None] * BC
            + offs_c[None, :]
        ).to(tl.float32)

        s_in = tl.load(
            sbound_ptr
            + pid_bh * NT * K * V
            + chunk * K * V
            + offs_k[:, None] * V
            + offs_v[None, :],
            mask=mask_k[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)

        b_dec = tl.load(
            dec_ptr + pid_bh * NT * K + chunk * K + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        dec_exp = tl.exp(b_dec)

        act_u = b_u0 - tl.dot(b_w, s_in, input_precision="tf32")
        g_u = tl.dot(b_kwf, gs, input_precision="tf32")
        g_u0 = g_u + tl.dot(tl.trans(n_mat), b_do, input_precision="tf32")

        tl.store(du0_ptr + ptrs_v, g_u0, mask=mask_v2)

        g_w = -tl.dot(g_u0, tl.trans(s_in), input_precision="tf32")
        g_kwf = tl.dot(act_u, tl.trans(gs), input_precision="tf32")
        g_qeff = tl.dot(b_do, tl.trans(s_in), input_precision="tf32")
        g_n = tl.dot(b_do, tl.trans(act_u), input_precision="tf32")
        g_dec = tl.sum(gs * (s_in * dec_exp[:, None]), axis=1)

        pk_ptrs = pid_vb * part_k_stride + ptrs_k
        tl.store(dw_part_ptr + pk_ptrs, g_w, mask=mask_k2)
        tl.store(dkwf_part_ptr + pk_ptrs, g_kwf, mask=mask_k2)
        tl.store(dqeff_part_ptr + pk_ptrs, g_qeff, mask=mask_k2)
        tl.store(
            dn_part_ptr
            + pid_vb * part_n_stride
            + pid_bh * NT * BC * BC
            + chunk * BC * BC
            + offs_c[:, None] * BC
            + offs_c[None, :],
            g_n,
        )
        tl.store(
            ddec_part_ptr
            + pid_vb * part_d_stride
            + pid_bh * NT * K
            + chunk * K
            + offs_k,
            g_dec,
            mask=mask_k,
        )

        gs = (
            gs * dec_exp[:, None]
            - tl.dot(tl.trans(b_w), g_u, input_precision="tf32")
            + tl.dot(tl.trans(b_qeff), b_do, input_precision="tf32")
        )


@triton.jit
def _bwd_inputs_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    gw_ptr,
    gkwf_ptr,
    gqeff_ptr,
    gn_ptr,
    gdec_ptr,
    du0_ptr,
    invmat_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dg_ptr,
    dbeta_ptr,
    h_ptr,
    T,
    K,
    V,
    NT,
    BC: tl.constexpr,
    BKB: tl.constexpr,
    BK2: tl.constexpr,
    BV2: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // NT
    pid_nt = pid % NT

    offs_c = tl.arange(0, BC)
    idx_t = pid_nt * BC + offs_c
    mask_t = idx_t < T

    base_k = pid_bh * T * K
    base_v = pid_bh * T * V
    base_b = pid_bh * T

    b_beta = tl.load(beta_ptr + base_b + idx_t, mask=mask_t, other=0.0).to(
        tl.float32
    )

    inv_mat = tl.load(
        invmat_ptr
        + pid_bh * NT * BC * BC
        + pid_nt * BC * BC
        + offs_c[:, None] * BC
        + offs_c[None, :]
    ).to(tl.float32)
    gn_mat = tl.load(
        gn_ptr
        + pid_bh * NT * BC * BC
        + pid_nt * BC * BC
        + offs_c[:, None] * BC
        + offs_c[None, :]
    ).to(tl.float32)

    loc_row = offs_c[:, None]
    loc_col = offs_c[None, :]

    dinv = tl.zeros([BC, BC], dtype=tl.float32)
    dbeta_acc = tl.zeros([BC], dtype=tl.float32)

    # ---- V blocks: dv, and contributions to dInv and dbeta.
    for vb in range(0, V, BV2):
        offs_vb = vb + tl.arange(0, BV2)
        mask_vb = offs_vb < V
        mask_v2 = mask_t[:, None] & mask_vb[None, :]
        ptrs_v = base_v + idx_t[:, None] * V + offs_vb[None, :]

        b_v = tl.load(v_ptr + ptrs_v, mask=mask_v2, other=0.0).to(tl.float32)
        g_u0 = tl.load(du0_ptr + ptrs_v, mask=mask_v2, other=0.0).to(tl.float32)

        d_xv = tl.dot(tl.trans(inv_mat), g_u0, input_precision="tf32")
        tl.store(dv_ptr + ptrs_v, b_beta[:, None] * d_xv, mask=mask_v2)
        dbeta_acc += tl.sum(d_xv * b_v, axis=1)
        dinv += tl.dot(
            g_u0, tl.trans(b_beta[:, None] * b_v), input_precision="tf32"
        )

    # ---- K blocks: pre-gradients for q/k and the cumsum adjoint seed.
    for kb in range(0, K, BK2):
        offs_kb = kb + tl.arange(0, BK2)
        mask_kb = offs_kb < K
        mask_k2 = mask_t[:, None] & mask_kb[None, :]
        ptrs = base_k + idx_t[:, None] * K + offs_kb[None, :]

        b_k = tl.load(k_ptr + ptrs, mask=mask_k2, other=0.0).to(tl.float32)
        b_q = tl.load(q_ptr + ptrs, mask=mask_k2, other=0.0).to(tl.float32)
        b_g = tl.load(g_ptr + ptrs, mask=mask_k2, other=0.0).to(tl.float32)
        g_w = tl.load(gw_ptr + ptrs, mask=mask_k2, other=0.0).to(tl.float32)
        g_kwf = tl.load(gkwf_ptr + ptrs, mask=mask_k2, other=0.0).to(tl.float32)
        g_qeff = tl.load(gqeff_ptr + ptrs, mask=mask_k2, other=0.0).to(tl.float32)

        b_cum = tl.cumsum(b_g, axis=0)
        b_cump = b_cum - b_g
        cum_last = tl.sum(b_g, axis=0)

        b_kp = b_k * tl.exp(b_cump)
        b_qf = b_q * tl.exp(b_cum)
        b_kwf = b_k * tl.exp(cum_last[None, :] - b_cump)
        x_k = b_beta[:, None] * b_kp

        d_xk = tl.dot(tl.trans(inv_mat), g_w, input_precision="tf32")
        dinv += tl.dot(g_w, tl.trans(x_k), input_precision="tf32")
        dbeta_acc += tl.sum(d_xk * b_kp, axis=1)

        d_kp = b_beta[:, None] * d_xk
        dk_pre = d_kp * tl.exp(b_cump) + g_kwf * tl.exp(
            cum_last[None, :] - b_cump
        )
        dq_pre = g_qeff * tl.exp(b_cum)
        h_pre = d_kp * b_kp - g_kwf * b_kwf + g_qeff * b_qf

        tl.store(dk_ptr + ptrs, dk_pre, mask=mask_k2)
        tl.store(dq_ptr + ptrs, dq_pre, mask=mask_k2)
        tl.store(h_ptr + ptrs, h_pre, mask=mask_k2)

    # ---- Backprop through Inv = (I + A)^-1:
    # dA = -Inv^T dInv Inv^T.
    tmp = tl.dot(dinv, tl.trans(inv_mat), input_precision="tf32")
    d_a = -tl.dot(tl.trans(inv_mat), tmp, input_precision="tf32")

    strict = loc_row > loc_col
    g_m = tl.where(strict, d_a, 0.0) * b_beta[:, None]
    g_n = tl.where(loc_row >= loc_col, gn_mat, 0.0)
    d_a_strict = tl.where(strict, d_a, 0.0)
    suffix_mat = tl.where(loc_col > loc_row, 1.0, 0.0)

    dbeta_a = tl.zeros([BC], dtype=tl.float32)

    # ---- Gram backprop over small K sub-blocks.
    for kb in range(0, K, BKB):
        offs_ks = kb + tl.arange(0, BKB)
        mask_ks = offs_ks < K
        mask_s = mask_t[:, None] & mask_ks[None, :]
        ptrs = base_k + idx_t[:, None] * K + offs_ks[None, :]

        ks = tl.load(k_ptr + ptrs, mask=mask_s, other=0.0).to(tl.float32)
        qs = tl.load(q_ptr + ptrs, mask=mask_s, other=0.0).to(tl.float32)
        gs_ = tl.load(g_ptr + ptrs, mask=mask_s, other=0.0).to(tl.float32)
        g_qeff = tl.load(gqeff_ptr + ptrs, mask=mask_s, other=0.0).to(tl.float32)
        g_kwf = tl.load(gkwf_ptr + ptrs, mask=mask_s, other=0.0).to(tl.float32)

        cum_s = tl.cumsum(gs_, axis=0)
        cump_s = cum_s - gs_
        cum_last_s = tl.sum(gs_, axis=0)

        diff = tl.expand_dims(cump_s, 1) - tl.expand_dims(cump_s, 0)
        diff = tl.minimum(diff, 0.0)
        e_mat = tl.exp(diff)

        a_q = qs * tl.exp(gs_)
        ks_i = tl.expand_dims(ks, 1)
        ks_j = tl.expand_dims(ks, 0)
        a_i = tl.expand_dims(a_q, 1)
        gm3 = tl.expand_dims(g_m, 2)
        gn3 = tl.expand_dims(g_n, 2)

        dk_gram = tl.sum(gm3 * ks_j * e_mat, axis=1)
        dk_gram += tl.sum(gm3 * ks_i * e_mat, axis=0)
        dk_gram += tl.sum(gn3 * a_i * e_mat, axis=0)

        q_sum = tl.sum(gn3 * ks_j * e_mat, axis=1)
        dq_gram = tl.exp(gs_) * q_sum
        dg_a = a_q * q_sum

        d_d = e_mat * ks_j * (gm3 * ks_i + gn3 * a_i)
        d_cump = tl.sum(d_d, axis=1) - tl.sum(d_d, axis=0)

        prod_a = tl.expand_dims(d_a_strict, 2) * ks_i * ks_j * e_mat
        dbeta_a += tl.sum(tl.sum(prod_a, axis=2), axis=1)

        dk_pre = tl.load(dk_ptr + ptrs, mask=mask_s, other=0.0).to(tl.float32)
        dq_pre = tl.load(dq_ptr + ptrs, mask=mask_s, other=0.0).to(tl.float32)
        h_pre = tl.load(h_ptr + ptrs, mask=mask_s, other=0.0).to(tl.float32)

        tl.store(dk_ptr + ptrs, dk_pre + dk_gram, mask=mask_s)
        tl.store(dq_ptr + ptrs, dq_pre + dq_gram, mask=mask_s)

        h_tot = h_pre + d_cump
        suffix = tl.dot(suffix_mat, h_tot, input_precision="ieee")

        qf_s = qs * tl.exp(cum_s)
        hc_qf = g_qeff * qf_s
        kwf_s = ks * tl.exp(cum_last_s[None, :] - cump_s)
        g_dec = tl.load(
            gdec_ptr + pid_bh * NT * K + pid_nt * K + offs_ks,
            mask=mask_ks,
            other=0.0,
        ).to(tl.float32)
        last_grad = g_dec + tl.sum(g_kwf * kwf_s, axis=0)

        dg_block = suffix + last_grad[None, :] + hc_qf + dg_a
        tl.store(dg_ptr + ptrs, dg_block, mask=mask_s)

    tl.store(
        dbeta_ptr + base_b + idx_t,
        dbeta_acc + dbeta_a,
        mask=mask_t,
    )


def _forward_impl(q_, k_, v_, g_, beta_, scale, BC=32):
    BH, T, K = q_.shape
    V = v_.shape[-1]
    device = q_.device
    NT = (T + BC - 1) // BC

    w = torch.empty((BH, T, K), device=device, dtype=torch.float32)
    kwf = torch.empty((BH, T, K), device=device, dtype=torch.float32)
    qeff = torch.empty((BH, T, K), device=device, dtype=torch.float32)
    u0 = torch.empty((BH, T, V), device=device, dtype=torch.float32)
    o0 = torch.empty((BH, T, V), device=device, dtype=torch.float32)
    out = torch.empty((BH, T, V), device=device, dtype=torch.float32)
    dec = torch.empty((BH, NT, K), device=device, dtype=torch.float32)
    nmat = torch.empty((BH, NT, BC, BC), device=device, dtype=torch.float32)
    invmat = torch.empty((BH, NT, BC, BC), device=device, dtype=torch.float32)
    sbound = torch.empty((BH, NT, K, V), device=device, dtype=torch.float32)

    _prep_kernel[(BH * NT,)](
        q_, k_, v_, g_, beta_,
        w, kwf, qeff, dec, u0, o0, nmat, invmat,
        T, K, V, NT,
        BC=BC, BKB=8, BK2=32, BV2=32,
        num_warps=8, num_stages=1,
    )

    BK = max(16, triton.next_power_of_2(K))
    BV = 64
    NV = (V + BV - 1) // BV

    _state_kernel[(BH * NV,)](
        w, kwf, qeff, dec, u0, o0, out, sbound,
        T, K, V, NV, NT,
        float(scale),
        BC=BC, BK=BK, BV=BV,
        num_warps=8, num_stages=1,
    )
    return out, (w, kwf, qeff, dec, u0, o0, nmat, invmat, sbound, NT)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        device = q.device

        q_ = q.contiguous().reshape(BH, T, K)
        k_ = k.contiguous().reshape(BH, T, K)
        v_ = v.contiguous().reshape(BH, T, V)
        g_ = g.contiguous().reshape(BH, T, K)
        beta_ = beta.contiguous().reshape(BH, T)

        if T == 0 or B == 0 or H == 0:
            out = torch.empty((BH, T, V), device=device, dtype=torch.float32)
            saved = None
            NT = 0
        else:
            out, saved = _forward_impl(q_, k_, v_, g_, beta_, float(scale))

        ctx.save_for_backward(
            q_, k_, v_, g_, beta_,
            *(saved[:-1] if saved is not None else []),
        )
        ctx.saved_ok = saved is not None
        ctx.NT = saved[-1] if saved is not None else 0
        ctx.BC = 32
        ctx.scale = float(scale)
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
        ctx.e_shape, ctx.e_dtype = e.shape, e.dtype
        ctx.gamma_shape, ctx.gamma_dtype = gamma.shape, gamma.dtype
        ctx.q_dtype, ctx.k_dtype, ctx.v_dtype = q.dtype, k.dtype, v.dtype
        ctx.g_dtype, ctx.beta_dtype = g.dtype, beta.dtype
        return out.reshape(B, H, T, V)

    @staticmethod
    def backward(ctx, grad_out):
        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        BH = B * H
        device = grad_out.device

        dq = torch.zeros((B, H, T, K), device=device, dtype=ctx.q_dtype)
        dk = torch.zeros((B, H, T, K), device=device, dtype=ctx.k_dtype)
        dv = torch.zeros((B, H, T, V), device=device, dtype=ctx.v_dtype)
        dg = torch.zeros((B, H, T, K), device=device, dtype=ctx.g_dtype)
        dbeta = torch.zeros((B, H, T), device=device, dtype=ctx.beta_dtype)
        de = torch.zeros(ctx.e_shape, device=device, dtype=ctx.e_dtype)
        dgamma = torch.zeros(
            ctx.gamma_shape, device=device, dtype=ctx.gamma_dtype
        )

        if not ctx.saved_ok or T == 0:
            return dq, dk, dv, de, dg, dbeta, dgamma, None

        saved = ctx.saved_tensors
        q_, k_, v_, g_, beta_ = saved[0], saved[1], saved[2], saved[3], saved[4]
        w, kwf, qeff, dec, u0, o0, nmat, invmat, sbound = saved[5:14]
        NT = ctx.NT
        BC = ctx.BC

        do_ = grad_out.contiguous().reshape(BH, T, V).to(torch.float32)

        # ---- Reverse state sweep (partials over V blocks).
        BV = 64
        NV = (V + BV - 1) // BV
        BK = max(16, triton.next_power_of_2(K))

        du0 = torch.empty((BH, T, V), device=device, dtype=torch.float32)
        dw_part = torch.empty((NV, BH, T, K), device=device, dtype=torch.float32)
        dkwf_part = torch.empty((NV, BH, T, K), device=device, dtype=torch.float32)
        dqeff_part = torch.empty((NV, BH, T, K), device=device, dtype=torch.float32)
        dn_part = torch.empty(
            (NV, BH, NT, BC, BC), device=device, dtype=torch.float32
        )
        ddec_part = torch.empty((NV, BH, NT, K), device=device, dtype=torch.float32)

        _bwd_state_kernel2[(BH * NV,)](
            w, kwf, qeff, dec, u0, nmat, sbound, do_,
            du0, dw_part, dkwf_part, dqeff_part, dn_part, ddec_part,
            T, K, V, NT, NV, BH,
            float(ctx.scale),
            BC=BC, BK=BK, BV=BV,
            num_warps=8, num_stages=1,
        )

        # Cheap reduction glue over V blocks.
        gw = dw_part.sum(0)
        gkwf = dkwf_part.sum(0)
        gqeff = dqeff_part.sum(0)
        gn = dn_part.sum(0)
        gdec = ddec_part.sum(0)

        # ---- Per-chunk input gradients.
        dq_ = torch.empty((BH, T, K), device=device, dtype=torch.float32)
        dk_ = torch.empty((BH, T, K), device=device, dtype=torch.float32)
        dv_ = torch.empty((BH, T, V), device=device, dtype=torch.float32)
        dg_ = torch.empty((BH, T, K), device=device, dtype=torch.float32)
        dbeta_ = torch.empty((BH, T), device=device, dtype=torch.float32)
        h_scratch = torch.empty((BH, T, K), device=device, dtype=torch.float32)

        _bwd_inputs_kernel[(BH * NT,)](
            q_, k_, v_, g_, beta_,
            gw, gkwf, gqeff, gn, gdec, du0, invmat,
            dq_, dk_, dv_, dg_, dbeta_, h_scratch,
            T, K, V, NT,
            BC=BC, BKB=8, BK2=32, BV2=32,
            num_warps=4, num_stages=1,
        )

        dq = dq_.reshape(B, H, T, K).to(ctx.q_dtype)
        dk = dk_.reshape(B, H, T, K).to(ctx.k_dtype)
        dv = dv_.reshape(B, H, T, V).to(ctx.v_dtype)
        dg = dg_.reshape(B, H, T, K).to(ctx.g_dtype)
        dbeta = dbeta_.reshape(B, H, T).to(ctx.beta_dtype)
        # e and gamma are unused by the forward, hence exact zero gradients.
        return dq, dk, dv, de, dg, dbeta, dgamma, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _GatedDelta.apply(q, k, v, e, g, beta, gamma, float(scale))
