```python
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
    u_ptr,
    o_ptr,
    T,
    K,
    V,
    NT,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // NT
    pid_nt = pid % NT

    offs_c = tl.arange(0, BC)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)

    idx_t = pid_nt * BC + offs_c
    mask_t = idx_t < T
    mask_k = offs_k < K
    mask_v = offs_v < V

    base_k = pid_bh * T * K
    base_v = pid_bh * T * V
    base_b = pid_bh * T

    ptrs_k = k_ptr + base_k + idx_t[:, None] * K + offs_k[None, :]
    ptrs_q = q_ptr + base_k + idx_t[:, None] * K + offs_k[None, :]
    ptrs_g = g_ptr + base_k + idx_t[:, None] * K + offs_k[None, :]
    ptrs_v = v_ptr + base_v + idx_t[:, None] * V + offs_v[None, :]

    mask_k2 = mask_t[:, None] & mask_k[None, :]
    mask_v2 = mask_t[:, None] & mask_v[None, :]

    b_k = tl.load(ptrs_k, mask=mask_k2, other=0.0).to(tl.float32)
    b_q = tl.load(ptrs_q, mask=mask_k2, other=0.0).to(tl.float32)
    b_g = tl.load(ptrs_g, mask=mask_k2, other=0.0).to(tl.float32)
    b_v = tl.load(ptrs_v, mask=mask_v2, other=0.0).to(tl.float32)
    b_beta = tl.load(beta_ptr + base_b + idx_t, mask=mask_t, other=0.0).to(
        tl.float32
    )

    # Cumulative decay inside this chunk.  cum_prev is the decay accumulated
    # strictly before the current time step.
    b_cum = tl.cumsum(b_g, axis=0)
    b_cum_prev = b_cum - b_g

    b_kp = b_k * tl.exp(b_cum_prev)          # keys as seen by incoming state
    b_qf = b_q * tl.exp(b_cum)               # queries as seen in checked state
    cum_last = tl.sum(b_g, axis=0)
    b_kwf = b_k * tl.exp(cum_last[None, :] - b_cum_prev)  # keys for final state

    # Intra-chunk interaction matrices:
    #   M[t, j] = sum_k k[t,k] k[j,k] exp(cum_prev[t,k] - cum_prev[j,k])
    #   N[t, j] = sum_k q[t,k] k[j,k] exp(cum[t,k]     - cum_prev[j,k])
    # For the entries that are actually used (j < t for M, j <= t for N) the
    # exponent is <= 0 because g <= 0; clamping makes the discarded entries
    # harmless as well.
    m_mat = tl.zeros([BC, BC], dtype=tl.float32)
    n_mat = tl.zeros([BC, BC], dtype=tl.float32)

    BKB: tl.constexpr = 16
    for kb in range(0, BK, BKB):
        offs_ks = kb + tl.arange(0, BKB)
        mask_ks = offs_ks < K
        mask_s = mask_t[:, None] & mask_ks[None, :]

        ks = tl.load(
            k_ptr + base_k + idx_t[:, None] * K + offs_ks[None, :],
            mask=mask_s,
            other=0.0,
        ).to(tl.float32)
        qs = tl.load(
            q_ptr + base_k + idx_t[:, None] * K + offs_ks[None, :],
            mask=mask_s,
            other=0.0,
        ).to(tl.float32)
        gs = tl.load(
            g_ptr + base_k + idx_t[:, None] * K + offs_ks[None, :],
            mask=mask_s,
            other=0.0,
        ).to(tl.float32)

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

    a_s = tl.where(loc_row > loc_col, m_mat, 0.0) * b_beta[:, None]
    n_mat = tl.where(loc_row >= loc_col, n_mat, 0.0)
    eye = tl.where(loc_row == loc_col, 1.0, 0.0)

    # Inverse of the unit lower-triangular matrix (I + a_s).  a_s is strictly
    # lower triangular, hence nilpotent with a_s**BC == 0, and
    #   (I + a_s)^-1 = (I - a_s)(I + a_s^2)(I + a_s^4)(I + a_s^8)(I + a_s^16)
    # for BC = 32.
    inv_mat = eye - a_s
    power_mat = tl.dot(a_s, a_s, input_precision="ieee")
    for _ in range(4):
        inv_mat = tl.dot(inv_mat, eye + power_mat, input_precision="ieee")
        power_mat = tl.dot(power_mat, power_mat, input_precision="ieee")

    # WY-style split of the corrected values:
    #   U = u0 - w_mat @ S_in
    #   output / scale = o0 + qeff @ S_in
    u0 = tl.dot(
        inv_mat, b_beta[:, None] * b_v, input_precision="ieee"
    )
    w_mat = tl.dot(
        inv_mat, b_beta[:, None] * b_kp, input_precision="ieee"
    )
    o0 = tl.dot(n_mat, u0, input_precision="ieee")
    qeff = b_qf - tl.dot(n_mat, w_mat, input_precision="ieee")

    tl.store(w_ptr + base_k + idx_t[:, None] * K + offs_k[None, :], w_mat, mask=mask_k2)
    tl.store(kwf_ptr + base_k + idx_t[:, None] * K + offs_k[None, :], b_kwf, mask=mask_k2)
    tl.store(qeff_ptr + base_k + idx_t[:, None] * K + offs_k[None, :], qeff, mask=mask_k2)
    tl.store(u_ptr + base_v + idx_t[:, None] * V + offs_v[None, :], u0, mask=mask_v2)
    tl.store(o_ptr + base_v + idx_t[:, None] * V + offs_v[None, :], o0, mask=mask_v2)


@triton.jit
def _state_kernel(
    w_ptr,
    kwf_ptr,
    qeff_ptr,
    u_ptr,
    o_ptr,
    g_ptr,
    out_ptr,
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

    # Recurrent state block S[pid_bh, :, pid_vb block], kept across chunks.
    s = tl.zeros([BK, BV], dtype=tl.float32)

    for chunk in range(NT):
        idx_t = chunk * BC + offs_c
        mask_t = idx_t < T
        mask_k2 = mask_t[:, None] & mask_k[None, :]
        mask_v2 = mask_t[:, None] & mask_v[None, :]

        offs_k2 = base_k + idx_t[:, None] * K + offs_k[None, :]
        offs_v2 = base_v + idx_t[:, None] * V + offs_v[None, :]

        b_w = tl.load(w_ptr + offs_k2, mask=mask_k2, other=0.0).to(tl.float32)
        b_kwf = tl.load(kwf_ptr + offs_k2, mask=mask_k2, other=0.0).to(tl.float32)
        b_qeff = tl.load(qeff_ptr + offs_k2, mask=mask_k2, other=0.0).to(tl.float32)
        b_g = tl.load(g_ptr + offs_k2, mask=mask_k2, other=0.0).to(tl.float32)
        b_u = tl.load(u_ptr + offs_v2, mask=mask_v2, other=0.0).to(tl.float32)
        b_o0 = tl.load(o_ptr + offs_v2, mask=mask_v2, other=0.0).to(tl.float32)

        u_corr = b_u - tl.dot(b_w, s, input_precision="ieee")
        b_o = b_o0 + tl.dot(b_qeff, s, input_precision="ieee")

        tl.store(out_ptr + offs_v2, b_o * scale, mask=mask_v2)

        cum_last = tl.sum(b_g, axis=0)
        s = s * tl.exp(cum_last)[:, None] + tl.dot(
            tl.trans(b_kwf), u_corr, input_precision="ieee"
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # e and gamma are intentionally unused: the reference (and the stated
    # recurrence) does not use them, so they must not affect the result.
    B, H, T, K = q.shape
    V = v.shape[-1]

    if T == 0 or B == 0 or H == 0:
        return torch.empty(
            (B, H, T, V), device=q.device, dtype=torch.float32
        )

    BH = B * H
    device = q.device

    q_ = q.contiguous().reshape(BH, T, K)
    k_ = k.contiguous().reshape(BH, T, K)
    v_ = v.contiguous().reshape(BH, T, V)
    g_ = g.contiguous().reshape(BH, T, K)
    beta_ = beta.contiguous().reshape(BH, T)

    w = torch.empty((BH, T, K), device=device, dtype=torch.float32)
    kwf = torch.empty((BH, T, K), device=device, dtype=torch.float32)
    qeff = torch.empty((BH, T, K), device=device, dtype=torch.float32)
    u0 = torch.empty((BH, T, V), device=device, dtype=torch.float32)
    o0 = torch.empty((BH, T, V), device=device, dtype=torch.float32)
    out = torch.empty((BH, T, V), device=device, dtype=torch.float32)

    BC = 32
    NT = (T + BC - 1) // BC
    BK = max(16, triton.next_power_of_2(K))
    BV_FULL = max(16, triton.next_power_of_2(V))
    BV = 32 if V >= 32 else max(16, triton.next_power_of_2(V))
    NV = (V + BV - 1) // BV

    _prep_kernel[(BH * NT,)](
        q_,
        k_,
        v_,
        g_,
        beta_,
        w,
        kwf,
        qeff,
        u0,
        o0,
        T,
        K,
        V,
        NT,
        BC=BC,
        BK=BK,
        BV=BV_FULL,
        num_warps=8,
    )

    _state_kernel[(BH * NV,)](
        w,
        kwf,
        qeff,
        u0,
        o0,
        g_,
        out,
        T,
        K,
        V,
        NV,
        NT,
        float(scale),
        BC=BC,
        BK=BK,
        BV=BV,
        num_warps=4,
    )

    return out.reshape(B, H, T, V)
```

My approach is a chunkwise form of the gated delta rule: within each 32-step chunk I build the strictly-lower interaction matrices with exponents measured relative to each key's own decay origin (so all used exponents are non-positive), invert the resulting unit lower-triangular system by the nilpotent product $(I-A)(I+A^2)(I+A^4)\cdots$, and split the corrected values WY-style into a state-independent part plus a term linear in the incoming state. A second kernel then sweeps the chunks once per (batch-head, value-block), keeping the `[K, V]` state block in registers, applying the post-write decay exactly as in the reference, and emitting outputs as it goes. I used IEEE-precision dots for accuracy, so the main uncertainty is whether Triton's scheduling of the 3-D interaction reduction and the register-resident state is as fast on the L4 as estimated, and whether extremely large per-chunk decay ranges (far beyond realistic gate values) could stress the clamped-exponent construction.