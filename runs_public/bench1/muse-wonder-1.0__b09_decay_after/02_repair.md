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
    dec_ptr,
    u_ptr,
    o_ptr,
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
    # harmless.  Tiles here are only [BC, BC, BKB] = 16x16x16.
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

    # (I + A)^-1 for strictly-lower-triangular, hence nilpotent, A (BC=16):
    # (I - A)(I + A^2)(I + A^4)(I + A^8).
    inv_mat = eye - a_mat
    pow_mat = tl.dot(a_mat, a_mat, input_precision="ieee")
    for _ in range(3):
        inv_mat = tl.dot(inv_mat, eye + pow_mat, input_precision="ieee")
        pow_mat = tl.dot(pow_mat, pow_mat, input_precision="ieee")

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

        b_w = tl.dot(
            inv_mat, b_beta[:, None] * b_kp, input_precision="ieee"
        )
        b_qeff = b_qf - tl.dot(n_mat, b_w, input_precision="ieee")

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
            inv_mat, b_beta[:, None] * b_v, input_precision="ieee"
        )
        b_o0 = tl.dot(n_mat, b_u0, input_precision="ieee")

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

        u_corr = b_u - tl.dot(b_w, s, input_precision="ieee")
        b_o = b_o0 + tl.dot(b_qeff, s, input_precision="ieee")
        tl.store(out_ptr + ptrs_v, b_o * scale, mask=mask_v2)

        s = s * tl.exp(b_dec)[:, None] + tl.dot(
            tl.trans(b_kwf), u_corr, input_precision="ieee"
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # e and gamma are intentionally unused: neither the reference loop nor
    # the stated recurrence uses them, so they must not affect the result.
    B, H, T, K = q.shape
    V = v.shape[-1]

    if T == 0 or B == 0 or H == 0:
        return torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    BH = B * H
    device = q.device

    # .contiguous() also normalises every strided/transposed layout the
    # checker passes in; afterwards plain [BH, T, *] indexing is valid.
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

    BC = 16
    NT = (T + BC - 1) // BC
    dec = torch.empty((BH, NT, K), device=device, dtype=torch.float32)

    # Small tiles plus num_stages=1 keep shared memory far below the
    # 101376-byte L4 limit (the previous version staged [32,128]/[32,128x128]
    # operands with the default pipelining and needed 122880 bytes).
    _prep_kernel[(BH * NT,)](
        q_, k_, v_, g_, beta_,
        w, kwf, qeff, dec, u0, o0,
        T, K, V, NT,
        BC=BC, BKB=16, BK2=32, BV2=32,
        num_warps=4, num_stages=1,
    )

    BK = max(16, triton.next_power_of_2(K))
    BV = 32
    NV = (V + BV - 1) // BV

    _state_kernel[(BH * NV,)](
        w, kwf, qeff, dec, u0, o0, out,
        T, K, V, NV, NT,
        float(scale),
        BC=BC, BK=BK, BV=BV,
        num_warps=8, num_stages=1,
    )

    return out.reshape(B, H, T, V)
```

The failure was purely a resource issue, not a numerical one: the previous prep kernel used a 32-step chunk with full 128-wide K/V tiles (including a `[32, 128]` value tile and `[32, 32, 16]` interaction tensor) and Triton's default software pipelining, which together required 122880 bytes of shared memory against the L4's 101376-byte limit — and 237568 bytes once K padded to 256. The fix shrinks the chunk to 16 steps, computes the Gram matrices over 16-wide K sub-blocks, computes W/queries/outputs over independent 32-wide K and V sub-blocks (so no tile is ever wider than 32 in the prep kernel), precomputes each chunk's total decay as a single K-vector so the state kernel no longer stages a `[BC, K]` decay tile, and launches both kernels with `num_stages=1`. The chunked delta-rule math itself (relative-decay Gram matrices, nilpotent triangular inverse, WY split, and post-write decay in the state sweep) is unchanged.