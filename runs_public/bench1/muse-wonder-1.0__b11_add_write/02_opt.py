import torch
import triton
import triton.language as tl


@triton.jit
def _chunked_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    H, NUM_CHUNKS,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    seb, seh, set_, sek,
    sgb, sgh, sgt, sgk,
    sbb, sbh, sbt,
    scb, sch, sct,
    sob, soh, sot, sov,
    scale,
    T, KDIM, VDIM,
    BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, BLOCK_C)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    mask_k = offs_k < KDIM
    mask_v = offs_v < VDIM

    q_ptrs = q_ptr + b * sqb + h * sqh + offs_k[None, :] * sqk
    k_ptrs = k_ptr + b * skb + h * skh + offs_k[None, :] * skk
    v_ptrs = v_ptr + b * svb + h * svh + offs_v[None, :] * svv
    e_ptrs = e_ptr + b * seb + h * seh + offs_k[None, :] * sek
    g_ptrs = g_ptr + b * sgb + h * sgh + offs_k[None, :] * sgk
    beta_ptrs = beta_ptr + b * sbb + h * sbh
    gamma_ptrs = gamma_ptr + b * scb + h * sch
    out_ptrs = out_ptr + b * sob + h * soh + offs_v[None, :] * sov

    # Original (untransformed) state for this value block.
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    row_idx = offs_c[:, None]
    col_idx = offs_c[None, :]
    strict_lower = row_idx > col_idx
    lower_incl = row_idx >= col_idx

    t_base = 0
    for _chunk in range(NUM_CHUNKS):
        mask_t = (t_base + offs_c) < T
        mask_ck = mask_t[:, None] & mask_k[None, :]
        mask_cv = mask_t[:, None] & mask_v[None, :]

        q_mat = tl.load(q_ptrs + offs_c[:, None] * sqt,
                        mask=mask_ck, other=0.0).to(tl.float32)
        k_mat = tl.load(k_ptrs + offs_c[:, None] * skt,
                        mask=mask_ck, other=0.0).to(tl.float32)
        v_mat = tl.load(v_ptrs + offs_c[:, None] * svt,
                        mask=mask_cv, other=0.0).to(tl.float32)
        e_mat = tl.load(e_ptrs + offs_c[:, None] * set_,
                        mask=mask_ck, other=0.0).to(tl.float32)
        g_mat = tl.load(g_ptrs + offs_c[:, None] * sgt,
                        mask=mask_ck, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptrs + offs_c * sbt,
                       mask=mask_t, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptrs + offs_c * sct,
                        mask=mask_t, other=0.0).to(tl.float32)

        # Cumulative log-decay inside this chunk: Gc[t, i] <= 0 and
        # non-increasing in t.  Recentre it at the chunk midpoint so
        # that neither exp(G-R) nor exp(R-G) can be astronomically
        # large for ordinary decay strengths.
        Gc = tl.cumsum(g_mat, axis=0)
        is_mid = offs_c == (BLOCK_C // 2)
        r_vec = tl.sum(tl.where(is_mid[:, None], Gc, 0.0), axis=0)

        diff = tl.maximum(-80.0, tl.minimum(80.0, Gc - r_vec[None, :]))
        exp_diff = tl.exp(diff)          # exp(G - R)
        exp_ndiff = tl.exp(-diff)        # exp(R - G)

        # Transformed quantities (see derivation in the summary):
        #   k_hat  = exp(G-R) * k      (prediction / query side)
        #   k_pr   = exp(R-G) * k      (write side)
        #   e_pr   = exp(R-G) * e
        #   q_til  = exp(G-R) * q * scale
        k_hat = k_mat * exp_diff
        k_pr = k_mat * exp_ndiff
        e_pr = e_mat * exp_ndiff
        q_til = q_mat * exp_diff * scale

        w_mat = gamma[:, None] * v_mat   # values of the e-write

        # State at chunk start in the centred coordinates:
        #   S_check_0 = exp(R - G_0) * S = exp(R) * S.
        exp_r = tl.exp(tl.minimum(0.0, r_vec))
        S_check = S * exp_r[:, None]

        # Intra-chunk couplings.
        a_mat = tl.dot(k_hat, tl.trans(k_pr))
        a_mat = tl.where(strict_lower, a_mat, 0.0)
        b_mat = tl.dot(k_hat, tl.trans(e_pr))
        b_mat = tl.where(strict_lower, b_mat, 0.0)

        # Quantities that only need the chunk-start state.
        inter_pred = tl.dot(k_hat, S_check)
        inter_out = tl.dot(q_til, S_check)
        aq_mat = tl.dot(q_til, tl.trans(k_pr))
        aq_mat = tl.where(lower_incl, aq_mat, 0.0)
        bq_mat = tl.dot(q_til, tl.trans(e_pr))
        bq_mat = tl.where(lower_incl, bq_mat, 0.0)

        # Delta values d solve  (I + diag(beta) A) d = diag(beta) r.
        r_mat = v_mat - inter_pred - tl.dot(b_mat, w_mat)
        rhs = beta[:, None] * r_mat

        # (I - M)^-1 with M = -diag(beta) A strictly lower triangular:
        # nilpotent, so the inverse is prod_j (I + M^(2^j)), applied
        # to rhs factor by factor -- all small tensor-core dots.
        m_mat = -beta[:, None] * a_mat
        m2 = tl.dot(m_mat, m_mat)
        m4 = tl.dot(m2, m2)
        m8 = tl.dot(m4, m4)
        m16 = tl.dot(m8, m8)

        d_mat = rhs
        d_mat = d_mat + tl.dot(m_mat, d_mat)
        d_mat = d_mat + tl.dot(m2, d_mat)
        d_mat = d_mat + tl.dot(m4, d_mat)
        if BLOCK_C > 8:
            d_mat = d_mat + tl.dot(m8, d_mat)
        if BLOCK_C > 16:
            d_mat = d_mat + tl.dot(m16, d_mat)
        if BLOCK_C > 32:
            m32 = tl.dot(m16, m16)
            d_mat = d_mat + tl.dot(m32, d_mat)

        out_mat = inter_out + tl.dot(aq_mat, d_mat) + tl.dot(bq_mat, w_mat)
        tl.store(out_ptrs + offs_c[:, None] * sot, out_mat, mask=mask_cv)

        # End-of-chunk state, still in centred coordinates, then map
        # back:  S_end = exp(G_last - R) * S_check_end.
        s_check_end = S_check + tl.dot(tl.trans(k_pr), d_mat) \
            + tl.dot(tl.trans(e_pr), w_mat)
        is_last = offs_c == (BLOCK_C - 1)
        exp_last = tl.sum(tl.where(is_last[:, None], exp_diff, 0.0), axis=0)
        S = s_check_end * exp_last[:, None]

        q_ptrs += BLOCK_C * sqt
        k_ptrs += BLOCK_C * skt
        v_ptrs += BLOCK_C * svt
        e_ptrs += BLOCK_C * set_
        g_ptrs += BLOCK_C * sgt
        beta_ptrs += BLOCK_C * sbt
        gamma_ptrs += BLOCK_C * sct
        out_ptrs += BLOCK_C * sot
        t_base += BLOCK_C


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    if B == 0 or H == 0 or T == 0 or V == 0 or K == 0:
        return torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    BLOCK_K = max(16, triton.next_power_of_2(K))
    if BLOCK_K <= 64:
        BLOCK_C = 64
    elif BLOCK_K <= 256:
        BLOCK_C = 32
    else:
        BLOCK_C = 16

    if V >= 64:
        BLOCK_V = 64
    elif V >= 32:
        BLOCK_V = 32
    else:
        BLOCK_V = 16

    num_chunks = (T + BLOCK_C - 1) // BLOCK_C
    grid = (B * H, (V + BLOCK_V - 1) // BLOCK_V)
    num_warps = 8 if BLOCK_K >= 128 else 4

    _chunked_kernel[grid](
        q, k, v, e, g, beta, gamma, out,
        H, num_chunks,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        e.stride(0), e.stride(1), e.stride(2), e.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        float(scale),
        T, K, V,
        BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        num_warps=num_warps, num_stages=2,
    )
    return out
