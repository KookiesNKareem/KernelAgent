```python
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

    q_base = q_ptr + b * sqb + h * sqh
    k_base = k_ptr + b * skb + h * skh
    v_base = v_ptr + b * svb + h * svh
    e_base = e_ptr + b * seb + h * seh
    g_base = g_ptr + b * sgb + h * sgh
    beta_base = beta_ptr + b * sbb + h * sbh
    gamma_base = gamma_ptr + b * scb + h * sch
    out_base = out_ptr + b * sob + h * soh

    # State slice owned by this program, in the original coordinates.
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    row_idx = offs_c[:, None]
    col_idx = offs_c[None, :]
    strict_lower = row_idx > col_idx
    lower_incl = row_idx >= col_idx
    is_mid = offs_c == (BLOCK_C // 2)
    is_last = offs_c == (BLOCK_C - 1)

    t_base = 0
    for _chunk in range(NUM_CHUNKS):
        mask_t = (t_base + offs_c) < T
        mask_ck = mask_t[:, None] & mask_k[None, :]

        g_mat = tl.load(
            g_base + offs_c[:, None] * sgt + offs_k[None, :] * sgk,
            mask=mask_ck, other=0.0,
        ).to(tl.float32)
        Gc = tl.cumsum(g_mat, axis=0)

        # Total (signed) cumulative-decay range in this chunk, including
        # the chunk-start point 0.  The tensor-core factorisation below
        # is only used when all its exponentials are safely in range.
        gc_hi = tl.max(Gc, axis=0)
        gc_lo = tl.min(Gc, axis=0)
        range_k = tl.maximum(gc_hi, 0.0) - tl.minimum(gc_lo, 0.0)
        range_all = tl.max(range_k, axis=0)

        if range_all > 50.0:
            # Exact per-step fallback for this chunk (extreme decay or
            # growth).  This reproduces the reference recurrence step
            # by step, so no reciprocal-decay factor can overflow or
            # turn a fully decayed coupling back into an O(1) one.
            for step in range(BLOCK_C):
                in_t = (t_base + step) < T
                kt = tl.load(k_base + step * skt + offs_k * skk,
                             mask=mask_k & in_t, other=0.0).to(tl.float32)
                vt = tl.load(v_base + step * svt + offs_v * svv,
                             mask=mask_v & in_t, other=0.0).to(tl.float32)
                et = tl.load(e_base + step * set_ + offs_k * sek,
                             mask=mask_k & in_t, other=0.0).to(tl.float32)
                qt = tl.load(q_base + step * sqt + offs_k * sqk,
                             mask=mask_k & in_t, other=0.0).to(tl.float32)
                gt = tl.load(g_base + step * sgt + offs_k * sgk,
                             mask=mask_k & in_t, other=0.0).to(tl.float32)
                bt = tl.load(beta_base + step * sbt,
                             mask=in_t, other=0.0).to(tl.float32)
                ct = tl.load(gamma_base + step * sct,
                             mask=in_t, other=0.0).to(tl.float32)

                S = S * tl.exp(gt)[:, None]
                pred = tl.sum(S * kt[:, None], axis=0)
                S = S + kt[:, None] * (bt * (vt - pred))[None, :]
                S = S + et[:, None] * (ct * vt)[None, :]
                ot = tl.sum(S * (qt * scale)[:, None], axis=0)
                tl.store(out_base + step * sot + offs_v * sov, ot,
                         mask=mask_v & in_t)
        else:
            mask_cv = mask_t[:, None] & mask_v[None, :]

            q_mat = tl.load(
                q_base + offs_c[:, None] * sqt + offs_k[None, :] * sqk,
                mask=mask_ck, other=0.0,
            ).to(tl.float32)
            k_mat = tl.load(
                k_base + offs_c[:, None] * skt + offs_k[None, :] * skk,
                mask=mask_ck, other=0.0,
            ).to(tl.float32)
            v_mat = tl.load(
                v_base + offs_c[:, None] * svt + offs_v[None, :] * svv,
                mask=mask_cv, other=0.0,
            ).to(tl.float32)
            e_mat = tl.load(
                e_base + offs_c[:, None] * set_ + offs_k[None, :] * sek,
                mask=mask_ck, other=0.0,
            ).to(tl.float32)
            beta = tl.load(beta_base + offs_c * sbt,
                           mask=mask_t, other=0.0).to(tl.float32)
            gamma = tl.load(gamma_base + offs_c * sct,
                            mask=mask_t, other=0.0).to(tl.float32)

            # Recentre the cumulative decay at the chunk midpoint.
            # With the range guard above, |Gc - R| <= 50 here.
            r_vec = tl.sum(tl.where(is_mid[:, None], Gc, 0.0), axis=0)
            diff = Gc - r_vec[None, :]
            exp_diff = tl.exp(diff)        # exp(G - R)
            exp_ndiff = tl.exp(-diff)      # exp(R - G)

            k_hat = k_mat * exp_diff
            k_pr = k_mat * exp_ndiff
            e_pr = e_mat * exp_ndiff
            q_til = q_mat * exp_diff * scale

            w_mat = gamma[:, None] * v_mat

            # Chunk-start state in centred coordinates: exp(R) * S.
            S_check = S * tl.exp(r_vec)[:, None]

            # Intra-chunk couplings and chunk-start contributions.
            a_mat = tl.dot(k_hat, tl.trans(k_pr))
            a_mat = tl.where(strict_lower, a_mat, 0.0)
            b_mat = tl.dot(k_hat, tl.trans(e_pr))
            b_mat = tl.where(strict_lower, b_mat, 0.0)

            inter_pred = tl.dot(k_hat, S_check)
            inter_out = tl.dot(q_til, S_check)
            aq_mat = tl.dot(q_til, tl.trans(k_pr))
            aq_mat = tl.where(lower_incl, aq_mat, 0.0)
            bq_mat = tl.dot(q_til, tl.trans(e_pr))
            bq_mat = tl.where(lower_incl, bq_mat, 0.0)

            # Solve (I + diag(beta) A) d = diag(beta) r.  With
            # M = -diag(beta) A strictly lower triangular (nilpotent),
            # (I - M)^-1 = prod_j (I + M^(2^j)); powers are squared
            # and applied one at a time so old powers are not kept live.
            r_mat = v_mat - inter_pred - tl.dot(b_mat, w_mat)
            d_mat = beta[:, None] * r_mat
            m_mat = -beta[:, None] * a_mat

            d_mat = d_mat + tl.dot(m_mat, d_mat)
            if BLOCK_C > 2:
                m2 = tl.dot(m_mat, m_mat)
                d_mat = d_mat + tl.dot(m2, d_mat)
                if BLOCK_C > 4:
                    m4 = tl.dot(m2, m2)
                    d_mat = d_mat + tl.dot(m4, d_mat)
                    if BLOCK_C > 8:
                        m8 = tl.dot(m4, m4)
                        d_mat = d_mat + tl.dot(m8, d_mat)
                        if BLOCK_C > 16:
                            m16 = tl.dot(m8, m8)
                            d_mat = d_mat + tl.dot(m16, d_mat)

            out_mat = inter_out + tl.dot(aq_mat, d_mat) \
                + tl.dot(bq_mat, w_mat)
            tl.store(
                out_base + offs_c[:, None] * sot + offs_v[None, :] * sov,
                out_mat, mask=mask_cv,
            )

            # End-of-chunk state, mapped back to original coordinates.
            s_check_end = S_check + tl.dot(tl.trans(k_pr), d_mat) \
                + tl.dot(tl.trans(e_pr), w_mat)
            exp_last = tl.sum(
                tl.where(is_last[:, None], exp_diff, 0.0), axis=0)
            S = s_check_end * exp_last[:, None]

        q_base += BLOCK_C * sqt
        k_base += BLOCK_C * skt
        v_base += BLOCK_C * svt
        e_base += BLOCK_C * set_
        g_base += BLOCK_C * sgt
        beta_base += BLOCK_C * sbt
        gamma_base += BLOCK_C * sct
        out_base += BLOCK_C * sot
        t_base += BLOCK_C


@triton.jit
def _recurrent_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    H, T,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    seb, seh, set_, sek,
    sgb, sgh, sgt, sgk,
    sbb, sbh, sbt,
    scb, sch, sct,
    sob, soh, sot, sov,
    scale,
    KDIM, VDIM,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    # Low-shared-memory exact recurrence; used only when K > 128,
    # where the chunked kernel's padded K block would be 256 wide.
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < KDIM
    mask_v = offs_v < VDIM

    q_ptrs = q_ptr + b * sqb + h * sqh + offs_k * sqk
    k_ptrs = k_ptr + b * skb + h * skh + offs_k * skk
    v_ptrs = v_ptr + b * svb + h * svh + offs_v * svv
    e_ptrs = e_ptr + b * seb + h * seh + offs_k * sek
    g_ptrs = g_ptr + b * sgb + h * sgh + offs_k * sgk
    out_ptrs = out_ptr + b * sob + h * soh + offs_v * sov
    beta_pos = beta_ptr + b * sbb + h * sbh
    gamma_pos = gamma_ptr + b * scb + h * sch

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for _t in range(T):
        kt = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        vt = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
        et = tl.load(e_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        qt = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        gt = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        bt = tl.load(beta_pos).to(tl.float32)
        ct = tl.load(gamma_pos).to(tl.float32)

        S = S * tl.exp(gt)[:, None]
        pred = tl.sum(S * kt[:, None], axis=0)
        S = S + kt[:, None] * (bt * (vt - pred))[None, :]
        S = S + et[:, None] * (ct * vt)[None, :]
        ot = tl.sum(S * (qt * scale)[:, None], axis=0)
        tl.store(out_ptrs, ot, mask=mask_v)

        q_ptrs += sqt
        k_ptrs += skt
        v_ptrs += svt
        e_ptrs += set_
        g_ptrs += sgt
        out_ptrs += sot
        beta_pos += sbt
        gamma_pos += sct


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    if B == 0 or H == 0 or T == 0 or V == 0 or K == 0:
        return torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    fscale = float(scale)

    if K > 128:
        BLOCK_K = triton.next_power_of_2(K)
        BLOCK_V = 32 if BLOCK_K <= 256 else 16
        grid = (B * H, (V + BLOCK_V - 1) // BLOCK_V)
        _recurrent_kernel[grid](
            q, k, v, e, g, beta, gamma, out,
            H, T,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            e.stride(0), e.stride(1), e.stride(2), e.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            fscale,
            K, V,
            BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
            num_warps=4, num_stages=1,
        )
        return out

    # The shared-memory fix: BLOCK_C 32 -> 16 halves every [C, K] and
    # [C, V] tile, and num_stages 2 -> 1 removes the second pipelined
    # copy of all load tiles (previously 143616 > 101376 bytes).
    BLOCK_K = max(16, triton.next_power_of_2(K))
    BLOCK_C = 16
    BLOCK_V = 64 if V >= 64 else 32
    num_warps = 8 if BLOCK_V == 64 else 4

    num_chunks = (T + BLOCK_C - 1) // BLOCK_C
    grid = (B * H, (V + BLOCK_V - 1) // BLOCK_V)

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
        fscale,
        T, K, V,
        BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        num_warps=num_warps, num_stages=1,
    )
    return out
```

The failing assumption was resource sizing, not the chunk algebra: `BLOCK_C=32, BLOCK_V=64, num_stages=2` made Triton allocate 143616 bytes of shared memory against the L4’s 101376-byte limit, so every case failed at launch. The fix halves the chunk length to 16, uses a single pipeline stage, keeps squared triangular-solve factors from staying simultaneously live, and routes `K>128` (whose padded block would be 256 wide) to a low-shared-memory exact recurrent kernel. As a guard exposed by the adversarial cases, chunks whose cumulative decay range exceeds 50 nats also fall back to the exact per-step update, since reciprocal decay factors there would otherwise overflow or resurrect fully decayed couplings.