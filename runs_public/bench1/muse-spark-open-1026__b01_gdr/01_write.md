```python
import torch
import triton
import triton.language as tl

# one chunk (BLOCK_C rows) of the gated delta rule.
# BH = B*H is grid, state [BH,K,V] is carried in HBM between chunks.
@triton.jit
def _chunk_kernel(
    q_ptr, k_ptr, v_ptr, logcum_ptr, beta_ptr, out_ptr, state_ptr, logprev_ptr,
    scale,
    T,                    # runtime int
    K: tl.constexpr, V: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_V: tl.constexpr,
    START: tl.constexpr,  # chunk start, passed as constexpr to avoid recompilation we keep it constexpr but wrapper loops
):
    pid = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)          # [C]
    offs_k = tl.arange(0, K)                # [K]
    offs_vb = tl.arange(0, BLOCK_V)         # [BLOCK_V]

    remaining = T - START
    mask_c = offs_c < remaining              # [C] valid rows in this chunk

    # ---- beta, logcum, D0, s ----
    beta = tl.load(beta_ptr + pid * T + START + offs_c, mask=mask_c, other=0.0)  # [C] fp32
    logcum = tl.load(logcum_ptr + pid * T + START + offs_c, mask=mask_c, other=0.0)  # [C]
    logprev = tl.load(logprev_ptr + pid)  # [1] scalar per program
    D0 = tl.exp(logcum - logprev)  # [C]  exp(logcum_i-logPrev) = prod d inside chunk
    D0 = tl.where(mask_c, D0, 0.0)
    # s_p = beta_p / D0_p  (0 for invalid rows)
    s_vec = tl.where(mask_c & (D0 != 0), beta / D0, 0.0)  # [C]

    # ---- K_chunk, Q_chunk  [C,K] ----
    # pointers: layout [BH,T,K] contiguous -> BH stride = T*K
    q_base = q_ptr + pid * T * K + START * K
    k_base = k_ptr + pid * T * K + START * K
    # 2-D pointers via broadcasting
    q_chunk = tl.load(
        q_base + offs_c[:, None] * K + offs_k[None, :],
        mask=mask_c[:, None], other=0.0
    ).to(tl.float32)  # [C,K]
    k_chunk = tl.load(
        k_base + offs_c[:, None] * K + offs_k[None, :],
        mask=mask_c[:, None], other=0.0
    ).to(tl.float32)  # [C,K]

    # ---- KK = K K^T , QK = Q K^T  [C,C] ----
    kk = tl.dot(k_chunk, tl.trans(k_chunk))  # [C,C]
    qk = tl.dot(q_chunk, tl.trans(k_chunk))  # [C,C]
    # diag of QK for the N update
    eye = (tl.arange(0, BLOCK_C)[:, None] == tl.arange(0, BLOCK_C)[None, :])
    qk_diag = tl.sum(qk * eye, axis=1)  # [C]

    # last index inside chunk for D_last
    last_idx = remaining - 1
    last_idx = tl.where(last_idx >= BLOCK_C, BLOCK_C - 1, last_idx)
    # D_last = D0[last_idx]
    last_mask = offs_c == last_idx
    D_last = tl.sum(D0 * last_mask)  # scalar

    # ---- loop over V tiles ----
    # V is constexpr, BLOCK_V constexpr => 4 iterations for 128/32
    for v_off in range(0, V, BLOCK_V):
        # v tile [C,BLOCK_V]
        v_base = v_ptr + pid * T * V + START * V + v_off
        v_tile = tl.load(
            v_base + offs_c[:, None] * V + offs_vb[None, :],
            mask=mask_c[:, None], other=0.0
        ).to(tl.float32)

        # state tile S0 [K,BLOCK_V]
        st_base = state_ptr + pid * K * V
        # K x BLOCK_V load
        offs_k2 = tl.arange(0, K)[:, None]  # [K,1]
        # offs_v2 = [1,BLOCK_V]
        s_tile = tl.load(
            st_base + offs_k2 * V + (v_off + offs_vb)[None, :],
            # all K rows are valid, mask only on V (always valid)
        ).to(tl.float32)  # [K,BLOCK_V]

        # p = K @ S0 , qs = Q @ S0  -> [C,BLOCK_V]
        p_tile = tl.dot(k_chunk, s_tile)
        qs_tile = tl.dot(q_chunk, s_tile)
        t_tile = v_tile - D0[:, None] * p_tile  # [C,BLOCK_V]

        # ---- forward solve (I+M) h = t ,  M_{i,p}=kk[i,p]*s_p*D0_i
        # we keep two accumulators acc_m / acc_n of shape [C,BLOCK_V]
        acc_m = tl.zeros((BLOCK_C, BLOCK_V), tl.float32)
        acc_n = tl.zeros((BLOCK_C, BLOCK_V), tl.float32)
        h = tl.zeros((BLOCK_C, BLOCK_V), tl.float32)
        out_tile = tl.zeros((BLOCK_C, BLOCK_V), tl.float32)

        # iterate p = 0 .. BLOCK_C-1  (constexpr unrolled)
        for p in range(BLOCK_C):
            is_p = tl.arange(0, BLOCK_C) == p  # [C]
            is_p_2d = is_p[:, None]  # [C,1]

            D0_p = tl.sum(D0 * is_p)          # scalar
            s_p = tl.sum(s_vec * is_p)        # scalar
            qk_diag_p = tl.sum(qk_diag * is_p)

            # rows for this p
            acc_m_p = tl.sum(acc_m * is_p_2d, axis=0)  # [BLOCK_V]
            acc_n_p = tl.sum(acc_n * is_p_2d, axis=0)
            t_p = tl.sum(t_tile * is_p_2d, axis=0)
            qs_p = tl.sum(qs_tile * is_p_2d, axis=0)

            h_p = t_p - D0_p * acc_m_p  # [BLOCK_V]
            # zero out invalid rows
            h_p = tl.where(remaining > p, h_p, 0.0)

            # write h row
            h = h + is_p_2d * h_p[None, :]

            # out_p = scale*D0_p*(qs_p + acc_n_p + qk_diag_p*s_p*h_p)
            out_p = D0_p * (qs_p + acc_n_p + qk_diag_p * s_p * h_p)
            out_p = out_p * scale
            out_p = tl.where(remaining > p, out_p, 0.0)
            out_tile = out_tile + is_p_2d * out_p[None, :]

            # column p of kk/qk for future accumulators
            col_mask = tl.arange(0, BLOCK_C)[None, :] == p  # [1,C]
            kk_col = tl.sum(kk * col_mask, axis=1)  # [C]
            qk_col = tl.sum(qk * col_mask, axis=1)  # [C]

            # acc_m += kk_col[:,None] * s_p * h_p ; acc_n += qk_col[:,None]*s_p*h_p
            acc_m = acc_m + kk_col[:, None] * s_p * h_p[None, :]
            acc_n = acc_n + qk_col[:, None] * s_p * h_p[None, :]

        # store out
        tl.store(
            out_ptr + pid * T * V + START * V + v_off + offs_c[:, None] * V + offs_vb[None, :],
            out_tile,
            mask=mask_c[:, None]
        )

        # ---- state update S_next = D_last*(S0 + K^T diag(s) h) ----
        # h_w = s_vec[:,None] * h   [C,BLOCK_V]
        h_w = s_vec[:, None] * h
        delta = tl.dot(tl.trans(k_chunk), h_w)  # [K,BLOCK_V]
        s_next = D_last * (s_tile + delta)
        tl.store(
            st_base + offs_k2 * V + (v_off + offs_vb)[None, :],
            s_next
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    """
    q,k,e: [B,H,T,K] bf16 unit-norm, v: [B,H,T,V] bf16, g: [B,H,T] fp32 <=0,
    beta,gamma: [B,H,T] fp32 (gamma unused), scale: float
    returns [B,H,T,V] (float32 or q.dtype)
    reference: S_t = exp(g_t)*(I-beta_t*k_t*k_t^T)*S_{t-1}+beta_t*k_t*v_t^T ; o_t=scale*S_t^T q_t
    """
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    BH = B * H
    # flatten B,H
    q_ = q.reshape(BH, T, K).contiguous()
    k_ = k.reshape(BH, T, K).contiguous()
    v_ = v.reshape(BH, T, Vd).contiguous()
    g_ = g.reshape(BH, T).contiguous().to(torch.float32)
    beta_ = beta.reshape(BH, T).contiguous().to(torch.float32)

    logcum = torch.cumsum(g_, dim=1)  # [BH,T] fp32

    out = torch.empty((BH, T, Vd), device=q.device, dtype=torch.float32)
    state = torch.zeros((BH, K, Vd), device=q.device, dtype=torch.float32)

    BLOCK_C = 32
    BLOCK_V = 32
    # ensure K,V are multiple of 32 for tl.dot (128 fits)
    grid = (BH,)

    # cheap scaling: pass scale as float
    scale_f = float(scale) if not isinstance(scale, float) else scale

    for start in range(0, T, BLOCK_C):
        if start == 0:
            logprev = torch.zeros((BH,), device=q.device, dtype=torch.float32)
        else:
            logprev = logcum[:, start - 1].contiguous()
        # Triton launch: START is constexpr for this launch (recompile per distinct START is ok,
        # we pass it as constexpr to keep addressing simple)
        _chunk_kernel[grid](
            q_, k_, v_, logcum, beta_, out, state, logprev,
            scale_f,
            T, K, Vd, BLOCK_C, BLOCK_V, start,
        )
    out = out.reshape(B, H, T, Vd)
    # return in q's dtype if requested (bf16) else float32 is also legal
    if q.dtype == torch.bfloat16:
        return out.to(torch.bfloat16)
    return out
```

This uses a chunked gated-delta formulation. By writing `S_t/d_t` we reduce the gated recurrence to the ungated delta rule, whose per-chunk effect is a lower-triangular system `(I+M_d)h=t` with `M_{i,p}= (k_i·k_p)β_p exp(cum_i-cum_p)`. The system is solved by forward substitution entirely in SRAM, the two GEMMs `K K^T` and `Q K^T` are done with `tl.dot`, and the state `S` is carried between chunks in HBM (`K×V` per head). Vector `V` is tiled (`32`) to keep SRAM <64KB, giving `O(T C V)` work with only `T/C` kernel launches.

Unsure: exact L4 SRAM limit and whether `tl.trans` + `tl.dot` with `64×128` shapes compiles on Triton 3.5 without `tl.make_block_ptr`; the intra-chunk sequential loop is written as fully unrolled `for p in range(BLOCK_C)` which inflates code size and register pressure – `BLOCK_C=32` was chosen to mitigate this but may still need hand-tuned `num_warps`/`num_stages`.
