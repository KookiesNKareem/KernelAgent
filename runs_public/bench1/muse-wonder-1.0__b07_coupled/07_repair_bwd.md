```python
import torch
import triton
import triton.language as tl


# Four real timesteps per chunk.  This bounds the spread of the chunk-local
# cumulative log-decay: even for g in [-25, -5], every exp(gc - r) and
# exp(r - gc) argument stays roughly within [-85, 70], instead of reaching
# several hundred (and overflowing float32) with 32-step chunks.
_CHUNK = 4
# The 8 real expanded sub-steps (erase+write for 4 steps) are padded to 16
# rows so that every tl.dot has M >= 16; padded rows have vec = beta = 0.
_BLOCK_N = 16


def _next_pow2(x):
    n = 1
    while n < x:
        n *= 2
    return n


@triton.jit
def _chunk_decay_kernel(
    g_ptr,
    gc_ptr,
    T,
    K,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_t = pid_c * CHUNK + tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BLOCK_K)
    mask = (offs_t[:, None] < T) & (offs_k[None, :] < K)

    base = pid_bh.to(tl.int64) * T * K
    offs = base + offs_t[:, None].to(tl.int64) * K + offs_k[None, :]
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Cumulative log-decay, local to this chunk.
    gc = tl.cumsum(g, axis=0)
    tl.store(gc_ptr + offs, gc, mask=mask)


@triton.jit
def _recurrent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    gc_ptr,
    beta_ptr,
    gamma_ptr,
    out_ptr,
    state_ptr,
    scale,
    T,
    K,
    V_DIM,
    NC,
    NV,
    CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SAVE_STATES: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_v = pid % NV
    pid_bh = pid // NV
    pid64 = pid_bh.to(tl.int64)

    offs_q = tl.arange(0, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    mask_k = offs_k < K
    mask_v = offs_v < V_DIM

    base_k = pid64 * T * K
    base_v = pid64 * T * V_DIM
    base_t = pid64 * T

    # Actual (untransformed) state at the start of the current chunk.
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for chunk in range(NC):
        t_start = chunk * CHUNK

        if SAVE_STATES:
            state_base = ((pid64 * NC + chunk) * K) * V_DIM
            state_offs = offs_k[:, None] * V_DIM + offs_v[None, :]
            tl.store(
                state_ptr + state_base + state_offs,
                S,
                mask=mask_k[:, None] & mask_v[None, :],
            )

        # Expanded sub-steps: 2*t is the erase step, 2*t+1 is the write step.
        # Rows >= 2*CHUNK are zero padding for tl.dot.
        t_rel_n = offs_n // 2
        t_n = t_start + t_rel_n
        mask_n = (offs_n < 2 * CHUNK) & (t_n < T)
        is_write = (offs_n % 2) == 1

        ptrs_nk = base_k + t_n[:, None].to(tl.int64) * K + offs_k[None, :]
        mask_nk = mask_n[:, None] & mask_k[None, :]

        k_exp = tl.load(k_ptr + ptrs_nk, mask=mask_nk, other=0.0).to(tl.float32)
        e_exp = tl.load(e_ptr + ptrs_nk, mask=mask_nk, other=0.0).to(tl.float32)
        vec = tl.where(is_write[:, None], k_exp, e_exp)

        gc_exp = tl.load(gc_ptr + ptrs_nk, mask=mask_nk, other=0.0).to(tl.float32)

        # Mid-chunk reference point for the decay-rescaled coordinates.
        r = tl.sum(
            tl.where(offs_n[:, None] == CHUNK, gc_exp, 0.0),
            axis=0,
        )
        w_scaled = vec * tl.exp(gc_exp - r[None, :])  # exp(l_i-r) * vec_i
        u_scaled = vec * tl.exp(r[None, :] - gc_exp)  # exp(r-l_i) * vec_i

        beta_t = tl.load(
            beta_ptr + base_t + t_n.to(tl.int64), mask=mask_n, other=0.0
        ).to(tl.float32)
        gamma_t = tl.load(
            gamma_ptr + base_t + t_n.to(tl.int64), mask=mask_n, other=0.0
        ).to(tl.float32)
        beta_step = tl.where(is_write, beta_t, gamma_t * (1.0 - beta_t))
        beta_step = tl.where(mask_n, beta_step, 0.0)

        v_exp = tl.load(
            v_ptr + base_v + t_n[:, None].to(tl.int64) * V_DIM + offs_v[None, :],
            mask=mask_n[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        v_star = tl.where(is_write[:, None], v_exp, 0.0)

        exp_r = tl.exp(r)
        S_r = S * exp_r[:, None]

        # Prediction from the chunk-initial state, in the decay-transformed
        # coordinates: P_i = (exp(l_i) vec_i)^T S_start.
        pred0 = tl.dot(w_scaled, S_r, input_precision="ieee")

        # Gram_ij = sum_k vec_i,k vec_j,k exp(l_i,k - l_j,k).
        gram = tl.dot(w_scaled, tl.trans(u_scaled), input_precision="ieee")
        lower = offs_n[None, :] < offs_n[:, None]
        l_mat = tl.where(lower, beta_step[:, None] * gram, 0.0)

        # (I + L) is unit lower triangular and L is nilpotent, so its inverse
        # is sum_{i=0}^{BLOCK_N-1} (-L)^i.  Evaluate by repeated doubling
        # (log2(16) = 4 squarings).
        inv_ser = tl.where(
            offs_n[:, None] == offs_n[None, :], 1.0, 0.0
        ).to(tl.float32)
        pow_x = -l_mat
        for _ in range(4):
            new_ser = inv_ser + tl.dot(pow_x, inv_ser, input_precision="ieee")
            new_pow = tl.dot(pow_x, pow_x, input_precision="ieee")
            inv_ser = new_ser
            pow_x = new_pow

        rhs = beta_step[:, None] * (v_star - pred0)
        v_hat = tl.dot(inv_ser, rhs, input_precision="ieee")

        # Outputs, one row per timestep of the chunk (rows >= CHUNK are
        # zero padding and are never stored).
        t_c = t_start + offs_q
        mask_c = (offs_q < CHUNK) & (t_c < T)
        ptrs_ck = base_k + t_c[:, None].to(tl.int64) * K + offs_k[None, :]
        mask_ck = mask_c[:, None] & mask_k[None, :]

        q_c = tl.load(q_ptr + ptrs_ck, mask=mask_ck, other=0.0).to(tl.float32)
        gc_c = tl.load(gc_ptr + ptrs_ck, mask=mask_ck, other=0.0).to(tl.float32)
        q_scaled = q_c * tl.exp(gc_c - r[None, :])

        out_val = tl.dot(q_scaled, S_r, input_precision="ieee")
        q_gram = tl.dot(q_scaled, tl.trans(u_scaled), input_precision="ieee")
        q_gram = tl.where(
            offs_n[None, :] <= (2 * offs_q[:, None] + 1), q_gram, 0.0
        )
        out_val = (out_val + tl.dot(q_gram, v_hat, input_precision="ieee")) * scale

        tl.store(
            out_ptr
            + base_v
            + t_c[:, None].to(tl.int64) * V_DIM
            + offs_v[None, :],
            out_val,
            mask=mask_c[:, None] & mask_v[None, :],
        )

        # Convert the transformed end-of-chunk state back to actual state:
        # S_end = exp(l_last) * S_start
        #         + sum_j exp(l_last-l_j) vec_j * v_hat_j.
        l_last = tl.sum(
            tl.where(offs_n[:, None] == (2 * CHUNK - 1), gc_exp, 0.0),
            axis=0,
        )
        state_delta = tl.dot(tl.trans(u_scaled), v_hat, input_precision="ieee")
        S = (
            S * tl.exp(l_last)[:, None]
            + state_delta * tl.exp(l_last - r)[:, None]
        )


# ---------------------------------------------------------------------------
# Backward kernel.
#
# Reverse-mode differentiation of the forward chunk computation above,
# chunk by chunk in reverse time.  For one chunk the forward map is
#   w = a*exp(l-r),   u = a*exp(r-l),   Sr = S_start*exp(r)
#   pred = w@Sr,      gram = w@u^T,     L = tril(beta*gram)
#   vhat = (I+L)^{-1} (beta*(v* - pred))
#   out/scale = q~@Sr + mask(q~@u^T)@vhat,   q~ = q*exp(lq-r)
#   S_end = S_start*exp(l_last) + (u^T@vhat)*exp(l_last-r)
# and every adjoint below is the transpose of one of those operations
# (for the inverse, dL = -inv^T dInv inv^T).  Adjoints w.r.t. the
# rescaled factors are then pushed through the exponentials onto the
# chunk-local cumulative decays l, lq; aggregating the erase/write rows
# of a timestep and reverse-cumsumming over the chunk gives d g.
# Shared (non-V) gradients are computed as per-V-tile partials and are
# summed across tiles by _reduce_kernel; d v is written directly.
# ---------------------------------------------------------------------------
@triton.jit
def _recurrent_backward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    gc_ptr,
    beta_ptr,
    gamma_ptr,
    go_ptr,
    state_ptr,
    gq_ptr,
    gk_ptr,
    gv_ptr,
    ge_ptr,
    gg_ptr,
    gbeta_ptr,
    ggamma_ptr,
    scale,
    T,
    K,
    V_DIM,
    NC,
    NV,
    NBH,
    CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_v = pid % NV
    pid_bh = pid // NV
    pid64 = pid_bh.to(tl.int64)
    pidv64 = pid_v.to(tl.int64)

    offs_q = tl.arange(0, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    mask_k = offs_k < K
    mask_v = offs_v < V_DIM

    base_k = pid64 * T * K
    base_v = pid64 * T * V_DIM
    base_t = pid64 * T
    # Partial-gradient slices for this V tile: [NV, BH, ...].
    part_k = ((pidv64 * NBH) + pid64) * T * K
    part_t = ((pidv64 * NBH) + pid64) * T

    lower = offs_n[None, :] < offs_n[:, None]
    lower_f = lower.to(tl.float32)
    sel_erase = (offs_n[None, :] == (2 * offs_q[:, None])).to(tl.float32)
    sel_write = (offs_n[None, :] == (2 * offs_q[:, None] + 1)).to(tl.float32)
    agg_mat = ((offs_n[None, :] // 2) == offs_q[:, None]).to(tl.float32)
    rev_mat = (offs_n[None, :] >= offs_q[:, None]).to(tl.float32)
    q_mask = offs_n[None, :] <= (2 * offs_q[:, None] + 1)

    # Adjoint of the state after the last chunk.
    Lam = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for rev in range(NC):
        chunk = NC - 1 - rev
        t_start = chunk * CHUNK

        # ---- recompute the forward chunk ----
        state_base = ((pid64 * NC + chunk) * K) * V_DIM
        S_start = tl.load(
            state_ptr + state_base + offs_k[:, None] * V_DIM + offs_v[None, :],
            mask=mask_k[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)

        t_rel_n = offs_n // 2
        t_n = t_start + t_rel_n
        mask_n = (offs_n < 2 * CHUNK) & (t_n < T)
        is_write = (offs_n % 2) == 1

        ptrs_nk = base_k + t_n[:, None].to(tl.int64) * K + offs_k[None, :]
        mask_nk = mask_n[:, None] & mask_k[None, :]

        k_exp = tl.load(k_ptr + ptrs_nk, mask=mask_nk, other=0.0).to(tl.float32)
        e_exp = tl.load(e_ptr + ptrs_nk, mask=mask_nk, other=0.0).to(tl.float32)
        vec = tl.where(is_write[:, None], k_exp, e_exp)

        gc_exp = tl.load(gc_ptr + ptrs_nk, mask=mask_nk, other=0.0).to(tl.float32)

        r = tl.sum(
            tl.where(offs_n[:, None] == CHUNK, gc_exp, 0.0),
            axis=0,
        )
        fw = tl.exp(gc_exp - r[None, :])
        fu = tl.exp(r[None, :] - gc_exp)
        w_scaled = vec * fw
        u_scaled = vec * fu

        beta_t = tl.load(
            beta_ptr + base_t + t_n.to(tl.int64), mask=mask_n, other=0.0
        ).to(tl.float32)
        gamma_t = tl.load(
            gamma_ptr + base_t + t_n.to(tl.int64), mask=mask_n, other=0.0
        ).to(tl.float32)
        beta_step = tl.where(is_write, beta_t, gamma_t * (1.0 - beta_t))
        beta_step = tl.where(mask_n, beta_step, 0.0)

        v_exp = tl.load(
            v_ptr + base_v + t_n[:, None].to(tl.int64) * V_DIM + offs_v[None, :],
            mask=mask_n[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        v_star = tl.where(is_write[:, None], v_exp, 0.0)

        exp_r = tl.exp(r)
        S_r = S_start * exp_r[:, None]

        pred0 = tl.dot(w_scaled, S_r, input_precision="ieee")
        gram = tl.dot(w_scaled, tl.trans(u_scaled), input_precision="ieee")
        l_mat = tl.where(lower, beta_step[:, None] * gram, 0.0)

        inv_ser = tl.where(
            offs_n[:, None] == offs_n[None, :], 1.0, 0.0
        ).to(tl.float32)
        pow_x = -l_mat
        for _ in range(4):
            new_ser = inv_ser + tl.dot(pow_x, inv_ser, input_precision="ieee")
            new_pow = tl.dot(pow_x, pow_x, input_precision="ieee")
            inv_ser = new_ser
            pow_x = new_pow

        rhs = beta_step[:, None] * (v_star - pred0)
        v_hat = tl.dot(inv_ser, rhs, input_precision="ieee")

        t_c = t_start + offs_q
        mask_c = (offs_q < CHUNK) & (t_c < T)
        ptrs_ck = base_k + t_c[:, None].to(tl.int64) * K + offs_k[None, :]
        mask_ck = mask_c[:, None] & mask_k[None, :]

        q_c = tl.load(q_ptr + ptrs_ck, mask=mask_ck, other=0.0).to(tl.float32)
        gc_c = tl.load(gc_ptr + ptrs_ck, mask=mask_ck, other=0.0).to(tl.float32)
        fq = tl.exp(gc_c - r[None, :])
        q_scaled = q_c * fq

        q_gram = tl.dot(q_scaled, tl.trans(u_scaled), input_precision="ieee")
        q_gram = tl.where(q_mask, q_gram, 0.0)

        state_delta = tl.dot(tl.trans(u_scaled), v_hat, input_precision="ieee")
        l_last = tl.sum(
            tl.where(offs_n[:, None] == (2 * CHUNK - 1), gc_exp, 0.0),
            axis=0,
        )
        exp_ll = tl.exp(l_last)
        efac2 = tl.exp(l_last - r)
        e1 = S_start * exp_ll[:, None]
        e2 = state_delta * efac2[:, None]

        # ---- adjoints ----
        Le = Lam
        go_c = tl.load(
            go_ptr + base_v + t_c[:, None].to(tl.int64) * V_DIM + offs_v[None, :],
            mask=mask_c[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        d_out = scale * go_c

        # out branch
        d_qsc = tl.dot(d_out, tl.trans(S_r), input_precision="ieee")
        dSr = tl.dot(tl.trans(q_scaled), d_out, input_precision="ieee")
        d_qgram_m = tl.dot(d_out, tl.trans(v_hat), input_precision="ieee")
        d_vhat = tl.dot(tl.trans(q_gram), d_out, input_precision="ieee")
        d_qgram = tl.where(q_mask, d_qgram_m, 0.0)
        d_qsc = d_qsc + tl.dot(d_qgram, u_scaled, input_precision="ieee")
        dU = tl.dot(tl.trans(d_qgram), q_scaled, input_precision="ieee")

        # S_end branch
        d_state_delta = Le * efac2[:, None]
        d_vhat = d_vhat + tl.dot(u_scaled, d_state_delta, input_precision="ieee")
        dU = dU + tl.dot(v_hat, tl.trans(d_state_delta), input_precision="ieee")

        # vhat = inv @ rhs ; rhs = beta * (v* - pred)
        d_inv = tl.dot(d_vhat, tl.trans(rhs), input_precision="ieee")
        d_rhs = tl.dot(tl.trans(inv_ser), d_vhat, input_precision="ieee")
        d_s = d_rhs * beta_step[:, None]
        d_pred0 = -d_s

        # inv = (I + L)^{-1}  ->  dL = -inv^T d_inv inv^T
        tmp = tl.dot(tl.trans(inv_ser), d_inv, input_precision="ieee")
        dL = -tl.dot(tmp, tl.trans(inv_ser), input_precision="ieee")
        d_gram = lower_f * beta_step[:, None] * dL
        # FIX: the previous version had an extra (x * 0.0) term here that
        # multiplied/divided a [BLOCK_N, BLOCK_V] tile by the bare 1-D
        # beta_step [BLOCK_N].  Triton aligns trailing dimensions, so
        # [BLOCK_N, BLOCK_V] against [BLOCK_N] is an illegal broadcast
        # and failed compilation for every shape.  The correct adjoint
        # needs only the two properly-shaped reductions below:
        #   rhs = beta_step * s         -> sum over V of d_rhs * s
        #   L   = beta_step * gram      -> sum over j of dL * tril(gram)
        d_beta_step = tl.sum(d_rhs * (v_star - pred0), axis=1) + tl.sum(
            dL * lower_f * gram, axis=1
        )

        # gram = w @ u^T ; pred = w @ Sr
        dW = tl.dot(d_gram, u_scaled, input_precision="ieee") + tl.dot(
            d_pred0, tl.trans(S_r), input_precision="ieee"
        )
        dU = dU + tl.dot(tl.trans(d_gram), w_scaled, input_precision="ieee")
        dSr = dSr + tl.dot(tl.trans(w_scaled), d_pred0, input_precision="ieee")

        # state adjoint for the previous chunk
        Lam = Le * exp_ll[:, None] + dSr * exp_r[:, None]

        # through the rescaled factors onto vec and the decays
        d_a = dW * fw + dU * fu
        dl_exp = dW * w_scaled - dU * u_scaled
        dl_q = d_qsc * q_scaled

        dr = -tl.sum(dW * w_scaled, axis=0) + tl.sum(dU * u_scaled, axis=0) - tl.sum(
            d_qsc * q_scaled, axis=0
        ) + tl.sum(dSr * S_r, axis=1) - tl.sum(Le * e2, axis=1)
        d_llast = tl.sum(Le * (e1 + e2), axis=1)
        dl_exp = dl_exp + tl.where(
            offs_n[:, None] == CHUNK, dr[None, :], 0.0
        ) + tl.where(offs_n[:, None] == (2 * CHUNK - 1), d_llast[None, :], 0.0)

        # ---- store this chunk's gradients ----
        mask_ck_store = mask_c[:, None] & mask_k[None, :]
        offs_pk = part_k + t_c[:, None].to(tl.int64) * K + offs_k[None, :]

        grad_q = d_qsc * fq
        tl.store(gq_ptr + offs_pk, grad_q, mask=mask_ck_store)
        tl.store(
            gk_ptr + offs_pk,
            tl.dot(sel_write, d_a, input_precision="ieee"),
            mask=mask_ck_store,
        )
        tl.store(
            ge_ptr + offs_pk,
            tl.dot(sel_erase, d_a, input_precision="ieee"),
            mask=mask_ck_store,
        )

        dgc_t = tl.dot(agg_mat, dl_exp, input_precision="ieee") + dl_q
        dg_chunk = tl.dot(rev_mat, dgc_t, input_precision="ieee")
        tl.store(gg_ptr + offs_pk, dg_chunk, mask=mask_ck_store)

        tl.store(
            gv_ptr + base_v + t_c[:, None].to(tl.int64) * V_DIM + offs_v[None, :],
            tl.dot(sel_write, d_s, input_precision="ieee"),
            mask=mask_c[:, None] & mask_v[None, :],
        )

        beta_c = tl.load(
            beta_ptr + base_t + t_c.to(tl.int64), mask=mask_c, other=0.0
        ).to(tl.float32)
        gamma_c = tl.load(
            gamma_ptr + base_t + t_c.to(tl.int64), mask=mask_c, other=0.0
        ).to(tl.float32)
        dbs_erase = tl.sum(
            tl.where(offs_n[None, :] == (2 * offs_q[:, None]),
                     d_beta_step[None, :], 0.0),
            axis=1,
        )
        dbs_write = tl.sum(
            tl.where(offs_n[None, :] == (2 * offs_q[:, None] + 1),
                     d_beta_step[None, :], 0.0),
            axis=1,
        )
        offs_pt = part_t + t_c.to(tl.int64)
        tl.store(gbeta_ptr + offs_pt, dbs_write - dbs_erase * gamma_c, mask=mask_c)
        tl.store(ggamma_ptr + offs_pt, dbs_erase * (1.0 - beta_c), mask=mask_c)


@triton.jit
def _reduce_kernel(
    in_ptr,
    out_ptr,
    N,
    NV,
    BLOCK_NV: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_x = pid * BLOCK_X + tl.arange(0, BLOCK_X)
    offs_n = tl.arange(0, BLOCK_NV)
    mask_x = offs_x < N
    ptrs = in_ptr + offs_n[:, None].to(tl.int64) * N + offs_x[None, :].to(tl.int64)
    vals = tl.load(
        ptrs,
        mask=(offs_n[:, None] < NV) & mask_x[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(out_ptr + offs_x.to(tl.int64), tl.sum(vals, axis=0), mask=mask_x)


def _tile_cfg(K, V):
    block_k = max(16, 1 << (K - 1).bit_length()) if K > 0 else 16
    block_v_full = max(16, 1 << (V - 1).bit_length()) if V > 0 else 16
    # K = 129 pads BLOCK_K to 256; halving the V tile (and using a single
    # pipeline stage below) keeps the tl.dot shared-memory footprint under
    # the L4 limit, while the small BLOCK_N above already shrinks it a lot.
    if block_k > 128:
        block_v = min(32, block_v_full)
        num_stages = 1
    else:
        block_v = min(64, block_v_full)
        num_stages = 2
    return block_k, block_v, num_stages


def _flatten(q, k, v, e, g, beta, gamma):
    B, H, T, K = q.shape
    V = v.shape[-1]
    BH = B * H
    return BH, (
        q.reshape(BH, T, K).contiguous(),
        k.reshape(BH, T, K).contiguous(),
        v.reshape(BH, T, V).contiguous(),
        e.reshape(BH, T, K).contiguous(),
        g.reshape(BH, T, K).contiguous(),
        beta.reshape(BH, T).contiguous(),
        gamma.reshape(BH, T).contiguous(),
    )


def _run_forward(flat, BH, T, K, V, scale, save_states):
    q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f = flat
    block_k, block_v, num_stages = _tile_cfg(K, V)
    num_v_tiles = (V + block_v - 1) // block_v
    num_chunks = (T + _CHUNK - 1) // _CHUNK

    gcum = torch.empty((BH, T, K), device=q_f.device, dtype=torch.float32)
    _chunk_decay_kernel[(BH, num_chunks)](
        g_f,
        gcum,
        T,
        K,
        CHUNK=_CHUNK,
        BLOCK_K=block_k,
        num_warps=4,
    )

    if save_states:
        states = torch.empty(
            (BH, num_chunks, K, V), device=q_f.device, dtype=torch.float32
        )
    else:
        states = torch.empty((1,), device=q_f.device, dtype=torch.float32)

    out = torch.empty((BH, T, V), device=q_f.device, dtype=torch.float32)
    _recurrent_kernel[(BH * num_v_tiles,)](
        q_f,
        k_f,
        v_f,
        e_f,
        gcum,
        beta_f,
        gamma_f,
        out,
        states,
        float(scale),
        T,
        K,
        V,
        num_chunks,
        num_v_tiles,
        CHUNK=_CHUNK,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        SAVE_STATES=save_states,
        num_warps=8,
        num_stages=num_stages,
    )
    return out, states, gcum


def _reduce_partial(partial, out_shape, device):
    NV = partial.shape[0]
    out = torch.empty(out_shape, device=device, dtype=torch.float32)
    n = out.numel()
    block_x = 1024
    block_nv = max(16, _next_pow2(NV))
    _reduce_kernel[((n + block_x - 1) // block_x,)](
        partial,
        out,
        n,
        NV,
        BLOCK_NV=block_nv,
        BLOCK_X=block_x,
        num_warps=4,
    )
    return out


class _LayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH, flat = _flatten(q, k, v, e, g, beta, gamma)
        out, states, gcum = _run_forward(flat, BH, T, K, V, scale, True)

        ctx.save_for_backward(*flat, gcum, states)
        ctx.shapes = (q.shape, k.shape, v.shape, e.shape,
                      g.shape, beta.shape, gamma.shape)
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype,
                      g.dtype, beta.dtype, gamma.dtype)
        ctx.scale = float(scale)
        ctx.dims = (BH, T, K, V)
        return out.reshape(B, H, T, V)

    @staticmethod
    def backward(ctx, grad_out):
        BH, T, K, V = ctx.dims
        (q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f, gcum, states) = \
            ctx.saved_tensors
        block_k, block_v, _ = _tile_cfg(K, V)
        num_v_tiles = (V + block_v - 1) // block_v
        num_chunks = (T + _CHUNK - 1) // _CHUNK
        device = q_f.device

        go_f = grad_out.reshape(BH, T, V).to(torch.float32).contiguous()

        gq_p = torch.zeros((num_v_tiles, BH, T, K), device=device, dtype=torch.float32)
        gk_p = torch.zeros((num_v_tiles, BH, T, K), device=device, dtype=torch.float32)
        ge_p = torch.zeros((num_v_tiles, BH, T, K), device=device, dtype=torch.float32)
        gg_p = torch.zeros((num_v_tiles, BH, T, K), device=device, dtype=torch.float32)
        gbeta_p = torch.zeros((num_v_tiles, BH, T), device=device, dtype=torch.float32)
        ggamma_p = torch.zeros((num_v_tiles, BH, T), device=device, dtype=torch.float32)
        gv = torch.empty((BH, T, V), device=device, dtype=torch.float32)

        _recurrent_backward_kernel[(BH * num_v_tiles,)](
            q_f,
            k_f,
            v_f,
            e_f,
            gcum,
            beta_f,
            gamma_f,
            go_f,
            states,
            gq_p,
            gk_p,
            gv,
            ge_p,
            gg_p,
            gbeta_p,
            ggamma_p,
            ctx.scale,
            T,
            K,
            V,
            num_chunks,
            num_v_tiles,
            BH,
            CHUNK=_CHUNK,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            num_warps=8,
            num_stages=1,
        )

        gq = _reduce_partial(gq_p, (BH, T, K), device)
        gk = _reduce_partial(gk_p, (BH, T, K), device)
        ge = _reduce_partial(ge_p, (BH, T, K), device)
        gg = _reduce_partial(gg_p, (BH, T, K), device)
        gbeta = _reduce_partial(gbeta_p, (BH, T), device)
        ggamma = _reduce_partial(ggamma_p, (BH, T), device)

        s = ctx.shapes
        d = ctx.dtypes
        return (
            gq.reshape(s[0]).to(d[0]),
            gk.reshape(s[1]).to(d[1]),
            gv.reshape(s[2]).to(d[2]),
            ge.reshape(s[3]).to(d[3]),
            gg.reshape(s[4]).to(d[4]),
            gbeta.reshape(s[5]).to(d[5]),
            ggamma.reshape(s[6]).to(d[6]),
            None,
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    if B == 0 or H == 0 or T == 0 or V == 0:
        return torch.empty(
            (B, H, T, V), device=q.device, dtype=torch.float32
        )

    if isinstance(scale, torch.Tensor):
        scale = float(scale.detach())

    needs_grad = any(
        isinstance(x, torch.Tensor) and x.requires_grad
        for x in (q, k, v, e, g, beta, gamma)
    )
    if needs_grad:
        return _LayerFn.apply(q, k, v, e, g, beta, gamma, float(scale))

    BH, flat = _flatten(q, k, v, e, g, beta, gamma)
    out, _, _ = _run_forward(flat, BH, T, K, V, scale, False)
    return out.reshape(B, H, T, V)
```

The compile failure was in the backward kernel's `beta_step` adjoint: a leftover `* 0.0` term multiplied and divided a `[BLOCK_N, BLOCK_V]` tile by the bare 1-D `beta_step` of length `BLOCK_N`, which Triton rejects because trailing-dimension broadcasting makes `BLOCK_V` collide with `BLOCK_N`. I deleted that dead term, leaving the two correctly shaped reductions (over V for the `rhs` path and over the Gram index for the `L` path), and switched the backward dots from TF32 to the IEEE precision already proven in the forward kernel. No other forward or backward math was changed.