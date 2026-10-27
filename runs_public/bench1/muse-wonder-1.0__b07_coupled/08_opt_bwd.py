import torch
import triton
import triton.language as tl


# Slow path: 4 timesteps/chunk, exactly the previously validated geometry
# (8 real erase/write sub-steps padded to 16 rows).  Fast path: 8
# timesteps/chunk, so all 16 sub-step rows are real work, with TF32 dots.
# The fast path is only used when the measured chunk-local cumulative
# decay spread is small enough that its exponents cannot overflow.
_CHUNK_SLOW = 4
_CHUNK_FAST = 8
_BLOCK_N = 16
_FAST_MAX_SPREAD = 40.0
_FAST_MAX_ABS_GC = 75.0


def _next_pow2(x):
    n = 1
    while n < x:
        n *= 2
    return n


@triton.jit
def _mdot(a, b, USE_TF32: tl.constexpr):
    if USE_TF32:
        return tl.dot(a, b, input_precision="tf32")
    else:
        return tl.dot(a, b, input_precision="ieee")


@triton.jit
def _chunk_decay_kernel(
    g_ptr,
    gc_ptr,
    spread_ptr,
    T,
    K,
    NC_DEC,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAVE_SPREAD: tl.constexpr,
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

    if SAVE_SPREAD:
        spread = tl.max(gc, axis=0) - tl.min(gc, axis=0)
        sbase = (pid_bh.to(tl.int64) * NC_DEC + pid_c) * K
        tl.store(spread_ptr + sbase + offs_k, spread, mask=offs_k < K)


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
    USE_TF32: tl.constexpr,
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

        # Prediction from the chunk-initial state.
        pred0 = _mdot(w_scaled, S_r, USE_TF32)

        # Gram_ij = sum_k vec_i,k vec_j,k exp(l_i,k - l_j,k).
        gram = _mdot(w_scaled, tl.trans(u_scaled), USE_TF32)
        lower = offs_n[None, :] < offs_n[:, None]
        l_mat = tl.where(lower, beta_step[:, None] * gram, 0.0)

        # Invert (I + L) via the Neumann series by repeated doubling.
        inv_ser = tl.where(
            offs_n[:, None] == offs_n[None, :], 1.0, 0.0
        ).to(tl.float32)
        pow_x = -l_mat
        for _ in range(4):
            new_ser = inv_ser + _mdot(pow_x, inv_ser, USE_TF32)
            new_pow = _mdot(pow_x, pow_x, USE_TF32)
            inv_ser = new_ser
            pow_x = new_pow

        rhs = beta_step[:, None] * (v_star - pred0)
        v_hat = _mdot(inv_ser, rhs, USE_TF32)

        # Outputs, one row per timestep of the chunk.
        t_c = t_start + offs_q
        mask_c = (offs_q < CHUNK) & (t_c < T)
        ptrs_ck = base_k + t_c[:, None].to(tl.int64) * K + offs_k[None, :]
        mask_ck = mask_c[:, None] & mask_k[None, :]

        q_c = tl.load(q_ptr + ptrs_ck, mask=mask_ck, other=0.0).to(tl.float32)
        gc_c = tl.load(gc_ptr + ptrs_ck, mask=mask_ck, other=0.0).to(tl.float32)
        q_scaled = q_c * tl.exp(gc_c - r[None, :])

        out_val = _mdot(q_scaled, S_r, USE_TF32)
        q_gram = _mdot(q_scaled, tl.trans(u_scaled), USE_TF32)
        q_gram = tl.where(
            offs_n[None, :] <= (2 * offs_q[:, None] + 1), q_gram, 0.0
        )
        out_val = (out_val + _mdot(q_gram, v_hat, USE_TF32)) * scale

        tl.store(
            out_ptr
            + base_v
            + t_c[:, None].to(tl.int64) * V_DIM
            + offs_v[None, :],
            out_val,
            mask=mask_c[:, None] & mask_v[None, :],
        )

        # Convert the transformed end-of-chunk state back to actual state.
        l_last = tl.sum(
            tl.where(offs_n[:, None] == (2 * CHUNK - 1), gc_exp, 0.0),
            axis=0,
        )
        state_delta = _mdot(tl.trans(u_scaled), v_hat, USE_TF32)
        S = (
            S * tl.exp(l_last)[:, None]
            + state_delta * tl.exp(l_last - r)[:, None]
        )


# ---------------------------------------------------------------------------
# Backward kernel: exact reverse mode of the forward chunk computation
# above, sweeping chunks in reverse time.  For the triangular solve,
# dL = -inv^T dInv inv^T; decay adjoints are aggregated from the
# erase/write sub-steps and reverse-cumsummed within the chunk to get dg.
# Shared (non-V) gradients are per-V-tile partials, summed afterwards by
# _reduce_kernel; dv is written directly.
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
    USE_TF32: tl.constexpr,
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

        pred0 = _mdot(w_scaled, S_r, USE_TF32)
        gram = _mdot(w_scaled, tl.trans(u_scaled), USE_TF32)
        l_mat = tl.where(lower, beta_step[:, None] * gram, 0.0)

        inv_ser = tl.where(
            offs_n[:, None] == offs_n[None, :], 1.0, 0.0
        ).to(tl.float32)
        pow_x = -l_mat
        for _ in range(4):
            new_ser = inv_ser + _mdot(pow_x, inv_ser, USE_TF32)
            new_pow = _mdot(pow_x, pow_x, USE_TF32)
            inv_ser = new_ser
            pow_x = new_pow

        rhs = beta_step[:, None] * (v_star - pred0)
        v_hat = _mdot(inv_ser, rhs, USE_TF32)

        t_c = t_start + offs_q
        mask_c = (offs_q < CHUNK) & (t_c < T)
        ptrs_ck = base_k + t_c[:, None].to(tl.int64) * K + offs_k[None, :]
        mask_ck = mask_c[:, None] & mask_k[None, :]

        q_c = tl.load(q_ptr + ptrs_ck, mask=mask_ck, other=0.0).to(tl.float32)
        gc_c = tl.load(gc_ptr + ptrs_ck, mask=mask_ck, other=0.0).to(tl.float32)
        fq = tl.exp(gc_c - r[None, :])
        q_scaled = q_c * fq

        q_gram = _mdot(q_scaled, tl.trans(u_scaled), USE_TF32)
        q_gram = tl.where(q_mask, q_gram, 0.0)

        state_delta = _mdot(tl.trans(u_scaled), v_hat, USE_TF32)
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
        d_qsc = _mdot(d_out, tl.trans(S_r), USE_TF32)
        dSr = _mdot(tl.trans(q_scaled), d_out, USE_TF32)
        d_qgram_m = _mdot(d_out, tl.trans(v_hat), USE_TF32)
        d_vhat = _mdot(tl.trans(q_gram), d_out, USE_TF32)
        d_qgram = tl.where(q_mask, d_qgram_m, 0.0)
        d_qsc = d_qsc + _mdot(d_qgram, u_scaled, USE_TF32)
        dU = _mdot(tl.trans(d_qgram), q_scaled, USE_TF32)

        # S_end branch
        d_state_delta = Le * efac2[:, None]
        d_vhat = d_vhat + _mdot(u_scaled, d_state_delta, USE_TF32)
        dU = dU + _mdot(v_hat, tl.trans(d_state_delta), USE_TF32)

        # vhat = inv @ rhs ; rhs = beta * (v* - pred)
        d_inv = _mdot(d_vhat, tl.trans(rhs), USE_TF32)
        d_rhs = _mdot(tl.trans(inv_ser), d_vhat, USE_TF32)
        d_s = d_rhs * beta_step[:, None]
        d_pred0 = -d_s

        # inv = (I + L)^{-1}  ->  dL = -inv^T d_inv inv^T
        tmp = _mdot(tl.trans(inv_ser), d_inv, USE_TF32)
        dL = -_mdot(tmp, tl.trans(inv_ser), USE_TF32)
        d_gram = lower_f * beta_step[:, None] * dL
        d_beta_step = tl.sum(d_rhs * (v_star - pred0), axis=1) + tl.sum(
            dL * lower_f * gram, axis=1
        )

        # gram = w @ u^T ; pred = w @ Sr
        dW = _mdot(d_gram, u_scaled, USE_TF32) + _mdot(
            d_pred0, tl.trans(S_r), USE_TF32
        )
        dU = dU + _mdot(tl.trans(d_gram), w_scaled, USE_TF32)
        dSr = dSr + _mdot(tl.trans(w_scaled), d_pred0, USE_TF32)

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
            _mdot(sel_write, d_a, USE_TF32),
            mask=mask_ck_store,
        )
        tl.store(
            ge_ptr + offs_pk,
            _mdot(sel_erase, d_a, USE_TF32),
            mask=mask_ck_store,
        )

        dgc_t = _mdot(agg_mat, dl_exp, USE_TF32) + dl_q
        dg_chunk = _mdot(rev_mat, dgc_t, USE_TF32)
        tl.store(gg_ptr + offs_pk, dg_chunk, mask=mask_ck_store)

        tl.store(
            gv_ptr + base_v + t_c[:, None].to(tl.int64) * V_DIM + offs_v[None, :],
            _mdot(sel_write, d_s, USE_TF32),
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
    # pipeline stage) keeps the tl.dot shared-memory footprint under the
    # L4 limit.
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
    device = q_f.device

    gcum = torch.empty((BH, T, K), device=device, dtype=torch.float32)

    # Decide chunk size / precision from the actual decay magnitudes,
    # measured by the Triton decay kernel itself on the fast geometry.
    if block_k <= 128:
        chunk = _CHUNK_FAST
        use_tf32 = True
        num_chunks = (T + chunk - 1) // chunk
        spread_buf = torch.empty((BH, num_chunks, K), device=device,
                                 dtype=torch.float32)
        _chunk_decay_kernel[(BH, num_chunks)](
            g_f, gcum, spread_buf, T, K, num_chunks,
            CHUNK=chunk, BLOCK_K=block_k, SAVE_SPREAD=True, num_warps=4,
        )
        max_spread = float(spread_buf.max())
        max_abs_gc = float(gcum.abs().max())
        if max_spread > _FAST_MAX_SPREAD or max_abs_gc > _FAST_MAX_ABS_GC:
            chunk = _CHUNK_SLOW
            use_tf32 = False
    else:
        chunk = _CHUNK_SLOW
        use_tf32 = False

    num_chunks = (T + chunk - 1) // chunk
    if chunk != _CHUNK_FAST:
        dummy = torch.empty((1,), device=device, dtype=torch.float32)
        _chunk_decay_kernel[(BH, num_chunks)](
            g_f, gcum, dummy, T, K, num_chunks,
            CHUNK=chunk, BLOCK_K=block_k, SAVE_SPREAD=False, num_warps=4,
        )

    if save_states:
        states = torch.empty(
            (BH, num_chunks, K, V), device=device, dtype=torch.float32
        )
    else:
        states = torch.empty((1,), device=device, dtype=torch.float32)

    out = torch.empty((BH, T, V), device=device, dtype=torch.float32)
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
        CHUNK=chunk,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        SAVE_STATES=save_states,
        USE_TF32=use_tf32,
        num_warps=8,
        num_stages=num_stages,
    )
    return out, states, gcum, chunk, use_tf32


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
        out, states, gcum, chunk, use_tf32 = _run_forward(
            flat, BH, T, K, V, scale, True
        )

        ctx.save_for_backward(*flat, gcum, states)
        ctx.shapes = (q.shape, k.shape, v.shape, e.shape,
                      g.shape, beta.shape, gamma.shape)
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype,
                      g.dtype, beta.dtype, gamma.dtype)
        ctx.scale = float(scale)
        ctx.dims = (BH, T, K, V)
        ctx.chunk = chunk
        ctx.use_tf32 = use_tf32
        return out.reshape(B, H, T, V)

    @staticmethod
    def backward(ctx, grad_out):
        BH, T, K, V = ctx.dims
        chunk = ctx.chunk
        (q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f, gcum, states) = \
            ctx.saved_tensors
        block_k, block_v, _ = _tile_cfg(K, V)
        num_v_tiles = (V + block_v - 1) // block_v
        num_chunks = (T + chunk - 1) // chunk
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
            CHUNK=chunk,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            USE_TF32=ctx.use_tf32,
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
    out, _, _, _, _ = _run_forward(flat, BH, T, K, V, scale, False)
    return out.reshape(B, H, T, V)
