import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernels (chunkwise affine form: S_out = P @ S_in + F per chunk)
# ---------------------------------------------------------------------------

@triton.jit
def _chunk_affine_kernel(
    k_ptr, e_ptr, g_ptr, v_ptr, beta_ptr, gamma_ptr,
    p_ptr, f_ptr,
    num_chunks, T,
    K: tl.constexpr, V: tl.constexpr, CHUNK: tl.constexpr,
    BK: tl.constexpr, BW: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_col = tl.program_id(1)
    bh = pid // num_chunks
    chunk = pid % num_chunks

    rows = tl.arange(0, BK)
    cols = tl.arange(0, BW)
    col_idx = pid_col * BW + cols
    row_mask = rows < K

    acc_f = tl.zeros((BK, BW), dtype=tl.float32)
    acc_p = tl.where(rows[:, None] == col_idx[None, :], 1.0, 0.0).to(tl.float32)

    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    g_base = g_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    beta_base = beta_ptr + bh * T
    gamma_base = gamma_ptr + bh * T

    t0 = chunk * CHUNK
    chunk_len = tl.minimum(CHUNK, T - t0)

    for i in range(chunk_len):
        t = t0 + i
        kt = tl.load(k_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        et = tl.load(e_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        gt = tl.load(g_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        vt = tl.load(v_base + t * V + col_idx, mask=col_idx < V, other=0.0).to(tl.float32)
        beta = tl.load(beta_base + t)
        gamma = tl.load(gamma_base + t)

        decay = tl.exp(gt)[:, None]
        acc_f = acc_f * decay
        acc_p = acc_p * decay

        pred_f = tl.sum(acc_f * kt[:, None], axis=0)
        pred_p = tl.sum(acc_p * kt[:, None], axis=0)
        acc_f = acc_f + kt[:, None] * (beta * (vt - pred_f))[None, :]
        acc_p = acc_p + kt[:, None] * (-beta * pred_p)[None, :]

        pred_f = tl.sum(acc_f * et[:, None], axis=0)
        pred_p = tl.sum(acc_p * et[:, None], axis=0)
        acc_f = acc_f - et[:, None] * (gamma * pred_f)[None, :]
        acc_p = acc_p - et[:, None] * (gamma * pred_p)[None, :]

    p_offsets = (bh * num_chunks * K * K + chunk * K * K
                 + rows[:, None] * K + col_idx[None, :])
    tl.store(p_ptr + p_offsets, acc_p,
             mask=row_mask[:, None] & (col_idx[None, :] < K))

    f_offsets = (bh * num_chunks * K * V + chunk * K * V
                 + rows[:, None] * V + col_idx[None, :])
    tl.store(f_ptr + f_offsets, acc_f,
             mask=row_mask[:, None] & (col_idx[None, :] < V))


@triton.jit
def _chunk_state_kernel(
    p_ptr, f_ptr, s_cur_ptr, s_next_ptr, s_hist_ptr,
    chunk, num_chunks,
    K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
):
    # One 64x64 tile of S_next = P_chunk @ S_cur + F_chunk (small tiles keep
    # shared memory far below the L4 limit for any K, V).
    bh = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)

    rows = pid_r * BT + tl.arange(0, BT)
    cols = pid_c * BT + tl.arange(0, BT)
    kidx = tl.arange(0, BT)
    row_mask = rows < K
    col_mask = cols < V
    tile_mask = row_mask[:, None] & col_mask[None, :]

    cur_offsets = bh * K * V + rows[:, None] * V + cols[None, :]
    hist_offsets = (bh * num_chunks * K * V + chunk * K * V
                    + rows[:, None] * V + cols[None, :])
    cur_tile = tl.load(s_cur_ptr + cur_offsets, mask=tile_mask, other=0.0).to(tl.float32)
    tl.store(s_hist_ptr + hist_offsets, cur_tile, mask=tile_mask)

    acc = tl.zeros((BT, BT), dtype=tl.float32)
    p_base = p_ptr + bh * num_chunks * K * K + chunk * K * K
    for j in range(0, tl.cdiv(K, BT)):
        kk = j * BT + kidx
        p_tile = tl.load(p_base + rows[:, None] * K + kk[None, :],
                         mask=row_mask[:, None] & (kk[None, :] < K),
                         other=0.0).to(tl.float32)
        s_tile = tl.load(s_cur_ptr + bh * K * V + kk[:, None] * V + cols[None, :],
                         mask=(kk[:, None] < K) & col_mask[None, :],
                         other=0.0).to(tl.float32)
        acc = tl.dot(p_tile, s_tile, acc=acc, input_precision="tf32")

    f_offsets = (bh * num_chunks * K * V + chunk * K * V
                 + rows[:, None] * V + cols[None, :])
    force = tl.load(f_ptr + f_offsets, mask=tile_mask, other=0.0).to(tl.float32)
    tl.store(s_next_ptr + cur_offsets, acc + force, mask=tile_mask)


@triton.jit
def _chunk_output_kernel(
    q_ptr, k_ptr, e_ptr, g_ptr, v_ptr, beta_ptr, gamma_ptr,
    s_ptr, sub_ptr, out_ptr,
    num_chunks, n_subs, T, scale,
    K: tl.constexpr, V: tl.constexpr, CHUNK: tl.constexpr,
    BCHUNK: tl.constexpr, BK: tl.constexpr, BW: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_col = tl.program_id(1)
    bh = pid // num_chunks
    chunk = pid % num_chunks

    rows = tl.arange(0, BK)
    cols = tl.arange(0, BW)
    col_idx = pid_col * BW + cols
    row_mask = rows < K
    col_mask = col_idx < V
    tile_mask = row_mask[:, None] & col_mask[None, :]

    s_offsets = (bh * num_chunks * K * V + chunk * K * V
                 + rows[:, None] * V + col_idx[None, :])
    acc = tl.load(s_ptr + s_offsets, mask=tile_mask, other=0.0).to(tl.float32)

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    g_base = g_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    beta_base = beta_ptr + bh * T
    gamma_base = gamma_ptr + bh * T
    out_base = out_ptr + bh * T * V

    t0 = chunk * CHUNK
    chunk_len = tl.minimum(CHUNK, T - t0)

    for i in range(chunk_len):
        t = t0 + i

        # Save the entering state of every backward sub-chunk for free:
        # this replay already visits it, so the backward no longer needs
        # a separate sub-state recomputation pass.
        is_bnd = (t % BCHUNK) == 0
        sub_idx = t // BCHUNK
        sub_offsets = ((bh * n_subs + sub_idx) * K * V
                       + rows[:, None] * V + col_idx[None, :])
        tl.store(sub_ptr + sub_offsets, acc, mask=tile_mask & is_bnd)

        kt = tl.load(k_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        et = tl.load(e_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        gt = tl.load(g_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        vt = tl.load(v_base + t * V + col_idx, mask=col_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_base + t)
        gamma = tl.load(gamma_base + t)

        acc = acc * tl.exp(gt)[:, None]
        pred = tl.sum(acc * kt[:, None], axis=0)
        acc = acc + kt[:, None] * (beta * (vt - pred))[None, :]
        pred_e = tl.sum(acc * et[:, None], axis=0)
        acc = acc - et[:, None] * (gamma * pred_e)[None, :]

        qt = tl.load(q_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        out = tl.sum(acc * (qt * scale)[:, None], axis=0)
        tl.store(out_base + t * V + col_idx, out, mask=col_mask)


# ---------------------------------------------------------------------------
# Backward kernel: fused snapshot + adjoint replay over one sub-chunk.
#
# The old code ran a separate snapshot kernel and reverse kernel with
# 64-step fp32 snapshots (128 MiB per launch at the timed shape, far above
# L2, so both kernels streamed ~4 GiB each way through DRAM).  Here a
# single kernel (a) replays 32 steps storing S_{t-1} in bf16 -- a 33 MiB
# working set at the timed shape, small enough to stay in L2 between the
# forward and reverse phases of the same launch -- and (b) immediately
# runs the adjoint recurrence backwards over those snapshots, reusing the
# still-hot inputs.  Wider 64-column tiles also halve the per-tile
# gradient partials that the host must reduce.
# ---------------------------------------------------------------------------

@triton.jit
def _bwd_fused_kernel(
    q_ptr, k_ptr, e_ptr, g_ptr, v_ptr, beta_ptr, gamma_ptr,
    dout_ptr, s_sub_ptr, snap_ptr, lam_cur_ptr, lam_next_ptr,
    dq_p_ptr, dk_p_ptr, de_p_ptr, dg_p_ptr,
    dbeta_p_ptr, dgamma_p_ptr, dv_ptr,
    sub_idx, n_subs, T, scale,
    K: tl.constexpr, V: tl.constexpr, BCHUNK: tl.constexpr,
    BK: tl.constexpr, BW: tl.constexpr, NT: tl.constexpr,
):
    bh = tl.program_id(0)
    tile = tl.program_id(1)

    rows = tl.arange(0, BK)
    cols = tl.arange(0, BW)
    col_idx = tile * BW + cols
    row_mask = rows < K
    col_mask = col_idx < V
    tile_mask = row_mask[:, None] & col_mask[None, :]

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    g_base = g_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    beta_base = beta_ptr + bh * T
    gamma_base = gamma_ptr + bh * T
    dout_base = dout_ptr + bh * T * V

    t0 = sub_idx * BCHUNK
    chunk_len = tl.minimum(BCHUNK, T - t0)

    # ---- Phase 1: forward replay, snapshotting S_{t-1} (bf16) per step.
    s_offsets = ((bh * n_subs + sub_idx) * K * V
                 + rows[:, None] * V + col_idx[None, :])
    acc = tl.load(s_sub_ptr + s_offsets, mask=tile_mask, other=0.0).to(tl.float32)

    for i in range(chunk_len):
        t = t0 + i
        snap_offsets = (bh * BCHUNK * K * V + i * K * V
                        + rows[:, None] * V + col_idx[None, :])
        tl.store(snap_ptr + snap_offsets, acc.to(tl.bfloat16), mask=tile_mask)

        kt = tl.load(k_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        et = tl.load(e_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        gt = tl.load(g_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        vt = tl.load(v_base + t * V + col_idx, mask=col_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_base + t)
        gamma = tl.load(gamma_base + t)

        acc = acc * tl.exp(gt)[:, None]
        pred = tl.sum(acc * kt[:, None], axis=0)
        acc = acc + kt[:, None] * (beta * (vt - pred))[None, :]
        pred_e = tl.sum(acc * et[:, None], axis=0)
        acc = acc - et[:, None] * (gamma * pred_e)[None, :]

    # ---- Phase 2: reverse adjoint replay over the snapshots.
    lam = tl.load(lam_cur_ptr + bh * K * V + rows[:, None] * V + col_idx[None, :],
                  mask=tile_mask, other=0.0).to(tl.float32)

    part_k = (bh * NT + tile) * T * K   # base for [BH, NT, T, K] partials
    part_s = (bh * NT + tile) * T       # base for [BH, NT, T] partials

    for rev in range(chunk_len):
        li = chunk_len - 1 - rev
        t = t0 + li

        s_prev = tl.load(
            snap_ptr + bh * BCHUNK * K * V + li * K * V
            + rows[:, None] * V + col_idx[None, :],
            mask=tile_mask, other=0.0).to(tl.float32)

        kt = tl.load(k_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        et = tl.load(e_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        gt = tl.load(g_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        vt = tl.load(v_base + t * V + col_idx, mask=col_mask, other=0.0).to(tl.float32)
        qt = tl.load(q_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        do = tl.load(dout_base + t * V + col_idx, mask=col_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_base + t)
        gamma = tl.load(gamma_base + t)

        # Rebuild the forward intermediates for this step.
        decay = tl.exp(gt)
        a_mat = s_prev * decay[:, None]
        p_vec = tl.sum(a_mat * kt[:, None], axis=0)
        u_vec = vt - p_vec
        b_mat = a_mat + kt[:, None] * (beta * u_vec)[None, :]
        r_vec = tl.sum(b_mat * et[:, None], axis=0)
        s_t = b_mat - et[:, None] * (gamma * r_vec)[None, :]

        # Output contribution: o_t = scale * S_t^T q_t.
        dq_vec = scale * tl.sum(s_t * do[None, :], axis=1)
        tl.store(dq_p_ptr + part_k + t * K + rows, dq_vec, mask=row_mask)
        lam = lam + (scale * qt)[:, None] * do[None, :]

        # Reverse erasure: S_t = (I - gamma e e^T) B.
        s_vec = tl.sum(lam * et[:, None], axis=0)
        lam_b = lam - et[:, None] * (gamma * s_vec)[None, :]
        dgamma = -tl.sum(s_vec * r_vec)
        tl.store(dgamma_p_ptr + part_s + t, dgamma)
        de_vec = -gamma * (tl.sum(lam * r_vec[None, :], axis=1)
                           + tl.sum(b_mat * s_vec[None, :], axis=1))
        tl.store(de_p_ptr + part_k + t * K + rows, de_vec, mask=row_mask)

        # Reverse delta: B = (I - beta k k^T) A + beta k v^T.
        w_vec = tl.sum(lam_b * kt[:, None], axis=0)
        lam_a = lam_b - kt[:, None] * (beta * w_vec)[None, :]
        dbeta = tl.sum(w_vec * u_vec)
        tl.store(dbeta_p_ptr + part_s + t, dbeta)
        tl.store(dv_ptr + bh * T * V + t * V + col_idx, beta * w_vec, mask=col_mask)
        dk_vec = beta * (tl.sum(lam_b * u_vec[None, :], axis=1)
                         - tl.sum(a_mat * w_vec[None, :], axis=1))
        tl.store(dk_p_ptr + part_k + t * K + rows, dk_vec, mask=row_mask)

        # Reverse decay: A = diag(exp(g)) S_{t-1}.
        dg_vec = tl.sum(lam_a * a_mat, axis=1)
        tl.store(dg_p_ptr + part_k + t * K + rows, dg_vec, mask=row_mask)
        lam = lam_a * decay[:, None]

    tl.store(lam_next_ptr + bh * K * V + rows[:, None] * V + col_idx[None, :],
             lam, mask=tile_mask)


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------

_BWD_CHUNK = 32


def _forward_impl(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    BH = B * H
    BK = max(16, triton.next_power_of_2(K))
    CHUNK = 256 if T > 256 else T
    num_chunks = (T + CHUNK - 1) // CHUNK
    n_subs = (T + _BWD_CHUNK - 1) // _BWD_CHUNK

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    # Entering states of the backward sub-chunks, filled by the output
    # replay kernel at no extra pass.
    s_sub = torch.empty((BH, n_subs, K, V), device=q.device, dtype=torch.float32)

    q2 = q.contiguous().reshape(BH, T, K)
    k2 = k.contiguous().reshape(BH, T, K)
    v2 = v.contiguous().reshape(BH, T, V)
    e2 = e.contiguous().reshape(BH, T, K)
    g2 = g.contiguous().reshape(BH, T, K)
    beta2 = beta.contiguous().reshape(BH, T)
    gamma2 = gamma.contiguous().reshape(BH, T)

    if num_chunks == 1:
        s_buf = torch.zeros((BH, 1, K, V), device=q.device, dtype=torch.float32)
    else:
        p_buf = torch.empty((BH, num_chunks, K, K), device=q.device, dtype=torch.float32)
        f_buf = torch.empty((BH, num_chunks, K, V), device=q.device, dtype=torch.float32)
        s_buf = torch.empty((BH, num_chunks, K, V), device=q.device, dtype=torch.float32)

        BW_A = 32
        grid_a = (BH * num_chunks, triton.cdiv(max(K, V), BW_A))
        _chunk_affine_kernel[grid_a](
            k2, e2, g2, v2, beta2, gamma2, p_buf, f_buf,
            num_chunks, T, K, V, CHUNK, BK, BW_A,
            num_warps=8, num_stages=2,
        )

        BT_S = 64
        grid_s = (BH, triton.cdiv(K, BT_S), triton.cdiv(V, BT_S))
        s_cur = torch.zeros((BH, K, V), device=q.device, dtype=torch.float32)
        s_next = torch.empty((BH, K, V), device=q.device, dtype=torch.float32)
        for chunk in range(num_chunks):
            _chunk_state_kernel[grid_s](
                p_buf, f_buf, s_cur, s_next, s_buf,
                chunk, num_chunks, K, V, BT_S,
                num_warps=4, num_stages=1,
            )
            s_cur, s_next = s_next, s_cur

    BW_O = 64
    grid_o = (BH * num_chunks, triton.cdiv(V, BW_O))
    _chunk_output_kernel[grid_o](
        q2, k2, e2, g2, v2, beta2, gamma2, s_buf, s_sub, out,
        num_chunks, n_subs, T, float(scale),
        K, V, CHUNK, _BWD_CHUNK, BK, BW_O,
        num_warps=8, num_stages=2,
    )
    return out, s_sub, n_subs


class _Layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        if B == 0 or H == 0 or T == 0 or V == 0 or K == 0:
            out = torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)
            dummy = torch.empty((0,), device=q.device, dtype=torch.float32)
            ctx.save_for_backward(q, k, v, e, g, beta, gamma, dummy)
            ctx.scale = float(scale)
            ctx.degenerate = True
            ctx.n_subs = 1
            return out

        out, s_sub, n_subs = _forward_impl(q, k, v, e, g, beta, gamma, scale)
        ctx.save_for_backward(q, k, v, e, g, beta, gamma, s_sub)
        ctx.scale = float(scale)
        ctx.degenerate = False
        ctx.n_subs = n_subs
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma, s_sub = ctx.saved_tensors
        if ctx.degenerate:
            return (torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v),
                    torch.zeros_like(e), torch.zeros_like(g), torch.zeros_like(beta),
                    torch.zeros_like(gamma), None)

        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        scale = ctx.scale
        dev = q.device
        n_subs = ctx.n_subs

        BK = max(16, triton.next_power_of_2(K))
        BCHUNK = _BWD_CHUNK
        BW_B = 64
        NT = triton.cdiv(V, BW_B)

        q2 = q.contiguous().reshape(BH, T, K)
        k2 = k.contiguous().reshape(BH, T, K)
        v2 = v.contiguous().reshape(BH, T, V)
        e2 = e.contiguous().reshape(BH, T, K)
        g2 = g.contiguous().reshape(BH, T, K)
        beta2 = beta.contiguous().reshape(BH, T)
        gamma2 = gamma.contiguous().reshape(BH, T)
        do2 = grad_out.contiguous().reshape(BH, T, V)

        # fp32 gradient accumulators.  q/k/e/g gradients are reduced over
        # V-tiles afterwards; v gradients are per-element, written directly.
        dq_p = torch.empty((BH, NT, T, K), device=dev, dtype=torch.float32)
        dk_p = torch.empty((BH, NT, T, K), device=dev, dtype=torch.float32)
        de_p = torch.empty((BH, NT, T, K), device=dev, dtype=torch.float32)
        dg_p = torch.empty((BH, NT, T, K), device=dev, dtype=torch.float32)
        dbeta_p = torch.empty((BH, NT, T), device=dev, dtype=torch.float32)
        dgamma_p = torch.empty((BH, NT, T), device=dev, dtype=torch.float32)
        dv_f = torch.empty((BH, T, V), device=dev, dtype=torch.float32)

        # Per-sub-chunk S_{t-1} snapshots in bf16: at the timed shape this
        # is a 33 MiB buffer whose live slice stays resident in L2 between
        # the fused kernel's forward and reverse phases.
        snap = torch.empty((BH, BCHUNK, K, V), device=dev, dtype=torch.bfloat16)
        lam_cur = torch.zeros((BH, K, V), device=dev, dtype=torch.float32)
        lam_next = torch.empty((BH, K, V), device=dev, dtype=torch.float32)

        grid_b = (BH, NT)
        for sub in range(n_subs - 1, -1, -1):
            _bwd_fused_kernel[grid_b](
                q2, k2, e2, g2, v2, beta2, gamma2,
                do2, s_sub, snap, lam_cur, lam_next,
                dq_p, dk_p, de_p, dg_p, dbeta_p, dgamma_p, dv_f,
                sub, n_subs, T, scale, K, V, BCHUNK, BK, BW_B, NT,
                num_warps=8, num_stages=1,
            )
            lam_cur, lam_next = lam_next, lam_cur

        grad_q = dq_p.sum(dim=1).reshape(B, H, T, K).to(q.dtype)
        grad_k = dk_p.sum(dim=1).reshape(B, H, T, K).to(k.dtype)
        grad_e = de_p.sum(dim=1).reshape(B, H, T, K).to(e.dtype)
        grad_g = dg_p.sum(dim=1).reshape(B, H, T, K).to(g.dtype)
        grad_v = dv_f.reshape(B, H, T, V).to(v.dtype)
        grad_beta = dbeta_p.sum(dim=1).reshape(B, H, T).to(beta.dtype)
        grad_gamma = dgamma_p.sum(dim=1).reshape(B, H, T).to(gamma.dtype)

        return grad_q, grad_k, grad_v, grad_e, grad_g, grad_beta, grad_gamma, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _Layer.apply(q, k, v, e, g, beta, gamma, scale)
