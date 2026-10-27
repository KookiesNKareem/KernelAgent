```python
import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_affine_kernel(
    k_ptr,
    e_ptr,
    g_ptr,
    v_ptr,
    beta_ptr,
    gamma_ptr,
    p_ptr,
    f_ptr,
    num_chunks,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BW: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_col = tl.program_id(1)

    bh = pid // num_chunks
    chunk = pid % num_chunks

    rows = tl.arange(0, BK)
    cols = tl.arange(0, BW)
    col_idx = pid_col * BW + cols
    row_mask = rows < K

    # F columns start at zero; P columns start as identity columns.
    acc_f = tl.zeros((BK, BW), dtype=tl.float32)
    acc_p = tl.where(
        rows[:, None] == col_idx[None, :], 1.0, 0.0
    ).to(tl.float32)

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
        vt = tl.load(
            v_base + t * V + col_idx,
            mask=col_idx < V,
            other=0.0,
        ).to(tl.float32)
        beta = tl.load(beta_base + t)
        gamma = tl.load(gamma_base + t)

        decay = tl.exp(gt)[:, None]
        acc_f = acc_f * decay
        acc_p = acc_p * decay

        # Delta update along k: v is the target for F, zero for P.
        pred_f = tl.sum(acc_f * kt[:, None], axis=0)
        pred_p = tl.sum(acc_p * kt[:, None], axis=0)
        acc_f = acc_f + kt[:, None] * (beta * (vt - pred_f))[None, :]
        acc_p = acc_p + kt[:, None] * (-beta * pred_p)[None, :]

        # Erasure update along e, with zero target for both.
        pred_f = tl.sum(acc_f * et[:, None], axis=0)
        pred_p = tl.sum(acc_p * et[:, None], axis=0)
        acc_f = acc_f - et[:, None] * (gamma * pred_f)[None, :]
        acc_p = acc_p - et[:, None] * (gamma * pred_p)[None, :]

    p_offsets = (
        bh * num_chunks * K * K
        + chunk * K * K
        + rows[:, None] * K
        + col_idx[None, :]
    )
    tl.store(
        p_ptr + p_offsets,
        acc_p,
        mask=row_mask[:, None] & (col_idx[None, :] < K),
    )

    f_offsets = (
        bh * num_chunks * K * V
        + chunk * K * V
        + rows[:, None] * V
        + col_idx[None, :]
    )
    tl.store(
        f_ptr + f_offsets,
        acc_f,
        mask=row_mask[:, None] & (col_idx[None, :] < V),
    )


@triton.jit
def _chunk_state_kernel(
    p_ptr,
    f_ptr,
    s_cur_ptr,
    s_next_ptr,
    s_hist_ptr,
    chunk,
    num_chunks,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    # One 64x64 output tile of one (batch, head) state.  A separate launch
    # per chunk replaces the old single kernel whose 128x128 (or, when
    # K = 129, padded 256x256) tl.dot operands, double-buffered with
    # num_stages=2, needed 128 KiB (384 KiB) of shared memory -- above the
    # L4's 99 KiB limit.  Here each tl.dot operand is only 64x64 fp32
    # (16 KiB) and num_stages=1, so shared usage stays far below the limit
    # for any K, V.
    bh = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)

    rows = pid_r * BT + tl.arange(0, BT)
    cols = pid_c * BT + tl.arange(0, BT)
    kidx = tl.arange(0, BT)
    row_mask = rows < K
    col_mask = cols < V

    cur_offsets = bh * K * V + rows[:, None] * V + cols[None, :]
    hist_offsets = (
        bh * num_chunks * K * V
        + chunk * K * V
        + rows[:, None] * V
        + cols[None, :]
    )
    cur_tile = tl.load(
        s_cur_ptr + cur_offsets,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    # Record the entering state for this chunk (used by the output replay).
    tl.store(s_hist_ptr + hist_offsets, cur_tile,
             mask=row_mask[:, None] & col_mask[None, :])

    # next = P_chunk @ cur + F_chunk, contracting over K in 64-wide blocks.
    acc = tl.zeros((BT, BT), dtype=tl.float32)
    p_base = p_ptr + bh * num_chunks * K * K + chunk * K * K
    for j in range(0, tl.cdiv(K, BT)):
        kk = j * BT + kidx
        p_tile = tl.load(
            p_base + rows[:, None] * K + kk[None, :],
            mask=row_mask[:, None] & (kk[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        s_tile = tl.load(
            s_cur_ptr + bh * K * V + kk[:, None] * V + cols[None, :],
            mask=(kk[:, None] < K) & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = tl.dot(p_tile, s_tile, acc=acc, input_precision="tf32")

    f_offsets = (
        bh * num_chunks * K * V
        + chunk * K * V
        + rows[:, None] * V
        + cols[None, :]
    )
    force = tl.load(
        f_ptr + f_offsets,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    tl.store(
        s_next_ptr + cur_offsets,
        acc + force,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _chunk_output_kernel(
    q_ptr,
    k_ptr,
    e_ptr,
    g_ptr,
    v_ptr,
    beta_ptr,
    gamma_ptr,
    s_ptr,
    out_ptr,
    num_chunks,
    T,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BW: tl.constexpr,
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

    s_offsets = (
        bh * num_chunks * K * V
        + chunk * K * V
        + rows[:, None] * V
        + col_idx[None, :]
    )
    acc = tl.load(
        s_ptr + s_offsets,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

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


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0 or V == 0:
        return out
    if K == 0:
        out.zero_()
        return out

    BH = B * H
    BK: int = max(16, triton.next_power_of_2(K))
    CHUNK: int = 256 if T > 256 else T
    num_chunks = (T + CHUNK - 1) // CHUNK

    q2 = q.contiguous().reshape(BH, T, K)
    k2 = k.contiguous().reshape(BH, T, K)
    v2 = v.contiguous().reshape(BH, T, V)
    e2 = e.contiguous().reshape(BH, T, K)
    g2 = g.contiguous().reshape(BH, T, K)
    beta2 = beta.contiguous().reshape(BH, T)
    gamma2 = gamma.contiguous().reshape(BH, T)

    if num_chunks == 1:
        # Single chunk: entering state is the zero initial state.
        s_buf = torch.zeros((BH, 1, K, V), device=q.device, dtype=torch.float32)
    else:
        p_buf = torch.empty((BH, num_chunks, K, K), device=q.device, dtype=torch.float32)
        f_buf = torch.empty((BH, num_chunks, K, V), device=q.device, dtype=torch.float32)
        s_buf = torch.empty((BH, num_chunks, K, V), device=q.device, dtype=torch.float32)

        # Fused F/P computation (unchanged): shares all k/e/g loads.
        BW_A: int = 32
        grid_a = (BH * num_chunks, triton.cdiv(max(K, V), BW_A))
        _chunk_affine_kernel[grid_a](
            k2, e2, g2, v2, beta2, gamma2, p_buf, f_buf,
            num_chunks, T,
            K, V, CHUNK, BK, BW_A,
            num_warps=8,
            num_stages=2,
        )

        # Cross-chunk scan, one small tiled matmul launch per chunk:
        # s_next = P_chunk @ s_cur + F_chunk, ping-ponging the state.
        BT_S: int = 64
        grid_s = (BH, triton.cdiv(K, BT_S), triton.cdiv(V, BT_S))
        s_cur = torch.zeros((BH, K, V), device=q.device, dtype=torch.float32)
        s_next = torch.empty((BH, K, V), device=q.device, dtype=torch.float32)
        for chunk in range(num_chunks):
            _chunk_state_kernel[grid_s](
                p_buf, f_buf, s_cur, s_next, s_buf,
                chunk, num_chunks,
                K, V, BT_S,
                num_warps=4,
                num_stages=1,
            )
            s_cur, s_next = s_next, s_cur

    BW_O: int = 64
    grid_o = (BH * num_chunks, triton.cdiv(V, BW_O))
    _chunk_output_kernel[grid_o](
        q2, k2, e2, g2, v2, beta2, gamma2, s_buf, out,
        num_chunks, T, float(scale),
        K, V, CHUNK, BK, BW_O,
        num_warps=8,
        num_stages=2,
    )

    return out
```

Diagnosis: the only assumption the failing inputs violated was about shared memory, not about values or shapes per se — the scan kernel's monolithic 128-wide (256-wide when K=129 pads up) `tl.dot` with `num_stages=2` double-buffered its fp32 operands past the L4's 99 KiB shared-memory limit, which is why every case with K=V=128 failed identically while the smaller `k_ne_v` cases passed. The fix replaces that kernel with per-chunk 64×64 tiled matmul launches at `num_stages=1`, so shared usage is independent of K and V and stays around tens of KiB, while keeping the 256-step chunking, TF32 tensor-core scan math, fused affine kernel, and exact fp32 replay that delivered the speedup.