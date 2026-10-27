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
    out_ptr,
    num_chunks,
    T,
    in_bh_k,
    in_bh_v,
    out_bh,
    out_chunk,
    out_row,
    K: tl.constexpr,
    V: tl.constexpr,
    COL_DIM: tl.constexpr,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BW: tl.constexpr,
    IS_P: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_col = tl.program_id(1)

    bh = pid // num_chunks
    chunk = pid % num_chunks

    rows = tl.arange(0, BK)
    cols = tl.arange(0, BW)
    col_idx = pid_col * BW + cols
    row_mask = rows < K

    acc = tl.zeros((BK, BW), dtype=tl.float32)
    if IS_P:
        # Columns of the chunk transition matrix P, initially the identity.
        acc = tl.where(rows[:, None] == col_idx[None, :], 1.0, 0.0)

    k_base = k_ptr + bh * in_bh_k
    e_base = e_ptr + bh * in_bh_k
    g_base = g_ptr + bh * in_bh_k
    v_base = v_ptr + bh * in_bh_v
    beta_base = beta_ptr + bh * T
    gamma_base = gamma_ptr + bh * T

    t0 = chunk * CHUNK
    chunk_len = tl.minimum(CHUNK, T - t0)

    for i in range(chunk_len):
        t = t0 + i

        kt = tl.load(k_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        et = tl.load(e_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        gt = tl.load(g_base + t * K + rows, mask=row_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_base + t)
        gamma = tl.load(gamma_base + t)

        # diag(exp(g_t)) S
        acc = acc * tl.exp(gt)[:, None]

        # Delta update with k_t.  For the transition-matrix columns the
        # target value is zero.
        pred = tl.sum(acc * kt[:, None], axis=0)
        if IS_P:
            delta = -beta * pred
        else:
            vt = tl.load(
                v_base + t * V + col_idx,
                mask=col_idx < V,
                other=0.0,
            ).to(tl.float32)
            delta = beta * (vt - pred)
        acc = acc + kt[:, None] * delta[None, :]

        # Erasure / second delta update with e_t and zero target.
        pred_e = tl.sum(acc * et[:, None], axis=0)
        acc = acc - et[:, None] * (gamma * pred_e)[None, :]

    out_offsets = (
        bh * out_bh
        + chunk * out_chunk
        + rows[:, None] * out_row
        + col_idx[None, :]
    )
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=row_mask[:, None] & (col_idx[None, :] < COL_DIM),
    )


@triton.jit
def _chunk_scan_kernel(
    p_ptr,
    f_ptr,
    s_ptr,
    num_chunks,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BW: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_col = tl.program_id(1)

    rows = tl.arange(0, BK)
    kcols = tl.arange(0, BK)
    cols = tl.arange(0, BW)
    col_idx = pid_col * BW + cols
    row_mask = rows < K

    acc = tl.zeros((BK, BW), dtype=tl.float32)

    p_bh = bh * num_chunks * K * K
    f_bh = bh * num_chunks * K * V
    s_bh = bh * num_chunks * K * V

    for chunk in range(num_chunks):
        # State entering this chunk.
        s_offsets = s_bh + chunk * K * V + rows[:, None] * V + col_idx[None, :]
        tl.store(
            s_ptr + s_offsets,
            acc,
            mask=row_mask[:, None] & (col_idx[None, :] < V),
        )

        p_offsets = (
            p_bh
            + chunk * K * K
            + rows[:, None] * K
            + kcols[None, :]
        )
        p_mat = tl.load(
            p_ptr + p_offsets,
            mask=row_mask[:, None] & (kcols[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        f_offsets = f_bh + chunk * K * V + rows[:, None] * V + col_idx[None, :]
        force = tl.load(
            f_ptr + f_offsets,
            mask=row_mask[:, None] & (col_idx[None, :] < V),
            other=0.0,
        ).to(tl.float32)

        # S_next = P_chunk @ S_current + F_chunk
        acc = tl.dot(p_mat, acc, input_precision="ieee") + force


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

    in_bh_k = T * K
    in_bh_v = T * V

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

    q_base = q_ptr + bh * in_bh_k
    k_base = k_ptr + bh * in_bh_k
    e_base = e_ptr + bh * in_bh_k
    g_base = g_ptr + bh * in_bh_k
    v_base = v_ptr + bh * in_bh_v
    beta_base = beta_ptr + bh * T
    gamma_base = gamma_ptr + bh * T
    out_base = out_ptr + bh * in_bh_v

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
    BW: int = 32
    BK: int = max(16, triton.next_power_of_2(K))
    CHUNK: int = 64 if T > 64 else T
    num_chunks = (T + CHUNK - 1) // CHUNK

    q2 = q.contiguous().reshape(BH, T, K)
    k2 = k.contiguous().reshape(BH, T, K)
    v2 = v.contiguous().reshape(BH, T, V)
    e2 = e.contiguous().reshape(BH, T, K)
    g2 = g.contiguous().reshape(BH, T, K)
    beta2 = beta.contiguous().reshape(BH, T)
    gamma2 = gamma.contiguous().reshape(BH, T)

    # For every chunk, the recurrence is affine in its entering state:
    #     S_out = P_chunk @ S_in + F_chunk.
    # _chunk_affine_kernel computes F_chunk (targets v / 0) and P_chunk
    # (targets 0, identity initial state) with the same sequential update.
    p_buf = torch.empty((BH, num_chunks, K, K), device=q.device, dtype=torch.float32)
    f_buf = torch.empty((BH, num_chunks, K, V), device=q.device, dtype=torch.float32)
    s_buf = torch.empty((BH, num_chunks, K, V), device=q.device, dtype=torch.float32)

    grid_b = (BH * num_chunks, triton.cdiv(V, BW))
    _chunk_affine_kernel[grid_b](
        k2, e2, g2, v2, beta2, gamma2, f_buf,
        num_chunks, T,
        T * K, T * V,
        num_chunks * K * V, K * V, V,
        K, V, V, CHUNK, BK, BW, False,
        num_warps=4,
        num_stages=2,
    )

    grid_p = (BH * num_chunks, triton.cdiv(K, BW))
    _chunk_affine_kernel[grid_p](
        k2, e2, g2, v2, beta2, gamma2, p_buf,
        num_chunks, T,
        T * K, T * V,
        num_chunks * K * K, K * K, K,
        K, V, K, CHUNK, BK, BW, True,
        num_warps=4,
        num_stages=2,
    )

    grid_scan = (BH, triton.cdiv(V, BW))
    _chunk_scan_kernel[grid_scan](
        p_buf, f_buf, s_buf,
        num_chunks,
        K, V, BK, BW,
        num_warps=8,
        num_stages=2,
    )

    grid_out = (BH * num_chunks, triton.cdiv(V, BW))
    _chunk_output_kernel[grid_out](
        q2, k2, e2, g2, v2, beta2, gamma2, s_buf, out,
        num_chunks, T, float(scale),
        K, V, CHUNK, BK, BW,
        num_warps=4,
        num_stages=2,
    )

    return out
```

My approach is chunkwise affine parallelization: each 64-step chunk is summarized as `S_out = P @ S_in + F` by running the exact decay/delta/erasure recurrence on the forcing columns and, separately, on identity columns; a tiny sequential scan then composes those affine maps across chunks, and a final highly parallel kernel replays each chunk from its true entering state to emit all outputs. This avoids cumulative-decay rescaling (and its overflow risks) because all intra-chunk work uses the original recurrence in fp32. I could not test compilation or timing, so the main uncertainties are Triton 3.5 codegen details (e.g. the `ieee` dot path) and whether the block/warp choices are optimal on an L4.