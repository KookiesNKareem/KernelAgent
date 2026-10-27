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


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    if B == 0 or H == 0 or T == 0 or V == 0:
        return torch.empty(
            (B, H, T, V), device=q.device, dtype=torch.float32
        )

    BH = B * H

    q_f = q.reshape(BH, T, K).contiguous()
    k_f = k.reshape(BH, T, K).contiguous()
    v_f = v.reshape(BH, T, V).contiguous()
    e_f = e.reshape(BH, T, K).contiguous()
    g_f = g.reshape(BH, T, K).contiguous()
    beta_f = beta.reshape(BH, T).contiguous()
    gamma_f = gamma.reshape(BH, T).contiguous()

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
    num_v_tiles = (V + block_v - 1) // block_v
    num_chunks = (T + _CHUNK - 1) // _CHUNK

    gcum = torch.empty((BH, T, K), device=q.device, dtype=torch.float32)
    _chunk_decay_kernel[(BH, num_chunks)](
        g_f,
        gcum,
        T,
        K,
        CHUNK=_CHUNK,
        BLOCK_K=block_k,
        num_warps=4,
    )

    out = torch.empty((BH, T, V), device=q.device, dtype=torch.float32)
    _recurrent_kernel[(BH * num_v_tiles,)](
        q_f,
        k_f,
        v_f,
        e_f,
        gcum,
        beta_f,
        gamma_f,
        out,
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
        num_warps=8,
        num_stages=num_stages,
    )
    return out.reshape(B, H, T, V)
```

Two assumptions broke: with 32-step chunks and per-step log-decay down to −25, the mid-chunk reference left `exp(±(gc−r))` arguments of several hundred, which overflow/underflow float32 and poison the Gram/state products with inf/NaN even though the true decayed quantities are tiny; and K=129 padded `BLOCK_K` to 256, doubling the `tl.dot` shared-memory footprint past the L4 limit. The fix shrinks chunks to 4 timesteps (8 real sub-steps padded to 16 rows for `tl.dot`), which bounds those exponent arguments to about [−85, 70] for the failing decay range, and it uses a 32-wide V tile with `num_stages=1` whenever `BLOCK_K` exceeds 128 to fit shared memory. The chunkwise erase/write expansion, triangular solve, and inter-chunk recurrence are otherwise unchanged.