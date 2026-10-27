```python
import torch
import triton
import triton.language as tl


@triton.jit
def _gated_delta_rule_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    scale,
    H, T, K_DIM, V_DIM, NUM_CHUNKS,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    sgb, sgh, sgt,
    sbb, sbh, sbt,
    sob, soh, sot, sov,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    b_idx = pid_bh // H
    h_idx = pid_bh % H

    q_base = q_ptr + b_idx * sqb + h_idx * sqh
    k_base = k_ptr + b_idx * skb + h_idx * skh
    v_base = v_ptr + b_idx * svb + h_idx * svh
    g_base = g_ptr + b_idx * sgb + h_idx * sgh
    beta_base = beta_ptr + b_idx * sbb + h_idx * sbh
    o_base = o_ptr + b_idx * sob + h_idx * soh

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    offs_v_global = pid_v * BV + offs_v

    mask_k = offs_k < K_DIM
    mask_v = offs_v_global < V_DIM

    row = offs_c[:, None]
    col = offs_c[None, :]
    eye = (row == col).to(tl.float32)
    lt_mask = col < row
    le_mask = col <= row

    # State for this (batch, head, V-block): [K, BV], kept in registers.
    state = tl.zeros([BK, BV], dtype=tl.float32)

    for chunk_idx in range(NUM_CHUNKS):
        start = chunk_idx * CHUNK
        offs_t = start + offs_c
        mask_t = offs_t < T

        k_tile = tl.load(
            k_base + offs_t[:, None] * skt + offs_k[None, :] * skk,
            mask=mask_t[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        q_tile = tl.load(
            q_base + offs_t[:, None] * sqt + offs_k[None, :] * sqk,
            mask=mask_t[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        v_tile = tl.load(
            v_base + offs_t[:, None] * svt + offs_v_global[None, :] * svv,
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)

        g_vec = tl.load(
            g_base + offs_t * sgt,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)

        beta_vec = tl.load(
            beta_base + offs_t * sbt,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)

        # Prefix log-decay within this chunk: cum[i] = sum_{j<=i} g[j].
        cum = tl.cumsum(g_vec, axis=0)
        a_vec = tl.exp(cum)                         # A_i, in (0, 1] for g <= 0
        total_log_decay = tl.sum(g_vec, axis=0)
        a_last = tl.exp(total_log_decay)

        cum_diff = cum[:, None] - cum[None, :]

        # Strictly-lower system matrix for the intra-chunk delta updates:
        # L[i,j] = beta_i * exp(cum_i - cum_j) * (k_i . k_j),  j < i.
        kk = tl.dot(k_tile, tl.trans(k_tile), input_precision="ieee")
        decay_lt = tl.exp(tl.where(lt_mask, cum_diff, 0.0))
        l_mat = tl.where(lt_mask, kk * beta_vec[:, None] * decay_lt, 0.0)

        # (I + L)^{-1} for nilpotent strictly-lower L (CHUNK <= 32):
        # (I+L)^{-1} = (I-L)(I+L^2)(I+L^4)(I+L^8)(I+L^16).
        inv = eye - l_mat
        power = tl.dot(l_mat, l_mat, input_precision="ieee")       # L^2
        inv = tl.dot(inv, eye + power, input_precision="ieee")
        power = tl.dot(power, power, input_precision="ieee")       # L^4
        inv = tl.dot(inv, eye + power, input_precision="ieee")
        power = tl.dot(power, power, input_precision="ieee")       # L^8
        inv = tl.dot(inv, eye + power, input_precision="ieee")
        power = tl.dot(power, power, input_precision="ieee")       # L^16
        inv = tl.dot(inv, eye + power, input_precision="ieee")

        # Solve (I+L) W = beta * (V - diag(A) K S_old).
        ks = tl.dot(k_tile, state, input_precision="ieee")
        rhs = beta_vec[:, None] * (v_tile - a_vec[:, None] * ks)
        w_tile = tl.dot(inv, rhs, input_precision="ieee")

        # Outputs from S_old plus intra-chunk contributions.
        qk = tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee")
        decay_le = tl.exp(tl.where(le_mask, cum_diff, 0.0))
        m_mat = tl.where(le_mask, qk * decay_le, 0.0)
        qs = tl.dot(q_tile, state, input_precision="ieee")
        mw = tl.dot(m_mat, w_tile, input_precision="ieee")
        o_tile = scale * (a_vec[:, None] * qs + mw)

        tl.store(
            o_base + offs_t[:, None] * sot + offs_v_global[None, :] * sov,
            o_tile,
            mask=mask_t[:, None] & mask_v[None, :],
        )

        # S_new = A_last * S_old + K_suffix^T W,
        # K_suffix[j] = exp(total_log_decay - cum_j) * k_j.
        k_suffix = k_tile * tl.exp(total_log_decay - cum)[:, None]
        state = a_last * state + tl.dot(
            tl.trans(k_suffix), w_tile, input_precision="ieee"
        )


def _next_power_of_2(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # e and gamma are intentionally unused, exactly as in the reference.
    B, H, T, K_DIM = q.shape
    V_DIM = v.shape[-1]

    out = torch.empty((B, H, T, V_DIM), device=q.device, dtype=torch.float32)
    if out.numel() == 0:
        return out

    BK = max(16, _next_power_of_2(K_DIM))
    if K_DIM > 128:
        # Wide-K fallback (e.g. K=129 -> BK=256): shrink the chunk and
        # V-block so the staged tiles and the [BK, BV] state fit on sm_89.
        CHUNK = 16
        BV = 16
        num_warps = 8
    else:
        CHUNK = 32
        BV = 32 if V_DIM > 32 else max(16, _next_power_of_2(V_DIM))
        num_warps = 4
    num_chunks = (T + CHUNK - 1) // CHUNK

    grid = (B * H, (V_DIM + BV - 1) // BV)

    _gated_delta_rule_kernel[grid](
        q, k, v, g, beta, out,
        float(scale),
        H, T, K_DIM, V_DIM, num_chunks,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        beta.stride(0), beta.stride(1), beta.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        CHUNK=CHUNK,
        BK=BK,
        BV=BV,
        num_warps=num_warps,
        num_stages=1,
    )
    return out
```

The failing assumption was resource sizing, not the math: the old `CHUNK=64, BV=64, num_stages=2` configuration staged enough fp32 tile and triangular-solve data to require 123392 bytes of shared memory, above the L4's 101376-byte limit. I halved the chunk and V-block sizes, dropped to a single pipeline stage, and added an even smaller configuration for `K > 128`, while leaving the chunkwise gated-delta algorithm and its decay handling unchanged. The main residual risk is some register pressure in the wide-K fallback, but its tiles are now small enough that shared memory should no longer be the binding constraint.