```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused chunkwise gated-delta-rule kernel.
#
# For one chunk, with b_t = 2*beta_t, G_t = sum_{i<=t} g_i and
# P_t = exp(G_t), define the scaled intra-chunk deltas e_t by
#
#   e_t + b_t * sum_{j<t} (k_t.k_j) exp(G_t-G_j) e_j
#       = b_t * (v_t - P_t * (k_t^T S_start))
#
# All exponentials of this form are <= 1 for g <= 0, so this formulation
# avoids the numerically dangerous exp(-G_t) factors.  The triangular
# factor is applied without explicitly forming its inverse:
#
#   (I+N)^-1 = (I-N)(I+N^2)(I+N^4)(I+N^8)(I+N^16)(I+N^32),
#              N strictly lower triangular and nilpotent for chunk size 64.
#
# Outputs and the outgoing state are then obtained with dense tl.dot
# operations from e, the decayed QK/KK matrices and S_start.
# ---------------------------------------------------------------------------
@triton.jit
def _chunk_delta_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    out_ptr,
    T,
    scale,
    num_chunks,
    K_DIM,
    V_DIM,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    mask_k = offs_k < K_DIM
    mask_v = offs_v < V_DIM

    row = offs_c[:, None]
    col = offs_c[None, :]
    mask_strict = row > col
    mask_incl = row >= col

    bh = pid_bh.to(tl.int64)
    q_base = bh * T * K_DIM
    k_base = bh * T * K_DIM
    v_base = bh * T * V_DIM
    t_base = bh * T

    state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK
        t_idx = chunk_start + offs_c
        mask_t = t_idx < T
        mask_tk = mask_t[:, None] & mask_k[None, :]
        mask_tv = mask_t[:, None] & mask_v[None, :]

        g_vec = tl.load(
            g_ptr + t_base + t_idx, mask=mask_t, other=0.0
        ).to(tl.float32)
        beta_vec = tl.load(
            beta_ptr + t_base + t_idx, mask=mask_t, other=0.0
        ).to(tl.float32)
        beta2 = 2.0 * beta_vec

        k_tile = tl.load(
            k_ptr + k_base + t_idx[:, None] * K_DIM + offs_k[None, :],
            mask=mask_tk,
            other=0.0,
        ).to(tl.float32)
        q_tile = tl.load(
            q_ptr + q_base + t_idx[:, None] * K_DIM + offs_k[None, :],
            mask=mask_tk,
            other=0.0,
        ).to(tl.float32)
        v_tile = tl.load(
            v_ptr + v_base + t_idx[:, None] * V_DIM + offs_v[None, :],
            mask=mask_tv,
            other=0.0,
        ).to(tl.float32)

        # Cumulative log-decay inside this chunk.
        cum_g = tl.cumsum(g_vec, axis=0)
        total_g = tl.sum(g_vec)
        p_vec = tl.exp(cum_g)
        p_last = tl.exp(total_g)

        # Decay matrix exp(G_t-G_j).  The minimum on entries that will be
        # masked out prevents inf*0 NaNs for very negative g.
        diff = cum_g[:, None] - cum_g[None, :]
        diff = tl.where(mask_incl, diff, tl.minimum(diff, 0.0))
        decay = tl.exp(diff)

        kk = tl.dot(
            k_tile, tl.trans(k_tile), input_precision="tf32"
        )
        qk = tl.dot(
            q_tile, tl.trans(k_tile), input_precision="tf32"
        )
        k_state = tl.dot(k_tile, state, input_precision="tf32")
        q_state = tl.dot(q_tile, state, input_precision="tf32")

        n_mat = tl.where(
            mask_strict, beta2[:, None] * kk * decay, 0.0
        )
        qk_decay = tl.where(mask_incl, qk * decay, 0.0)

        rhs = beta2[:, None] * (
            v_tile - p_vec[:, None] * k_state
        )

        # e = (I+N)^-1 rhs, applied as
        # (I-N)(I+N^2)(I+N^4)(I+N^8)(I+N^16)(I+N^32) rhs.
        e_tile = rhs - tl.dot(n_mat, rhs, input_precision="tf32")
        p_mat = n_mat
        for _ in range(5):
            p_mat = tl.dot(p_mat, p_mat, input_precision="tf32")
            e_tile = e_tile + tl.dot(
                p_mat, e_tile, input_precision="tf32"
            )

        out_tile = scale * (
            p_vec[:, None] * q_state
            + tl.dot(qk_decay, e_tile, input_precision="tf32")
        )
        tl.store(
            out_ptr + v_base + t_idx[:, None] * V_DIM + offs_v[None, :],
            out_tile,
            mask=mask_tv,
        )

        # S_end = P_last*S_start
        #         + sum_j exp(G_last-G_j) k_j e_j^T
        decay_last = tl.exp(total_g - cum_g)
        k_decayed = k_tile * decay_last[:, None]
        state = state * p_last + tl.dot(
            tl.trans(k_decayed), e_tile, input_precision="tf32"
        )


# Direct recurrent fallback, used only for shapes unsuitable for the
# chunk kernel (very large K or very short sequences).
@triton.jit
def _recurrent_delta_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    out_ptr,
    T,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < V

    bh = pid_bh.to(tl.int64)
    q_ptrs = q_ptr + bh * T * K + offs_k
    k_ptrs = k_ptr + bh * T * K + offs_k
    v_ptrs = v_ptr + bh * T * V + offs_v
    out_ptrs = out_ptr + bh * T * V + offs_v
    g_ptrs = g_ptr + bh * T
    beta_ptrs = beta_ptr + bh * T

    state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for _ in range(T):
        decay = tl.exp(tl.load(g_ptrs))
        beta_val = tl.load(beta_ptrs)

        k_vec = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)

        state *= decay
        pred = tl.sum(state * k_vec[:, None], axis=0)
        delta = (2.0 * beta_val) * (v_vec - pred)
        state += k_vec[:, None] * delta[None, :]

        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        out_vec = tl.sum(state * (q_vec * scale)[:, None], axis=0)
        tl.store(out_ptrs, out_vec, mask=mask_v)

        q_ptrs += K
        k_ptrs += K
        v_ptrs += V
        out_ptrs += V
        g_ptrs += 1
        beta_ptrs += 1


def _next_power_of_2(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _ceil_div(a, b):
    return (a + b - 1) // b


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # e and gamma are intentionally unused: they do not occur in the
    # reference recurrence.
    B, H, T, K = q.shape
    V = v.shape[-1]

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0 or V == 0:
        return out

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = g.contiguous()
    beta_c = beta.contiguous()

    bh = B * H
    q_f = q_c.view(bh, T, K)
    k_f = k_c.view(bh, T, K)
    v_f = v_c.view(bh, T, V)
    g_f = g_c.view(bh, T)
    beta_f = beta_c.view(bh, T)
    out_f = out.view(bh, T, V)

    use_chunk = T >= 64 and K <= 128

    with torch.cuda.device(q.device):
        if use_chunk:
            block_k = max(16, _next_power_of_2(K))
            if V <= 32:
                block_v = max(16, _next_power_of_2(V))
            elif bh <= 48:
                block_v = 32
            else:
                block_v = 64

            num_v_blocks = _ceil_div(V, block_v)
            chunk = 64
            num_chunks = _ceil_div(T, chunk)
            num_warps = 8 if block_k >= 64 else 4

            _chunk_delta_kernel[(bh, num_v_blocks)](
                q_f,
                k_f,
                v_f,
                g_f,
                beta_f,
                out_f,
                T,
                float(scale),
                num_chunks,
                K,
                V,
                CHUNK=chunk,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                num_warps=num_warps,
                num_stages=2,
            )
        else:
            block_k = _next_power_of_2(K)
            if block_k <= 32:
                block_v_cap = 128
            elif block_k <= 64:
                block_v_cap = 64
            elif block_k <= 128:
                block_v_cap = 32
            elif block_k <= 256:
                block_v_cap = 16
            else:
                block_v_cap = 8

            block_v = min(block_v_cap, _next_power_of_2(V))
            while block_v > 8 and bh * _ceil_div(V, block_v) < 128:
                block_v //= 2

            num_v_blocks = _ceil_div(V, block_v)
            tile_elems = block_k * block_v
            if tile_elems >= 4096:
                num_warps = 8
            elif tile_elems >= 1024:
                num_warps = 4
            else:
                num_warps = 2

            _recurrent_delta_kernel[(bh, num_v_blocks)](
                q_f,
                k_f,
                v_f,
                g_f,
                beta_f,
                out_f,
                T,
                float(scale),
                K=K,
                V=V,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                num_warps=num_warps,
            )

    return out
```

This replaces the 2,048-step scalar recurrence with 64-step chunks whose Gram matrices, state interactions, output computation, and state propagation are dense TF32 tensor-core `tl.dot` operations. The intra-chunk delta-rule triangular system is applied through the exact nilpotent factorization `(I−N)(I+N²)(I+N⁴)(I+N⁸)(I+N¹⁶)(I+N³²)`, and all decay factors are expressed as `exp(G_t−G_j) ≤ 1` to avoid overflow-prone reciprocal decays. The main uncertainty is register pressure from the several 64-wide tiles, mitigated by splitting `V` into 32-wide blocks for the benchmark shape.