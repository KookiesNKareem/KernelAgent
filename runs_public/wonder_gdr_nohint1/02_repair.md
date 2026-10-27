```python
import torch
import triton
import triton.language as tl


@triton.jit
def _inv_kernel(
    k_ptr,
    g_ptr,
    beta_ptr,
    inv_ptr,
    T,
    K,
    NT,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NT
    chunk = pid % NT

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_t = chunk * CHUNK + offs_c
    t_mask = offs_t < T
    k_mask = offs_k < K

    k_ptrs = k_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :]
    k_vals = tl.load(
        k_ptrs, mask=t_mask[:, None] & k_mask[None, :], other=0.0
    ).to(tl.float32)

    g_ptrs = g_ptr + bh * T + offs_t
    g_vals = tl.load(g_ptrs, mask=t_mask, other=0.0).to(tl.float32)

    beta_ptrs = beta_ptr + bh * T + offs_t
    beta_vals = tl.load(beta_ptrs, mask=t_mask, other=0.0).to(tl.float32)

    cumg = tl.cumsum(g_vals, axis=0)

    # L[t, j] = beta[t] * exp(cumg[t] - cumg[j]) * <k[t], k[j]> for j < t.
    # Since g <= 0 in the specified regime, cumg[t] - cumg[j] <= 0 here.
    kk = tl.dot(k_vals, tl.trans(k_vals), input_precision="tf32")
    exp_diff = tl.exp(cumg[:, None] - cumg[None, :])
    lower = offs_c[:, None] > offs_c[None, :]
    mat_l = tl.where(lower, kk * exp_diff * beta_vals[:, None], 0.0)

    # Invert (I + L) in place by forward substitution on the identity.
    # A regular (non-static) loop keeps code size and shared memory small.
    inv_mat = (offs_c[:, None] == offs_c[None, :]).to(tl.float32)
    for i in range(CHUNK):
        row_i = tl.sum(tl.where(offs_c[:, None] == i, inv_mat, 0.0), axis=0)
        col_i = tl.sum(tl.where(offs_c[None, :] == i, mat_l, 0.0), axis=1)
        inv_mat = inv_mat - col_i[:, None] * row_i[None, :]

    inv_ptrs = (
        inv_ptr
        + (bh * NT + chunk) * CHUNK * CHUNK
        + offs_c[:, None] * CHUNK
        + offs_c[None, :]
    )
    tl.store(inv_ptrs, inv_mat)


@triton.jit
def _wy_kernel(
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    inv_ptr,
    w_ptr,
    u_ptr,
    T,
    K,
    V,
    NT,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_blk = tl.program_id(1)
    bh = pid // NT
    chunk = pid % NT

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    offs_t = chunk * CHUNK + offs_c
    t_mask = offs_t < T

    g_ptrs = g_ptr + bh * T + offs_t
    g_vals = tl.load(g_ptrs, mask=t_mask, other=0.0).to(tl.float32)
    beta_ptrs = beta_ptr + bh * T + offs_t
    beta_vals = tl.load(beta_ptrs, mask=t_mask, other=0.0).to(tl.float32)
    cumg = tl.cumsum(g_vals, axis=0)

    inv_ptrs = (
        inv_ptr
        + (bh * NT + chunk) * CHUNK * CHUNK
        + offs_c[:, None] * CHUNK
        + offs_c[None, :]
    )
    inv_mat = tl.load(inv_ptrs).to(tl.float32)

    if pid_blk == 0:
        # W = (I + L)^-1 @ (beta * exp(cumg) * k)
        k_mask = offs_k < K
        k_ptrs = k_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :]
        k_vals = tl.load(
            k_ptrs, mask=t_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)
        rhs_w = beta_vals[:, None] * tl.exp(cumg)[:, None] * k_vals
        w_vals = tl.dot(inv_mat, rhs_w, input_precision="tf32")
        w_ptrs = w_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :]
        tl.store(w_ptrs, w_vals, mask=t_mask[:, None] & k_mask[None, :])
    else:
        # Uhat = (I + L)^-1 @ (beta * v), one V block per program.
        pid_v = pid_blk - 1
        v_cols = pid_v * BV + offs_v
        v_mask = v_cols < V
        v_ptrs = v_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :]
        v_vals = tl.load(
            v_ptrs, mask=t_mask[:, None] & v_mask[None, :], other=0.0
        ).to(tl.float32)
        rhs_u = beta_vals[:, None] * v_vals
        u_vals = tl.dot(inv_mat, rhs_u, input_precision="tf32")
        u_ptrs = u_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :]
        tl.store(u_ptrs, u_vals, mask=t_mask[:, None] & v_mask[None, :])


@triton.jit
def _state_kernel(
    q_ptr,
    k_ptr,
    g_ptr,
    w_ptr,
    u_ptr,
    out_ptr,
    T,
    K,
    V,
    NT,
    scale,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    v_cols = pid_v * BV + offs_v
    v_mask = v_cols < V
    k_mask = offs_k < K

    # S block [BK, BV], carried across chunks in registers.
    state = tl.zeros([BK, BV], dtype=tl.float32)

    for chunk in range(NT):
        offs_t = chunk * CHUNK + offs_c
        t_mask = offs_t < T

        g_ptrs = g_ptr + bh * T + offs_t
        g_vals = tl.load(g_ptrs, mask=t_mask, other=0.0).to(tl.float32)
        cumg = tl.cumsum(g_vals, axis=0)
        exp_cum = tl.exp(cumg)

        q_ptrs = q_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :]
        q_vals = tl.load(
            q_ptrs, mask=t_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)

        k_ptrs = k_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :]
        k_vals = tl.load(
            k_ptrs, mask=t_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)

        w_ptrs = w_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :]
        w_vals = tl.load(
            w_ptrs, mask=t_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)

        u_ptrs = u_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :]
        u_hat = tl.load(
            u_ptrs, mask=t_mask[:, None] & v_mask[None, :], other=0.0
        ).to(tl.float32)

        # Effective intra-chunk updates given the incoming state.
        u_vals = u_hat - tl.dot(w_vals, state, input_precision="tf32")

        # Output from the incoming state.
        q_state = tl.dot(q_vals, state, input_precision="tf32")

        # Output from interactions inside this chunk (j <= t).
        qk = tl.dot(q_vals, tl.trans(k_vals), input_precision="tf32")
        exp_diff = tl.exp(cumg[:, None] - cumg[None, :])
        causal = offs_c[None, :] <= offs_c[:, None]
        mat_m = tl.where(causal, qk * exp_diff, 0.0)
        intra = tl.dot(mat_m, u_vals, input_precision="tf32")

        out_vals = scale * (exp_cum[:, None] * q_state + intra)
        out_ptrs = out_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :]
        tl.store(out_ptrs, out_vals, mask=t_mask[:, None] & v_mask[None, :])

        # Propagate state to the end of the chunk. Padded tail rows have
        # g = 0 and k = 0, so cumg at the last block row is still correct.
        last_cum = tl.sum(tl.where(offs_c == CHUNK - 1, cumg, 0.0))
        k_scaled = k_vals * tl.exp(last_cum - cumg)[:, None]
        state = state * tl.exp(last_cum) + tl.dot(
            tl.trans(k_scaled), u_vals, input_precision="tf32"
        )


def kernel_function(q, k, v, g, beta, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    if T == 0 or V == 0:
        return torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if K == 0:
        return torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)

    if isinstance(scale, torch.Tensor):
        scale_val = float(scale.detach().item())
    else:
        scale_val = float(scale)

    # Small tiles plus num_stages=1 are the fix for the shared-memory
    # blow-up: previously CHUNK=64, BK=128, BV=64 with default pipelining
    # needed 262656 bytes of shared memory (limit 101376).
    BK = 16
    while BK < K:
        BK *= 2
    if BK <= 128:
        CHUNK = 32
        BV = 32
    else:
        # e.g. K = 129 -> BK = 256: shrink the other tiles to compensate.
        CHUNK = 16
        BV = 16

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = g.contiguous()
    beta_c = beta.contiguous()

    BH = B * H
    q_f = q_c.reshape(BH, T, K)
    k_f = k_c.reshape(BH, T, K)
    v_f = v_c.reshape(BH, T, V)
    g_f = g_c.reshape(BH, T)
    beta_f = beta_c.reshape(BH, T)

    NT = (T + CHUNK - 1) // CHUNK
    num_v_blocks = (V + BV - 1) // BV

    inv = torch.empty((BH * NT, CHUNK, CHUNK), device=q.device, dtype=torch.float32)
    w = torch.empty((BH, T, K), device=q.device, dtype=torch.float32)
    u = torch.empty((BH, T, V), device=q.device, dtype=torch.float32)
    out = torch.empty((BH, T, V), device=q.device, dtype=torch.float32)

    _inv_kernel[(BH * NT,)](
        k_f, g_f, beta_f, inv,
        T, K, NT,
        CHUNK, BK,
        num_warps=4, num_stages=1,
    )

    _wy_kernel[(BH * NT, 1 + num_v_blocks)](
        k_f, v_f, g_f, beta_f, inv, w, u,
        T, K, V, NT,
        CHUNK, BK, BV,
        num_warps=4, num_stages=1,
    )

    _state_kernel[(BH, num_v_blocks)](
        q_f, k_f, g_f, w, u, out,
        T, K, V, NT,
        scale_val,
        CHUNK, BK, BV,
        num_warps=4, num_stages=1,
    )

    return out.reshape(B, H, T, V)
```

The failing assumption was resource sizing, not the math: the original tiles (chunk 64, full 128-wide K, 64-wide V) with Triton's default software pipelining needed 262 KB of shared memory against the L4's 101 KB limit, so every launch failed before any numerics ran. The fix keeps the same chunkwise WY/state-carrying algorithm but shrinks the chunk to 32 (16 when K pads to 256, e.g. K=129, with V tiles also halved), tiles V at 32, computes the small triangular inverse in its own kernel with a non-unrolled substitution loop, and forces `num_stages=1`/`num_warps=4` on every launch to stop pipelining from multiplying shared-memory buffers. I remain unsure whether the checker's tolerance is tight enough to prefer full fp32 `ieee` dots over the tf32 tensor-core dots used here, and whether register pressure (rather than shared memory) could spill for the K=129 fallback path.