import torch
import triton
import triton.language as tl


@triton.jit
def _wy_kernel(
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
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
    pid_v = tl.program_id(1)

    bh = pid // NT
    chunk = pid % NT

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)

    offs_t = chunk * CHUNK + offs_c
    t_mask = offs_t < T
    v_cols = pid_v * BV + offs_v
    v_mask = v_cols < V

    # Load k for this chunk: [CHUNK, BK]
    k_ptrs = k_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :]
    k_vals = tl.load(
        k_ptrs,
        mask=t_mask[:, None] & (offs_k[None, :] < K),
        other=0.0,
    ).to(tl.float32)

    # Load v block: [CHUNK, BV]
    v_ptrs = v_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :]
    v_vals = tl.load(
        v_ptrs,
        mask=t_mask[:, None] & v_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    g_ptrs = g_ptr + bh * T + offs_t
    g_vals = tl.load(g_ptrs, mask=t_mask, other=0.0).to(tl.float32)

    beta_ptrs = beta_ptr + bh * T + offs_t
    beta_vals = tl.load(beta_ptrs, mask=t_mask, other=0.0).to(tl.float32)

    cumg = tl.cumsum(g_vals, axis=0)

    # Strictly-lower intra-chunk interaction matrix.
    # L[t, j] = beta[t] * exp(cumg[t] - cumg[j]) * <k[t], k[j]>, j < t.
    # The exponent is always <= 0 for non-positive g, so this is stable.
    kk = tl.dot(k_vals, tl.trans(k_vals), input_precision="ieee")
    exp_diff = tl.exp(cumg[:, None] - cumg[None, :])
    lower = offs_c[:, None] > offs_c[None, :]
    mat_l = tl.where(lower, kk * exp_diff * beta_vals[:, None], 0.0)

    # Invert (I + L) by forward substitution, maintaining an updated RHS.
    eye = (offs_c[:, None] == offs_c[None, :]).to(tl.float32)
    bmat = eye
    inv_mat = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)
    for i in range(CHUNK):
        row_i = tl.sum(tl.where(offs_c[:, None] == i, bmat, 0.0), axis=0)
        inv_mat += tl.where(offs_c[:, None] == i, row_i[None, :], 0.0)
        col_i = tl.sum(tl.where(offs_c[None, :] == i, mat_l, 0.0), axis=1)
        bmat -= col_i[:, None] * row_i[None, :]

    # W = (I+L)^-1 (beta * exp(cumg) * k), used for the state-dependent part.
    rhs_w = beta_vals[:, None] * tl.exp(cumg)[:, None] * k_vals
    w_vals = tl.dot(inv_mat, rhs_w, input_precision="ieee")

    if pid_v == 0:
        w_ptrs = w_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :]
        tl.store(
            w_ptrs,
            w_vals,
            mask=t_mask[:, None] & (offs_k[None, :] < K),
        )

    # Uhat = (I+L)^-1 (beta * v), the state-independent part.
    rhs_u = beta_vals[:, None] * v_vals
    u_vals = tl.dot(inv_mat, rhs_u, input_precision="ieee")

    u_ptrs = u_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :]
    tl.store(
        u_ptrs,
        u_vals,
        mask=t_mask[:, None] & v_mask[None, :],
    )


@triton.jit
def _chunk_state_kernel(
    q_ptr,
    k_ptr,
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

    # State block S: [BK, BV], initially zero.
    state = tl.zeros([BK, BV], dtype=tl.float32)

    for chunk in range(NT):
        offs_t = chunk * CHUNK + offs_c
        t_mask = offs_t < T

        g_ptrs = q_ptr  # dummy base to keep pointer types obvious; replaced below
        # g is not stored separately in this kernel's argument list, so
        # cumulative decay is passed through w/u?  (No: see wrapper note.)
        # This placeholder is overwritten by the loads that follow.
        _ = g_ptrs

        # Load q, k: [CHUNK, BK]
        q_ptrs = q_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :]
        q_vals = tl.load(
            q_ptrs,
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        k_ptrs = k_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :]
        k_vals = tl.load(
            k_ptrs,
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Recompute cumg from the w/u encoding is not possible; the wrapper
        # therefore supplies g through the k_ptr-adjacent global layout is
        # intentionally avoided.  Instead, cumg is loaded in the wrapper's
        # packed layout documented there.
        # (The actual g load is performed just below using G_PTR.)
        break

    # The body above is never used: this kernel is replaced by
    # _chunk_state_kernel_impl below through the wrapper.
    _ = state


@triton.jit
def _chunk_state_kernel_impl(
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

        # Effective per-token update vectors for this chunk.
        u_vals = u_hat - tl.dot(w_vals, state, input_precision="ieee")

        # Output: state carried into the chunk.
        q_state = tl.dot(q_vals, state, input_precision="ieee")

        # Output: interactions inside the chunk, j <= t.
        qk = tl.dot(q_vals, tl.trans(k_vals), input_precision="ieee")
        exp_diff = tl.exp(cumg[:, None] - cumg[None, :])
        causal = offs_c[None, :] <= offs_c[:, None]
        mat_m = tl.where(causal, qk * exp_diff, 0.0)
        intra = tl.dot(mat_m, u_vals, input_precision="ieee")

        out_vals = scale * (exp_cum[:, None] * q_state + intra)

        out_ptrs = out_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :]
        tl.store(
            out_ptrs,
            out_vals,
            mask=t_mask[:, None] & v_mask[None, :],
        )

        # Propagate the state to the end of the chunk.  For a partial final
        # chunk the padded rows have g = 0 and k = 0, so cumg at the last
        # block position is still the correct cumulative decay.
        last_cum = tl.sum(tl.where(offs_c == CHUNK - 1, cumg, 0.0))
        k_scaled = k_vals * tl.exp(last_cum - cumg)[:, None]
        state = state * tl.exp(last_cum) + tl.dot(
            tl.trans(k_scaled), u_vals, input_precision="ieee"
        )


def kernel_function(q, k, v, g, beta, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    if T == 0:
        return torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    CHUNK = 64
    BV = 64
    BK = max(16, 1 << (K - 1).bit_length()) if K > 0 else 16

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

    w = torch.empty((BH, T, K), device=q.device, dtype=torch.float32)
    u = torch.empty((BH, T, V), device=q.device, dtype=torch.float32)
    out = torch.empty((BH, T, V), device=q.device, dtype=torch.float32)

    _wy_kernel[(BH * NT, num_v_blocks)](
        k_f,
        v_f,
        g_f,
        beta_f,
        w,
        u,
        T,
        K,
        V,
        NT,
        CHUNK,
        BK,
        BV,
        num_warps=8,
    )

    _chunk_state_kernel_impl[(BH, num_v_blocks)](
        q_f,
        k_f,
        g_f,
        w,
        u,
        out,
        T,
        K,
        V,
        NT,
        float(scale),
        CHUNK,
        BK,
        BV,
        num_warps=8,
    )

    return out.reshape(B, H, T, V)
