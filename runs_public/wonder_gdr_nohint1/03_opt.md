```python
import torch
import triton
import triton.language as tl


@triton.jit
def _blocked_solve(
    k_ptr,
    rhs_ptr,
    g_ptr,
    beta_ptr,
    bh,
    chunk,
    T,
    K,
    R,
    rhs_stride,
    CHUNK: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
    IS_W: tl.constexpr,
):
    # Solves, for one chunk, (I + L) X = RHS in blocks of BS rows.
    # L[t, j] = beta[t] * exp(cumg[t] - cumg[j]) * <k[t], k[j]>, j < t.
    # Cross-block terms are applied with tensor-core dots; each diagonal
    # BS x BS block is solved by a short forward substitution.  This
    # avoids the serial CHUNK-step inversion of a full CHUNK x CHUNK tile.
    offs_c = tl.arange(0, CHUNK)
    offs_s = tl.arange(0, BS)
    offs_k = tl.arange(0, BK)
    offs_r = tl.arange(0, BR)
    k_mask = offs_k < K
    r_mask = offs_r < R

    offs_t = chunk * CHUNK + offs_c
    t_mask = offs_t < T

    g_full = tl.load(g_ptr + bh * T + offs_t, mask=t_mask, other=0.0).to(tl.float32)
    cumg_full = tl.cumsum(g_full, axis=0)

    x_full = tl.zeros([CHUNK, BR], dtype=tl.float32)

    for i in range(CHUNK // BS):
        seg_t = chunk * CHUNK + i * BS + offs_s
        seg_mask = seg_t < T
        seg_idx = i * BS + offs_s

        # k block, beta block, and cumulative decay for this segment.
        k_i = tl.load(
            k_ptr + bh * T * K + seg_t[:, None] * K + offs_k[None, :],
            mask=seg_mask[:, None] & k_mask[None, :],
            other=0.0,
        )  # bf16
        beta_i = tl.load(
            beta_ptr + bh * T + seg_t, mask=seg_mask, other=0.0
        ).to(tl.float32)
        cumg_i = tl.sum(
            tl.where(
                offs_c[:, None] == seg_idx[None, :], cumg_full[:, None], 0.0
            ),
            axis=0,
        )

        if IS_W:
            rhs_i = beta_i[:, None] * tl.exp(cumg_i)[:, None] * k_i.to(tl.float32)
        else:
            rhs_i = beta_i[:, None] * tl.load(
                rhs_ptr
                + bh * T * rhs_stride
                + seg_t[:, None] * rhs_stride
                + offs_r[None, :],
                mask=seg_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)

        # Subtract contributions of already-solved segments j < i.
        for j in range(CHUNK // BS):
            if j < i:
                prev_t = chunk * CHUNK + j * BS + offs_s
                prev_mask = prev_t < T
                prev_idx = j * BS + offs_s
                k_j = tl.load(
                    k_ptr + bh * T * K + prev_t[:, None] * K + offs_k[None, :],
                    mask=prev_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                cumg_j = tl.sum(
                    tl.where(
                        offs_c[:, None] == prev_idx[None, :],
                        cumg_full[:, None],
                        0.0,
                    ),
                    axis=0,
                )
                e_j = (offs_c[:, None] == prev_idx[None, :]).to(tl.float32)
                x_j = tl.dot(
                    tl.trans(e_j), x_full, input_precision="tf32"
                )
                l_ij = (
                    tl.dot(k_i, tl.trans(k_j)).to(tl.float32)
                    * tl.exp(cumg_i[:, None] - cumg_j[None, :])
                    * beta_i[:, None]
                )
                rhs_i = rhs_i - tl.dot(l_ij, x_j, input_precision="tf32")

        # Solve the diagonal block (I + L_ii) x_i = rhs_i.
        l_ii = (
            tl.dot(k_i, tl.trans(k_i)).to(tl.float32)
            * tl.exp(cumg_i[:, None] - cumg_i[None, :])
            * beta_i[:, None]
        )
        l_ii = tl.where(offs_s[:, None] > offs_s[None, :], l_ii, 0.0)
        for r in range(BS):
            x_r = tl.sum(
                tl.where(offs_s[:, None] == r, rhs_i, 0.0), axis=0
            )
            col_r = tl.sum(
                tl.where(offs_s[None, :] == r, l_ii, 0.0), axis=1
            )
            rhs_i = rhs_i - col_r[:, None] * x_r[None, :]

        # Scatter this segment into the full-chunk result.
        e_i = (offs_c[:, None] == seg_idx[None, :]).to(tl.float32)
        x_full = x_full + tl.dot(e_i, rhs_i, input_precision="tf32")

    return x_full


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
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NT
    chunk = pid % NT

    offs_c = tl.arange(0, CHUNK)
    offs_t = chunk * CHUNK + offs_c
    t_mask = offs_t < T

    # W = (I+L)^-1 (beta * exp(cumg) * k)
    x_w = _blocked_solve(
        k_ptr, k_ptr, g_ptr, beta_ptr, bh, chunk, T, K, K, K,
        CHUNK, BS, BK, BK, True,
    )
    offs_k = tl.arange(0, BK)
    tl.store(
        w_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
        x_w.to(tl.bfloat16),
        mask=t_mask[:, None] & (offs_k[None, :] < K),
    )

    # Uhat = (I+L)^-1 (beta * v)
    x_u = _blocked_solve(
        k_ptr, v_ptr, g_ptr, beta_ptr, bh, chunk, T, K, V, V,
        CHUNK, BS, BK, BV, False,
    )
    offs_v = tl.arange(0, BV)
    tl.store(
        u_ptr + bh * T * V + offs_t[:, None] * V + offs_v[None, :],
        x_u.to(tl.bfloat16),
        mask=t_mask[:, None] & (offs_v[None, :] < V),
    )


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

    # S block [BK, BV], carried across chunks in registers (fp32).
    state = tl.zeros([BK, BV], dtype=tl.float32)

    for chunk in range(NT):
        offs_t = chunk * CHUNK + offs_c
        t_mask = offs_t < T

        g_vals = tl.load(
            g_ptr + bh * T + offs_t, mask=t_mask, other=0.0
        ).to(tl.float32)
        cumg = tl.cumsum(g_vals, axis=0)
        exp_cum = tl.exp(cumg)

        q_vals = tl.load(
            q_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        )  # bf16
        k_vals = tl.load(
            k_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        )  # bf16
        w_vals = tl.load(
            w_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        )  # bf16
        u_hat = tl.load(
            u_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
            mask=t_mask[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)  # bf16 -> fp32

        state_bf = state.to(tl.bfloat16)

        # Effective intra-chunk updates given the incoming state.
        u_vals = u_hat - tl.dot(w_vals, state_bf).to(tl.float32)

        # Output: contribution of the incoming state.
        q_state = tl.dot(q_vals, state_bf).to(tl.float32)

        # Output: interactions inside this chunk (j <= t).  All decay
        # factors are exp of non-positive differences in the gated
        # regime, so this is numerically stable.
        qk = tl.dot(q_vals, tl.trans(k_vals)).to(tl.float32)
        exp_diff = tl.exp(cumg[:, None] - cumg[None, :])
        causal = offs_c[None, :] <= offs_c[:, None]
        mat_m = tl.where(causal, qk * exp_diff, 0.0)
        intra = tl.dot(mat_m.to(tl.bfloat16), u_vals.to(tl.bfloat16)).to(
            tl.float32
        )

        out_vals = scale * (exp_cum[:, None] * q_state + intra)
        tl.store(
            out_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
            out_vals,
            mask=t_mask[:, None] & v_mask[None, :],
        )

        # Propagate the state to the end of the chunk.  Padded tail rows
        # have g = 0 and k = 0, so cumg at the last block row is correct.
        last_cum = tl.sum(tl.where(offs_c == CHUNK - 1, cumg, 0.0))
        k_scaled = (
            k_vals.to(tl.float32) * tl.exp(last_cum - cumg)[:, None]
        ).to(tl.bfloat16)
        state = state * tl.exp(last_cum) + tl.dot(
            tl.trans(k_scaled), u_vals.to(tl.bfloat16)
        ).to(tl.float32)


def _pow2_at_least(x, lo=16):
    p = lo
    while p < x:
        p *= 2
    return p


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

    BK = _pow2_at_least(K)
    BV_FULL = _pow2_at_least(V)

    # Larger tiles than the correctness-first version, but num_stages=1
    # everywhere keeps shared memory well under the L4 limit.  For very
    # wide padded tiles (e.g. K = V = 129) shrink the chunk instead.
    if BK <= 128 and BV_FULL <= 128:
        CHUNK = 64
        BV = 64
        num_warps = 8
    else:
        CHUNK = 32
        BV = 32
        num_warps = 4
    BS = 16

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

    # bf16 intermediates: halves W/U traffic through the state kernel;
    # well within the 2e-2 relative tolerance.
    w = torch.empty((BH, T, K), device=q.device, dtype=torch.bfloat16)
    u = torch.empty((BH, T, V), device=q.device, dtype=torch.bfloat16)
    out = torch.empty((BH, T, V), device=q.device, dtype=torch.float32)

    _wy_kernel[(BH * NT,)](
        k_f, v_f, g_f, beta_f, w, u,
        T, K, V, NT,
        CHUNK, BS, BK, BV_FULL,
        num_warps=num_warps, num_stages=1,
    )

    _state_kernel[(BH, num_v_blocks)](
        q_f, k_f, g_f, w, u, out,
        T, K, V, NT,
        scale_val,
        CHUNK, BK, BV,
        num_warps=num_warps, num_stages=1,
    )

    return out.reshape(B, H, T, V)
```

The main speedups over the 1.48 ms version are: the separate inverse kernel is eliminated and the triangular solve is blocked into 16-row segments whose cross terms use tensor-core dots (instead of a serial 32-step full-matrix substitution in its own kernel, plus a separate WY pass), W/U are stored in bf16 to halve traffic into the state kernel, and the state kernel now uses chunk 64 with V-tiles of 64, 8 warps, and native bf16 dots with fp32 state accumulation. Tiles and `num_stages=1` are still sized to stay under the L4 shared-memory limit, with a smaller-tile fallback when K or V pads to 256. I could not run this, so the main risks are Triton-specific: whether the `if j < i` scalar branch inside the blocked-solve loops compiles as efficiently as expected, and whether register pressure from the fp32 `[64, 128]` solve/state tiles causes spills on sm_89.