```python
import torch
import triton
import triton.language as tl


@triton.jit
def _wy_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    w_ptr,
    u_ptr,
    a_ptr,
    T,
    K,
    V,
    NT,
    CHUNK: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BVF: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NT
    chunk = pid % NT

    offs_c = tl.arange(0, CHUNK)
    offs_s = tl.arange(0, BS)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BVF)
    k_mask = offs_k < K
    v_mask = offs_v < V

    offs_t_all = chunk * CHUNK + offs_c
    t_mask_all = offs_t_all < T

    g_full = tl.load(
        g_ptr + bh * T + offs_t_all, mask=t_mask_all, other=0.0
    ).to(tl.float32)
    cumg_full = tl.cumsum(g_full, axis=0)

    # Blocked forward solve of (I + L) X = RHS for both RHS blocks
    # (W from beta*exp(cumg)*k, Uhat from beta*v), in BS-row segments.
    # Diagonal BS x BS blocks are inverted explicitly (tensor-core
    # friendly block size); cross-segment terms use dots.
    k_prev = tl.zeros([BS, BK], dtype=tl.bfloat16)
    xw_prev = tl.zeros([BS, BK], dtype=tl.float32)
    xu_prev = tl.zeros([BS, BVF], dtype=tl.float32)
    cumg_prev = tl.zeros([BS], dtype=tl.float32)

    for seg in range(CHUNK // BS):
        seg_idx = seg * BS + offs_s
        seg_t = chunk * CHUNK + seg_idx
        seg_mask = seg_t < T

        k_i = tl.load(
            k_ptr + bh * T * K + seg_t[:, None] * K + offs_k[None, :],
            mask=seg_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        v_i = tl.load(
            v_ptr + bh * T * V + seg_t[:, None] * V + offs_v[None, :],
            mask=seg_mask[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        beta_i = tl.load(
            beta_ptr + bh * T + seg_t, mask=seg_mask, other=0.0
        ).to(tl.float32)
        cumg_i = tl.sum(
            tl.where(
                offs_c[:, None] == seg_idx[None, :],
                cumg_full[:, None],
                0.0,
            ),
            axis=0,
        )

        rhs_w = beta_i[:, None] * tl.exp(cumg_i)[:, None] * k_i.to(tl.float32)
        rhs_u = beta_i[:, None] * v_i.to(tl.float32)

        if seg > 0:
            l_cross = (
                tl.dot(k_i, tl.trans(k_prev)).to(tl.float32)
                * tl.exp(cumg_i[:, None] - cumg_prev[None, :])
                * beta_i[:, None]
            ).to(tl.bfloat16)
            rhs_w = rhs_w - tl.dot(l_cross, xw_prev.to(tl.bfloat16)).to(
                tl.float32
            )
            rhs_u = rhs_u - tl.dot(l_cross, xu_prev.to(tl.bfloat16)).to(
                tl.float32
            )

        # Invert the diagonal block I + L_ii by forward substitution.
        gram = tl.dot(k_i, tl.trans(k_i)).to(tl.float32)
        diff = cumg_i[:, None] - cumg_i[None, :]
        lower = offs_s[:, None] > offs_s[None, :]
        exp_diff = tl.where(
            lower, tl.exp(tl.where(lower, diff, 0.0)), 0.0
        )
        l_ii = tl.where(lower, gram * exp_diff * beta_i[:, None], 0.0)

        inv_b = (offs_s[:, None] == offs_s[None, :]).to(tl.float32)
        for r in range(BS):
            row_r = tl.sum(
                tl.where(offs_s[:, None] == r, inv_b, 0.0), axis=0
            )
            col_r = tl.sum(
                tl.where(offs_s[None, :] == r, l_ii, 0.0), axis=1
            )
            inv_b = inv_b - col_r[:, None] * row_r[None, :]

        inv_bf = inv_b.to(tl.bfloat16)
        x_w = tl.dot(inv_bf, rhs_w.to(tl.bfloat16)).to(tl.float32)
        x_u = tl.dot(inv_bf, rhs_u.to(tl.bfloat16)).to(tl.float32)

        tl.store(
            w_ptr + bh * T * K + seg_t[:, None] * K + offs_k[None, :],
            x_w.to(tl.bfloat16),
            mask=seg_mask[:, None] & k_mask[None, :],
        )
        tl.store(
            u_ptr + bh * T * V + seg_t[:, None] * V + offs_v[None, :],
            x_u.to(tl.bfloat16),
            mask=seg_mask[:, None] & v_mask[None, :],
        )

        k_prev = k_i
        xw_prev = x_w
        xu_prev = x_u
        cumg_prev = cumg_i

    # Precompute the decay-weighted intra-chunk q-k matrix for the
    # (highly parallel) output kernel, so the sequential state kernel
    # never has to touch q.
    q_full = tl.load(
        q_ptr + bh * T * K + offs_t_all[:, None] * K + offs_k[None, :],
        mask=t_mask_all[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    k_full = tl.load(
        k_ptr + bh * T * K + offs_t_all[:, None] * K + offs_k[None, :],
        mask=t_mask_all[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    qk = tl.dot(q_full, tl.trans(k_full)).to(tl.float32)
    diff_all = cumg_full[:, None] - cumg_full[None, :]
    causal = offs_c[None, :] <= offs_c[:, None]
    exp_causal = tl.where(
        causal, tl.exp(tl.where(causal, diff_all, 0.0)), 0.0
    )
    a_mat = tl.where(causal, qk * exp_causal, 0.0)

    a_ptrs = (
        a_ptr
        + pid * CHUNK * CHUNK
        + offs_c[:, None] * CHUNK
        + offs_c[None, :]
    )
    tl.store(a_ptrs, a_mat.to(tl.bfloat16))


@triton.jit
def _state_kernel(
    k_ptr,
    g_ptr,
    w_ptr,
    u_ptr,
    h_ptr,
    T,
    K,
    V,
    NT,
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

        w_vals = tl.load(
            w_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        u_hat = tl.load(
            u_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
            mask=t_mask[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        state_bf = state.to(tl.bfloat16)

        # Effective update for this chunk given the incoming state.
        u_eff = u_hat - tl.dot(w_vals, state_bf).to(tl.float32)

        # Snapshot the incoming state for the parallel output kernel.
        h_ptrs = (
            h_ptr
            + ((bh * NT + chunk) * K + offs_k[:, None]) * V
            + v_cols[None, :]
        )
        tl.store(
            h_ptrs,
            state_bf,
            mask=k_mask[:, None] & v_mask[None, :],
        )

        # Overwrite Uhat with the effective update (disjoint V slices
        # per program, so no cross-program hazard).
        tl.store(
            u_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
            u_eff.to(tl.bfloat16),
            mask=t_mask[:, None] & v_mask[None, :],
        )

        # Propagate the state to the end of the chunk.  Padded tail
        # rows have g = 0 and k = 0, so cumg at the last row is exact.
        k_vals = tl.load(
            k_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        last_cum = tl.sum(tl.where(offs_c == CHUNK - 1, cumg, 0.0))
        k_scaled = (
            k_vals.to(tl.float32) * tl.exp(last_cum - cumg)[:, None]
        ).to(tl.bfloat16)
        state = state * tl.exp(last_cum) + tl.dot(
            tl.trans(k_scaled), u_eff.to(tl.bfloat16)
        ).to(tl.float32)


@triton.jit
def _out_kernel(
    q_ptr,
    g_ptr,
    a_ptr,
    u_ptr,
    h_ptr,
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
    pid = tl.program_id(0)
    pid_v = tl.program_id(1)
    bh = pid // NT
    chunk = pid % NT

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    v_cols = pid_v * BV + offs_v
    v_mask = v_cols < V
    k_mask = offs_k < K

    offs_t = chunk * CHUNK + offs_c
    t_mask = offs_t < T

    g_vals = tl.load(
        g_ptr + bh * T + offs_t, mask=t_mask, other=0.0
    ).to(tl.float32)
    exp_cum = tl.exp(tl.cumsum(g_vals, axis=0))

    q_vals = tl.load(
        q_ptr + bh * T * K + offs_t[:, None] * K + offs_k[None, :],
        mask=t_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    h_vals = tl.load(
        h_ptr + (pid * K + offs_k[:, None]) * V + v_cols[None, :],
        mask=k_mask[:, None] & v_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    a_vals = tl.load(
        a_ptr + pid * CHUNK * CHUNK + offs_c[:, None] * CHUNK + offs_c[None, :]
    ).to(tl.bfloat16)

    u_vals = tl.load(
        u_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
        mask=t_mask[:, None] & v_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    q_state = tl.dot(q_vals, h_vals).to(tl.float32)
    intra = tl.dot(a_vals, u_vals).to(tl.float32)

    out_vals = scale * (exp_cum[:, None] * q_state + intra)
    tl.store(
        out_ptr + bh * T * V + offs_t[:, None] * V + v_cols[None, :],
        out_vals,
        mask=t_mask[:, None] & v_mask[None, :],
    )


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
    BVF = _pow2_at_least(V)

    # Chunk 64 everywhere; smaller tiles only when K/V pad to 256, to
    # stay well under the L4 shared-memory limit with num_stages=1.
    if BK <= 128 and BVF <= 128:
        CHUNK = 64
        BS = 32
        BV = 64
        num_warps = 8
    else:
        CHUNK = 32
        BS = 16
        BV = 32
        num_warps = 4

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

    w = torch.empty((BH, T, K), device=q.device, dtype=torch.bfloat16)
    u = torch.empty((BH, T, V), device=q.device, dtype=torch.bfloat16)
    a_qk = torch.empty((BH * NT, CHUNK, CHUNK), device=q.device, dtype=torch.bfloat16)
    h_snap = torch.empty((BH * NT, K, V), device=q.device, dtype=torch.bfloat16)
    out = torch.empty((BH, T, V), device=q.device, dtype=torch.float32)

    _wy_kernel[(BH * NT,)](
        q_f, k_f, v_f, g_f, beta_f, w, u, a_qk,
        T, K, V, NT,
        CHUNK, BS, BK, BVF,
        num_warps=num_warps, num_stages=1,
    )

    _state_kernel[(BH, num_v_blocks)](
        k_f, g_f, w, u, h_snap,
        T, K, V, NT,
        CHUNK, BK, BV,
        num_warps=num_warps, num_stages=1,
    )

    _out_kernel[(BH * NT, num_v_blocks)](
        q_f, g_f, a_qk, u, h_snap, out,
        T, K, V, NT,
        scale_val,
        CHUNK, BK, BV,
        num_warps=num_warps, num_stages=1,
    )

    return out.reshape(B, H, T, V)
```

The restructuring attacks the old bottleneck directly: the sequential state kernel (previously 0.98 ms doing state propagation plus all output math) now only performs the two state dots per chunk, while the WY kernel additionally precomputes the decay-weighted intra-chunk qk matrix and a fully parallel output kernel (one program per chunk instead of one per sequence) evaluates outputs from bf16 state snapshots. The WY work is also restructured into a blocked 32-row forward solve that shares the Gram/inverse computation between the W and U right-hand sides and replaces the old separate inverse kernel, and all intermediates move in bf16 under the 2e-2 tolerance. Remaining risks I could not verify without running: register pressure in the WY kernel from holding both solved blocks, and whether the snapshot traffic (~33 MB each way at the benchmark shape) is fully repaid by the extra output parallelism on the L4.