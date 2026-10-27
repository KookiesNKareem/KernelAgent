```python
import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_inv_kernel(
    q_ptr, k_ptr, g_ptr, beta_ptr, inv_ptr, qk_ptr,
    T, H, NT, D_K,
    sq_b, sq_h, sq_t, sq_d,
    sk_b, sk_h, sk_t, sk_d,
    sg_b, sg_h, sg_t, sg_d,
    sb_b, sb_h, sb_t,
    CHUNK: tl.constexpr, NUM_ITER: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_c = tl.arange(0, CHUNK)
    offs_t = pid_c * CHUNK + offs_c
    mask_t = offs_t < T

    q_base = q_ptr + pid_b * sq_b + pid_h * sq_h
    k_base = k_ptr + pid_b * sk_b + pid_h * sk_h
    g_base = g_ptr + pid_b * sg_b + pid_h * sg_h
    beta_base = beta_ptr + pid_b * sb_b + pid_h * sb_h

    beta = tl.load(
        beta_base + offs_t * sb_t,
        mask=mask_t,
        other=0.0,
    ).to(tl.float32)

    # A[t, s] and QK[t, s] for s <= t are accumulated one K channel at a
    # time.  Every exponential needed here is exp(G[t] - G[s]) with
    # s <= t: for a log-decay G is non-increasing, so the exponent is
    # <= 0, the factor lies in (0, 1], and strong decay simply
    # underflows it to 0 -- which is the correct contribution.  No
    # shared reference level and no positive exponentials are involved,
    # so nothing is ever clamped into a wrong ratio.
    a_mat = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)
    qk_mat = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)

    for kk in range(D_K):
        q_vec = tl.load(
            q_base + offs_t * sq_t + kk * sq_d,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)
        k_vec = tl.load(
            k_base + offs_t * sk_t + kk * sk_d,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)
        g_vec = tl.load(
            g_base + offs_t * sg_t + kk * sg_d,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)

        gcum = tl.cumsum(g_vec, axis=0)
        diff = gcum[:, None] - gcum[None, :]
        # The minimum only guards the discarded s > t triangle (and
        # hypothetically positive g); legitimate s <= t exponents are
        # <= 0 and pass through unchanged, however negative they are.
        exp_diff = tl.exp(tl.minimum(diff, 80.0))

        common = (k_vec[:, None] * k_vec[None, :]) * exp_diff
        a_mat += beta[:, None] * common
        qk_mat += q_vec[:, None] * common

    row = offs_c[:, None]
    col = offs_c[None, :]
    eye = tl.where(col == row, 1.0, 0.0)
    m_mat = eye + tl.where(col < row, a_mat, 0.0)
    qk_mat = tl.where(col <= row, qk_mat, 0.0)

    # Exact Newton inversion of I + A for nilpotent strictly-lower A.
    x_mat = eye
    for _ in range(NUM_ITER):
        mx = tl.dot(m_mat, x_mat, input_precision="ieee")
        x_mat = tl.dot(x_mat, 2.0 * eye - mx, input_precision="ieee")

    base_idx = (pid_bh * NT + pid_c) * CHUNK
    inv_ptrs = (
        inv_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :]
    )
    tl.store(inv_ptrs, x_mat)
    qk_ptrs = (
        qk_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :]
    )
    tl.store(qk_ptrs, qk_mat)


@triton.jit
def _chunk_state_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, inv_ptr, qk_ptr, state_ptr, out_ptr,
    T, H, NT, D_K, D_V, NK, scale,
    sq_b, sq_h, sq_t, sq_d,
    sk_b, sk_h, sk_t, sk_d,
    sv_b, sv_h, sv_t, sv_d,
    sg_b, sg_h, sg_t, sg_d,
    sb_b, sb_h, sb_t,
    so_b, so_h, so_t, so_d,
    CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_c = tl.arange(0, CHUNK)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < D_V

    q_base = q_ptr + pid_b * sq_b + pid_h * sq_h
    k_base = k_ptr + pid_b * sk_b + pid_h * sk_h
    v_base = v_ptr + pid_b * sv_b + pid_h * sv_h
    g_base = g_ptr + pid_b * sg_b + pid_h * sg_h
    beta_base = beta_ptr + pid_b * sb_b + pid_h * sb_h
    out_base = out_ptr + pid_b * so_b + pid_h * so_h
    state_base = state_ptr + pid_bh * D_K * D_V

    for chunk_idx in range(NT):
        t_start = chunk_idx * CHUNK
        offs_t = t_start + offs_c
        mask_t = offs_t < T

        vval = tl.load(
            v_base + offs_t[:, None] * sv_t + offs_v[None, :] * sv_d,
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        beta = tl.load(
            beta_base + offs_t * sb_t,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)

        base_idx = (pid_bh * NT + chunk_idx) * CHUNK
        inv_mat = tl.load(
            inv_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :]
        )
        qk_mat = tl.load(
            qk_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :]
        )

        chunk_len = tl.minimum(T - t_start, CHUNK)

        pred = tl.zeros([CHUNK, BLOCK_V], dtype=tl.float32)
        out_state = tl.zeros([CHUNK, BLOCK_V], dtype=tl.float32)

        # Incoming-state terms.  gcum <= 0 for a log-decay, so
        # exp(gcum) is the true decay of the state up to each step:
        # at most 1, and 0 once the state has been erased.
        for kb in range(NK):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D_K

            qval = tl.load(
                q_base + offs_t[:, None] * sq_t + offs_k[None, :] * sq_d,
                mask=mask_t[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            kval = tl.load(
                k_base + offs_t[:, None] * sk_t + offs_k[None, :] * sk_d,
                mask=mask_t[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            gval = tl.load(
                g_base + offs_t[:, None] * sg_t + offs_k[None, :] * sg_d,
                mask=mask_t[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            state = tl.load(
                state_base + offs_k[:, None] * D_V + offs_v[None, :],
                mask=mask_k[:, None] & mask_v[None, :],
                other=0.0,
            )

            gcum = tl.cumsum(gval, axis=0)
            exp_g = tl.exp(tl.minimum(gcum, 80.0))

            pred += tl.dot(kval * exp_g, state, input_precision="ieee")
            out_state += tl.dot(qval * exp_g, state, input_precision="ieee")

        rhs = beta[:, None] * (vval - pred)
        wmat = tl.dot(inv_mat, rhs, input_precision="ieee")
        out_mat = (out_state + tl.dot(qk_mat, wmat, input_precision="ieee")) * scale

        tl.store(
            out_base + offs_t[:, None] * so_t + offs_v[None, :] * so_d,
            out_mat,
            mask=mask_t[:, None] & mask_v[None, :],
        )

        # End-of-chunk state.  Each in-chunk update s is propagated by
        # exp(G[last] - G[s]) <= 1, and the incoming state by
        # exp(G[last]) <= 1; both are formed directly, never as a
        # ratio of a huge and a tiny exponential.
        for kb in range(NK):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D_K

            kval = tl.load(
                k_base + offs_t[:, None] * sk_t + offs_k[None, :] * sk_d,
                mask=mask_t[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            gval = tl.load(
                g_base + offs_t[:, None] * sg_t + offs_k[None, :] * sg_d,
                mask=mask_t[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            state = tl.load(
                state_base + offs_k[:, None] * D_V + offs_v[None, :],
                mask=mask_k[:, None] & mask_v[None, :],
                other=0.0,
            )

            gcum = tl.cumsum(gval, axis=0)
            last_gc = tl.sum(
                tl.where(offs_c[:, None] == (chunk_len - 1), gcum, 0.0),
                axis=0,
            )
            exp_last = tl.exp(tl.minimum(last_gc, 80.0))
            exp_end = tl.exp(tl.minimum(last_gc[None, :] - gcum, 80.0))

            state_new = state * exp_last[:, None] + tl.dot(
                tl.trans(kval * exp_end), wmat, input_precision="ieee"
            )
            tl.store(
                state_base + offs_k[:, None] * D_V + offs_v[None, :],
                state_new,
                mask=mask_k[:, None] & mask_v[None, :],
            )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # e and gamma are intentionally unused, exactly as in the reference.
    B, H, T, K = q.shape
    V = v.shape[-1]

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0 or K == 0 or V == 0:
        return out

    CHUNK = 32
    BLOCK_K = 64
    BLOCK_V = 32

    num_iter = 0
    while (1 << num_iter) < CHUNK:
        num_iter += 1

    bh = B * H
    nt = (T + CHUNK - 1) // CHUNK
    nk = (K + BLOCK_K - 1) // BLOCK_K

    inv = torch.empty(
        (bh * nt * CHUNK * CHUNK,), device=q.device, dtype=torch.float32
    )
    qk = torch.empty(
        (bh * nt * CHUNK * CHUNK,), device=q.device, dtype=torch.float32
    )
    state = torch.zeros((bh, K, V), device=q.device, dtype=torch.float32)

    _chunk_inv_kernel[(nt, bh)](
        q_ptr=q, k_ptr=k, g_ptr=g, beta_ptr=beta, inv_ptr=inv, qk_ptr=qk,
        T=T, H=H, NT=nt, D_K=K,
        sq_b=q.stride(0), sq_h=q.stride(1), sq_t=q.stride(2), sq_d=q.stride(3),
        sk_b=k.stride(0), sk_h=k.stride(1), sk_t=k.stride(2), sk_d=k.stride(3),
        sg_b=g.stride(0), sg_h=g.stride(1), sg_t=g.stride(2), sg_d=g.stride(3),
        sb_b=beta.stride(0), sb_h=beta.stride(1), sb_t=beta.stride(2),
        CHUNK=CHUNK, NUM_ITER=num_iter,
        num_warps=8, num_stages=1,
    )

    _chunk_state_kernel[(bh, triton.cdiv(V, BLOCK_V))](
        q_ptr=q, k_ptr=k, v_ptr=v, g_ptr=g, beta_ptr=beta,
        inv_ptr=inv, qk_ptr=qk, state_ptr=state, out_ptr=out,
        T=T, H=H, NT=nt, D_K=K, D_V=V, NK=nk, scale=float(scale),
        sq_b=q.stride(0), sq_h=q.stride(1), sq_t=q.stride(2), sq_d=q.stride(3),
        sk_b=k.stride(0), sk_h=k.stride(1), sk_t=k.stride(2), sk_d=k.stride(3),
        sv_b=v.stride(0), sv_h=v.stride(1), sv_t=v.stride(2), sv_d=v.stride(3),
        sg_b=g.stride(0), sg_h=g.stride(1), sg_t=g.stride(2), sg_d=g.stride(3),
        sb_b=beta.stride(0), sb_h=beta.stride(1), sb_t=beta.stride(2),
        so_b=out.stride(0), so_h=out.stride(1), so_t=out.stride(2), so_d=out.stride(3),
        CHUNK=CHUNK, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        num_warps=4, num_stages=1,
    )
    return out
```

The violated assumption was that a chunk's cumulative decay fits around one balanced reference level: with `g` in `[-25, -5]`, a 32-step chunk accumulates roughly `-160` to `-800` of log-decay, so the old `exp(±(gcum − cref))` factors hit the ±80 clamp and reconstructed ratios such as `exp(G_t − G_s)` as 1 instead of their true tiny values. The fix removes the reference level entirely — the coupling/output matrices are now accumulated per K-channel from the directly-formed factor `exp(G_t − G_s) ∈ (0, 1]`, and the incoming/end-of-chunk state terms use `exp(G_t)` and `exp(G_last − G_s)`, all non-positive exponents that correctly underflow to zero under strong decay. Tile sizes, `num_stages=1`, chunk size, and all other previously-passing behaviour are unchanged.