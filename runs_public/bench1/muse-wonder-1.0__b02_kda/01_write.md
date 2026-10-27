```python
import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_inv_kernel(
    k_ptr, g_ptr, beta_ptr, inv_ptr,
    T, H, NT, D_K,
    sk_b, sk_h, sk_t, sk_d,
    sg_b, sg_h, sg_t, sg_d,
    sb_b, sb_h, sb_t,
    CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, NUM_ITER: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BLOCK_K)
    offs_t = pid_c * CHUNK + offs_c
    mask_t = offs_t < T
    mask_k = offs_k < D_K

    k_base = k_ptr + pid_b * sk_b + pid_h * sk_h
    g_base = g_ptr + pid_b * sg_b + pid_h * sg_h
    beta_base = beta_ptr + pid_b * sb_b + pid_h * sb_h

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
    beta = tl.load(
        beta_base + offs_t * sb_t,
        mask=mask_t,
        other=0.0,
    ).to(tl.float32)

    gcum = tl.cumsum(gval, axis=0)

    chunk_len = T - pid_c * CHUNK
    chunk_len = tl.minimum(chunk_len, CHUNK)
    last_gc = tl.sum(
        tl.where(offs_c[:, None] == (chunk_len - 1), gcum, 0.0),
        axis=0,
    )

    # Balanced reference point for this chunk: both exp(diff) and
    # exp(-diff) then have at most half the chunk's decay range.
    cref = 0.5 * last_gc
    diff = gcum - cref[None, :]
    diff = tl.minimum(tl.maximum(diff, -80.0), 80.0)
    exp_p = tl.exp(diff)
    exp_m = tl.exp(-diff)

    k_up = kval * exp_p
    k_down = kval * exp_m

    row = offs_c[:, None]
    col = offs_c[None, :]

    a_mat = tl.dot(
        k_up * beta[:, None],
        tl.trans(k_down),
        input_precision="ieee",
    )
    a_mat = tl.where(col < row, a_mat, 0.0)
    eye = tl.where(col == row, 1.0, 0.0)
    m_mat = eye + a_mat

    # (I + A)^-1 for strictly-lower A by Newton iteration.  A is
    # nilpotent, so ceil(log2(CHUNK)) squarings make the error
    # exactly zero in exact arithmetic.
    x_mat = eye
    for _ in range(NUM_ITER):
        mx = tl.dot(m_mat, x_mat, input_precision="ieee")
        x_mat = tl.dot(x_mat, 2.0 * eye - mx, input_precision="ieee")

    inv_ptrs = (
        inv_ptr
        + ((pid_bh * NT + pid_c) * CHUNK + offs_c[:, None]) * CHUNK
        + offs_c[None, :]
    )
    tl.store(inv_ptrs, x_mat)


@triton.jit
def _chunk_state_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, inv_ptr, out_ptr,
    T, H, NT, D_K, D_V, scale,
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
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < D_K
    mask_v = offs_v < D_V

    q_base = q_ptr + pid_b * sq_b + pid_h * sq_h
    k_base = k_ptr + pid_b * sk_b + pid_h * sk_h
    v_base = v_ptr + pid_b * sv_b + pid_h * sv_h
    g_base = g_ptr + pid_b * sg_b + pid_h * sg_h
    beta_base = beta_ptr + pid_b * sb_b + pid_h * sb_h
    out_base = out_ptr + pid_b * so_b + pid_h * so_h

    state = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    for chunk_idx in range(NT):
        t_start = chunk_idx * CHUNK
        offs_t = t_start + offs_c
        mask_t = offs_t < T

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
        vval = tl.load(
            v_base + offs_t[:, None] * sv_t + offs_v[None, :] * sv_d,
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        gval = tl.load(
            g_base + offs_t[:, None] * sg_t + offs_k[None, :] * sg_d,
            mask=mask_t[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        beta = tl.load(
            beta_base + offs_t * sb_t,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)

        gcum = tl.cumsum(gval, axis=0)

        chunk_len = T - t_start
        chunk_len = tl.minimum(chunk_len, CHUNK)
        last_gc = tl.sum(
            tl.where(offs_c[:, None] == (chunk_len - 1), gcum, 0.0),
            axis=0,
        )

        cref = 0.5 * last_gc
        exp_c = tl.exp(tl.minimum(tl.maximum(cref, -80.0), 80.0))
        diff = gcum - cref[None, :]
        diff = tl.minimum(tl.maximum(diff, -80.0), 80.0)
        exp_p = tl.exp(diff)
        exp_m = tl.exp(-diff)

        k_up = kval * exp_p
        k_down = kval * exp_m
        q_up = qval * exp_p

        # State expressed in this chunk's balanced coordinates.
        state_hat = state * exp_c[:, None]

        inv_mat = tl.load(
            inv_ptr
            + ((pid_bh * NT + chunk_idx) * CHUNK + offs_c[:, None]) * CHUNK
            + offs_c[None, :]
        )

        pred = tl.dot(k_up, state_hat, input_precision="ieee")
        rhs = beta[:, None] * (vval - pred)
        wmat = tl.dot(inv_mat, rhs, input_precision="ieee")

        out_state = tl.dot(q_up, state_hat, input_precision="ieee")
        qk = tl.dot(q_up, tl.trans(k_down), input_precision="ieee")
        qk = tl.where(offs_c[None, :] <= offs_c[:, None], qk, 0.0)
        out_mat = (out_state + tl.dot(qk, wmat, input_precision="ieee")) * scale

        tl.store(
            out_base + offs_t[:, None] * so_t + offs_v[None, :] * so_d,
            out_mat,
            mask=mask_t[:, None] & mask_v[None, :],
        )

        state_hat = state_hat + tl.dot(
            tl.trans(k_down), wmat, input_precision="ieee"
        )
        # exp(last_gc - cref) == exp(cref), back to unscaled state.
        state = state_hat * exp_c[:, None]


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # e and gamma are intentionally unused, exactly as in the reference.
    B, H, T, K = q.shape
    V = v.shape[-1]

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0 or V == 0:
        return out

    CHUNK = 64
    BLOCK_K = triton.next_power_of_2(max(K, 16))
    if V >= 64:
        BLOCK_V = 64
    elif V >= 32:
        BLOCK_V = 32
    else:
        BLOCK_V = 16

    num_iter = 0
    while (1 << num_iter) < CHUNK:
        num_iter += 1

    bh = B * H
    nt = (T + CHUNK - 1) // CHUNK

    inv = torch.empty(
        (bh * nt * CHUNK * CHUNK,), device=q.device, dtype=torch.float32
    )

    _chunk_inv_kernel[(nt, bh)](
        k_ptr=k, g_ptr=g, beta_ptr=beta, inv_ptr=inv,
        T=T, H=H, NT=nt, D_K=K,
        sk_b=k.stride(0), sk_h=k.stride(1), sk_t=k.stride(2), sk_d=k.stride(3),
        sg_b=g.stride(0), sg_h=g.stride(1), sg_t=g.stride(2), sg_d=g.stride(3),
        sb_b=beta.stride(0), sb_h=beta.stride(1), sb_t=beta.stride(2),
        CHUNK=CHUNK, BLOCK_K=BLOCK_K, NUM_ITER=num_iter,
        num_warps=4,
    )

    _chunk_state_kernel[(bh, triton.cdiv(V, BLOCK_V))](
        q_ptr=q, k_ptr=k, v_ptr=v, g_ptr=g, beta_ptr=beta,
        inv_ptr=inv, out_ptr=out,
        T=T, H=H, NT=nt, D_K=K, D_V=V, scale=float(scale),
        sq_b=q.stride(0), sq_h=q.stride(1), sq_t=q.stride(2), sq_d=q.stride(3),
        sk_b=k.stride(0), sk_h=k.stride(1), sk_t=k.stride(2), sk_d=k.stride(3),
        sv_b=v.stride(0), sv_h=v.stride(1), sv_t=v.stride(2), sv_d=v.stride(3),
        sg_b=g.stride(0), sg_h=g.stride(1), sg_t=g.stride(2), sg_d=g.stride(3),
        sb_b=beta.stride(0), sb_h=beta.stride(1), sb_t=beta.stride(2),
        so_b=out.stride(0), so_h=out.stride(1), so_t=out.stride(2), so_d=out.stride(3),
        CHUNK=CHUNK, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        num_warps=8,
    )
    return out
```

I used the chunkwise form of the per-channel gated delta rule: within each chunk, cumulative decay is folded into scaled queries/keys, the strictly-lower-triangular delta-rule coupling is inverted in parallel (via Newton iteration, exact for a nilpotent matrix), and a second kernel carries the `[K,V]` state across chunks while computing outputs and the state update with `tl.dot`. All arithmetic inside the kernels is fp32, with decay exponentials recentred per chunk for stability, and `e`/`gamma` are ignored as in the reference. Since I could not execute anything, I am least certain about Triton 3.5 API details such as the `input_precision` keyword and about register pressure/performance at `BLOCK_V=64`, as well as behaviour for extreme decay ranges where the balanced exponentials are clamped at ±80.