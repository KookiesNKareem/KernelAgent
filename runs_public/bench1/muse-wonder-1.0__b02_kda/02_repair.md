```python
import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_inv_kernel(
    k_ptr, g_ptr, beta_ptr, inv_ptr,
    T, H, NT, D_K, NK,
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
    offs_t = pid_c * CHUNK + offs_c
    mask_t = offs_t < T

    k_base = k_ptr + pid_b * sk_b + pid_h * sk_h
    g_base = g_ptr + pid_b * sg_b + pid_h * sg_h
    beta_base = beta_ptr + pid_b * sb_b + pid_h * sb_h

    beta = tl.load(
        beta_base + offs_t * sb_t,
        mask=mask_t,
        other=0.0,
    ).to(tl.float32)

    chunk_len = tl.minimum(T - pid_c * CHUNK, CHUNK)

    a_mat = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)

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

        gcum = tl.cumsum(gval, axis=0)
        last_gc = tl.sum(
            tl.where(offs_c[:, None] == (chunk_len - 1), gcum, 0.0),
            axis=0,
        )
        cref = 0.5 * last_gc
        diff = gcum - cref[None, :]
        diff = tl.minimum(tl.maximum(diff, -80.0), 80.0)
        exp_p = tl.exp(diff)
        exp_m = tl.exp(-diff)

        k_up = kval * exp_p
        k_down = kval * exp_m

        a_mat += tl.dot(
            k_up * beta[:, None],
            tl.trans(k_down),
            input_precision="ieee",
        )

    row = offs_c[:, None]
    col = offs_c[None, :]
    eye = tl.where(col == row, 1.0, 0.0)
    m_mat = eye + tl.where(col < row, a_mat, 0.0)

    # Exact Newton inversion of I + A for nilpotent strictly-lower A.
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
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, inv_ptr, state_ptr, out_ptr,
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

        inv_mat = tl.load(
            inv_ptr
            + ((pid_bh * NT + chunk_idx) * CHUNK + offs_c[:, None]) * CHUNK
            + offs_c[None, :]
        )

        chunk_len = tl.minimum(T - t_start, CHUNK)

        pred = tl.zeros([CHUNK, BLOCK_V], dtype=tl.float32)
        out_state = tl.zeros([CHUNK, BLOCK_V], dtype=tl.float32)
        qk = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)

        # First pass over K tiles: accumulate everything that needs
        # the incoming state.
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

            state_hat = state * exp_c[:, None]
            pred += tl.dot(kval * exp_p, state_hat, input_precision="ieee")
            out_state += tl.dot(qval * exp_p, state_hat, input_precision="ieee")
            qk += tl.dot(
                qval * exp_p,
                tl.trans(kval * exp_m),
                input_precision="ieee",
            )

        qk = tl.where(offs_c[None, :] <= offs_c[:, None], qk, 0.0)
        rhs = beta[:, None] * (vval - pred)
        wmat = tl.dot(inv_mat, rhs, input_precision="ieee")
        out_mat = (out_state + tl.dot(qk, wmat, input_precision="ieee")) * scale

        tl.store(
            out_base + offs_t[:, None] * so_t + offs_v[None, :] * so_d,
            out_mat,
            mask=mask_t[:, None] & mask_v[None, :],
        )

        # Second pass over K tiles: write the updated state back.
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
            cref = 0.5 * last_gc
            exp_c = tl.exp(tl.minimum(tl.maximum(cref, -80.0), 80.0))
            diff = gcum - cref[None, :]
            diff = tl.minimum(tl.maximum(diff, -80.0), 80.0)
            exp_m = tl.exp(-diff)

            state_hat = state * exp_c[:, None]
            state_hat += tl.dot(
                tl.trans(kval * exp_m), wmat, input_precision="ieee"
            )
            tl.store(
                state_base + offs_k[:, None] * D_V + offs_v[None, :],
                state_hat * exp_c[:, None],
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
    state = torch.zeros((bh, K, V), device=q.device, dtype=torch.float32)

    _chunk_inv_kernel[(nt, bh)](
        k_ptr=k, g_ptr=g, beta_ptr=beta, inv_ptr=inv,
        T=T, H=H, NT=nt, D_K=K, NK=nk,
        sk_b=k.stride(0), sk_h=k.stride(1), sk_t=k.stride(2), sk_d=k.stride(3),
        sg_b=g.stride(0), sg_h=g.stride(1), sg_t=g.stride(2), sg_d=g.stride(3),
        sb_b=beta.stride(0), sb_h=beta.stride(1), sb_t=beta.stride(2),
        CHUNK=CHUNK, BLOCK_K=BLOCK_K, NUM_ITER=num_iter,
        num_warps=4, num_stages=1,
    )

    _chunk_state_kernel[(bh, triton.cdiv(V, BLOCK_V))](
        q_ptr=q, k_ptr=k, v_ptr=v, g_ptr=g, beta_ptr=beta,
        inv_ptr=inv, state_ptr=state, out_ptr=out,
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

The failure was purely a resource assumption: the original 64×128×64 tiles plus default software pipelining needed 311 KB of shared memory, over the L4's 101 KB limit, on every input regardless of values. I kept the chunkwise gated-delta mathematics unchanged but shrank the tiles to 32 (time) × 64 (K, now looped over so K > 64 and non-powers of two work) × 32 (V), keep the cross-chunk state in a global scratch buffer instead of one huge register/shared tile, and force `num_stages=1` on both launches. I could not test locally, so the residual risk is that Triton's layout conversions still allocate more shared memory than my estimate or that the extra global state traffic costs some speed.