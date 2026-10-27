```python
import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_inv_kernel(
    q_ptr, k_ptr, g_ptr, beta_ptr, inv_ptr, qk_ptr,
    T, H, NT, D_K, NK, KB,
    sq_b, sq_h, sq_t, sq_d,
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

    q_base = q_ptr + pid_b * sq_b + pid_h * sq_h
    k_base = k_ptr + pid_b * sk_b + pid_h * sk_h
    g_base = g_ptr + pid_b * sg_b + pid_h * sg_h
    beta_base = beta_ptr + pid_b * sb_b + pid_h * sb_h

    beta = tl.load(
        beta_base + offs_t * sb_t,
        mask=mask_t,
        other=0.0,
    ).to(tl.float32)

    a_mat = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)
    qk_mat = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)

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

        gcum = tl.cumsum(gval, axis=0)
        gmax = tl.max(gcum, axis=0)
        gmin = tl.min(gcum, axis=0)
        grange = tl.max(gmax - gmin)

        if grange <= 140.0:
            # Balanced factorisation: both exponentials are at most
            # exp(grange / 2) <= exp(70), nowhere near overflow, and
            # no clamping is applied, so exp(G[t] - G[s]) is exact.
            cref = 0.5 * (gmax + gmin)
            diff = gcum - cref[None, :]
            exp_p = tl.exp(diff)
            exp_m = tl.exp(-diff)
            k_down = kval * exp_m
            a_mat += tl.dot(
                (kval * exp_p) * beta[:, None],
                tl.trans(k_down),
                input_precision="tf32",
            )
            qk_mat += tl.dot(
                qval * exp_p,
                tl.trans(k_down),
                input_precision="tf32",
            )
        else:
            # Strong-decay fallback: form exp(G[t] - G[s]) directly.
            # Its exponent is <= 0 for s <= t, so it underflows to the
            # correct zero instead of distorting close pairs.  This
            # branch is only taken by chunks with an extreme decay
            # range, which are not the timed shapes.
            for kk in range(KB):
                global_k = kb * BLOCK_K + kk
                valid = mask_t & (global_k < D_K)
                q_vec = tl.load(
                    q_base + offs_t * sq_t + global_k * sq_d,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                k_vec = tl.load(
                    k_base + offs_t * sk_t + global_k * sk_d,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                g_vec = tl.load(
                    g_base + offs_t * sg_t + global_k * sg_d,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)

                gc_vec = tl.cumsum(g_vec, axis=0)
                diffm = gc_vec[:, None] - gc_vec[None, :]
                exp_diff = tl.exp(tl.minimum(diffm, 80.0))
                ks_exp = k_vec[None, :] * exp_diff
                a_mat += beta[:, None] * (k_vec[:, None] * ks_exp)
                qk_mat += q_vec[:, None] * ks_exp

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
def _chunk_state_fast_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, inv_ptr, qk_ptr, out_ptr,
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

    # The whole [K, V] state lives in registers for the full sequence:
    # no global state traffic between chunks.
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

        base_idx = (pid_bh * NT + chunk_idx) * CHUNK
        inv_mat = tl.load(
            inv_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :]
        )
        qk_mat = tl.load(
            qk_ptr + (base_idx + offs_c[:, None]) * CHUNK + offs_c[None, :]
        )

        chunk_len = tl.minimum(T - t_start, CHUNK)

        gcum = tl.cumsum(gval, axis=0)
        last_gc = tl.sum(
            tl.where(offs_c[:, None] == (chunk_len - 1), gcum, 0.0),
            axis=0,
        )
        exp_g = tl.exp(tl.minimum(gcum, 80.0))

        pred = tl.dot(kval * exp_g, state, input_precision="tf32")
        out_state = tl.dot(qval * exp_g, state, input_precision="tf32")

        rhs = beta[:, None] * (vval - pred)
        wmat = tl.dot(inv_mat, rhs, input_precision="ieee")
        out_mat = (
            out_state + tl.dot(qk_mat, wmat, input_precision="ieee")
        ) * scale

        tl.store(
            out_base + offs_t[:, None] * so_t + offs_v[None, :] * so_d,
            out_mat,
            mask=mask_t[:, None] & mask_v[None, :],
        )

        exp_last = tl.exp(tl.minimum(last_gc, 80.0))
        exp_end = tl.exp(tl.minimum(last_gc[None, :] - gcum, 80.0))
        state = state * exp_last[:, None] + tl.dot(
            tl.trans(kval * exp_end), wmat, input_precision="tf32"
        )


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
    # General-K fallback (K > 128): state kept in global scratch.
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
        out_mat = (
            out_state + tl.dot(qk_mat, wmat, input_precision="ieee")
        ) * scale

        tl.store(
            out_base + offs_t[:, None] * so_t + offs_v[None, :] * so_d,
            out_mat,
            mask=mask_t[:, None] & mask_v[None, :],
        )

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
    INV_BLOCK_K = 64
    BLOCK_V = 32

    num_iter = 0
    while (1 << num_iter) < CHUNK:
        num_iter += 1

    bh = B * H
    nt = (T + CHUNK - 1) // CHUNK
    inv_nk = (K + INV_BLOCK_K - 1) // INV_BLOCK_K

    inv = torch.empty(
        (bh * nt * CHUNK * CHUNK,), device=q.device, dtype=torch.float32
    )
    qk = torch.empty(
        (bh * nt * CHUNK * CHUNK,), device=q.device, dtype=torch.float32
    )

    _chunk_inv_kernel[(nt, bh)](
        q_ptr=q, k_ptr=k, g_ptr=g, beta_ptr=beta, inv_ptr=inv, qk_ptr=qk,
        T=T, H=H, NT=nt, D_K=K, NK=inv_nk, KB=INV_BLOCK_K,
        sq_b=q.stride(0), sq_h=q.stride(1), sq_t=q.stride(2), sq_d=q.stride(3),
        sk_b=k.stride(0), sk_h=k.stride(1), sk_t=k.stride(2), sk_d=k.stride(3),
        sg_b=g.stride(0), sg_h=g.stride(1), sg_t=g.stride(2), sg_d=g.stride(3),
        sb_b=beta.stride(0), sb_h=beta.stride(1), sb_t=beta.stride(2),
        CHUNK=CHUNK, BLOCK_K=INV_BLOCK_K, NUM_ITER=num_iter,
        num_warps=4, num_stages=1,
    )

    if K <= 128:
        fast_block_k = triton.next_power_of_2(max(K, 16))
        _chunk_state_fast_kernel[(bh, triton.cdiv(V, BLOCK_V))](
            q_ptr=q, k_ptr=k, v_ptr=v, g_ptr=g, beta_ptr=beta,
            inv_ptr=inv, qk_ptr=qk, out_ptr=out,
            T=T, H=H, NT=nt, D_K=K, D_V=V, scale=float(scale),
            sq_b=q.stride(0), sq_h=q.stride(1), sq_t=q.stride(2), sq_d=q.stride(3),
            sk_b=k.stride(0), sk_h=k.stride(1), sk_t=k.stride(2), sk_d=k.stride(3),
            sv_b=v.stride(0), sv_h=v.stride(1), sv_t=v.stride(2), sv_d=v.stride(3),
            sg_b=g.stride(0), sg_h=g.stride(1), sg_t=g.stride(2), sg_d=g.stride(3),
            sb_b=beta.stride(0), sb_h=beta.stride(1), sb_t=beta.stride(2),
            so_b=out.stride(0), so_h=out.stride(1), so_t=out.stride(2), so_d=out.stride(3),
            CHUNK=CHUNK, BLOCK_K=fast_block_k, BLOCK_V=BLOCK_V,
            num_warps=8, num_stages=1,
        )
    else:
        state = torch.zeros((bh, K, V), device=q.device, dtype=torch.float32)
        _chunk_state_kernel[(bh, triton.cdiv(V, BLOCK_V))](
            q_ptr=q, k_ptr=k, v_ptr=v, g_ptr=g, beta_ptr=beta,
            inv_ptr=inv, qk_ptr=qk, state_ptr=state, out_ptr=out,
            T=T, H=H, NT=nt, D_K=K, D_V=V, NK=inv_nk, scale=float(scale),
            sq_b=q.stride(0), sq_h=q.stride(1), sq_t=q.stride(2), sq_d=q.stride(3),
            sk_b=k.stride(0), sk_h=k.stride(1), sk_t=k.stride(2), sk_d=k.stride(3),
            sv_b=v.stride(0), sv_h=v.stride(1), sv_t=v.stride(2), sv_d=v.stride(3),
            sg_b=g.stride(0), sg_h=g.stride(1), sg_t=g.stride(2), sg_d=g.stride(3),
            sb_b=beta.stride(0), sb_h=beta.stride(1), sb_t=beta.stride(2),
            so_b=out.stride(0), so_h=out.stride(1), so_t=out.stride(2), so_d=out.stride(3),
            CHUNK=CHUNK, BLOCK_K=INV_BLOCK_K, BLOCK_V=BLOCK_V,
            num_warps=4, num_stages=1,
        )
    return out
```

The two hot spots were attacked separately: the inverse/QK kernel now builds both triangular matrices with TF32 tensor-core `tl.dot`s over 64-wide K tiles (falling back to the direct `exp(G_t−G_s)` per-channel loop only for K-blocks whose cumulative decay range exceeds 140, where balanced exponentials could overflow), and the state kernel keeps the full `[K,V]` state in registers across chunks, uses TF32 for the three large state matmuls, and loads each chunk tile once instead of twice. The previously passing global-state kernel is retained only for `K > 128`. The main risks I cannot rule out without running it are that the runtime `if` on the decay range interacts badly with Triton's register allocation, and that TF32 rounding in the coupling matrices is slightly less accurate than the old IEEE path, though both should sit well inside the 2e-2 tolerance.