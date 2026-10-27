```python
import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    out_ptr,
    T,
    H,
    K,
    V,
    NVB,
    scale,
    sq_b,
    sq_h,
    sq_t,
    sq_k,
    sk_b,
    sk_h,
    sk_t,
    sk_k,
    sv_b,
    sv_h,
    sv_t,
    sv_v,
    sg_b,
    sg_h,
    sg_t,
    sg_k,
    sb_b,
    sb_h,
    sb_t,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    bh = pid // NVB
    vb = pid % NVB
    b_idx = bh // H
    h_idx = bh % H

    base_q = b_idx * sq_b + h_idx * sq_h
    base_k = b_idx * sk_b + h_idx * sk_h
    base_v = b_idx * sv_b + h_idx * sv_h
    base_g = b_idx * sg_b + h_idx * sg_h
    base_beta = b_idx * sb_b + h_idx * sb_h
    base_out = bh * T * V

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = vb * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V

    strict = offs_c[:, None] > offs_c[None, :]
    eye = tl.where(offs_c[:, None] == offs_c[None, :], 1.0, 0.0)

    # State for this (batch, head, value-block), in registers.
    S = tl.zeros([BK, BV], dtype=tl.float32)

    for start in range(0, T, CHUNK):
        t_idx = start + offs_c
        mask_t = t_idx < T
        mask_ck = mask_t[:, None] & mask_k[None, :]
        mask_cv = mask_t[:, None] & mask_v[None, :]

        g = tl.load(
            g_ptr + base_g + t_idx[:, None] * sg_t + offs_k[None, :] * sg_k,
            mask=mask_ck,
            other=0.0,
        ).to(tl.float32)

        c = tl.cumsum(g, axis=0)
        c_min = tl.min(c, axis=0)
        c_max = tl.max(c, axis=0)
        range_max = tl.max(c_max - c_min)

        if range_max <= 100.0:
            # Chunkwise matrix path.  CHUNK=32 and BV=32 (with one
            # pipeline stage) keep the dot/shared-memory footprint
            # far below the L4 limit, unlike 64x128 blocking.
            total = tl.sum(g, axis=0)
            pivot = (c_min + c_max) * 0.5

            q = tl.load(
                q_ptr + base_q + t_idx[:, None] * sq_t + offs_k[None, :] * sq_k,
                mask=mask_ck,
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                k_ptr + base_k + t_idx[:, None] * sk_t + offs_k[None, :] * sk_k,
                mask=mask_ck,
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptr + base_v + t_idx[:, None] * sv_t + offs_v[None, :] * sv_v,
                mask=mask_cv,
                other=0.0,
            ).to(tl.float32)
            beta = tl.load(
                beta_ptr + base_beta + t_idx * sb_t,
                mask=mask_t,
                other=0.0,
            ).to(tl.float32)

            ec = tl.exp(c - pivot[None, :])
            enc = tl.exp(pivot[None, :] - c)
            q_hat = q * ec
            k_hat = k * ec
            k_tilde = k * enc

            kk = tl.dot(k_hat, tl.trans(k_tilde), input_precision="ieee")
            a_mat = tl.where(strict, kk * beta[:, None], 0.0)

            # (I + A)^-1 for strictly-lower-triangular nilpotent A;
            # five Newton steps give all 32 terms exactly.
            b_mat = a_mat + eye
            inv = eye
            for _ in range(5):
                inv = inv + tl.dot(
                    inv,
                    eye - tl.dot(b_mat, inv, input_precision="ieee"),
                    input_precision="ieee",
                )

            s_hat = S * tl.exp(pivot)[:, None]

            w = tl.dot(inv, beta[:, None] * k_hat, input_precision="ieee")
            u = tl.dot(inv, beta[:, None] * v, input_precision="ieee") - tl.dot(
                w, s_hat, input_precision="ieee"
            )

            qk = tl.dot(q_hat, tl.trans(k_tilde), input_precision="ieee")
            qk = tl.where(strict, qk, 0.0)
            o = tl.dot(qk, u, input_precision="ieee") + tl.dot(
                q_hat, s_hat, input_precision="ieee"
            )

            tl.store(
                out_ptr + base_out + t_idx[:, None] * V + offs_v[None, :],
                o * scale,
                mask=mask_cv,
            )

            k_end = k * tl.exp(total[None, :] - c)
            S = S * tl.exp(total)[:, None] + tl.dot(
                tl.trans(k_end), u, input_precision="ieee"
            )
        else:
            # Very strong (or otherwise wide-range) decay in this chunk:
            # factored exponentials could overflow, so replay these
            # steps sequentially, using only per-step exp(g) exactly
            # as the reference does.
            for s in tl.static_range(CHUNK):
                t = start + s
                in_t = t < T
                mk = mask_k & in_t
                mv = mask_v & in_t

                qv = tl.load(
                    q_ptr + base_q + t * sq_t + offs_k * sq_k,
                    mask=mk,
                    other=0.0,
                ).to(tl.float32)
                kv = tl.load(
                    k_ptr + base_k + t * sk_t + offs_k * sk_k,
                    mask=mk,
                    other=0.0,
                ).to(tl.float32)
                gv = tl.load(
                    g_ptr + base_g + t * sg_t + offs_k * sg_k,
                    mask=mk,
                    other=0.0,
                ).to(tl.float32)
                vv = tl.load(
                    v_ptr + base_v + t * sv_t + offs_v * sv_v,
                    mask=mv,
                    other=0.0,
                ).to(tl.float32)
                bv_scalar = tl.load(
                    beta_ptr + base_beta + t * sb_t, mask=in_t, other=0.0
                ).to(tl.float32)

                s_decayed = S * tl.exp(gv)[:, None]
                ov = tl.sum(s_decayed * qv[:, None], axis=0)
                tl.store(
                    out_ptr + base_out + t * V + offs_v,
                    ov * scale,
                    mask=mv,
                )
                pred = tl.sum(s_decayed * kv[:, None], axis=0)
                S = s_decayed + kv[:, None] * (
                    bv_scalar * (vv - pred)
                )[None, :]


@triton.jit
def _recurrent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    state_ptr,
    out_ptr,
    T,
    H,
    K,
    V,
    NVB,
    scale,
    sq_b,
    sq_h,
    sq_t,
    sq_k,
    sk_b,
    sk_h,
    sk_t,
    sk_k,
    sv_b,
    sv_h,
    sv_t,
    sv_v,
    sg_b,
    sg_h,
    sg_t,
    sg_k,
    sb_b,
    sb_h,
    sb_t,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # Generic fallback for K > 128: state lives in global scratch and
    # is processed one step at a time, in K-blocks of BK.
    pid = tl.program_id(axis=0)
    bh = pid // NVB
    vb = pid % NVB
    b_idx = bh // H
    h_idx = bh % H

    base_q = b_idx * sq_b + h_idx * sq_h
    base_k = b_idx * sk_b + h_idx * sk_h
    base_v = b_idx * sv_b + h_idx * sv_h
    base_g = b_idx * sg_b + h_idx * sg_h
    base_beta = b_idx * sb_b + h_idx * sb_h
    base_state = bh * K * V
    base_out = bh * T * V

    offs_v = vb * BV + tl.arange(0, BV)
    offs_kb = tl.arange(0, BK)
    mask_v = offs_v < V

    for t in range(0, T):
        o_vec = tl.zeros([BV], dtype=tl.float32)
        pred = tl.zeros([BV], dtype=tl.float32)

        for ks in range(0, K, BK):
            offs_k = ks + offs_kb
            mask_k = offs_k < K
            mask_kv = mask_k[:, None] & mask_v[None, :]
            s_ptrs = (
                state_ptr
                + base_state
                + offs_k[:, None] * V
                + offs_v[None, :]
            )
            s_tile = tl.load(s_ptrs, mask=mask_kv, other=0.0).to(tl.float32)
            qv = tl.load(
                q_ptr + base_q + t * sq_t + offs_k * sq_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            kv = tl.load(
                k_ptr + base_k + t * sk_t + offs_k * sk_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            gv = tl.load(
                g_ptr + base_g + t * sg_t + offs_k * sg_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)

            s_decayed = s_tile * tl.exp(gv)[:, None]
            tl.store(s_ptrs, s_decayed, mask=mask_kv)
            o_vec = o_vec + tl.sum(s_decayed * qv[:, None], axis=0)
            pred = pred + tl.sum(s_decayed * kv[:, None], axis=0)

        tl.store(out_ptr + base_out + t * V + offs_v, o_vec * scale, mask=mask_v)

        vv = tl.load(
            v_ptr + base_v + t * sv_t + offs_v * sv_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        bv_scalar = tl.load(beta_ptr + base_beta + t * sb_t).to(tl.float32)
        delta = bv_scalar * (vv - pred)

        for ks in range(0, K, BK):
            offs_k = ks + offs_kb
            mask_k = offs_k < K
            mask_kv = mask_k[:, None] & mask_v[None, :]
            s_ptrs = (
                state_ptr
                + base_state
                + offs_k[:, None] * V
                + offs_v[None, :]
            )
            s_decayed = tl.load(s_ptrs, mask=mask_kv, other=0.0).to(tl.float32)
            kv = tl.load(
                k_ptr + base_k + t * sk_t + offs_k * sk_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                s_ptrs,
                s_decayed + kv[:, None] * delta[None, :],
                mask=mask_kv,
            )


def _next_pow2(x):
    n = 1
    while n < x:
        n *= 2
    return n


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # e and gamma are unused by the reference recurrence.
    B, H, T, K = q.shape
    V = v.shape[-1]

    if B == 0 or H == 0 or T == 0 or V == 0:
        return torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if K == 0:
        return torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    BV = 32
    nvb = (V + BV - 1) // BV
    grid = (B * H * nvb,)
    scale_f = float(scale)

    common = (
        T,
        H,
        K,
        V,
        nvb,
        scale_f,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        g.stride(0),
        g.stride(1),
        g.stride(2),
        g.stride(3),
        beta.stride(0),
        beta.stride(1),
        beta.stride(2),
    )

    if K <= 128:
        _chunk_kernel[grid](
            q,
            k,
            v,
            g,
            beta,
            out,
            *common,
            BK=max(16, _next_pow2(K)),
            BV=BV,
            CHUNK=32,
            num_warps=4,
            num_stages=1,
        )
    else:
        state = torch.zeros((B * H * K * V,), device=q.device, dtype=torch.float32)
        _recurrent_kernel[grid](
            q,
            k,
            v,
            g,
            beta,
            state,
            out,
            *common,
            BK=32,
            BV=BV,
            num_warps=4,
            num_stages=1,
        )
    return out
```

The failure was purely a resource-limit violation from my 64-step chunks, 128-wide value blocks, and two pipeline stages, so the fix reduces the chunk to 32, tiles values in blocks of 32, and uses `num_stages=1`. I also pass the input strides explicitly (handling strided/transposed views without copies), replay individual chunks sequentially when their cumulative-decay range is too wide for factored exponentials, and use a K-blocked recurrent fallback for K > 128. I remain unsure about the exact timing margin of the IEEE-precision dots on the L4, though the much smaller tiles should now compile comfortably within shared memory.