```python
import torch
import triton
import triton.language as tl


@triton.jit
def _register_kernel(
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
    K_FULL: tl.constexpr,
    V_FULL: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    bh = pid // NVB
    vb = pid % NVB
    b_idx = bh // H
    h_idx = bh % H

    offs_k = tl.arange(0, BK)
    offs_v = vb * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V

    q_ptrs = q_ptr + b_idx * sq_b + h_idx * sq_h + offs_k * sq_k
    k_ptrs = k_ptr + b_idx * sk_b + h_idx * sk_h + offs_k * sk_k
    g_ptrs = g_ptr + b_idx * sg_b + h_idx * sg_h + offs_k * sg_k
    v_ptrs = v_ptr + b_idx * sv_b + h_idx * sv_h + offs_v * sv_v
    beta_ptr_cur = beta_ptr + b_idx * sb_b + h_idx * sb_h
    out_ptrs = out_ptr + bh * T * V + offs_v

    # State transposed to [V, K]: each program owns a V-slice of S^T.
    # Different V-slices need no communication.
    S = tl.zeros([BV, BK], dtype=tl.float32)

    for _ in range(0, T):
        if K_FULL:
            qv = tl.load(q_ptrs).to(tl.float32)
            kv = tl.load(k_ptrs).to(tl.float32)
            gv = tl.load(g_ptrs).to(tl.float32)
        else:
            qv = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
            kv = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
            gv = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        if V_FULL:
            vv = tl.load(v_ptrs).to(tl.float32)
        else:
            vv = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)

        beta_t = tl.load(beta_ptr_cur).to(tl.float32)

        # Decay, then read (exactly the reference order).
        S = S * tl.exp(gv)[None, :]

        o = tl.sum(S * qv[None, :], axis=1)
        if V_FULL:
            tl.store(out_ptrs, o * scale)
        else:
            tl.store(out_ptrs, o * scale, mask=mask_v)

        # Delta-rule write for this V-slice.
        pred = tl.sum(S * kv[None, :], axis=1)
        delta = beta_t * (vv - pred)
        S = S + delta[:, None] * kv[None, :]

        q_ptrs += sq_t
        k_ptrs += sk_t
        g_ptrs += sg_t
        v_ptrs += sv_t
        beta_ptr_cur += sb_t
        out_ptrs += V


@triton.jit
def _global_kernel(
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
    # Fallback for very large K: same recurrence, state in global scratch.
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
            s_ptrs = state_ptr + base_state + offs_k[:, None] * V + offs_v[None, :]
            s_tile = tl.load(s_ptrs, mask=mask_kv, other=0.0).to(tl.float32)
            qv = tl.load(
                q_ptr + base_q + t * sq_t + offs_k * sq_k, mask=mask_k, other=0.0
            ).to(tl.float32)
            kv = tl.load(
                k_ptr + base_k + t * sk_t + offs_k * sk_k, mask=mask_k, other=0.0
            ).to(tl.float32)
            gv = tl.load(
                g_ptr + base_g + t * sg_t + offs_k * sg_k, mask=mask_k, other=0.0
            ).to(tl.float32)

            s_decayed = s_tile * tl.exp(gv)[:, None]
            tl.store(s_ptrs, s_decayed, mask=mask_kv)
            o_vec = o_vec + tl.sum(s_decayed * qv[:, None], axis=0)
            pred = pred + tl.sum(s_decayed * kv[:, None], axis=0)

        tl.store(out_ptr + base_out + t * V + offs_v, o_vec * scale, mask=mask_v)

        vv = tl.load(
            v_ptr + base_v + t * sv_t + offs_v * sv_v, mask=mask_v, other=0.0
        ).to(tl.float32)
        beta_t = tl.load(beta_ptr + base_beta + t * sb_t).to(tl.float32)
        delta = beta_t * (vv - pred)

        for ks in range(0, K, BK):
            offs_k = ks + offs_kb
            mask_k = offs_k < K
            mask_kv = mask_k[:, None] & mask_v[None, :]
            s_ptrs = state_ptr + base_state + offs_k[:, None] * V + offs_v[None, :]
            s_tile = tl.load(s_ptrs, mask=mask_kv, other=0.0).to(tl.float32)
            kv = tl.load(
                k_ptr + base_k + t * sk_t + offs_k * sk_k, mask=mask_k, other=0.0
            ).to(tl.float32)
            tl.store(s_ptrs, s_tile + kv[:, None] * delta[None, :], mask=mask_kv)


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
    scale_f = float(scale)

    strides = (
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
    )

    BK = max(16, _next_pow2(K))

    if BK <= 256:
        if BK <= 64:
            BV = 64
        elif BK <= 128:
            BV = 32
        else:
            BV = 16
        nvb = (V + BV - 1) // BV
        _register_kernel[(B * H * nvb,)](
            q, k, v, g, beta, out,
            T, H, K, V, nvb, scale_f,
            *strides,
            BK=BK,
            BV=BV,
            K_FULL=(K == BK),
            V_FULL=(V % BV == 0),
            num_warps=4,
            num_stages=2,
        )
    else:
        BV = 32
        nvb = (V + BV - 1) // BV
        state = torch.zeros((B * H * K * V,), device=q.device, dtype=torch.float32)
        _global_kernel[(B * H * nvb,)](
            q, k, v, g, beta, state, out,
            T, H, K, V, nvb, scale_f,
            *strides,
            BK=32,
            BV=BV,
            num_warps=4,
            num_stages=1,
        )
    return out
```

The previous kernel spent its time in many small IEEE-precision chunk matmuls and a Newton inversion. This version removes that entirely: because the delta-rule state partitions independently across `V`, each program keeps a `[BV, BK]` slice of the transposed state in registers and runs the exact decay-read-write recurrence in fp32, with pointer increments and mask-free loads specialized for the common full-tile case. The only remaining uncertainty is Triton's reduction layout for the state tile, though eliminating the chunk inversion and repeated state traffic should dominate that cost.