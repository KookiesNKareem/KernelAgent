```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward.
#
# One program owns a [BV, BK] slice of S^T in registers.  It follows the
# reference step-by-step and checkpoint S (in its original [K, V] layout)
# at every 32-step chunk boundary for the backward pass.
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    out_ptr,
    cp_ptr,
    T,
    H,
    K,
    V,
    NVB,
    NCH,
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
    mask_kv = mask_k[:, None] & mask_v[None, :]

    q_ptrs = q_ptr + b_idx * sq_b + h_idx * sq_h + offs_k * sq_k
    k_ptrs = k_ptr + b_idx * sk_b + h_idx * sk_h + offs_k * sk_k
    g_ptrs = g_ptr + b_idx * sg_b + h_idx * sg_h + offs_k * sg_k
    v_ptrs = v_ptr + b_idx * sv_b + h_idx * sv_h + offs_v * sv_v
    beta_ptr_cur = beta_ptr + b_idx * sb_b + h_idx * sb_h
    out_ptrs = out_ptr + bh * T * V + offs_v
    cp_base = cp_ptr + bh * (NCH + 1) * K * V

    S = tl.zeros([BV, BK], dtype=tl.float32)

    for chunk in range(0, NCH):
        tl.store(
            cp_base
            + chunk * K * V
            + offs_k[:, None] * V
            + offs_v[None, :],
            tl.trans(S),
            mask=mask_kv,
        )

        chunk_len = T - chunk * CHUNK
        chunk_len = tl.where(chunk_len > CHUNK, CHUNK, chunk_len)

        for _ in range(0, chunk_len):
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

            S = S * tl.exp(gv)[None, :]

            o = tl.sum(S * qv[None, :], axis=1)
            if V_FULL:
                tl.store(out_ptrs, o * scale)
            else:
                tl.store(out_ptrs, o * scale, mask=mask_v)

            pred = tl.sum(S * kv[None, :], axis=1)
            delta = beta_t * (vv - pred)
            S = S + delta[:, None] * kv[None, :]

            q_ptrs += sq_t
            k_ptrs += sk_t
            g_ptrs += sg_t
            v_ptrs += sv_t
            beta_ptr_cur += sb_t
            out_ptrs += V

    tl.store(
        cp_base + NCH * K * V + offs_k[:, None] * V + offs_v[None, :],
        tl.trans(S),
        mask=mask_kv,
    )


# ---------------------------------------------------------------------------
# Backward.
#
# No matrix multiplication and no chunk-inverse branch.  For each chunk,
# first recompute its decayed states S_t^d into a private global scratch
# slice.  Then walk the chunk backwards with the exact one-step adjoint.
#
# With D = dL/dS_t, W = D^T k_t, and
# B = D + scale*q_t*do_t^T - beta_t*k_t*W^T:
#   adjoint(S_t^d)   = B
#   adjoint(S_{t-1}) = exp(g_t) * B
#   dq_t    = scale * S_t^d @ do_t
#   dk_t    = D @ delta_t - beta_t * S_t^d @ W
#   dg_t    = sum_V (B * S_t^d)
#   dv_t    = beta_t * W
#   dbeta_t = (v_t - pred_t) . W
# All quantities below use the transposed [BV, BK] state slice.
# ---------------------------------------------------------------------------
@triton.jit
def _bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    do_ptr,
    cp_ptr,
    scr_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dg_ptr,
    dbeta_ptr,
    T,
    H,
    K,
    V,
    NVB,
    NCH,
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
    so_b,
    so_h,
    so_t,
    so_v,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNK: tl.constexpr,
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
    mask_kv = mask_k[:, None] & mask_v[None, :]
    mask_vk = mask_v[:, None] & mask_k[None, :]

    base_q = b_idx * sq_b + h_idx * sq_h
    base_k = b_idx * sk_b + h_idx * sk_h
    base_v = b_idx * sv_b + h_idx * sv_h
    base_g = b_idx * sg_b + h_idx * sg_h
    base_beta = b_idx * sb_b + h_idx * sb_h
    base_do = b_idx * so_b + h_idx * so_h

    cp_base = bh * (NCH + 1) * K * V
    scr_base = pid * CHUNK * BV * K
    grad_k_base = bh * T * K
    grad_v_base = bh * T * V

    # Adjoint of S at the current (reverse) position.
    D = tl.zeros([BV, BK], dtype=tl.float32)

    for chunk_rev in range(0, NCH):
        chunk = NCH - 1 - chunk_rev
        start = chunk * CHUNK

        chunk_len = T - start
        chunk_len = tl.where(chunk_len > CHUNK, CHUNK, chunk_len)

        # Recompute this chunk, storing every decayed state.
        S = tl.trans(
            tl.load(
                cp_ptr
                + cp_base
                + chunk * K * V
                + offs_k[:, None] * V
                + offs_v[None, :],
                mask=mask_kv,
                other=0.0,
            ).to(tl.float32)
        )

        for i in range(0, chunk_len):
            t = start + i
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
            vv = tl.load(
                v_ptr + base_v + t * sv_t + offs_v * sv_v,
                mask=mask_v,
                other=0.0,
            ).to(tl.float32)
            beta_t = tl.load(beta_ptr + base_beta + t * sb_t).to(tl.float32)

            S = S * tl.exp(gv)[None, :]
            tl.store(
                scr_ptr
                + scr_base
                + i * BV * K
                + offs_v[:, None] * K
                + offs_k[None, :],
                S,
                mask=mask_vk,
            )

            pred = tl.sum(S * kv[None, :], axis=1)
            delta = beta_t * (vv - pred)
            S = S + delta[:, None] * kv[None, :]

        # Exact reverse pass over the same chunk.
        for rev_i in range(0, chunk_len):
            i = chunk_len - 1 - rev_i
            t = start + i

            S = tl.load(
                scr_ptr
                + scr_base
                + i * BV * K
                + offs_v[:, None] * K
                + offs_k[None, :],
                mask=mask_vk,
                other=0.0,
            ).to(tl.float32)

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
            vv = tl.load(
                v_ptr + base_v + t * sv_t + offs_v * sv_v,
                mask=mask_v,
                other=0.0,
            ).to(tl.float32)
            dov = tl.load(
                do_ptr + base_do + t * so_t + offs_v * so_v,
                mask=mask_v,
                other=0.0,
            ).to(tl.float32)
            beta_t = tl.load(beta_ptr + base_beta + t * sb_t).to(tl.float32)

            w = tl.sum(D * kv[None, :], axis=1)
            pred = tl.sum(S * kv[None, :], axis=1)
            resid = vv - pred
            delta = beta_t * resid

            B = (
                D
                + scale * dov[:, None] * qv[None, :]
                - beta_t * w[:, None] * kv[None, :]
            )

            dq_part = scale * tl.sum(S * dov[:, None], axis=0)
            dk_part = tl.sum(D * delta[:, None], axis=0) - beta_t * tl.sum(
                S * w[:, None], axis=0
            )
            dg_part = tl.sum(B * S, axis=0)

            tl.store(
                dv_ptr + grad_v_base + t * V + offs_v,
                beta_t * w,
                mask=mask_v,
            )
            tl.atomic_add(dbeta_ptr + bh * T + t, tl.sum(w * resid))
            tl.atomic_add(
                dq_ptr + grad_k_base + t * K + offs_k,
                dq_part,
                mask=mask_k,
            )
            tl.atomic_add(
                dk_ptr + grad_k_base + t * K + offs_k,
                dk_part,
                mask=mask_k,
            )
            tl.atomic_add(
                dg_ptr + grad_k_base + t * K + offs_k,
                dg_part,
                mask=mask_k,
            )

            D = B * tl.exp(gv)[None, :]


def _next_pow2(x):
    n = 1
    while n < x:
        n *= 2
    return n


def _tile_sizes(K, V):
    BK = max(16, _next_pow2(K))
    if BK <= 64:
        BV = 64
    elif BK <= 128:
        BV = 32
    elif BK <= 256:
        BV = 16
    elif BK <= 512:
        BV = 8
    else:
        BV = 4
    return BK, BV, (V + BV - 1) // BV


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        ctx.shape_info = (B, H, T, K, V)
        ctx.scale_f = float(scale)

        chunk = 32
        ctx.chunk = chunk
        trivial = B == 0 or H == 0 or T == 0 or V == 0 or K == 0
        ctx.trivial = trivial

        out = torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)
        if trivial:
            ctx.nch = 0
            cp = torch.empty((0,), device=q.device, dtype=torch.float32)
            ctx.save_for_backward(q, k, v, e, g, beta, gamma, cp)
            return out

        nch = (T + chunk - 1) // chunk
        ctx.nch = nch

        cp = torch.empty(
            (B * H * (nch + 1) * K * V,), device=q.device, dtype=torch.float32
        )
        ctx.save_for_backward(q, k, v, e, g, beta, gamma, cp)

        BK, BV, nvb = _tile_sizes(K, V)
        _fwd_kernel[(B * H * nvb,)](
            q,
            k,
            v,
            g,
            beta,
            out,
            cp,
            T,
            H,
            K,
            V,
            nvb,
            nch,
            float(scale),
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
            BK=BK,
            BV=BV,
            CHUNK=chunk,
            K_FULL=(K == BK),
            V_FULL=(V % BV == 0),
            num_warps=4,
            num_stages=1,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma, cp = ctx.saved_tensors
        B, H, T, K, V = ctx.shape_info

        if ctx.trivial:
            return (
                torch.zeros_like(q),
                torch.zeros_like(k),
                torch.zeros_like(v),
                torch.zeros_like(e),
                torch.zeros_like(g),
                torch.zeros_like(beta),
                torch.zeros_like(gamma),
                None,
            )

        go = grad_out.contiguous()

        dq = torch.zeros((B * H * T * K,), device=q.device, dtype=torch.float32)
        dk = torch.zeros((B * H * T * K,), device=q.device, dtype=torch.float32)
        dg = torch.zeros((B * H * T * K,), device=q.device, dtype=torch.float32)
        dv = torch.empty((B * H * T * V,), device=q.device, dtype=torch.float32)
        dbeta = torch.zeros((B * H * T,), device=q.device, dtype=torch.float32)

        BK, BV, nvb = _tile_sizes(K, V)
        scratch = torch.empty(
            (B * H * nvb * ctx.chunk * BV * K,),
            device=q.device,
            dtype=torch.float32,
        )

        _bwd_kernel[(B * H * nvb,)](
            q,
            k,
            v,
            g,
            beta,
            go,
            cp,
            scratch,
            dq,
            dk,
            dv,
            dg,
            dbeta,
            T,
            H,
            K,
            V,
            nvb,
            ctx.nch,
            ctx.scale_f,
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
            go.stride(0),
            go.stride(1),
            go.stride(2),
            go.stride(3),
            BK=BK,
            BV=BV,
            CHUNK=ctx.chunk,
            num_warps=4,
            num_stages=1,
        )

        return (
            dq.view(B, H, T, K).to(q.dtype),
            dk.view(B, H, T, K).to(k.dtype),
            dv.view(B, H, T, V).to(v.dtype),
            torch.zeros_like(e),
            dg.view(B, H, T, K).to(g.dtype),
            dbeta.view(B, H, T).to(beta.dtype),
            torch.zeros_like(gamma),
            None,
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _GatedDelta.apply(q, k, v, e, g, beta, gamma, float(scale))
```

The failure was entirely in the Python autograd plumbing, not the kernels: the previous module delegated through a second `Function` whose backward called a nonexistent `_GatedDelta.backward_with_scale` method and contained a dead-code expression that would also have passed scale `0.0` to the backward kernel. This fix removes that delegation, uses a single `autograd.Function`, stores the scale on `ctx` in forward, and passes `ctx.scale_f` to the backward kernel. The previously introduced checkpointed forward and exact one-step-adjoint backward kernels (no `tl.dot`, no static nested replay, `num_stages=1`) are otherwise unchanged, so `e` and `gamma` still correctly receive zero gradients.