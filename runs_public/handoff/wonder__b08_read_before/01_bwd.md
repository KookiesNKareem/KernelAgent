```python
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, out_ptr, chk_ptr,
    T, K, V, BH,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, CHUNK: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V
    state_mask = k_mask[:, None] & v_mask[None, :]

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    d_base = d_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    o_base = out_ptr + bh * T * V
    b_base = beta_ptr + bh * T

    for t in range(T):
        if t % CHUNK == 0:
            chunk = t // CHUNK
            chk_off = (chunk * BH + bh) * K * V + k_offs[:, None] * V + v_offs[None, :]
            tl.store(chk_ptr + chk_off, S, mask=state_mask)

        q_t = tl.load(q_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        k_t = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        dec = tl.load(d_base + t * K + k_offs, mask=k_mask, other=1.0).to(tl.float32)
        v_t = tl.load(v_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
        b_t = tl.load(b_base + t).to(tl.float32)

        S = S * dec[:, None]

        # Output is read before the delta-rule write.
        out_t = tl.sum(S * (q_t * scale)[:, None], axis=0)
        tl.store(o_base + t * V + v_offs, out_t, mask=v_mask)

        pred = tl.sum(S * k_t[:, None], axis=0)
        r = v_t - pred
        S = S + k_t[:, None] * (b_t * r)[None, :]


@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, do_ptr,
    chk_ptr, scr_ptr,
    dq_ptr, dk_ptr, dv_ptr, dg_ptr, dbeta_ptr,
    T, K, V, BH,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    CHUNK: tl.constexpr, NCHUNK: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V
    state_mask = k_mask[:, None] & v_mask[None, :]

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    d_base = d_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    do_base = do_ptr + bh * T * V
    b_base = beta_ptr + bh * T
    dq_base = dq_ptr + bh * T * K
    dk_base = dk_ptr + bh * T * K
    dv_base = dv_ptr + bh * T * V
    dg_base = dg_ptr + bh * T * K
    db_base = dbeta_ptr + bh * T

    # G = dL / dS_t for the currently processed (reverse) time step.
    G = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for rev_chunk in range(NCHUNK):
        chunk = NCHUNK - 1 - rev_chunk
        c0 = chunk * CHUNK
        c_end = c0 + CHUNK
        if c_end > T:
            c_end = T
        chunk_len = c_end - c0

        # Recompute and stash the decayed states S~ for this chunk,
        # starting from the checkpoint taken before the chunk.
        chk_off = (chunk * BH + bh) * K * V + k_offs[:, None] * V + v_offs[None, :]
        S = tl.load(chk_ptr + chk_off, mask=state_mask, other=0.0).to(tl.float32)
        for local in range(chunk_len):
            t = c0 + local
            k_t = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            dec = tl.load(d_base + t * K + k_offs, mask=k_mask, other=1.0).to(tl.float32)
            v_t = tl.load(v_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
            b_t = tl.load(b_base + t).to(tl.float32)
            S_tilde = S * dec[:, None]
            scr_off = (bh * CHUNK + local) * K * V + k_offs[:, None] * V + v_offs[None, :]
            tl.store(scr_ptr + scr_off, S_tilde, mask=state_mask)
            pred = tl.sum(S_tilde * k_t[:, None], axis=0)
            r = v_t - pred
            S = S_tilde + k_t[:, None] * (b_t * r)[None, :]

        # Reverse sweep through the chunk using the stashed S~.
        for rev in range(chunk_len):
            t = c_end - 1 - rev
            local = t - c0
            q_t = tl.load(q_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            k_t = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            dec = tl.load(d_base + t * K + k_offs, mask=k_mask, other=1.0).to(tl.float32)
            v_t = tl.load(v_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
            b_t = tl.load(b_base + t).to(tl.float32)
            do_t = tl.load(do_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
            scr_off = (bh * CHUNK + local) * K * V + k_offs[:, None] * V + v_offs[None, :]
            S_tilde = tl.load(scr_ptr + scr_off, mask=state_mask, other=0.0).to(tl.float32)

            pred = tl.sum(S_tilde * k_t[:, None], axis=0)
            r = v_t - pred
            # wG[j] = sum_i k[i] * G[i, j]
            wG = tl.sum(G * k_t[:, None], axis=0)

            # Gradient w.r.t. the decayed state S~_t:
            #   from S_t = (I - beta k k^T) S~_t + beta k v^T
            #   plus from o_t = scale * S~_t^T q_t.
            G_tilde = G - (b_t * k_t)[:, None] * wG[None, :] \
                + (scale * q_t)[:, None] * do_t[None, :]

            dq_part = scale * tl.sum(S_tilde * do_t[None, :], axis=1)
            dk_part = b_t * (tl.sum(G * r[None, :], axis=1)
                             - tl.sum(S_tilde * wG[None, :], axis=1))
            dg_part = tl.sum(G_tilde * S_tilde, axis=1)
            tl.atomic_add(dq_base + t * K + k_offs, dq_part, mask=k_mask)
            tl.atomic_add(dk_base + t * K + k_offs, dk_part, mask=k_mask)
            tl.atomic_add(dg_base + t * K + k_offs, dg_part, mask=k_mask)

            # dv_t = beta * (k^T G); dbeta_t = (k^T G) . (v_t - pred_t)
            tl.store(dv_base + t * V + v_offs, b_t * wG, mask=v_mask)
            tl.atomic_add(db_base + t, tl.sum(wG * r))

            # Propagate to S_{t-1}:  S~_t = diag(dec) S_{t-1}.
            G = G_tilde * dec[:, None]


def _next_pow2(x):
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


def _block_sizes(K, V):
    block_k = _next_pow2(K)
    if block_k < 16:
        block_k = 16
    if V <= 32:
        block_v = _next_pow2(V)
        if block_v < 16:
            block_v = 16
    else:
        block_v = 32
    return block_k, block_v


def _chunk_len(BH, T, K, V):
    per_state = max(1, BH * K * V)
    budget = 1 << 24  # ~64 MiB of fp32 scratch states
    c = budget // per_state
    if c > 64:
        c = 64
    if c < 1:
        c = 1
    return int(c)


class _LayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
        ctx.g_dtype = g.dtype

        qc = q.contiguous()
        kc = k.contiguous()
        vc = v.contiguous()
        bc = beta.contiguous()
        gc = g.contiguous()
        dc = torch.exp(gc)
        if not dc.is_contiguous():
            dc = dc.contiguous()

        qc3 = qc.reshape(BH, T, K)
        kc3 = kc.reshape(BH, T, K)
        dc3 = dc.reshape(BH, T, K)
        vc3 = vc.reshape(BH, T, V)
        bc2 = bc.reshape(BH, T)

        out3 = torch.empty(BH, T, V, device=q.device, dtype=torch.float32)
        chunk = _chunk_len(BH, T, K, V)
        n_chunks = (T + chunk - 1) // chunk if T > 0 else 0
        chk = torch.empty(n_chunks * BH * K * V, device=q.device, dtype=torch.float32)
        ctx.chunk = chunk
        ctx.n_chunks = n_chunks

        needs = BH > 0 and T > 0 and K > 0 and V > 0
        ctx.needs_kernel = needs
        if needs:
            block_k, block_v = _block_sizes(K, V)
            num_warps = 8 if block_k * block_v > 2048 else 4
            grid = (BH, (V + block_v - 1) // block_v)
            _fwd_kernel[grid](
                qc3, kc3, vc3, dc3, bc2, out3, chk,
                T, K, V, BH, float(scale),
                BLOCK_K=block_k, BLOCK_V=block_v, CHUNK=chunk,
                num_warps=num_warps, num_stages=2,
            )

        ctx.save_for_backward(qc3, kc3, vc3, dc3, bc2, chk, e, gamma)
        return out3.reshape(B, H, T, V)

    @staticmethod
    def backward(ctx, grad_output):
        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        BH = B * H
        qc3, kc3, vc3, dc3, bc2, chk, e, gamma = ctx.saved_tensors

        grad_e = torch.zeros_like(e)
        grad_gamma = torch.zeros_like(gamma)

        if (not ctx.needs_kernel) or grad_output is None:
            return (
                torch.zeros_like(qc3).reshape(B, H, T, K),
                torch.zeros_like(kc3).reshape(B, H, T, K),
                torch.zeros_like(vc3).reshape(B, H, T, V),
                grad_e,
                torch.zeros(B, H, T, K, device=qc3.device, dtype=ctx.g_dtype),
                torch.zeros_like(bc2).reshape(B, H, T),
                grad_gamma,
                None,
            )

        device = qc3.device
        go3 = grad_output.reshape(BH, T, V).contiguous().to(torch.float32)

        dq_acc = torch.zeros(BH, T, K, device=device, dtype=torch.float32)
        dk_acc = torch.zeros(BH, T, K, device=device, dtype=torch.float32)
        dg_acc = torch.zeros(BH, T, K, device=device, dtype=torch.float32)
        dv_acc = torch.empty(BH, T, V, device=device, dtype=torch.float32)
        db_acc = torch.zeros(BH, T, device=device, dtype=torch.float32)
        scratch = torch.empty(BH * ctx.chunk * K * V, device=device, dtype=torch.float32)

        block_k, block_v = _block_sizes(K, V)
        num_warps = 8 if block_k * block_v > 2048 else 4
        grid = (BH, (V + block_v - 1) // block_v)
        _bwd_kernel[grid](
            qc3, kc3, vc3, dc3, bc2, go3,
            chk, scratch,
            dq_acc, dk_acc, dv_acc, dg_acc, db_acc,
            T, K, V, BH, ctx.scale if hasattr(ctx, "scale") else _scale_holder[0],
            BLOCK_K=block_k, BLOCK_V=block_v,
            CHUNK=ctx.chunk, NCHUNK=ctx.n_chunks,
            num_warps=num_warps, num_stages=2,
        )

        return (
            dq_acc.reshape(B, H, T, K).to(qc3.dtype),
            dk_acc.reshape(B, H, T, K).to(kc3.dtype),
            dv_acc.reshape(B, H, T, V).to(vc3.dtype),
            grad_e,
            dg_acc.reshape(B, H, T, K).to(ctx.g_dtype),
            db_acc.reshape(B, H, T).to(bc2.dtype),
            grad_gamma,
            None,
        )


# The scale is a plain float for autograd purposes; it is threaded through
# apply() as the last argument, so forward receives it directly.  The holder
# below is only a fallback and is set in kernel_function before apply().
_scale_holder = [1.0]


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)
    _scale_holder[0] = scale_f
    return _LayerFn.apply(q, k, v, e, g, beta, gamma, scale_f)
```

The forward reuses the blocked state-in-registers recurrence and also stores the `[K,V]` state at each chunk boundary (chunk length chosen so the per-chunk scratch stays near 64 MiB). The backward walks chunks in reverse: for each chunk it first recomputes the decayed states `S~_t = diag(exp g_t) S_{t-1}` from the chunk checkpoint into a small global scratch buffer, then sweeps the chunk backward carrying `G = dL/dS_t` in registers, using `G~ = (I - beta k k^T) G + scale * q (d o)^T`, `dq = scale * S~ do`, `dv = beta * (k^T G)`, `dbeta = (k^T G)·(v - S~^T k)`, `dk = beta (G r - S~ (k^T G))`, `dg = rowsum(G~ * S~)`, and `G <- diag(exp g_t) G~`. Cross-`V`-block contributions to `dq`, `dk`, `dg`, and `dbeta` are combined with fp32 atomic adds and cast to each input's dtype at the end, while `e` and `gamma` never enter the recurrence and correctly receive zero gradients. One uncertainty is the fallback handling of the scalar `scale` inside the autograd function—it is passed through `apply` as a float, so the kernel always sees the exact forward value.