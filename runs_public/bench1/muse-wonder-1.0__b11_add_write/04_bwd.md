```python
import torch
import triton
import triton.language as tl

_CB = 32


def _cfg(K, V):
    BLOCK_K = triton.next_power_of_2(K)
    v_pow2 = triton.next_power_of_2(V)
    BLOCK_V = 16
    while BLOCK_V < v_pow2 and BLOCK_K * BLOCK_V < 4096:
        BLOCK_V *= 2
    tile = BLOCK_K * BLOCK_V
    if tile >= 8192:
        warps = 8
    elif tile >= 4096:
        warps = 4
    elif tile >= 2048:
        warps = 2
    else:
        warps = 1
    return BLOCK_K, BLOCK_V, warps


@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    bnd_ptr,
    H, T, NUM_CHUNKS, BND_STRIDE_BH,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    seb, seh, set_, sek,
    sgb, sgh, sgt, sgk,
    sbb, sbh, sbt,
    scb, sch, sct,
    sob, soh, sot, sov,
    scale,
    KDIM, VDIM,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    CB: tl.constexpr, SAVE_BWD: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < KDIM
    mask_v = offs_v < VDIM

    q_ptrs = q_ptr + b * sqb + h * sqh + offs_k * sqk
    k_ptrs = k_ptr + b * skb + h * skh + offs_k * skk
    v_ptrs = v_ptr + b * svb + h * svh + offs_v * svv
    e_ptrs = e_ptr + b * seb + h * seh + offs_k * sek
    g_ptrs = g_ptr + b * sgb + h * sgh + offs_k * sgk
    out_ptrs = out_ptr + b * sob + h * soh + offs_v * sov

    beta_pos = beta_ptr + b * sbb + h * sbh
    gamma_pos = gamma_ptr + b * scb + h * sch

    bnd_ptrs = bnd_ptr + pid_bh.to(tl.int64) * BND_STRIDE_BH \
        + offs_k[:, None] * VDIM + offs_v[None, :]
    mask_kv = mask_k[:, None] & mask_v[None, :]

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in range(T):
        if SAVE_BWD:
            if t % CB == 0:
                tl.store(bnd_ptrs + (t // CB) * KDIM * VDIM, S, mask=mask_kv)

        kt = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        gt = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        vt = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
        et = tl.load(e_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        qt = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        bt = tl.load(beta_pos).to(tl.float32)
        ct = tl.load(gamma_pos).to(tl.float32)

        S = S * tl.exp(gt)[:, None]
        pred = tl.sum(S * kt[:, None], axis=0)
        S = S + kt[:, None] * (bt * (vt - pred))[None, :]
        S = S + et[:, None] * (ct * vt)[None, :]

        ot = tl.sum(S * (qt * scale)[:, None], axis=0)
        tl.store(out_ptrs, ot, mask=mask_v)

        q_ptrs += sqt
        k_ptrs += skt
        v_ptrs += svt
        e_ptrs += set_
        g_ptrs += sgt
        out_ptrs += sot
        beta_pos += sbt
        gamma_pos += sct

    if SAVE_BWD:
        tl.store(bnd_ptrs + NUM_CHUNKS * KDIM * VDIM, S, mask=mask_kv)


@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, go_ptr,
    bnd_ptr, scr_ptr,
    dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
    H, NUM_CHUNKS,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    seb, seh, set_, sek,
    sgb, sgh, sgt, sgk,
    sbb, sbh, sbt,
    scb, sch, sct,
    sob, soh, sot, sov,
    sgob, sgoh, sgot, sgov,
    sdqb, sdqh, sdqt, sdqk,
    sdvb, sdvh, sdvt, sdvv,
    sdbb, sdbh, sdbt,
    scale,
    T, KDIM, VDIM,
    BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < KDIM
    mask_v = offs_v < VDIM
    mask_kv = mask_k[:, None] & mask_v[None, :]

    q_b = q_ptr + b * sqb + h * sqh
    k_b = k_ptr + b * skb + h * skh
    v_b = v_ptr + b * svb + h * svh
    e_b = e_ptr + b * seb + h * seh
    g_b = g_ptr + b * sgb + h * sgh
    go_b = go_ptr + b * sgob + h * sgoh
    beta_b = beta_ptr + b * sbb + h * sbh
    gamma_b = gamma_ptr + b * scb + h * sch

    dq_b = dq_ptr + b * sdqb + h * sdqh
    dk_b = dk_ptr + b * sdqb + h * sdqh
    de_b = de_ptr + b * sdqb + h * sdqh
    dg_b = dg_ptr + b * sdqb + h * sdqh
    dv_b = dv_ptr + b * sdvb + h * sdvh
    dbeta_b = dbeta_ptr + b * sdbb + h * sdbh
    dgamma_b = dgamma_ptr + b * sdbb + h * sdbh

    KV = KDIM * VDIM
    bnd_b = bnd_ptr + pid_bh.to(tl.int64) * (NUM_CHUNKS + 1) * KV \
        + offs_k[:, None] * VDIM + offs_v[None, :]
    scr_b = scr_ptr + pid_bh.to(tl.int64) * BLOCK_C * KV \
        + offs_k[:, None] * VDIM + offs_v[None, :]

    # Adjoint state dL/dS_t for this value block.
    lam = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for chunk_rev in range(NUM_CHUNKS):
        chunk = NUM_CHUNKS - 1 - chunk_rev
        t_start = chunk * BLOCK_C

        # Phase A: recompute this chunk forward from its checkpoint,
        # stashing the decayed states Y_t = diag(exp g_t) S_{t-1}.
        S_cur = tl.load(bnd_b + chunk * KV, mask=mask_kv, other=0.0).to(tl.float32)
        for step in range(BLOCK_C):
            t = t_start + step
            valid = t < T
            kt = tl.load(k_b + t * skt + offs_k * skk,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            vt = tl.load(v_b + t * svt + offs_v * svv,
                         mask=mask_v & valid, other=0.0).to(tl.float32)
            et = tl.load(e_b + t * set_ + offs_k * sek,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            gt = tl.load(g_b + t * sgt + offs_k * sgk,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            bt = tl.load(beta_b + t * sbt, mask=valid, other=0.0).to(tl.float32)
            ct = tl.load(gamma_b + t * sct, mask=valid, other=0.0).to(tl.float32)

            Y = S_cur * tl.exp(gt)[:, None]
            tl.store(scr_b + step * KV, Y, mask=mask_kv)
            pred = tl.sum(Y * kt[:, None], axis=0)
            S_cur = Y + kt[:, None] * (bt * (vt - pred))[None, :] \
                + et[:, None] * (ct * vt)[None, :]

        tl.debug_barrier()

        # Phase B: walk the chunk backwards, applying the adjoint
        # recurrence and accumulating all seven input gradients.
        for rev_step in range(BLOCK_C):
            step = BLOCK_C - 1 - rev_step
            t = t_start + step
            valid = t < T

            qt = tl.load(q_b + t * sqt + offs_k * sqk,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            kt = tl.load(k_b + t * skt + offs_k * skk,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            vt = tl.load(v_b + t * svt + offs_v * svv,
                         mask=mask_v & valid, other=0.0).to(tl.float32)
            et = tl.load(e_b + t * set_ + offs_k * sek,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            gt = tl.load(g_b + t * sgt + offs_k * sgk,
                         mask=mask_k & valid, other=0.0).to(tl.float32)
            got = tl.load(go_b + t * sgot + offs_v * sgov,
                          mask=mask_v & valid, other=0.0).to(tl.float32)
            bt = tl.load(beta_b + t * sbt, mask=valid, other=0.0).to(tl.float32)
            ct = tl.load(gamma_b + t * sct, mask=valid, other=0.0).to(tl.float32)
            Y = tl.load(scr_b + step * KV, mask=mask_kv, other=0.0).to(tl.float32)

            # Add this step's output contribution A_t = scale q_t go_t^T.
            lam = lam + (qt * scale)[:, None] * got[None, :]

            lam_k = tl.sum(lam * kt[:, None], axis=0)     # k^T Lambda
            lam_e = tl.sum(lam * et[:, None], axis=0)     # e^T Lambda
            lam_v = tl.sum(lam * vt[None, :], axis=1)     # Lambda v

            pred = tl.sum(Y * kt[:, None], axis=0)
            delta = vt - pred
            lam_delta = tl.sum(lam * delta[None, :], axis=1)  # Lambda delta
            y_lamk = tl.sum(Y * lam_k[None, :], axis=1)       # Y (k^T Lambda)^T

            gk_vec = bt * (lam_delta - y_lamk)
            grad_y = lam - (bt * kt)[:, None] * lam_k[None, :]
            gg_vec = tl.sum(grad_y * Y, axis=1)
            ge_vec = ct * lam_v
            gv_vec = bt * lam_k + ct * lam_e
            gbeta_val = tl.sum(lam_k * delta)
            ggamma_val = tl.sum(lam_v * et)

            s_go = tl.sum(Y * got[None, :], axis=1) \
                + bt * kt * tl.sum(delta * got) \
                + ct * et * tl.sum(vt * got)
            gq_vec = scale * s_go

            # Propagate the adjoint to S_{t-1}.
            lam = tl.exp(gt)[:, None] * grad_y

            tl.store(dv_b + t * sdvt + offs_v * sdvv, gv_vec,
                     mask=mask_v & valid)
            tl.atomic_add(dq_b + t * sdqt + offs_k * sdqk, gq_vec,
                          mask=mask_k & valid)
            tl.atomic_add(dk_b + t * sdqt + offs_k * sdqk, gk_vec,
                          mask=mask_k & valid)
            tl.atomic_add(de_b + t * sdqt + offs_k * sdqk, ge_vec,
                          mask=mask_k & valid)
            tl.atomic_add(dg_b + t * sdqt + offs_k * sdqk, gg_vec,
                          mask=mask_k & valid)
            tl.atomic_add(dbeta_b + t * sdbt, gbeta_val, mask=valid)
            tl.atomic_add(dgamma_b + t * sdbt, ggamma_val, mask=valid)

        tl.debug_barrier()


def _forward_launch(q, k, v, e, g, beta, gamma, scale, want_bwd):
    B, H, T, K = q.shape
    V = v.shape[-1]
    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    BLOCK_K, BLOCK_V, warps = _cfg(K, V)
    num_chunks = (T + _CB - 1) // _CB
    if want_bwd:
        bnd = torch.empty((B, H, num_chunks + 1, K, V),
                          device=q.device, dtype=torch.float32)
    else:
        bnd = torch.empty(0, device=q.device, dtype=torch.float32)

    grid = (B * H, (V + BLOCK_V - 1) // BLOCK_V)
    _fwd_kernel[grid](
        q, k, v, e, g, beta, gamma, out, bnd,
        H, T, num_chunks, (num_chunks + 1) * K * V,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        e.stride(0), e.stride(1), e.stride(2), e.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        float(scale), K, V,
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, CB=_CB, SAVE_BWD=want_bwd,
        num_warps=warps, num_stages=2,
    )
    return out, bnd


class _Layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        ctx.scale = float(scale)
        ctx.empty_case = (B == 0 or H == 0 or T == 0 or K == 0 or V == 0)
        if ctx.empty_case:
            ctx.save_for_backward(q, k, v, e, g, beta, gamma)
            return torch.zeros((B, H, T, V), device=q.device,
                               dtype=torch.float32)

        want_bwd = any(x.requires_grad for x in (q, k, v, e, g, beta, gamma))
        ctx.need_bwd = want_bwd
        out, bnd = _forward_launch(q, k, v, e, g, beta, gamma,
                                   scale, want_bwd)
        ctx.save_for_backward(q, k, v, e, g, beta, gamma, bnd)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.empty_case:
            q, k, v, e, g, beta, gamma = ctx.saved_tensors
            return (torch.zeros(q.shape, dtype=q.dtype, device=q.device),
                    torch.zeros(k.shape, dtype=k.dtype, device=k.device),
                    torch.zeros(v.shape, dtype=v.dtype, device=v.device),
                    torch.zeros(e.shape, dtype=e.dtype, device=e.device),
                    torch.zeros(g.shape, dtype=g.dtype, device=g.device),
                    torch.zeros(beta.shape, dtype=beta.dtype,
                                device=beta.device),
                    torch.zeros(gamma.shape, dtype=gamma.dtype,
                                device=gamma.device),
                    None)

        q, k, v, e, g, beta, gamma, bnd = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        dev = q.device
        fp = torch.float32

        if not ctx.need_bwd:
            return (torch.zeros(q.shape, dtype=q.dtype, device=dev),
                    torch.zeros(k.shape, dtype=k.dtype, device=dev),
                    torch.zeros(v.shape, dtype=v.dtype, device=dev),
                    torch.zeros(e.shape, dtype=e.dtype, device=dev),
                    torch.zeros(g.shape, dtype=g.dtype, device=dev),
                    torch.zeros(beta.shape, dtype=beta.dtype, device=dev),
                    torch.zeros(gamma.shape, dtype=gamma.dtype, device=dev),
                    None)

        dq = torch.zeros((B, H, T, K), device=dev, dtype=fp)
        dk = torch.zeros((B, H, T, K), device=dev, dtype=fp)
        de = torch.zeros((B, H, T, K), device=dev, dtype=fp)
        dg = torch.zeros((B, H, T, K), device=dev, dtype=fp)
        dv = torch.zeros((B, H, T, V), device=dev, dtype=fp)
        dbeta = torch.zeros((B, H, T), device=dev, dtype=fp)
        dgamma = torch.zeros((B, H, T), device=dev, dtype=fp)
        scratch = torch.empty(B * H * _CB * K * V, device=dev, dtype=fp)

        BLOCK_K, BLOCK_V, warps = _cfg(K, V)
        num_chunks = (T + _CB - 1) // _CB
        grid = (B * H, (V + BLOCK_V - 1) // BLOCK_V)

        _bwd_kernel[grid](
            q, k, v, e, g, beta, gamma, grad_out, bnd, scratch,
            dq, dk, dv, de, dg, dbeta, dgamma,
            H, num_chunks,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            e.stride(0), e.stride(1), e.stride(2), e.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            0, 0, 0, 0,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            grad_out.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            dbeta.stride(0), dbeta.stride(1), dbeta.stride(2),
            ctx.scale, T, K, V,
            BLOCK_C=_CB, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
            num_warps=warps, num_stages=1,
        )

        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                de.to(e.dtype), dg.to(g.dtype), dbeta.to(beta.dtype),
                dgamma.to(gamma.dtype), None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if torch.is_grad_enabled() and any(
            x.requires_grad for x in (q, k, v, e, g, beta, gamma)):
        return _Layer.apply(q, k, v, e, g, beta, gamma, float(scale))

    B, H, T, K = q.shape
    V = v.shape[-1]
    if B == 0 or H == 0 or T == 0 or V == 0 or K == 0:
        return torch.zeros((B, H, T, V), device=q.device,
                           dtype=torch.float32)
    out, _ = _forward_launch(q, k, v, e, g, beta, gamma, scale, False)
    return out
```

The forward is the previously passing per-step recurrent kernel, augmented only to checkpoint the state every 32 steps when gradients are needed. The backward runs the exact adjoint recurrence — $\Lambda_{t-1} = \mathrm{diag}(e^{g_t})(I-\beta_t k_t k_t^\top)\Lambda_t$ after adding $\text{scale}\cdot q_t\,{\bar o}_t^\top$ — with each chunk’s decayed states $Y_t$ recomputed once from its checkpoint into a small reused scratch buffer, so no full $T \times K \times V$ state history is stored. All seven gradients are evaluated in closed form from $Y_t$, $\Lambda_t$, and $\delta_t = v_t - Y_t^\top k_t$, accumulated in fp32 (atomically across value blocks where a gradient is shared) and cast to each input’s dtype. I could not test on hardware, so Triton details around `tl.debug_barrier()` (used to make the scratch reuse safe within a program) and scalar masked atomic adds are the main items I am unsure about.