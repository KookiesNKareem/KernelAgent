```python
import torch
import triton
import triton.language as tl


@triton.jit
def _recurrent_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    T, scale, DK, DV, DE, DG,
    BK: tl.constexpr, BV: tl.constexpr, EXACT: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    mask_k = offs_k < DK
    mask_v = offs_v < DV
    mask_e = offs_v < DE

    q_p = q_ptr + bh * T * DK
    k_p = k_ptr + bh * T * DK
    v_p = v_ptr + bh * T * DV
    e_p = e_ptr + bh * T * DE
    g_p = g_ptr + bh * T * DG
    beta_p = beta_ptr + bh * T
    gamma_p = gamma_ptr + bh * T
    out_p = out_ptr + bh * T * DV

    S = tl.zeros([BK, BV], dtype=tl.float32)

    for _ in range(T):
        if EXACT:
            k_vec = tl.load(k_p + offs_k).to(tl.float32)
            v_vec = tl.load(v_p + offs_v).to(tl.float32)
            e_vec = tl.load(e_p + offs_v).to(tl.float32)
            g_vec = tl.load(g_p + offs_k).to(tl.float32)
        else:
            k_vec = tl.load(k_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            v_vec = tl.load(v_p + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            e_vec = tl.load(e_p + offs_v, mask=mask_e, other=0.0).to(tl.float32)
            g_vec = tl.load(g_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        beta_s = tl.load(beta_p)
        gamma_s = tl.load(gamma_p)

        S = S * tl.exp(g_vec)[:, None]
        u = tl.sum(S * e_vec[None, :], axis=1)
        S = S - gamma_s * u[:, None] * e_vec[None, :]
        pred = tl.sum(S * k_vec[:, None], axis=0)
        S = S + k_vec[:, None] * (beta_s * (v_vec - pred))[None, :]

        if EXACT:
            q_vec = tl.load(q_p + offs_k).to(tl.float32)
            tl.store(out_p + offs_v,
                     tl.sum(S * (q_vec * scale)[:, None], axis=0))
        else:
            q_vec = tl.load(q_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            tl.store(out_p + offs_v,
                     tl.sum(S * (q_vec * scale)[:, None], axis=0),
                     mask=mask_v)

        q_p += DK
        k_p += DK
        v_p += DV
        e_p += DE
        g_p += DG
        beta_p += 1
        gamma_p += 1
        out_p += DV


@triton.jit
def _checkpoint_kernel(
    k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, chk_ptr,
    T, DK, DV, DE, DG, NCHK,
    BK: tl.constexpr, BV: tl.constexpr, EXACT: tl.constexpr,
    CHUNK: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    mask_k = offs_k < DK
    mask_v = offs_v < DV
    mask_e = offs_v < DE
    offs_s = offs_k[:, None] * DV + offs_v[None, :]
    mask_s = mask_k[:, None] & mask_v[None, :]

    k_p = k_ptr + bh * T * DK
    v_p = v_ptr + bh * T * DV
    e_p = e_ptr + bh * T * DE
    g_p = g_ptr + bh * T * DG
    beta_p = beta_ptr + bh * T
    gamma_p = gamma_ptr + bh * T

    S = tl.zeros([BK, BV], dtype=tl.float32)

    for t in range(T):
        if t % CHUNK == 0:
            cp = chk_ptr + (bh * NCHK + (t // CHUNK)) * DK * DV
            if EXACT:
                tl.store(cp + offs_s, S)
            else:
                tl.store(cp + offs_s, S, mask=mask_s)

        if EXACT:
            k_vec = tl.load(k_p + offs_k).to(tl.float32)
            v_vec = tl.load(v_p + offs_v).to(tl.float32)
            e_vec = tl.load(e_p + offs_v).to(tl.float32)
            g_vec = tl.load(g_p + offs_k).to(tl.float32)
        else:
            k_vec = tl.load(k_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            v_vec = tl.load(v_p + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            e_vec = tl.load(e_p + offs_v, mask=mask_e, other=0.0).to(tl.float32)
            g_vec = tl.load(g_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        beta_s = tl.load(beta_p)
        gamma_s = tl.load(gamma_p)

        S = S * tl.exp(g_vec)[:, None]
        u = tl.sum(S * e_vec[None, :], axis=1)
        S = S - gamma_s * u[:, None] * e_vec[None, :]
        pred = tl.sum(S * k_vec[:, None], axis=0)
        S = S + k_vec[:, None] * (beta_s * (v_vec - pred))[None, :]

        k_p += DK
        v_p += DV
        e_p += DE
        g_p += DG
        beta_p += 1
        gamma_p += 1


@triton.jit
def _backward_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr,
    do_ptr, chk_ptr, scr_ptr,
    dq_ptr, dk_ptr, dv_ptr, de_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
    T, scale, DK, DV, DE, DG, NCHK,
    BK: tl.constexpr, BV: tl.constexpr, EXACT: tl.constexpr,
    CHUNK: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    mask_k = offs_k < DK
    mask_v = offs_v < DV
    mask_e = offs_v < DE
    offs_s = offs_k[:, None] * DV + offs_v[None, :]
    mask_s = mask_k[:, None] & mask_v[None, :]

    q_base = q_ptr + bh * T * DK
    k_base = k_ptr + bh * T * DK
    v_base = v_ptr + bh * T * DV
    e_base = e_ptr + bh * T * DE
    g_base = g_ptr + bh * T * DG
    beta_base = beta_ptr + bh * T
    gamma_base = gamma_ptr + bh * T
    do_base = do_ptr + bh * T * DV
    dq_base = dq_ptr + bh * T * DK
    dk_base = dk_ptr + bh * T * DK
    dv_base = dv_ptr + bh * T * DV
    de_base = de_ptr + bh * T * DE
    dg_base = dg_ptr + bh * T * DG
    dbeta_base = dbeta_ptr + bh * T
    dgamma_base = dgamma_ptr + bh * T
    scr_base = scr_ptr + bh * (CHUNK + 1) * DK * DV

    # Adjoint dL/dS_t, carried backwards across chunk boundaries.
    G = tl.zeros([BK, BV], dtype=tl.float32)

    for chunk_rev in range(NCHK):
        ci = (NCHK - 1) - chunk_rev
        start = ci * CHUNK
        chunk_len = tl.minimum(CHUNK, T - start)

        # Exact state at the start of this chunk, S_{start-1}.
        cp = chk_ptr + (bh * NCHK + ci) * DK * DV
        if EXACT:
            S = tl.load(cp + offs_s)
        else:
            S = tl.load(cp + offs_s, mask=mask_s, other=0.0)

        # Replay the chunk forward, materializing every state of the
        # chunk (slot i holds the state before step start+i, slot
        # chunk_len the chunk's final state).  The reverse pass below
        # reads these exact forward states; it never inverts the
        # recurrence.
        if EXACT:
            tl.store(scr_base + offs_s, S)
        else:
            tl.store(scr_base + offs_s, S, mask=mask_s)

        k_p = k_base + start * DK
        v_p = v_base + start * DV
        e_p = e_base + start * DE
        g_p = g_base + start * DG
        beta_p = beta_base + start
        gamma_p = gamma_base + start

        for lt in range(chunk_len):
            if EXACT:
                k_vec = tl.load(k_p + offs_k).to(tl.float32)
                v_vec = tl.load(v_p + offs_v).to(tl.float32)
                e_vec = tl.load(e_p + offs_v).to(tl.float32)
                g_vec = tl.load(g_p + offs_k).to(tl.float32)
            else:
                k_vec = tl.load(k_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)
                v_vec = tl.load(v_p + offs_v, mask=mask_v, other=0.0).to(tl.float32)
                e_vec = tl.load(e_p + offs_v, mask=mask_e, other=0.0).to(tl.float32)
                g_vec = tl.load(g_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)

            beta_s = tl.load(beta_p)
            gamma_s = tl.load(gamma_p)
            S = S * tl.exp(g_vec)[:, None]
            u = tl.sum(S * e_vec[None, :], axis=1)
            S = S - gamma_s * u[:, None] * e_vec[None, :]
            pred = tl.sum(S * k_vec[:, None], axis=0)
            S = S + k_vec[:, None] * (beta_s * (v_vec - pred))[None, :]

            if EXACT:
                tl.store(scr_base + (lt + 1) * DK * DV + offs_s, S)
            else:
                tl.store(scr_base + (lt + 1) * DK * DV + offs_s, S,
                         mask=mask_s)

            k_p += DK
            v_p += DV
            e_p += DE
            g_p += DG
            beta_p += 1
            gamma_p += 1

        # Point everything at the last timestep of the chunk.  Here S
        # is the chunk's final state, i.e. S_t for the first reverse
        # iteration; afterwards it is carried from S_prev.
        k_p -= DK
        v_p -= DV
        e_p -= DE
        g_p -= DG
        beta_p -= 1
        gamma_p -= 1
        last = start + chunk_len - 1
        q_p = q_base + last * DK
        do_p = do_base + last * DV
        dq_p = dq_base + last * DK
        dk_p = dk_base + last * DK
        dv_p = dv_base + last * DV
        de_p = de_base + last * DE
        dg_p = dg_base + last * DG
        dbeta_p = dbeta_base + last
        dgamma_p = dgamma_base + last

        for rev in range(chunk_len):
            lt = chunk_len - 1 - rev
            if EXACT:
                S_prev = tl.load(scr_base + lt * DK * DV + offs_s)
                q_vec = tl.load(q_p + offs_k).to(tl.float32)
                k_vec = tl.load(k_p + offs_k).to(tl.float32)
                v_vec = tl.load(v_p + offs_v).to(tl.float32)
                e_vec = tl.load(e_p + offs_v).to(tl.float32)
                g_vec = tl.load(g_p + offs_k).to(tl.float32)
                do_vec = tl.load(do_p + offs_v).to(tl.float32)
            else:
                S_prev = tl.load(scr_base + lt * DK * DV + offs_s,
                                 mask=mask_s, other=0.0)
                q_vec = tl.load(q_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)
                k_vec = tl.load(k_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)
                v_vec = tl.load(v_p + offs_v, mask=mask_v, other=0.0).to(tl.float32)
                e_vec = tl.load(e_p + offs_v, mask=mask_e, other=0.0).to(tl.float32)
                g_vec = tl.load(g_p + offs_k, mask=mask_k, other=0.0).to(tl.float32)
                do_vec = tl.load(do_p + offs_v, mask=mask_v, other=0.0).to(tl.float32)

            beta_s = tl.load(beta_p)
            gamma_s = tl.load(gamma_p)
            decay = tl.exp(g_vec)

            # dL/dq_t from the exact stored S_t.
            dq_vec = scale * tl.sum(S * do_vec[None, :], axis=1)
            if EXACT:
                tl.store(dq_p + offs_k, dq_vec)
            else:
                tl.store(dq_p + offs_k, dq_vec, mask=mask_k)

            # Forward intermediates of this step, recomputed directly
            # from the exact stored S_{t-1} (= tilde after decay).
            # No division by (1 - beta*||k||^2), (1 - gamma*||e||^2)
            # or exp(g) occurs anywhere.
            tmp = S_prev * decay[:, None]
            u = tl.sum(tmp * e_vec[None, :], axis=1)
            pred = tl.sum(tmp * k_vec[:, None], axis=0) \
                - gamma_s * tl.sum(u * k_vec) * e_vec
            resid = v_vec - pred

            # Add this timestep's output contribution to the adjoint.
            G = G + (scale * q_vec)[:, None] * do_vec[None, :]

            # Through S_t = (I - beta k k^T) hat + beta k v^T.
            a_vec = tl.sum(G * k_vec[:, None], axis=0)
            g_r = tl.sum(G * resid[None, :], axis=1)
            til_a = tl.sum(tmp * a_vec[None, :], axis=1)
            hat_a = til_a - gamma_s * u * tl.sum(e_vec * a_vec)
            dk_vec = beta_s * (g_r - hat_a)
            if EXACT:
                tl.store(dk_p + offs_k, dk_vec)
                tl.store(dv_p + offs_v, beta_s * a_vec)
            else:
                tl.store(dk_p + offs_k, dk_vec, mask=mask_k)
                tl.store(dv_p + offs_v, beta_s * a_vec, mask=mask_v)
            tl.store(dbeta_p, tl.sum(a_vec * resid))
            G = G - k_vec[:, None] * (beta_s * a_vec)[None, :]

            # Through hat = tilde (I - gamma e e^T).
            b_vec = tl.sum(G * e_vec[None, :], axis=1)
            de_vec = -gamma_s * (
                tl.sum(tmp * b_vec[:, None], axis=0)
                + tl.sum(G * u[:, None], axis=0))
            if EXACT:
                tl.store(de_p + offs_v, de_vec)
            else:
                tl.store(de_p + offs_v, de_vec, mask=mask_e)
            tl.store(dgamma_p, -tl.sum(u * b_vec))
            G = G - b_vec[:, None] * (gamma_s * e_vec)[None, :]

            # Through tilde = diag(exp(g)) S_{t-1}.
            dg_vec = tl.sum(G * tmp, axis=1)
            if EXACT:
                tl.store(dg_p + offs_k, dg_vec)
            else:
                tl.store(dg_p + offs_k, dg_vec, mask=mask_k)
            G = G * decay[:, None]

            # S_t of the next (earlier) reverse iteration.
            S = S_prev

            q_p -= DK
            k_p -= DK
            v_p -= DV
            e_p -= DE
            g_p -= DG
            beta_p -= 1
            gamma_p -= 1
            do_p -= DV
            dq_p -= DK
            dk_p -= DK
            dv_p -= DV
            de_p -= DE
            dg_p -= DG
            dbeta_p -= 1
            dgamma_p -= 1


def _pow2_at_least_16(n):
    x = 16
    while x < n:
        x *= 2
    return x


def _num_warps(elems):
    if elems >= 65536:
        return 32
    if elems >= 16384:
        return 16
    if elems >= 4096:
        return 8
    return 4


class _Layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        e = e.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        gamma = gamma.contiguous()

        B, H, T, DK = q.shape
        DV = v.shape[-1]
        DE = e.shape[-1]
        DG = g.shape[-1]
        out = torch.empty((B, H, T, DV), device=q.device, dtype=torch.float32)

        BH = B * H
        if BH != 0 and T != 0:
            BK = _pow2_at_least_16(DK)
            BV = _pow2_at_least_16(DV)
            exact = (BK == DK) and (BV == DV) and (BV == DE)
            _recurrent_kernel[(BH,)](
                q, k, v, e, g, beta, gamma, out,
                T, float(scale), DK, DV, DE, DG, BK, BV, exact,
                num_warps=_num_warps(BK * BV),
            )

        ctx.save_for_backward(q, k, v, e, g, beta, gamma)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma = ctx.saved_tensors
        B, H, T, DK = q.shape
        DV = v.shape[-1]
        DE = e.shape[-1]
        DG = g.shape[-1]
        BH = B * H

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        de = torch.empty_like(e)
        dg = torch.empty_like(g)
        dbeta = torch.empty_like(beta)
        dgamma = torch.empty_like(gamma)

        if BH == 0 or T == 0:
            for t in (dq, dk, dv, de, dg, dbeta, dgamma):
                t.zero_()
            return dq, dk, dv, de, dg, dbeta, dgamma, None

        do = grad_out.contiguous()
        if do.dtype != torch.float32:
            do = do.float()

        BK = _pow2_at_least_16(DK)
        BV = _pow2_at_least_16(DV)
        exact = (BK == DK) and (BV == DV) and (BV == DE)
        warps = _num_warps(BK * BV)

        chunk = 32
        while (T + chunk - 1) // chunk > 64:
            chunk *= 2
        nchk = (T + chunk - 1) // chunk
        chk = torch.empty((BH * nchk * DK * DV),
                          device=q.device, dtype=torch.float32)
        scr = torch.empty((BH * (chunk + 1) * DK * DV),
                          device=q.device, dtype=torch.float32)

        _checkpoint_kernel[(BH,)](
            k, v, e, g, beta, gamma, chk,
            T, DK, DV, DE, DG, nchk, BK, BV, exact, chunk,
            num_warps=warps,
        )
        _backward_kernel[(BH,)](
            q, k, v, e, g, beta, gamma, do, chk, scr,
            dq, dk, dv, de, dg, dbeta, dgamma,
            T, ctx.scale, DK, DV, DE, DG, nchk, BK, BV, exact, chunk,
            num_warps=32 if BK * BV >= 16384 else warps,
        )
        return dq, dk, dv, de, dg, dbeta, dgamma, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _Layer.apply(q, k, v, e, g, beta, gamma, scale)
```

The failing assumption was in the old backward's state recovery: it inverted each step, dividing by `1 - beta*||k||^2`, `1 - gamma*||e||^2`, and `exp(g)`, which is only safe for unit-norm `k`/`e`, gates in `(0,1)`, and mild decay. The transformed checker inputs violate all of these (non-unit or rescaled/shifted vectors, gates far outside `(0,1)`, decay down to `exp(-25)`), so each inversion amplified errors enormously, producing the observed `1e32` blow-ups and non-finite values exactly in the non-typical cases. The fix removes inversion entirely: the backward kernel replays each 32-step chunk forward and materializes its exact fp32 states in a per-head scratch buffer, then computes all gradients directly from the stored `S_{t-1}`/`S_t` pair, which is valid for arbitrary input magnitudes, signs, and gate/decay values. Forward, checkpointing, the (already general) adjoint formulas, and dtype/layout handling are otherwise unchanged.