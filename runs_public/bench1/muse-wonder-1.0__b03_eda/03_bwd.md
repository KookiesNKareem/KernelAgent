```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel (with the same semantics as the reference loop).
# ---------------------------------------------------------------------------
@triton.jit
def _recurrent_delta_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    out_ptr,
    T,
    K,
    V,
    scale,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid_v = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V

    bh = pid_bh.to(tl.int64)
    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    o_base = out_ptr + bh * T * V
    s_base = g_ptr + bh * T

    S = tl.zeros((BK, BV), dtype=tl.float32)

    for t in range(T):
        tk = t * K
        tv = t * V

        kt = tl.load(k_base + tk + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        et = tl.load(e_base + tk + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        qt = tl.load(q_base + tk + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        vt = tl.load(v_base + tv + offs_v, mask=mask_v, other=0.0).to(tl.float32)

        decay = tl.exp(tl.load(s_base + t))
        beta_t = tl.load(beta_ptr + bh * T + t)
        gamma_t = tl.load(gamma_ptr + bh * T + t)

        S = S * decay
        erased = tl.sum(S * et[:, None], axis=0)
        S = S - et[:, None] * (gamma_t * erased)[None, :]
        pred = tl.sum(S * kt[:, None], axis=0)
        S = S + kt[:, None] * (beta_t * (vt - pred))[None, :]

        ot = tl.sum(S * (qt * scale)[:, None], axis=0)
        tl.store(o_base + tv + offs_v, ot, mask=mask_v)


# ---------------------------------------------------------------------------
# Backward kernel.
#
# Adjoint recurrence (per step, with DS = dL/dS_t including the output term):
#   w = DS^T k,  u = v - A2^T k
#   dv = beta*w,  dbeta = w.u,  dk = beta*(DS u - A2 w)
#   dA2 = DS - beta*k w^T
#   s = dA2^T e,  r = A1^T e
#   dgamma = -s.r,  de = -gamma*(dA2 r + A1 s)
#   dA1 = dA2 - gamma*e s^T
#   dg = <dA1, A1>,  dS_{t-1} = exp(g) * dA1,  dq = scale * S_t dO_t
#
# The kernel first recomputes and stores chunk-start states, then processes
# chunks backwards.  Within a chunk it replays the forward step, storing each
# S_t in a small reusable scratch buffer, and then walks the chunk backwards.
# K-side gradients are accumulated per value-block and reduced in the wrapper.
# ---------------------------------------------------------------------------
@triton.jit
def _backward_delta_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    go_ptr,
    chk_ptr,
    scratch_ptr,
    dq_ptr,
    dk_ptr,
    de_ptr,
    dv_ptr,
    dg_ptr,
    dbeta_ptr,
    dgamma_ptr,
    T,
    K,
    V,
    BH,
    NV,
    scale,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
):
    pid_v = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    offs_k = tl.arange(0, BK)
    offs_vl = tl.arange(0, BV)
    offs_v = pid_v * BV + offs_vl
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_kv = mask_k[:, None] & mask_v[None, :]

    bh = pid_bh.to(tl.int64)
    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    go_base = go_ptr + bh * T * V
    s_base = g_ptr + bh * T
    b_base = beta_ptr + bh * T
    c_base = gamma_ptr + bh * T

    num_chunks = tl.cdiv(T, BC)
    chk_base = chk_ptr + bh * num_chunks * K * V
    scratch_base = scratch_ptr + ((bh * NV + pid_v) * BC) * BK * BV
    part_base = pid_v.to(tl.int64) * BH + bh

    # ---- Phase 1: store the state at the start of every chunk. ----
    S = tl.zeros((BK, BV), dtype=tl.float32)
    for ci in range(num_chunks):
        tl.store(
            chk_base + ci * K * V + offs_k[:, None] * V + offs_v[None, :],
            S,
            mask=mask_kv,
        )
        c0 = ci * BC
        for i in range(BC):
            t = c0 + i
            valid = t < T
            kt = tl.load(k_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            et = tl.load(e_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            vt = tl.load(v_base + t * V + offs_v, mask=valid & mask_v, other=0.0).to(tl.float32)
            decay = tl.exp(tl.load(s_base + t, mask=valid, other=0.0))
            beta_t = tl.load(b_base + t, mask=valid, other=0.0)
            gamma_t = tl.load(c_base + t, mask=valid, other=0.0)

            S = S * decay
            erased = tl.sum(S * et[:, None], axis=0)
            S = S - et[:, None] * (gamma_t * erased)[None, :]
            pred = tl.sum(S * kt[:, None], axis=0)
            S = S + kt[:, None] * (beta_t * (vt - pred))[None, :]

    # ---- Phase 2: reverse over chunks. ----
    DS = tl.zeros((BK, BV), dtype=tl.float32)
    for rev_ci in range(num_chunks):
        ci = num_chunks - 1 - rev_ci
        c0 = ci * BC

        S_chk = tl.load(
            chk_base + ci * K * V + offs_k[:, None] * V + offs_v[None, :],
            mask=mask_kv,
            other=0.0,
        )

        # Replay this chunk, storing each post-step state in scratch.
        S_cur = S_chk
        for i in range(BC):
            t = c0 + i
            valid = t < T
            kt = tl.load(k_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            et = tl.load(e_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            vt = tl.load(v_base + t * V + offs_v, mask=valid & mask_v, other=0.0).to(tl.float32)
            decay = tl.exp(tl.load(s_base + t, mask=valid, other=0.0))
            beta_t = tl.load(b_base + t, mask=valid, other=0.0)
            gamma_t = tl.load(c_base + t, mask=valid, other=0.0)

            S_cur = S_cur * decay
            erased = tl.sum(S_cur * et[:, None], axis=0)
            S_cur = S_cur - et[:, None] * (gamma_t * erased)[None, :]
            pred = tl.sum(S_cur * kt[:, None], axis=0)
            S_cur = S_cur + kt[:, None] * (beta_t * (vt - pred))[None, :]

            tl.store(
                scratch_base + i * BK * BV + offs_k[:, None] * BV + offs_vl[None, :],
                S_cur,
                mask=mask_kv,
            )

        # Walk the chunk backwards, applying the adjoint step.
        for rev_i in range(BC):
            i = BC - 1 - rev_i
            t = c0 + i
            valid = t < T

            S_t = tl.load(
                scratch_base + i * BK * BV + offs_k[:, None] * BV + offs_vl[None, :],
                mask=mask_kv,
                other=0.0,
            )
            if i == 0:
                S_prev = S_chk
            else:
                S_prev = tl.load(
                    scratch_base + (i - 1) * BK * BV + offs_k[:, None] * BV + offs_vl[None, :],
                    mask=mask_kv,
                    other=0.0,
                )

            kt = tl.load(k_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            et = tl.load(e_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            qt = tl.load(q_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            vt = tl.load(v_base + t * V + offs_v, mask=valid & mask_v, other=0.0).to(tl.float32)
            do_t = tl.load(go_base + t * V + offs_v, mask=valid & mask_v, other=0.0).to(tl.float32)
            g_t = tl.load(s_base + t, mask=valid, other=0.0)
            beta_t = tl.load(b_base + t, mask=valid, other=0.0)
            gamma_t = tl.load(c_base + t, mask=valid, other=0.0)
            decay = tl.exp(g_t)

            # Rebuild this step's intermediate states from S_prev.
            A1 = decay * S_prev
            r = tl.sum(A1 * et[:, None], axis=0)
            A2 = A1 - et[:, None] * (gamma_t * r)[None, :]
            pred = tl.sum(A2 * kt[:, None], axis=0)
            u = vt - pred

            # Output contribution.
            DS_tot = DS + (qt * scale)[:, None] * do_t[None, :]
            dq_part = scale * tl.sum(S_t * do_t[None, :], axis=1)
            tl.store(
                dq_ptr + part_base * T * K + t * K + offs_k,
                dq_part,
                mask=valid & mask_k,
            )

            # Delta-write adjoint.
            w = tl.sum(DS_tot * kt[:, None], axis=0)
            tl.store(
                dv_ptr + bh * T * V + t * V + offs_v,
                beta_t * w,
                mask=valid & mask_v,
            )
            tl.store(
                dbeta_ptr + part_base * T + t,
                tl.sum(w * u),
                mask=valid,
            )
            dk_part = beta_t * (
                tl.sum(DS_tot * u[None, :], axis=1)
                - tl.sum(A2 * w[None, :], axis=1)
            )
            tl.store(
                dk_ptr + part_base * T * K + t * K + offs_k,
                dk_part,
                mask=valid & mask_k,
            )
            DA2 = DS_tot - kt[:, None] * w[None, :] * beta_t

            # Erase adjoint.
            s_vec = tl.sum(DA2 * et[:, None], axis=0)
            tl.store(
                dgamma_ptr + part_base * T + t,
                -tl.sum(s_vec * r),
                mask=valid,
            )
            de_part = -gamma_t * (
                tl.sum(DA2 * r[None, :], axis=1)
                + tl.sum(A1 * s_vec[None, :], axis=1)
            )
            tl.store(
                de_ptr + part_base * T * K + t * K + offs_k,
                de_part,
                mask=valid & mask_k,
            )
            DA1 = DA2 - et[:, None] * (gamma_t * s_vec)[None, :]

            # Decay adjoint.
            tl.store(
                dg_ptr + part_base * T + t,
                tl.sum(DA1 * A1),
                mask=valid,
            )
            DS = decay * DA1


def _next_power_of_two(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _launch_forward(q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f, out, T, K, V, scale):
    BH = q_f.shape[0]
    BK = max(16, _next_power_of_two(K))
    BV = 16 if BK >= 256 else 32
    num_v_blocks = triton.cdiv(V, BV)
    num_warps = 8 if BK * BV > 4096 else 4
    _recurrent_delta_kernel[(num_v_blocks, BH)](
        q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f, out,
        T, K, V, float(scale),
        BK=BK, BV=BV,
        num_warps=num_warps,
        num_stages=2,
    )


class _DeltaEraseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        ctx.input_metas = [
            (tuple(x.shape), x.dtype, x.device) for x in (q, k, v, e, g, beta, gamma)
        ]
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
        ctx.scale = float(scale)
        ctx.empty_case = (B == 0 or H == 0 or T == 0 or V == 0 or K == 0)

        if ctx.empty_case:
            ctx.save_for_backward()
            if K == 0 and B > 0 and H > 0 and T > 0 and V > 0:
                return torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)
            return torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

        input_device = q.device
        ctx.input_device = input_device
        work_device = input_device if input_device.type == "cuda" else torch.device("cuda")
        ctx.work_device = work_device

        q_c = q.to(work_device).contiguous()
        k_c = k.to(work_device).contiguous()
        v_c = v.to(work_device).contiguous()
        e_c = e.to(work_device).contiguous()
        g_c = g.to(work_device).contiguous()
        beta_c = beta.to(work_device).contiguous()
        gamma_c = gamma.to(work_device).contiguous()

        BH = B * H
        q_f = q_c.reshape(BH, T, K)
        k_f = k_c.reshape(BH, T, K)
        v_f = v_c.reshape(BH, T, V)
        e_f = e_c.reshape(BH, T, K)
        g_f = g_c.reshape(BH, T)
        beta_f = beta_c.reshape(BH, T)
        gamma_f = gamma_c.reshape(BH, T)

        out = torch.empty((BH, T, V), device=work_device, dtype=torch.float32)
        _launch_forward(q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f, out, T, K, V, ctx.scale)

        ctx.save_for_backward(q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f)
        out = out.reshape(B, H, T, V)
        if out.device != input_device:
            out = out.to(input_device)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.empty_case:
            grads = [
                torch.zeros(shape, dtype=dtype, device=device)
                for (shape, dtype, device) in ctx.input_metas
            ]
            return grads[0], grads[1], grads[2], grads[3], grads[4], grads[5], grads[6], None

        q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f = ctx.saved_tensors
        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        BH = B * H
        work_device = ctx.work_device

        go = grad_output.to(device=work_device, dtype=torch.float32).contiguous()
        go_f = go.reshape(BH, T, V)

        BK = max(16, _next_power_of_two(K))
        BV = 16 if BK >= 256 else 32
        NV = triton.cdiv(V, BV)
        BC = 32
        num_chunks = (T + BC - 1) // BC

        chk = torch.empty((BH * num_chunks * K * V,), device=work_device, dtype=torch.float32)
        scratch = torch.empty((BH * NV * BC * BK * BV,), device=work_device, dtype=torch.float32)

        dq_part = torch.empty((NV, BH, T, K), device=work_device, dtype=torch.float32)
        dk_part = torch.empty((NV, BH, T, K), device=work_device, dtype=torch.float32)
        de_part = torch.empty((NV, BH, T, K), device=work_device, dtype=torch.float32)
        dv_f = torch.empty((BH, T, V), device=work_device, dtype=torch.float32)
        dg_part = torch.empty((NV, BH, T), device=work_device, dtype=torch.float32)
        dbeta_part = torch.empty((NV, BH, T), device=work_device, dtype=torch.float32)
        dgamma_part = torch.empty((NV, BH, T), device=work_device, dtype=torch.float32)

        num_warps = 8 if BK * BV > 4096 else 4
        _backward_delta_kernel[(NV, BH)](
            q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f, go_f,
            chk, scratch,
            dq_part, dk_part, de_part, dv_f,
            dg_part, dbeta_part, dgamma_part,
            T, K, V, BH, NV, ctx.scale,
            BK=BK, BV=BV, BC=BC,
            num_warps=num_warps,
            num_stages=2,
        )

        metas = ctx.input_metas
        dq = dq_part.sum(dim=0).reshape(B, H, T, K).to(device=metas[0][2], dtype=metas[0][1])
        dk = dk_part.sum(dim=0).reshape(B, H, T, K).to(device=metas[1][2], dtype=metas[1][1])
        dv = dv_f.reshape(B, H, T, V).to(device=metas[2][2], dtype=metas[2][1])
        de = de_part.sum(dim=0).reshape(B, H, T, K).to(device=metas[3][2], dtype=metas[3][1])
        dg = dg_part.sum(dim=0).reshape(B, H, T).to(device=metas[4][2], dtype=metas[4][1])
        dbeta = dbeta_part.sum(dim=0).reshape(B, H, T).to(device=metas[5][2], dtype=metas[5][1])
        dgamma = dgamma_part.sum(dim=0).reshape(B, H, T).to(device=metas[6][2], dtype=metas[6][1])

        return dq, dk, dv, de, dg, dbeta, dgamma, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _DeltaEraseFunction.apply(q, k, v, e, g, beta, gamma, scale)
```

The forward reuses the fast recurrent Triton kernel, while the backward implements the exact adjoint recurrence derived from the decay, erase, and delta-write stages in a second Triton kernel. To avoid storing every state, the backward first records chunk-start states, then replays each 32-step chunk into a small reusable scratch buffer and walks it backwards, rebuilding each step's intermediates from the previous state. Value-block partial gradients for the K-side and scalar inputs are reduced in the wrapper and cast back to each input's dtype. My main uncertainties are Triton's handling of the scalar conditional load for the first state in each chunk and register pressure on unusually large head dimensions.