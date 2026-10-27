```python
import torch
import triton
import triton.language as tl


@triton.jit
def _forward_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr, ckpt_ptr,
    T, Kdim, Vdim, NV, NC,
    scale,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BT: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    iv = pid % NV

    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    mask_k = off_k < Kdim
    v_idx = iv * BLOCK_V + off_v
    mask_v = v_idx < Vdim
    tile_mask = mask_k[:, None] & mask_v[None, :]
    tile_off = off_k[:, None] * Vdim + v_idx[None, :]

    qk_base = bh * T * Kdim
    v_base = bh * T * Vdim
    beta_base = bh * T
    ckpt_base = bh * NC * Kdim * Vdim

    # S[bh] sliced over V: [K, BLOCK_V]
    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    for t in range(T):
        if t % BT == 0:
            tl.store(
                ckpt_ptr + ckpt_base + (t // BT) * Kdim * Vdim + tile_off,
                S,
                mask=tile_mask,
            )

        kt = tl.load(
            k_ptr + qk_base + t * Kdim + off_k,
            mask=mask_k, other=0.0,
        ).to(tl.float32)
        gt = tl.load(
            g_ptr + qk_base + t * Kdim + off_k,
            mask=mask_k, other=0.0,
        ).to(tl.float32)
        vt = tl.load(
            v_ptr + v_base + t * Vdim + v_idx,
            mask=mask_v, other=0.0,
        ).to(tl.float32)
        bt = tl.load(beta_ptr + beta_base + t).to(tl.float32)

        pred = tl.sum(S * kt[:, None], axis=0)
        delta = bt * (vt - pred)
        S = S + kt[:, None] * delta[None, :]
        S = S * tl.exp(gt)[:, None]

        qt = tl.load(
            q_ptr + qk_base + t * Kdim + off_k,
            mask=mask_k, other=0.0,
        ).to(tl.float32)
        ot = tl.sum(S * (qt * scale)[:, None], axis=0)
        tl.store(out_ptr + v_base + t * Vdim + v_idx, ot, mask=mask_v)


@triton.jit
def _backward_chunk_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, do_ptr, ckpt_ptr,
    scratch_ptr, grad_s_ptr,
    dq_ptr, dk_ptr, dv_ptr, dg_ptr, dbeta_ptr,
    T, Kdim, Vdim, NV, NC,
    chunk_start, chunk_len, chunk_idx,
    scale,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BT: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    iv = pid % NV

    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    mask_k = off_k < Kdim
    v_idx = iv * BLOCK_V + off_v
    mask_v = v_idx < Vdim
    tile_mask = mask_k[:, None] & mask_v[None, :]
    tile_off = off_k[:, None] * Vdim + v_idx[None, :]

    qk_base = bh * T * Kdim
    v_base = bh * T * Vdim
    beta_base = bh * T
    ckpt_off = (bh * NC + chunk_idx) * Kdim * Vdim + tile_off
    scratch_base = bh * BT * Kdim * Vdim
    grad_s_off = bh * Kdim * Vdim + tile_off

    S_ckpt = tl.load(ckpt_ptr + ckpt_off, mask=tile_mask, other=0.0)
    # Adjoint w.r.t. S at the last timestep of this chunk, excluding the
    # output contribution at that timestep (added inside the loop).
    G = tl.load(grad_s_ptr + grad_s_off, mask=tile_mask, other=0.0)

    # Phase 1: recompute this chunk's forward states from its checkpoint.
    S = S_ckpt
    for i in range(chunk_len):
        t = chunk_start + i
        kt = tl.load(
            k_ptr + qk_base + t * Kdim + off_k,
            mask=mask_k, other=0.0,
        ).to(tl.float32)
        gt = tl.load(
            g_ptr + qk_base + t * Kdim + off_k,
            mask=mask_k, other=0.0,
        ).to(tl.float32)
        vt = tl.load(
            v_ptr + v_base + t * Vdim + v_idx,
            mask=mask_v, other=0.0,
        ).to(tl.float32)
        bt = tl.load(beta_ptr + beta_base + t).to(tl.float32)

        pred = tl.sum(S * kt[:, None], axis=0)
        delta = bt * (vt - pred)
        S = S + kt[:, None] * delta[None, :]
        S = S * tl.exp(gt)[:, None]

        tl.store(
            scratch_ptr + scratch_base + i * Kdim * Vdim + tile_off,
            S,
            mask=tile_mask,
        )

    # Phase 2: reverse sweep through the chunk.
    for rev in range(chunk_len):
        i = chunk_len - 1 - rev
        t = chunk_start + i

        S_t = tl.load(
            scratch_ptr + scratch_base + i * Kdim * Vdim + tile_off,
            mask=tile_mask, other=0.0,
        )
        S_prev = S_ckpt
        if i != 0:
            S_prev = tl.load(
                scratch_ptr + scratch_base + (i - 1) * Kdim * Vdim + tile_off,
                mask=tile_mask, other=0.0,
            )

        kt = tl.load(
            k_ptr + qk_base + t * Kdim + off_k,
            mask=mask_k, other=0.0,
        ).to(tl.float32)
        gt = tl.load(
            g_ptr + qk_base + t * Kdim + off_k,
            mask=mask_k, other=0.0,
        ).to(tl.float32)
        vt = tl.load(
            v_ptr + v_base + t * Vdim + v_idx,
            mask=mask_v, other=0.0,
        ).to(tl.float32)
        bt = tl.load(beta_ptr + beta_base + t).to(tl.float32)
        qt = tl.load(
            q_ptr + qk_base + t * Kdim + off_k,
            mask=mask_k, other=0.0,
        ).to(tl.float32)
        do_t = tl.load(
            do_ptr + v_base + t * Vdim + v_idx,
            mask=mask_v, other=0.0,
        ).to(tl.float32)

        d_t = tl.exp(gt)

        # Total adjoint w.r.t. S_t, including this timestep's output.
        G_tot = G + scale * qt[:, None] * do_t[None, :]
        # Adjoint w.r.t. the pre-decay state.
        Gt = G_tot * d_t[:, None]
        du = tl.sum(Gt * kt[:, None], axis=0)

        pred = tl.sum(S_prev * kt[:, None], axis=0)
        resid = vt - pred
        u_t = bt * resid

        dq_part = scale * tl.sum(S_t * do_t[None, :], axis=1)
        dk_part = tl.sum(Gt * u_t[None, :], axis=1) - bt * tl.sum(
            S_prev * du[None, :], axis=1
        )
        dg_part = tl.sum(G_tot * S_t, axis=1)
        dv_part = bt * du
        dbeta_part = tl.sum(du * resid, axis=0)

        tl.store(dv_ptr + v_base + t * Vdim + v_idx, dv_part, mask=mask_v)
        tl.atomic_add(dq_ptr + qk_base + t * Kdim + off_k, dq_part, mask=mask_k)
        tl.atomic_add(dk_ptr + qk_base + t * Kdim + off_k, dk_part, mask=mask_k)
        tl.atomic_add(dg_ptr + qk_base + t * Kdim + off_k, dg_part, mask=mask_k)
        tl.atomic_add(dbeta_ptr + beta_base + t, dbeta_part)

        # Adjoint w.r.t. S_{t-1}: (I - beta k k^T) applied to Gt.
        G = Gt - bt * kt[:, None] * du[None, :]

    tl.store(grad_s_ptr + grad_s_off, G, mask=tile_mask)


def _pow2_at_least(n, minimum=16):
    p = triton.next_power_of_2(max(int(n), 1))
    return max(p, minimum)


def _block_config(Kdim, Vdim):
    block_k = _pow2_at_least(Kdim, 16)
    if Vdim <= 16:
        block_v = 16
    elif Vdim <= 32:
        block_v = 32
    else:
        block_v = 64
    while block_k * block_v > 16384 and block_v > 16:
        block_v //= 2
    num_warps = 8 if block_k * block_v > 8192 else 4
    return block_k, block_v, num_warps


def _as_float(x):
    if torch.is_tensor(x):
        return float(x.detach().item())
    return float(x)


class _RecurrentFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, Kdim = q.shape
        Vdim = v.shape[-1]
        BT = 32
        NC = (T + BT - 1) // BT if T > 0 else 0

        qc = q.contiguous()
        kc = k.contiguous()
        vc = v.contiguous()
        gc = g.contiguous()
        bc = beta.contiguous()

        out = torch.empty(B, H, T, Vdim, device=q.device, dtype=torch.float32)
        ckpt = torch.empty(
            B, H, NC, Kdim, Vdim, device=q.device, dtype=torch.float32
        )

        if out.numel() > 0 and T > 0:
            block_k, block_v, num_warps = _block_config(Kdim, Vdim)
            NV = (Vdim + block_v - 1) // block_v
            _forward_kernel[(B * H * NV,)](
                qc, kc, vc, gc, bc, out, ckpt,
                T, Kdim, Vdim, NV, NC,
                float(scale),
                BLOCK_K=block_k, BLOCK_V=block_v, BT=BT,
                num_warps=num_warps,
            )

        ctx.save_for_backward(qc, kc, vc, e, gc, bc, gamma, ckpt)
        ctx.scale = float(scale)
        ctx.BT = BT
        ctx.NC = NC
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, e, g, beta, gamma, ckpt = ctx.saved_tensors
        B, H, T, Kdim = q.shape
        Vdim = v.shape[-1]
        BT = ctx.BT
        NC = ctx.NC
        scale = ctx.scale

        dq32 = torch.zeros(B, H, T, Kdim, device=q.device, dtype=torch.float32)
        dk32 = torch.zeros(B, H, T, Kdim, device=q.device, dtype=torch.float32)
        dv32 = torch.zeros(B, H, T, Vdim, device=q.device, dtype=torch.float32)
        dg32 = torch.zeros(B, H, T, Kdim, device=q.device, dtype=torch.float32)
        dbeta32 = torch.zeros(B, H, T, device=q.device, dtype=torch.float32)

        if T > 0 and grad_output.numel() > 0:
            go = grad_output.contiguous().to(torch.float32)
            grad_s = torch.zeros(
                B, H, Kdim, Vdim, device=q.device, dtype=torch.float32
            )
            scratch = torch.empty(
                B, H, BT, Kdim, Vdim, device=q.device, dtype=torch.float32
            )
            block_k, block_v, num_warps = _block_config(Kdim, Vdim)
            NV = (Vdim + block_v - 1) // block_v
            grid = (B * H * NV,)
            for ci in range(NC - 1, -1, -1):
                start = ci * BT
                length = min(BT, T - start)
                _backward_chunk_kernel[grid](
                    q, k, v, g, beta, go, ckpt,
                    scratch, grad_s,
                    dq32, dk32, dv32, dg32, dbeta32,
                    T, Kdim, Vdim, NV, NC,
                    start, length, ci,
                    scale,
                    BLOCK_K=block_k, BLOCK_V=block_v, BT=BT,
                    num_warps=num_warps,
                )

        grad_q = dq32.to(q.dtype)
        grad_k = dk32.to(k.dtype)
        grad_v = dv32.to(v.dtype)
        grad_g = dg32.to(g.dtype)
        grad_beta = dbeta32.to(beta.dtype)
        grad_e = torch.zeros_like(e) if e.requires_grad else None
        grad_gamma = torch.zeros_like(gamma) if gamma.requires_grad else None
        return grad_q, grad_k, grad_v, grad_e, grad_g, grad_beta, grad_gamma, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _RecurrentFn.apply(q, k, v, e, g, beta, gamma, _as_float(scale))
```

The forward kernel keeps the recurrent state in registers, splitting only the $V$ dimension across programs, and stores a state checkpoint every 32 timesteps. The backward is the exact adjoint recurrence: each chunk is relaunched in reverse order, its forward states are recomputed from the checkpoint into a small reused scratch buffer, and the reverse sweep propagates the state adjoint while accumulating gradients, using atomics only for the reductions across $V$-blocks ($dq$, $dk$, $dg$, $d\beta$). Since $e$ and $\gamma$ never enter the reference computation, their gradients are exact zeros. I am unsure about Triton 3.5's handling of the tensor reassignment under the `if i != 0` branch in the backward kernel, and checkpoint/scratch memory (roughly 200 MB at the stated size) could be tight alongside other training state.