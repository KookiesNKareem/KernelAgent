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
def _backward_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, do_ptr, ckpt_ptr,
    scratch_ptr,
    dq_ptr, dk_ptr, dv_ptr, dg_ptr, dbeta_ptr,
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
    scratch_base = bh * BT * Kdim * Vdim

    # Adjoint w.r.t. S at the current (reverse) position.  It is carried
    # in registers across all chunks: one launch for the whole backward.
    G = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    for chunk_rev in range(NC):
        chunk_idx = NC - 1 - chunk_rev
        start = chunk_idx * BT
        remaining = T - start
        chunk_len = tl.where(remaining < BT, remaining, BT)

        # Recompute this chunk, storing S_{t-1} (the state *before* each
        # step).  All gradient formulae can be rebuilt from S_{t-1}
        # alone, so the reverse pass needs exactly one scratch load
        # per step, and no separate S_t load.
        S = tl.load(
            ckpt_ptr + ckpt_base + chunk_idx * Kdim * Vdim + tile_off,
            mask=tile_mask, other=0.0,
        )
        for i in range(chunk_len):
            t = start + i
            tl.store(
                scratch_ptr + scratch_base + i * Kdim * Vdim + tile_off,
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

        # Reverse sweep through the chunk.
        for rev in range(chunk_len):
            i = chunk_len - 1 - rev
            t = start + i

            s_prev = tl.load(
                scratch_ptr + scratch_base + i * Kdim * Vdim + tile_off,
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

            # Total adjoint w.r.t. S_t, then w.r.t. the pre-decay state.
            G_tot = G + scale * qt[:, None] * do_t[None, :]
            Gt = G_tot * d_t[:, None]
            du = tl.sum(Gt * kt[:, None], axis=0)

            pred = tl.sum(s_prev * kt[:, None], axis=0)
            resid = vt - pred
            u_t = bt * resid

            # Reductions involving S_{t-1}, taken before forming S_pre.
            dk_part = tl.sum(Gt * u_t[None, :], axis=1) - bt * tl.sum(
                s_prev * du[None, :], axis=1
            )

            # S_pre = S_{t-1} + k u^T, and S_t = d * S_pre.  Hence
            #   dq = scale * d * (S_pre @ do),  dg = Gt^T-reduced S_pre,
            # so S_t itself never has to be materialised or loaded.
            s_pre = s_prev + kt[:, None] * u_t[None, :]
            dq_part = scale * d_t * tl.sum(s_pre * do_t[None, :], axis=1)
            dg_part = tl.sum(Gt * s_pre, axis=1)
            dv_part = bt * du
            dbeta_part = tl.sum(du * resid, axis=0)

            tl.store(dv_ptr + v_base + t * Vdim + v_idx, dv_part, mask=mask_v)
            tl.atomic_add(dq_ptr + qk_base + t * Kdim + off_k, dq_part, mask=mask_k)
            tl.atomic_add(dk_ptr + qk_base + t * Kdim + off_k, dk_part, mask=mask_k)
            tl.atomic_add(dg_ptr + qk_base + t * Kdim + off_k, dg_part, mask=mask_k)
            tl.atomic_add(dbeta_ptr + beta_base + t, dbeta_part)

            # Adjoint w.r.t. S_{t-1}.
            G = Gt - bt * kt[:, None] * du[None, :]


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
    return block_k, block_v


def _as_float(x):
    if torch.is_tensor(x):
        return float(x.detach().item())
    return float(x)


class _RecurrentFn(torch.autograd.Function):
    # Checkpoint interval.  With BT=8 the per-chunk scratch working set
    # is B*H*BT*K*V*4 = 16 MB at the target shape (vs 64 MB at BT=32),
    # small enough to stay largely resident in the L4's 48 MB L2 while
    # a chunk is recomputed and immediately re-read in reverse.
    BT = 8

    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, Kdim = q.shape
        Vdim = v.shape[-1]
        BT = _RecurrentFn.BT
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
            block_k, block_v = _block_config(Kdim, Vdim)
            NV = (Vdim + block_v - 1) // block_v
            _forward_kernel[(B * H * NV,)](
                qc, kc, vc, gc, bc, out, ckpt,
                T, Kdim, Vdim, NV, NC,
                float(scale),
                BLOCK_K=block_k, BLOCK_V=block_v, BT=BT,
                num_warps=4,
            )

        ctx.save_for_backward(qc, kc, vc, e, gc, bc, gamma, ckpt)
        ctx.scale = float(scale)
        ctx.NC = NC
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, e, g, beta, gamma, ckpt = ctx.saved_tensors
        B, H, T, Kdim = q.shape
        Vdim = v.shape[-1]
        BT = _RecurrentFn.BT
        NC = ctx.NC
        scale = ctx.scale

        dq32 = torch.zeros(B, H, T, Kdim, device=q.device, dtype=torch.float32)
        dk32 = torch.zeros(B, H, T, Kdim, device=q.device, dtype=torch.float32)
        dv32 = torch.zeros(B, H, T, Vdim, device=q.device, dtype=torch.float32)
        dg32 = torch.zeros(B, H, T, Kdim, device=q.device, dtype=torch.float32)
        dbeta32 = torch.zeros(B, H, T, device=q.device, dtype=torch.float32)

        if T > 0 and grad_output.numel() > 0:
            go = grad_output.contiguous().to(torch.float32)
            scratch = torch.empty(
                B, H, BT, Kdim, Vdim, device=q.device, dtype=torch.float32
            )
            block_k, block_v = _block_config(Kdim, Vdim)
            NV = (Vdim + block_v - 1) // block_v
            _backward_kernel[(B * H * NV,)](
                q, k, v, g, beta, go, ckpt,
                scratch,
                dq32, dk32, dv32, dg32, dbeta32,
                T, Kdim, Vdim, NV, NC,
                scale,
                BLOCK_K=block_k, BLOCK_V=block_v, BT=BT,
                num_warps=8,
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

The main change is shrinking the checkpoint interval from 32 to 8, which cuts the backward scratch working set from ~64 MB to ~16 MB so the recomputed states stay largely in L2 instead of round-tripping to DRAM, and replacing the 64 separate per-chunk backward launches (each reloading/storing the state adjoint through HBM) with a single launch that carries the adjoint in registers across all chunks. The recompute phase now stores the pre-step state $S_{t-1}$ rather than $S_t$; since $dq$ and $dg$ can be rewritten in terms of the pre-decay state, the reverse pass needs only one scratch load per step instead of two, with no carried-state bookkeeping. Everything else (exact adjoint recurrence, atomics only across the two $V$-blocks, exact zero gradients for the unused $e$ and $\gamma$) is unchanged, so correctness should be preserved. The main risk is whether Triton's handling of the register-carried adjoint across the nested chunk/step loops compiles as expected, and whether the denser checkpointing (512 MB written in forward at the target shape) costs more than the L2 residency saves, but both should be net wins given the profile pointed entirely at the old backward kernel.