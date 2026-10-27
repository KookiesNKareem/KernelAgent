```python
import torch
import triton
import triton.language as tl


@triton.jit
def _forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    out_ptr,
    state_ptr,
    T,
    K_DIM,
    V_DIM,
    scale,
    NUM_V_BLOCKS,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_bh = pid // NUM_V_BLOCKS
    pid_v = pid - pid_bh * NUM_V_BLOCKS

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K_DIM
    mask_v = offs_v < V_DIM

    # h[v, k] is the transpose of the state S[k, v].
    h = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)

    bh = pid_bh.to(tl.int64)
    base_k = bh * T * K_DIM
    base_v = bh * T * V_DIM
    base_s = bh * T
    base_state = bh * T * V_DIM * K_DIM

    for t in range(T):
        # Save S_{t-1} (transposed, [V, K] layout) for the backward pass.
        tl.store(
            state_ptr
            + base_state
            + t * V_DIM * K_DIM
            + offs_v[:, None] * K_DIM
            + offs_k[None, :],
            h,
            mask=mask_v[:, None] & mask_k[None, :],
        )

        off_k = base_k + t * K_DIM + offs_k
        off_v = base_v + t * V_DIM + offs_v

        q_vec = tl.load(q_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        k_vec = tl.load(k_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        e_vec = tl.load(e_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        g_vec = tl.load(g_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptr + off_v, mask=mask_v, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + base_s + t).to(tl.float32)
        gamma_t = tl.load(gamma_ptr + base_s + t).to(tl.float32)

        h = h * tl.exp(g_vec)[None, :]

        erased = tl.sum(h * e_vec[None, :], axis=1)
        h = h - (gamma_t * erased)[:, None] * e_vec[None, :]

        pred = tl.sum(h * k_vec[None, :], axis=1)
        delta = beta_t * (v_vec - pred)
        h = h + delta[:, None] * k_vec[None, :]

        out_vec = tl.sum(h * (q_vec * scale)[None, :], axis=1)
        tl.store(out_ptr + off_v, out_vec, mask=mask_v)


# do_not_specialize is retained from the compiling version: it prevents the
# divisibility-by-16 specialization of the dimension arguments under which
# Triton's layout-conversion pass failed on this kernel.  Unlike the
# previous version, gradients over K are now reduced inside the kernel
# (axis=0 sums over the local V tile) and written as one partial per
# V-block, instead of writing one contribution per (V, K) element and
# reducing huge [BH, T, V, K] buffers in PyTorch.
@triton.jit(do_not_specialize=["T", "K_DIM", "V_DIM", "BH_DIM", "NUM_V_BLOCKS"])
def _backward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    go_ptr,
    state_ptr,
    gq_part_ptr,
    gk_part_ptr,
    gv_ptr,
    ge_part_ptr,
    gg_part_ptr,
    gbeta_full_ptr,
    ggamma_full_ptr,
    T,
    K_DIM,
    V_DIM,
    BH_DIM,
    scale,
    NUM_V_BLOCKS,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_bh = pid // NUM_V_BLOCKS
    pid_v = pid - pid_bh * NUM_V_BLOCKS

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K_DIM
    mask_v = offs_v < V_DIM
    mask_h = mask_v[:, None] & mask_k[None, :]

    bh = pid_bh.to(tl.int64)
    base_k = bh * T * K_DIM
    base_v = bh * T * V_DIM
    base_s = bh * T
    base_state = bh * T * V_DIM * K_DIM
    part_base = pid_v.to(tl.int64) * BH_DIM + bh

    # adj[v, k] is the transpose of dL/dS_t.
    adj = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)

    for rev in range(T):
        t = T - 1 - rev

        off_k = base_k + t * K_DIM + offs_k
        off_v = base_v + t * V_DIM + offs_v
        off_h = (
            base_state
            + t * V_DIM * K_DIM
            + offs_v[:, None] * K_DIM
            + offs_k[None, :]
        )
        off_part_k = (part_base * T + t) * K_DIM + offs_k

        q_vec = tl.load(q_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        k_vec = tl.load(k_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        e_vec = tl.load(e_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        g_vec = tl.load(g_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptr + off_v, mask=mask_v, other=0.0).to(tl.float32)
        go_vec = tl.load(go_ptr + off_v, mask=mask_v, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + base_s + t).to(tl.float32)
        gamma_t = tl.load(gamma_ptr + base_s + t).to(tl.float32)

        # Reload S_{t-1} and rebuild the intermediate states of step t.
        h_prev = tl.load(state_ptr + off_h, mask=mask_h, other=0.0).to(tl.float32)

        decay = tl.exp(g_vec)
        h_d = h_prev * decay[None, :]
        u_vec = tl.sum(h_d * e_vec[None, :], axis=1)
        h_e = h_d - (gamma_t * u_vec)[:, None] * e_vec[None, :]
        p_vec = tl.sum(h_e * k_vec[None, :], axis=1)
        delta_v = v_vec - p_vec
        w_vec = beta_t * delta_v

        # Include dL/do_t: adj becomes dL/dS_t (transposed).
        adj = adj + (scale * go_vec)[:, None] * q_vec[None, :]

        a_vec = tl.sum(adj * k_vec[None, :], axis=1)

        tl.store(gv_ptr + off_v, beta_t * a_vec, mask=mask_v)
        tl.store(gbeta_full_ptr + off_v, a_vec * delta_v, mask=mask_v)

        # dL/dq_t from S_t (transposed) = h_e + w k^T, reduced over V here.
        h_t = h_e + w_vec[:, None] * k_vec[None, :]
        gq_vec = scale * tl.sum(h_t * go_vec[:, None], axis=0)
        tl.store(gq_part_ptr + off_part_k, gq_vec, mask=mask_k)

        gk_vec = tl.sum(adj * w_vec[:, None], axis=0) - beta_t * tl.sum(
            h_e * a_vec[:, None], axis=0
        )
        tl.store(gk_part_ptr + off_part_k, gk_vec, mask=mask_k)

        # Adjoint through the erase update.
        adj_e = adj - (beta_t * a_vec)[:, None] * k_vec[None, :]
        b_vec = tl.sum(adj_e * e_vec[None, :], axis=1)

        tl.store(ggamma_full_ptr + off_v, -b_vec * u_vec, mask=mask_v)

        adj_d = adj_e - (gamma_t * b_vec)[:, None] * e_vec[None, :]

        ge_vec = -gamma_t * (
            tl.sum(adj_e * u_vec[:, None], axis=0)
            + tl.sum(h_d * b_vec[:, None], axis=0)
        )
        tl.store(ge_part_ptr + off_part_k, ge_vec, mask=mask_k)

        gg_vec = tl.sum(adj_d * h_d, axis=0)
        tl.store(gg_part_ptr + off_part_k, gg_vec, mask=mask_k)

        # Propagate the adjoint to S_{t-1}.
        adj = adj_d * decay[None, :]


def _next_power_of_2(x):
    n = 1
    while n < x:
        n *= 2
    return n


def _tile_config(K, V):
    block_k = max(16, _next_power_of_2(max(1, K)))
    if block_k >= 256:
        desired_v = 16
    elif block_k == 128:
        desired_v = 32
    else:
        desired_v = 64
    if V < desired_v:
        block_v = max(16, _next_power_of_2(max(1, V)))
    else:
        block_v = desired_v
    num_v_blocks = (V + block_v - 1) // block_v
    num_warps = 4 if block_k * block_v >= 4096 else 2
    return block_k, block_v, num_v_blocks, num_warps


class _RecurrentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        scale_f = float(scale)

        q_flat = q.contiguous().reshape(BH, T, K)
        k_flat = k.contiguous().reshape(BH, T, K)
        e_flat = e.contiguous().reshape(BH, T, K)
        g_flat = g.contiguous().reshape(BH, T, K)
        v_flat = v.contiguous().reshape(BH, T, V)
        beta_flat = beta.contiguous().reshape(BH, T)
        gamma_flat = gamma.contiguous().reshape(BH, T)

        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        out_flat = out.reshape(BH, T, V)

        states = torch.empty((BH, T, V, K), device=q.device, dtype=torch.float32)

        block_k, block_v, num_v_blocks, num_warps = _tile_config(K, V)

        if BH > 0 and T > 0 and K > 0 and V > 0:
            _forward_kernel[(BH * num_v_blocks,)](
                q_flat,
                k_flat,
                v_flat,
                e_flat,
                g_flat,
                beta_flat,
                gamma_flat,
                out_flat,
                states,
                T,
                K,
                V,
                scale_f,
                num_v_blocks,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                num_warps=num_warps,
                num_stages=2,
            )

        ctx.save_for_backward(q, k, v, e, g, beta, gamma)
        ctx.states = states
        ctx.scale_f = scale_f
        ctx.block_k = block_k
        ctx.block_v = block_v
        ctx.num_v_blocks = num_v_blocks
        ctx.num_warps = num_warps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        nv = ctx.num_v_blocks
        dev = q.device

        if BH == 0 or T == 0 or K == 0 or V == 0:
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

        q_flat = q.contiguous().reshape(BH, T, K)
        k_flat = k.contiguous().reshape(BH, T, K)
        e_flat = e.contiguous().reshape(BH, T, K)
        g_flat = g.contiguous().reshape(BH, T, K)
        v_flat = v.contiguous().reshape(BH, T, V)
        beta_flat = beta.contiguous().reshape(BH, T)
        gamma_flat = gamma.contiguous().reshape(BH, T)
        go_flat = grad_out.contiguous().reshape(BH, T, V)

        # One partial per V-block: [nv, BH, T, K], summed over nv below.
        # This is BLOCK_V times smaller than the previous per-element
        # contribution buffers.
        gq_part = torch.empty((nv, BH, T, K), device=dev, dtype=torch.float32)
        gk_part = torch.empty((nv, BH, T, K), device=dev, dtype=torch.float32)
        ge_part = torch.empty((nv, BH, T, K), device=dev, dtype=torch.float32)
        gg_part = torch.empty((nv, BH, T, K), device=dev, dtype=torch.float32)
        gv_flat = torch.empty((BH, T, V), device=dev, dtype=torch.float32)
        gbeta_full = torch.empty((BH, T, V), device=dev, dtype=torch.float32)
        ggamma_full = torch.empty((BH, T, V), device=dev, dtype=torch.float32)

        bwd_warps = 8 if ctx.block_k * ctx.block_v >= 4096 else ctx.num_warps

        _backward_kernel[(BH * nv,)](
            q_flat,
            k_flat,
            v_flat,
            e_flat,
            g_flat,
            beta_flat,
            gamma_flat,
            go_flat,
            ctx.states,
            gq_part,
            gk_part,
            gv_flat,
            ge_part,
            gg_part,
            gbeta_full,
            ggamma_full,
            T,
            K,
            V,
            BH,
            ctx.scale_f,
            nv,
            BLOCK_K=ctx.block_k,
            BLOCK_V=ctx.block_v,
            num_warps=bwd_warps,
            num_stages=2,
        )

        gq = gq_part.sum(dim=0).reshape(B, H, T, K)
        gk = gk_part.sum(dim=0).reshape(B, H, T, K)
        ge = ge_part.sum(dim=0).reshape(B, H, T, K)
        gg = gg_part.sum(dim=0).reshape(B, H, T, K)
        gv = gv_flat.reshape(B, H, T, V)
        gbeta = gbeta_full.sum(dim=2).reshape(B, H, T)
        ggamma = ggamma_full.sum(dim=2).reshape(B, H, T)

        return (
            gq.to(q.dtype),
            gk.to(k.dtype),
            gv.to(v.dtype),
            ge.to(e.dtype),
            gg.to(g.dtype),
            gbeta.to(beta.dtype),
            ggamma.to(gamma.dtype),
            None,
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if isinstance(scale, torch.Tensor):
        scale = float(scale.detach().item())
    return _RecurrentFunction.apply(q, k, v, e, g, beta, gamma, float(scale))
```

The profile shows the old backward spent most of its time writing four `[BH, T, V, K]` fp32 contribution buffers (about 17 GB at the timed shape) and then reading them back in a 67 ms PyTorch reduction. The backward kernel now reduces each V-tile internally and writes only one `[K]` partial per V-block per step — 32× less gradient traffic — so the remaining PyTorch sums shrink from ~17 GB to ~0.5 GB of reads, and the backward also gets 8 warps to spread its several live state tiles across more threads. The forward kernel, saved states, adjoint formulas, and the `do_not_specialize` guard that fixed the earlier `PassManager` failure are all unchanged. The main uncertainty is whether the new in-kernel axis-0 reductions compile as cleanly on the checker's Triton as the non-specialized shapes did before, and how much their cross-lane cost offsets the eliminated memory traffic.