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


# The only shapes that compiled previously were K=V=65 and K=V=129, i.e.
# dimensions NOT divisible by 16.  Triton specializes such integer
# arguments (divisibility by 16), and under that specialization the
# layout-conversion pass failed on this kernel's axis=0 (cross-V)
# reductions.  The fix below therefore
#   (a) disables that specialization for the dimension arguments, and
#   (b) removes every axis=0 / scalar reduction from the kernel: it now
#       contains only axis=1 reductions, exactly like the forward kernel
#       that always compiled.  The cross-V sums are instead formed by
#       storing per-V contributions and reducing them in PyTorch.
@triton.jit(do_not_specialize=["T", "K_DIM", "V_DIM", "NUM_V_BLOCKS"])
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
    gq_contrib_ptr,
    gk_contrib_ptr,
    gv_ptr,
    ge_contrib_ptr,
    gg_contrib_ptr,
    gbeta_full_ptr,
    ggamma_full_ptr,
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
    mask_h = mask_v[:, None] & mask_k[None, :]

    bh = pid_bh.to(tl.int64)
    base_k = bh * T * K_DIM
    base_v = bh * T * V_DIM
    base_s = bh * T
    base_state = bh * T * V_DIM * K_DIM

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

        # Per-V contributions whose sum over V gives dL/dq_t.  Here
        # S_t (transposed) is h_e + w k^T.
        tl.store(
            gq_contrib_ptr + off_h,
            scale * (h_e + w_vec[:, None] * k_vec[None, :]) * go_vec[:, None],
            mask=mask_h,
        )

        # Per-V contributions whose sum over V gives dL/dk_t.
        tl.store(
            gk_contrib_ptr + off_h,
            adj * w_vec[:, None] - (beta_t * a_vec)[:, None] * h_e,
            mask=mask_h,
        )

        # Adjoint through the erase update.
        adj_e = adj - (beta_t * a_vec)[:, None] * k_vec[None, :]
        b_vec = tl.sum(adj_e * e_vec[None, :], axis=1)

        tl.store(ggamma_full_ptr + off_v, -b_vec * u_vec, mask=mask_v)

        adj_d = adj_e - (gamma_t * b_vec)[:, None] * e_vec[None, :]

        # Per-V contributions whose sums over V give dL/de_t and dL/dg_t.
        tl.store(
            ge_contrib_ptr + off_h,
            -gamma_t
            * (adj_e * u_vec[:, None] + h_d * b_vec[:, None]),
            mask=mask_h,
        )
        tl.store(gg_contrib_ptr + off_h, adj_d * h_d, mask=mask_h)

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

        # Per-V gradient contributions, [BH, T, V, K]; summed over V below.
        gq_contrib = torch.empty((BH, T, V, K), device=dev, dtype=torch.float32)
        gk_contrib = torch.empty((BH, T, V, K), device=dev, dtype=torch.float32)
        ge_contrib = torch.empty((BH, T, V, K), device=dev, dtype=torch.float32)
        gg_contrib = torch.empty((BH, T, V, K), device=dev, dtype=torch.float32)
        gv_flat = torch.empty((BH, T, V), device=dev, dtype=torch.float32)
        gbeta_full = torch.empty((BH, T, V), device=dev, dtype=torch.float32)
        ggamma_full = torch.empty((BH, T, V), device=dev, dtype=torch.float32)

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
            gq_contrib,
            gk_contrib,
            gv_flat,
            ge_contrib,
            gg_contrib,
            gbeta_full,
            ggamma_full,
            T,
            K,
            V,
            ctx.scale_f,
            nv,
            BLOCK_K=ctx.block_k,
            BLOCK_V=ctx.block_v,
            num_warps=ctx.num_warps,
            num_stages=2,
        )

        gq = gq_contrib.sum(dim=2).reshape(B, H, T, K)
        gk = gk_contrib.sum(dim=2).reshape(B, H, T, K)
        ge = ge_contrib.sum(dim=2).reshape(B, H, T, K)
        gg = gg_contrib.sum(dim=2).reshape(B, H, T, K)
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
