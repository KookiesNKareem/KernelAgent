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
    chk_ptr,
    T,
    K_DIM,
    V_DIM,
    scale,
    NUM_V_BLOCKS,
    NC,
    CHUNK: tl.constexpr,
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
    base_chk = bh * NC * K_DIM * V_DIM

    for t in range(T):
        if t % CHUNK == 0:
            chunk_idx = t // CHUNK
            tl.store(
                chk_ptr
                + base_chk
                + chunk_idx * K_DIM * V_DIM
                + offs_k[None, :] * V_DIM
                + offs_v[:, None],
                h,
                mask=mask_k[None, :] & mask_v[:, None],
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


@triton.jit
def _backward_kernel(
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
    gq_ptr,
    gk_ptr,
    gv_ptr,
    ge_ptr,
    gg_ptr,
    gbeta_ptr,
    ggamma_ptr,
    T,
    K_DIM,
    V_DIM,
    scale,
    NUM_V_BLOCKS,
    NC,
    CHUNK: tl.constexpr,
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
    mask_h = mask_k[None, :] & mask_v[:, None]

    bh = pid_bh.to(tl.int64)
    base_k = bh * T * K_DIM
    base_v = bh * T * V_DIM
    base_s = bh * T
    base_chk = bh * NC * K_DIM * V_DIM
    base_scratch = bh * CHUNK * K_DIM * V_DIM

    # adj[v, k] is the transpose of dL/dS for the current time step.
    adj = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)

    for chunk_rev in range(NC):
        c = NC - 1 - chunk_rev
        t_start = c * CHUNK
        remaining = T - t_start
        chunk_len = CHUNK
        if remaining < CHUNK:
            chunk_len = remaining

        # Reload the exact state at the beginning of this chunk.
        h_cur = tl.load(
            chk_ptr
            + base_chk
            + c * K_DIM * V_DIM
            + offs_k[None, :] * V_DIM
            + offs_v[:, None],
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)

        # Recompute the chunk forward, snapshotting S_{t-1} for each step.
        # Different V-blocks own disjoint V slices of the scratch buffer.
        for i in range(chunk_len):
            t = t_start + i
            tl.store(
                scratch_ptr
                + base_scratch
                + i * K_DIM * V_DIM
                + offs_k[None, :] * V_DIM
                + offs_v[:, None],
                h_cur,
                mask=mask_h,
            )

            off_k = base_k + t * K_DIM + offs_k
            off_v = base_v + t * V_DIM + offs_v
            k_vec = tl.load(k_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
            e_vec = tl.load(e_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
            g_vec = tl.load(g_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
            v_vec = tl.load(v_ptr + off_v, mask=mask_v, other=0.0).to(tl.float32)
            beta_t = tl.load(beta_ptr + base_s + t).to(tl.float32)
            gamma_t = tl.load(gamma_ptr + base_s + t).to(tl.float32)

            h_cur = h_cur * tl.exp(g_vec)[None, :]
            erased = tl.sum(h_cur * e_vec[None, :], axis=1)
            h_cur = h_cur - (gamma_t * erased)[:, None] * e_vec[None, :]
            pred = tl.sum(h_cur * k_vec[None, :], axis=1)
            delta = beta_t * (v_vec - pred)
            h_cur = h_cur + delta[:, None] * k_vec[None, :]

        tl.debug_barrier()

        # Walk the chunk backwards.  Per step, with G = dL/dS_t (adj = G^T):
        #   a = G^T k,  w = beta (v - p),  p = S_e^T k
        #   G_Se = G - beta k a^T
        #   b = G_Se^T e
        #   G_Sd = G_Se - gamma e b^T,  G_{t-1} = diag(exp g) G_Sd
        for rev_i in range(chunk_len):
            i = chunk_len - 1 - rev_i
            t = t_start + i

            h_prev = tl.load(
                scratch_ptr
                + base_scratch
                + i * K_DIM * V_DIM
                + offs_k[None, :] * V_DIM
                + offs_v[:, None],
                mask=mask_h,
                other=0.0,
            ).to(tl.float32)

            off_k = base_k + t * K_DIM + offs_k
            off_v = base_v + t * V_DIM + offs_v

            q_vec = tl.load(q_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
            k_vec = tl.load(k_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
            e_vec = tl.load(e_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
            g_vec = tl.load(g_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
            v_vec = tl.load(v_ptr + off_v, mask=mask_v, other=0.0).to(tl.float32)
            go_vec = tl.load(go_ptr + off_v, mask=mask_v, other=0.0).to(tl.float32)
            beta_t = tl.load(beta_ptr + base_s + t).to(tl.float32)
            gamma_t = tl.load(gamma_ptr + base_s + t).to(tl.float32)

            decay = tl.exp(g_vec)
            h_d = h_prev * decay[None, :]
            u_vec = tl.sum(h_d * e_vec[None, :], axis=1)
            h_e = h_d - (gamma_t * u_vec)[:, None] * e_vec[None, :]
            p_vec = tl.sum(h_e * k_vec[None, :], axis=1)
            h_t = h_e + (beta_t * (v_vec - p_vec))[:, None] * k_vec[None, :]

            # Output contribution: dL/dS_t += scale * q_t (dL/do_t)^T.
            adj = adj + (scale * go_vec)[:, None] * q_vec[None, :]

            a_vec = tl.sum(adj * k_vec[None, :], axis=1)
            delta_v = v_vec - p_vec
            w_vec = beta_t * delta_v

            tl.store(
                gv_ptr + off_v, beta_t * a_vec, mask=mask_v
            )
            tl.atomic_add(
                gbeta_ptr + base_s + t, tl.sum(a_vec * delta_v)
            )

            dk_partial = tl.sum(adj * w_vec[:, None], axis=0) - beta_t * tl.sum(
                h_e * a_vec[:, None], axis=0
            )
            tl.atomic_add(
                gk_ptr + base_k + t * K_DIM + offs_k,
                dk_partial,
                mask=mask_k,
            )

            gq_partial = scale * tl.sum(h_t * go_vec[:, None], axis=0)
            tl.atomic_add(
                gq_ptr + base_k + t * K_DIM + offs_k,
                gq_partial,
                mask=mask_k,
            )

            # Transpose of G_Se.
            adj_e = adj - (beta_t * a_vec)[:, None] * k_vec[None, :]
            b_vec = tl.sum(adj_e * e_vec[None, :], axis=1)

            tl.atomic_add(
                ggamma_ptr + base_s + t, -tl.sum(b_vec * u_vec)
            )

            # Transpose of G_Sd.
            adj_d = adj_e - (gamma_t * b_vec)[:, None] * e_vec[None, :]

            de_partial = -gamma_t * (
                tl.sum(adj_e * u_vec[:, None], axis=0)
                + tl.sum(h_d * b_vec[:, None], axis=0)
            )
            tl.atomic_add(
                ge_ptr + base_k + t * K_DIM + offs_k,
                de_partial,
                mask=mask_k,
            )

            dg_partial = tl.sum(adj_d * h_d, axis=0)
            tl.atomic_add(
                gg_ptr + base_k + t * K_DIM + offs_k,
                dg_partial,
                mask=mask_k,
            )

            # Propagate to S_{t-1}: rows of G are scaled by exp(g).
            adj = adj_d * decay[None, :]

        tl.debug_barrier()


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

        chunk = 64 if T >= 64 else max(1, T)
        nc = (T + chunk - 1) // chunk
        checkpoints = torch.empty(
            (BH, nc, K, V), device=q.device, dtype=torch.float32
        )
        chk_flat = checkpoints.reshape(BH, nc, K * V)

        block_k, block_v, num_v_blocks, num_warps = _tile_config(K, V)

        _forward_kernel[(BH * num_v_blocks,)](
            q_flat,
            k_flat,
            v_flat,
            e_flat,
            g_flat,
            beta_flat,
            gamma_flat,
            out_flat,
            chk_flat,
            T,
            K,
            V,
            scale_f,
            num_v_blocks,
            nc,
            CHUNK=chunk,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            num_warps=num_warps,
            num_stages=2,
        )

        ctx.save_for_backward(q, k, v, e, g, beta, gamma, checkpoints)
        ctx.scale_f = scale_f
        ctx.chunk = chunk
        ctx.nc = nc
        ctx.block_k = block_k
        ctx.block_v = block_v
        ctx.num_v_blocks = num_v_blocks
        ctx.num_warps = num_warps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma, checkpoints = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        chunk = ctx.chunk
        nc = ctx.nc

        q_flat = q.contiguous().reshape(BH, T, K)
        k_flat = k.contiguous().reshape(BH, T, K)
        e_flat = e.contiguous().reshape(BH, T, K)
        g_flat = g.contiguous().reshape(BH, T, K)
        v_flat = v.contiguous().reshape(BH, T, V)
        beta_flat = beta.contiguous().reshape(BH, T)
        gamma_flat = gamma.contiguous().reshape(BH, T)
        go_flat = grad_out.contiguous().reshape(BH, T, V)
        chk_flat = checkpoints.reshape(BH, nc, K * V)

        dev = q.device
        gq = torch.zeros((B, H, T, K), device=dev, dtype=torch.float32)
        gk = torch.zeros((B, H, T, K), device=dev, dtype=torch.float32)
        ge = torch.zeros((B, H, T, K), device=dev, dtype=torch.float32)
        gg = torch.zeros((B, H, T, K), device=dev, dtype=torch.float32)
        gv = torch.zeros((B, H, T, V), device=dev, dtype=torch.float32)
        gbeta = torch.zeros((B, H, T), device=dev, dtype=torch.float32)
        ggamma = torch.zeros((B, H, T), device=dev, dtype=torch.float32)

        scratch = torch.empty(
            (BH * chunk * K * V,), device=dev, dtype=torch.float32
        )

        _backward_kernel[(BH * ctx.num_v_blocks,)](
            q_flat,
            k_flat,
            v_flat,
            e_flat,
            g_flat,
            beta_flat,
            gamma_flat,
            go_flat,
            chk_flat,
            scratch,
            gq.reshape(BH, T, K),
            gk.reshape(BH, T, K),
            gv.reshape(BH, T, V),
            ge.reshape(BH, T, K),
            gg.reshape(BH, T, K),
            gbeta.reshape(BH, T),
            ggamma.reshape(BH, T),
            T,
            K,
            V,
            ctx.scale_f,
            ctx.num_v_blocks,
            nc,
            CHUNK=chunk,
            BLOCK_K=ctx.block_k,
            BLOCK_V=ctx.block_v,
            num_warps=ctx.num_warps,
            num_stages=2,
        )

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
    B, H, T, K = q.shape
    V = v.shape[-1]
    if B == 0 or H == 0 or T == 0 or V == 0 or K == 0:
        return torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    return _RecurrentFunction.apply(q, k, v, e, g, beta, gamma, scale)
