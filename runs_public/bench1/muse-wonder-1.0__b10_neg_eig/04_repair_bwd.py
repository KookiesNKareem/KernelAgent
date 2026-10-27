import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Gated delta rule (reference semantics):
#
#   S <- exp(g_t) * S
#   pred = S^T k_t
#   S <- S + k_t * (2*beta_t*(v_t - pred))^T
#   o_t = scale * S^T q_t
#
# Inputs e and gamma do not occur in the reference, so their gradients
# are exactly zero.
#
# This implementation deliberately uses only register-resident
# recurrent kernels (no large chunk matrices / tl.dot factorisations).
# The forward kernel checkpoints the state every CHUNK_LEN steps.  The
# backward kernel, for every chunk in reverse order, first recomputes
# that chunk's per-step states into a small per-programme scratch buffer
# and then walks the chunk backwards applying the exact adjoint of one
# delta-rule step.
# ---------------------------------------------------------------------------


@triton.jit
def _fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    out_ptr,
    ckpt_ptr,
    T,
    scale,
    chunk_len,
    K_DIM,
    V_DIM,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K_DIM
    mask_v = offs_v < V_DIM

    bh = pid_bh.to(tl.int64)

    q_ptrs = q_ptr + bh * T * K_DIM + offs_k
    k_ptrs = k_ptr + bh * T * K_DIM + offs_k
    v_ptrs = v_ptr + bh * T * V_DIM + offs_v
    out_ptrs = out_ptr + bh * T * V_DIM + offs_v
    g_ptrs = g_ptr + bh * T
    beta_ptrs = beta_ptr + bh * T
    ckpt_base = ckpt_ptr + bh * ((T + chunk_len - 1) // chunk_len) * K_DIM * V_DIM

    state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in range(T):
        # Checkpoint S_start of each chunk for the backward pass.
        is_start = (t % chunk_len) == 0
        ckpt_idx = t // chunk_len
        tl.store(
            ckpt_base
            + ckpt_idx * K_DIM * V_DIM
            + offs_k[:, None] * V_DIM
            + offs_v[None, :],
            state,
            mask=is_start & mask_k[:, None] & mask_v[None, :],
        )

        decay = tl.exp(tl.load(g_ptrs))
        beta_val = tl.load(beta_ptrs)
        k_vec = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)

        state *= decay
        pred = tl.sum(state * k_vec[:, None], axis=0)
        delta = (2.0 * beta_val) * (v_vec - pred)
        state += k_vec[:, None] * delta[None, :]

        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        out_vec = tl.sum(state * (q_vec * scale)[:, None], axis=0)
        tl.store(out_ptrs, out_vec, mask=mask_v)

        q_ptrs += K_DIM
        k_ptrs += K_DIM
        v_ptrs += V_DIM
        out_ptrs += V_DIM
        g_ptrs += 1
        beta_ptrs += 1


@triton.jit
def _bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    do_ptr,
    ckpt_ptr,
    scratch_ptr,
    dq_part_ptr,
    dk_part_ptr,
    dv_ptr,
    dg_part_ptr,
    dbeta_part_ptr,
    T,
    scale,
    num_chunks,
    chunk_len,
    num_v_blocks,
    bh_num,
    K_DIM,
    V_DIM,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_vl = tl.arange(0, BLOCK_V)
    offs_v = pid_v * BLOCK_V + offs_vl
    mask_k = offs_k < K_DIM
    mask_v = offs_v < V_DIM
    mask_kv = mask_k[:, None] & mask_v[None, :]
    mask_kvl = mask_k[:, None] & (offs_vl < BLOCK_V)

    bh = pid_bh.to(tl.int64)
    q_base = q_ptr + bh * T * K_DIM
    k_base = k_ptr + bh * T * K_DIM
    v_base = v_ptr + bh * T * V_DIM
    t_base = bh * T
    ckpt_base = ckpt_ptr + bh * num_chunks * K_DIM * V_DIM

    part = pid_v.to(tl.int64) * bh_num + bh
    part_qk = dq_part_ptr + part * T * K_DIM
    part_dk = dk_part_ptr + part * T * K_DIM
    part_t = part * T

    scratch_base = (
        scratch_ptr
        + (bh * num_v_blocks + pid_v) * chunk_len * K_DIM * BLOCK_V
    )

    # Adjoint of the state coming from steps after the current chunk.
    dstate = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for rev_chunk in range(num_chunks):
        chunk_idx = num_chunks - 1 - rev_chunk
        chunk_start = chunk_idx * chunk_len

        state_start = tl.load(
            ckpt_base
            + chunk_idx * K_DIM * V_DIM
            + offs_k[:, None] * V_DIM
            + offs_v[None, :],
            mask=mask_kv,
            other=0.0,
        ).to(tl.float32)

        # Recompute this chunk forwards, storing S_t in scratch.
        state = state_start
        for i in range(chunk_len):
            t = chunk_start + i
            mask_t = t < T
            k_vec = tl.load(
                k_base + t * K_DIM + offs_k,
                mask=mask_t & mask_k,
                other=0.0,
            ).to(tl.float32)
            v_vec = tl.load(
                v_base + t * V_DIM + offs_v,
                mask=mask_t & mask_v,
                other=0.0,
            ).to(tl.float32)
            g_val = tl.load(g_ptr + t_base + t, mask=mask_t, other=0.0).to(
                tl.float32
            )
            beta_val = tl.load(
                beta_ptr + t_base + t, mask=mask_t, other=0.0
            ).to(tl.float32)

            state *= tl.exp(g_val)
            pred = tl.sum(state * k_vec[:, None], axis=0)
            delta = (2.0 * beta_val) * (v_vec - pred)
            state += k_vec[:, None] * delta[None, :]

            tl.store(
                scratch_base
                + i * K_DIM * BLOCK_V
                + offs_k[:, None] * BLOCK_V
                + offs_vl[None, :],
                state,
                mask=mask_kvl,
            )

        # Walk this chunk backwards.
        for rev_i in range(chunk_len):
            i = chunk_len - 1 - rev_i
            t = chunk_start + i
            mask_t = t < T

            q_vec = tl.load(
                q_base + t * K_DIM + offs_k,
                mask=mask_t & mask_k,
                other=0.0,
            ).to(tl.float32)
            k_vec = tl.load(
                k_base + t * K_DIM + offs_k,
                mask=mask_t & mask_k,
                other=0.0,
            ).to(tl.float32)
            v_vec = tl.load(
                v_base + t * V_DIM + offs_v,
                mask=mask_t & mask_v,
                other=0.0,
            ).to(tl.float32)
            do_vec = tl.load(
                do_ptr + v_base + t * V_DIM + offs_v,
                mask=mask_t & mask_v,
                other=0.0,
            ).to(tl.float32)
            g_val = tl.load(g_ptr + t_base + t, mask=mask_t, other=0.0).to(
                tl.float32
            )
            beta_val = tl.load(
                beta_ptr + t_base + t, mask=mask_t, other=0.0
            ).to(tl.float32)

            s_t = tl.load(
                scratch_base
                + i * K_DIM * BLOCK_V
                + offs_k[:, None] * BLOCK_V
                + offs_vl[None, :],
                mask=mask_kvl,
                other=0.0,
            ).to(tl.float32)

            prev_i = i - 1
            prev_i = tl.where(prev_i < 0, 0, prev_i)
            s_prev_load = tl.load(
                scratch_base
                + prev_i * K_DIM * BLOCK_V
                + offs_k[:, None] * BLOCK_V
                + offs_vl[None, :],
                mask=mask_kvl,
                other=0.0,
            ).to(tl.float32)
            s_prev = tl.where(i == 0, state_start, s_prev_load)

            a_val = tl.exp(g_val)
            beta2 = 2.0 * beta_val
            a_mat = a_val * s_prev
            pred = tl.sum(a_mat * k_vec[:, None], axis=0)
            u_vec = v_vec - pred
            dvec = beta2 * u_vec

            dos = scale * do_vec
            # Total adjoint w.r.t. S_t (future + this step's output).
            d_mat = dstate + q_vec[:, None] * dos[None, :]

            dq_vec = tl.sum(s_t * dos[None, :], axis=1)
            dd_vec = tl.sum(d_mat * k_vec[:, None], axis=0)
            dpred = -beta2 * dd_vec

            dk_vec = tl.sum(d_mat * dvec[None, :], axis=1)
            dk_vec += tl.sum(a_mat * dpred[None, :], axis=1)
            dv_vec = beta2 * dd_vec
            dbeta_val = 2.0 * tl.sum(dd_vec * u_vec)

            da_mat = d_mat + k_vec[:, None] * dpred[None, :]
            dg_val = a_val * tl.sum(da_mat * s_prev)
            dstate = a_val * da_mat

            tl.store(
                part_qk + t * K_DIM + offs_k,
                dq_vec,
                mask=mask_t & mask_k,
            )
            tl.store(
                part_dk + t * K_DIM + offs_k,
                dk_vec,
                mask=mask_t & mask_k,
            )
            tl.store(
                dv_ptr + v_base + t * V_DIM + offs_v,
                dv_vec,
                mask=mask_t & mask_v,
            )
            tl.store(dg_part_ptr + part_t + t, dg_val, mask=mask_t)
            tl.store(dbeta_part_ptr + part_t + t, dbeta_val, mask=mask_t)


def _next_power_of_2(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _ceil_div(a, b):
    return (a + b - 1) // b


def _config(bh, K, V):
    block_k = max(16, _next_power_of_2(K))
    if block_k <= 32:
        cap = 128
    elif block_k <= 64:
        cap = 64
    elif block_k <= 128:
        cap = 32
    elif block_k <= 256:
        cap = 16
    else:
        cap = 8
    block_v = min(cap, max(16, _next_power_of_2(V)))
    if V < block_v:
        block_v = max(16, _next_power_of_2(V))
    # Keep enough programmes for small batch*head counts.
    while block_v > 16 and bh * _ceil_div(V, block_v) < 32:
        block_v //= 2
    tile = block_k * block_v
    if tile >= 4096:
        warps = 8
    elif tile >= 1024:
        warps = 4
    else:
        warps = 2
    return block_k, block_v, warps


class _DeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        ctx.out_shape = (B, H, T, V)
        ctx.k_shape = (B, H, T, K)
        ctx.t_shape = (B, H, T)
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, e.dtype, g.dtype,
                      beta.dtype, gamma.dtype)
        ctx.scale = float(scale)
        ctx.device = q.device
        ctx.is_empty = (B == 0 or H == 0 or T == 0 or K == 0 or V == 0)

        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        if ctx.is_empty:
            ctx._tensors = ()
            return out

        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        g_c = g.contiguous()
        beta_c = beta.contiguous()

        bh = B * H
        q_f = q_c.view(bh, T, K)
        k_f = k_c.view(bh, T, K)
        v_f = v_c.view(bh, T, V)
        g_f = g_c.view(bh, T)
        beta_f = beta_c.view(bh, T)
        out_f = out.view(bh, T, V)

        chunk_len = 64
        num_chunks = _ceil_div(T, chunk_len)
        block_k, block_v, warps = _config(bh, K, V)
        num_v_blocks = _ceil_div(V, block_v)

        ctx.meta = (bh, T, K, V, chunk_len, num_chunks, block_k,
                    block_v, num_v_blocks, warps)

        checkpoints = torch.empty(
            (bh, num_chunks, K, V), device=q.device, dtype=torch.float32
        )
        with torch.cuda.device(q.device):
            _fwd_kernel[(bh, num_v_blocks)](
                q_f, k_f, v_f, g_f, beta_f, out_f, checkpoints,
                T, float(scale), chunk_len, K, V,
                BLOCK_K=block_k, BLOCK_V=block_v,
                num_warps=warps, num_stages=1,
            )
        ctx._tensors = (q_f, k_f, v_f, g_f, beta_f, checkpoints)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        dq_d, dk_d, dv_d, de_d, dg_d, dbeta_d, dgamma_d = ctx.dtypes
        B, H, T, V = ctx.out_shape
        _, _, _, K = ctx.k_shape
        if ctx.is_empty:
            return (
                torch.zeros(ctx.k_shape, device=ctx.device, dtype=dq_d),
                torch.zeros(ctx.k_shape, device=ctx.device, dtype=dk_d),
                torch.zeros(ctx.out_shape, device=ctx.device, dtype=dv_d),
                torch.zeros(ctx.k_shape, device=ctx.device, dtype=de_d),
                torch.zeros(ctx.t_shape, device=ctx.device, dtype=dg_d),
                torch.zeros(ctx.t_shape, device=ctx.device, dtype=dbeta_d),
                torch.zeros(ctx.t_shape, device=ctx.device, dtype=dgamma_d),
                None,
            )

        q_f, k_f, v_f, g_f, beta_f, checkpoints = ctx._tensors
        (bh, T_, K_, V_, chunk_len, num_chunks, block_k, block_v,
         num_v_blocks, warps) = ctx.meta
        device = grad_output.device

        go = grad_output.contiguous().view(bh, T, V).to(torch.float32)

        dq_part = torch.zeros((num_v_blocks, bh, T, K), device=device,
                              dtype=torch.float32)
        dk_part = torch.zeros((num_v_blocks, bh, T, K), device=device,
                              dtype=torch.float32)
        dv_f = torch.zeros((bh, T, V), device=device, dtype=torch.float32)
        dg_part = torch.zeros((num_v_blocks, bh, T), device=device,
                              dtype=torch.float32)
        dbeta_part = torch.zeros((num_v_blocks, bh, T), device=device,
                                 dtype=torch.float32)
        scratch = torch.empty((bh, num_v_blocks, chunk_len, K, block_v),
                              device=device, dtype=torch.float32)

        with torch.cuda.device(device):
            _bwd_kernel[(bh, num_v_blocks)](
                q_f, k_f, v_f, g_f, beta_f, go, checkpoints, scratch,
                dq_part, dk_part, dv_f, dg_part, dbeta_part,
                T, ctx.scale, num_chunks, chunk_len, num_v_blocks, bh,
                K, V,
                BLOCK_K=block_k, BLOCK_V=block_v,
                num_warps=warps, num_stages=1,
            )

        dq = dq_part.sum(0).view(B, H, T, K).to(dq_d)
        dk = dk_part.sum(0).view(B, H, T, K).to(dk_d)
        dv = dv_f.view(B, H, T, V).to(dv_d)
        dg = dg_part.sum(0).view(B, H, T).to(dg_d)
        dbeta = dbeta_part.sum(0).view(B, H, T).to(dbeta_d)
        de = torch.zeros((B, H, T, K), device=device, dtype=de_d)
        dgamma = torch.zeros((B, H, T), device=device, dtype=dgamma_d)
        return dq, dk, dv, de, dg, dbeta, dgamma, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _DeltaRuleFunction.apply(q, k, v, e, g, beta, gamma, scale)
