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
# Forward: register-resident recurrent kernel, checkpointing S every
# CHUNK_LEN steps.
#
# Backward: for each chunk in reverse order, recompute the chunk
# forwards while writing S_{t-1} (the state *before* each step) into a
# small scratch buffer, then walk the chunk backwards.  The backward
# kernel uses only increment/decrement pointer arithmetic, exactly like
# the compiled forward kernel: no per-step dynamically indexed loads,
# no scalar-vector masks, and no dynamic scratch indexing.
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
    do_base = do_ptr + bh * T * V_DIM
    dv_base = dv_ptr + bh * T * V_DIM
    g_base = g_ptr + bh * T
    beta_base = beta_ptr + bh * T
    ckpt_ptrs = (
        ckpt_ptr
        + bh * num_chunks * K_DIM * V_DIM
        + offs_k[:, None] * V_DIM
        + offs_v[None, :]
    )

    part = pid_v.to(tl.int64) * bh_num + bh
    dq_part_base = dq_part_ptr + part * T * K_DIM
    dk_part_base = dk_part_ptr + part * T * K_DIM
    dg_part_base = dg_part_ptr + part * T
    dbeta_part_base = dbeta_part_ptr + part * T

    scratch_ptrs = (
        scratch_ptr
        + (bh * num_v_blocks + pid_v) * chunk_len * K_DIM * BLOCK_V
        + offs_k[:, None] * BLOCK_V
        + offs_vl[None, :]
    )

    # Adjoint of the state coming from steps after the current chunk.
    dstate = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for rev_chunk in range(num_chunks):
        chunk_idx = num_chunks - 1 - rev_chunk
        chunk_start = chunk_idx * chunk_len

        state = tl.load(
            ckpt_ptrs + chunk_idx * K_DIM * V_DIM,
            mask=mask_kv,
            other=0.0,
        ).to(tl.float32)
        state_start = state

        q_ptrs = q_base + chunk_start * K_DIM + offs_k
        k_ptrs = k_base + chunk_start * K_DIM + offs_k
        v_ptrs = v_base + chunk_start * V_DIM + offs_v
        do_ptrs = do_base + chunk_start * V_DIM + offs_v
        dv_ptrs = dv_base + chunk_start * V_DIM + offs_v
        g_ptrs = g_base + chunk_start
        beta_ptrs = beta_base + chunk_start
        dq_ptrs = dq_part_base + chunk_start * K_DIM + offs_k
        dk_ptrs = dk_part_base + chunk_start * K_DIM + offs_k
        dg_ptrs = dg_part_base + chunk_start
        dbeta_ptrs = dbeta_part_base + chunk_start
        scr_ptrs = scratch_ptrs

        # Recompute this chunk forwards, storing S_{t-1} per step.
        for _i in range(chunk_len):
            tl.store(scr_ptrs, state, mask=mask_kvl)

            k_vec = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
            v_vec = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
            g_val = tl.load(g_ptrs).to(tl.float32)
            beta_val = tl.load(beta_ptrs).to(tl.float32)

            state *= tl.exp(g_val)
            pred = tl.sum(state * k_vec[:, None], axis=0)
            delta = (2.0 * beta_val) * (v_vec - pred)
            state += k_vec[:, None] * delta[None, :]

            q_ptrs += K_DIM
            k_ptrs += K_DIM
            v_ptrs += V_DIM
            do_ptrs += V_DIM
            dv_ptrs += V_DIM
            dq_ptrs += K_DIM
            dk_ptrs += K_DIM
            g_ptrs += 1
            beta_ptrs += 1
            dg_ptrs += 1
            dbeta_ptrs += 1
            scr_ptrs += K_DIM * BLOCK_V

        # Step all pointers back to the last step of the chunk.
        q_ptrs -= K_DIM
        k_ptrs -= K_DIM
        v_ptrs -= V_DIM
        do_ptrs -= V_DIM
        dv_ptrs -= V_DIM
        dq_ptrs -= K_DIM
        dk_ptrs -= K_DIM
        g_ptrs -= 1
        beta_ptrs -= 1
        dg_ptrs -= 1
        dbeta_ptrs -= 1
        scr_ptrs -= K_DIM * BLOCK_V

        # Walk this chunk backwards.  Padded tail steps have zero
        # q/k/v/do/g/beta, so they leave state and dstate unchanged
        # and produce zero gradients that the wrapper slices away.
        for _ri in range(chunk_len):
            q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
            k_vec = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
            v_vec = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
            do_vec = tl.load(do_ptrs, mask=mask_v, other=0.0).to(tl.float32)
            g_val = tl.load(g_ptrs).to(tl.float32)
            beta_val = tl.load(beta_ptrs).to(tl.float32)
            s_prev = tl.load(scr_ptrs, mask=mask_kvl, other=0.0).to(tl.float32)

            a_val = tl.exp(g_val)
            beta2 = 2.0 * beta_val
            a_mat = a_val * s_prev
            pred = tl.sum(a_mat * k_vec[:, None], axis=0)
            u_vec = v_vec - pred
            dvec = beta2 * u_vec
            s_t = a_mat + k_vec[:, None] * dvec[None, :]

            dos = scale * do_vec
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

            tl.store(dq_ptrs, dq_vec, mask=mask_k)
            tl.store(dk_ptrs, dk_vec, mask=mask_k)
            tl.store(dv_ptrs, dv_vec, mask=mask_v)
            tl.store(dg_ptrs, dg_val)
            tl.store(dbeta_ptrs, dbeta_val)

            q_ptrs -= K_DIM
            k_ptrs -= K_DIM
            v_ptrs -= V_DIM
            do_ptrs -= V_DIM
            dv_ptrs -= V_DIM
            dq_ptrs -= K_DIM
            dk_ptrs -= K_DIM
            g_ptrs -= 1
            beta_ptrs -= 1
            dg_ptrs -= 1
            dbeta_ptrs -= 1
            scr_ptrs -= K_DIM * BLOCK_V

        # state_start is retained only to keep the checkpoint value
        # live across the chunk; s_prev for the first step comes from
        # scratch slot 0, which stored exactly this tensor.
        dstate += state_start * 0.0


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


def _pad_time(x, t_pad):
    # x: (bh, T, D) -> zero-padded (bh, t_pad, D)
    bh, T, D = x.shape
    out = torch.zeros((bh, t_pad, D), device=x.device, dtype=x.dtype)
    out[:, :T, :] = x
    return out


def _pad_time_1d(x, t_pad):
    bh, T = x.shape
    out = torch.zeros((bh, t_pad), device=x.device, dtype=x.dtype)
    out[:, :T] = x
    return out


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
            ctx.save_for_backward()
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

        ctx.meta = (bh, chunk_len, num_chunks, block_k, block_v,
                    num_v_blocks, warps)

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
        ctx.save_for_backward(q_f, k_f, v_f, g_f, beta_f, checkpoints)
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

        q_f, k_f, v_f, g_f, beta_f, checkpoints = ctx.saved_tensors
        (bh, chunk_len, num_chunks, block_k, block_v,
         num_v_blocks, warps) = ctx.meta
        device = grad_output.device
        t_pad = num_chunks * chunk_len

        go = grad_output.contiguous().view(bh, T, V).to(torch.float32)

        if t_pad == T:
            q_p, k_p, v_p, go_p = q_f, k_f, v_f, go
            g_p, beta_p = g_f, beta_f
        else:
            q_p = _pad_time(q_f, t_pad)
            k_p = _pad_time(k_f, t_pad)
            v_p = _pad_time(v_f, t_pad)
            go_p = _pad_time(go, t_pad)
            g_p = _pad_time_1d(g_f, t_pad)
            beta_p = _pad_time_1d(beta_f, t_pad)

        dq_part = torch.zeros((num_v_blocks, bh, t_pad, K), device=device,
                              dtype=torch.float32)
        dk_part = torch.zeros((num_v_blocks, bh, t_pad, K), device=device,
                              dtype=torch.float32)
        dv_p = torch.zeros((bh, t_pad, V), device=device, dtype=torch.float32)
        dg_part = torch.zeros((num_v_blocks, bh, t_pad), device=device,
                              dtype=torch.float32)
        dbeta_part = torch.zeros((num_v_blocks, bh, t_pad), device=device,
                                 dtype=torch.float32)
        scratch = torch.empty((bh * num_v_blocks, chunk_len, K, block_v),
                              device=device, dtype=torch.float32)

        with torch.cuda.device(device):
            _bwd_kernel[(bh, num_v_blocks)](
                q_p, k_p, v_p, g_p, beta_p, go_p, checkpoints, scratch,
                dq_part, dk_part, dv_p, dg_part, dbeta_part,
                t_pad, ctx.scale, num_chunks, chunk_len, num_v_blocks, bh,
                K, V,
                BLOCK_K=block_k, BLOCK_V=block_v,
                num_warps=warps, num_stages=1,
            )

        dq = dq_part[:, :, :T, :].sum(0).view(B, H, T, K).to(dq_d)
        dk = dk_part[:, :, :T, :].sum(0).view(B, H, T, K).to(dk_d)
        dv = dv_p[:, :T, :].view(B, H, T, V).to(dv_d)
        dg = dg_part[:, :, :T].sum(0).view(B, H, T).to(dg_d)
        dbeta = dbeta_part[:, :, :T].sum(0).view(B, H, T).to(dbeta_d)
        de = torch.zeros((B, H, T, K), device=device, dtype=de_d)
        dgamma = torch.zeros((B, H, T), device=device, dtype=dgamma_d)
        return dq, dk, dv, de, dg, dbeta, dgamma, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _DeltaRuleFunction.apply(q, k, v, e, g, beta, gamma, scale)
