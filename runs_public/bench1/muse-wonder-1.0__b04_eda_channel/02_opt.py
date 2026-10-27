import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_transition_kernel(
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    aug_ptr,
    T,
    K_DIM,
    V_DIM,
    W_DIM,
    NC,
    NUM_W_BLOCKS,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_w = pid % NUM_W_BLOCKS
    tmp = pid // NUM_W_BLOCKS
    pid_chunk = tmp % NC
    pid_bh = tmp // NC

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    offs_k = tl.arange(0, BLOCK_K)
    mask_w = offs_w < W_DIM
    mask_k = offs_k < K_DIM

    is_trans = offs_w < K_DIM
    col_v = tl.maximum(offs_w - K_DIM, 0)
    mask_vcol = (offs_w >= K_DIM) & (offs_w < K_DIM + V_DIM)

    # Columns [0, K) hold the chunk transition matrix P (initially I).
    # Columns [K, K+V) hold the chunk contribution B (initially 0, driven by v).
    # h is the transposed augmented state, shape [W, K].
    h = (is_trans[:, None] & (offs_k[None, :] == offs_w[:, None])).to(tl.float32)

    bh = pid_bh.to(tl.int64)
    base_k = bh * T * K_DIM
    base_v = bh * T * V_DIM
    base_s = bh * T
    base_aug = ((bh * NC + pid_chunk) * K_DIM) * W_DIM

    t_start = pid_chunk * CHUNK
    remaining = T - t_start
    chunk_len = CHUNK
    if remaining < CHUNK:
        chunk_len = remaining

    for i in range(chunk_len):
        t = t_start + i
        off_k = base_k + t * K_DIM + offs_k

        k_vec = tl.load(k_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        e_vec = tl.load(e_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        g_vec = tl.load(g_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(
            v_ptr + base_v + t * V_DIM + col_v, mask=mask_vcol, other=0.0
        ).to(tl.float32)
        beta_t = tl.load(beta_ptr + base_s + t).to(tl.float32)
        gamma_t = tl.load(gamma_ptr + base_s + t).to(tl.float32)

        h = h * tl.exp(g_vec)[None, :]

        erased = tl.sum(h * e_vec[None, :], axis=1)
        h = h - (gamma_t * erased)[:, None] * e_vec[None, :]

        pred = tl.sum(h * k_vec[None, :], axis=1)
        delta = beta_t * (v_vec - pred)
        h = h + delta[:, None] * k_vec[None, :]

    tl.store(
        aug_ptr + base_aug + offs_k[None, :] * W_DIM + offs_w[:, None],
        h,
        mask=mask_k[None, :] & mask_w[:, None],
    )


@triton.jit
def _chunk_boundary_kernel(
    aug_ptr,
    starts_ptr,
    K_DIM,
    V_DIM,
    W_DIM,
    NC,
    NUM_V_BLOCKS,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_v = pid % NUM_V_BLOCKS
    pid_bh = pid // NUM_V_BLOCKS

    offs_kr = tl.arange(0, BLOCK_K)
    offs_kc = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_kr = offs_kr < K_DIM
    mask_kc = offs_kc < K_DIM
    mask_v = offs_v < V_DIM

    bh = pid_bh.to(tl.int64)

    # State in the ordinary [K, V] orientation for a tensor-core product.
    s_state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for chunk in range(NC):
        base_aug = ((bh * NC + chunk) * K_DIM) * W_DIM
        base_start = ((bh * NC + chunk) * K_DIM) * V_DIM

        tl.store(
            starts_ptr + base_start + offs_kr[:, None] * V_DIM + offs_v[None, :],
            s_state,
            mask=mask_kr[:, None] & mask_v[None, :],
        )

        p_tile = tl.load(
            aug_ptr + base_aug + offs_kr[:, None] * W_DIM + offs_kc[None, :],
            mask=mask_kr[:, None] & mask_kc[None, :],
            other=0.0,
        ).to(tl.float32)
        b_tile = tl.load(
            aug_ptr + base_aug + offs_kr[:, None] * W_DIM + (K_DIM + offs_v)[None, :],
            mask=mask_kr[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)

        s_state = tl.dot(p_tile, s_state, input_precision="ieee") + b_tile


@triton.jit
def _chunk_output_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    starts_ptr,
    out_ptr,
    T,
    K_DIM,
    V_DIM,
    NC,
    NUM_V_BLOCKS,
    scale,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_v = pid % NUM_V_BLOCKS
    tmp = pid // NUM_V_BLOCKS
    pid_chunk = tmp % NC
    pid_bh = tmp // NC

    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_k = tl.arange(0, BLOCK_K)
    mask_v = offs_v < V_DIM
    mask_k = offs_k < K_DIM

    bh = pid_bh.to(tl.int64)
    base_k = bh * T * K_DIM
    base_v = bh * T * V_DIM
    base_s = bh * T
    base_start = ((bh * NC + pid_chunk) * K_DIM) * V_DIM

    # Transposed state [V, K], loaded from this chunk's starting state.
    h = tl.load(
        starts_ptr + base_start + offs_k[None, :] * V_DIM + offs_v[:, None],
        mask=mask_k[None, :] & mask_v[:, None],
        other=0.0,
    ).to(tl.float32)

    t_start = pid_chunk * CHUNK
    remaining = T - t_start
    chunk_len = CHUNK
    if remaining < CHUNK:
        chunk_len = remaining

    for i in range(chunk_len):
        t = t_start + i
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
def _recurrent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    out_ptr,
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

    h = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)

    bh = pid_bh.to(tl.int64)
    base_k = bh * T * K_DIM
    base_v = bh * T * V_DIM
    base_s = bh * T

    for t in range(T):
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


def _next_power_of_2(x):
    n = 1
    while n < x:
        n *= 2
    return n


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0 or V == 0:
        return out

    q_flat = q.contiguous().reshape(B * H, T, K)
    k_flat = k.contiguous().reshape(B * H, T, K)
    e_flat = e.contiguous().reshape(B * H, T, K)
    g_flat = g.contiguous().reshape(B * H, T, K)
    v_flat = v.contiguous().reshape(B * H, T, V)
    beta_flat = beta.contiguous().reshape(B * H, T)
    gamma_flat = gamma.contiguous().reshape(B * H, T)
    out_flat = out.reshape(B * H, T, V)

    block_k = max(16, _next_power_of_2(max(1, K)))
    BH = B * H

    # Chunk-parallel exact path: short sequential depth, many more programs.
    if K <= 128 and K + V <= 512 and T >= 64:
        CHUNK = 128
        BLOCK_W = 64
        BLOCK_V = 64
        NC = (T + CHUNK - 1) // CHUNK
        W_DIM = K + V
        num_w_blocks = (W_DIM + BLOCK_W - 1) // BLOCK_W
        num_v_blocks = (V + BLOCK_V - 1) // BLOCK_V

        aug = torch.empty((BH, NC, K, W_DIM), device=q.device, dtype=torch.float32)
        starts = torch.empty((BH, NC, K, V), device=q.device, dtype=torch.float32)
        aug_flat = aug.reshape(BH, NC, K * W_DIM)
        starts_flat = starts.reshape(BH, NC, K * V)

        _chunk_transition_kernel[(BH * NC * num_w_blocks,)](
            k_flat,
            v_flat,
            e_flat,
            g_flat,
            beta_flat,
            gamma_flat,
            aug_flat,
            T,
            K,
            V,
            W_DIM,
            NC,
            num_w_blocks,
            CHUNK=CHUNK,
            BLOCK_K=block_k,
            BLOCK_W=BLOCK_W,
            num_warps=8,
            num_stages=2,
        )

        _chunk_boundary_kernel[(BH * num_v_blocks,)](
            aug_flat,
            starts_flat,
            K,
            V,
            W_DIM,
            NC,
            num_v_blocks,
            BLOCK_K=block_k,
            BLOCK_V=BLOCK_V,
            num_warps=8,
            num_stages=2,
        )

        _chunk_output_kernel[(BH * NC * num_v_blocks,)](
            q_flat,
            k_flat,
            v_flat,
            e_flat,
            g_flat,
            beta_flat,
            gamma_flat,
            starts_flat,
            out_flat,
            T,
            K,
            V,
            NC,
            num_v_blocks,
            float(scale),
            CHUNK=CHUNK,
            BLOCK_K=block_k,
            BLOCK_V=BLOCK_V,
            num_warps=8,
            num_stages=2,
        )
        return out

    # Fallback for unusual shapes.
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

    _recurrent_kernel[(BH * num_v_blocks,)](
        q_flat,
        k_flat,
        v_flat,
        e_flat,
        g_flat,
        beta_flat,
        gamma_flat,
        out_flat,
        T,
        K,
        V,
        float(scale),
        num_v_blocks,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=num_warps,
        num_stages=2,
    )
    return out
