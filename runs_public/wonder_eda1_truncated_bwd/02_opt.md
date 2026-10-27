```python
import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_inverse_kernel(
    k_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    inv_ptr,
    T,
    Kdim,
    num_chunks,
    BLOCK_K: tl.constexpr,
    CHUNK: tl.constexpr,
    LCHUNK: tl.constexpr,
    LOG_STEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // num_chunks
    chunk = pid % num_chunks

    offs_k = tl.arange(0, BLOCK_K)
    offs_l = tl.arange(0, LCHUNK)
    mask_k = offs_k < Kdim

    start = chunk * CHUNK
    t_l = start + (offs_l // 2)
    mask_l = t_l < T
    is_write = (offs_l % 2) == 1

    base = bh * T
    e_ptrs = e_ptr + ((base + t_l)[:, None] * Kdim + offs_k[None, :])
    k_ptrs = k_ptr + ((base + t_l)[:, None] * Kdim + offs_k[None, :])
    e_tile = tl.load(
        e_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0
    ).to(tl.float32)
    k_tile = tl.load(
        k_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0
    ).to(tl.float32)
    U = tl.where(is_write[:, None], k_tile, e_tile)

    g_l = tl.load(g_ptr + base + t_l, mask=mask_l, other=0.0).to(tl.float32)
    beta_l = tl.load(beta_ptr + base + t_l, mask=mask_l, other=0.0).to(tl.float32)
    gamma_l = tl.load(gamma_ptr + base + t_l, mask=mask_l, other=0.0).to(tl.float32)
    b_l = tl.where(is_write, beta_l, gamma_l)
    g_step = tl.where(is_write, 0.0, g_l)
    cum_l = tl.cumsum(g_step, axis=0)

    A = tl.dot(U, tl.trans(U), input_precision="tf32")

    row = offs_l[:, None]
    col = offs_l[None, :]
    strict = col < row
    diff = cum_l[:, None] - cum_l[None, :]
    ratio = tl.where(strict, tl.exp(tl.where(strict, diff, 0.0)), 0.0)
    # Matrix whose inverse (via Neumann series below) solves the
    # intra-chunk delta-rule triangular system.
    Nmat = -(b_l[:, None] * A) * ratio

    inv_mat = tl.where(row == col, 1.0, 0.0)
    pow_mat = Nmat
    for _ in range(LOG_STEPS - 1):
        inv_mat = inv_mat + tl.dot(pow_mat, inv_mat, input_precision="tf32")
        pow_mat = tl.dot(pow_mat, pow_mat, input_precision="tf32")
    inv_mat = inv_mat + tl.dot(pow_mat, inv_mat, input_precision="tf32")

    inv_base = ((bh * num_chunks + chunk) * LCHUNK) * LCHUNK
    inv_ptrs = inv_ptr + inv_base + offs_l[:, None] * LCHUNK + offs_l[None, :]
    tl.store(inv_ptrs, inv_mat)


@triton.jit
def _chunk_apply_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    inv_ptr,
    o_ptr,
    T,
    Kdim,
    Vdim,
    scale,
    num_chunks,
    num_v_blocks,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    CHUNK: tl.constexpr,
    LCHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // num_v_blocks
    vb = pid % num_v_blocks

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_c = tl.arange(0, CHUNK)
    offs_l = tl.arange(0, LCHUNK)
    mask_k = offs_k < Kdim
    mask_v = offs_v < Vdim

    base = bh * T
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for chunk in range(num_chunks):
        start = chunk * CHUNK

        # Expanded intra-chunk sequence: erase_0, write_0, erase_1, ...
        t_l = start + (offs_l // 2)
        mask_l = t_l < T
        is_write = (offs_l % 2) == 1

        e_ptrs = e_ptr + ((base + t_l)[:, None] * Kdim + offs_k[None, :])
        k_ptrs = k_ptr + ((base + t_l)[:, None] * Kdim + offs_k[None, :])
        e_tile = tl.load(
            e_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)
        k_tile = tl.load(
            k_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)
        U = tl.where(is_write[:, None], k_tile, e_tile)

        v_ptrs = v_ptr + ((base + t_l)[:, None] * Vdim + offs_v[None, :])
        v_tile = tl.load(
            v_ptrs, mask=mask_l[:, None] & mask_v[None, :], other=0.0
        ).to(tl.float32)
        target = tl.where(is_write[:, None], v_tile, 0.0)

        g_l = tl.load(g_ptr + base + t_l, mask=mask_l, other=0.0).to(tl.float32)
        beta_l = tl.load(beta_ptr + base + t_l, mask=mask_l, other=0.0).to(
            tl.float32
        )
        gamma_l = tl.load(gamma_ptr + base + t_l, mask=mask_l, other=0.0).to(
            tl.float32
        )
        b_l = tl.where(is_write, beta_l, gamma_l)
        g_step = tl.where(is_write, 0.0, g_l)
        cum_l = tl.cumsum(g_step, axis=0)
        lam_l = tl.exp(cum_l)
        total_g = tl.sum(tl.where(is_write, 0.0, g_l), axis=0)

        inv_base = ((bh * num_chunks + chunk) * LCHUNK) * LCHUNK
        inv_ptrs = inv_ptr + inv_base + offs_l[:, None] * LCHUNK + offs_l[None, :]
        inv_mat = tl.load(inv_ptrs)

        # Solve for the scaled residuals of all erase/write steps at once.
        US = tl.dot(U, S, input_precision="tf32")
        rhs = b_l[:, None] * (target - lam_l[:, None] * US)
        W = tl.dot(inv_mat, rhs, input_precision="tf32")

        # Outputs for the ordinary (post-write) timesteps in this chunk.
        t_c = start + offs_c
        mask_c = t_c < T
        q_ptrs = q_ptr + ((base + t_c)[:, None] * Kdim + offs_k[None, :])
        q_tile = tl.load(
            q_ptrs, mask=mask_c[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)
        g_c = tl.load(g_ptr + base + t_c, mask=mask_c, other=0.0).to(tl.float32)
        cg = tl.cumsum(g_c, axis=0)

        QS = tl.dot(q_tile, S, input_precision="tf32")
        QUT = tl.dot(q_tile, tl.trans(U), input_precision="tf32")

        ready = offs_l[None, :] <= (2 * offs_c[:, None] + 1)
        diff_o = cg[:, None] - cum_l[None, :]
        ratio_o = tl.where(ready, tl.exp(tl.where(ready, diff_o, 0.0)), 0.0)
        out_tile = (
            tl.exp(cg)[:, None] * QS
            + tl.dot(QUT * ratio_o, W, input_precision="tf32")
        ) * scale

        o_ptrs = o_ptr + ((base + t_c)[:, None] * Vdim + offs_v[None, :])
        tl.store(o_ptrs, out_tile, mask=mask_c[:, None] & mask_v[None, :])

        # Carry the exact chunk-end state forward.
        end_ratio = tl.exp(total_g - cum_l)
        S = tl.exp(total_g) * S + tl.dot(
            tl.trans(U), W * end_ratio[:, None], input_precision="tf32"
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    BH = B * H
    out_device = q.device

    if not q.is_cuda:
        q, k, v, e, g, beta, gamma = (
            x.cuda() for x in (q, k, v, e, g, beta, gamma)
        )

    q2 = q.reshape(BH, T, K).contiguous()
    k2 = k.reshape(BH, T, K).contiguous()
    e2 = e.reshape(BH, T, K).contiguous()
    v2 = v.reshape(BH, T, V).contiguous()
    g2 = g.reshape(BH, T).contiguous()
    beta2 = beta.reshape(BH, T).contiguous()
    gamma2 = gamma.reshape(BH, T).contiguous()

    out = torch.empty((BH, T, V), device=q2.device, dtype=torch.float32)

    CHUNK = 16
    LCHUNK = 32
    LOG_STEPS = 5  # 2**5 == LCHUNK
    num_chunks = triton.cdiv(T, CHUNK)

    BLOCK_K = max(16, triton.next_power_of_2(K))
    BLOCK_V = 64 if V > 64 else max(16, triton.next_power_of_2(V))
    num_v_blocks = triton.cdiv(V, BLOCK_V)

    inv = torch.empty(
        (BH * num_chunks * LCHUNK * LCHUNK,), device=q2.device, dtype=torch.float32
    )

    _chunk_inverse_kernel[(BH * num_chunks,)](
        k2,
        e2,
        g2,
        beta2,
        gamma2,
        inv,
        T,
        K,
        num_chunks,
        BLOCK_K=BLOCK_K,
        CHUNK=CHUNK,
        LCHUNK=LCHUNK,
        LOG_STEPS=LOG_STEPS,
        num_warps=4,
        num_stages=2,
    )

    _chunk_apply_kernel[(BH * num_v_blocks,)](
        q2,
        k2,
        v2,
        e2,
        g2,
        beta2,
        gamma2,
        inv,
        out,
        T,
        K,
        V,
        float(scale),
        num_chunks,
        num_v_blocks,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        CHUNK=CHUNK,
        LCHUNK=LCHUNK,
        num_warps=8,
        num_stages=2,
    )

    result = out.reshape(B, H, T, V)
    if result.device != out_device:
        result = result.to(out_device)
    return result
```

The main change is splitting the work into two kernels: a highly parallel first kernel precomputes each 16-step chunk's triangular-system inverse once per batch-head (instead of once per value block), using a smaller expanded system of size 32 whose Neumann inverse is far cheaper, and all matrix products now use TF32 tensor cores instead of IEEE FP32. The second kernel then only needs one inverse application per chunk while carrying the state, rather than redoing the inversion for every value block. I am unsure whether the smaller chunk's lower per-dot efficiency or the extra global-memory traffic for the inverses could offset part of the gain on the L4, since I cannot benchmark the tuning.