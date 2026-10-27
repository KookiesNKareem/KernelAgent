```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Generic forward kernels (fallback for shapes not handled by the fast path)
# ---------------------------------------------------------------------------

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
        U =