```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Generic kernels (fallback for shapes not handled by the fast path)
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
    sb_ptr,
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
        # Save the incoming state; the backward pass restarts from here.
        sb_ptrs = sb_ptr + (
            ((bh * num_chunks + chunk) * Kdim + offs_k[:, None]) * Vdim
            + offs_v[None, :]
        )
        tl.store(sb_ptrs, S, mask=mask_k[:, None] & mask_v[None, :])

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
        total_g = tl.sum(g_step, axis=0)

        inv_base = ((bh * num_chunks + chunk) * LCHUNK) * LCHUNK
        inv_ptrs = inv_ptr + inv_base + offs_l[:, None] * LCHUNK + offs_l[None, :]
        inv_mat = tl.load(inv_ptrs)

        US = tl.dot(U, S, input_precision="tf32")
        rhs = b_l[:, None] * (target - lam_l[:, None] * US)
        W = tl.dot(inv_mat, rhs, input_precision="tf32")

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

        end_ratio = tl.exp(total_g - cum_l)
        S = tl.exp(total_g) * S + tl.dot(
            tl.trans(U), W * end_ratio[:, None], input_precision="tf32"
        )


@triton.jit
def _chunk_backward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    inv_ptr,
    sb_ptr,
    go_ptr,
    dq_ptr,
    dk_ptr,
    de_ptr,
    dv_ptr,
    dg_ptr,
    dbeta_ptr,
    dgamma_ptr,
    BH,
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
    row_t = offs_l // 2
    is_write = (offs_l % 2) == 1

    base = bh * T
    # Flat offsets into the [num_v_blocks, BH, T, *] partial buffers.
    part_base_k = (vb * BH + bh) * T * Kdim
    part_base_1 = (vb * BH + bh) * T

    dS = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)  # adjoint of S_end

    for rev in range(num_chunks):
        chunk = num_chunks - 1 - rev
        start = chunk * CHUNK

        # ----- reload chunk inputs and recompute the forward quantities ---
        t_l = start + row_t
        mask_l = t_l < T

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
        total_g = tl.sum(g_step, axis=0)
        exp_total = tl.exp(total_g)

        sb_ptrs = sb_ptr + (
            ((bh * num_chunks + chunk) * Kdim + offs_k[:, None]) * Vdim
            + offs_v[None, :]
        )
        S0 = tl.load(sb_ptrs, mask=mask_k[:, None] & mask_v[None, :], other=0.0)

        inv_base = ((bh * num_chunks + chunk) * LCHUNK) * LCHUNK
        inv_ptrs = inv_ptr + inv_base + offs_l[:, None] * LCHUNK + offs_l[None, :]
        P = tl.load(inv_ptrs)

        US = tl.dot(U, S0, input_precision="tf32")
        diff = target - lam_l[:, None] * US
        rhs = b_l[:, None] * diff
        W = tl.dot(P, rhs, input_precision="tf32")

        t_c = start + offs_c
        mask_c = t_c < T
        q_ptrs = q_ptr + ((base + t_c)[:, None] * Kdim + offs_k[None, :])
        q_tile = tl.load(
            q_ptrs, mask=mask_c[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)
        go_ptrs = go_ptr + ((base + t_c)[:, None] * Vdim + offs_v[None, :])
        GO = tl.load(
            go_ptrs, mask=mask_c[:, None] & mask_v[None, :], other=0.0
        ).to(tl.float32)
        GO = GO * scale

        g_c = tl.load(g_ptr + base + t_c, mask=mask_c, other=0.0).to(tl.float32)
        cg = tl.cumsum(g_c, axis=0)
        exp_cg = tl.exp(cg)

        QS = tl.dot(q_tile, S0, input_precision="tf32")
        QUT = tl.dot(q_tile, tl.trans(U), input_precision="tf32")

        ready = offs_l[None, :] <= (2 * offs_c[:, None] + 1)
        diff_o = cg[:, None] - cum_l[None, :]
        Ro = tl.where(ready, tl.exp(tl.where(ready, diff_o, 0.0)), 0.0)
        Rend = tl.exp(total_g - cum_l)

        # ----- gradients flowing into W ----------------------------------
        UGsend = tl.dot(U, dS, input_precision="tf32")
        GW = tl.dot(tl.trans(QUT * Ro), GO, input_precision="tf32") + (
            Rend[:, None] * UGsend
        )

        # ----- output-term gradients w.r.t. S0, Q, U, and decay scalars --
        GO_cg = exp_cg[:, None] * GO
        dS0 = tl.dot(tl.trans(q_tile), GO_cg, input_precision="tf32") + (
            exp_total * dS
        )
        GQ = tl.dot(GO_cg, tl.trans(S0), input_precision="tf32")
        GM = tl.dot(GO, tl.trans(W), input_precision="tf32")
        GQUT = GM * Ro
        GQ = GQ + tl.dot(GQUT, U, input_precision="tf32")
        GU = tl.dot(tl.trans(GQUT), q_tile, input_precision="tf32")

        GRo = GM * QUT
        dc_c = exp_cg * tl.sum(GO * QS, axis=1) + tl.sum(GRo * Ro, axis=1)
        dcum = -tl.sum(GRo * Ro, axis=0)

        # ----- chunk-end state term ---------------------------------------
        GRend = tl.sum(UGsend * W, axis=1)
        dcum = dcum - GRend * Rend
        d_total = exp_total * tl.sum(
            tl.sum(dS * S0, axis=0), axis=0
        ) + tl.sum(GRend * Rend, axis=0)
        GU = GU + tl.dot(
            W * Rend[:, None], tl.trans(dS), input_precision="tf32"
        )

        # ----- differentiate through P = (I - N)^-1 ------------------------
        row = offs_l[:, None]
        col = offs_l[None, :]
        strict = col < row
        diff_n = cum_l[:, None] - cum_l[None, :]
        ratio_n = tl.where(strict, tl.exp(tl.where(strict, diff_n, 0.0)), 0.0)

        A = tl.dot(U, tl.trans(U), input_precision="tf32")
        GP = tl.dot(GW, tl.trans(rhs), input_precision="tf32")
        GN = tl.dot(
            tl.trans(P), tl.dot(GP, tl.trans(P), input_precision="tf32"),
            input_precision="tf32",
        )
        GN = tl.where(strict, GN, 0.0)

        neg_bA = -(b_l[:, None] * A)
        GRmat = GN * neg_bA
        Gb = tl.sum(GN * (-(A * ratio_n)), axis=1)
        GA = -(b_l[:, None] * GN * ratio_n)
        dcum = dcum + tl.sum(GRmat * ratio_n, axis=1) - tl.sum(
            GRmat * ratio_n, axis=0
        )
        GU = GU + tl.dot(GA + tl.trans(GA), U, input_precision="tf32")

        # ----- right-hand side ---------------------------------------------
        Grhs = tl.dot(tl.trans(P), GW, input_precision="tf32")
        Gb = Gb + tl.sum(Grhs * diff, axis=1)
        Gdiff = b_l[:, None] * Grhs
        GUS = -lam_l[:, None] * Gdiff
        dcum = dcum - lam_l * tl.sum(Gdiff * US, axis=1)
        dS0 = dS0 + tl.dot(tl.trans(U), GUS, input_precision="tf32")
        GU = GU + tl.dot(GUS, tl.trans(S0), input_precision="tf32")

        # ----- map decay adjoints back to per-timestep g ------------------
        group_mat = ((row_t[None, :] == offs_c[:, None])).to(tl.float32)
        grouped = tl.sum(group_mat * dcum[None, :], axis=1)
        prefix_adj = grouped + dc_c
        suffix_mat = (offs_c[None, :] >= offs_c[:, None]).to(tl.float32)
        dg_c = tl.sum(suffix_mat * prefix_adj[None, :], axis=1) + d_total

        # ----- store gradients ---------------------------------------------
        tl.store(
            dq_ptr + part_base_k + (t_c[:, None] * Kdim + offs_k[None, :]),
            GQ,
            mask=mask_c[:, None] & mask_k[None, :],
        )
        tl.store(
            dk_ptr + part_base_k + (t_l[:, None] * Kdim + offs_k[None, :]),
            GU,
            mask=mask_l[:, None] & mask_k[None, :] & is_write[:, None],
        )
        tl.store(
            de_ptr + part_base_k + (t_l[:, None] * Kdim + offs_k[None, :]),
            GU,
            mask=mask_l[:, None] & mask_k[None, :] & (~is_write)[:, None],
        )
        tl.store(
            dv_ptr + (base + t_l)[:, None] * Vdim + offs_v[None, :],
            Gdiff,
            mask=mask_l[:, None] & mask_v[None, :] & is_write[:, None],
        )
        tl.store(dg_ptr + part_base_1 + t_c, dg_c, mask=mask_c)
        tl.store(
            dbeta_ptr + part_base_1 + t_l, Gb, mask=mask_l & is_write
        )
        tl.store(
            dgamma_ptr + part_base_1 + t_l, Gb, mask=mask_l & (~is_write)
        )

        dS = dS0


# ---------------------------------------------------------------------------
# Fast kernels: K == V == 128, bf16 q/k/v/e, T % 32 == 0.
# Chunk = 32 timesteps = 64 expanded erase/write steps, BLOCK_V = 64.
# Large K/V contractions use bf16 tensor cores with fp32 accumulation and
# an fp32 carried state; the 64x64 system-inverse products stay TF32.
# ---------------------------------------------------------------------------

@triton.jit
def _fast_inverse_kernel(
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
    row_t = offs_l // 2
    is_write = (offs_l % 2) == 1

    t_l = chunk * CHUNK + row_t
    base = bh * T

    e_tile = tl.load(e_ptr + ((base + t_l)[:, None] * Kdim + offs_k[None, :]))
    k_tile = tl.load(k_ptr + ((base + t_l)[:, None] * Kdim + offs_k[None, :]))
    U = tl.where(is_write[:, None], k_tile, e_tile)

    g_l = tl.load(g_ptr + base + t_l).to(tl.float32)
    beta_l = tl.load(beta_ptr + base + t_l).to(tl.float32)
    gamma_l = tl.load(gamma_ptr + base + t_l).to(tl.float32)
    b_l = tl.where(is_write, beta_l, gamma_l)
    g_step = tl.where(is_write, 0.0, g_l)
    cum_l = tl.cumsum(g_step, axis=0)

    A = tl.dot(U, tl.trans(U))

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

    inv_ptrs = (
        inv_ptr
        + ((bh * num_chunks + chunk) * LCHUNK) * LCHUNK
        + offs_l[:, None] * LCHUNK
        + offs_l[None, :]
    )
    tl.store(inv_ptrs, inv_mat)


@triton.jit
def _fast_apply_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    inv_ptr,
    sb_ptr,
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
    row_t = offs_l // 2
    is_write = (offs_l % 2) == 1

    bh_off = bh * T
    q_cur = q_ptr + bh_off * Kdim
    k_cur = k_ptr + bh_off * Kdim
    e_cur = e_ptr + bh_off * Kdim
    v_cur = v_ptr + bh_off * Vdim
    o_cur = o_ptr + bh_off * Vdim
    g_cur = g_ptr + bh_off
    beta_cur = beta_ptr + bh_off
    gamma_cur = gamma_ptr + bh_off
    sb_cur = sb_ptr + bh * num_chunks * Kdim * Vdim
    inv_cur = inv_ptr + bh * num_chunks * LCHUNK * LCHUNK

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for _chunk in range(num_chunks):
        tl.store(
            sb_cur + offs_k[:, None] * Vdim + offs_v[None, :], S
        )

        e_tile = tl.load(e_cur + row_t[:, None] * Kdim + offs_k[None, :])
        k_tile = tl.load(k_cur + row_t[:, None] * Kdim + offs_k[None, :])
        U = tl.where(is_write[:, None], k_tile, e_tile)

        v_tile = tl.load(v_cur + row_t[:, None] * Vdim + offs_v[None, :])
        target = tl.where(is_write[:, None], v_tile, 0.0).to(tl.float32)

        g_l = tl.load(g_cur + row_t).to(tl.float32)
        beta_l = tl.load(beta_cur + row_t).to(tl.float32)
        gamma_l = tl.load(gamma_cur + row_t).to(tl.float32)
        b_l = tl.where(is_write, beta_l, gamma_l)
        g_step = tl.where(is_write, 0.0, g_l)
        cum_l = tl.cumsum(g_step, axis=0)
        lam_l = tl.exp(cum_l)
        total_g = tl.sum(g_step, axis=0)

        P = tl.load(inv_cur + offs_l[:, None] * LCHUNK + offs_l[None, :])

        S_b = S.to(tl.bfloat16)
        US = tl.dot(U, S_b)
        rhs = b_l[:, None] * (target - lam_l[:, None] * US)
        W = tl.dot(P, rhs, input_precision="tf32")
        W_b = W.to(tl.bfloat16)

        q_tile = tl.load(q_cur + offs_c[:, None] * Kdim + offs_k[None, :])
        g_c = tl.load(g_cur + offs_c).to(tl.float32)
        cg = tl.cumsum(g_c, axis=0)

        QS = tl.dot(q_tile, S_b)
        QUT = tl.dot(q_tile, tl.trans(U))

        ready = offs_l[None, :] <= (2 * offs_c[:, None] + 1)
        diff_o = cg[:, None] - cum_l[None, :]
        Ro = tl.where(ready, tl.exp(tl.where(ready, diff_o, 0.0)), 0.0)
        out_tile = (
            tl.exp(cg)[:, None] * QS
            + tl.dot((QUT * Ro).to(tl.bfloat16), W_b)
        ) * scale
        tl.store(o_cur + offs_c[:, None] * Vdim + offs_v[None, :], out_tile)

        end_ratio = tl.exp(total_g - cum_l)
        W_scaled = (W * end_ratio[:, None]).to(tl.bfloat16)
        S = tl.exp(total_g) * S + tl.dot(tl.trans(U), W_scaled)

        q_cur += CHUNK * Kdim
        k_cur += CHUNK * Kdim
        e_cur += CHUNK * Kdim
        v_cur += CHUNK * Vdim
        o_cur += CHUNK * Vdim
        g_cur += CHUNK
        beta_cur += CHUNK
        gamma_cur += CHUNK
        sb_cur += Kdim * Vdim
        inv_cur += LCHUNK * LCHUNK


@triton.jit
def _fast_backward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    inv_ptr,
    sb_ptr,
    go_ptr,
    dq_ptr,
    dk_ptr,
    de_ptr,
    dv_ptr,
    dg_ptr,
    dbeta_ptr,
    dgamma_ptr,
    BH,
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
    row_t = offs_l // 2
    is_write = (offs_l % 2) == 1
    mask_k = offs_k[None, :] < Kdim
    mask_v = offs_v[None, :] < Vdim

    # Start every pointer at the last chunk and walk backwards.
    last_t = (num_chunks - 1) * CHUNK
    q_cur = q_ptr + (bh * T + last_t) * Kdim
    k_cur = k_ptr + (bh * T + last_t) * Kdim
    e_cur = e_ptr + (bh * T + last_t) * Kdim
    v_cur = v_ptr + (bh * T + last_t) * Vdim
    go_cur = go_ptr + (bh * T + last_t) * Vdim
    g_cur = g_ptr + bh * T + last_t
    beta_cur = beta_ptr + bh * T + last_t
    gamma_cur = gamma_ptr + bh * T + last_t
    sb_cur = sb_ptr + (bh * num_chunks + num_chunks - 1) * Kdim * Vdim
    inv_cur = inv_ptr + (bh * num_chunks + num_chunks - 1) * LCHUNK * LCHUNK

    part_k = (vb * BH + bh) * T * Kdim + last_t * Kdim
    dq_cur = dq_ptr + part_k
    dk_cur = dk_ptr + part_k
    de_cur = de_ptr + part_k
    dv_cur = dv_ptr + (bh * T + last_t) * Vdim
    part_1 = (vb * BH + bh) * T + last_t
    dg_cur = dg_ptr + part_1
    dbeta_cur = dbeta_ptr + part_1
    dgamma_cur = dgamma_ptr + part_1

    dS = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for _rev in range(num_chunks):
        # ----- recompute the chunk forward quantities ---------------------
        e_tile = tl.load(e_cur + row_t[:, None] * Kdim + offs_k[None, :])
        k_tile = tl.load(k_cur + row_t[:, None] * Kdim + offs_k[None, :])
        U = tl.where(is_write[:, None], k_tile, e_tile)

        v_tile = tl.load(v_cur + row_t[:, None] * Vdim + offs_v[None, :])
        target = tl.where(is_write[:, None], v_tile, 0.0).to(tl.float32)

        g_l = tl.load(g_cur + row_t).to(tl.float32)
        beta_l = tl.load(beta_cur + row_t).to(tl.float32)
        gamma_l = tl.load(gamma_cur + row_t).to(tl.float32)
        b_l = tl.where(is_write, beta_l, gamma_l)
        g_step = tl.where(is_write, 0.0, g_l)
        cum_l = tl.cumsum(g_step, axis=0)
        lam_l = tl.exp(cum_l)
        total_g = tl.sum(g_step, axis=0)
        exp_total = tl.exp(total_g)

        S0 = tl.load(sb_cur + offs_k[:, None] * Vdim + offs_v[None, :])
        P = tl.load(inv_cur + offs_l[:, None] * LCHUNK + offs_l[None, :])

        S0_b = S0.to(tl.bfloat16)
        dS_b = dS.to(tl.bfloat16)

        US = tl.dot(U, S0_b)
        diff = target - lam_l[:, None] * US
        rhs = b_l[:, None] * diff
        W = tl.dot(P, rhs, input_precision="tf32")
        W_b = W.to(tl.bfloat16)

        q_tile = tl.load(q_cur + offs_c[:, None] * Kdim + offs_k[None, :])
        GO = tl.load(go_cur + offs_c[:, None] * Vdim + offs_v[None, :]).to(
            tl.float32
        ) * scale
        GO_b = GO.to(tl.bfloat16)

        g_c = tl.load(g_cur + offs_c).to(tl.float32)
        cg = tl.cumsum(g_c, axis=0)
        exp_cg = tl.exp(cg)

        QS = tl.dot(q_tile, S0_b)
        QUT = tl.dot(q_tile, tl.trans(U))

        ready = offs_l[None, :] <= (2 * offs_c[:, None] + 1)
        diff_o = cg[:, None] - cum_l[None, :]
        Ro = tl.where(ready, tl.exp(tl.where(ready, diff_o, 0.0)), 0.0)
        Rend = tl.exp(total_g - cum_l)

        # ----- gradients flowing into W -----------------------------------
        UGsend = tl.dot(U, dS_b)
        GW = tl.dot(
            tl.trans((QUT * Ro).to(tl.bfloat16)), GO_b
        ) + Rend[:, None] * UGsend

        # ----- output-term gradients --------------------------------------
        GO_cg = exp_cg[:, None] * GO
        GO_cg_b = GO_cg.to(tl.bfloat16)
        dS0 = tl.dot(tl.trans(q_tile), GO_cg_b) + exp_total * dS
        GQ = tl.dot(GO_cg_b, tl.trans(S0_b))
        GM = tl.dot(GO_b, tl.trans(W_b))
        GQUT_b = (GM * Ro).to(tl.bfloat16)
        GQ = GQ + tl.dot(GQUT_b, U)
        GU = tl.dot(tl.trans(GQUT_b), q_tile)

        GRo = GM * QUT
        dc_c = exp_cg * tl.sum(GO * QS, axis=1) + tl.sum(GRo * Ro, axis=1)
        dcum = -tl.sum(GRo * Ro, axis=0)

        # ----- chunk-end state term ----------------------------------------
        GRend = tl.sum(UGsend * W, axis=1)
        dcum = dcum - GRend * Rend
        d_total = exp_total * tl.sum(
            tl.sum(dS * S0, axis=0), axis=0
        ) + tl.sum(GRend * Rend, axis=0)
        GU = GU + tl.dot((W * Rend[:, None]).to(tl.bfloat16), tl.trans(dS_b))

        # ----- differentiate through P = (I - N)^-1 -------------------------
        row = offs_l[:, None]
        col = offs_l[None, :]
        strict = col < row
        diff_n = cum_l[:, None] - cum_l[None, :]
        ratio_n = tl.where(strict, tl.exp(tl.where(strict, diff_n, 0.0)), 0.0)

        A = tl.dot(U, tl.trans(U))
        GP = tl.dot(GW.to(tl.bfloat16), tl.trans(rhs.to(tl.bfloat16)))
        GN = tl.dot(
            tl.trans(P),
            tl.dot(GP, tl.trans(P), input_precision="tf32"),
            input_precision="tf32",
        )
        GN = tl.where(strict, GN, 0.0)

        GRmat = GN * (-(b_l[:, None] * A))
        Gb = tl.sum(GN * (-(A * ratio_n)), axis=1)
        GA = -(b_l[:, None] * GN * ratio_n)
        dcum = dcum + tl.sum(GRmat * ratio_n, axis=1) - tl.sum(
            GRmat * ratio_n, axis=0
        )
        GU = GU + tl.dot((GA + tl.trans(GA)).to(tl.bfloat16), U)

        # ----- right-hand side ----------------------------------------------
        Grhs = tl.dot(tl.trans(P), GW, input_precision="tf32")
        Gb = Gb + tl.sum(Grhs * diff, axis=1)
        Gdiff = b_l[:, None] * Grhs
        GUS = -lam_l[:, None] * Gdiff
        dcum = dcum - lam_l * tl.sum(Gdiff * US, axis=1)
        GUS_b = GUS.to(tl.bfloat16)
        dS0 = dS0 + tl.dot(tl.trans(U), GUS_b)
        GU = GU + tl.dot(GUS_b, tl.trans(S0_b))

        # ----- decay adjoints back to per-timestep g -------------------------
        group_mat = (row_t[None, :] == offs_c[:, None]).to(tl.float32)
        grouped = tl.sum(group_mat * dcum[None, :], axis=1)
        prefix_adj = grouped + dc_c
        suffix_mat = (offs_c[None, :] >= offs_c[:, None]).to(tl.float32)
        dg_c = tl.sum(suffix_mat * prefix_adj[None, :], axis=1) + d_total

        # ----- store gradients (pointers are in the inputs' dtypes) ----------
        tl.store(dq_cur + offs_c[:, None] * Kdim + offs_k[None, :], GQ)
        tl.store(
            dk_cur + row_t[:, None] * Kdim + offs_k[None, :],
            GU,
            mask=is_write[:, None] & mask_k,
        )
        tl.store(
            de_cur + row_t[:, None] * Kdim + offs_k[None, :],
            GU,
            mask=(~is_write)[:, None] & mask_k,
        )
        tl.store(
            dv_cur + row_t[:, None] * Vdim + offs_v[None, :],
            Gdiff,
            mask=is_write[:, None] & mask_v,
        )
        tl.store(dg_cur + offs_c, dg_c)
        tl.store(dbeta_cur + row_t, Gb, mask=is_write)
        tl.store(dgamma_cur + row_t, Gb, mask=~is_write)

        dS = dS0

        q_cur -= CHUNK * Kdim
        k_cur -= CHUNK * Kdim
        e_cur -= CHUNK * Kdim
        v_cur -= CHUNK * Vdim
        go_cur -= CHUNK * Vdim
        g_cur -= CHUNK
        beta_cur -= CHUNK
        gamma_cur -= CHUNK
        sb_cur -= Kdim * Vdim
        inv_cur -= LCHUNK * LCHUNK
        dq_cur -= CHUNK * Kdim
        dk_cur -= CHUNK * Kdim
        de_cur -= CHUNK * Kdim
        dv_cur -= CHUNK * Vdim
        dg_cur -= CHUNK
        dbeta_cur -= CHUNK
        dgamma_cur -= CHUNK


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------

_CHUNK = 16
_LCHUNK = 32
_LOG_STEPS = 5

_FCHUNK = 32
_FLCHUNK = 64
_FLOG_STEPS = 6


class _EraseWriteDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H

        q2 = q.reshape(BH, T, K).contiguous()
        k2 = k.reshape(BH, T, K).contiguous()
        e2 = e.reshape(BH, T, K).contiguous()
        v2 = v.reshape(BH, T, V).contiguous()
        g2 = g.reshape(BH, T).contiguous()
        beta2 = beta.reshape(BH, T).contiguous()
        gamma2 = gamma.reshape(BH, T).contiguous()

        out = torch.empty((BH, T, V), device=q2.device, dtype=torch.float32)

        fast = (
            K == 128
            and V == 128
            and (T % _FCHUNK) == 0
            and q2.dtype == torch.bfloat16
            and k2.dtype == torch.bfloat16
            and v2.dtype == torch.bfloat16
            and e2.dtype == torch.bfloat16
        )

        if fast:
            num_chunks = T // _FCHUNK
            num_v_blocks = 2

            inv = torch.empty(
                (BH * num_chunks * _FLCHUNK * _FLCHUNK,),
                device=q2.device,
                dtype=torch.float32,
            )
            boundaries = torch.empty(
                (BH * num_chunks * K * V,), device=q2.device, dtype=torch.float32
            )

            _fast_inverse_kernel[(BH * num_chunks,)](
                k2, e2, g2, beta2, gamma2, inv, T, K, num_chunks,
                BLOCK_K=128, CHUNK=_FCHUNK, LCHUNK=_FLCHUNK,
                LOG_STEPS=_FLOG_STEPS, num_warps=4, num_stages=1,
            )
            _fast_apply_kernel[(BH * num_v_blocks,)](
                q2, k2, v2, e2, g2, beta2, gamma2, inv, boundaries, out,
                T, K, V, float(scale), num_chunks, num_v_blocks,
                BLOCK_K=128, BLOCK_V=64, CHUNK=_FCHUNK, LCHUNK=_FLCHUNK,
                num_warps=8, num_stages=1,
            )

            ctx.fast = True
            ctx.block_v = 64
        else:
            num_chunks = triton.cdiv(T, _CHUNK)
            BLOCK_K = max(16, triton.next_power_of_2(K))
            BLOCK_V = 64 if V > 64 else max(16, triton.next_power_of_2(V))
            num_v_blocks = triton.cdiv(V, BLOCK_V)
            # BLOCK_K = 256 (e.g. K = 129) tiles need less pipelining to fit
            # in the L4's shared memory.
            stages = 1 if BLOCK_K > 128 else 2

            inv = torch.empty(
                (BH * num_chunks * _LCHUNK * _LCHUNK,),
                device=q2.device,
                dtype=torch.float32,
            )
            boundaries = torch.empty(
                (BH * num_chunks * K * V,), device=q2.device, dtype=torch.float32
            )

            _chunk_inverse_kernel[(BH * num_chunks,)](
                k2, e2, g2, beta2, gamma2, inv, T, K, num_chunks,
                BLOCK_K=BLOCK_K, CHUNK=_CHUNK, LCHUNK=_LCHUNK,
                LOG_STEPS=_LOG_STEPS, num_warps=4, num_stages=stages,
            )
            _chunk_apply_kernel[(BH * num_v_blocks,)](
                q2, k2, v2, e2, g2, beta2, gamma2, inv, boundaries, out,
                T, K, V, float(scale), num_chunks, num_v_blocks,
                BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, CHUNK=_CHUNK,
                LCHUNK=_LCHUNK, num_warps=8, num_stages=stages,
            )

            ctx.fast = False
            ctx.block_v = BLOCK_V

        ctx.save_for_backward(q2, k2, v2, e2, g2, beta2, gamma2, inv, boundaries)
        ctx.dims = (B, H, T, K, V)
        ctx.scale = float(scale)
        ctx.num_chunks = num_chunks
        ctx.num_v_blocks = num_v_blocks
        return out.reshape(B, H, T, V)

    @staticmethod
    def backward(ctx, grad_out):
        q2, k2, v2, e2, g2, beta2, gamma2, inv, boundaries = ctx.saved_tensors
        B, H, T, K, V = ctx.dims
        BH = B * H
        num_chunks = ctx.num_chunks
        num_v_blocks = ctx.num_v_blocks

        go2 = grad_out.reshape(BH, T, V).contiguous().float()

        if ctx.fast:
            # Partial buffers are written by the kernel directly in the
            # inputs' dtypes, so the only wrapper work is the sum over the
            # two value blocks.
            dq_buf = torch.empty(
                (num_v_blocks, BH, T, K), device=q2.device, dtype=q2.dtype
            )
            dk_buf = torch.empty(
                (num_v_blocks, BH, T, K), device=q2.device, dtype=k2.dtype
            )
            de_buf = torch.empty(
                (num_v_blocks, BH, T, K), device=q2.device, dtype=e2.dtype
            )
            dv = torch.empty((BH, T, V), device=q2.device, dtype=v2.dtype)
            dg_buf = torch.empty(
                (num_v_blocks, BH, T), device=q2.device, dtype=torch.float32
            )
            dbeta_buf = torch.empty(
                (num_v_blocks, BH, T), device=q2.device, dtype=torch.float32
            )
            dgamma_buf = torch.empty(
                (num_v_blocks, BH, T), device=q2.device, dtype=torch.float32
            )

            _fast_backward_kernel[(BH * num_v_blocks,)](
                q2, k2, v2, e2, g2, beta2, gamma2, inv, boundaries, go2,
                dq_buf, dk_buf, de_buf, dv, dg_buf, dbeta_buf, dgamma_buf,
                BH, T, K, V, ctx.scale, num_chunks, num_v_blocks,
                BLOCK_K=128, BLOCK_V=64, CHUNK=_FCHUNK, LCHUNK=_FLCHUNK,
                num_warps=8, num_stages=1,
            )

            return (
                dq_buf.sum(0).reshape(B, H, T, K),
                dk_buf.sum(0).reshape(B, H, T, K),
                dv.reshape(B, H, T, V),
                de_buf.sum(0).reshape(B, H, T, K),
                dg_buf.sum(0).to(g2.dtype).reshape(B, H, T),
                dbeta_buf.sum(0).to(beta2.dtype).reshape(B, H, T),
                dgamma_buf.sum(0).to(gamma2.dtype).reshape(B, H, T),
                None,
            )

        dq_buf = torch.empty((num_v_blocks, BH, T, K), device=q2.device, dtype=torch.float32)
        dk_buf = torch.empty((num_v_blocks, BH, T, K), device=q2.device, dtype=torch.float32)
        de_buf = torch.empty((num_v_blocks, BH, T, K), device=q2.device, dtype=torch.float32)
        dv = torch.empty((BH, T, V), device=q2.device, dtype=torch.float32)
        dg_buf = torch.empty((num_v_blocks, BH, T), device=q2.device, dtype=torch.float32)
        dbeta_buf = torch.empty((num_v_blocks, BH, T), device=q2.device, dtype=torch.float32)
        dgamma_buf = torch.empty((num_v_blocks, BH, T), device=q2.device, dtype=torch.float32)

        BLOCK_K = max(16, triton.next_power_of_2(K))
        stages = 1 if BLOCK_K > 128 else 2
        _chunk_backward_kernel[(BH * num_v_blocks,)](
            q2, k2, v2, e2, g2, beta2, gamma2, inv, boundaries, go2,
            dq_buf, dk_buf, de_buf, dv, dg_buf, dbeta_buf, dgamma_buf,
            BH, T, K, V, ctx.scale, num_chunks, num_v_blocks,
            BLOCK_K=BLOCK_K, BLOCK_V=ctx.block_v, CHUNK=_CHUNK, LCHUNK=_LCHUNK,
            num_warps=8, num_stages=stages,
        )

        if num_v_blocks > 1:
            dq = dq_buf.sum(0)
            dk = dk_buf.sum(0)
            de = de_buf.sum(0)
            dg = dg_buf.sum(0)
            dbeta = dbeta_buf.sum(0)
            dgamma = dgamma_buf.sum(0)
        else:
            dq, dk, de = dq_buf[0], dk_buf[0], de_buf[0]
            dg, dbeta, dgamma = dg_buf[0], dbeta_buf[0], dgamma_buf[0]

        return (
            dq.to(q2.dtype).reshape(B, H, T, K),
            dk.to(k2.dtype).reshape(B, H, T, K),
            dv.to(v2.dtype).reshape(B, H, T, V),
            de.to(e2.dtype).reshape(B, H, T, K),
            dg.to(g2.dtype).reshape(B, H, T),
            dbeta.to(beta2.dtype).reshape(B, H, T),
            dgamma.to(gamma2.dtype).reshape(B, H, T),
            None,
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if not q.is_cuda:
        moved = tuple(x.cuda() for x in (q, k, v, e, g, beta, gamma))
        return _EraseWriteDelta.apply(*moved, float(scale)).to(q.device)
    return _EraseWriteDelta.apply(q, k, v, e, g, beta, gamma, float(scale))
```

The optimization is a mask-free fast path for the benchmark shape (K=V=128, bf16 inputs, T divisible by 32) that doubles the chunk to 32 timesteps, walks memory with incrementing/decrementing pointers instead of recomputed global offsets, and keeps all non-fast shapes on the proven generic kernels. Both carried states (forward `S` and backward `dS`) remain fp32, but their per-chunk snapshots and all large K/V contractions now run as bf16 tensor-core dots with fp32 accumulation, while the sensitive 64×64 triangular-inverse products in the forward, its adjoint (via `dP = P dN P`), and the chunk-end state paths stay TF32. The fast backward also stores `dq/dk/de/dv` directly in bf16 through the kernel, eliminating the separate fp32-to-bf16 cast kernels that appeared in the profile and halving the partial-sum traffic. I am unsure whether the larger 64-step system measurably raises the inverse kernel's cost or whether bf16 rounding of the state snapshots stays comfortably inside the 2e-2 tolerance across all checker distributions, since I cannot benchmark or test locally.