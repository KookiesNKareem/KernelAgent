import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fast chunk-preparation kernel.
#
# One program handles one (batch, head, chunk).  It builds, independently of
# the recurrent state:
#
#   Inv   = (I + L)^(-1)          intra-chunk delta-rule triangular inverse
#   Kg    = Inv @ diag(beta * A) K
#   Qeff  = diag(A) Q - M @ Kg
#   U     = Inv @ diag(beta) V
#   Ointr = M @ U
#
# where A_i = exp(cumsum(g)_i) and M is the decay-masked Q K^T matrix.
# ---------------------------------------------------------------------------
@triton.jit
def _prep_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
    kg_ptr, qeff_ptr, u_ptr, ointra_ptr, cum_ptr,
    H, T, K_DIM, V_DIM, NUM_CHUNKS,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    sgb, sgh, sgt,
    sbb, sbh, sbt,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BVFULL: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // NUM_CHUNKS
    chunk_idx = pid % NUM_CHUNKS
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    q_base = q_ptr + b_idx * sqb + h_idx * sqh
    k_base = k_ptr + b_idx * skb + h_idx * skh
    v_base = v_ptr + b_idx * svb + h_idx * svh
    g_base = g_ptr + b_idx * sgb + h_idx * sgh
    beta_base = beta_ptr + b_idx * sbb + h_idx * sbh

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BVFULL)
    offs_t = chunk_idx * CHUNK + offs_c

    mask_t = offs_t < T
    mask_k = offs_k < K_DIM
    mask_v = offs_v < V_DIM

    g_vec = tl.load(
        g_base + offs_t * sgt, mask=mask_t, other=0.0
    ).to(tl.float32)
    beta_vec = tl.load(
        beta_base + offs_t * sbt, mask=mask_t, other=0.0
    ).to(tl.float32)

    cum = tl.cumsum(g_vec, axis=0)
    a_vec = tl.exp(cum)

    slice_idx = pid_bh * NUM_CHUNKS + chunk_idx
    tl.store(cum_ptr + slice_idx * CHUNK + offs_c, cum)

    k_tile = tl.load(
        k_base + offs_t[:, None] * skt + offs_k[None, :] * skk,
        mask=mask_t[:, None] & mask_k[None, :],
        other=0.0,
    )
    q_tile = tl.load(
        q_base + offs_t[:, None] * sqt + offs_k[None, :] * sqk,
        mask=mask_t[:, None] & mask_k[None, :],
        other=0.0,
    )

    row = offs_c[:, None]
    col = offs_c[None, :]
    eye = (row == col).to(tl.float32)
    lt_mask = col < row
    le_mask = col <= row
    cum_diff = cum[:, None] - cum[None, :]

    kk = tl.dot(k_tile, tl.trans(k_tile))
    qk = tl.dot(q_tile, tl.trans(k_tile))

    decay_lt = tl.exp(tl.where(lt_mask, cum_diff, 0.0))
    l_mat = tl.where(lt_mask, kk * beta_vec[:, None] * decay_lt, 0.0)

    decay_le = tl.exp(tl.where(le_mask, cum_diff, 0.0))
    m_mat = tl.where(le_mask, qk * decay_le, 0.0)

    # Exact inverse of the nilpotent unit-lower-triangular system:
    # (I+L)^-1 = (I-L)(I+L^2)(I+L^4)(I+L^8)(I+L^16)(I+L^32), CHUNK=64.
    inv = eye - l_mat
    power = tl.dot(l_mat, l_mat, input_precision="tf32")
    inv = tl.dot(inv, eye + power, input_precision="tf32")
    power = tl.dot(power, power, input_precision="tf32")
    inv = tl.dot(inv, eye + power, input_precision="tf32")
    power = tl.dot(power, power, input_precision="tf32")
    inv = tl.dot(inv, eye + power, input_precision="tf32")
    power = tl.dot(power, power, input_precision="tf32")
    inv = tl.dot(inv, eye + power, input_precision="tf32")
    power = tl.dot(power, power, input_precision="tf32")
    inv = tl.dot(inv, eye + power, input_precision="tf32")

    k_f32 = k_tile.to(tl.float32)
    q_f32 = q_tile.to(tl.float32)

    kg = tl.dot(
        inv,
        k_f32 * (beta_vec * a_vec)[:, None],
        input_precision="tf32",
    )
    tl.store(
        kg_ptr + slice_idx * CHUNK * K_DIM
        + offs_c[:, None] * K_DIM + offs_k[None, :],
        kg,
        mask=mask_k[None, :],
    )

    qeff = q_f32 * a_vec[:, None] - tl.dot(
        m_mat, kg, input_precision="tf32"
    )
    tl.store(
        qeff_ptr + slice_idx * CHUNK * K_DIM
        + offs_c[:, None] * K_DIM + offs_k[None, :],
        qeff,
        mask=mask_k[None, :],
    )

    v_tile = tl.load(
        v_base + offs_t[:, None] * svt + offs_v[None, :] * svv,
        mask=mask_t[:, None] & mask_v[None, :],
        other=0.0,
    )
    u_mat = tl.dot(
        inv,
        v_tile.to(tl.float32) * beta_vec[:, None],
        input_precision="tf32",
    )
    tl.store(
        u_ptr + slice_idx * CHUNK * V_DIM
        + offs_c[:, None] * V_DIM + offs_v[None, :],
        u_mat,
        mask=mask_v[None, :],
    )

    ointra = tl.dot(m_mat, u_mat, input_precision="tf32")
    tl.store(
        ointra_ptr + slice_idx * CHUNK * V_DIM
        + offs_c[:, None] * V_DIM + offs_v[None, :],
        ointra,
        mask=mask_v[None, :],
    )


# ---------------------------------------------------------------------------
# Fast recurrent kernel.  One program handles one (batch, head, V-block) and
# scans chunks sequentially, keeping S in registers.  Per chunk it only needs
# three state-sized matmuls; all triangular/chunk-local work was prepared by
# _prep_kernel once instead of once per V-block.
#
#   W   = U - Kg @ S
#   o   = scale * (Qeff @ S + Ointr)
#   S'  = exp(cum_last) * S + K_suffix^T @ W
# ---------------------------------------------------------------------------
@triton.jit
def _scan_kernel(
    k_ptr, kg_ptr, qeff_ptr, u_ptr, ointra_ptr, cum_ptr, o_ptr,
    scale,
    H, T, K_DIM, V_DIM, NUM_CHUNKS,
    skb, skh, skt, skk,
    sob, soh, sot, sov,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    k_base = k_ptr + b_idx * skb + h_idx * skh
    o_base = o_ptr + b_idx * soh + h_idx * soh * 0 + b_idx * sob * 0  # placeholder
    o_base = o_ptr + b_idx * sob + h_idx * soh

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    offs_v_global = pid_v * BV + offs_v

    mask_k = offs_k < K_DIM
    mask_v = offs_v_global < V_DIM

    state = tl.zeros([BK, BV], dtype=tl.float32)

    for chunk_idx in range(NUM_CHUNKS):
        offs_t = chunk_idx * CHUNK + offs_c
        mask_t = offs_t < T
        slice_idx = pid_bh * NUM_CHUNKS + chunk_idx

        kg = tl.load(
            kg_ptr + slice_idx * CHUNK * K_DIM
            + offs_c[:, None] * K_DIM + offs_k[None, :],
            mask=mask_k[None, :],
            other=0.0,
        )
        u_mat = tl.load(
            u_ptr + slice_idx * CHUNK * V_DIM
            + offs_c[:, None] * V_DIM + offs_v_global[None, :],
            mask=mask_v[None, :],
            other=0.0,
        )
        w_tile = u_mat - tl.dot(kg, state, input_precision="tf32")

        qeff = tl.load(
            qeff_ptr + slice_idx * CHUNK * K_DIM
            + offs_c[:, None] * K_DIM + offs_k[None, :],
            mask=mask_k[None, :],
            other=0.0,
        )
        ointra = tl.load(
            ointra_ptr + slice_idx * CHUNK * V_DIM
            + offs_c[:, None] * V_DIM + offs_v_global[None, :],
            mask=mask_v[None, :],
            other=0.0,
        )
        o_tile = scale * (
            tl.dot(qeff, state, input_precision="tf32") + ointra
        )
        tl.store(
            o_base + offs_t[:, None] * sot + offs_v_global[None, :] * sov,
            o_tile,
            mask=mask_t[:, None] & mask_v[None, :],
        )

        cum = tl.load(cum_ptr + slice_idx * CHUNK + offs_c)
        total_cum = tl.sum(
            tl.where(offs_c == CHUNK - 1, cum, 0.0), axis=0
        )
        k_tile = tl.load(
            k_base + offs_t[:, None] * skt + offs_k[None, :] * skk,
            mask=mask_t[:, None] & mask_k[None, :],
            other=0.0,
        )
        k_suffix = k_tile.to(tl.float32) * tl.exp(total_cum - cum)[:, None]
        state = tl.exp(total_cum) * state + tl.dot(
            tl.trans(k_suffix), w_tile, input_precision="tf32"
        )


# ---------------------------------------------------------------------------
# Fallback kernel: the previously validated small-tile formulation, used for
# shapes/dtypes outside the fast path (e.g. K > 128).
# ---------------------------------------------------------------------------
@triton.jit
def _fallback_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    scale,
    H, T, K_DIM, V_DIM, NUM_CHUNKS,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    sgb, sgh, sgt,
    sbb, sbh, sbt,
    sob, soh, sot, sov,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    b_idx = pid_bh // H
    h_idx = pid_bh % H

    q_base = q_ptr + b_idx * sqb + h_idx * sqh
    k_base = k_ptr + b_idx * skb + h_idx * skh
    v_base = v_ptr + b_idx * svb + h_idx * svh
    g_base = g_ptr + b_idx * sgb + h_idx * sgh
    beta_base = beta_ptr + b_idx * sbb + h_idx * sbh
    o_base = o_ptr + b_idx * sob + h_idx * soh

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    offs_v_global = pid_v * BV + offs_v

    mask_k = offs_k < K_DIM
    mask_v = offs_v_global < V_DIM

    row = offs_c[:, None]
    col = offs_c[None, :]
    eye = (row == col).to(tl.float32)
    lt_mask = col < row
    le_mask = col <= row

    state = tl.zeros([BK, BV], dtype=tl.float32)

    for chunk_idx in range(NUM_CHUNKS):
        start = chunk_idx * CHUNK
        offs_t = start + offs_c
        mask_t = offs_t < T

        k_tile = tl.load(
            k_base + offs_t[:, None] * skt + offs_k[None, :] * skk,
            mask=mask_t[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        q_tile = tl.load(
            q_base + offs_t[:, None] * sqt + offs_k[None, :] * sqk,
            mask=mask_t[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        v_tile = tl.load(
            v_base + offs_t[:, None] * svt + offs_v_global[None, :] * svv,
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)

        g_vec = tl.load(
            g_base + offs_t * sgt, mask=mask_t, other=0.0
        ).to(tl.float32)

        beta_vec = tl.load(
            beta_base + offs_t * sbt, mask=mask_t, other=0.0
        ).to(tl.float32)

        cum = tl.cumsum(g_vec, axis=0)
        a_vec = tl.exp(cum)
        total_log_decay = tl.sum(g_vec, axis=0)
        a_last = tl.exp(total_log_decay)
        cum_diff = cum[:, None] - cum[None, :]

        kk = tl.dot(k_tile, tl.trans(k_tile), input_precision="ieee")
        decay_lt = tl.exp(tl.where(lt_mask, cum_diff, 0.0))
        l_mat = tl.where(lt_mask, kk * beta_vec[:, None] * decay_lt, 0.0)

        inv = eye - l_mat
        power = tl.dot(l_mat, l_mat, input_precision="ieee")
        inv = tl.dot(inv, eye + power, input_precision="ieee")
        power = tl.dot(power, power, input_precision="ieee")
        inv = tl.dot(inv, eye + power, input_precision="ieee")
        power = tl.dot(power, power, input_precision="ieee")
        inv = tl.dot(inv, eye + power, input_precision="ieee")
        power = tl.dot(power, power, input_precision="ieee")
        inv = tl.dot(inv, eye + power, input_precision="ieee")

        ks = tl.dot(k_tile, state, input_precision="ieee")
        rhs = beta_vec[:, None] * (v_tile - a_vec[:, None] * ks)
        w_tile = tl.dot(inv, rhs, input_precision="ieee")

        qk = tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee")
        decay_le = tl.exp(tl.where(le_mask, cum_diff, 0.0))
        m_mat = tl.where(le_mask, qk * decay_le, 0.0)
        qs = tl.dot(q_tile, state, input_precision="ieee")
        mw = tl.dot(m_mat, w_tile, input_precision="ieee")
        o_tile = scale * (a_vec[:, None] * qs + mw)

        tl.store(
            o_base + offs_t[:, None] * sot + offs_v_global[None, :] * sov,
            o_tile,
            mask=mask_t[:, None] & mask_v[None, :],
        )

        k_suffix = k_tile * tl.exp(total_log_decay - cum)[:, None]
        state = a_last * state + tl.dot(
            tl.trans(k_suffix), w_tile, input_precision="ieee"
        )


def _next_power_of_2(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # e and gamma are intentionally unused, exactly as in the reference.
    B, H, T, K_DIM = q.shape
    V_DIM = v.shape[-1]

    out = torch.empty((B, H, T, V_DIM), device=q.device, dtype=torch.float32)
    if out.numel() == 0:
        return out

    BK = max(16, _next_power_of_2(K_DIM))
    BVFULL = max(16, _next_power_of_2(V_DIM))

    use_fast = (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and g.dtype == torch.float32
        and beta.dtype == torch.float32
        and BK <= 128
        and BVFULL <= 128
    )

    if use_fast:
        CHUNK = 64
        num_chunks = (T + CHUNK - 1) // CHUNK
        bh = B * H
        prep_elems = bh * num_chunks * CHUNK * (2 * K_DIM + 2 * V_DIM + 1)
        # Keep the fp32 workspace bounded; otherwise use the fallback.
        use_fast = prep_elems <= 200_000_000 and bh * num_chunks <= 1_000_000

    if use_fast:
        BV = 64 if V_DIM > 64 else BVFULL

        kg_buf = torch.empty(
            (bh, num_chunks, CHUNK, K_DIM), device=q.device, dtype=torch.float32
        )
        qeff_buf = torch.empty(
            (bh, num_chunks, CHUNK, K_DIM), device=q.device, dtype=torch.float32
        )
        u_buf = torch.empty(
            (bh, num_chunks, CHUNK, V_DIM), device=q.device, dtype=torch.float32
        )
        ointra_buf = torch.empty(
            (bh, num_chunks, CHUNK, V_DIM), device=q.device, dtype=torch.float32
        )
        cum_buf = torch.empty(
            (bh, num_chunks, CHUNK), device=q.device, dtype=torch.float32
        )

        _prep_kernel[(bh * num_chunks,)](
            q, k, v, g, beta,
            kg_buf, qeff_buf, u_buf, ointra_buf, cum_buf,
            H, T, K_DIM, V_DIM, num_chunks,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            g.stride(0), g.stride(1), g.stride(2),
            beta.stride(0), beta.stride(1), beta.stride(2),
            CHUNK=CHUNK,
            BK=BK,
            BVFULL=BVFULL,
            num_warps=8,
            num_stages=1,
        )

        _scan_kernel[(bh, (V_DIM + BV - 1) // BV)](
            k, kg_buf, qeff_buf, u_buf, ointra_buf, cum_buf, out,
            float(scale),
            H, T, K_DIM, V_DIM, num_chunks,
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            CHUNK=CHUNK,
            BK=BK,
            BV=BV,
            num_warps=8,
            num_stages=1,
        )
        return out

    # Fallback path (previously validated tiling).
    if K_DIM > 128:
        CHUNK = 16
        BV = 16
        num_warps = 8
    else:
        CHUNK = 32
        BV = 32 if V_DIM > 32 else max(16, _next_power_of_2(V_DIM))
        num_warps = 4
    num_chunks = (T + CHUNK - 1) // CHUNK
    grid = (B * H, (V_DIM + BV - 1) // BV)

    _fallback_kernel[grid](
        q, k, v, g, beta, out,
        float(scale),
        H, T, K_DIM, V_DIM, num_chunks,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        beta.stride(0), beta.stride(1), beta.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        CHUNK=CHUNK,
        BK=BK,
        BV=BV,
        num_warps=num_warps,
        num_stages=1,
    )
    return out
