import torch
import triton
import triton.language as tl


_BLOCK_N = 16
_CHUNK_SLOW = 4
_CHUNK_FAST = 8


@triton.jit
def _exp80(x):
    # exp() with the argument clamped to a float32-safe range.
    return tl.exp(tl.minimum(tl.maximum(x, -80.0), 80.0))


@triton.jit
def _chunk_decay_kernel(
    g_ptr,
    gc_ptr,
    T,
    K,
    CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_t = pid_c * CHUNK + tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BLOCK_K)
    mask = (offs_t[:, None] < T) & (offs_k[None, :] < K)

    base = pid_bh.to(tl.int64) * T * K
    offs = base + offs_t[:, None].to(tl.int64) * K + offs_k[None, :]
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    gc = tl.cumsum(g, axis=0)
    tl.store(gc_ptr + offs, gc, mask=mask)


# ---------------------------------------------------------------------------
# Fast kernel: 8 timesteps per chunk (= 16 erase/write sub-steps), TF32 dots.
#
# Fix for strong decay: a single mid-chunk decay reference puts the first
# timesteps ~8 decay steps away from the reference, so with g in [-25, -5]
# the scaled factors exp(+(gc-r)) / exp(-(gc-r)) hit the +/-80 clamp with
# *different* amounts for the destination and source factors of the same
# Gram / q-Gram entry.  The clamp therefore no longer cancels in the
# product exp(gc_i - gc_j), corrupting exactly the intra-chunk couplings.
# The Gram and q-Gram are now computed twice, once with a reference at the
# start of each 4-timestep half of the chunk (so every destination row is
# at most 4 decay steps from its reference), and the appropriate half is
# selected row-wise.  With a source-start reference, any remaining clamped
# factor belongs to a pair whose true coupling is <= e^-30 and hence
# negligible at the 2e-2 tolerance.  All other terms keep the previous
# single mid-chunk reference, which was already accurate for them.
# ---------------------------------------------------------------------------
@triton.jit
def _recurrent_kernel_fast(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    gc_ptr,
    beta_ptr,
    gamma_ptr,
    out_ptr,
    scale,
    T,
    K,
    V_DIM,
    NC,
    NV,
    CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_v = pid % NV
    pid_bh = pid // NV
    pid64 = pid_bh.to(tl.int64)

    offs_q = tl.arange(0, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    mask_k = offs_k < K
    mask_v = offs_v < V_DIM

    base_k = pid64 * T * K
    base_v = pid64 * T * V_DIM
    base_t = pid64 * T

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for chunk in range(NC):
        t_start = chunk * CHUNK

        # Expanded sub-steps: 2*t is the erase step, 2*t+1 is the write step.
        t_rel_n = offs_n // 2
        t_n = t_start + t_rel_n
        mask_n = t_n < T
        is_write = (offs_n % 2) == 1

        ptrs_nk = base_k + t_n[:, None].to(tl.int64) * K + offs_k[None, :]
        mask_nk = mask_n[:, None] & mask_k[None, :]

        k_exp = tl.load(k_ptr + ptrs_nk, mask=mask_nk, other=0.0).to(tl.float32)
        e_exp = tl.load(e_ptr + ptrs_nk, mask=mask_nk, other=0.0).to(tl.float32)
        vec = tl.where(is_write[:, None], k_exp, e_exp)

        gc_exp = tl.load(gc_ptr + ptrs_nk, mask=mask_nk, other=0.0).to(tl.float32)

        beta_t = tl.load(
            beta_ptr + base_t + t_n.to(tl.int64), mask=mask_n, other=0.0
        ).to(tl.float32)
        gamma_t = tl.load(
            gamma_ptr + base_t + t_n.to(tl.int64), mask=mask_n, other=0.0
        ).to(tl.float32)
        beta_step = tl.where(is_write, beta_t, gamma_t * (1.0 - beta_t))
        beta_step = tl.where(mask_n, beta_step, 0.0)

        v_exp = tl.load(
            v_ptr + base_v + t_n[:, None].to(tl.int64) * V_DIM + offs_v[None, :],
            mask=mask_n[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        v_star = tl.where(is_write[:, None], v_exp, 0.0)

        # Query rows (one per timestep of the chunk), loaded early because
        # the q-Gram is built together with the Gram below.
        t_c = t_start + offs_q
        mask_c = (offs_q < CHUNK) & (t_c < T)
        ptrs_ck = base_k + t_c[:, None].to(tl.int64) * K + offs_k[None, :]
        mask_ck = mask_c[:, None] & mask_k[None, :]

        q_c = tl.load(q_ptr + ptrs_ck, mask=mask_ck, other=0.0).to(tl.float32)
        gc_c = tl.load(gc_ptr + ptrs_ck, mask=mask_ck, other=0.0).to(tl.float32)

        # --- Gram and q-Gram, first half (reference = gc of sub-step 0) ---
        r_a = tl.sum(
            tl.where(offs_n[:, None] == 0, gc_exp, 0.0),
            axis=0,
        )
        w_a = vec * _exp80(gc_exp - r_a[None, :])
        u_a = vec * _exp80(r_a[None, :] - gc_exp)
        gram_a = tl.dot(w_a, tl.trans(u_a), input_precision="tf32")
        q_a = q_c * _exp80(gc_c - r_a[None, :])
        q_gram_a = tl.dot(q_a, tl.trans(u_a), input_precision="tf32")

        # --- Gram and q-Gram, second half (reference = gc of sub-step 8) ---
        r_b = tl.sum(
            tl.where(offs_n[:, None] == (BLOCK_N // 2), gc_exp, 0.0),
            axis=0,
        )
        w_b = vec * _exp80(gc_exp - r_b[None, :])
        u_b = vec * _exp80(r_b[None, :] - gc_exp)
        gram_b = tl.dot(w_b, tl.trans(u_b), input_precision="tf32")
        q_b = q_c * _exp80(gc_c - r_b[None, :])
        q_gram_b = tl.dot(q_b, tl.trans(u_b), input_precision="tf32")

        # Each destination row uses the reference of its own half.
        gram = tl.where(offs_n[:, None] < (BLOCK_N // 2), gram_a, gram_b)
        q_gram = tl.where(offs_q[:, None] < (CHUNK // 2), q_gram_a, q_gram_b)

        # --- Remaining terms with the single mid-chunk reference ---
        r = tl.sum(
            tl.where(offs_n[:, None] == (BLOCK_N // 2), gc_exp, 0.0),
            axis=0,
        )
        w_scaled = vec * _exp80(gc_exp - r[None, :])
        u_scaled = vec * _exp80(r[None, :] - gc_exp)

        S_r = S * _exp80(r)[:, None]

        # Prediction of every sub-step from the chunk-initial state.
        pred0 = tl.dot(w_scaled, S_r, input_precision="tf32")

        lower = offs_n[None, :] < offs_n[:, None]
        l_mat = tl.where(lower, beta_step[:, None] * gram, 0.0)

        # Invert (I + L), L strictly lower triangular / nilpotent, via the
        # Neumann series evaluated by repeated doubling (log2(16) = 4).
        inv_ser = tl.where(
            offs_n[:, None] == offs_n[None, :], 1.0, 0.0
        ).to(tl.float32)
        pow_x = -l_mat
        for _ in range(4):
            new_ser = inv_ser + tl.dot(pow_x, inv_ser, input_precision="tf32")
            new_pow = tl.dot(pow_x, pow_x, input_precision="tf32")
            inv_ser = new_ser
            pow_x = new_pow

        rhs = beta_step[:, None] * (v_star - pred0)
        v_hat = tl.dot(inv_ser, rhs, input_precision="tf32")

        # Outputs.
        q_scaled = q_c * _exp80(gc_c - r[None, :])

        out_val = tl.dot(q_scaled, S_r, input_precision="tf32")
        q_gram = tl.where(
            offs_n[None, :] <= (2 * offs_q[:, None] + 1), q_gram, 0.0
        )
        out_val = (out_val + tl.dot(q_gram, v_hat, input_precision="tf32")) * scale

        tl.store(
            out_ptr
            + base_v
            + t_c[:, None].to(tl.int64) * V_DIM
            + offs_v[None, :],
            out_val,
            mask=mask_c[:, None] & mask_v[None, :],
        )

        # End-of-chunk state, converted back out of the rescaled coordinates.
        l_last = tl.sum(
            tl.where(offs_n[:, None] == (2 * CHUNK - 1), gc_exp, 0.0),
            axis=0,
        )
        state_delta = tl.dot(tl.trans(u_scaled), v_hat, input_precision="tf32")
        S = S * _exp80(l_last)[:, None] + state_delta * _exp80(l_last - r)[:, None]


# ---------------------------------------------------------------------------
# Slow (fallback) kernel: the previously validated 4-step / IEEE version.
# Used only when BLOCK_K pads past 128 (e.g. K = 129), where the fast
# kernel's tiles would exceed the L4 shared-memory limit.
# ---------------------------------------------------------------------------
@triton.jit
def _recurrent_kernel_slow(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    gc_ptr,
    beta_ptr,
    gamma_ptr,
    out_ptr,
    scale,
    T,
    K,
    V_DIM,
    NC,
    NV,
    CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_v = pid % NV
    pid_bh = pid // NV
    pid64 = pid_bh.to(tl.int64)

    offs_q = tl.arange(0, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    mask_k = offs_k < K
    mask_v = offs_v < V_DIM

    base_k = pid64 * T * K
    base_v = pid64 * T * V_DIM
    base_t = pid64 * T

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for chunk in range(NC):
        t_start = chunk * CHUNK

        t_rel_n = offs_n // 2
        t_n = t_start + t_rel_n
        mask_n = (offs_n < 2 * CHUNK) & (t_n < T)
        is_write = (offs_n % 2) == 1

        ptrs_nk = base_k + t_n[:, None].to(tl.int64) * K + offs_k[None, :]
        mask_nk = mask_n[:, None] & mask_k[None, :]

        k_exp = tl.load(k_ptr + ptrs_nk, mask=mask_nk, other=0.0).to(tl.float32)
        e_exp = tl.load(e_ptr + ptrs_nk, mask=mask_nk, other=0.0).to(tl.float32)
        vec = tl.where(is_write[:, None], k_exp, e_exp)

        gc_exp = tl.load(gc_ptr + ptrs_nk, mask=mask_nk, other=0.0).to(tl.float32)

        r = tl.sum(
            tl.where(offs_n[:, None] == CHUNK, gc_exp, 0.0),
            axis=0,
        )
        w_scaled = vec * tl.exp(gc_exp - r[None, :])
        u_scaled = vec * tl.exp(r[None, :] - gc_exp)

        beta_t = tl.load(
            beta_ptr + base_t + t_n.to(tl.int64), mask=mask_n, other=0.0
        ).to(tl.float32)
        gamma_t = tl.load(
            gamma_ptr + base_t + t_n.to(tl.int64), mask=mask_n, other=0.0
        ).to(tl.float32)
        beta_step = tl.where(is_write, beta_t, gamma_t * (1.0 - beta_t))
        beta_step = tl.where(mask_n, beta_step, 0.0)

        v_exp = tl.load(
            v_ptr + base_v + t_n[:, None].to(tl.int64) * V_DIM + offs_v[None, :],
            mask=mask_n[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        v_star = tl.where(is_write[:, None], v_exp, 0.0)

        exp_r = tl.exp(r)
        S_r = S * exp_r[:, None]

        pred0 = tl.dot(w_scaled, S_r, input_precision="ieee")
        gram = tl.dot(w_scaled, tl.trans(u_scaled), input_precision="ieee")
        lower = offs_n[None, :] < offs_n[:, None]
        l_mat = tl.where(lower, beta_step[:, None] * gram, 0.0)

        inv_ser = tl.where(
            offs_n[:, None] == offs_n[None, :], 1.0, 0.0
        ).to(tl.float32)
        pow_x = -l_mat
        for _ in range(4):
            new_ser = inv_ser + tl.dot(pow_x, inv_ser, input_precision="ieee")
            new_pow = tl.dot(pow_x, pow_x, input_precision="ieee")
            inv_ser = new_ser
            pow_x = new_pow

        rhs = beta_step[:, None] * (v_star - pred0)
        v_hat = tl.dot(inv_ser, rhs, input_precision="ieee")

        t_c = t_start + offs_q
        mask_c = (offs_q < CHUNK) & (t_c < T)
        ptrs_ck = base_k + t_c[:, None].to(tl.int64) * K + offs_k[None, :]
        mask_ck = mask_c[:, None] & mask_k[None, :]

        q_c = tl.load(q_ptr + ptrs_ck, mask=mask_ck, other=0.0).to(tl.float32)
        gc_c = tl.load(gc_ptr + ptrs_ck, mask=mask_ck, other=0.0).to(tl.float32)
        q_scaled = q_c * tl.exp(gc_c - r[None, :])

        out_val = tl.dot(q_scaled, S_r, input_precision="ieee")
        q_gram = tl.dot(q_scaled, tl.trans(u_scaled), input_precision="ieee")
        q_gram = tl.where(
            offs_n[None, :] <= (2 * offs_q[:, None] + 1), q_gram, 0.0
        )
        out_val = (out_val + tl.dot(q_gram, v_hat, input_precision="ieee")) * scale

        tl.store(
            out_ptr
            + base_v
            + t_c[:, None].to(tl.int64) * V_DIM
            + offs_v[None, :],
            out_val,
            mask=mask_c[:, None] & mask_v[None, :],
        )

        l_last = tl.sum(
            tl.where(offs_n[:, None] == (2 * CHUNK - 1), gc_exp, 0.0),
            axis=0,
        )
        state_delta = tl.dot(tl.trans(u_scaled), v_hat, input_precision="ieee")
        S = (
            S * tl.exp(l_last)[:, None]
            + state_delta * tl.exp(l_last - r)[:, None]
        )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    if B == 0 or H == 0 or T == 0 or V == 0:
        return torch.empty(
            (B, H, T, V), device=q.device, dtype=torch.float32
        )

    BH = B * H

    q_f = q.reshape(BH, T, K).contiguous()
    k_f = k.reshape(BH, T, K).contiguous()
    v_f = v.reshape(BH, T, V).contiguous()
    e_f = e.reshape(BH, T, K).contiguous()
    g_f = g.reshape(BH, T, K).contiguous()
    beta_f = beta.reshape(BH, T).contiguous()
    gamma_f = gamma.reshape(BH, T).contiguous()

    block_k = max(16, 1 << (K - 1).bit_length()) if K > 0 else 16
    block_v_full = max(16, 1 << (V - 1).bit_length()) if V > 0 else 16

    out = torch.empty((BH, T, V), device=q.device, dtype=torch.float32)

    if block_k <= 128:
        chunk = _CHUNK_FAST
        block_v = min(64, block_v_full)
        num_stages = 2
        kernel = _recurrent_kernel_fast
    else:
        chunk = _CHUNK_SLOW
        block_v = min(32, block_v_full)
        num_stages = 1
        kernel = _recurrent_kernel_slow

    num_v_tiles = (V + block_v - 1) // block_v
    num_chunks = (T + chunk - 1) // chunk

    gcum = torch.empty((BH, T, K), device=q.device, dtype=torch.float32)
    _chunk_decay_kernel[(BH, num_chunks)](
        g_f,
        gcum,
        T,
        K,
        CHUNK=chunk,
        BLOCK_K=block_k,
        num_warps=4,
    )

    kernel[(BH * num_v_tiles,)](
        q_f,
        k_f,
        v_f,
        e_f,
        gcum,
        beta_f,
        gamma_f,
        out,
        float(scale),
        T,
        K,
        V,
        num_chunks,
        num_v_tiles,
        CHUNK=chunk,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=8,
        num_stages=num_stages,
    )
    return out.reshape(B, H, T, V)
