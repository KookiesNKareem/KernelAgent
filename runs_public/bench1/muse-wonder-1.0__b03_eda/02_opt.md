```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Chunkwise kernel.
#
# Each original step is split into two micro-steps (erase with e, then
# delta-write with k).  Within a chunk of 32 tokens (64 micro-steps) the
# recurrence becomes
#     S <- exp(g) * S + u_n z_n^T
# where z satisfies a strictly-lower-triangular system (I + B) z = rhs.
# B only depends on Gram matrices of the interleaved e/k vectors and on
# intra-chunk decay ratios (all <= 1, so nothing is divided by a tiny decay).
# (I + B)^-1 is obtained by 6 Newton iterations (B^64 = 0), after which all
# outputs and the chunk-final state are ordinary matmuls against S_in.
# ---------------------------------------------------------------------------
@triton.jit
def _chunk_delta_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    out_ptr,
    T,
    K,
    V,
    scale,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
    BN: tl.constexpr,
):
    pid_v = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    bh = pid_bh.to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_c = tl.arange(0, BC)
    offs_n = tl.arange(0, BN)
    mask_k = offs_k < K
    mask_v = offs_v < V

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    o_base = out_ptr + bh * T * V
    s_base = g_ptr + bh * T

    eye = tl.where(offs_n[:, None] == offs_n[None, :], 1.0, 0.0)
    strict = offs_n[:, None] > offs_n[None, :]

    tok_n = offs_n // 2
    is_write = (offs_n % 2) == 1
    is_erase = (offs_n % 2) == 0

    S = tl.zeros((BK, BV), dtype=tl.float32)

    for chunk in range(0, tl.cdiv(T, BC)):
        c0 = chunk * BC
        toks_c = c0 + offs_c
        valid_c = toks_c < T
        toks_n = c0 + tok_n
        valid_n = toks_n < T

        # ---- per-token cumulative decay (inclusive) ----
        g_c = tl.load(s_base + toks_c, mask=valid_c, other=0.0).to(tl.float32)
        logE_c = tl.cumsum(g_c, axis=0)
        logE_total = tl.sum(g_c)
        E_total = tl.exp(logE_total)

        # ---- micro-step cumulative decay ----
        g_n = tl.load(s_base + toks_n, mask=is_erase & valid_n, other=0.0).to(tl.float32)
        logE_n = tl.cumsum(g_n, axis=0)
        E_n = tl.exp(logE_n)

        beta_n = tl.where(
            is_write,
            tl.load(beta_ptr + bh * T + toks_n, mask=is_write & valid_n, other=0.0).to(tl.float32),
            tl.load(gamma_ptr + bh * T + toks_n, mask=is_erase & valid_n, other=0.0).to(tl.float32),
        )

        # ---- interleaved U rows: e_0, k_0, e_1, k_1, ... ----
        e_ptrs = e_base + toks_n[:, None] * K + offs_k[None, :]
        k_ptrs = k_base + toks_n[:, None] * K + offs_k[None, :]
        U = tl.load(
            e_ptrs,
            mask=is_erase[:, None] & valid_n[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32) + tl.load(
            k_ptrs,
            mask=is_write[:, None] & valid_n[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        v_ptrs = v_base + toks_n[:, None] * V + offs_v[None, :]
        x_n = tl.load(
            v_ptrs,
            mask=is_write[:, None] & valid_n[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)

        # rhs = beta * (x - E * U @ S_in)
        proj = tl.dot(U, S, input_precision="tf32")
        rhs = beta_n[:, None] * (x_n - E_n[:, None] * proj)

        # B[n, j] = beta_n * (E_n / E_j) * (u_n . u_j),  j < n
        gram = tl.dot(U, tl.trans(U), input_precision="tf32")
        diff = logE_n[:, None] - logE_n[None, :]
        ratio = tl.exp(tl.where(strict, diff, 0.0))
        Bm = tl.where(strict, gram * ratio * beta_n[:, None], 0.0)
        Mmat = eye + Bm

        # Newton inverse of the unit lower-triangular Mmat.
        Minv = eye
        for _ in range(6):
            mx = tl.dot(Mmat, Minv, input_precision="tf32")
            Minv = tl.dot(Minv, eye + eye - mx, input_precision="tf32")

        Z = tl.dot(Minv, rhs, input_precision="tf32")

        # ---- outputs for this chunk ----
        q_ptrs = q_base + toks_c[:, None] * K + offs_k[None, :]
        qt = tl.load(
            q_ptrs, mask=valid_c[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)
        proj_q = tl.dot(qt, S, input_precision="tf32")
        qgram = tl.dot(qt, tl.trans(U), input_precision="tf32")
        dec_mask = tok_n[None, :] <= offs_c[:, None]
        dec_diff = logE_c[:, None] - logE_n[None, :]
        dec = tl.exp(tl.where(dec_mask, dec_diff, 0.0))
        dec = tl.where(dec_mask, dec, 0.0)
        intra = tl.dot(dec * qgram, Z, input_precision="tf32")
        out_t = scale * (tl.exp(logE_c)[:, None] * proj_q + intra)
        o_ptrs = o_base + toks_c[:, None] * V + offs_v[None, :]
        tl.store(o_ptrs, out_t, mask=valid_c[:, None] & mask_v[None, :])

        # ---- chunk-final state ----
        z_scaled = Z * tl.exp(logE_total - logE_n)[:, None]
        S = E_total * S + tl.dot(tl.trans(U), z_scaled, input_precision="tf32")


# ---------------------------------------------------------------------------
# Plain recurrent kernel, used as a fallback for shapes the chunk kernel
# is not specialised for.
# ---------------------------------------------------------------------------
@triton.jit
def _recurrent_delta_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    out_ptr,
    T,
    K,
    V,
    scale,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid_v = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V

    bh = pid_bh.to(tl.int64)
    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    o_base = out_ptr + bh * T * V
    s_base = g_ptr + bh * T

    S = tl.zeros((BK, BV), dtype=tl.float32)

    for t in range(T):
        tk = t * K
        tv = t * V
        kt = tl.load(k_base + tk + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        et = tl.load(e_base + tk + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        qt = tl.load(q_base + tk + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        vt = tl.load(v_base + tv + offs_v, mask=mask_v, other=0.0).to(tl.float32)

        decay = tl.exp(tl.load(s_base + t))
        beta_t = tl.load(beta_ptr + bh * T + t)
        gamma_t = tl.load(gamma_ptr + bh * T + t)

        S = S * decay
        erased = tl.sum(S * et[:, None], axis=0)
        S = S - et[:, None] * (gamma_t * erased)[None, :]
        pred = tl.sum(S * kt[:, None], axis=0)
        S = S + kt[:, None] * (beta_t * (vt - pred))[None, :]
        ot = tl.sum(S * (qt * scale)[:, None], axis=0)
        tl.store(o_base + tv + offs_v, ot, mask=mask_v)


def _next_power_of_two(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    if T == 0 or B == 0 or H == 0 or V == 0 or K == 0:
        return torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)

    input_device = q.device
    work_device = input_device if input_device.type == "cuda" else torch.device("cuda")

    q_c = q.to(work_device).contiguous()
    k_c = k.to(work_device).contiguous()
    v_c = v.to(work_device).contiguous()
    e_c = e.to(work_device).contiguous()
    g_c = g.to(work_device).contiguous()
    beta_c = beta.to(work_device).contiguous()
    gamma_c = gamma.to(work_device).contiguous()

    BH = B * H
    q_f = q_c.reshape(BH, T, K)
    k_f = k_c.reshape(BH, T, K)
    v_f = v_c.reshape(BH, T, V)
    e_f = e_c.reshape(BH, T, K)
    g_f = g_c.reshape(BH, T)
    beta_f = beta_c.reshape(BH, T)
    gamma_f = gamma_c.reshape(BH, T)

    out = torch.empty((BH, T, V), device=work_device, dtype=torch.float32)

    BK = max(16, _next_power_of_two(K))
    use_chunk = work_device.type == "cuda" and BK <= 128 and V <= 128

    if use_chunk:
        if V <= 16:
            BV = 16
        elif V <= 64:
            BV = _next_power_of_two(V)
        else:
            BV = 64
        num_v_blocks = triton.cdiv(V, BV)
        num_warps = 8 if BK * BV >= 8192 else 4
        _chunk_delta_kernel[(num_v_blocks, BH)](
            q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f, out,
            T, K, V, float(scale),
            BK=BK, BV=BV, BC=32, BN=64,
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        BV = 16 if BK >= 256 else 32
        num_v_blocks = triton.cdiv(V, BV)
        num_warps = 8 if BK * BV > 4096 else 4
        _recurrent_delta_kernel[(num_v_blocks, BH)](
            q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f, out,
            T, K, V, float(scale),
            BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=2,
        )

    out = out.reshape(B, H, T, V)
    if out.device != input_device:
        out = out.to(input_device)
    return out
```

The main optimisation is a chunkwise reformulation: each token is split into erase/write micro-steps, turning a 32-token chunk into a 64×64 unit lower-triangular system whose inverse is computed exactly (in exact arithmetic) by six Newton iterations, so the sequential dependency only remains between 64 chunks instead of 2048 steps and everything else becomes TF32 tensor-core matmuls against the state. Decay is only ever applied through ratios ≤ 1, avoiding division by tiny cumulative decays. The value dimension is split into blocks of 64 for parallelism, and the original recurrent kernel is retained as a fallback for out-of-range shapes; the main uncertainty is register pressure from holding the state plus several 64×128 tiles per program.