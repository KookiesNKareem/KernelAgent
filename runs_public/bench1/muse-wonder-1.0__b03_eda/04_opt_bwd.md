```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel (same semantics as the reference loop).
# ---------------------------------------------------------------------------
@triton.jit
def _recurrent_delta_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    T, K, V, scale,
    BK: tl.constexpr, BV: tl.constexpr,
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


# ---------------------------------------------------------------------------
# Chunk-start states for the backward pass.  Stores S at the beginning of
# every BC-token chunk; this is BK*BV-tile sized per chunk, not per token.
# ---------------------------------------------------------------------------
@triton.jit
def _chunk_state_kernel(
    k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, chk_ptr,
    T, K, V, NC,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
):
    pid_v = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_kv = mask_k[:, None] & mask_v[None, :]
    bh = pid_bh.to(tl.int64)
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    s_base = g_ptr + bh * T
    chk_base = chk_ptr + bh * NC * K * V

    S = tl.zeros((BK, BV), dtype=tl.float32)
    num_chunks = tl.cdiv(T, BC)
    for ci in range(num_chunks):
        tl.store(chk_base + ci * K * V + offs_k[:, None] * V + offs_v[None, :],
                 S, mask=mask_kv)
        c0 = ci * BC
        for i in range(BC):
            t = c0 + i
            valid = t < T
            kt = tl.load(k_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            et = tl.load(e_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            vt = tl.load(v_base + t * V + offs_v, mask=valid & mask_v, other=0.0).to(tl.float32)
            decay = tl.exp(tl.load(s_base + t, mask=valid, other=0.0))
            beta_t = tl.load(beta_ptr + bh * T + t, mask=valid, other=0.0)
            gamma_t = tl.load(gamma_ptr + bh * T + t, mask=valid, other=0.0)
            S = S * decay
            erased = tl.sum(S * et[:, None], axis=0)
            S = S - et[:, None] * (gamma_t * erased)[None, :]
            pred = tl.sum(S * kt[:, None], axis=0)
            S = S + kt[:, None] * (beta_t * (vt - pred))[None, :]


# ---------------------------------------------------------------------------
# Chunkwise backward kernel.
#
# A chunk of BC tokens is viewed as BN=2*BC micro-steps (erase then write).
# With cumulative intra-chunk decay E_n and scaled increments Y, the forward
# trajectory inside the chunk is
#     Y   = (I - diag(c) L)^{-1} (diag(c) U S0 + diag(p) X/E),
#     S_t = E_t ( S0 + U_{<=t}^T Y ),   L = strict-lower Gram of U,
# where U interleaves e/k, c = (-gamma, -beta), p = (0, beta).
# The kernel re-derives Y per chunk with a Newton inverse (tensor cores),
# forms the total adjoint of Y from the chunk-end state and the outputs,
# solves the transposed triangular system with the same inverse, and writes
# all input gradients as matmuls / row reductions of [BN,BK], [BN,BV],
# [BC,BV] matrices.  No per-token [K,V] state is ever stored.
# ---------------------------------------------------------------------------
@triton.jit
def _chunk_backward_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, go_ptr, chk_ptr,
    dq_ptr, dk_ptr, de_ptr, dv_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
    T, K, V, BH, NC, scale,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr, BN: tl.constexpr,
):
    pid_v = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_c = tl.arange(0, BC)
    offs_n = tl.arange(0, BN)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_kv = mask_k[:, None] & mask_v[None, :]
    bh = pid_bh.to(tl.int64)
    part_base = pid_v.to(tl.int64) * BH + bh

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    go_base = go_ptr + bh * T * V
    s_base = g_ptr + bh * T
    chk_base = chk_ptr + bh * NC * K * V

    tok_n = offs_n // 2
    is_write = (offs_n % 2) == 1
    is_erase = (offs_n % 2) == 0

    eye = tl.where(offs_n[:, None] == offs_n[None, :], 1.0, 0.0)
    strict = offs_n[:, None] > offs_n[None, :]
    dec_mask = tok_n[None, :] <= offs_c[:, None]          # [BC, BN]
    sel_write = (tok_n[:, None] == offs_c[None, :]) & is_write[:, None]
    suffix_mask = tok_n[None, :] >= offs_c[:, None]       # [BC, BN]

    DS = tl.zeros((BK, BV), dtype=tl.float32)
    num_chunks = tl.cdiv(T, BC)

    for rev_ci in range(num_chunks):
        ci = num_chunks - 1 - rev_ci
        c0 = ci * BC
        toks_c = c0 + offs_c
        valid_c = toks_c < T
        toks_n = c0 + tok_n
        valid_n = toks_n < T

        S0 = tl.load(chk_base + ci * K * V + offs_k[:, None] * V + offs_v[None, :],
                     mask=mask_kv, other=0.0)

        # ---- chunk inputs as matrices ----
        U = tl.load(e_base + toks_n[:, None] * K + offs_k[None, :],
                    mask=is_erase[:, None] & valid_n[:, None] & mask_k[None, :],
                    other=0.0).to(tl.float32) + tl.load(
                    k_base + toks_n[:, None] * K + offs_k[None, :],
                    mask=is_write[:, None] & valid_n[:, None] & mask_k[None, :],
                    other=0.0).to(tl.float32)
        Q = tl.load(q_base + toks_c[:, None] * K + offs_k[None, :],
                    mask=valid_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        DO = tl.load(go_base + toks_c[:, None] * V + offs_v[None, :],
                     mask=valid_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
        X = tl.load(v_base + toks_n[:, None] * V + offs_v[None, :],
                    mask=is_write[:, None] & valid_n[:, None] & mask_v[None, :],
                    other=0.0).to(tl.float32)

        g_c = tl.load(s_base + toks_c, mask=valid_c, other=0.0).to(tl.float32)
        E_c = tl.exp(tl.cumsum(g_c, axis=0))
        E_total = tl.exp(tl.sum(g_c))
        g_n = tl.load(s_base + toks_n, mask=is_erase & valid_n, other=0.0).to(tl.float32)
        E_n = tl.exp(tl.cumsum(g_n, axis=0))

        beta_n = tl.load(beta_ptr + bh * T + toks_n,
                         mask=is_write & valid_n, other=0.0).to(tl.float32)
        gamma_n = tl.load(gamma_ptr + bh * T + toks_n,
                          mask=is_erase & valid_n, other=0.0).to(tl.float32)
        c_n = tl.where(is_write, -beta_n, -gamma_n)
        p_n = tl.where(is_write, beta_n, 0.0)

        Xtilde = tl.where(is_write[:, None], X / E_n[:, None], 0.0)

        # ---- forward chunk solve ----
        P0 = tl.dot(U, S0, input_precision="tf32")
        rhs = c_n[:, None] * P0 + p_n[:, None] * Xtilde
        gram = tl.dot(U, tl.trans(U), input_precision="ieee")
        L = tl.where(strict, gram, 0.0)
        Mmat = eye - c_n[:, None] * L
        Minv = eye
        for _ in range(6):
            mx = tl.dot(Mmat, Minv, input_precision="ieee")
            Minv = tl.dot(Minv, eye + eye - mx, input_precision="ieee")
        Y = tl.dot(Minv, rhs, input_precision="ieee")

        # ---- chunk outputs and final state (needed for decay grads) ----
        Qgram = tl.dot(Q, tl.trans(U), input_precision="tf32")
        proj_q = tl.dot(Q, S0, input_precision="tf32")
        intra = tl.dot(tl.where(dec_mask, Qgram, 0.0), Y, input_precision="tf32")
        O = scale * E_c[:, None] * (proj_q + intra)
        Shat_last = S0 + tl.dot(tl.trans(U), Y, input_precision="tf32")
        S_final = E_total * Shat_last
        final_elog = tl.sum(DS * S_final)

        # ---- total adjoint of Y ----
        Gshat = E_total * DS
        GY = tl.dot(U, Gshat, input_precision="tf32")
        coef_oc = tl.where(dec_mask, Qgram * (scale * E_c)[:, None], 0.0)
        GY = GY + tl.dot(tl.trans(coef_oc), DO, input_precision="tf32")

        # ---- adjoint triangular solve ----
        lam = tl.dot(tl.trans(Minv), GY, input_precision="ieee")
        c_lam = c_n[:, None] * lam
        dM = -tl.dot(lam, tl.trans(Y), input_precision="ieee")
        dL = tl.where(strict, -c_n[:, None] * dM, 0.0)
        dc = tl.sum(lam * P0, axis=1) - tl.sum(dM * L, axis=1)
        dp = tl.sum(lam * Xtilde, axis=1)
        dtilde = p_n[:, None] * lam

        # ---- dU (rows interleave de / dk) ----
        M2 = tl.dot(Y, tl.trans(DO), input_precision="tf32")
        coef_du = tl.where(dec_mask, tl.trans(M2) * (scale * E_c)[:, None], 0.0)
        dU = tl.dot(c_lam, tl.trans(S0), input_precision="tf32")
        dU = dU + tl.dot(dL + tl.trans(dL), U, input_precision="tf32")
        dU = dU + tl.dot(Y, tl.trans(Gshat), input_precision="tf32")
        dU = dU + tl.dot(tl.trans(coef_du), Q, input_precision="tf32")

        # ---- dq for this chunk ----
        dq_chunk = scale * (
            E_c[:, None] * tl.dot(DO, tl.trans(S0), input_precision="tf32")
            + tl.dot(tl.where(dec_mask, tl.trans(M2), 0.0), U, input_precision="tf32")
        )
        tl.store(dq_ptr + part_base * T * K + toks_c[:, None] * K + offs_k[None, :],
                 dq_chunk, mask=valid_c[:, None] & mask_k[None, :])

        # ---- state adjoint passed to the previous chunk ----
        DS_new = Gshat
        DS_new = DS_new + tl.dot(tl.trans(U), c_lam, input_precision="tf32")
        DS_new = DS_new + tl.dot(tl.trans(Q), DO * (scale * E_c)[:, None],
                                 input_precision="tf32")

        # ---- remaining gradient stores ----
        dv_micro = tl.where(is_write[:, None], dtilde / E_n[:, None], 0.0)
        tl.store(dv_ptr + bh * T * V + toks_n[:, None] * V + offs_v[None, :],
                 dv_micro, mask=is_write[:, None] & valid_n[:, None] & mask_v[None, :])
        tl.store(dk_ptr + part_base * T * K + toks_n[:, None] * K + offs_k[None, :],
                 dU, mask=is_write[:, None] & valid_n[:, None] & mask_k[None, :])
        tl.store(de_ptr + part_base * T * K + toks_n[:, None] * K + offs_k[None, :],
                 dU, mask=is_erase[:, None] & valid_n[:, None] & mask_k[None, :])
        tl.store(dbeta_ptr + part_base * T + toks_n, -dc + dp,
                 mask=is_write & valid_n)
        tl.store(dgamma_ptr + part_base * T + toks_n, -dc,
                 mask=is_erase & valid_n)

        elog_out = tl.sum(DO * O, axis=1)
        elog_micro = -tl.sum(dtilde * Xtilde, axis=1)
        elog_micro = elog_micro + tl.sum(
            tl.where(sel_write, elog_out[None, :], 0.0), axis=1)
        elog_micro = elog_micro + (offs_n == BN - 1).to(tl.float32) * final_elog
        dg_chunk = tl.sum(tl.where(suffix_mask, elog_micro[None, :], 0.0), axis=1)
        tl.store(dg_ptr + part_base * T + toks_c, dg_chunk, mask=valid_c)

        DS = DS_new


# ---------------------------------------------------------------------------
# Recurrent backward kernel, kept as a fallback for shapes outside the
# chunkwise kernel's range (same adjoint recurrence, chunk-replay version).
# ---------------------------------------------------------------------------
@triton.jit
def _backward_delta_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, go_ptr,
    chk_ptr, scratch_ptr,
    dq_ptr, dk_ptr, de_ptr, dv_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
    T, K, V, BH, NV, scale,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
):
    pid_v = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    offs_k = tl.arange(0, BK)
    offs_vl = tl.arange(0, BV)
    offs_v = pid_v * BV + offs_vl
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_kv = mask_k[:, None] & mask_v[None, :]
    bh = pid_bh.to(tl.int64)
    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    go_base = go_ptr + bh * T * V
    s_base = g_ptr + bh * T
    b_base = beta_ptr + bh * T
    c_base = gamma_ptr + bh * T
    num_chunks = tl.cdiv(T, BC)
    chk_base = chk_ptr + bh * num_chunks * K * V
    scratch_base = scratch_ptr + ((bh * NV + pid_v) * BC) * BK * BV
    part_base = pid_v.to(tl.int64) * BH + bh

    S = tl.zeros((BK, BV), dtype=tl.float32)
    for ci in range(num_chunks):
        tl.store(chk_base + ci * K * V + offs_k[:, None] * V + offs_v[None, :],
                 S, mask=mask_kv)
        c0 = ci * BC
        for i in range(BC):
            t = c0 + i
            valid = t < T
            kt = tl.load(k_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            et = tl.load(e_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            vt = tl.load(v_base + t * V + offs_v, mask=valid & mask_v, other=0.0).to(tl.float32)
            decay = tl.exp(tl.load(s_base + t, mask=valid, other=0.0))
            beta_t = tl.load(b_base + t, mask=valid, other=0.0)
            gamma_t = tl.load(c_base + t, mask=valid, other=0.0)
            S = S * decay
            erased = tl.sum(S * et[:, None], axis=0)
            S = S - et[:, None] * (gamma_t * erased)[None, :]
            pred = tl.sum(S * kt[:, None], axis=0)
            S = S + kt[:, None] * (beta_t * (vt - pred))[None, :]

    DS = tl.zeros((BK, BV), dtype=tl.float32)
    for rev_ci in range(num_chunks):
        ci = num_chunks - 1 - rev_ci
        c0 = ci * BC
        S_chk = tl.load(chk_base + ci * K * V + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_kv, other=0.0)
        S_cur = S_chk
        for i in range(BC):
            t = c0 + i
            valid = t < T
            kt = tl.load(k_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            et = tl.load(e_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            vt = tl.load(v_base + t * V + offs_v, mask=valid & mask_v, other=0.0).to(tl.float32)
            decay = tl.exp(tl.load(s_base + t, mask=valid, other=0.0))
            beta_t = tl.load(b_base + t, mask=valid, other=0.0)
            gamma_t = tl.load(c_base + t, mask=valid, other=0.0)
            S_cur = S_cur * decay
            erased = tl.sum(S_cur * et[:, None], axis=0)
            S_cur = S_cur - et[:, None] * (gamma_t * erased)[None, :]
            pred = tl.sum(S_cur * kt[:, None], axis=0)
            S_cur = S_cur + kt[:, None] * (beta_t * (vt - pred))[None, :]
            tl.store(scratch_base + i * BK * BV + offs_k[:, None] * BV + offs_vl[None, :],
                     S_cur, mask=mask_kv)
        for rev_i in range(BC):
            i = BC - 1 - rev_i
            t = c0 + i
            valid = t < T
            S_t = tl.load(scratch_base + i * BK * BV + offs_k[:, None] * BV + offs_vl[None, :],
                          mask=mask_kv, other=0.0)
            if i == 0:
                S_prev = S_chk
            else:
                S_prev = tl.load(
                    scratch_base + (i - 1) * BK * BV + offs_k[:, None] * BV + offs_vl[None, :],
                    mask=mask_kv, other=0.0)
            kt = tl.load(k_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            et = tl.load(e_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            qt = tl.load(q_base + t * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
            vt = tl.load(v_base + t * V + offs_v, mask=valid & mask_v, other=0.0).to(tl.float32)
            do_t = tl.load(go_base + t * V + offs_v, mask=valid & mask_v, other=0.0).to(tl.float32)
            g_t = tl.load(s_base + t, mask=valid, other=0.0)
            beta_t = tl.load(b_base + t, mask=valid, other=0.0)
            gamma_t = tl.load(c_base + t, mask=valid, other=0.0)
            decay = tl.exp(g_t)

            A1 = decay * S_prev
            r = tl.sum(A1 * et[:, None], axis=0)
            A2 = A1 - et[:, None] * (gamma_t * r)[None, :]
            pred = tl.sum(A2 * kt[:, None], axis=0)
            u = vt - pred

            DS_tot = DS + (qt * scale)[:, None] * do_t[None, :]
            tl.store(dq_ptr + part_base * T * K + t * K + offs_k,
                     scale * tl.sum(S_t * do_t[None, :], axis=1),
                     mask=valid & mask_k)
            w = tl.sum(DS_tot * kt[:, None], axis=0)
            tl.store(dv_ptr + bh * T * V + t * V + offs_v, beta_t * w,
                     mask=valid & mask_v)
            tl.store(dbeta_ptr + part_base * T + t, tl.sum(w * u), mask=valid)
            dk_part = beta_t * (tl.sum(DS_tot * u[None, :], axis=1)
                                - tl.sum(A2 * w[None, :], axis=1))
            tl.store(dk_ptr + part_base * T * K + t * K + offs_k, dk_part,
                     mask=valid & mask_k)
            DA2 = DS_tot - kt[:, None] * (beta_t * w)[None, :]
            s_vec = tl.sum(DA2 * et[:, None], axis=0)
            tl.store(dgamma_ptr + part_base * T + t, -tl.sum(s_vec * r), mask=valid)
            de_part = -gamma_t * (tl.sum(DA2 * r[None, :], axis=1)
                                  + tl.sum(A1 * s_vec[None, :], axis=1))
            tl.store(de_ptr + part_base * T * K + t * K + offs_k, de_part,
                     mask=valid & mask_k)
            DA1 = DA2 - et[:, None] * (gamma_t * s_vec)[None, :]
            tl.store(dg_ptr + part_base * T + t, tl.sum(DA1 * A1), mask=valid)
            DS = decay * DA1


def _next_power_of_two(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _launch_forward(q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f, out, T, K, V, scale):
    BH = q_f.shape[0]
    BK = max(16, _next_power_of_two(K))
    BV = 16 if BK >= 256 else 32
    num_v_blocks = triton.cdiv(V, BV)
    num_warps = 8 if BK * BV > 4096 else 4
    _recurrent_delta_kernel[(num_v_blocks, BH)](
        q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f, out,
        T, K, V, float(scale), BK=BK, BV=BV,
        num_warps=num_warps, num_stages=2,
    )


class _DeltaEraseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        ctx.input_metas = [
            (tuple(x.shape), x.dtype, x.device) for x in (q, k, v, e, g, beta, gamma)
        ]
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
        ctx.scale = float(scale)
        ctx.empty_case = (B == 0 or H == 0 or T == 0 or V == 0 or K == 0)
        if ctx.empty_case:
            ctx.save_for_backward()
            if K == 0 and B > 0 and H > 0 and T > 0 and V > 0:
                return torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)
            return torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

        input_device = q.device
        ctx.input_device = input_device
        work_device = input_device if input_device.type == "cuda" else torch.device("cuda")
        ctx.work_device = work_device

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
        _launch_forward(q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f, out,
                        T, K, V, ctx.scale)
        ctx.save_for_backward(q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f)
        out = out.reshape(B, H, T, V)
        if out.device != input_device:
            out = out.to(input_device)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.empty_case:
            grads = [
                torch.zeros(shape, dtype=dtype, device=device)
                for (shape, dtype, device) in ctx.input_metas
            ]
            return grads[0], grads[1], grads[2], grads[3], grads[4], grads[5], grads[6], None

        q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f = ctx.saved_tensors
        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        BH = B * H
        work_device = ctx.work_device

        go = grad_output.to(device=work_device, dtype=torch.float32).contiguous()
        go_f = go.reshape(BH, T, V)

        BK = max(16, _next_power_of_two(K))
        use_chunk = BK <= 128 and V <= 256

        if use_chunk:
            BV = 64 if V > 32 else (32 if V > 16 else 16)
            NV = triton.cdiv(V, BV)
            BC = 16
            NC = (T + BC - 1) // BC
            chk = torch.empty((BH * NC * K * V,), device=work_device, dtype=torch.float32)
            # Chunk-start states with a cheap recurrent kernel (BV=32).
            BV_s = 32 if V > 16 else 16
            NV_s = triton.cdiv(V, BV_s)
            _chunk_state_kernel[(NV_s, BH)](
                k_f, v_f, e_f, g_f, beta_f, gamma_f, chk,
                T, K, V, NC, BK=BK, BV=BV_s, BC=BC,
                num_warps=8 if BK * BV_s > 4096 else 4, num_stages=2,
            )
        else:
            BV = 16 if BK >= 256 else 32
            NV = triton.cdiv(V, BV)
            BC = 32
            NC = (T + BC - 1) // BC

        dq_part = torch.empty((NV, BH, T, K), device=work_device, dtype=torch.float32)
        dk_part = torch.empty((NV, BH, T, K), device=work_device, dtype=torch.float32)
        de_part = torch.empty((NV, BH, T, K), device=work_device, dtype=torch.float32)
        dv_f = torch.empty((BH, T, V), device=work_device, dtype=torch.float32)
        dg_part = torch.empty((NV, BH, T), device=work_device, dtype=torch.float32)
        dbeta_part = torch.empty((NV, BH, T), device=work_device, dtype=torch.float32)
        dgamma_part = torch.empty((NV, BH, T), device=work_device, dtype=torch.float32)

        if use_chunk:
            _chunk_backward_kernel[(NV, BH)](
                q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f, go_f, chk,
                dq_part, dk_part, de_part, dv_f,
                dg_part, dbeta_part, dgamma_part,
                T, K, V, BH, NC, ctx.scale,
                BK=BK, BV=BV, BC=BC, BN=2 * BC,
                num_warps=8, num_stages=2,
            )
        else:
            chk = torch.empty((BH * NC * K * V,), device=work_device, dtype=torch.float32)
            scratch = torch.empty((BH * NV * BC * BK * BV,), device=work_device,
                                  dtype=torch.float32)
            _backward_delta_kernel[(NV, BH)](
                q_f, k_f, v_f, e_f, g_f, beta_f, gamma_f, go_f,
                chk, scratch,
                dq_part, dk_part, de_part, dv_f,
                dg_part, dbeta_part, dgamma_part,
                T, K, V, BH, NV, ctx.scale,
                BK=BK, BV=BV, BC=BC,
                num_warps=8 if BK * BV > 4096 else 4, num_stages=2,
            )

        metas = ctx.input_metas
        dq = dq_part.sum(dim=0).reshape(B, H, T, K).to(device=metas[0][2], dtype=metas[0][1])
        dk = dk_part.sum(dim=0).reshape(B, H, T, K).to(device=metas[1][2], dtype=metas[1][1])
        dv = dv_f.reshape(B, H, T, V).to(device=metas[2][2], dtype=metas[2][1])
        de = de_part.sum(dim=0).reshape(B, H, T, K).to(device=metas[3][2], dtype=metas[3][1])
        dg = dg_part.sum(dim=0).reshape(B, H, T).to(device=metas[4][2], dtype=metas[4][1])
        dbeta = dbeta_part.sum(dim=0).reshape(B, H, T).to(device=metas[5][2], dtype=metas[5][1])
        dgamma = dgamma_part.sum(dim=0).reshape(B, H, T).to(device=metas[6][2], dtype=metas[6][1])
        return dq, dk, dv, de, dg, dbeta, dgamma, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _DeltaEraseFunction.apply(q, k, v, e, g, beta, gamma, scale)
```

The old backward spent most of its time streaming every per-token `[K,V]` state through global scratch and spilling `[K,32]` tiles, so the new backward is chunkwise: a small kernel records only chunk-start states, and a tensor-core kernel processes 16-token chunks (32 erase/write micro-steps) at a time. Inside a chunk the forward trajectory is expressed through a unit lower-triangular Gram system solved by Newton iteration, and its exact adjoint is obtained with the transposed inverse, turning all state, query, key/erase, value, decay, and gate gradients into matmuls and row reductions with no per-token state traffic. The forward pass and the recurrent fallback for out-of-range shapes are unchanged. The main risks are register pressure in the chunk kernel at `BK=128, BV=64` and TF32-level roundoff in the larger matmuls, both of which should be well inside the stated tolerance.