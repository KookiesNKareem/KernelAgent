```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# (I - N)^-1 = I + N + N^2 + ... for strictly triangular (nilpotent) N,
# evaluated exactly (in exact arithmetic) by repeated squaring.
# ---------------------------------------------------------------------------
@triton.jit
def _series_sum(N, BT: tl.constexpr, STEPS: tl.constexpr):
    offs = tl.arange(0, BT)
    S = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0).to(tl.float32)
    P = N
    for _ in range(STEPS):
        S = S + tl.dot(P, S, input_precision="tf32")
        P = tl.dot(P, P, input_precision="tf32")
    return S


# Dot product with ~16-bit effective mantissa at tensor-core speed:
# split each fp32 operand into a bf16 high part plus a bf16 residual.
@triton.jit
def _dot_split(a, b):
    a_hi = a.to(tl.bfloat16)
    a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
    b_hi = b.to(tl.bfloat16)
    b_lo = (b - b_hi.to(tl.float32)).to(tl.bfloat16)
    acc = tl.dot(a_hi, b_hi)
    acc += tl.dot(a_hi, b_lo)
    acc += tl.dot(a_lo, b_hi)
    return acc


# ---------------------------------------------------------------------------
# Chunkwise kernel.
#
# Inside a chunk, with F_i = diag(exp(cumsum g)) and P_i = R_0 ... R_i,
# R_j = I - gamma_j u_j u_j^T, the transformed state
#     Shat_i = F_i^-1 S_i P_i^-1      (rows rescaled by exp(shift))
# obeys an ordinary delta-rule recurrence with keys k_i/F_i, k_i*F_i and
# transformed values vhat_i = P_i^-T v_i.  vhat, the in-chunk outputs and
# the end-of-chunk state are recovered from Shat with unit-triangular
# systems in the Gram matrix U U^T, solved by _series_sum.
#
# The K dimension is processed in KB=32 slices (the [K, V] state itself is
# kept in a global scratch buffer between the two slice loops), so no
# [128, 128] operand is ever staged for a tl.dot: this is what keeps the
# shared-memory usage below the L4 limit.  One program handles one
# (batch, head) and carries the state across chunks via the scratch.
# ---------------------------------------------------------------------------
@triton.jit
def _chunk_kernel(
    q_ptr, k_ptr, v_ptr, u_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr,
    state_ptr, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, KB: tl.constexpr,
    BV: tl.constexpr, NUM_KB: tl.constexpr, STEPS: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    off_t = tl.arange(0, BT)
    off_kb = tl.arange(0, KB)
    off_v = tl.arange(0, BV)
    row = off_t[:, None]
    col = off_t[None, :]
    causal = col <= row
    upper = row <= col
    strict_lower = col < row
    strict_upper = col > row

    for start in range(0, T, BT):
        rem = T - start
        m_t = off_t < rem
        base_v = bh * T * BV + (start + off_t)[:, None] * BV + off_v[None, :]

        vv = tl.load(v_ptr + base_v, mask=m_t[:, None], other=0.0).to(tl.float32)
        uu = tl.load(u_ptr + base_v, mask=m_t[:, None], other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + bh * T + start + off_t, mask=m_t, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + bh * T + start + off_t, mask=m_t, other=0.0).to(tl.float32)

        gram_u = tl.dot(uu, tl.trans(uu), input_precision="tf32")

        # ---- pass 1 over K slices: accumulate the BT x BT couplings and
        # the contributions of the incoming state.
        ka = tl.zeros([BT, BT], dtype=tl.float32)
        aq = tl.zeros([BT, BT], dtype=tl.float32)
        pred = tl.zeros([BT, BV], dtype=tl.float32)
        opred = tl.zeros([BT, BV], dtype=tl.float32)
        for kb in range(0, NUM_KB):
            koff = kb * KB
            base_k = bh * T * BK + (start + off_t)[:, None] * BK + (koff + off_kb)[None, :]
            qq = tl.load(q_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)
            kk = tl.load(k_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)
            gg = tl.load(g_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)

            cum_g = tl.cumsum(gg, axis=0)
            total = tl.sum(tl.where(off_t[:, None] == rem - 1, cum_g, 0.0), axis=0)
            shift = 0.5 * total
            rel = cum_g - shift[None, :]
            exp_rel = tl.exp(rel)
            a_key = kk * tl.exp(-rel)
            c_key = kk * exp_rel
            q_tilde = qq * exp_rel

            s_blk = tl.load(
                state_ptr + bh * BK * BV + (koff + off_kb)[:, None] * BV + off_v[None, :]
            )
            s_hat = s_blk * tl.exp(shift)[:, None]

            ka += tl.dot(c_key, tl.trans(a_key), input_precision="tf32")
            aq += tl.dot(q_tilde, tl.trans(a_key), input_precision="tf32")
            pred += _dot_split(c_key, s_hat)
            opred += _dot_split(q_tilde, s_hat)

        # vhat_i = P_i^-T v_i = v_i + sum_{j<=i} alpha_{j,i} u_j.
        delta = gamma / (1.0 - gamma)
        b_alpha = tl.where(
            upper,
            delta[:, None] * tl.dot(uu, tl.trans(vv), input_precision="tf32"),
            0.0,
        )
        alpha_op = _series_sum(
            delta[:, None] * tl.where(strict_upper, gram_u, 0.0), BT, STEPS
        )
        alpha = tl.dot(alpha_op, b_alpha, input_precision="tf32")
        v_hat = vv + tl.dot(tl.trans(alpha), uu, input_precision="tf32")

        # Ordinary delta-rule solve inside the chunk.
        rhs = beta[:, None] * (v_hat - pred)
        left_op = _series_sum(
            -(beta[:, None] * tl.where(strict_lower, ka, 0.0)), BT, STEPS
        )
        w = tl.dot(left_op, rhs, input_precision="tf32")

        # Transformed outputs, then apply P_i^T row by row.
        aq = tl.where(causal, aq, 0.0)
        o_hat = opred + tl.dot(aq, w, input_precision="tf32")
        b_out = tl.where(
            upper,
            gamma[:, None] * tl.dot(uu, tl.trans(o_hat), input_precision="tf32"),
            0.0,
        )
        out_op = _series_sum(
            -(gamma[:, None] * tl.where(strict_lower, gram_u, 0.0)), BT, STEPS
        )
        coeff = tl.dot(out_op, b_out, input_precision="tf32")
        coeff = tl.where(upper, coeff, 0.0)
        out = (o_hat - tl.dot(tl.trans(coeff), uu, input_precision="tf32")) * scale
        tl.store(o_ptr + base_v, out, mask=m_t[:, None])

        # ---- pass 2 over K slices: end-of-chunk state Shat_last @ P_last,
        # then undo the decay scaling and write the new state back.
        end_op = _series_sum(
            -(tl.where(strict_upper, gram_u, 0.0) * gamma[None, :]), BT, STEPS
        )
        for kb in range(0, NUM_KB):
            koff = kb * KB
            base_k = bh * T * BK + (start + off_t)[:, None] * BK + (koff + off_kb)[None, :]
            kk = tl.load(k_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)
            gg = tl.load(g_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)

            cum_g = tl.cumsum(gg, axis=0)
            total = tl.sum(tl.where(off_t[:, None] == rem - 1, cum_g, 0.0), axis=0)
            shift = 0.5 * total
            rel = cum_g - shift[None, :]
            a_key = kk * tl.exp(-rel)

            s_ptr = state_ptr + bh * BK * BV + (koff + off_kb)[:, None] * BV + off_v[None, :]
            s_hat = tl.load(s_ptr) * tl.exp(shift)[:, None]

            x_hat = s_hat + tl.dot(tl.trans(a_key), w, input_precision="tf32")
            e_mat = _dot_split(x_hat, tl.trans(uu))
            d_coeff = tl.dot(e_mat * gamma[None, :], end_op, input_precision="tf32")
            x_end = x_hat - tl.dot(d_coeff, uu, input_precision="tf32")
            tl.store(s_ptr, x_end * tl.exp(shift)[:, None])


# ---------------------------------------------------------------------------
# Direct recurrent kernel (fallback for shapes / input regimes where the
# chunk transform is not applicable: K or V > 128, gamma near 1, extreme
# decay, out-of-range gates).  One program per (batch, head).
# ---------------------------------------------------------------------------
@triton.jit
def _recurrent_kernel(
    q_ptr, k_ptr, v_ptr, u_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr,
    T, scale,
    BK: tl.constexpr, BV: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    off_k = tl.arange(0, BK)
    off_v = tl.arange(0, BV)
    S = tl.zeros([BK, BV], dtype=tl.float32)

    for t in range(T):
        r = bh * T + t
        g = tl.load(g_ptr + r * BK + off_k).to(tl.float32)
        u = tl.load(u_ptr + r * BV + off_v).to(tl.float32)
        kk = tl.load(k_ptr + r * BK + off_k).to(tl.float32)
        vv = tl.load(v_ptr + r * BV + off_v).to(tl.float32)
        qq = tl.load(q_ptr + r * BK + off_k).to(tl.float32)
        beta = tl.load(beta_ptr + r).to(tl.float32)
        gamma = tl.load(gamma_ptr + r).to(tl.float32)

        S = S * tl.exp(g)[:, None]
        su = tl.sum(S * u[None, :], axis=1)
        S = S - gamma * su[:, None] * u[None, :]
        pred = tl.sum(S * kk[:, None], axis=0)
        S = S + kk[:, None] * (beta * (vv - pred))[None, :]
        out = scale * tl.sum(S * qq[:, None], axis=0)
        tl.store(o_ptr + r * BV + off_v, out)


def _pad_last_dim(x, padded_size):
    if x.shape[-1] == padded_size:
        return x.contiguous()
    result = torch.zeros((*x.shape[:-1], padded_size), device=x.device, dtype=x.dtype)
    result[..., : x.shape[-1]] = x
    return result


def kernel_function(q, k, v, u, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    if B * H == 0 or T == 0 or K == 0 or V == 0:
        return torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)

    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))

    q_ = _pad_last_dim(q, BK).reshape(B * H, T, BK)
    k_ = _pad_last_dim(k, BK).reshape(B * H, T, BK)
    g_ = _pad_last_dim(g, BK).reshape(B * H, T, BK)
    v_ = _pad_last_dim(v, BV).reshape(B * H, T, BV)
    u_ = _pad_last_dim(u, BV).reshape(B * H, T, BV)
    beta_ = beta.contiguous().reshape(B * H, T)
    gamma_ = gamma.contiguous().reshape(B * H, T)
    out_ = torch.empty((B * H, T, BV), device=q.device, dtype=torch.float32)

    # The chunk transform divides by (1 - gamma) and exponentiates the
    # cumulative decay, so use it only where it is well conditioned; pick
    # a smaller chunk when gamma is close to 1 or the decay is strong.
    # Everything else goes to the direct recurrent kernel.
    use_chunk = False
    BT = 32
    if BK <= 128 and BV <= 128:
        b_lo, b_hi = (float(x) for x in beta.aminmax())
        ga_lo, ga_hi = (float(x) for x in gamma.aminmax())
        g_lo, g_hi = (float(x) for x in g.aminmax())
        if ga_hi > 0.99:
            BT = 16
        use_chunk = (
            0.0 <= b_lo and b_hi <= 1.0
            and 0.0 <= ga_lo and ga_hi <= 0.9999
            and -30.0 <= g_lo and g_hi <= 0.0
            and -g_lo * BT <= 140.0
        )

    if use_chunk:
        KB = min(32, BK)
        state = torch.zeros((B * H, BK, BV), device=q.device, dtype=torch.float32)
        _chunk_kernel[(B * H,)](
            q_, k_, v_, u_, g_, beta_, gamma_, out_, state,
            T, float(scale),
            BT=BT, BK=BK, KB=KB, BV=BV, NUM_KB=BK // KB,
            STEPS=(BT - 1).bit_length(),
            num_warps=8, num_stages=1,
        )
    else:
        state_elements = BK * BV
        if state_elements <= 4096:
            num_warps = 4
        elif state_elements <= 16384:
            num_warps = 16
        else:
            num_warps = 32
        _recurrent_kernel[(B * H,)](
            q_, k_, v_, u_, g_, beta_, gamma_, out_,
            T, float(scale),
            BK=BK, BV=BV,
            num_warps=num_warps, num_stages=1,
        )

    return out_[:, :, :V].reshape(B, H, T, V)
```

The previous chunk kernel staged a full `[128, 128]` state operand (plus its bf16 hi/lo splits) for the `k^T S` / `q^T S` products, which is what pushed shared memory to 114688 bytes over the 101376 limit. The fix keeps the same chunkwise mathematics but slices the K dimension into blocks of 32, carrying the state through a global scratch buffer between the two slice passes, so the largest staged dot operand is now `[32, 128]` and shared usage fits comfortably. Everything else — the triangular-series solves, the bf16 hi/lo split dots, the adaptive chunk size, and the recurrent fallback for out-of-regime inputs — is unchanged.