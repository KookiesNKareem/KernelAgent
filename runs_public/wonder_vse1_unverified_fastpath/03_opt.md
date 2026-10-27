```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triangular series:  return I + N + N^2 + ... = (I - N)^-1.
# Used only for strictly triangular (hence nilpotent) N, so the series is
# exact in exact arithmetic (evaluated by repeated squaring).
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


# Accurate dot for the large [*, 128] x [128, 128] products: split each
# fp32 operand into a bf16 high part plus a bf16 residual and accumulate
# the three leading products in fp32.  This keeps tensor-core speed and
# bf16 shared-memory traffic while giving ~16-bit effective mantissa.
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
#     Shat_i = F_i^-1 S_i P_i^-1      (row-rescaled by exp(shift))
# obeys an ordinary delta-rule recurrence with keys k_i/F_i, k_i*F_i and
# transformed values vhat_i = P_i^-T v_i.  vhat, the in-chunk outputs and
# the end-of-chunk state are all recovered from Shat with unit-triangular
# systems in the Gram matrix U U^T, solved by _series_sum.  One program
# handles one (batch, head) and carries the fp32 state across chunks.
# ---------------------------------------------------------------------------
@triton.jit
def _chunk_kernel(
    q_ptr, k_ptr, v_ptr, u_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr,
    T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    STEPS: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    off_t = tl.arange(0, BT)
    off_k = tl.arange(0, BK)
    off_v = tl.arange(0, BV)
    row = off_t[:, None]
    col = off_t[None, :]
    causal = col <= row          # source index <= destination index
    upper = row <= col           # coefficient index <= timestep index
    strict_lower = col < row
    strict_upper = col > row

    S = tl.zeros([BK, BV], dtype=tl.float32)

    for start in range(0, T, BT):
        rem = T - start
        m_t = off_t < rem
        base_k = bh * T * BK + (start + off_t)[:, None] * BK + off_k[None, :]
        base_v = bh * T * BV + (start + off_t)[:, None] * BV + off_v[None, :]

        q = tl.load(q_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)
        kk = tl.load(k_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)
        gg = tl.load(g_ptr + base_k, mask=m_t[:, None], other=0.0).to(tl.float32)
        vv = tl.load(v_ptr + base_v, mask=m_t[:, None], other=0.0).to(tl.float32)
        uu = tl.load(u_ptr + base_v, mask=m_t[:, None], other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + bh * T + start + off_t, mask=m_t, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + bh * T + start + off_t, mask=m_t, other=0.0).to(tl.float32)

        # Balanced per-channel decay scaling for this chunk.
        cum_g = tl.cumsum(gg, axis=0)
        total = tl.sum(tl.where(off_t[:, None] == rem - 1, cum_g, 0.0), axis=0)
        shift = 0.5 * total
        rel = cum_g - shift[None, :]
        exp_rel = tl.exp(rel)
        a_key = kk * tl.exp(-rel)        # k_i / F_i
        c_key = kk * exp_rel             # k_i * F_i
        q_tilde = q * exp_rel            # q_i * F_i
        s_hat = S * tl.exp(shift)[:, None]

        gram_u = tl.dot(uu, tl.trans(uu), input_precision="tf32")

        # vhat_i = P_i^-T v_i = v_i + sum_{j<=i} alpha_{j,i} u_j.
        delta = gamma / (1.0 - gamma)
        b_alpha = tl.where(
            upper,
            delta[:, None] * tl.dot(uu, tl.trans(vv), input_precision="tf32"),
            0.0,
        )
        alpha_op = _series_sum(delta[:, None] * tl.where(strict_upper, gram_u, 0.0), BT, STEPS)
        alpha = tl.dot(alpha_op, b_alpha, input_precision="tf32")
        v_hat = vv + tl.dot(tl.trans(alpha), uu, input_precision="tf32")

        # Ordinary delta-rule solve inside the chunk.
        rhs = beta[:, None] * (v_hat - _dot_split(c_key, s_hat))
        ka = tl.dot(c_key, tl.trans(a_key), input_precision="tf32")
        left_op = _series_sum(-(beta[:, None] * tl.where(strict_lower, ka, 0.0)), BT, STEPS)
        w = tl.dot(left_op, rhs, input_precision="tf32")

        # Transformed outputs, then apply P_i^T row by row.
        aq = tl.where(causal, tl.dot(q_tilde, tl.trans(a_key), input_precision="tf32"), 0.0)
        o_hat = _dot_split(q_tilde, s_hat) + tl.dot(aq, w, input_precision="tf32")
        b_out = tl.where(
            upper,
            gamma[:, None] * tl.dot(uu, tl.trans(o_hat), input_precision="tf32"),
            0.0,
        )
        out_op = _series_sum(-(gamma[:, None] * tl.where(strict_lower, gram_u, 0.0)), BT, STEPS)
        coeff = tl.dot(out_op, b_out, input_precision="tf32")
        coeff = tl.where(upper, coeff, 0.0)
        out = (o_hat - tl.dot(tl.trans(coeff), uu, input_precision="tf32")) * scale
        tl.store(o_ptr + base_v, out, mask=m_t[:, None])

        # End-of-chunk state: Shat_last @ P_last, then undo decay scaling.
        x_hat = s_hat + _dot_split(tl.trans(a_key), w)
        e_mat = _dot_split(x_hat, tl.trans(uu))                    # [K, BT]
        end_op = _series_sum(-(tl.where(strict_upper, gram_u, 0.0) * gamma[None, :]), BT, STEPS)
        d_coeff = tl.dot(e_mat * gamma[None, :], end_op, input_precision="tf32")
        x_end = x_hat - tl.dot(d_coeff, uu, input_precision="tf32")
        S = x_end * tl.exp(shift)[:, None]


# ---------------------------------------------------------------------------
# Direct recurrent kernel (fallback for shapes / input regimes where the
# chunk transform is not applicable, e.g. K or V > 128, gamma near 1, or
# very strong decay).  One program per (batch, head), state in registers.
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
    # cumulative decay, so use it only in the regime where it is well
    # conditioned; everything else goes to the direct recurrent kernel.
    use_chunk = False
    if BK <= 128 and BV <= 128:
        b_lo, b_hi = (float(x) for x in beta.aminmax())
        ga_lo, ga_hi = (float(x) for x in gamma.aminmax())
        g_lo, g_hi = (float(x) for x in g.aminmax())
        g_mean = float(g.float().mean())
        use_chunk = (
            0.0 <= b_lo and b_hi <= 1.0
            and 0.0 <= ga_lo and ga_hi <= 0.999
            and -30.0 <= g_lo and g_hi <= 0.0
            and abs(g_mean) * 32.0 <= 120.0
        )

    if use_chunk:
        BT = 32
        _chunk_kernel[(B * H,)](
            q_, k_, v_, u_, g_, beta_, gamma_, out_,
            T, float(scale),
            BT=BT, BK=BK, BV=BV, STEPS=(BT - 1).bit_length(),
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

The 36 ms kernel was a step-by-step recurrence with only 32 programs and three register-tile reductions per timestep; this version reformulates each 32-step chunk as tensor-core matrix algebra: cumulative decay is folded into oppositely scaled keys/queries, the right-hand erase product is inverted away (turning the chunk into a standard delta-rule recurrence whose couplings are `32x32` Gram systems solved exactly by a nilpotent Neumann series), and the state is carried between chunks in fp32. The large state products use a bf16 hi/lo split with fp32 accumulation, which keeps tensor-core speed and small shared-memory tiles (`BT=32`, `num_stages=1`) while retaining well over tf32 precision, and out-of-regime inputs (oversized dims, `gamma` near 1, extreme decay, out-of-range gates) fall back to the previously passing recurrent kernel. I am unsure of the exact wall-clock on the L4 since occupancy is still one program per `(batch, head)`, and of accuracy in the unlikely event the timed inputs have many `gamma` values extremely close to 1, where the transformed representation is inherently ill-conditioned.