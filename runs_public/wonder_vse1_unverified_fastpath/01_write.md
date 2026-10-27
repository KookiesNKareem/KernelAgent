```python
import torch
import triton
import triton.language as tl


@triton.jit
def _inv_unitri(N, BT: tl.constexpr, STEPS: tl.constexpr):
    # Return (I - N)^(-1) = I + N + N^2 + ... for strictly triangular N.
    offs = tl.arange(0, BT)
    eye = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0).to(tl.float32)
    S = eye
    P = N
    for _ in range(STEPS):
        S = S + tl.dot(P, S, input_precision="ieee")
        P = tl.dot(P, P, input_precision="ieee")
    return S


@triton.jit
def _layer_kernel(
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
    causal = col >= row
    strict_lower = col < row
    strict_upper = col > row
    mask_k = off_k < tl.num_programs(0) * 0 + 1000000000  # placeholder, replaced below
    # Real dimension masks are recomputed from pointer-independent constexpr-free
    # values passed through the loads' index expressions; K/V are encoded by
    # BK/BV tile bounds in the caller for the supported power-of-two padded
    # tensors.  (See kernel_function: it pads logically via masked loads using
    # K and V arguments below.)
    S = tl.zeros([BK, BV], dtype=tl.float32)

    # The loop body is inlined in kernel_function-generated specialization via
    # closure over constexprs is not supported by Triton, so K and V are passed
    # as regular arguments through the following symbols.
    # NOTE: this kernel is launched only by kernel_function below, which sets
    # BK/BV to the padded dimensions and guarantees masked loads are used.
    for start in range(0, T, BT):
        rem = T - start
        m_t = off_t < rem

        # Loads are patched by the wrapper's tensor layout: tensors are
        # contiguous [BH, T, D], and off_k/off_v beyond the true dimension
        # are never present because BK/BV equal the padded true dimensions
        # chosen as next_power_of_2 and the tensors are viewed with those
        # exact last dimensions when needed.  For non-power-of-two sizes the
        # wrapper pads the last dimension (see kernel_function).
        q = tl.load(
            q_ptr + bh * T * BK + (start + off_t)[:, None] * BK + off_k[None, :],
            mask=m_t[:, None], other=0.0,
        ).to(tl.float32)
        kk = tl.load(
            k_ptr + bh * T * BK + (start + off_t)[:, None] * BK + off_k[None, :],
            mask=m_t[:, None], other=0.0,
        ).to(tl.float32)
        gg = tl.load(
            g_ptr + bh * T * BK + (start + off_t)[:, None] * BK + off_k[None, :],
            mask=m_t[:, None], other=0.0,
        ).to(tl.float32)
        vv = tl.load(
            v_ptr + bh * T * BV + (start + off_t)[:, None] * BV + off_v[None, :],
            mask=m_t[:, None], other=0.0,
        ).to(tl.float32)
        uu = tl.load(
            u_ptr + bh * T * BV + (start + off_t)[:, None] * BV + off_v[None, :],
            mask=m_t[:, None], other=0.0,
        ).to(tl.float32)
        beta = tl.load(beta_ptr + bh * T + start + off_t, mask=m_t, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + bh * T + start + off_t, mask=m_t, other=0.0).to(tl.float32)

        # Absorb the per-channel decay into scaled keys/queries.  The shift
        # (half the chunk's total log-decay) keeps the exponentials balanced.
        cum_g = tl.cumsum(gg, axis=0)
        last_row = tl.sum(
            tl.where(off_t[:, None] == rem - 1, cum_g, 0.0), axis=0
        )
        shift = 0.5 * last_row
        rel = cum_g - shift[None, :]
        decay = tl.exp(rel)
        a_key = kk * tl.exp(-rel)          # k_i / F_i
        c_key = kk * decay                 # k_i * F_i
        q_tilde = q * decay                # q_i * F_i
        s_hat = S * tl.exp(-shift)[:, None]

        gram_u = tl.dot(uu, tl.trans(uu), input_precision="ieee")

        # Values transformed by (P_i^T)^-1, P_i = R_0 ... R_i:
        # alpha_s - delta_s * sum_{r>s} (u_s.u_r) alpha_r = delta_s (u_s.v_i)
        delta = gamma / (1.0 - gamma)
        b_alpha = tl.where(
            causal,
            delta[:, None] * tl.dot(uu, tl.trans(vv), input_precision="ieee"),
            0.0,
        )
        inv_alpha = _inv_unitri(
            delta[:, None] * tl.where(strict_upper, gram_u, 0.0), BT, STEPS
        )
        alpha = tl.dot(inv_alpha, b_alpha, input_precision="ieee")
        v_hat = vv + tl.dot(tl.trans(alpha), uu, input_precision="ieee")

        # Ordinary (transformed) delta-rule solve inside the chunk.
        rhs = beta[:, None] * (
            v_hat - tl.dot(c_key, s_hat, input_precision="ieee")
        )
        ka = tl.dot(c_key, tl.trans(a_key), input_precision="ieee")
        inv_left = _inv_unitri(
            -(beta[:, None] * tl.where(strict_lower, ka, 0.0)), BT, STEPS
        )
        h = tl.dot(inv_left, rhs, input_precision="ieee")

        # Outputs in the transformed state, then apply P_i^T row-wise.
        aq = tl.where(
            causal,
            tl.dot(q_tilde, tl.trans(a_key), input_precision="ieee"),
            0.0,
        )
        o_hat = tl.dot(q_tilde, s_hat, input_precision="ieee") + tl.dot(
            aq, h, input_precision="ieee"
        )
        b_out = tl.where(
            causal,
            gamma[:, None] * tl.dot(uu, tl.trans(o_hat), input_precision="ieee"),
            0.0,
        )
        inv_out = _inv_unitri(
            -(gamma[:, None] * tl.where(strict_lower, gram_u, 0.0)), BT, STEPS
        )
        coeff = tl.dot(inv_out, b_out, input_precision="ieee")
        coeff = tl.where(causal, coeff, 0.0)
        out = (o_hat - tl.dot(tl.trans(coeff), uu, input_precision="ieee")) * scale

        tl.store(
            o_ptr + bh * T * BV + (start + off_t)[:, None] * BV + off_v[None, :],
            out,
            mask=m_t[:, None],
        )

        # End-of-chunk transformed state, apply P_{C-1} on the right, then
        # undo the decay scaling to obtain the true next state.
        x_hat = s_hat + tl.dot(tl.trans(a_key), h, input_precision="ieee")
        d_coeff = tl.dot(x_hat, tl.trans(uu), input_precision="ieee")
        inv_ag = _inv_unitri(-tl.where(strict_upper, gram_u, 0.0), BT, STEPS)
        d_coeff = tl.dot(d_coeff, inv_ag, input_precision="ieee")
        x_end = x_hat - tl.dot(d_coeff, uu, input_precision="ieee")
        S = x_end * tl.exp(shift)[:, None]


def _pad_last_dim(x, padded):
    # Cheap elementwise/layout prep only: zero-pad a non-power-of-two last dim.
    if x.shape[-1] == padded:
        return x.contiguous()
    y = torch.zeros(
        (*x.shape[:-1], padded), device=x.device, dtype=x.dtype
    )
    y[..., : x.shape[-1]] = x
    return y


def kernel_function(q, k, v, u, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B * H == 0 or T == 0:
        return out

    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))
    BT = 64

    # Pad K/V to the kernel tile sizes (identity when already 128, as in the
    # target workload); padded channels are identically zero.
    q_ = _pad_last_dim(q, BK).reshape(B * H, T, BK)
    k_ = _pad_last_dim(k, BK).reshape(B * H, T, BK)
    g_ = _pad_last_dim(g, BK).reshape(B * H, T, BK)
    v_ = _pad_last_dim(v, BV).reshape(B * H, T, BV)
    u_ = _pad_last_dim(u, BV).reshape(B * H, T, BV)
    beta_ = beta.contiguous().reshape(B * H, T)
    gamma_ = gamma.contiguous().reshape(B * H, T)
    out_ = torch.zeros((B * H, T, BV), device=q.device, dtype=torch.float32)

    steps = (BT - 1).bit_length()
    _layer_kernel[(B * H,)](
        q_, k_, v_, u_, g_, beta_, gamma_, out_,
        T, float(scale),
        BT=BT, BK=BK, BV=BV, STEPS=steps,
        num_warps=8, num_stages=2,
    )
    return out_[:, :, :V].reshape(B, H, T, V)
```

My approach is a chunkwise reformulation: cumulative decay is folded into oppositely scaled keys and queries, the right-hand erase product is removed by transforming the state with its inverse (turning the chunk into a standard delta-rule recurrence), both erase/output transformations are evaluated with unit-triangular Gram-matrix systems solved by a Neumann series, and one Triton program per `(batch, head)` carries the `128x128` state across 64-step chunks using matrix multiplies. I am unsure about numerical robustness for extreme inputs (decay over a chunk large enough to overflow the balanced exponentials, or `gamma` extremely close to 1, where the true transformed quantities are themselves huge), and about register pressure/speed for head dimensions far from the target `K=V=128`, which the tile-per-state design is not optimized for.