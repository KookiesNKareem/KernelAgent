import torch
import triton
import triton.language as tl


@triton.jit
def _fused_recurrent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    u_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    o_ptr,
    T,
    scale,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)

    off_k = tl.arange(0, BK)
    off_v = tl.arange(0, BV)

    # The whole [K, V] state is held in registers.  No large matrix-multiply
    # tiles are materialised, so this kernel needs only the small amount of
    # shared memory used internally by the two vector reductions below.
    S = tl.zeros([BK, BV], dtype=tl.float32)

    for t in range(T):
        row = bh * T + t

        g = tl.load(g_ptr + row * BK + off_k).to(tl.float32)
        u = tl.load(u_ptr + row * BV + off_v).to(tl.float32)
        kk = tl.load(k_ptr + row * BK + off_k).to(tl.float32)
        vv = tl.load(v_ptr + row * BV + off_v).to(tl.float32)
        qq = tl.load(q_ptr + row * BK + off_k).to(tl.float32)
        beta = tl.load(beta_ptr + row).to(tl.float32)
        gamma = tl.load(gamma_ptr + row).to(tl.float32)

        # Per-key-channel decay.
        S = S * tl.exp(g)[:, None]

        # Right-hand erase: S <- S - gamma * (S u) u^T.
        su = tl.sum(S * u[None, :], axis=1)
        S = S - gamma * su[:, None] * u[None, :]

        # Delta-rule write: S <- S + k (beta * (v - S^T k))^T.
        pred = tl.sum(S * kk[:, None], axis=0)
        S = S + kk[:, None] * (beta * (vv - pred))[None, :]

        # Output: o = scale * S^T q.
        out = scale * tl.sum(S * qq[:, None], axis=0)
        tl.store(o_ptr + row * BV + off_v, out)


def _pad_last_dim(x, padded_size):
    """Return a contiguous tensor whose last dimension is padded_size."""
    if x.shape[-1] == padded_size:
        return x.contiguous()
    result = torch.zeros(
        (*x.shape[:-1], padded_size), device=x.device, dtype=x.dtype
    )
    result[..., : x.shape[-1]] = x
    return result


def kernel_function(q, k, v, u, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    if B * H == 0 or T == 0 or K == 0 or V == 0:
        return torch.zeros(
            (B, H, T, V), device=q.device, dtype=torch.float32
        )

    # Triton ranges must be powers of two.  Padded key channels have
    # q = k = g = 0 and padded value channels have v = u = 0, so they do
    # not affect the unpadded outputs.
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

    state_elements = BK * BV
    if state_elements <= 4096:
        num_warps = 4
    elif state_elements <= 16384:
        num_warps = 16
    else:
        num_warps = 32

    _fused_recurrent_kernel[(B * H,)](
        q_,
        k_,
        v_,
        u_,
        g_,
        beta_,
        gamma_,
        out_,
        T,
        float(scale),
        BK=BK,
        BV=BV,
        num_warps=num_warps,
        num_stages=1,
    )

    return out_[:, :, :V].reshape(B, H, T, V)
