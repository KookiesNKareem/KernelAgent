import torch
import triton
import triton.language as tl


@triton.jit
def _gated_delta_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    out_ptr,
    T,
    K,
    V,
    scale,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per (B, H)

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V

    strict = offs_c[:, None] > offs_c[None, :]
    eye = tl.where(offs_c[:, None] == offs_c[None, :], 1.0, 0.0)

    # State S in the original (unscaled) domain, shape [K, V].
    S = tl.zeros([BK, BV], dtype=tl.float32)

    q_base = pid * T * K
    v_base = pid * T * V
    beta_base = pid * T

    for start in range(0, T, CHUNK):
        t_idx = start + offs_c
        mask_t = t_idx < T
        mask_ck = mask_t[:, None] & mask_k[None, :]
        mask_cv = mask_t[:, None] & mask_v[None, :]

        q = tl.load(
            q_ptr + q_base + t_idx[:, None] * K + offs_k[None, :],
            mask=mask_ck,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + q_base + t_idx[:, None] * K + offs_k[None, :],
            mask=mask_ck,
            other=0.0,
        ).to(tl.float32)
        g = tl.load(
            g_ptr + q_base + t_idx[:, None] * K + offs_k[None, :],
            mask=mask_ck,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + v_base + t_idx[:, None] * V + offs_v[None, :],
            mask=mask_cv,
            other=0.0,
        ).to(tl.float32)
        beta = tl.load(beta_ptr + beta_base + t_idx, mask=mask_t, other=0.0).to(
            tl.float32
        )

        # Cumulative log-decay within this chunk.  A per-K pivot at the
        # middle of the chunk's decay range keeps both exp(c-pivot) and
        # exp(pivot-c) as small as possible; all pairwise products below
        # still equal exp(c_i - c_j) exactly as in the recurrence.
        c = tl.cumsum(g, axis=0)
        total = tl.sum(g, axis=0)  # c at the last row of the chunk
        pivot = total * 0.5

        ec = tl.exp(c - pivot[None, :])
        enc = tl.exp(pivot[None, :] - c)
        q_hat = q * ec
        k_hat = k * ec
        k_tilde = k * enc

        # A_ij = beta_i * <k_hat_i, k_tilde_j> for j < i.
        kk = tl.dot(k_hat, tl.trans(k_tilde), input_precision="ieee")
        a_mat = tl.where(strict, kk * beta[:, None], 0.0)

        # inv = (I + A)^-1 via Newton iteration.  A is strictly lower
        # triangular and nilpotent of index CHUNK, so for CHUNK=64 six
        # iterations (each squaring the residual) are exact in exact
        # arithmetic: residuals go A, A^2, A^4, ..., A^64 = 0.
        b_mat = a_mat + eye
        inv = eye
        for _ in range(6):
            inv = inv + tl.dot(
                inv,
                eye - tl.dot(b_mat, inv, input_precision="ieee"),
                input_precision="ieee",
            )

        # Transformed-domain state at chunk entry: Shat = exp(pivot) * S,
        # so that q_hat^T Shat (resp. k_hat^T Shat) is the decayed read.
        s_hat = S * tl.exp(pivot)[:, None]

        # Solve (I + A) U = beta * (v - k_hat @ Shat) in factored form.
        u = tl.dot(inv, beta[:, None] * v, input_precision="ieee") - tl.dot(
            tl.dot(inv, beta[:, None] * k_hat, input_precision="ieee"),
            s_hat,
            input_precision="ieee",
        )

        # o_i = scale * (q_hat_i^T Shat + sum_{j<i} <q_hat_i, k_tilde_j> u_j)
        qk = tl.dot(q_hat, tl.trans(k_tilde), input_precision="ieee")
        qk = tl.where(strict, qk, 0.0)
        o = tl.dot(qk, u, input_precision="ieee") + tl.dot(
            q_hat, s_hat, input_precision="ieee"
        )

        tl.store(
            out_ptr + v_base + t_idx[:, None] * V + offs_v[None, :],
            o * scale,
            mask=mask_cv,
        )

        # Advance the state to the end of the chunk:
        # S <- exp(total) * S + sum_j k_j * exp(total - c_j) * u_j^T.
        k_end = k * tl.exp(total[None, :] - c)
        S = S * tl.exp(total)[:, None] + tl.dot(
            tl.trans(k_end), u, input_precision="ieee"
        )


def _next_pow2(x):
    n = 1
    while n < x:
        n *= 2
    return n


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # e and gamma are intentionally unused: the reference recurrence
    # never reads them, so the output must not depend on them.
    B, H, T, K = q.shape
    V = v.shape[-1]

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0:
        return out

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = g.contiguous()
    beta_c = beta.contiguous()

    BK = max(16, _next_pow2(K))
    BV = max(16, _next_pow2(V))

    _gated_delta_kernel[(B * H,)](
        q_c,
        k_c,
        v_c,
        g_c,
        beta_c,
        out,
        T,
        K,
        V,
        float(scale),
        BK=BK,
        BV=BV,
        CHUNK=64,
        num_warps=8,
        num_stages=2,
    )
    return out
