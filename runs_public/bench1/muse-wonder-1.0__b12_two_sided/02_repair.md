```python
import torch
import triton
import triton.language as tl


@triton.jit
def _forward_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    q_sb, q_sh, q_st, q_sk,
    k_sb, k_sh, k_st, k_sk,
    v_sb, v_sh, v_st, v_sv,
    e_sb, e_sh, e_st, e_sk,
    g_sb, g_sh, g_st, g_sk,
    beta_sb, beta_sh, beta_st,
    gamma_sb, gamma_sh, gamma_st,
    out_sb, out_sh, out_st, out_sv,
    num_heads, seq_len, head_k, head_v, scale_f,
    BK: tl.constexpr, BV: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // num_heads
    h_idx = pid % num_heads
    b64 = b_idx.to(tl.int64)
    h64 = h_idx.to(tl.int64)

    q_cur = q_ptr + b64 * q_sb + h64 * q_sh
    k_cur = k_ptr + b64 * k_sb + h64 * k_sh
    v_cur = v_ptr + b64 * v_sb + h64 * v_sh
    e_cur = e_ptr + b64 * e_sb + h64 * e_sh
    g_cur = g_ptr + b64 * g_sb + h64 * g_sh
    beta_cur = beta_ptr + b64 * beta_sb + h64 * beta_sh
    gamma_cur = gamma_ptr + b64 * gamma_sb + h64 * gamma_sh
    out_cur = out_ptr + b64 * out_sb + h64 * out_sh

    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    offs_k64 = offs_k.to(tl.int64)
    offs_v64 = offs_v.to(tl.int64)
    mask_k = offs_k < head_k
    mask_v = offs_v < head_v
    # e has length head_k; it is also used along the value axis (head_k == head_v
    # in every case the reference itself can evaluate).
    mask_ev = offs_v < head_k

    state = tl.zeros([BK, BV], dtype=tl.float32)

    for _ in range(seq_len):
        q_vec = tl.load(q_cur + offs_k64 * q_sk, mask=mask_k, other=0.0).to(tl.float32)
        k_vec = tl.load(k_cur + offs_k64 * k_sk, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_cur + offs_v64 * v_sv, mask=mask_v, other=0.0).to(tl.float32)
        e_k = tl.load(e_cur + offs_k64 * e_sk, mask=mask_k, other=0.0).to(tl.float32)
        e_v = tl.load(e_cur + offs_v64 * e_sk, mask=mask_ev, other=0.0).to(tl.float32)
        g_vec = tl.load(g_cur + offs_k64 * g_sk, mask=mask_k, other=0.0).to(tl.float32)
        beta_val = tl.load(beta_cur).to(tl.float32)
        gamma_val = tl.load(gamma_cur).to(tl.float32)

        # S = diag(exp(g)) S
        state = state * tl.exp(g_vec)[:, None]

        # S = (I - gamma e e^T) S
        left_dot = tl.sum(state * e_k[:, None], axis=0)
        state = state - (gamma_val * e_k)[:, None] * left_dot[None, :]

        # S = S (I - gamma e e^T)
        right_dot = tl.sum(state * e_v[None, :], axis=1)
        state = state - right_dot[:, None] * (gamma_val * e_v)[None, :]

        # S = S + beta k (v - S^T k)^T ;  o = scale * S^T q
        pred = tl.sum(state * k_vec[:, None], axis=0)
        state = state + k_vec[:, None] * (beta_val * (v_vec - pred))[None, :]
        out_vec = tl.sum(state * (q_vec * scale_f)[:, None], axis=0)
        tl.store(out_cur + offs_v64 * out_sv, out_vec, mask=mask_v)

        q_cur += q_st
        k_cur += k_st
        v_cur += v_st
        e_cur += e_st
        g_cur += g_st
        beta_cur += beta_st
        gamma_cur += gamma_st
        out_cur += out_st


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0 or V == 0:
        return out

    bk = max(16, triton.next_power_of_2(K))
    bv = max(16, triton.next_power_of_2(V))
    num_warps = int(max(4, min(32, (bk * bv) // 2048)))

    _forward_kernel[(B * H,)](
        q, k, v, e, g, beta, gamma, out,
        q_sb=q.stride(0), q_sh=q.stride(1), q_st=q.stride(2), q_sk=q.stride(3),
        k_sb=k.stride(0), k_sh=k.stride(1), k_st=k.stride(2), k_sk=k.stride(3),
        v_sb=v.stride(0), v_sh=v.stride(1), v_st=v.stride(2), v_sv=v.stride(3),
        e_sb=e.stride(0), e_sh=e.stride(1), e_st=e.stride(2), e_sk=e.stride(3),
        g_sb=g.stride(0), g_sh=g.stride(1), g_st=g.stride(2), g_sk=g.stride(3),
        beta_sb=beta.stride(0), beta_sh=beta.stride(1), beta_st=beta.stride(2),
        gamma_sb=gamma.stride(0), gamma_sh=gamma.stride(1), gamma_st=gamma.stride(2),
        out_sb=out.stride(0), out_sh=out.stride(1), out_st=out.stride(2), out_sv=out.stride(3),
        num_heads=H, seq_len=T, head_k=K, head_v=V, scale_f=float(scale),
        BK=bk, BV=bv, num_warps=num_warps, num_stages=1,
    )
    return out
```

The previous chunkwise version inverted the cumulative erase/decay operator within each chunk, implicitly assuming that operator was invertible and well-conditioned (unit-norm `e`, `gamma` bounded away from 1, mild decay); the failing inputs violate that, since a contractive or even singular erase has an exponentially amplifying inverse, producing the non-finite/wrong results. The fix removes every inverse, division by decay, and norm/gate assumption: each program keeps the full state in registers and applies only the reference's forward decay, two-sided erase, delta update, and readout, in the original order. This remains a single fused Triton launch with no per-step host or memory traffic for the state, so it is still much faster than the Python loop.