```python
import torch
import triton
import triton.language as tl


@triton.jit
def _recurrent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    out_ptr,
    T,
    K_DIM,
    V_DIM,
    scale,
    NUM_V_BLOCKS,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_bh = pid // NUM_V_BLOCKS
    pid_v = pid - pid_bh * NUM_V_BLOCKS

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K_DIM
    mask_v = offs_v < V_DIM

    # The state is stored transposed as h[v, k] = S[k, v].  This makes every
    # per-time-step contraction a reduction over the last axis.
    h = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)

    bh = pid_bh.to(tl.int64)
    base_k = bh * T * K_DIM
    base_v = bh * T * V_DIM
    base_s = bh * T

    for t in range(T):
        off_k = base_k + t * K_DIM + offs_k
        off_v = base_v + t * V_DIM + offs_v

        q_vec = tl.load(q_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        k_vec = tl.load(k_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        e_vec = tl.load(e_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        g_vec = tl.load(g_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptr + off_v, mask=mask_v, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + base_s + t).to(tl.float32)
        gamma_t = tl.load(gamma_ptr + base_s + t).to(tl.float32)

        # S <- diag(exp(g_t)) S, represented on the transposed state.
        h = h * tl.exp(g_vec)[None, :]

        # S <- (I - gamma_t e_t e_t^T) S
        erased = tl.sum(h * e_vec[None, :], axis=1)
        h = h - (gamma_t * erased)[:, None] * e_vec[None, :]

        # S <- S + beta_t k_t (v_t - S^T k_t)^T
        pred = tl.sum(h * k_vec[None, :], axis=1)
        delta = beta_t * (v_vec - pred)
        h = h + delta[:, None] * k_vec[None, :]

        # o_t = scale * S_t^T q_t
        out_vec = tl.sum(h * (q_vec * scale)[None, :], axis=1)
        tl.store(out_ptr + off_v, out_vec, mask=mask_v)


def _next_power_of_2(x):
    n = 1
    while n < x:
        n *= 2
    return n


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0 or V == 0:
        return out

    q_flat = q.contiguous().reshape(B * H, T, K)
    k_flat = k.contiguous().reshape(B * H, T, K)
    e_flat = e.contiguous().reshape(B * H, T, K)
    g_flat = g.contiguous().reshape(B * H, T, K)
    v_flat = v.contiguous().reshape(B * H, T, V)
    beta_flat = beta.contiguous().reshape(B * H, T)
    gamma_flat = gamma.contiguous().reshape(B * H, T)
    out_flat = out.reshape(B * H, T, V)

    block_k = max(16, _next_power_of_2(max(1, K)))
    if block_k >= 256:
        desired_v = 16
    elif block_k == 128:
        desired_v = 32
    else:
        desired_v = 64

    if V < desired_v:
        block_v = max(16, _next_power_of_2(max(1, V)))
    else:
        block_v = desired_v

    num_v_blocks = (V + block_v - 1) // block_v
    grid = (B * H * num_v_blocks,)
    num_warps = 4 if block_k * block_v >= 4096 else 2

    _recurrent_kernel[grid](
        q_flat,
        k_flat,
        v_flat,
        e_flat,
        g_flat,
        beta_flat,
        gamma_flat,
        out_flat,
        T,
        K,
        V,
        float(scale),
        num_v_blocks,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=num_warps,
        num_stages=2,
    )
    return out
```

The implementation assigns one Triton program to each batch-head and a tile of the value dimension, keeps the transposed fp32 state in registers, and applies the decay, erase, delta update, and output projection in the reference order at every time step. Splitting only the value dimension preserves the exact recurrence while exposing substantially more parallelism than one program per sequence. Because nothing could be executed here, the chosen tile sizes and warp count are reasonable but unprofiled choices for the L4, and Triton’s generated reduction layout may leave room for further tuning.