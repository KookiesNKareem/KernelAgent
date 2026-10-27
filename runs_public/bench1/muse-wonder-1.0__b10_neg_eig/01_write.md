```python
import torch
import triton
import triton.language as tl


@triton.jit
def _delta_rule_recurrent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    out_ptr,
    T,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < V

    bh = pid_bh.to(tl.int64)

    q_ptrs = q_ptr + bh * T * K + offs_k
    k_ptrs = k_ptr + bh * T * K + offs_k
    v_ptrs = v_ptr + bh * T * V + offs_v
    out_ptrs = out_ptr + bh * T * V + offs_v
    g_ptrs = g_ptr + bh * T
    beta_ptrs = beta_ptr + bh * T

    # FP32 state tile for this (batch, head, V-block), held in registers.
    state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for _ in range(T):
        decay = tl.exp(tl.load(g_ptrs))
        beta_val = tl.load(beta_ptrs)

        k_vec = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)

        state *= decay
        pred = tl.sum(state * k_vec[:, None], axis=0)
        delta = (2.0 * beta_val) * (v_vec - pred)
        state += k_vec[:, None] * delta[None, :]

        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        out_vec = tl.sum(state * (q_vec * scale)[:, None], axis=0)
        tl.store(out_ptrs, out_vec, mask=mask_v)

        q_ptrs += K
        k_ptrs += K
        v_ptrs += V
        out_ptrs += V
        g_ptrs += 1
        beta_ptrs += 1


def _next_power_of_2(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # e and gamma are intentionally unused: the reference recurrence
    # never uses them.
    B, H, T, K = q.shape
    V = v.shape[-1]

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0 or V == 0:
        return out

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = g.contiguous()
    beta_c = beta.contiguous()

    bh = B * H
    q_f = q_c.view(bh, T, K)
    k_f = k_c.view(bh, T, K)
    v_f = v_c.view(bh, T, V)
    g_f = g_c.view(bh, T)
    beta_f = beta_c.view(bh, T)
    out_f = out.view(bh, T, V)

    block_k = _next_power_of_2(K)
    if block_k <= 32:
        block_v_cap = 128
    elif block_k <= 64:
        block_v_cap = 64
    elif block_k <= 128:
        block_v_cap = 32
    elif block_k <= 256:
        block_v_cap = 16
    else:
        block_v_cap = 8

    block_v = min(block_v_cap, _next_power_of_2(V))
    # Add V-block parallelism when there are few (batch, head) pairs.
    while block_v > 8 and bh * ((V + block_v - 1) // block_v) < 128:
        block_v //= 2

    num_v_blocks = (V + block_v - 1) // block_v
    tile_elems = block_k * block_v
    if tile_elems >= 4096:
        num_warps = 8
    elif tile_elems >= 1024:
        num_warps = 4
    else:
        num_warps = 2

    grid = (bh, num_v_blocks)
    with torch.cuda.device(q.device):
        _delta_rule_recurrent_kernel[grid](
            q_f,
            k_f,
            v_f,
            g_f,
            beta_f,
            out_f,
            T,
            float(scale),
            K=K,
            V=V,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            num_warps=num_warps,
        )
    return out
```

The kernel assigns one Triton program to each `(batch, head, V-block)`, keeps the FP32 state tile in registers, and applies the exact time recurrence sequentially, including the `2 * beta` Householder-style update. `e` and `gamma` are ignored because the reference implementation never uses them. I avoided chunkwise cumulative-decay rescaling, which can overflow for strongly negative `g`; the main uncertainty is whether register pressure for unusual `K, V` shapes larger than those benchmarked could reduce occupancy.