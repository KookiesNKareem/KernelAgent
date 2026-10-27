```python
import torch
import triton
import triton.language as tl


@triton.jit
def fast_kernel(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, out_ptr,
                BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, scale: tl.constexpr,
                BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, NTILES: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // NTILES
    tile = pid % NTILES
    v_start = tile * BLOCK_V

    k_idx = tl.arange(0, BLOCK_K)
    v_idx = tl.arange(0, BLOCK_V)

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    q_base = bh * T * K
    k_base = bh * T * K
    e_base = bh * T * K
    d_base = bh * T * K
    v_base = bh * T * V
    o_base = bh * T * V
    b_base = bh * T

    for t in range(T):
        q = tl.load(q_ptr + q_base + t * K + k_idx).to(tl.float32)
        kk = tl.load(k_ptr + k_base + t * K + k_idx).to(tl.float32)
        ee = tl.load(e_ptr + e_base + t * K + k_idx).to(tl.float32)
        dd = tl.load(d_ptr + d_base + t * K + k_idx).to(tl.float32)
        beta = tl.load(beta_ptr + b_base + t)
        gamma = tl.load(gamma_ptr + b_base + t)
        vv = tl.load(v_ptr + v_base + t * V + v_start + v_idx).to(tl.float32)

        S = S * dd[:, None]

        dot_e = tl.sum(ee[:, None] * S, axis=0)
        S = S - gamma * ee[:, None] * dot_e[None, :]

        pred = tl.sum(kk[:, None] * S, axis=0)
        delta = beta * (vv - pred)
        S = S + kk[:, None] * delta[None, :]

        out = tl.sum(S * q[:, None], axis=0) * scale
        tl.store(out_ptr + o_base + t * V + v_start + v_idx, out)


@triton.jit
def generic_kernel(q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, out_ptr,
                   BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, scale: tl.constexpr,
                   BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, NTILES: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // NTILES
    tile = pid % NTILES
    v_start = tile * BLOCK_V

    k_idx = tl.arange(0, BLOCK_K)
    v_idx = tl.arange(0, BLOCK_V)
    k_mask = k_idx < K
    v_mask = (v_start + v_idx) < V

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    q_base = bh * T * K
    k_base = bh * T * K
    e_base = bh * T * K
    d_base = bh * T * K
    v_base = bh * T * V
    o_base = bh * T * V
    b_base = bh * T

    for t in range(T):
        q = tl.load(q_ptr + q_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        kk = tl.load(k_ptr + k_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        ee = tl.load(e_ptr + e_base + t * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        dd = tl.load(d_ptr + d_base + t * K + k_idx, mask=k_mask, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + b_base + t)
        gamma = tl.load(gamma_ptr + b_base + t)
        vv = tl.load(v_ptr + v_base + t * V + v_start + v_idx, mask=v_mask, other=0.0).to(tl.float32)

        S = S * dd[:, None]

        dot_e = tl.sum(ee[:, None] * S, axis=0)
        S = S - gamma * ee[:, None] * dot_e[None, :]

        pred = tl.sum(kk[:, None] * S, axis=0)
        delta = beta * (vv - pred)
        delta = tl.where(v_mask, delta, 0.0)
        S = S + kk[:, None] * delta[None, :]

        out = tl.sum(S * q[:, None], axis=0) * scale
        tl.store(out_ptr + o_base + t * V + v_start + v_idx, out, mask=v_mask)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    if T == 0:
        return torch.empty((B, H, 0, V), device=q.device, dtype=torch.float32)
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)

    BH = B * H

    # cheap elementwise prep: hoist exp out of the loop (same 8M exps, vectorized)
    g_exp = torch.exp(g.contiguous())

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    e_c = e.contiguous()
    d_c = g_exp.contiguous()
    beta_c = beta.contiguous()
    gamma_c = gamma.contiguous()

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    def is_pow2(x: int) -> bool:
        return (x & (x - 1)) == 0

    # fast path: K power-of-two and V divisible by BLOCK_V -> no masks, no tl.exp
    if V <= 32:
        BLOCK_V_fast = 1 << (V - 1).bit_length() if V > 1 else 1
    else:
        BLOCK_V_fast = 32

    if is_pow2(K) and K <= 256 and V % BLOCK_V_fast == 0:
        BLOCK_K = K
        BLOCK_V = BLOCK_V_fast
        NTILES = V // BLOCK_V
        grid = (BH * NTILES,)
        fast_kernel[grid](
            q_c, k_c, v_c, e_c, d_c, beta_c, gamma_c, out,
            BH, T, K, V, scale_f,
            BLOCK_K, BLOCK_V, NTILES,
            num_warps=8, num_stages=3,
        )
        return out
    else:
        # generic correct path for arbitrary K,V
        BLOCK_K = 1 << (K - 1).bit_length() if K > 1 else 1
        if V <= 32:
            BLOCK_V = 1 << (V - 1).bit_length() if V > 1 else 1
        else:
            BLOCK_V = 32
        NTILES = (V + BLOCK_V - 1) // BLOCK_V
        grid = (BH * NTILES,)
        generic_kernel[grid](
            q_c, k_c, v_c, e_c, d_c, beta_c, gamma_c, out,
            BH, T, K, V, scale_f,
            BLOCK_K, BLOCK_V, NTILES,
            num_warps=4, num_stages=2,
        )
        return out
```

Fused T-loop keeps `S[K,BLOCK_V]` in registers and tiles only `V` across programs so reductions stay on-chip, matching reference order `decay→e-erase→k-update`. Hoisting `exp(g)` to a single vectorized `torch.exp` removes the SFU bottleneck from the inner loop and removing masks/predicates for the `K=V=128` power-of-two case cuts predication overhead. Unsure about optimal `num_warps/num_stages` for Ada/SM89 without profiling and whether `torch.exp` counts as allowed cheap prep, plus `tl.sum` API stability on Triton 3.5.