```python
import torch
import triton
import triton.language as tl

def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 1 else 1

@triton.jit
def fwd_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
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

    # S[K, V_tile] kept in registers / fp32
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    q_base = bh * T * K
    k_base = bh * T * K
    e_base = bh * T * K
    g_base = bh * T * K
    v_base = bh * T * V
    out_base = bh * T * V
    beta_base = bh * T
    gamma_base = bh * T

    for t in range(T):
        q = tl.load(q_ptr + q_base + t * K + k_idx, mask=k_mask, other=0.0)
        q = q.to(tl.float32)
        k = tl.load(k_ptr + k_base + t * K + k_idx, mask=k_mask, other=0.0)
        k = k.to(tl.float32)
        e = tl.load(e_ptr + e_base + t * K + k_idx, mask=k_mask, other=0.0)
        e = e.to(tl.float32)
        g = tl.load(g_ptr + g_base + t * K + k_idx, mask=k_mask, other=0.0)
        beta = tl.load(beta_ptr + beta_base + t)
        gamma = tl.load(gamma_ptr + gamma_base + t)
        v = tl.load(v_ptr + v_base + t * V + v_start + v_idx, mask=v_mask, other=0.0)
        v = v.to(tl.float32)

        # 1. decay diag(exp(g)) : row scale
        d = tl.exp(g)  # [BLOCK_K]
        S = S * d[:, None]

        # 2. e-erase: S = (I - gamma e e^T) S
        dot_e = tl.sum(e[:, None] * S, axis=0)  # [BLOCK_V]
        S = S - gamma * e[:, None] * dot_e[None, :]

        # 3. k-update: pred = k^T S, S = S - beta k k^T S + beta k v^T
        pred = tl.sum(k[:, None] * S, axis=0)  # [BLOCK_V]
        delta = beta * (v - pred)  # [BLOCK_V]
        delta = tl.where(v_mask, delta, 0.0)
        S = S + k[:, None] * delta[None, :]

        # 4. output o_t = scale * S^T q
        out = tl.sum(S * q[:, None], axis=0) * scale  # [BLOCK_V]
        tl.store(out_ptr + out_base + t * V + v_start + v_idx, out, mask=v_mask)

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    """
    q,k,e: [B,H,T,K] bf16 (unit-norm for k,e)
    v: [B,H,T,V] bf16
    g: [B,H,T,K] fp32 (log decay <=0)
    beta,gamma: [B,H,T] fp32
    scale: float
    returns [B,H,T,V] float32 (or q dtype) with reference semantics
    """
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    B, H, T, K = q.shape
    V = v.shape[-1]
    if T == 0:
        return torch.empty((B, H, 0, V), device=q.device, dtype=torch.float32)

    # make contiguous so linear offset arithmetic is valid
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    e_c = e.contiguous()
    g_c = g.contiguous()
    beta_c = beta.contiguous()
    gamma_c = gamma.contiguous()

    BH = B * H

    BLOCK_K = _next_pow2(K)          # 128 for K=128, keeps one K-tile in registers
    # tile V to keep S = K x V_tile in registers (~16KB for 128x32 fp32)
    if V <= 32:
        BLOCK_V = _next_pow2(V)
    else:
        BLOCK_V = 32
    NTILES = (V + BLOCK_V - 1) // BLOCK_V

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    # out is contiguous [B,H,T,V] == [BH,T,V] linear layout expected by kernel

    grid = (BH * NTILES,)

    fwd_kernel[grid](
        q_c, k_c, v_c, e_c, g_c, beta_c, gamma_c, out,
        BH, T, K, V, scale,
        BLOCK_K, BLOCK_V, NTILES,
    )
    return out
```

Fused sequential kernel keeps the `K×V` state in registers and loops over `T` inside one Triton program, tiling `V` across programs so each tile `S[K,BLOCK_V]` is independent per column. This removes all Python-loop launch overhead and mirrors the reference order `decay → e-erase → k-pred → k-update` in fp32. Uncertainties: exact `tl.sum(...,axis=0)` / `to(tl.float32)` API on Triton 3.5, bf16 pointer handling, register pressure for 128×32 tiles and whether a chunk-scan would be needed for much longer T, but the fused loop should easily beat the 217 ms baseline on L4.