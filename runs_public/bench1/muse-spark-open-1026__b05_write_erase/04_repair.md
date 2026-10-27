```python
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr, state_ptr,
    scale,
    B, H, T, K, Vd,
    Q_B, Q_H, Q_T, Q_K,
    K_B, K_H, K_T, K_K,
    V_B, V_H, V_T, V_V,
    E_B, E_H, E_T, E_K,
    G_B, G_H, G_T, G_K,
    Bet_B, Bet_H, Bet_T,
    Gam_B, Gam_H, Gam_T,
    O_B, O_H, O_T, O_V,
    S_B, S_H, S_K, S_V,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    NK: tl.constexpr, NV: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    state_base = state_ptr + b * S_B + h * S_H

    t = 0
    while t < T:
        beta_t = tl.load(beta_ptr + b * Bet_B + h * Bet_H + t * Bet_T).to(tl.float32)
        gamma_t = tl.load(gamma_ptr + b * Gam_B + h * Gam_H + t * Gam_T).to(tl.float32)

        for j in range(NV):
            v_off = j * BLOCK_V
            offs_v = tl.arange(0, BLOCK_V)
            mask_v = (v_off + offs_v) < Vd
            v_tile = tl.load(v_ptr + b * V_B + h * V_H + t * V_T + (v_off + offs_v) * V_V, mask=mask_v, other=0.0).to(tl.float32)

            # pred = k^T (diag(exp g) S)  and write back decayed S
            pred = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for i in range(NK):
                k_off = i * BLOCK_K
                offs_k = tl.arange(0, BLOCK_K)
                mask_k = (k_off + offs_k) < K
                k_tile = tl.load(k_ptr + b * K_B + h * K_H + t * K_T + (k_off + offs_k) * K_K, mask=mask_k, other=0.0).to(tl.float32)
                g_tile = tl.load(g_ptr + b * G_B + h * G_H + t * G_T + (k_off + offs_k) * G_K, mask=mask_k, other=0.0).to(tl.float32)
                d_tile = tl.exp(g_tile)
                s_ptr = state_base + (offs_k[:, None] * S_K + offs_v[None, :] * S_V) + k_off * S_K + v_off * S_V
                mask_2d = mask_k[:, None] & mask_v[None, :]
                s_tile = tl.load(s_ptr, mask=mask_2d, other=0.0)
                s_tile = s_tile * d_tile[:, None]
                tl.store(s_ptr, s_tile, mask=mask_2d)
                pred += tl.sum(k_tile[:, None] * s_tile, axis=0)

            w_tile = v_tile - pred

            # S = S + beta k w^T ;  proj = e^T S
            proj = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for i in range(NK):
                k_off = i * BLOCK_K
                offs_k = tl.arange(0, BLOCK_K)
                mask_k = (k_off + offs_k) < K
                k_tile = tl.load(k_ptr + b * K_B + h * K_H + t * K_T + (k_off + offs_k) * K_K, mask=mask_k, other=0.0).to(tl.float32)
                e_tile = tl.load(e_ptr + b * E_B + h * E_H + t * E_T + (k_off + offs_k) * E_K, mask=mask_k, other=0.0).to(tl.float32)
                s_ptr = state_base + (offs_k[:, None] * S_K + offs_v[None, :] * S_V) + k_off * S_K + v_off * S_V
                mask_2d = mask_k[:, None] & mask_v[None, :]
                s_tile = tl.load(s_ptr, mask=mask_2d, other=0.0)
                s_tile = s_tile + beta_t * (k_tile[:, None] * w_tile[None, :])
                proj += tl.sum(e_tile[:, None] * s_tile, axis=0)
                tl.store(s_ptr, s_tile, mask=mask_2d)

            # S = S - gamma e proj^T ; out = scale * S^T q
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for i in range(NK):
                k_off = i * BLOCK_K
                offs_k = tl.arange(0, BLOCK_K)
                mask_k = (k_off + offs_k) < K
                q_tile = tl.load(q_ptr + b * Q_B + h * Q_H + t * Q_T + (k_off + offs_k) * Q_K, mask=mask_k, other=0.0).to(tl.float32)
                e_tile = tl.load(e_ptr + b * E_B + h * E_H + t * E_T + (k_off + offs_k) * E_K, mask=mask_k, other=0.0).to(tl.float32)
                s_ptr = state_base + (offs_k[:, None] * S_K + offs_v[None, :] * S_V) + k_off * S_K + v_off * S_V
                mask_2d = mask_k[:, None] & mask_v[None, :]
                s_tile = tl.load(s_ptr, mask=mask_2d, other=0.0)
                s_tile = s_tile - gamma_t * (e_tile[:, None] * proj[None, :])
                acc += tl.sum(q_tile[:, None] * s_tile, axis=0)
                tl.store(s_ptr, s_tile, mask=mask_2d)

            acc = acc * scale
            tl.store(out_ptr + b * O_B + h * O_H + t * O_T + (v_off + offs_v) * O_V, acc, mask=mask_v)

        t += 1


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    out = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    scale = float(scale)
    state = torch.zeros((B, H, K, Vd), device=q.device, dtype=torch.float32)
    BLOCK_K = 32
    BLOCK_V = 32
    NK = (K + BLOCK_K - 1) // BLOCK_K
    NV = (Vd + BLOCK_V - 1) // BLOCK_V

    Q_B, Q_H, Q_T, Q_K = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
    K_B, K_H, K_T, K_K = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
    V_B, V_H, V_T, V_V = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
    E_B, E_H, E_T, E_K = e.stride(0), e.stride(1), e.stride(2), e.stride(3)
    G_B, G_H, G_T, G_K = g.stride(0), g.stride(1), g.stride(2), g.stride(3)
    Bet_B, Bet_H, Bet_T = beta.stride(0), beta.stride(1), beta.stride(2)
    Gam_B, Gam_H, Gam_T = gamma.stride(0), gamma.stride(1), gamma.stride(2)
    O_B, O_H, O_T, O_V = out.stride(0), out.stride(1), out.stride(2), out.stride(3)
    S_B, S_H, S_K, S_V = state.stride(0), state.stride(1), state.stride(2), state.stride(3)

    grid = (B * H,)
    _fwd_kernel[grid](
        q, k, v, e, g, beta, gamma, out, state, scale,
        B, H, T, K, Vd,
        Q_B, Q_H, Q_T, Q_K,
        K_B, K_H, K_T, K_K,
        V_B, V_H, V_T, V_V,
        E_B, E_H, E_T, E_K,
        G_B, G_H, G_T, G_K,
        Bet_B, Bet_H, Bet_T,
        Gam_B, Gam_H, Gam_T,
        O_B, O_H, O_T, O_V,
        S_B, S_H, S_K, S_V,
        BLOCK_K, BLOCK_V, NK, NV,
        num_warps=4, num_stages=1,
    )
    return out
```
Prior version computed `pred = k^T S` before applying `diag(exp g)`, while reference does `S←diag(exp g)S` first, causing ~1% error and zero error when `g=0`. Fix applies decay to every tile before `pred`, storing decayed `S` back, then does beta-update+proj and gamma-update+output in two further passes per `V`-tile with no Python lists and a file-defined `@triton.jit` kernel, handling arbitrary strides/masks.

Unsure about optimal `BLOCK_K/V` and `num_warps` for `K=129` on L4 and exact `tl.exp` numerics vs PyTorch.
