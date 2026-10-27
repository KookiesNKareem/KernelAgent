```python
import torch
import triton
import triton.language as tl

# generic fallback (correct for any K,V, strided, etc.)
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


# fast path for the timed shape K=V=128 : S in registers, no HBM traffic
@triton.jit
def _fwd_fast_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    scale,
    B, H, T,
    Q_B, Q_H, Q_T, Q_K,
    K_B, K_H, K_T, K_K,
    V_B, V_H, V_T, V_V,
    E_B, E_H, E_T, E_K,
    G_B, G_H, G_T, G_K,
    Bet_B, Bet_H, Bet_T,
    Gam_B, Gam_H, Gam_T,
    O_B, O_H, O_T, O_V,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # S 128x128 in registers as 4 tiles 64x64
    s00 = tl.zeros((64, 64), dtype=tl.float32)
    s01 = tl.zeros((64, 64), dtype=tl.float32)
    s10 = tl.zeros((64, 64), dtype=tl.float32)
    s11 = tl.zeros((64, 64), dtype=tl.float32)

    t = 0
    while t < T:
        offs0 = tl.arange(0, 64)
        offs1 = tl.arange(0, 64) + 64

        q0 = tl.load(q_ptr + b * Q_B + h * Q_H + t * Q_T + offs0 * Q_K).to(tl.float32)
        q1 = tl.load(q_ptr + b * Q_B + h * Q_H + t * Q_T + offs1 * Q_K).to(tl.float32)
        k0 = tl.load(k_ptr + b * K_B + h * K_H + t * K_T + offs0 * K_K).to(tl.float32)
        k1 = tl.load(k_ptr + b * K_B + h * K_H + t * K_T + offs1 * K_K).to(tl.float32)
        e0 = tl.load(e_ptr + b * E_B + h * E_H + t * E_T + offs0 * E_K).to(tl.float32)
        e1 = tl.load(e_ptr + b * E_B + h * E_H + t * E_T + offs1 * E_K).to(tl.float32)
        g0 = tl.load(g_ptr + b * G_B + h * G_H + t * G_T + offs0 * G_K).to(tl.float32)
        g1 = tl.load(g_ptr + b * G_B + h * G_H + t * G_T + offs1 * G_K).to(tl.float32)
        v0 = tl.load(v_ptr + b * V_B + h * V_H + t * V_T + offs0 * V_V).to(tl.float32)
        v1 = tl.load(v_ptr + b * V_B + h * V_H + t * V_T + offs1 * V_V).to(tl.float32)
        beta_t = tl.load(beta_ptr + b * Bet_B + h * Bet_H + t * Bet_T).to(tl.float32)
        gamma_t = tl.load(gamma_ptr + b * Gam_B + h * Gam_H + t * Gam_T).to(tl.float32)

        d0 = tl.exp(g0)
        d1 = tl.exp(g1)

        # pred = k^T diag(d) S
        pred0 = tl.zeros((64,), dtype=tl.float32)
        pred1 = tl.zeros((64,), dtype=tl.float32)
        s00 = s00 * d0[:, None]
        pred0 += tl.sum(k0[:, None] * s00, axis=0)
        s01 = s01 * d0[:, None]
        pred1 += tl.sum(k0[:, None] * s01, axis=0)
        s10 = s10 * d1[:, None]
        pred0 += tl.sum(k1[:, None] * s10, axis=0)
        s11 = s11 * d1[:, None]
        pred1 += tl.sum(k1[:, None] * s11, axis=0)

        w0 = v0 - pred0
        w1 = v1 - pred1

        # S += beta k w^T ; proj = e^T S
        proj0 = tl.zeros((64,), dtype=tl.float32)
        proj1 = tl.zeros((64,), dtype=tl.float32)
        s00 = s00 + beta_t * (k0[:, None] * w0[None, :])
        proj0 += tl.sum(e0[:, None] * s00, axis=0)
        s01 = s01 + beta_t * (k0[:, None] * w1[None, :])
        proj1 += tl.sum(e0[:, None] * s01, axis=0)
        s10 = s10 + beta_t * (k1[:, None] * w0[None, :])
        proj0 += tl.sum(e1[:, None] * s10, axis=0)
        s11 = s11 + beta_t * (k1[:, None] * w1[None, :])
        proj1 += tl.sum(e1[:, None] * s11, axis=0)

        # S -= gamma e proj^T ; out = S^T q
        acc0 = tl.zeros((64,), dtype=tl.float32)
        acc1 = tl.zeros((64,), dtype=tl.float32)
        s00 = s00 - gamma_t * (e0[:, None] * proj0[None, :])
        acc0 += tl.sum(q0[:, None] * s00, axis=0)
        s01 = s01 - gamma_t * (e0[:, None] * proj1[None, :])
        acc1 += tl.sum(q0[:, None] * s01, axis=0)
        s10 = s10 - gamma_t * (e1[:, None] * proj0[None, :])
        acc0 += tl.sum(q1[:, None] * s10, axis=0)
        s11 = s11 - gamma_t * (e1[:, None] * proj1[None, :])
        acc1 += tl.sum(q1[:, None] * s11, axis=0)

        acc0 = acc0 * scale
        acc1 = acc1 * scale
        tl.store(out_ptr + b * O_B + h * O_H + t * O_T + offs0 * O_V, acc0)
        tl.store(out_ptr + b * O_B + h * O_H + t * O_T + offs1 * O_V, acc1)

        t += 1


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    out = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    scale = float(scale)

    Q_B, Q_H, Q_T, Q_K = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
    K_B, K_H, K_T, K_K = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
    V_B, V_H, V_T, V_V = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
    E_B, E_H, E_T, E_K = e.stride(0), e.stride(1), e.stride(2), e.stride(3)
    G_B, G_H, G_T, G_K = g.stride(0), g.stride(1), g.stride(2), g.stride(3)
    Bet_B, Bet_H, Bet_T = beta.stride(0), beta.stride(1), beta.stride(2)
    Gam_B, Gam_H, Gam_T = gamma.stride(0), gamma.stride(1), gamma.stride(2)
    O_B, O_H, O_T, O_V = out.stride(0), out.stride(1), out.stride(2), out.stride(3)

    grid = (B * H,)

    if K == 128 and Vd == 128:
        _fwd_fast_kernel[grid](
            q, k, v, e, g, beta, gamma, out, scale,
            B, H, T,
            Q_B, Q_H, Q_T, Q_K,
            K_B, K_H, K_T, K_K,
            V_B, V_H, V_T, V_V,
            E_B, E_H, E_T, E_K,
            G_B, G_H, G_T, G_K,
            Bet_B, Bet_H, Bet_T,
            Gam_B, Gam_H, Gam_T,
            O_B, O_H, O_T, O_V,
            num_warps=4, num_stages=2,
        )
    else:
        BLOCK_K = 32
        BLOCK_V = 32
        NK = (K + BLOCK_K - 1) // BLOCK_K
        NV = (Vd + BLOCK_V - 1) // BLOCK_V
        state = torch.zeros((B, H, K, Vd), device=q.device, dtype=torch.float32)
        S_B, S_H, S_K, S_V = state.stride(0), state.stride(1), state.stride(2), state.stride(3)
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

Previous 37ms version kept `S` in HBM and did 3 HBM passes per step and decay before `pred` was wrong. The fix fuses the recurrence into one `while T` loop, keeps `S` (16k) in registers as 4×64×64 tiles for the hot `K=V=128` shape (fallback to masked HBM scratch for generic shapes) and respects the reference order `S←diag(e^g)S`, `pred=kᵀS`, `S+=βk(v-pred)ᵀ`, `proj=eᵀS`, `S-=γe projᵀ`, `o=scale·Sᵀq` with correct strides/masks. Unsure about optimal `num_warps/num_stages` and register pressure for 64×64 tiles on L4.