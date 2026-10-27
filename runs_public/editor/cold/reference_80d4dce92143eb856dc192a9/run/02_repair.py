import torch
import triton
import triton.language as tl


@triton.jit
def _fused_fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr,
    s_qb, s_qh, s_qt, s_qk,
    s_kb, s_kh, s_kt, s_kk,
    s_vb, s_vh, s_vt, s_vv,
    s_eb, s_eh, s_et, s_ek,
    s_gb, s_gh, s_gt, s_gk,
    s_betab, s_betah, s_betat,
    s_gammab, s_gammah, s_gammat,
    s_ob, s_oh, s_ot, s_ov,
    H,
    T: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    off_k = tl.arange(0, BLOCK_K)
    off_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = off_k < K_DIM
    mask_v = off_v < V_DIM

    state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in range(T):
        q_ptrs = q_ptr + b * s_qb + h * s_qh + t * s_qt + off_k * s_qk
        k_ptrs = k_ptr + b * s_kb + h * s_kh + t * s_kt + off_k * s_kk
        e_ptrs = e_ptr + b * s_eb + h * s_eh + t * s_et + off_k * s_ek
        g_ptrs = g_ptr + b * s_gb + h * s_gh + t * s_gt + off_k * s_gk
        v_ptrs = v_ptr + b * s_vb + h * s_vh + t * s_vt + off_v * s_vv

        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0)
        k_vec = tl.load(k_ptrs, mask=mask_k, other=0.0)
        e_vec = tl.load(e_ptrs, mask=mask_k, other=0.0)
        g_vec = tl.load(g_ptrs, mask=mask_k, other=0.0)
        v_vec = tl.load(v_ptrs, mask=mask_v, other=0.0)

        beta_val = tl.load(beta_ptr + b * s_betab + h * s_betah + t * s_betat)
        gamma_val = tl.load(gamma_ptr + b * s_gammab + h * s_gammah + t * s_gammat)

        # row decay: S *= exp(g)
        decay = tl.exp(g_vec)
        state = state * tl.expand_dims(decay, 1)

        # erase: S -= gamma * e * (e^T S)
        e_col = tl.expand_dims(e_vec, 1)
        a_vec = tl.sum(state * e_col, 0)
        state = state - gamma_val * (e_col * tl.expand_dims(a_vec, 0))

        # delta: pred = k^T S, S += beta * k (v-pred)^T
        k_col = tl.expand_dims(k_vec, 1)
        p_vec = tl.sum(state * k_col, 0)
        diff_vec = beta_val * (v_vec - p_vec)
        state = state + k_col * tl.expand_dims(diff_vec, 0)

        # readout: o = scale * S^T q
        q_scaled = q_vec * scale
        q_col = tl.expand_dims(q_scaled, 1)
        o_vec = tl.sum(state * q_col, 0)

        o_ptrs = o_ptr + b * s_ob + h * s_oh + t * s_ot + off_v * s_ov
        tl.store(o_ptrs, o_vec, mask=mask_v)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V_DIM = v.shape[-1]

    def _prom(x):
        return x.to(torch.promote_types(x.dtype, torch.float32))

    q32 = _prom(q)
    k32 = _prom(k)
    v32 = _prom(v)
    e32 = _prom(e)
    g32 = _prom(g)
    beta32 = _prom(beta)
    gamma32 = _prom(gamma)

    out = torch.empty((B, H, T, V_DIM), device=q.device, dtype=torch.float32)

    s_qb, s_qh, s_qt, s_qk = q32.stride()
    s_kb, s_kh, s_kt, s_kk = k32.stride()
    s_vb, s_vh, s_vt, s_vv = v32.stride()
    s_eb, s_eh, s_et, s_ek = e32.stride()
    s_gb, s_gh, s_gt, s_gk = g32.stride()
    s_betab, s_betah, s_betat = beta32.stride()
    s_gammab, s_gammah, s_gammat = gamma32.stride()
    s_ob, s_oh, s_ot, s_ov = out.stride()

    BLOCK_K = triton.next_power_of_2(K)
    if V_DIM <= 32:
        BLOCK_V = triton.next_power_of_2(V_DIM)
    else:
        BLOCK_V = 32
    n_vblocks = (V_DIM + BLOCK_V - 1) // BLOCK_V
    grid = (B * H, n_vblocks)

    _fused_fwd_kernel[grid](
        q32, k32, v32, e32, g32, beta32, gamma32, out,
        s_qb, s_qh, s_qt, s_qk,
        s_kb, s_kh, s_kt, s_kk,
        s_vb, s_vh, s_vt, s_vv,
        s_eb, s_eh, s_et, s_ek,
        s_gb, s_gh, s_gt, s_gk,
        s_betab, s_betah, s_betat,
        s_gammab, s_gammah, s_gammat,
        s_ob, s_oh, s_ot, s_ov,
        H,
        T, K, V_DIM, float(scale),
        BLOCK_K, BLOCK_V,
        num_warps=4, num_stages=1,
    )
    return out
