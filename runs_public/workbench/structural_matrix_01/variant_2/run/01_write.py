import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    s_qb, s_qh, s_qt, s_qk,
    s_kb, s_kh, s_kt, s_kk,
    s_vb, s_vh, s_vt, s_vv,
    s_eb, s_eh, s_et, s_ek,
    s_gb, s_gh, s_gt, s_gk,
    s_betab, s_betah, s_betat,
    s_gammab, s_gammah, s_gammat,
    s_ob, s_oh, s_ot, s_ov,
    B, H, T, K, V,
    scale,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    nv = (V + BLOCK_V - 1) // BLOCK_V
    bh = pid // nv
    iv = pid % nv
    b = bh // H
    h = bh % H

    rk = tl.arange(0, BLOCK_K)
    rv = tl.arange(0, BLOCK_V)
    mask_k = rk < K
    v_idx = iv * BLOCK_V + rv
    mask_v = v_idx < V

    q_base = q_ptr + b * s_qb + h * s_qh
    k_base = k_ptr + b * s_kb + h * s_kh
    v_base = v_ptr + b * s_vb + h * s_vh
    e_base = e_ptr + b * s_eb + h * s_eh
    g_base = g_ptr + b * s_gb + h * s_gh
    beta_base = beta_ptr + b * s_betab + h * s_betah
    gamma_base = gamma_ptr + b * s_gammab + h * s_gammah
    o_base = out_ptr + b * s_ob + h * s_oh

    S = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)

    for t in range(T):
        q_t = tl.load(q_base + t * s_qt + rk * s_qk, mask=mask_k, other=0.0).to(tl.float32)
        k_t = tl.load(k_base + t * s_kt + rk * s_kk, mask=mask_k, other=0.0).to(tl.float32)
        e_t = tl.load(e_base + t * s_et + rk * s_ek, mask=mask_k, other=0.0).to(tl.float32)
        g_t = tl.load(g_base + t * s_gt + rk * s_gk, mask=mask_k, other=0.0).to(tl.float32)
        v_t = tl.load(v_base + t * s_vt + v_idx * s_vv, mask=mask_v, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_base + t * s_betat)
        gamma_t = tl.load(gamma_base + t * s_gammat)

        decay = tl.exp(g_t)
        S = S * tl.expand_dims(decay, 1)

        k1 = tl.expand_dims(k_t, 1)
        pred = tl.sum(S * k1, axis=0)
        delta = (v_t - pred) * beta_t
        S = S + k1 * tl.expand_dims(delta, 0)

        e1 = tl.expand_dims(e_t, 1)
        s_e = tl.sum(S * e1, axis=0)
        ge = tl.expand_dims(e_t * gamma_t, 1)
        S = S - ge * tl.expand_dims(s_e, 0)

        qs = tl.expand_dims(q_t * scale, 1)
        o_t = tl.sum(S * qs, axis=0)
        tl.store(o_base + t * s_ot + v_idx * s_ov, o_t, mask=mask_v)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    out = torch.empty(B, H, T, V, dtype=torch.float32, device=q.device)
    if out.numel() == 0:
        return out
    BLOCK_K = triton.next_power_of_2(K)
    if V <= 32:
        BLOCK_V = triton.next_power_of_2(V)
    else:
        BLOCK_V = 32
    nv = (V + BLOCK_V - 1) // BLOCK_V
    grid = (B * H * nv,)

    s_qb, s_qh, s_qt, s_qk = q.stride()
    s_kb, s_kh, s_kt, s_kk = k.stride()
    s_vb, s_vh, s_vt, s_vv = v.stride()
    s_eb, s_eh, s_et, s_ek = e.stride()
    s_gb, s_gh, s_gt, s_gk = g.stride()
    s_betab, s_betah, s_betat = beta.stride()
    s_gammab, s_gammah, s_gammat = gamma.stride()
    s_ob, s_oh, s_ot, s_ov = out.stride()

    try:
        scale_f = float(scale)
    except Exception:
        scale_f = scale

    _fwd_kernel[grid](
        q, k, v, e, g, beta, gamma, out,
        s_qb, s_qh, s_qt, s_qk,
        s_kb, s_kh, s_kt, s_kk,
        s_vb, s_vh, s_vt, s_vv,
        s_eb, s_eh, s_et, s_ek,
        s_gb, s_gh, s_gt, s_gk,
        s_betab, s_betah, s_betat,
        s_gammab, s_gammah, s_gammat,
        s_ob, s_oh, s_ot, s_ov,
        B, H, T, K, V,
        scale_f,
        BLOCK_K, BLOCK_V,
        num_warps=4,
    )
    return out
