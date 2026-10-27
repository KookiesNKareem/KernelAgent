import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr,
    s_qb, s_qh, s_qt, s_qk,
    s_kb, s_kh, s_kt, s_kk,
    s_vb, s_vh, s_vt, s_vv,
    s_eb, s_eh, s_et, s_ek,
    s_gb, s_gh, s_gt, s_gk,
    s_betab, s_betah, s_betat,
    s_gammab, s_gammah, s_gammat,
    s_ob, s_oh, s_ot, s_od,
    B, H, T, Ksz, Vsz,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    mask_k = off_k < Ksz
    mask_v = off_v < Vsz
    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    for t in range(T):
        k_vec = tl.load(k_ptr + b * s_kb + h * s_kh + t * s_kt + off_k * s_kk, mask=mask_k, other=0.0).to(tl.float32)
        e_vec = tl.load(e_ptr + b * s_eb + h * s_eh + t * s_et + off_k * s_ek, mask=mask_k, other=0.0).to(tl.float32)
        g_vec = tl.load(g_ptr + b * s_gb + h * s_gh + t * s_gt + off_k * s_gk, mask=mask_k, other=0.0).to(tl.float32)
        q_vec = tl.load(q_ptr + b * s_qb + h * s_qh + t * s_qt + off_k * s_qk, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptr + b * s_vb + h * s_vh + t * s_vt + off_v * s_vv, mask=mask_v, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + b * s_betab + h * s_betah + t * s_betat)
        gamma_t = tl.load(gamma_ptr + b * s_gammab + h * s_gammah + t * s_gammat)
        d_vec = tl.exp(g_vec)
        S = S * d_vec[:, None]
        a_vec = tl.sum(S * e_vec[:, None], axis=0)
        S = S - gamma_t * e_vec[:, None] * a_vec[None, :]
        pred_vec = tl.sum(S * k_vec[:, None], axis=0)
        delta_vec = (v_vec - pred_vec) * beta_t
        S = S + k_vec[:, None] * delta_vec[None, :]
        qs_vec = (q_vec * scale).to(tl.float32)
        out_vec = tl.sum(S * qs_vec[:, None], axis=0)
        tl.store(o_ptr + b * s_ob + h * s_oh + t * s_ot + off_v * s_od, out_vec, mask=mask_v)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    out = torch.empty(B, H, T, Vd, dtype=torch.float32, device=q.device)
    # normalize channel-decay to [B,H,T,K] view (expand is stride-0, no copy)
    if g.dim() == 3:
        g = g.unsqueeze(-1).expand(B, H, T, K)
    elif g.dim() == 4 and g.shape[-1] != K:
        if g.shape[-1] == 1:
            g = g.expand(B, H, T, K)
    def _np2(x):
        return 1 << (x - 1).bit_length() if x > 1 else 1
    BLOCK_K = max(16, _np2(K))
    BLOCK_V = max(16, _np2(Vd))
    scale_f = float(scale)
    grid = (B * H,)
    nw = 8 if BLOCK_K * BLOCK_V > 1024 else 4
    _fwd_kernel[grid](
        q, k, v, e, g, beta, gamma, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        e.stride(0), e.stride(1), e.stride(2), e.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, T, K, Vd,
        scale_f,
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        num_warps=nw,
    )
    return out
