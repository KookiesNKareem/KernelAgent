import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr,
    H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    s_qb, s_qh, s_qt, s_qk,
    s_kb, s_kh, s_kt, s_kk,
    s_vb, s_vh, s_vt, s_vk,
    s_eb, s_eh, s_et, s_ek,
    s_gb, s_gh, s_gt, s_gk,
    s_betab, s_betah, s_betat,
    s_gammab, s_gammah, s_gammat,
    s_ob, s_oh, s_ot, s_ov,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    q_base = b * s_qb + h * s_qh
    k_base = b * s_kb + h * s_kh
    v_base = b * s_vb + h * s_vh
    e_base = b * s_eb + h * s_eh
    g_base = b * s_gb + h * s_gh
    beta_base = b * s_betab + h * s_betah
    gamma_base = b * s_gammab + h * s_gammah
    o_base = b * s_ob + h * s_oh

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_ev = offs_v < K

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in range(T):
        kt = tl.load(k_ptr + k_base + t * s_kt + offs_k * s_kk, mask=mask_k, other=0.0).to(tl.float32)
        qt = tl.load(q_ptr + q_base + t * s_qt + offs_k * s_qk, mask=mask_k, other=0.0).to(tl.float32)
        et = tl.load(e_ptr + e_base + t * s_et + offs_v * s_ek, mask=mask_ev, other=0.0).to(tl.float32)
        gt = tl.load(g_ptr + g_base + t * s_gt + offs_k * s_gk, mask=mask_k, other=0.0).to(tl.float32)
        vt = tl.load(v_ptr + v_base + t * s_vt + offs_v * s_vk, mask=mask_v, other=0.0).to(tl.float32)
        bt = tl.load(beta_ptr + beta_base + t * s_betat).to(tl.float32)
        ct = tl.load(gamma_ptr + gamma_base + t * s_gammat).to(tl.float32)

        d = tl.exp(gt)
        S = S * d[:, None]

        p = tl.sum(S * et[None, :], axis=1)
        S = S - (ct * p)[:, None] * et[None, :]

        pred = tl.sum(S * kt[:, None], axis=0)
        err = (vt - pred) * bt
        S = S + kt[:, None] * err[None, :]

        qts = qt * scale
        out = tl.sum(S * qts[:, None], axis=0)
        tl.store(o_ptr + o_base + t * s_ot + offs_v * s_ov, out, mask=mask_v)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    out_dtype = torch.promote_types(q.dtype, torch.float32)
    o = torch.empty(B, H, T, V, dtype=out_dtype, device=q.device)
    if T == 0 or K == 0 or V == 0:
        return o
    def _np2(x):
        return 1 if x <= 1 else 1 << (x - 1).bit_length()
    BLOCK_K = _np2(K)
    BLOCK_V = _np2(V)
    s_qb, s_qh, s_qt, s_qk = q.stride()
    s_kb, s_kh, s_kt, s_kk = k.stride()
    s_vb, s_vh, s_vt, s_vk = v.stride()
    s_eb, s_eh, s_et, s_ek = e.stride()
    s_gb, s_gh, s_gt, s_gk = g.stride()
    s_betab, s_betah, s_betat = beta.stride()
    s_gammab, s_gammah, s_gammat = gamma.stride()
    s_ob, s_oh, s_ot, s_ov = o.stride()
    try:
        scale_f = float(scale)
    except Exception:
        scale_f = scale
    grid = (B * H,)
    nw = 4
    if BLOCK_K * BLOCK_V > 2048:
        nw = 8
    _fwd_kernel[grid](
        q, k, v, e, g, beta, gamma, o,
        H, T, K, V,
        s_qb, s_qh, s_qt, s_qk,
        s_kb, s_kh, s_kt, s_kk,
        s_vb, s_vh, s_vt, s_vk,
        s_eb, s_eh, s_et, s_ek,
        s_gb, s_gh, s_gt, s_gk,
        s_betab, s_betah, s_betat,
        s_gammab, s_gammah, s_gammat,
        s_ob, s_oh, s_ot, s_ov,
        scale_f,
        BLOCK_K, BLOCK_V,
        num_warps=nw,
    )
    return o
