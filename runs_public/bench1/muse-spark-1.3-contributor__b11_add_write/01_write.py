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
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = bh // H
    h = bh % H

    k_offsets = tl.arange(0, BLOCK_K)
    v_offsets = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = k_offsets < K
    mask_v = v_offsets < V

    q_bh = b * s_qb + h * s_qh
    k_bh = b * s_kb + h * s_kh
    v_bh = b * s_vb + h * s_vh
    e_bh = b * s_eb + h * s_eh
    g_bh = b * s_gb + h * s_gh
    beta_bh = b * s_betab + h * s_betah
    gamma_bh = b * s_gammab + h * s_gammah
    o_bh = b * s_ob + h * s_oh

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in tl.range(T):
        off_q = q_bh + t * s_qt + k_offsets * s_qk
        off_k = k_bh + t * s_kt + k_offsets * s_kk
        off_e = e_bh + t * s_et + k_offsets * s_ek
        off_g = g_bh + t * s_gt + k_offsets * s_gk
        off_v = v_bh + t * s_vt + v_offsets * s_vv
        off_o = o_bh + t * s_ot + v_offsets * s_ov

        qv = tl.load(q_ptr + off_q, mask=mask_k, other=0.0).to(tl.float32)
        kv = tl.load(k_ptr + off_k, mask=mask_k, other=0.0).to(tl.float32)
        ev = tl.load(e_ptr + off_e, mask=mask_k, other=0.0).to(tl.float32)
        gv = tl.load(g_ptr + off_g, mask=mask_k, other=0.0).to(tl.float32)
        vv = tl.load(v_ptr + off_v, mask=mask_v, other=0.0).to(tl.float32)
        bv = tl.load(beta_ptr + beta_bh + t * s_betat).to(tl.float32)
        cv = tl.load(gamma_ptr + gamma_bh + t * s_gammat).to(tl.float32)

        qv = qv * scale
        decay = tl.exp(gv)

        decay_e = tl.expand_dims(decay, 1)
        k_e = tl.expand_dims(kv, 1)
        e_e = tl.expand_dims(ev, 1)
        q_e = tl.expand_dims(qv, 1)
        v_e = tl.expand_dims(vv, 0)

        S = S * decay_e
        pred = tl.sum(S * k_e, 0)
        diff = (vv - pred) * bv
        diff_e = tl.expand_dims(diff, 0)
        S = S + k_e * diff_e
        S = S + cv * e_e * v_e
        outv = tl.sum(S * q_e, 0)
        tl.store(out_ptr + off_o, outv, mask=mask_v)


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    device = q.device
    out = torch.empty((B, H, T, V), dtype=torch.float32, device=device)
    if B == 0 or H == 0 or T == 0 or V == 0:
        return out
    if K == 0:
        out.zero_()
        return out

    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)

    BLOCK_K = _next_pow2(K)
    if BLOCK_K < 16:
        BLOCK_K = 16
    if BLOCK_K <= 32:
        BLOCK_V = 64
    elif BLOCK_K <= 128:
        BLOCK_V = 32
    else:
        BLOCK_V = 16

    NV = (V + BLOCK_V - 1) // BLOCK_V
    BH = B * H
    grid = (BH, NV)

    s_qb, s_qh, s_qt, s_qk = q.stride()
    s_kb, s_kh, s_kt, s_kk = k.stride()
    s_vb, s_vh, s_vt, s_vv = v.stride()
    s_eb, s_eh, s_et, s_ek = e.stride()
    s_gb, s_gh, s_gt, s_gk = g.stride()
    s_betab, s_betah, s_betat = beta.stride()
    s_gammab, s_gammah, s_gammat = gamma.stride()
    s_ob, s_oh, s_ot, s_ov = out.stride()

    num_warps = 8 if (BLOCK_K * BLOCK_V > 4096) else 4

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
        num_warps=num_warps,
    )
    return out
