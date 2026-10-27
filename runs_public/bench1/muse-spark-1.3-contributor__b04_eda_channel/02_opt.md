```python
import torch
import triton
import triton.language as tl


@triton.jit
def _fast128_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr,
    H,
    T: tl.constexpr,
    s_qb, s_qh, s_qt, s_qk,
    s_kb, s_kh, s_kt, s_kk,
    s_vb, s_vh, s_vt, s_vv,
    s_eb, s_eh, s_et, s_ek,
    s_gb, s_gh, s_gt, s_gk,
    s_betab, s_betah, s_betat,
    s_gammab, s_gammah, s_gammat,
    s_ob, s_oh, s_ot, s_ov,
    scale,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    offs_k = tl.arange(0, 128)
    offs_v = tl.arange(0, 128)

    q_bh = b * s_qb + h * s_qh
    k_bh = b * s_kb + h * s_kh
    v_bh = b * s_vb + h * s_vh
    e_bh = b * s_eb + h * s_eh
    g_bh = b * s_gb + h * s_gh
    beta_bh = b * s_betab + h * s_betah
    gamma_bh = b * s_gammab + h * s_gammah
    o_bh = b * s_ob + h * s_oh

    S = tl.zeros([128, 128], dtype=tl.float32)

    for t in tl.range(0, T):
        q_vec = tl.load(q_ptr + q_bh + t * s_qt + offs_k * s_qk).to(tl.float32)
        k_vec = tl.load(k_ptr + k_bh + t * s_kt + offs_k * s_kk).to(tl.float32)
        e_vec = tl.load(e_ptr + e_bh + t * s_et + offs_k * s_ek).to(tl.float32)
        g_vec = tl.load(g_ptr + g_bh + t * s_gt + offs_k * s_gk).to(tl.float32)
        v_vec = tl.load(v_ptr + v_bh + t * s_vt + offs_v * s_vv).to(tl.float32)
        bt = tl.load(beta_ptr + beta_bh + t * s_betat).to(tl.float32)
        ct = tl.load(gamma_ptr + gamma_bh + t * s_gammat).to(tl.float32)

        d_vec = tl.exp(g_vec)
        S = S * tl.expand_dims(d_vec, 1)
        a_vec = tl.sum(S * tl.expand_dims(e_vec, 1), axis=0)
        S = S - tl.expand_dims(ct * e_vec, 1) * tl.expand_dims(a_vec, 0)
        pred_vec = tl.sum(S * tl.expand_dims(k_vec, 1), axis=0)
        delta_vec = (v_vec - pred_vec) * bt
        S = S + tl.expand_dims(k_vec, 1) * tl.expand_dims(delta_vec, 0)
        out_vec = tl.sum(S * tl.expand_dims(q_vec * scale, 1), axis=0)
        tl.store(o_ptr + o_bh + t * s_ot + offs_v * s_ov, out_vec)


@triton.jit
def _fused_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr,
    H,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    s_qb, s_qh, s_qt, s_qk,
    s_kb, s_kh, s_kt, s_kk,
    s_vb, s_vh, s_vt, s_vv,
    s_eb, s_eh, s_et, s_ek,
    s_gb, s_gh, s_gt, s_gk,
    s_betab, s_betah, s_betat,
    s_gammab, s_gammah, s_gammat,
    s_ob, s_oh, s_ot, s_ov,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    nV = tl.cdiv(V, BLOCK_V)
    bh = pid // nV
    iv = pid % nV
    b = bh // H
    h = bh % H
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    offs_v = iv * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    q_bh = b * s_qb + h * s_qh
    k_bh = b * s_kb + h * s_kh
    v_bh = b * s_vb + h * s_vh
    e_bh = b * s_eb + h * s_eh
    g_bh = b * s_gb + h * s_gh
    beta_bh = b * s_betab + h * s_betah
    gamma_bh = b * s_gammab + h * s_gammah
    o_bh = b * s_ob + h * s_oh

    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    for t in tl.range(0, T):
        q_vec = tl.load(q_ptr + q_bh + t * s_qt + offs_k * s_qk, mask=mask_k, other=0.0).to(tl.float32)
        k_vec = tl.load(k_ptr + k_bh + t * s_kt + offs_k * s_kk, mask=mask_k, other=0.0).to(tl.float32)
        e_vec = tl.load(e_ptr + e_bh + t * s_et + offs_k * s_ek, mask=mask_k, other=0.0).to(tl.float32)
        g_vec = tl.load(g_ptr + g_bh + t * s_gt + offs_k * s_gk, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptr + v_bh + t * s_vt + offs_v * s_vv, mask=mask_v, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + beta_bh + t * s_betat).to(tl.float32)
        gamma = tl.load(gamma_ptr + gamma_bh + t * s_gammat).to(tl.float32)

        d_vec = tl.exp(g_vec)
        S = S * tl.expand_dims(d_vec, 1)
        a_vec = tl.sum(S * tl.expand_dims(e_vec, 1), axis=0)
        S = S - tl.expand_dims(gamma * e_vec, 1) * tl.expand_dims(a_vec, 0)
        pred_vec = tl.sum(S * tl.expand_dims(k_vec, 1), axis=0)
        delta_vec = (v_vec - pred_vec) * beta
        S = S + tl.expand_dims(k_vec, 1) * tl.expand_dims(delta_vec, 0)
        qs_vec = q_vec * scale
        out_vec = tl.sum(S * tl.expand_dims(qs_vec, 1), axis=0)
        tl.store(o_ptr + o_bh + t * s_ot + offs_v * s_ov, out_vec, mask=mask_v)


@triton.jit
def _stream_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr, state_ptr,
    H,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    s_qb, s_qh, s_qt, s_qk,
    s_kb, s_kh, s_kt, s_kk,
    s_vb, s_vh, s_vt, s_vv,
    s_eb, s_eh, s_et, s_ek,
    s_gb, s_gh, s_gt, s_gk,
    s_betab, s_betah, s_betat,
    s_gammab, s_gammah, s_gammat,
    s_ob, s_oh, s_ot, s_ov,
    s_sb, s_sh, s_sk, s_sv,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    nV = tl.cdiv(V, BLOCK_V)
    bh = pid // nV
    iv = pid % nV
    b = bh // H
    h = bh % H
    offs_v = iv * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V
    nK = tl.cdiv(K, BLOCK_K)

    q_bh = b * s_qb + h * s_qh
    k_bh = b * s_kb + h * s_kh
    v_bh = b * s_vb + h * s_vh
    e_bh = b * s_eb + h * s_eh
    g_bh = b * s_gb + h * s_gh
    beta_bh = b * s_betab + h * s_betah
    gamma_bh = b * s_gammab + h * s_gammah
    o_bh = b * s_ob + h * s_oh
    s_bh = b * s_sb + h * s_sh
    ev = tl.expand_dims(offs_v, 0)
    mv = tl.expand_dims(mask_v, 0)

    for t in tl.range(0, T):
        v_vec = tl.load(v_ptr + v_bh + t * s_vt + offs_v * s_vv, mask=mask_v, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + beta_bh + t * s_betat).to(tl.float32)
        gamma = tl.load(gamma_ptr + gamma_bh + t * s_gammat).to(tl.float32)

        a_vec = tl.zeros([BLOCK_V], dtype=tl.float32)
        for ik in tl.range(0, nK):
            k0 = ik * BLOCK_K
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            g_vec = tl.load(g_ptr + g_bh + t * s_gt + offs_k * s_gk, mask=mask_k, other=0.0).to(tl.float32)
            e_vec = tl.load(e_ptr + e_bh + t * s_et + offs_k * s_ek, mask=mask_k, other=0.0).to(tl.float32)
            s_ptrs = state_ptr + s_bh + tl.expand_dims(offs_k, 1) * s_sk + ev * s_sv
            mask_2d = tl.expand_dims(mask_k, 1) & mv
            s_tile = tl.load(s_ptrs, mask=mask_2d, other=0.0)
            d_vec = tl.exp(g_vec)
            s_tile = s_tile * tl.expand_dims(d_vec, 1)
            tl.store(s_ptrs, s_tile, mask=mask_2d)
            a_vec = a_vec + tl.sum(s_tile * tl.expand_dims(e_vec, 1), axis=0)

        pred_vec = tl.zeros([BLOCK_V], dtype=tl.float32)
        for ik in tl.range(0, nK):
            k0 = ik * BLOCK_K
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            e_vec = tl.load(e_ptr + e_bh + t * s_et + offs_k * s_ek, mask=mask_k, other=0.0).to(tl.float32)
            k_vec = tl.load(k_ptr + k_bh + t * s_kt + offs_k * s_kk, mask=mask_k, other=0.0).to(tl.float32)
            s_ptrs = state_ptr + s_bh + tl.expand_dims(offs_k, 1) * s_sk + ev * s_sv
            mask_2d = tl.expand_dims(mask_k, 1) & mv
            s_tile = tl.load(s_ptrs, mask=mask_2d, other=0.0)
            s_tile = s_tile - tl.expand_dims(gamma * e_vec, 1) * tl.expand_dims(a_vec, 0)
            pred_vec = pred_vec + tl.sum(s_tile * tl.expand_dims(k_vec, 1), axis=0)
            tl.store(s_ptrs, s_tile, mask=mask_2d)

        delta_vec = (v_vec - pred_vec) * beta

        out_vec = tl.zeros([BLOCK_V], dtype=tl.float32)
        for ik in tl.range(0, nK):
            k0 = ik * BLOCK_K
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            k_vec = tl.load(k_ptr + k_bh + t * s_kt + offs_k * s_kk, mask=mask_k, other=0.0).to(tl.float32)
            q_vec = tl.load(q_ptr + q_bh + t * s_qt + offs_k * s_qk, mask=mask_k, other=0.0).to(tl.float32)
            qs_vec = q_vec * scale
            s_ptrs = state_ptr + s_bh + tl.expand_dims(offs_k, 1) * s_sk + ev * s_sv
            mask_2d = tl.expand_dims(mask_k, 1) & mv
            s_tile = tl.load(s_ptrs, mask=mask_2d, other=0.0)
            s_tile = s_tile + tl.expand_dims(k_vec, 1) * tl.expand_dims(delta_vec, 0)
            out_vec = out_vec + tl.sum(s_tile * tl.expand_dims(qs_vec, 1), axis=0)
            tl.store(s_ptrs, s_tile, mask=mask_2d)
        tl.store(o_ptr + o_bh + t * s_ot + offs_v * s_ov, out_vec, mask=mask_v)


def _np2(x, minimum=16):
    if x <= minimum:
        return minimum
    return 1 << ((x - 1).bit_length())


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    device = q.device
    if isinstance(scale, torch.Tensor):
        scale_val = float(scale.item())
    else:
        scale_val = float(scale)
    out = torch.empty(B, H, T, V, dtype=torch.float32, device=device)
    if T == 0 or V == 0 or K == 0:
        return out
    s_qb, s_qh, s_qt, s_qk = q.stride()
    s_kb, s_kh, s_kt, s_kk = k.stride()
    s_vb, s_vh, s_vt, s_vv = v.stride()
    s_eb, s_eh, s_et, s_ek = e.stride()
    s_gb, s_gh, s_gt, s_gk = g.stride()
    s_betab, s_betah, s_betat = beta.stride()
    s_gammab, s_gammah, s_gammat = gamma.stride()
    s_ob, s_oh, s_ot, s_ov = out.stride()

    if K == 128 and V == 128:
        grid = (B * H,)
        _fast128_kernel[grid](
            q, k, v, e, g, beta, gamma, out,
            H, T,
            s_qb, s_qh, s_qt, s_qk,
            s_kb, s_kh, s_kt, s_kk,
            s_vb, s_vh, s_vt, s_vv,
            s_eb, s_eh, s_et, s_ek,
            s_gb, s_gh, s_gt, s_gk,
            s_betab, s_betah, s_betat,
            s_gammab, s_gammah, s_gammat,
            s_ob, s_oh, s_ot, s_ov,
            scale_val,
            num_warps=8, num_stages=2,
        )
    elif K <= 128:
        BLOCK_K = _np2(K)
        BLOCK_V = _np2(V) if V <= 64 else 64
        nV = (V + BLOCK_V - 1) // BLOCK_V
        grid = (B * H * nV,)
        _fused_kernel[grid](
            q, k, v, e, g, beta, gamma, out,
            H, T, K, V,
            s_qb, s_qh, s_qt, s_qk,
            s_kb, s_kh, s_kt, s_kk,
            s_vb, s_vh, s_vt, s_vv,
            s_eb, s_eh, s_et, s_ek,
            s_gb, s_gh, s_gt, s_gk,
            s_betab, s_betah, s_betat,
            s_gammab, s_gammah, s_gammat,
            s_ob, s_oh, s_ot, s_ov,
            scale_val,
            BLOCK_K, BLOCK_V,
            num_warps=8, num_stages=1,
        )
    else:
        BLOCK_K = 64
        BLOCK_V = _np2(V) if V <= 64 else 64
        nV = (V + BLOCK_V - 1) // BLOCK_V
        state = torch.zeros(B, H, K, V, dtype=torch.float32, device=device)
        s_sb, s_sh, s_sk, s_sv = state.stride()
        grid = (B * H * nV,)
        _stream_kernel[grid](
            q, k, v, e, g, beta, gamma, out, state,
            H, T, K, V,
            s_qb, s_qh, s_qt, s_qk,
            s_kb, s_kh, s_kt, s_kk,
            s_vb, s_vh, s_vt, s_vv,
            s_eb, s_eh, s_et, s_ek,
            s_gb, s_gh, s_gt, s_gk,
            s_betab, s_betah, s_betat,
            s_gammab, s_gammah, s_gammat,
            s_ob, s_oh, s_ot, s_ov,
            s_sb, s_sh, s_sk, s_sv,
            scale_val,
            BLOCK_K, BLOCK_V,
            num_warps=4, num_stages=1,
        )
    return out
```
My approach keeps the full 128×128 state in registers with one CTA per head, removing the 4× duplication of K-vector loads/exps and all masking/predication for the benchmark shape. Generic shapes fall back to V-sharded register or HBM-streaming kernels. I am unsure about register pressure/spilling for the 16k-float tile and exact Triton pipelining behavior on sm_89, but fp32 order matches the reference so the 2e-2 tolerance should hold.