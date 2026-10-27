import math

import torch
import triton
import triton.language as tl


@triton.jit
def _inv_iplus(L, CHUNK: tl.constexpr, STEPS: tl.constexpr):
    # Inverse of (I + L), where L is strictly lower triangular.
    # (I+L)^-1 = (I-L)(I+L^2)(I+L^4)... because L is nilpotent.
    offs = tl.arange(0, CHUNK)
    eye = (offs[:, None] == offs[None, :]).to(tl.float32)
    inv = eye - L
    power = tl.dot(L, L)
    for _ in range(STEPS):
        inv = tl.dot(inv, eye + power)
        power = tl.dot(power, power)
    return inv


@triton.jit
def _layer_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    q_sb, q_sh, q_st, q_sk,
    k_sb, k_sh, k_st, k_sk,
    v_sb, v_sh, v_st, v_sv,
    e_sb, e_sh, e_st, e_sk,
    g_sb, g_sh, g_st, g_sk,
    beta_sb, beta_sh, beta_st,
    gamma_sb, gamma_sh, gamma_st,
    out_sb, out_sh, out_st, out_sv,
    num_heads, seq_len, head_k, head_v, scale,
    CHUNK: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    ISTEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // num_heads
    h_idx = pid % num_heads
    b64 = b_idx.to(tl.int64)
    h64 = h_idx.to(tl.int64)

    q_base = q_ptr + b64 * q_sb + h64 * q_sh
    k_base = k_ptr + b64 * k_sb + h64 * k_sh
    v_base = v_ptr + b64 * v_sb + h64 * v_sh
    e_base = e_ptr + b64 * e_sb + h64 * e_sh
    g_base = g_ptr + b64 * g_sb + h64 * g_sh
    beta_base = beta_ptr + b64 * beta_sb + h64 * beta_sh
    gamma_base = gamma_ptr + b64 * gamma_sb + h64 * gamma_sh
    out_base = out_ptr + b64 * out_sb + h64 * out_sh

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    strict_f = (offs_c[:, None] > offs_c[None, :]).to(tl.float32)
    incl_f = (offs_c[:, None] >= offs_c[None, :]).to(tl.float32)
    mask_k = offs_k[None, :] < head_k
    mask_v = offs_v[None, :] < head_v

    state = tl.zeros([BK, BV], dtype=tl.float32)
    num_chunks = tl.cdiv(seq_len, CHUNK)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK
        t_idx = start + offs_c
        mask_c = t_idx < seq_len
        t64 = t_idx.to(tl.int64)

        qmat = tl.load(
            q_base + t64[:, None] * q_st + offs_k[None, :].to(tl.int64) * q_sk,
            mask=mask_c[:, None] & mask_k, other=0.0,
        ).to(tl.float32)
        kmat = tl.load(
            k_base + t64[:, None] * k_st + offs_k[None, :].to(tl.int64) * k_sk,
            mask=mask_c[:, None] & mask_k, other=0.0,
        ).to(tl.float32)
        vmat = tl.load(
            v_base + t64[:, None] * v_st + offs_v[None, :].to(tl.int64) * v_sv,
            mask=mask_c[:, None] & mask_v, other=0.0,
        ).to(tl.float32)
        emat = tl.load(
            e_base + t64[:, None] * e_st + offs_k[None, :].to(tl.int64) * e_sk,
            mask=mask_c[:, None] & mask_k, other=0.0,
        ).to(tl.float32)
        gmat = tl.load(
            g_base + t64[:, None] * g_st + offs_k[None, :].to(tl.int64) * g_sk,
            mask=mask_c[:, None] & mask_k, other=0.0,
        ).to(tl.float32)
        beta = tl.load(beta_base + t64 * beta_st, mask=mask_c, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_base + t64 * gamma_st, mask=mask_c, other=0.0).to(tl.float32)
        alpha = gamma / (1.0 - gamma)

        # Cumulative decay inside this chunk.  p_prev is the decay before step i.
        cum_g = tl.cumsum(gmat, axis=0)
        decay = tl.exp(cum_g)
        decay_prev = tl.exp(cum_g - gmat)
        fmat = decay_prev * emat          # decayed-before e, for A^T / A^-1 factors
        hmat = emat / decay_prev          # inverse-decayed e
        ftmat = decay * emat              # decayed-through e, for final A product

        gram_ee = tl.dot(emat, tl.trans(emat))

        # \hat q_i = A_i^T q_i and \hat k_i = A_i^T k_i, where
        # A_i = (I-gamma_i e_i e_i^T) diag(exp(g_i)) ... (first step).
        lh = gamma[:, None] * tl.dot(fmat, tl.trans(emat)) * strict_f
        inv_lh = _inv_iplus(lh, CHUNK, ISTEPS)
        coef = tl.dot(inv_lh, gamma[:, None] * tl.dot(fmat, tl.trans(kmat)))
        hat_k = decay * (kmat - tl.dot(tl.trans(coef), emat))
        coef = tl.dot(inv_lh, gamma[:, None] * tl.dot(fmat, tl.trans(qmat)))
        hat_q = decay * (qmat - tl.dot(tl.trans(coef), emat))

        # \tilde k_i = A_i^-1 k_i.
        lk = alpha[:, None] * tl.dot(hmat, tl.trans(fmat)) * strict_f
        inv_lk = _inv_iplus(-lk, CHUNK, ISTEPS)
        coef = tl.dot(inv_lk, alpha[:, None] * tl.dot(hmat, tl.trans(kmat)))
        tilde_k = (kmat + tl.dot(tl.trans(coef), fmat)) / decay

        # \tilde v_i = (E_0 ... E_i)^-T v_i.
        lv = alpha[:, None] * gram_ee * strict_f
        inv_lv = _inv_iplus(-lv, CHUNK, ISTEPS)
        coef = tl.dot(inv_lv, alpha[:, None] * tl.dot(emat, tl.trans(vmat)))
        tilde_v = vmat + tl.dot(tl.trans(coef), emat)

        # Generalized delta rule in the transformed coordinates:
        # S_i = S_{i-1} + tilde_k_i u_i^T,
        # u_i = beta_i (tilde_v_i - S_{i-1}^T hat_k_i).
        hk_state = tl.dot(hat_k, state)
        hq_state = tl.dot(hat_q, state)

        inter_kk = tl.dot(hat_k, tl.trans(tilde_k)) * strict_f
        inv_delta = _inv_iplus(beta[:, None] * inter_kk, CHUNK, ISTEPS)
        upd = tl.dot(inv_delta, beta[:, None] * (tilde_v - hk_state))

        inter_qk = tl.dot(hat_q, tl.trans(tilde_k)) * incl_f
        out_tilde = hq_state + tl.dot(inter_qk, upd)

        # Back to original coordinates on the value side:
        # o_i = scale * (E_0 ... E_i)^T \tilde o_i.
        lo = gamma[:, None] * gram_ee * strict_f
        inv_lo = _inv_iplus(lo, CHUNK, ISTEPS)
        coef = tl.dot(inv_lo, gamma[:, None] * tl.dot(emat, tl.trans(out_tilde)))
        out_mat = (out_tilde - tl.dot(tl.trans(coef), emat)) * scale

        tl.store(
            out_base + t64[:, None] * out_st + offs_v[None, :].to(tl.int64) * out_sv,
            out_mat,
            mask=mask_c[:, None] & mask_v,
        )

        # Final transformed state, then S_end = A_chunk \tilde S_end R_chunk.
        state_tilde = state + tl.dot(tl.trans(tilde_k), upd)

        ls = gamma[:, None] * tl.dot(ftmat, tl.trans(emat)) * strict_f
        inv_ls = _inv_iplus(ls, CHUNK, ISTEPS)
        coef_s = tl.dot(inv_ls, gamma[:, None] * tl.dot(ftmat, state_tilde))
        decay_last = tl.sum(
            tl.where(offs_c[:, None] == (CHUNK - 1), decay, 0.0), axis=0
        )
        state_left = decay_last[:, None] * (
            state_tilde - tl.dot(tl.trans(emat), coef_s)
        )

        coef_r = tl.dot(
            inv_lo, gamma[:, None] * tl.dot(emat, tl.trans(state_left))
        )
        state = state_left - tl.dot(tl.trans(coef_r), emat)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0:
        return out

    chunk = 32
    bk = max(16, triton.next_power_of_2(K))
    bv = max(16, triton.next_power_of_2(V))
    isteps = int(math.log2(chunk)) - 1

    _layer_kernel[(B * H,)](
        q, k, v, e, g, beta, gamma, out,
        q_sb=q.stride(0), q_sh=q.stride(1), q_st=q.stride(2), q_sk=q.stride(3),
        k_sb=k.stride(0), k_sh=k.stride(1), k_st=k.stride(2), k_sk=k.stride(3),
        v_sb=v.stride(0), v_sh=v.stride(1), v_st=v.stride(2), v_sv=v.stride(3),
        e_sb=e.stride(0), e_sh=e.stride(1), e_st=e.stride(2), e_sk=e.stride(3),
        g_sb=g.stride(0), g_sh=g.stride(1), g_st=g.stride(2), g_sk=g.stride(3),
        beta_sb=beta.stride(0), beta_sh=beta.stride(1), beta_st=beta.stride(2),
        gamma_sb=gamma.stride(0), gamma_sh=gamma.stride(1), gamma_st=gamma.stride(2),
        out_sb=out.stride(0), out_sh=out.stride(1), out_st=out.stride(2), out_sv=out.stride(3),
        num_heads=H, seq_len=T, head_k=K, head_v=V, scale=float(scale),
        CHUNK=chunk, BK=bk, BV=bv, ISTEPS=isteps,
        num_warps=8, num_stages=2,
    )
    return out
