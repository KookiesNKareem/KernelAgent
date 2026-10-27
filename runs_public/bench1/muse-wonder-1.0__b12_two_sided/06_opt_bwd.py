import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward: exact per-step recurrence, with optional chunk checkpoints
# (state at the start of every _CHUNK steps) for the backward pass.
# ---------------------------------------------------------------------------
@triton.jit
def _forward_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr, chk_ptr,
    q_sb, q_sh, q_st, q_sk,
    k_sb, k_sh, k_st, k_sk,
    v_sb, v_sh, v_st, v_sv,
    e_sb, e_sh, e_st, e_sk,
    g_sb, g_sh, g_st, g_sk,
    beta_sb, beta_sh, beta_st,
    gamma_sb, gamma_sh, gamma_st,
    out_sb, out_sh, out_st, out_sv,
    num_heads, seq_len, head_k, head_v, state_elems, chk_bh_stride, scale_f,
    BK: tl.constexpr, BV: tl.constexpr, WRITE_CHK: tl.constexpr,
    CKPT_CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // num_heads
    h_idx = pid % num_heads
    b64 = b_idx.to(tl.int64)
    h64 = h_idx.to(tl.int64)

    q_cur = q_ptr + b64 * q_sb + h64 * q_sh
    k_cur = k_ptr + b64 * k_sb + h64 * k_sh
    v_cur = v_ptr + b64 * v_sb + h64 * v_sh
    e_cur = e_ptr + b64 * e_sb + h64 * e_sh
    g_cur = g_ptr + b64 * g_sb + h64 * g_sh
    beta_cur = beta_ptr + b64 * beta_sb + h64 * beta_sh
    gamma_cur = gamma_ptr + b64 * gamma_sb + h64 * gamma_sh
    out_cur = out_ptr + b64 * out_sb + h64 * out_sh
    chk_cur = chk_ptr + (b64 * num_heads + h64) * chk_bh_stride

    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    offs_k64 = offs_k.to(tl.int64)
    offs_v64 = offs_v.to(tl.int64)
    mask_k = offs_k < head_k
    mask_v = offs_v < head_v
    mask_ev = offs_v < head_k
    tile_off = offs_k[:, None] * head_v + offs_v[None, :]
    tile_mask = mask_k[:, None] & mask_v[None, :]

    state = tl.zeros([BK, BV], dtype=tl.float32)
    step_in_chunk = 0

    for _ in range(seq_len):
        if WRITE_CHK:
            if step_in_chunk == 0:
                tl.store(chk_cur + tile_off, state, mask=tile_mask)

        q_vec = tl.load(q_cur + offs_k64 * q_sk, mask=mask_k, other=0.0).to(tl.float32)
        k_vec = tl.load(k_cur + offs_k64 * k_sk, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_cur + offs_v64 * v_sv, mask=mask_v, other=0.0).to(tl.float32)
        e_k = tl.load(e_cur + offs_k64 * e_sk, mask=mask_k, other=0.0).to(tl.float32)
        e_v = tl.load(e_cur + offs_v64 * e_sk, mask=mask_ev, other=0.0).to(tl.float32)
        g_vec = tl.load(g_cur + offs_k64 * g_sk, mask=mask_k, other=0.0).to(tl.float32)
        beta_val = tl.load(beta_cur).to(tl.float32)
        gamma_val = tl.load(gamma_cur).to(tl.float32)

        state = state * tl.exp(g_vec)[:, None]
        left_dot = tl.sum(state * e_k[:, None], axis=0)
        state = state - (gamma_val * e_k)[:, None] * left_dot[None, :]
        right_dot = tl.sum(state * e_v[None, :], axis=1)
        state = state - right_dot[:, None] * (gamma_val * e_v)[None, :]
        pred = tl.sum(state * k_vec[:, None], axis=0)
        state = state + k_vec[:, None] * (beta_val * (v_vec - pred))[None, :]
        out_vec = tl.sum(state * (q_vec * scale_f)[:, None], axis=0)
        tl.store(out_cur + offs_v64 * out_sv, out_vec, mask=mask_v)

        q_cur += q_st
        k_cur += k_st
        v_cur += v_st
        e_cur += e_st
        g_cur += g_st
        beta_cur += beta_st
        gamma_cur += gamma_st
        out_cur += out_st

        if WRITE_CHK:
            step_in_chunk += 1
            if step_in_chunk == CKPT_CHUNK:
                step_in_chunk = 0
                chk_cur += state_elems


def _run_forward(q, k, v, e, g, beta, gamma, out, scale_f, chk, num_chunks):
    B, H, T, K = q.shape
    V = v.shape[-1]
    bk = max(16, triton.next_power_of_2(K))
    bv = max(16, triton.next_power_of_2(V))
    num_warps = 16 if bk * bv >= 16384 else int(max(4, min(32, (bk * bv) // 2048)))
    write_chk = chk is not None
    chk_arg = chk if write_chk else out
    _forward_kernel[(B * H,)](
        q, k, v, e, g, beta, gamma, out, chk_arg,
        q_sb=q.stride(0), q_sh=q.stride(1), q_st=q.stride(2), q_sk=q.stride(3),
        k_sb=k.stride(0), k_sh=k.stride(1), k_st=k.stride(2), k_sk=k.stride(3),
        v_sb=v.stride(0), v_sh=v.stride(1), v_st=v.stride(2), v_sv=v.stride(3),
        e_sb=e.stride(0), e_sh=e.stride(1), e_st=e.stride(2), e_sk=e.stride(3),
        g_sb=g.stride(0), g_sh=g.stride(1), g_st=g.stride(2), g_sk=g.stride(3),
        beta_sb=beta.stride(0), beta_sh=beta.stride(1), beta_st=beta.stride(2),
        gamma_sb=gamma.stride(0), gamma_sh=gamma.stride(1), gamma_st=gamma.stride(2),
        out_sb=out.stride(0), out_sh=out.stride(1), out_st=out.stride(2), out_sv=out.stride(3),
        num_heads=H, seq_len=T, head_k=K, head_v=V,
        state_elems=K * V, chk_bh_stride=(num_chunks * K * V if write_chk else 0),
        scale_f=float(scale_f),
        BK=bk, BV=bv, WRITE_CHK=write_chk, CKPT_CHUNK=_CHUNK,
        num_warps=num_warps, num_stages=1,
    )


# ---------------------------------------------------------------------------
# Backward.
#
# Per step, with Sa = diag(exp(g)) S_{t-1} (snapshotted), Sb/Sc the states
# after the left/right erase and St = S_t:
#   dq, dv, dbeta, dk, dgamma, de, dg and the adjoint propagation
#   Lam <- diag(exp(g)) (I-gamma e e^T) (I-beta k k^T applied) Lam ...
# are all expressible through *vector* actions of Sa and Lam alone:
# e.g. St @ go = Sc @ go + beta k (res.go), Sc @ w is recovered from
# Sa @ w by two scalar corrections, and dg uses rowsum(Lam_a * Sa) with
# Lam_a built from Lam by three rank-1 updates.  So the reverse sweep
# keeps only two [K,V] tiles (Sa, Lam) live instead of five, and the
# replay sweep transforms a single tile in place.  No inverse of any
# decay/erase/delta operator appears anywhere.
# ---------------------------------------------------------------------------
@triton.jit
def _backward_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr,
    go_ptr, chk_ptr, snap_ptr,
    gq_ptr, gk_ptr, gv_ptr, ge_ptr, gg_ptr, gbeta_ptr, ggamma_ptr,
    q_sb, q_sh, q_st, q_sk,
    k_sb, k_sh, k_st, k_sk,
    v_sb, v_sh, v_st, v_sv,
    e_sb, e_sh, e_st, e_sk,
    g_sb, g_sh, g_st, g_sk,
    beta_sb, beta_sh, beta_st,
    gamma_sb, gamma_sh, gamma_st,
    go_sb, go_sh, go_st, go_sv,
    num_heads, seq_len, head_k, head_v, num_chunks, state_elems,
    chk_bh_stride, snap_bh_stride, scale_f,
    CHUNK: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    SAME_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // num_heads
    h_idx = pid % num_heads
    b64 = b_idx.to(tl.int64)
    h64 = h_idx.to(tl.int64)
    bh64 = b64 * num_heads + h64

    q_base = q_ptr + b64 * q_sb + h64 * q_sh
    k_base = k_ptr + b64 * k_sb + h64 * k_sh
    v_base = v_ptr + b64 * v_sb + h64 * v_sh
    e_base = e_ptr + b64 * e_sb + h64 * e_sh
    g_base = g_ptr + b64 * g_sb + h64 * g_sh
    beta_base = beta_ptr + b64 * beta_sb + h64 * beta_sh
    gamma_base = gamma_ptr + b64 * gamma_sb + h64 * gamma_sh
    go_base = go_ptr + b64 * go_sb + h64 * go_sh

    gq_base = gq_ptr + bh64 * seq_len * head_k
    gk_base = gk_ptr + bh64 * seq_len * head_k
    gv_base = gv_ptr + bh64 * seq_len * head_v
    ge_base = ge_ptr + bh64 * seq_len * head_k
    gg_base = gg_ptr + bh64 * seq_len * head_k
    gbeta_base = gbeta_ptr + bh64 * seq_len
    ggamma_base = ggamma_ptr + bh64 * seq_len
    snap_base = snap_ptr + bh64 * snap_bh_stride

    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    offs_k64 = offs_k.to(tl.int64)
    offs_v64 = offs_v.to(tl.int64)
    mask_k = offs_k < head_k
    mask_v = offs_v < head_v
    mask_ev = offs_v < head_k
    tile_off = offs_k[:, None] * head_v + offs_v[None, :]
    tile_mask = mask_k[:, None] & mask_v[None, :]

    lam = tl.zeros([BK, BV], dtype=tl.float32)

    # Walk chunks backwards (checkpoints were written by the forward).
    # Replay each chunk, snapshotting Sa per step, then sweep it in reverse.
    chk_cur = chk_ptr + bh64 * chk_bh_stride + (num_chunks - 1) * state_elems
    for chunk_rev in range(num_chunks):
        chunk_idx = num_chunks - 1 - chunk_rev
        chunk_start = chunk_idx * CHUNK
        state = tl.load(chk_cur + tile_off, mask=tile_mask, other=0.0).to(tl.float32)

        for s in range(CHUNK):
            t_idx = chunk_start + s
            if t_idx < seq_len:
                t64 = t_idx.to(tl.int64)
                kt = tl.load(k_base + t64 * k_st + offs_k64 * k_sk,
                             mask=mask_k, other=0.0).to(tl.float32)
                vt = tl.load(v_base + t64 * v_st + offs_v64 * v_sv,
                             mask=mask_v, other=0.0).to(tl.float32)
                ek = tl.load(e_base + t64 * e_st + offs_k64 * e_sk,
                             mask=mask_k, other=0.0).to(tl.float32)
                ev = tl.load(e_base + t64 * e_st + offs_v64 * e_sk,
                             mask=mask_ev, other=0.0).to(tl.float32)
                gt = tl.load(g_base + t64 * g_st + offs_k64 * g_sk,
                             mask=mask_k, other=0.0).to(tl.float32)
                beta_s = tl.load(beta_base + t64 * beta_st).to(tl.float32)
                gamma_s = tl.load(gamma_base + t64 * gamma_st).to(tl.float32)

                state = tl.exp(gt)[:, None] * state
                tl.store(snap_base + s * state_elems + tile_off, state,
                         mask=tile_mask)
                z = tl.sum(state * ek[:, None], axis=0)
                state = state - (gamma_s * ek)[:, None] * z[None, :]
                r = tl.sum(state * ev[None, :], axis=1)
                state = state - r[:, None] * (gamma_s * ev)[None, :]
                pred = tl.sum(state * kt[:, None], axis=0)
                state = state + kt[:, None] * (beta_s * (vt - pred))[None, :]

        for s_rev in range(CHUNK):
            s = CHUNK - 1 - s_rev
            t_idx = chunk_start + s
            if t_idx < seq_len:
                t64 = t_idx.to(tl.int64)
                qt = tl.load(q_base + t64 * q_st + offs_k64 * q_sk,
                             mask=mask_k, other=0.0).to(tl.float32)
                kt = tl.load(k_base + t64 * k_st + offs_k64 * k_sk,
                             mask=mask_k, other=0.0).to(tl.float32)
                vt = tl.load(v_base + t64 * v_st + offs_v64 * v_sv,
                             mask=mask_v, other=0.0).to(tl.float32)
                ek = tl.load(e_base + t64 * e_st + offs_k64 * e_sk,
                             mask=mask_k, other=0.0).to(tl.float32)
                ev = tl.load(e_base + t64 * e_st + offs_v64 * e_sk,
                             mask=mask_ev, other=0.0).to(tl.float32)
                gt = tl.load(g_base + t64 * g_st + offs_k64 * g_sk,
                             mask=mask_k, other=0.0).to(tl.float32)
                beta_s = tl.load(beta_base + t64 * beta_st).to(tl.float32)
                gamma_s = tl.load(gamma_base + t64 * gamma_st).to(tl.float32)
                go_vec = tl.load(go_base + t64 * go_st + offs_v64 * go_sv,
                                 mask=mask_v, other=0.0).to(tl.float32)
                sa = tl.load(snap_base + s * state_elems + tile_off,
                             mask=tile_mask, other=0.0).to(tl.float32)
                exp_g = tl.exp(gt)

                lam = lam + (scale_f * qt)[:, None] * go_vec[None, :]

                # --- Vector actions of Sa (forward intermediates) ---
                ek_k = tl.sum(ek * kt)
                ev_go = tl.sum(ev * go_vec)
                z = tl.sum(sa * ek[:, None], axis=0)          # Sa^T e_k
                sa_go = tl.sum(sa * go_vec[None, :], axis=1)  # Sa @ go
                sa_ev = tl.sum(sa * ev[None, :], axis=1)      # Sa @ e_v
                sa_tk = tl.sum(sa * kt[:, None], axis=0)      # Sa^T k
                z_go = tl.sum(z * go_vec)
                z_ev = tl.sum(z * ev)
                sb_go = sa_go - gamma_s * ek * z_go
                r = sa_ev - gamma_s * ek * z_ev               # Sb @ e_v
                sb_tk = sa_tk - gamma_s * z * ek_k
                r_k = tl.sum(r * kt)
                res = vt - (sb_tk - gamma_s * ev * r_k)       # v - Sc^T k
                sc_go = sb_go - gamma_s * r * ev_go
                dq = scale_f * (sc_go + kt * (beta_s * tl.sum(res * go_vec)))
                tl.store(gq_base + t64 * head_k + offs_k64, dq, mask=mask_k)

                # --- Delta-update adjoints ---
                lam_tk = tl.sum(lam * kt[:, None], axis=0)    # Lam^T k
                w = beta_s * lam_tk
                tl.store(gv_base + t64 * head_v + offs_v64, w, mask=mask_v)
                tl.store(gbeta_base + t64, tl.sum(lam_tk * res))
                lam_res = tl.sum(lam * res[None, :], axis=1)  # Lam @ res
                sa_w = tl.sum(sa * w[None, :], axis=1)        # Sa @ w
                z_w = tl.sum(z * w)
                w_ev = tl.sum(w * ev)
                sc_w = (sa_w - gamma_s * ek * z_w) - gamma_s * r * w_ev
                dk = beta_s * lam_res - sc_w                  # Sc @ w removed
                tl.store(gk_base + t64 * head_k + offs_k64, dk, mask=mask_k)

                # --- Right erase adjoints (dSc = Lam - k w^T) ---
                lam_ev = tl.sum(lam * ev[None, :], axis=1)
                ce = lam_ev - kt * w_ev                       # dSc @ e_v
                lam_t_ek = tl.sum(lam * ek[:, None], axis=0)
                ce_ek = tl.sum(ce * ek)
                be = (lam_t_ek - w * ek_k) - gamma_s * ev * ce_ek  # Lam_b^T e_k
                sa_t_ce = tl.sum(sa * ce[:, None], axis=0)
                lam_t_r = tl.sum(lam * r[:, None], axis=0)
                de_right = -gamma_s * ((sa_t_ce - gamma_s * z * ce_ek)
                                       + (lam_t_r - w * r_k))
                dgamma_r = -tl.sum(r * ce)

                # --- Left erase / decay adjoints ---
                lam_z = tl.sum(lam * z[None, :], axis=1)
                lam_b_z = (lam_z - kt * z_w) - gamma_s * ce * z_ev
                sa_be = tl.sum(sa * be[None, :], axis=1)      # Sa @ be
                de_left = -gamma_s * (lam_b_z + sa_be)
                tl.store(ggamma_base + t64, -tl.sum(be * z) + dgamma_r)
                if SAME_DIM:
                    tl.store(ge_base + t64 * head_k + offs_k64,
                             de_left + de_right, mask=mask_k)
                else:
                    tl.store(ge_base + t64 * head_k + offs_k64,
                             de_left, mask=mask_k)
                lam_sa = tl.sum(lam * sa, axis=1)
                dg = ((lam_sa - kt * sa_w) - gamma_s * ce * sa_ev) \
                    - gamma_s * ek * sa_be
                tl.store(gg_base + t64 * head_k + offs_k64, dg, mask=mask_k)

                # --- Propagate adjoint: Lam <- diag(exp(g)) Lam_a ---
                lam = lam - kt[:, None] * w[None, :]
                lam = lam - (gamma_s * ce)[:, None] * ev[None, :]
                lam = lam - (gamma_s * ek)[:, None] * be[None, :]
                lam = lam * exp_g[:, None]

        chk_cur -= state_elems


_CHUNK = 32


class _LayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        scale_is_tensor = isinstance(scale, torch.Tensor)
        scale_f = float(scale)
        num_chunks = (T + _CHUNK - 1) // _CHUNK if T > 0 else 0
        chk = None
        if B > 0 and H > 0 and T > 0 and V > 0 and K > 0:
            chk = torch.empty((B, H, num_chunks, K, V),
                              device=q.device, dtype=torch.float32)
            _run_forward(q, k, v, e, g, beta, gamma, out, scale_f, chk, num_chunks)
        ctx.save_for_backward(q, k, v, e, g, beta, gamma)
        ctx.scale_f = scale_f
        ctx.scale_is_tensor = scale_is_tensor
        ctx.scale_requires_grad = bool(scale.requires_grad) if scale_is_tensor else False
        ctx.scale_dtype = scale.dtype if scale_is_tensor else None
        ctx.out_ref = out
        ctx.chk = chk
        ctx.num_chunks = num_chunks
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]

        gq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        gk = torch.empty(k.shape, device=k.device, dtype=k.dtype)
        gv = torch.empty(v.shape, device=v.device, dtype=v.dtype)
        ge = torch.empty(e.shape, device=e.device, dtype=e.dtype)
        gg = torch.empty(g.shape, device=g.device, dtype=g.dtype)
        gbeta = torch.empty(beta.shape, device=beta.device, dtype=beta.dtype)
        ggamma = torch.empty(gamma.shape, device=gamma.device, dtype=gamma.dtype)

        if B > 0 and H > 0 and T > 0 and V > 0 and K > 0:
            num_chunks = ctx.num_chunks
            snap = torch.empty((B, H, _CHUNK, K, V),
                               device=q.device, dtype=torch.float32)
            bk = max(16, triton.next_power_of_2(K))
            bv = max(16, triton.next_power_of_2(V))
            num_warps = 16 if bk * bv >= 16384 else 8
            _backward_kernel[(B * H,)](
                q, k, v, e, g, beta, gamma, grad_out, ctx.chk, snap,
                gq, gk, gv, ge, gg, gbeta, ggamma,
                q_sb=q.stride(0), q_sh=q.stride(1), q_st=q.stride(2), q_sk=q.stride(3),
                k_sb=k.stride(0), k_sh=k.stride(1), k_st=k.stride(2), k_sk=k.stride(3),
                v_sb=v.stride(0), v_sh=v.stride(1), v_st=v.stride(2), v_sv=v.stride(3),
                e_sb=e.stride(0), e_sh=e.stride(1), e_st=e.stride(2), e_sk=e.stride(3),
                g_sb=g.stride(0), g_sh=g.stride(1), g_st=g.stride(2), g_sk=g.stride(3),
                beta_sb=beta.stride(0), beta_sh=beta.stride(1), beta_st=beta.stride(2),
                gamma_sb=gamma.stride(0), gamma_sh=gamma.stride(1), gamma_st=gamma.stride(2),
                go_sb=grad_out.stride(0), go_sh=grad_out.stride(1),
                go_st=grad_out.stride(2), go_sv=grad_out.stride(3),
                num_heads=H, seq_len=T, head_k=K, head_v=V,
                num_chunks=num_chunks, state_elems=K * V,
                chk_bh_stride=num_chunks * K * V, snap_bh_stride=_CHUNK * K * V,
                scale_f=ctx.scale_f,
                CHUNK=_CHUNK, BK=bk, BV=bv, SAME_DIM=(K == V),
                num_warps=num_warps, num_stages=1,
            )

        grad_scale = None
        if ctx.scale_is_tensor and ctx.scale_requires_grad:
            grad_scale = torch.sum(
                grad_out.float() * ctx.out_ref.float()
            ).to(ctx.scale_dtype)
        return gq, gk, gv, ge, gg, gbeta, ggamma, grad_scale


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _LayerFn.apply(q, k, v, e, g, beta, gamma, scale)
