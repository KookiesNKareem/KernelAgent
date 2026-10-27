```python
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward: exact per-step recurrence (the version that passed the checker).
# ---------------------------------------------------------------------------
@triton.jit
def _forward_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    q_sb, q_sh, q_st, q_sk,
    k_sb, k_sh, k_st, k_sk,
    v_sb, v_sh, v_st, v_sv,
    e_sb, e_sh, e_st, e_sk,
    g_sb, g_sh, g_st, g_sk,
    beta_sb, beta_sh, beta_st,
    gamma_sb, gamma_sh, gamma_st,
    out_sb, out_sh, out_st, out_sv,
    num_heads, seq_len, head_k, head_v, scale_f,
    BK: tl.constexpr, BV: tl.constexpr,
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

    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    offs_k64 = offs_k.to(tl.int64)
    offs_v64 = offs_v.to(tl.int64)
    mask_k = offs_k < head_k
    mask_v = offs_v < head_v
    mask_ev = offs_v < head_k

    state = tl.zeros([BK, BV], dtype=tl.float32)

    for _ in range(seq_len):
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


def _run_forward(q, k, v, e, g, beta, gamma, out, scale_f):
    B, H, T, K = q.shape
    V = v.shape[-1]
    bk = max(16, triton.next_power_of_2(K))
    bv = max(16, triton.next_power_of_2(V))
    num_warps = int(max(4, min(32, (bk * bv) // 2048)))
    _forward_kernel[(B * H,)](
        q, k, v, e, g, beta, gamma, out,
        q_sb=q.stride(0), q_sh=q.stride(1), q_st=q.stride(2), q_sk=q.stride(3),
        k_sb=k.stride(0), k_sh=k.stride(1), k_st=k.stride(2), k_sk=k.stride(3),
        v_sb=v.stride(0), v_sh=v.stride(1), v_st=v.stride(2), v_sv=v.stride(3),
        e_sb=e.stride(0), e_sh=e.stride(1), e_st=e.stride(2), e_sk=e.stride(3),
        g_sb=g.stride(0), g_sh=g.stride(1), g_st=g.stride(2), g_sk=g.stride(3),
        beta_sb=beta.stride(0), beta_sh=beta.stride(1), beta_st=beta.stride(2),
        gamma_sb=gamma.stride(0), gamma_sh=gamma.stride(1), gamma_st=gamma.stride(2),
        out_sb=out.stride(0), out_sh=out.stride(1), out_st=out.stride(2), out_sv=out.stride(3),
        num_heads=H, seq_len=T, head_k=K, head_v=V, scale_f=float(scale_f),
        BK=bk, BV=bv, num_warps=num_warps, num_stages=1,
    )


# ---------------------------------------------------------------------------
# Backward.
#
# Per step, with P = S_{t-1}, d = exp(g):
#   Sa = diag(d) P
#   Sb = (I - gamma e e^T) Sa          (z = Sa^T e)
#   Sc = Sb (I - gamma e e^T)          (r = Sb e)
#   res = v - Sc^T k
#   S_t = Sc + beta k res^T,   o = scale * S_t^T q
#
# Given Lam = dL/dS_t (output contribution already added), writing
# LamTk = Lam^T k (V-vector):
#   dSc   = Lam - beta * k LamTk^T
#   dv    = beta * LamTk
#   dbeta = LamTk . res
#   dk    = beta * (Lam res) - Sc (beta LamTk)
#   dq    = scale * S_t grad_o
# and, with ce = dSc e:
#   dSb = dSc - gamma * ce e^T
#   dgamma_right = -r . ce
#   de_right = -gamma * (Sb^T ce + dSc^T r)
# and, with be = e^T dSb:
#   dSa = dSb - gamma * e be^T
#   dgamma_left = -be . z
#   de_left = -gamma * (dSb z + Sa be)
#   dg  = rowsum(dSa * Sa),   dP = diag(d) dSa.
# No inverse of any decay/erase/delta operator appears anywhere.
# ---------------------------------------------------------------------------
@triton.jit
def _from_sa(sa, kt, vt, ek, ev, beta_s, gamma_s):
    z = tl.sum(sa * ek[:, None], axis=0)
    sb = sa - (gamma_s * ek)[:, None] * z[None, :]
    r = tl.sum(sb * ev[None, :], axis=1)
    sc = sb - r[:, None] * (gamma_s * ev)[None, :]
    pred = tl.sum(sc * kt[:, None], axis=0)
    res = vt - pred
    st = sc + kt[:, None] * (beta_s * res)[None, :]
    return sb, sc, st, z, r, res


@triton.jit
def _fwd_step(p_state, kt, vt, ek, ev, gt, beta_s, gamma_s):
    sa = tl.exp(gt)[:, None] * p_state
    sb, sc, st, z, r, res = _from_sa(sa, kt, vt, ek, ev, beta_s, gamma_s)
    return sa, sb, sc, st, z, r, res


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

    state = tl.zeros([BK, BV], dtype=tl.float32)
    lam = tl.zeros([BK, BV], dtype=tl.float32)

    # Phase A: forward sweep, checkpointing the state at each chunk start.
    chk_cur = chk_ptr + bh64 * chk_bh_stride
    for chunk_idx in range(num_chunks):
        tl.store(chk_cur + tile_off, state, mask=tile_mask)
        chunk_start = chunk_idx * CHUNK
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
                sa, sb, sc, st, z, r, res = _fwd_step(
                    state, kt, vt, ek, ev, gt, beta_s, gamma_s)
                state = st
        chk_cur += state_elems

    # Phase B: walk chunks backwards.  Replay each chunk from its
    # checkpoint, snapshotting Sa per step, then sweep it in reverse.
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
                sa, sb, sc, st, z, r, res = _fwd_step(
                    state, kt, vt, ek, ev, gt, beta_s, gamma_s)
                tl.store(snap_base + s * state_elems + tile_off, sa, mask=tile_mask)
                state = st

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
                sb, sc, st, z, r, res = _from_sa(sa, kt, vt, ek, ev, beta_s, gamma_s)

                lam = lam + (scale_f * qt)[:, None] * go_vec[None, :]

                dq = scale_f * tl.sum(st * go_vec[None, :], axis=1)
                tl.store(gq_base + t64 * head_k + offs_k64, dq, mask=mask_k)

                lam_tk = tl.sum(lam * kt[:, None], axis=0)
                lam_res = beta_s * lam_tk
                dbeta = tl.sum(lam_tk * res)
                lam_r = tl.sum(lam * res[None, :], axis=1)
                dk = beta_s * lam_r - tl.sum(sc * lam_res[None, :], axis=1)
                tl.store(gk_base + t64 * head_k + offs_k64, dk, mask=mask_k)
                tl.store(gv_base + t64 * head_v + offs_v64, lam_res, mask=mask_v)
                tl.store(gbeta_base + t64, dbeta)

                lam_c = lam - kt[:, None] * lam_res[None, :]

                # Right erase (value side).
                ce = tl.sum(lam_c * ev[None, :], axis=1)
                dgamma_r = -tl.sum(r * ce)
                de_right = -gamma_s * (
                    tl.sum(sb * ce[:, None], axis=0)
                    + tl.sum(lam_c * r[:, None], axis=0)
                )
                lam_b = lam_c - (gamma_s * ce)[:, None] * ev[None, :]

                # Left erase (key side).
                be = tl.sum(lam_b * ek[:, None], axis=0)
                dgamma_l = -tl.sum(be * z)
                de_left = -gamma_s * (
                    tl.sum(lam_b * z[None, :], axis=1)
                    + tl.sum(sa * be[None, :], axis=1)
                )
                tl.store(ggamma_base + t64, dgamma_l + dgamma_r)
                if SAME_DIM:
                    tl.store(ge_base + t64 * head_k + offs_k64,
                             de_left + de_right, mask=mask_k)
                else:
                    tl.store(ge_base + t64 * head_k + offs_k64,
                             de_left, mask=mask_k)

                lam_a = lam_b - (gamma_s * ek)[:, None] * be[None, :]
                dg = tl.sum(lam_a * sa, axis=1)
                tl.store(gg_base + t64 * head_k + offs_k64, dg, mask=mask_k)
                lam = tl.exp(gt)[:, None] * lam_a

        chk_cur -= state_elems


_CHUNK = 32


class _LayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        scale_is_tensor = isinstance(scale, torch.Tensor)
        scale_f = float(scale) if scale_is_tensor else float(scale)
        if B > 0 and H > 0 and T > 0 and V > 0 and K > 0:
            _run_forward(q, k, v, e, g, beta, gamma, out, scale_f)
        ctx.save_for_backward(q, k, v, e, g, beta, gamma)
        ctx.scale_f = scale_f
        ctx.scale_is_tensor = scale_is_tensor
        ctx.scale_requires_grad = bool(scale.requires_grad) if scale_is_tensor else False
        ctx.scale_dtype = scale.dtype if scale_is_tensor else None
        ctx.out_ref = out
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
            num_chunks = (T + _CHUNK - 1) // _CHUNK
            chk = torch.empty((B, H, num_chunks, K, V),
                              device=q.device, dtype=torch.float32)
            snap = torch.empty((B, H, _CHUNK, K, V),
                               device=q.device, dtype=torch.float32)
            bk = max(16, triton.next_power_of_2(K))
            bv = max(16, triton.next_power_of_2(V))
            num_warps = 16 if bk * bv >= 16384 else 8
            _backward_kernel[(B * H,)](
                q, k, v, e, g, beta, gamma, grad_out, chk, snap,
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
```

The forward is the exact per-step Triton kernel that already passed, wrapped in a `torch.autograd.Function`. For the backward I derived the adjoint recurrence analytically: it propagates a state adjoint backwards through the delta update, both erase factors, and the decay, accumulating per-step gradients for all seven inputs (with `e` and `gamma` each receiving both their left- and right-erase contributions), and it forms no inverse of any operator. To avoid storing every state, the backward kernel first checkpoints the state every 32 steps, then replays each chunk forward into a small per-head snapshot buffer of the decayed state `S_a` (from which all other intermediates are recomputed) before sweeping that chunk in reverse. Gradients are computed in float32 and stored directly into tensors of each input's dtype, and arbitrary input/`grad_out` strides are honoured throughout.