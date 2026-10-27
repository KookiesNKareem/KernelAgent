import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Chunkwise kernels.
#
# Per chunk, with S0 the incoming state, cum_t = sum_{s<=t} g_s,
# P_t = exp(cum_t), u_t the delta-rule update direction:
#   S~_t = diag(P_t) S0 + sum_{s<t} diag(P_t/P_s) k_s u_s^T
#   A[t,s] = k_t^T diag(P_t/P_s) k_s ,  B[t,s] = q_t^T diag(P_t/P_s) k_s
#   U = (I + diag(beta) A_strict)^-1 diag(beta) (V - (P*k) S0)
#   O = scale * ( (P*q) S0 + B_strict U )
#   S_end = diag(P_last) S0 + K_end^T U,  K_end[s] = exp(cum_last-cum_s) k_s
# Pairwise decay factors use a per-key mid-chunk reference.  This path is
# only used when |g| is small enough that a chunk's total decay range is
# bounded (checked in the wrapper); otherwise the sequential kernels below
# are used, so exp() factors can never overflow on strong decay.
#
# The V dimension is tiled (BLOCK_V <= 32) so no [K, V]=[128, 128] dot
# operands are ever staged: this is what keeps shared memory far below
# the 101376-byte L4 limit.  num_stages=1 everywhere on this path.
# ---------------------------------------------------------------------------

@triton.jit
def _chunk_fwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr, chk_ptr,
    T, K, V, BH,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    CHUNK: tl.constexpr, NCHUNK: tl.constexpr, NEWTON: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    rows = tl.arange(0, CHUNK)
    kidx = tl.arange(0, BLOCK_K)
    vidx = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = kidx < K
    v_mask = vidx < V
    state_mask = k_mask[:, None] & v_mask[None, :]

    ident = (rows[:, None] == rows[None, :]).to(tl.float32)
    strict = rows[None, :] < rows[:, None]

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    g_base = g_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    o_base = out_ptr + bh * T * V
    b_base = beta_ptr + bh * T

    for chunk in range(NCHUNK):
        chk_off = (chunk * BH + bh) * K * V + kidx[:, None] * V + vidx[None, :]
        tl.store(chk_ptr + chk_off, S, mask=state_mask)

        c0 = chunk * CHUNK
        tids = c0 + rows
        t_mask = tids < T
        kmask2 = t_mask[:, None] & k_mask[None, :]
        vmask2 = t_mask[:, None] & v_mask[None, :]

        q_mat = tl.load(q_base + tids[:, None] * K + kidx[None, :],
                        mask=kmask2, other=0.0).to(tl.float32)
        k_mat = tl.load(k_base + tids[:, None] * K + kidx[None, :],
                        mask=kmask2, other=0.0).to(tl.float32)
        g_mat = tl.load(g_base + tids[:, None] * K + kidx[None, :],
                        mask=kmask2, other=0.0).to(tl.float32)
        v_mat = tl.load(v_base + tids[:, None] * V + vidx[None, :],
                        mask=vmask2, other=0.0).to(tl.float32)
        beta_vec = tl.load(b_base + tids, mask=t_mask, other=0.0).to(tl.float32)

        cum = tl.cumsum(g_mat, axis=0)
        ref = tl.sum(tl.where(rows[:, None] == (CHUNK // 2), cum, 0.0), axis=0)
        exp_up = tl.exp(cum - ref[None, :])
        exp_dn = tl.exp(ref[None, :] - cum)
        p_exp = tl.exp(cum)
        ka = k_mat * exp_up
        kb = k_mat * exp_dn
        qa = q_mat * exp_up
        qp = q_mat * p_exp
        kp = k_mat * p_exp

        a_mat = tl.dot(ka, tl.trans(kb), input_precision="ieee")
        b_mat = tl.dot(qa, tl.trans(kb), input_precision="ieee")
        a_strict = tl.where(strict, a_mat, 0.0)
        b_strict = tl.where(strict, b_mat, 0.0)

        l_mat = ident + beta_vec[:, None] * a_strict
        inv_l = ident
        for _ in range(NEWTON):
            tmp = tl.dot(l_mat, inv_l, input_precision="ieee")
            inv_l = tl.dot(inv_l, 2.0 * ident - tmp, input_precision="ieee")

        pred = tl.dot(kp, S, input_precision="ieee")
        y_mat = v_mat - pred
        u_mat = tl.dot(inv_l, beta_vec[:, None] * y_mat, input_precision="ieee")

        out_mat = scale * (tl.dot(qp, S, input_precision="ieee")
                           + tl.dot(b_strict, u_mat, input_precision="ieee"))
        tl.store(o_base + tids[:, None] * V + vidx[None, :], out_mat, mask=vmask2)

        cum_last = tl.sum(tl.where(rows[:, None] == (CHUNK - 1), cum, 0.0), axis=0)
        kend = k_mat * tl.exp(cum_last[None, :] - cum)
        S = S * tl.exp(cum_last)[:, None] \
            + tl.dot(tl.trans(kend), u_mat, input_precision="ieee")


@triton.jit
def _chunk_bwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, do_ptr, chk_ptr,
    dq_ptr, dk_ptr, dv_ptr, dg_ptr, dbeta_ptr,
    T, K, V, BH,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    CHUNK: tl.constexpr, NCHUNK: tl.constexpr, NEWTON: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    rows = tl.arange(0, CHUNK)
    kidx = tl.arange(0, BLOCK_K)
    vidx = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = kidx < K
    v_mask = vidx < V
    state_mask = k_mask[:, None] & v_mask[None, :]

    ident = (rows[:, None] == rows[None, :]).to(tl.float32)
    strict = rows[None, :] < rows[:, None]

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    g_base = g_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    do_base = do_ptr + bh * T * V
    b_base = beta_ptr + bh * T
    # dq/dk/dg/dbeta are per-V-tile partial sums; the wrapper reduces them.
    dq_base = dq_ptr + (pid_v * BH + bh) * T * K
    dk_base = dk_ptr + (pid_v * BH + bh) * T * K
    dv_base = dv_ptr + bh * T * V
    dg_base = dg_ptr + (pid_v * BH + bh) * T * K
    db_base = dbeta_ptr + (pid_v * BH + bh) * T

    g_end = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for rev in range(NCHUNK):
        chunk = NCHUNK - 1 - rev
        chk_off = (chunk * BH + bh) * K * V + kidx[:, None] * V + vidx[None, :]
        s0 = tl.load(chk_ptr + chk_off, mask=state_mask, other=0.0).to(tl.float32)

        c0 = chunk * CHUNK
        tids = c0 + rows
        t_mask = tids < T
        kmask2 = t_mask[:, None] & k_mask[None, :]
        vmask2 = t_mask[:, None] & v_mask[None, :]

        q_mat = tl.load(q_base + tids[:, None] * K + kidx[None, :],
                        mask=kmask2, other=0.0).to(tl.float32)
        k_mat = tl.load(k_base + tids[:, None] * K + kidx[None, :],
                        mask=kmask2, other=0.0).to(tl.float32)
        g_mat = tl.load(g_base + tids[:, None] * K + kidx[None, :],
                        mask=kmask2, other=0.0).to(tl.float32)
        v_mat = tl.load(v_base + tids[:, None] * V + vidx[None, :],
                        mask=vmask2, other=0.0).to(tl.float32)
        do_mat = tl.load(do_base + tids[:, None] * V + vidx[None, :],
                         mask=vmask2, other=0.0).to(tl.float32)
        beta_vec = tl.load(b_base + tids, mask=t_mask, other=0.0).to(tl.float32)

        # ---- recompute forward chunk quantities ----
        cum = tl.cumsum(g_mat, axis=0)
        ref = tl.sum(tl.where(rows[:, None] == (CHUNK // 2), cum, 0.0), axis=0)
        exp_up = tl.exp(cum - ref[None, :])
        exp_dn = tl.exp(ref[None, :] - cum)
        p_exp = tl.exp(cum)
        ka = k_mat * exp_up
        kb = k_mat * exp_dn
        qa = q_mat * exp_up
        qp = q_mat * p_exp
        kp = k_mat * p_exp

        a_mat = tl.dot(ka, tl.trans(kb), input_precision="ieee")
        b_mat = tl.dot(qa, tl.trans(kb), input_precision="ieee")
        a_strict = tl.where(strict, a_mat, 0.0)
        b_strict = tl.where(strict, b_mat, 0.0)

        l_mat = ident + beta_vec[:, None] * a_strict
        inv_l = ident
        for _ in range(NEWTON):
            tmp = tl.dot(l_mat, inv_l, input_precision="ieee")
            inv_l = tl.dot(inv_l, 2.0 * ident - tmp, input_precision="ieee")

        pred = tl.dot(kp, s0, input_precision="ieee")
        y_mat = v_mat - pred
        u_mat = tl.dot(inv_l, beta_vec[:, None] * y_mat, input_precision="ieee")

        cum_last = tl.sum(tl.where(rows[:, None] == (CHUNK - 1), cum, 0.0), axis=0)
        plast = tl.exp(cum_last)
        exp_end = tl.exp(cum_last[None, :] - cum)
        kend = k_mat * exp_end

        # ---- backward through the chunk (this V tile) ----
        gu = scale * tl.dot(tl.trans(b_strict), do_mat, input_precision="ieee") \
            + tl.dot(kend, g_end, input_precision="ieee")
        g_rhs = tl.dot(tl.trans(inv_l), gu, input_precision="ieee")
        gy = beta_vec[:, None] * g_rhs

        ds0 = tl.dot(tl.trans(kp), -gy, input_precision="ieee") \
            + scale * tl.dot(tl.trans(qp), do_mat, input_precision="ieee") \
            + plast[:, None] * g_end

        dkp = tl.dot(-gy, tl.trans(s0), input_precision="ieee")
        dqp = scale * tl.dot(do_mat, tl.trans(s0), input_precision="ieee")

        gl_mat = -tl.dot(g_rhs, tl.trans(u_mat), input_precision="ieee")
        da = tl.where(strict, beta_vec[:, None] * gl_mat, 0.0)
        db_mat = tl.where(
            strict,
            scale * tl.dot(do_mat, tl.trans(u_mat), input_precision="ieee"),
            0.0)

        dka = tl.dot(da, kb, input_precision="ieee")
        dkb = tl.dot(tl.trans(da), ka, input_precision="ieee") \
            + tl.dot(tl.trans(db_mat), qa, input_precision="ieee")
        dqa = tl.dot(db_mat, kb, input_precision="ieee")

        dq_mat = dqa * exp_up + dqp * p_exp
        dk_mat = dka * exp_up + dkb * exp_dn + dkp * p_exp
        dcum = dqa * qa + dqp * qp + dka * ka - dkb * kb

        dkend = tl.dot(u_mat, tl.trans(g_end), input_precision="ieee")
        dk_mat = dk_mat + dkend * exp_end
        dcum = dcum - dkend * kend
        dplast = tl.sum(g_end * s0, axis=1)
        last_add = tl.sum(dkend * kend, axis=0) + dplast * plast
        dcum = tl.where(rows[:, None] == (CHUNK - 1),
                        dcum + last_add[None, :], dcum)
        dg_mat = tl.cumsum(dcum, axis=0, reverse=True)

        dbeta_vec = tl.sum(g_rhs * y_mat, axis=1) \
            + tl.sum(gl_mat * a_strict, axis=1)

        tl.store(dq_base + tids[:, None] * K + kidx[None, :], dq_mat, mask=kmask2)
        tl.store(dk_base + tids[:, None] * K + kidx[None, :], dk_mat, mask=kmask2)
        tl.store(dg_base + tids[:, None] * K + kidx[None, :], dg_mat, mask=kmask2)
        tl.store(dv_base + tids[:, None] * V + vidx[None, :], gy, mask=vmask2)
        tl.store(db_base + tids, dbeta_vec, mask=t_mask)

        g_end = ds0


# ---------------------------------------------------------------------------
# Sequential kernels: used whenever the chunkwise path is not applicable
# (large head dims, or decay so strong / so large that separated chunkwise
# exponentials could overflow).  These apply exp(g_t) one step at a time,
# exactly like the reference, so they stay finite wherever it does.
# ---------------------------------------------------------------------------

@triton.jit
def _seq_fwd_kernel(
    q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, out_ptr, chk_ptr,
    T, K, V, BH,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, CHUNK: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V
    state_mask = k_mask[:, None] & v_mask[None, :]

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    d_base = d_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    o_base = out_ptr + bh * T * V
    b_base = beta_ptr + bh * T

    for t in range(T):
        if t % CHUNK == 0:
            chunk = t // CHUNK
            chk_off = (chunk * BH + bh) * K * V + k_offs[:, None] * V + v_offs[None, :]
            tl.store(chk_ptr + chk_off, S, mask=state_mask)
        q_t = tl.load(q_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        k_t = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        dec = tl.load(d_base + t * K + k_offs, mask=k_mask, other=1.0).to(tl.float32)
        v_t = tl.load(v_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
        b_t = tl.load(b_base + t).to(tl.float32)
        S = S * dec[:, None]
        out_t = tl.sum(S * (q_t * scale)[:, None], axis=0)
        tl.store(o_base + t * V + v_offs, out_t, mask=v_mask)
        pred = tl.sum(S * k_t[:, None], axis=0)
        r = v_t - pred
        S = S + k_t[:, None] * (b_t * r)[None, :]


@triton.jit
def _seq_bwd_kernel(
    q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, do_ptr,
    chk_ptr, scr_ptr,
    dq_ptr, dk_ptr, dv_ptr, dg_ptr, dbeta_ptr,
    T, K, V, BH,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    CHUNK: tl.constexpr, NCHUNK: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V
    state_mask = k_mask[:, None] & v_mask[None, :]

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    d_base = d_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    do_base = do_ptr + bh * T * V
    b_base = beta_ptr + bh * T
    dq_base = dq_ptr + bh * T * K
    dk_base = dk_ptr + bh * T * K
    dv_base = dv_ptr + bh * T * V
    dg_base = dg_ptr + bh * T * K
    db_base = dbeta_ptr + bh * T

    G = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for rev_chunk in range(NCHUNK):
        chunk = NCHUNK - 1 - rev_chunk
        c0 = chunk * CHUNK
        c_end = c0 + CHUNK
        if c_end > T:
            c_end = T
        chunk_len = c_end - c0

        chk_off = (chunk * BH + bh) * K * V + k_offs[:, None] * V + v_offs[None, :]
        S = tl.load(chk_ptr + chk_off, mask=state_mask, other=0.0).to(tl.float32)
        for local in range(chunk_len):
            t = c0 + local
            k_t = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            dec = tl.load(d_base + t * K + k_offs, mask=k_mask, other=1.0).to(tl.float32)
            v_t = tl.load(v_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
            b_t = tl.load(b_base + t).to(tl.float32)
            s_tilde = S * dec[:, None]
            scr_off = (bh * CHUNK + local) * K * V + k_offs[:, None] * V + v_offs[None, :]
            tl.store(scr_ptr + scr_off, s_tilde, mask=state_mask)
            pred = tl.sum(s_tilde * k_t[:, None], axis=0)
            r = v_t - pred
            S = s_tilde + k_t[:, None] * (b_t * r)[None, :]

        for rev in range(chunk_len):
            t = c_end - 1 - rev
            local = t - c0
            q_t = tl.load(q_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            k_t = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            dec = tl.load(d_base + t * K + k_offs, mask=k_mask, other=1.0).to(tl.float32)
            v_t = tl.load(v_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
            b_t = tl.load(b_base + t).to(tl.float32)
            do_t = tl.load(do_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
            scr_off = (bh * CHUNK + local) * K * V + k_offs[:, None] * V + v_offs[None, :]
            s_tilde = tl.load(scr_ptr + scr_off, mask=state_mask, other=0.0).to(tl.float32)

            pred = tl.sum(s_tilde * k_t[:, None], axis=0)
            r = v_t - pred
            wg = tl.sum(G * k_t[:, None], axis=0)
            g_tilde = G - (b_t * k_t)[:, None] * wg[None, :] \
                + (scale * q_t)[:, None] * do_t[None, :]

            dq_part = scale * tl.sum(s_tilde * do_t[None, :], axis=1)
            dk_part = b_t * (tl.sum(G * r[None, :], axis=1)
                             - tl.sum(s_tilde * wg[None, :], axis=1))
            dg_part = tl.sum(g_tilde * s_tilde, axis=1)
            tl.atomic_add(dq_base + t * K + k_offs, dq_part, mask=k_mask)
            tl.atomic_add(dk_base + t * K + k_offs, dk_part, mask=k_mask)
            tl.atomic_add(dg_base + t * K + k_offs, dg_part, mask=k_mask)
            tl.store(dv_base + t * V + v_offs, b_t * wg, mask=v_mask)
            tl.atomic_add(db_base + t, tl.sum(wg * r))

            G = g_tilde * dec[:, None]


def _next_pow2(x):
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


def _block_sizes(K, V):
    block_k = _next_pow2(K)
    if block_k < 16:
        block_k = 16
    if V <= 32:
        block_v = _next_pow2(V)
        if block_v < 16:
            block_v = 16
    else:
        block_v = 32
    return block_k, block_v


def _seq_chunk_len(BH, K, V):
    per_state = max(1, BH * K * V)
    c = (1 << 24) // per_state
    if c > 64:
        c = 64
    if c < 1:
        c = 1
    return int(c)


class _LayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
        ctx.scale_f = float(scale)
        ctx.g_dtype = g.dtype
        device = q.device

        ctx.empty_case = (BH == 0 or T == 0 or K == 0 or V == 0)

        qc = q.contiguous()
        kc = k.contiguous()
        vc = v.contiguous()
        bc = beta.contiguous()
        gc = g.contiguous()

        out3 = torch.empty(max(BH, 0), max(T, 0), max(V, 0),
                           device=device, dtype=torch.float32)

        if ctx.empty_case:
            ctx.fast = False
            ctx.save_for_backward(q, k, v, e, g, beta, gamma)
            return out3.reshape(B, H, T, V)

        qc3 = qc.reshape(BH, T, K)
        kc3 = kc.reshape(BH, T, K)
        vc3 = vc.reshape(BH, T, V)
        bc2 = bc.reshape(BH, T)
        gc3 = gc.reshape(BH, T, K)
        qc3 = qc3 if qc3.is_contiguous() else qc3.contiguous()
        kc3 = kc3 if kc3.is_contiguous() else kc3.contiguous()
        vc3 = vc3 if vc3.is_contiguous() else vc3.contiguous()
        bc2 = bc2 if bc2.is_contiguous() else bc2.contiguous()
        gc3 = gc3 if gc3.is_contiguous() else gc3.contiguous()

        # Chunkwise path only when the per-chunk decay range is provably
        # small: range <= (CHUNK-1) * max|g|.  Strong decay, huge scaled
        # decays, or positive-overflow decays fall back to the sequential
        # kernels, which apply exp(g_t) stepwise exactly like the reference.
        ctx.fast = False
        ctx.chunk_size = 32
        ctx.newton = 5
        if device.type == "cuda" and K <= 128 and V <= 128:
            gmax = float(gc3.abs().max().item())
            if gmax <= 1.25:
                ctx.fast = True
                ctx.chunk_size = 32
                ctx.newton = 5
            elif gmax <= 2.5:
                ctx.fast = True
                ctx.chunk_size = 16
                ctx.newton = 4

        if ctx.fast:
            block_k = max(16, _next_pow2(K))
            block_v = 32 if V > 32 else max(16, _next_pow2(V))
            ctx.block_k, ctx.block_v = block_k, block_v
            ctx.n_v = (V + block_v - 1) // block_v
            chunk = ctx.chunk_size
            n_chunks = (T + chunk - 1) // chunk
            ctx.n_chunks = n_chunks
            chk = torch.empty(n_chunks * BH * K * V, device=device,
                              dtype=torch.float32)
            _chunk_fwd_kernel[(BH, ctx.n_v)](
                qc3, kc3, vc3, gc3, bc2, out3, chk,
                T, K, V, BH, ctx.scale_f,
                BLOCK_K=block_k, BLOCK_V=block_v,
                CHUNK=chunk, NCHUNK=n_chunks, NEWTON=ctx.newton,
                num_warps=4, num_stages=1,
            )
            ctx.save_for_backward(qc3, kc3, vc3, gc3, bc2, chk, e, gamma)
            return out3.reshape(B, H, T, V)

        # Sequential fallback path.
        dc = torch.exp(gc)
        dc3 = dc.reshape(BH, T, K)
        dc3 = dc3 if dc3.is_contiguous() else dc3.contiguous()
        chunk = _seq_chunk_len(BH, K, V)
        n_chunks = (T + chunk - 1) // chunk
        ctx.chunk = chunk
        ctx.n_chunks = n_chunks
        chk = torch.empty(n_chunks * BH * K * V, device=device, dtype=torch.float32)
        block_k, block_v = _block_sizes(K, V)
        num_warps = 8 if block_k * block_v > 2048 else 4
        grid = (BH, (V + block_v - 1) // block_v)
        _seq_fwd_kernel[grid](
            qc3, kc3, vc3, dc3, bc2, out3, chk,
            T, K, V, BH, ctx.scale_f,
            BLOCK_K=block_k, BLOCK_V=block_v, CHUNK=chunk,
            num_warps=num_warps, num_stages=1,
        )
        ctx.save_for_backward(qc3, kc3, vc3, dc3, bc2, chk, e, gamma)
        return out3.reshape(B, H, T, V)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.empty_case:
            q, k, v, e, g, beta, gamma = ctx.saved_tensors
            return (torch.zeros_like(q), torch.zeros_like(k),
                    torch.zeros_like(v), torch.zeros_like(e),
                    torch.zeros_like(g), torch.zeros_like(beta),
                    torch.zeros_like(gamma), None)

        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        BH = B * H

        if ctx.fast:
            qc3, kc3, vc3, gc3, bc2, chk, e, gamma = ctx.saved_tensors
            if grad_output is None:
                return (torch.zeros_like(qc3).reshape(B, H, T, K),
                        torch.zeros_like(kc3).reshape(B, H, T, K),
                        torch.zeros_like(vc3).reshape(B, H, T, V),
                        torch.zeros_like(e),
                        torch.zeros_like(gc3).reshape(B, H, T, K),
                        torch.zeros_like(bc2).reshape(B, H, T),
                        torch.zeros_like(gamma), None)
            device = qc3.device
            go3 = grad_output.reshape(BH, T, V)
            go3 = go3 if go3.is_contiguous() else go3.contiguous()
            n_v = ctx.n_v
            dq_part = torch.empty(n_v, BH, T, K, device=device, dtype=torch.float32)
            dk_part = torch.empty(n_v, BH, T, K, device=device, dtype=torch.float32)
            dg_part = torch.empty(n_v, BH, T, K, device=device, dtype=torch.float32)
            db_part = torch.empty(n_v, BH, T, device=device, dtype=torch.float32)
            dv3 = torch.empty_like(vc3)
            _chunk_bwd_kernel[(BH, n_v)](
                qc3, kc3, vc3, gc3, bc2, go3, chk,
                dq_part, dk_part, dv3, dg_part, db_part,
                T, K, V, BH, ctx.scale_f,
                BLOCK_K=ctx.block_k, BLOCK_V=ctx.block_v,
                CHUNK=ctx.chunk_size, NCHUNK=ctx.n_chunks, NEWTON=ctx.newton,
                num_warps=4, num_stages=1,
            )
            return (dq_part.sum(dim=0).to(qc3.dtype).reshape(B, H, T, K),
                    dk_part.sum(dim=0).to(kc3.dtype).reshape(B, H, T, K),
                    dv3.reshape(B, H, T, V),
                    torch.zeros_like(e),
                    dg_part.sum(dim=0).to(gc3.dtype).reshape(B, H, T, K),
                    db_part.sum(dim=0).to(bc2.dtype).reshape(B, H, T),
                    torch.zeros_like(gamma), None)

        # Sequential fallback path.
        qc3, kc3, vc3, dc3, bc2, chk, e, gamma = ctx.saved_tensors
        if grad_output is None:
            return (torch.zeros_like(qc3).reshape(B, H, T, K),
                    torch.zeros_like(kc3).reshape(B, H, T, K),
                    torch.zeros_like(vc3).reshape(B, H, T, V),
                    torch.zeros_like(e),
                    torch.zeros(B, H, T, K, device=qc3.device, dtype=ctx.g_dtype),
                    torch.zeros_like(bc2).reshape(B, H, T),
                    torch.zeros_like(gamma), None)
        device = qc3.device
        go3 = grad_output.reshape(BH, T, V).contiguous().to(torch.float32)
        dq_acc = torch.zeros(BH, T, K, device=device, dtype=torch.float32)
        dk_acc = torch.zeros(BH, T, K, device=device, dtype=torch.float32)
        dg_acc = torch.zeros(BH, T, K, device=device, dtype=torch.float32)
        dv_acc = torch.empty(BH, T, V, device=device, dtype=torch.float32)
        db_acc = torch.zeros(BH, T, device=device, dtype=torch.float32)
        scratch = torch.empty(BH * ctx.chunk * K * V, device=device,
                              dtype=torch.float32)
        block_k, block_v = _block_sizes(K, V)
        num_warps = 8 if block_k * block_v > 2048 else 4
        grid = (BH, (V + block_v - 1) // block_v)
        _seq_bwd_kernel[grid](
            qc3, kc3, vc3, dc3, bc2, go3,
            chk, scratch,
            dq_acc, dk_acc, dv_acc, dg_acc, db_acc,
            T, K, V, BH, ctx.scale_f,
            BLOCK_K=block_k, BLOCK_V=block_v,
            CHUNK=ctx.chunk, NCHUNK=ctx.n_chunks,
            num_warps=num_warps, num_stages=1,
        )
        return (dq_acc.reshape(B, H, T, K).to(qc3.dtype),
                dk_acc.reshape(B, H, T, K).to(kc3.dtype),
                dv_acc.reshape(B, H, T, V).to(vc3.dtype),
                torch.zeros_like(e),
                dg_acc.reshape(B, H, T, K).to(ctx.g_dtype),
                db_acc.reshape(B, H, T).to(bc2.dtype),
                torch.zeros_like(gamma), None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)
    return _LayerFn.apply(q, k, v, e, g, beta, gamma, scale_f)
