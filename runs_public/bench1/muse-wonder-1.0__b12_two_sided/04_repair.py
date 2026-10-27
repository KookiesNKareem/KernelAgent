import torch
import triton
import triton.language as tl


@triton.jit
def _apply_a(x, kt, et, dt, beta_s, gamma_s):
    # A_s x = (I - beta k k^T) (I - gamma e e^T) (diag(exp(g)) x), per row.
    y = x * dt[None, :]
    de = tl.sum(y * et[None, :], axis=1)
    y = y - (gamma_s * de)[:, None] * et[None, :]
    dk = tl.sum(y * kt[None, :], axis=1)
    y = y - (beta_s * dk)[:, None] * kt[None, :]
    return y


@triton.jit
def _apply_a_t(x, kt, et, dt, beta_s, gamma_s):
    # A_s^T x = diag(exp(g)) (I - gamma e e^T) (I - beta k k^T) x, per row.
    dk = tl.sum(x * kt[None, :], axis=1)
    y = x - (beta_s * dk)[:, None] * kt[None, :]
    de = tl.sum(y * et[None, :], axis=1)
    y = y - (gamma_s * de)[:, None] * et[None, :]
    return y * dt[None, :]


@triton.jit
def _chunk_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    q_sb, q_sh, q_st, q_sk,
    k_sb, k_sh, k_st, k_sk,
    v_sb, v_sh, v_st, v_sv,
    e_sb, e_sh, e_st, e_sk,
    g_sb, g_sh, g_st, g_sk,
    beta_sb, beta_sh, beta_st,
    gamma_sb, gamma_sh, gamma_st,
    out_sb, out_sh, out_st, out_sv,
    num_heads, seq_len, head_dim, scale_f,
    CHUNK: tl.constexpr, BD: tl.constexpr,
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
    offs_d = tl.arange(0, BD)
    offs_d64 = offs_d.to(tl.int64)
    mask_d = offs_d < head_dim

    state = tl.zeros([BD, BD], dtype=tl.float32)
    num_chunks = tl.cdiv(seq_len, CHUNK)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK
        t_idx = start + offs_c
        mask_c = t_idx < seq_len
        t64 = t_idx.to(tl.int64)
        cmask = mask_c[:, None] & mask_d[None, :]

        q_mat = tl.load(q_base + t64[:, None] * q_st + offs_d64[None, :] * q_sk,
                        mask=cmask, other=0.0).to(tl.float32)
        k_mat = tl.load(k_base + t64[:, None] * k_st + offs_d64[None, :] * k_sk,
                        mask=cmask, other=0.0).to(tl.float32)
        v_mat = tl.load(v_base + t64[:, None] * v_st + offs_d64[None, :] * v_sv,
                        mask=cmask, other=0.0).to(tl.float32)
        e_mat = tl.load(e_base + t64[:, None] * e_st + offs_d64[None, :] * e_sk,
                        mask=cmask, other=0.0).to(tl.float32)
        g_mat = tl.load(g_base + t64[:, None] * g_st + offs_d64[None, :] * g_sk,
                        mask=cmask, other=0.0).to(tl.float32)
        beta = tl.load(beta_base + t64 * beta_st, mask=mask_c, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_base + t64 * gamma_st, mask=mask_c, other=0.0).to(tl.float32)

        d_rows = tl.exp(g_mat)
        d_cum = tl.exp(tl.sum(g_mat, axis=0))

        # Low-rank factorisation of the chunk's left operator product
        #   P_A = A_{C-1} ... A_0 = diag(d_cum) + U_k W_k^T + U_e W_e^T
        # (grouping the expansion of each A_t by its latest rank-1 term).
        # U_k is (minus) the forward-transformed beta*k tile computed below;
        # U_e is forward-transformed here, W_k / W_e backward-transformed.
        ke_dot = tl.sum(k_mat * e_mat, axis=1)
        u_e = -gamma[:, None] * e_mat + (beta * gamma * ke_dot)[:, None] * k_mat
        w_bk = d_rows * k_mat
        w_be = d_rows * e_mat

        # WY factor for the right product P_R = R_0 ... R_{C-1},
        #   P_R = I - E^T W_R E,
        # built by applying the erase factors one at a time:
        #   W_R[:, s] += gamma_s * (unit_s - W_R @ gram[:, s]).
        # This is a forward product, not an inverse: W_R stays bounded
        # (it represents I - P_R for a contractive P_R), so no division,
        # no inverse of any erase/decay operator, and no large shared tile
        # beyond the small [CHUNK, CHUNK] Gram/W_R matrices.
        gram = tl.dot(e_mat, tl.trans(e_mat))
        w_r = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)
        for s in range(CHUNK):
            gamma_s = tl.sum(tl.where(offs_c == s, gamma, 0.0))
            g_col = tl.sum(tl.where(offs_c[:, None] == s, gram, 0.0), axis=0)
            wg = tl.sum(w_r * g_col[None, :], axis=1)
            unit = (offs_c == s).to(tl.float32)
            w_r = tl.where(
                offs_c[None, :] == s,
                w_r + (gamma_s * (unit - wg))[:, None],
                w_r,
            )

        # Backward pass: transformed queries and W factors.
        q_hat = q_mat
        for idx in range(CHUNK):
            s = CHUNK - 1 - idx
            kt = tl.sum(tl.where(offs_c[:, None] == s, k_mat, 0.0), axis=0)
            et = tl.sum(tl.where(offs_c[:, None] == s, e_mat, 0.0), axis=0)
            gt = tl.sum(tl.where(offs_c[:, None] == s, g_mat, 0.0), axis=0)
            beta_s = tl.sum(tl.where(offs_c == s, beta, 0.0))
            gamma_s = tl.sum(tl.where(offs_c == s, gamma, 0.0))
            dt = tl.exp(gt)
            new = _apply_a_t(q_hat, kt, et, dt, beta_s, gamma_s)
            q_hat = tl.where(offs_c[:, None] >= s, new, q_hat)
            new = _apply_a_t(w_bk, kt, et, dt, beta_s, gamma_s)
            w_bk = tl.where(offs_c[:, None] > s, new, w_bk)
            new = _apply_a_t(w_be, kt, et, dt, beta_s, gamma_s)
            w_be = tl.where(offs_c[:, None] > s, new, w_be)

        out_t = tl.dot(q_hat, state)

        # Forward pass: evolve intra-chunk source factors b_j, c_j under
        # A_s / R_s, finish the right erase of the state contribution row
        # by row, and emit each row's intra-chunk contribution.
        b_t = tl.zeros([CHUNK, BD], dtype=tl.float32)
        c_t = tl.zeros([CHUNK, BD], dtype=tl.float32)
        for s in range(CHUNK):
            kt = tl.sum(tl.where(offs_c[:, None] == s, k_mat, 0.0), axis=0)
            et = tl.sum(tl.where(offs_c[:, None] == s, e_mat, 0.0), axis=0)
            gt = tl.sum(tl.where(offs_c[:, None] == s, g_mat, 0.0), axis=0)
            vt = tl.sum(tl.where(offs_c[:, None] == s, v_mat, 0.0), axis=0)
            qi = tl.sum(tl.where(offs_c[:, None] == s, q_mat, 0.0), axis=0)
            beta_s = tl.sum(tl.where(offs_c == s, beta, 0.0))
            gamma_s = tl.sum(tl.where(offs_c == s, gamma, 0.0))
            dt = tl.exp(gt)

            new = _apply_a(b_t, kt, et, dt, beta_s, gamma_s)
            b_t = tl.where(offs_c[:, None] < s, new, b_t)
            new = _apply_a(u_e, kt, et, dt, beta_s, gamma_s)
            u_e = tl.where(offs_c[:, None] < s, new, u_e)
            dc = tl.sum(c_t * et[None, :], axis=1)
            new = c_t - (gamma_s * dc)[:, None] * et[None, :]
            c_t = tl.where(offs_c[:, None] < s, new, c_t)

            b_t = tl.where(offs_c[:, None] == s, (beta_s * kt)[None, :], b_t)
            c_t = tl.where(offs_c[:, None] == s, vt[None, :], c_t)

            do = tl.sum(out_t * et[None, :], axis=1)
            new = out_t - (gamma_s * do)[:, None] * et[None, :]
            out_t = tl.where(offs_c[:, None] >= s, new, out_t)

            p = tl.sum(b_t * qi[None, :], axis=1)
            oi = tl.sum(c_t * p[:, None], axis=0)
            out_t = out_t + tl.where(offs_c[:, None] == s, oi[None, :], 0.0)

        tl.store(out_base + t64[:, None] * out_st + offs_d64[None, :] * out_sv,
                 out_t * scale_f, mask=cmask)

        # State update: S <- P_A (S P_R) + B^T C, all via tensor-core dots.
        se = tl.dot(state, tl.trans(e_mat))
        sw = tl.dot(se, w_r)
        y_state = state - tl.dot(sw, e_mat)
        tk_state = tl.dot(w_bk, y_state)
        te_state = tl.dot(w_be, y_state)
        state = (d_cum[:, None] * y_state
                 - tl.dot(tl.trans(b_t), tk_state)
                 + tl.dot(tl.trans(u_e), te_state)
                 + tl.dot(tl.trans(b_t), c_t))


@triton.jit
def _direct_kernel(
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


def _launch_direct(q, k, v, e, g, beta, gamma, out, B, H, T, K, V, scale):
    bk = max(16, triton.next_power_of_2(K))
    bv = max(16, triton.next_power_of_2(V))
    num_warps = int(max(4, min(32, (bk * bv) // 2048)))
    _direct_kernel[(B * H,)](
        q, k, v, e, g, beta, gamma, out,
        q_sb=q.stride(0), q_sh=q.stride(1), q_st=q.stride(2), q_sk=q.stride(3),
        k_sb=k.stride(0), k_sh=k.stride(1), k_st=k.stride(2), k_sk=k.stride(3),
        v_sb=v.stride(0), v_sh=v.stride(1), v_st=v.stride(2), v_sv=v.stride(3),
        e_sb=e.stride(0), e_sh=e.stride(1), e_st=e.stride(2), e_sk=e.stride(3),
        g_sb=g.stride(0), g_sh=g.stride(1), g_st=g.stride(2), g_sk=g.stride(3),
        beta_sb=beta.stride(0), beta_sh=beta.stride(1), beta_st=beta.stride(2),
        gamma_sb=gamma.stride(0), gamma_sh=gamma.stride(1), gamma_st=gamma.stride(2),
        out_sb=out.stride(0), out_sh=out.stride(1), out_st=out.stride(2), out_sv=out.stride(3),
        num_heads=H, seq_len=T, head_k=K, head_v=V, scale_f=float(scale),
        BK=bk, BV=bv, num_warps=num_warps, num_stages=1,
    )


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    if B == 0 or H == 0 or T == 0 or V == 0:
        return out

    # The chunkwise path assumes the documented operating regime
    # (K == V <= 128, unit-norm e, gates in [0, 1], log-decay <= 0).
    # Outside it, fall back to the exact per-step kernel.
    use_chunk = False
    if K == V and K <= 128:
        e_norm = torch.sqrt(torch.sum(e.float() * e.float(), dim=-1))
        checks = torch.stack((
            torch.all((e_norm > 0.85) & (e_norm < 1.18)),
            torch.all((beta >= 0) & (beta <= 1)),
            torch.all((gamma >= 0) & (gamma <= 1)),
            torch.all(g <= 0),
        ))
        use_chunk = bool(checks.all())

    if use_chunk:
        bd = max(16, triton.next_power_of_2(K))
        _chunk_kernel[(B * H,)](
            q, k, v, e, g, beta, gamma, out,
            q_sb=q.stride(0), q_sh=q.stride(1), q_st=q.stride(2), q_sk=q.stride(3),
            k_sb=k.stride(0), k_sh=k.stride(1), k_st=k.stride(2), k_sk=k.stride(3),
            v_sb=v.stride(0), v_sh=v.stride(1), v_st=v.stride(2), v_sv=v.stride(3),
            e_sb=e.stride(0), e_sh=e.stride(1), e_st=e.stride(2), e_sk=e.stride(3),
            g_sb=g.stride(0), g_sh=g.stride(1), g_st=g.stride(2), g_sk=g.stride(3),
            beta_sb=beta.stride(0), beta_sh=beta.stride(1), beta_st=beta.stride(2),
            gamma_sb=gamma.stride(0), gamma_sh=gamma.stride(1), gamma_st=gamma.stride(2),
            out_sb=out.stride(0), out_sh=out.stride(1), out_st=out.stride(2), out_sv=out.stride(3),
            num_heads=H, seq_len=T, head_dim=K, scale_f=float(scale),
            CHUNK=16, BD=bd, num_warps=8, num_stages=1,
        )
    else:
        _launch_direct(q, k, v, e, g, beta, gamma, out, B, H, T, K, V, scale)
    return out
