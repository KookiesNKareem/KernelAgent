import torch
import triton
import triton.language as tl


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


@triton.jit
def _fwd_chunk_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, g_ptr, beta_ptr, state_ptr, ckpt_ptr,
    sQB, sQH, sQT, sQK,
    sKB, sKH, sKT, sKK,
    sVB, sVH, sVT, sVV,
    sOB, sOH, sOT, sOV,
    sGB, sGH, sGT,
    sBB, sBH, sBT,
    B, H, K, V, NVB, Vpad, t_start, L,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    EVEN_K: tl.constexpr, EVEN_V: tl.constexpr,
):
    pid = tl.program_id(0)
    vb = pid % NVB
    tmp = pid // NVB
    h = tmp % H
    b = tmp // H
    v_start = vb * BLOCK_V
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = tl.arange(0, BLOCK_V)
    Kp = BLOCK_K
    off = (b * H + h) * Kp * Vpad + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
    S = tl.load(state_ptr + off).to(tl.float32)
    if EVEN_K:
        k_mask = None
    else:
        k_mask = k_offs < K
    if EVEN_V:
        v_mask = None
    else:
        v_mask = (v_start + v_offs) < V
    for t_off in range(L):
        t = t_start + t_off
        k_ptrs = k_ptr + b * sKB + h * sKH + t * sKT + k_offs * sKK
        q_ptrs = q_ptr + b * sQB + h * sQH + t * sQT + k_offs * sQK
        v_ptrs = v_ptr + b * sVB + h * sVH + t * sVT + (v_start + v_offs) * sVV
        o_ptrs = o_ptr + b * sOB + h * sOH + t * sOT + (v_start + v_offs) * sOV
        g_p = g_ptr + b * sGB + h * sGH + t * sGT
        be_p = beta_ptr + b * sBB + h * sBH + t * sBT
        if EVEN_K:
            k_vec = tl.load(k_ptrs).to(tl.float32)
            q_vec = tl.load(q_ptrs).to(tl.float32)
        else:
            k_vec = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
            q_vec = tl.load(q_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        if EVEN_V:
            v_vec = tl.load(v_ptrs).to(tl.float32)
        else:
            v_vec = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
        g_t = tl.load(g_p).to(tl.float32)
        beta_t = tl.load(be_p).to(tl.float32)
        d = tl.exp(g_t)
        b2 = beta_t * 2.0
        S = S * d
        k_col = k_vec[:, None]
        pred = tl.sum(S * k_col, axis=0)
        diff = v_vec - pred
        S = S + (k_col * b2) * diff[None, :]
        out_vec = tl.sum(S * q_vec[:, None], axis=0) * scale
        if EVEN_V:
            tl.store(o_ptrs, out_vec)
        else:
            tl.store(o_ptrs, out_vec, mask=v_mask)
    tl.store(state_ptr + off, S)
    tl.store(ckpt_ptr + off, S)


@triton.jit
def _bwd_chunk_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, g_ptr, beta_ptr,
    sstart_ptr, carry_ptr, scratch_ptr,
    pdq_ptr, pdk_ptr, pdg_ptr, pdb_ptr, dv_ptr,
    sQB, sQH, sQT, sQK,
    sKB, sKH, sKT, sKK,
    sVB, sVH, sVT, sVV,
    sDB, sDH, sDT, sDV,
    sGB, sGH, sGT,
    sBB, sBH, sBT,
    sDVB, sDVH, sDVT, sDVV,
    B, H, K, V, NVB, Vpad, Cmax, t_start, L,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    EVEN_K: tl.constexpr, EVEN_V: tl.constexpr,
):
    pid = tl.program_id(0)
    vb = pid % NVB
    tmp = pid // NVB
    h = tmp % H
    b = tmp // H
    v_start = vb * BLOCK_V
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = tl.arange(0, BLOCK_V)
    Kp = BLOCK_K
    bh = b * H + h
    off_state = bh * Kp * Vpad + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
    S0 = tl.load(sstart_ptr + off_state).to(tl.float32)
    S = S0
    if EVEN_K:
        k_mask = None
    else:
        k_mask = k_offs < K
    if EVEN_V:
        v_mask = None
    else:
        v_mask = (v_start + v_offs) < V
    for i in range(L):
        t = t_start + i
        k_ptrs = k_ptr + b * sKB + h * sKH + t * sKT + k_offs * sKK
        v_ptrs = v_ptr + b * sVB + h * sVH + t * sVT + (v_start + v_offs) * sVV
        g_p = g_ptr + b * sGB + h * sGH + t * sGT
        be_p = beta_ptr + b * sBB + h * sBH + t * sBT
        if EVEN_K:
            k_vec = tl.load(k_ptrs).to(tl.float32)
        else:
            k_vec = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        if EVEN_V:
            v_vec = tl.load(v_ptrs).to(tl.float32)
        else:
            v_vec = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
        g_t = tl.load(g_p).to(tl.float32)
        beta_t = tl.load(be_p).to(tl.float32)
        d = tl.exp(g_t)
        b2 = beta_t * 2.0
        S = S * d
        pred = tl.sum(S * k_vec[:, None], axis=0)
        diff = v_vec - pred
        S = S + (k_vec[:, None] * b2) * diff[None, :]
        scr = scratch_ptr + (bh * Cmax + i) * Kp * Vpad + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
        tl.store(scr, S)
    F = tl.load(carry_ptr + off_state).to(tl.float32)
    for ii in range(L - 1):
        t_off = L - 1 - ii
        t = t_start + t_off
        q_ptrs = q_ptr + b * sQB + h * sQH + t * sQT + k_offs * sQK
        k_ptrs = k_ptr + b * sKB + h * sKH + t * sKT + k_offs * sKK
        v_ptrs = v_ptr + b * sVB + h * sVH + t * sVT + (v_start + v_offs) * sVV
        do_ptrs = do_ptr + b * sDB + h * sDH + t * sDT + (v_start + v_offs) * sDV
        dv_ptrs = dv_ptr + b * sDVB + h * sDVH + t * sDVT + (v_start + v_offs) * sDVV
        g_p = g_ptr + b * sGB + h * sGH + t * sGT
        be_p = beta_ptr + b * sBB + h * sBH + t * sBT
        if EVEN_K:
            q_vec = tl.load(q_ptrs).to(tl.float32)
            k_vec2 = tl.load(k_ptrs).to(tl.float32)
        else:
            q_vec = tl.load(q_ptrs, mask=k_mask, other=0.0).to(tl.float32)
            k_vec2 = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        if EVEN_V:
            v_vec = tl.load(v_ptrs).to(tl.float32)
            do_vec = tl.load(do_ptrs).to(tl.float32)
        else:
            v_vec = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
            do_vec = tl.load(do_ptrs, mask=v_mask, other=0.0).to(tl.float32)
        g_t = tl.load(g_p).to(tl.float32)
        beta_t = tl.load(be_p).to(tl.float32)
        d = tl.exp(g_t)
        b2 = beta_t * 2.0
        st_ptr = scratch_ptr + (bh * Cmax + t_off) * Kp * Vpad + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
        S_t = tl.load(st_ptr).to(tl.float32)
        sp_ptr = scratch_ptr + (bh * Cmax + t_off - 1) * Kp * Vpad + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
        S_prev = tl.load(sp_ptr).to(tl.float32)
        G = F + scale * q_vec[:, None] * do_vec[None, :]
        k_col = k_vec2[:, None]
        c_vec = tl.sum(G * k_col, axis=0)
        a_vec = tl.sum(S_prev * k_col, axis=0)
        e_vec = v_vec - d * a_vec
        dv_vec = b2 * c_vec
        if EVEN_V:
            tl.store(dv_ptrs, dv_vec)
        else:
            tl.store(dv_ptrs, dv_vec, mask=v_mask)
        pdq_vec = scale * tl.sum(S_t * do_vec[None, :], axis=1)
        pdk_vec = b2 * tl.sum(G * e_vec[None, :] - d * S_prev * c_vec[None, :], axis=1)
        s1 = tl.sum(G * S_t)
        s2 = tl.sum(c_vec * v_vec)
        s3 = tl.sum(c_vec * e_vec)
        base_q = (bh * NVB + vb) * L * BLOCK_K + t_off * BLOCK_K
        tl.store(pdq_ptr + base_q + k_offs, pdq_vec)
        tl.store(pdk_ptr + base_q + k_offs, pdk_vec)
        base_s = (bh * NVB + vb) * L + t_off
        tl.store(pdg_ptr + base_s, s1 - b2 * s2)
        tl.store(pdb_ptr + base_s, 2.0 * s3)
        F = d * G - (d * b2) * k_col * c_vec[None, :]
    # t_off == 0
    t = t_start
    q_ptrs0 = q_ptr + b * sQB + h * sQH + t * sQT + k_offs * sQK
    k_ptrs0 = k_ptr + b * sKB + h * sKH + t * sKT + k_offs * sKK
    v_ptrs0 = v_ptr + b * sVB + h * sVH + t * sVT + (v_start + v_offs) * sVV
    do_ptrs0 = do_ptr + b * sDB + h * sDH + t * sDT + (v_start + v_offs) * sDV
    dv_ptrs0 = dv_ptr + b * sDVB + h * sDVH + t * sDVT + (v_start + v_offs) * sDVV
    g_p0 = g_ptr + b * sGB + h * sGH + t * sGT
    be_p0 = beta_ptr + b * sBB + h * sBH + t * sBT
    if EVEN_K:
        q_vec0 = tl.load(q_ptrs0).to(tl.float32)
        k_vec0 = tl.load(k_ptrs0).to(tl.float32)
    else:
        q_vec0 = tl.load(q_ptrs0, mask=k_mask, other=0.0).to(tl.float32)
        k_vec0 = tl.load(k_ptrs0, mask=k_mask, other=0.0).to(tl.float32)
    if EVEN_V:
        v_vec0 = tl.load(v_ptrs0).to(tl.float32)
        do_vec0 = tl.load(do_ptrs0).to(tl.float32)
    else:
        v_vec0 = tl.load(v_ptrs0, mask=v_mask, other=0.0).to(tl.float32)
        do_vec0 = tl.load(do_ptrs0, mask=v_mask, other=0.0).to(tl.float32)
    g_t0 = tl.load(g_p0).to(tl.float32)
    beta_t0 = tl.load(be_p0).to(tl.float32)
    d0 = tl.exp(g_t0)
    b20 = beta_t0 * 2.0
    st_ptr0 = scratch_ptr + (bh * Cmax) * Kp * Vpad + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
    S_t0 = tl.load(st_ptr0).to(tl.float32)
    S_prev0 = S0
    G0 = F + scale * q_vec0[:, None] * do_vec0[None, :]
    k_col0 = k_vec0[:, None]
    c0 = tl.sum(G0 * k_col0, axis=0)
    a0 = tl.sum(S_prev0 * k_col0, axis=0)
    e0 = v_vec0 - d0 * a0
    dv0 = b20 * c0
    if EVEN_V:
        tl.store(dv_ptrs0, dv0)
    else:
        tl.store(dv_ptrs0, dv0, mask=v_mask)
    pdq0 = scale * tl.sum(S_t0 * do_vec0[None, :], axis=1)
    pdk0 = b20 * tl.sum(G0 * e0[None, :] - d0 * S_prev0 * c0[None, :], axis=1)
    s10 = tl.sum(G0 * S_t0)
    s20 = tl.sum(c0 * v_vec0)
    s30 = tl.sum(c0 * e0)
    base_q0 = (bh * NVB + vb) * L * BLOCK_K
    tl.store(pdq_ptr + base_q0 + k_offs, pdq0)
    tl.store(pdk_ptr + base_q0 + k_offs, pdk0)
    base_s0 = (bh * NVB + vb) * L
    tl.store(pdg_ptr + base_s0, s10 - b20 * s20)
    tl.store(pdb_ptr + base_s0, 2.0 * s30)
    F = d0 * G0 - (d0 * b20) * k_col0 * c0[None, :]
    tl.store(carry_ptr + off_state, F)


class _DeltaFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        if torch.is_tensor(scale):
            try:
                scale_f = float(scale.detach().to(torch.float32).reshape(-1)[0].item()) if scale.numel() >= 1 else 1.0
            except Exception:
                scale_f = 1.0
        else:
            scale_f = float(scale)
        dev = q.device
        if T == 0 or B == 0 or H == 0 or K == 0 or V == 0:
            out = torch.empty(B, H, T, V, dtype=torch.float32, device=dev)
            ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
            ctx.NVB, ctx.Cmax, ctx.N = 1, 64, 0
            ctx.BLOCK_K, ctx.BLOCK_V, ctx.Vpad = 1, 1, 1
            ctx.EVEN_K, ctx.EVEN_V, ctx.scale_f = True, True, scale_f
            ctx.save_for_backward(q, k, v, e, g, beta, gamma, torch.empty(0, dtype=torch.float32, device=dev))
            return out
        BLOCK_K = _next_pow2(K)
        if V <= 32:
            BLOCK_V = _next_pow2(V)
        else:
            BLOCK_V = 32 if BLOCK_K <= 128 else 16
        NVB = (V + BLOCK_V - 1) // BLOCK_V
        EVEN_K = (K == BLOCK_K)
        EVEN_V = (V % BLOCK_V == 0)
        Vpad = NVB * BLOCK_V
        Kp = BLOCK_K
        Cmax = 64
        N = (T + Cmax - 1) // Cmax
        M = B * H * Kp * Vpad
        out = torch.empty(B, H, T, V, dtype=torch.float32, device=dev)
        state = torch.zeros(M, dtype=torch.float32, device=dev)
        ckpt = torch.empty(N * M, dtype=torch.float32, device=dev)
        sQB, sQH, sQT, sQK = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
        sKB, sKH, sKT, sKK = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
        sVB, sVH, sVT, sVV = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
        sOB, sOH, sOT, sOV = out.stride(0), out.stride(1), out.stride(2), out.stride(3)
        sGB, sGH, sGT = g.stride(0), g.stride(1), g.stride(2)
        sBB, sBH, sBT = beta.stride(0), beta.stride(1), beta.stride(2)
        grid = (B * H * NVB,)
        for c in range(N):
            t0 = c * Cmax
            Lc = Cmax if (t0 + Cmax) <= T else (T - t0)
            ckpt_c = ckpt[c * M:(c + 1) * M]
            _fwd_chunk_kernel[grid](
                q, k, v, out, g, beta, state, ckpt_c,
                sQB, sQH, sQT, sQK, sKB, sKH, sKT, sKK,
                sVB, sVH, sVT, sVV, sOB, sOH, sOT, sOV,
                sGB, sGH, sGT, sBB, sBH, sBT,
                B, H, K, V, NVB, Vpad, t0, Lc, scale_f,
                BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, EVEN_K=EVEN_K, EVEN_V=EVEN_V,
                num_warps=4, num_stages=2,
            )
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
        ctx.NVB, ctx.Cmax, ctx.N = NVB, Cmax, N
        ctx.BLOCK_K, ctx.BLOCK_V, ctx.Vpad = BLOCK_K, BLOCK_V, Vpad
        ctx.EVEN_K, ctx.EVEN_V, ctx.scale_f = EVEN_K, EVEN_V, scale_f
        ctx.save_for_backward(q, k, v, e, g, beta, gamma, ckpt)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma, ckpt = ctx.saved_tensors
        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        NVB, Cmax, N = ctx.NVB, ctx.Cmax, ctx.N
        BLOCK_K, BLOCK_V, Vpad = ctx.BLOCK_K, ctx.BLOCK_V, ctx.Vpad
        EVEN_K, EVEN_V, scale_f = ctx.EVEN_K, ctx.EVEN_V, ctx.scale_f
        dev = q.device
        if N == 0 or T == 0 or B == 0 or H == 0 or K == 0 or V == 0:
            return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v), torch.zeros_like(e), torch.zeros_like(g), torch.zeros_like(beta), torch.zeros_like(gamma), None
        do = grad_out.to(torch.float32) if grad_out.dtype != torch.float32 else grad_out
        dq32 = torch.empty(B, H, T, K, dtype=torch.float32, device=dev)
        dk32 = torch.empty(B, H, T, K, dtype=torch.float32, device=dev)
        dv32 = torch.empty(B, H, T, V, dtype=torch.float32, device=dev)
        dg32 = torch.empty(B, H, T, dtype=torch.float32, device=dev)
        db32 = torch.empty(B, H, T, dtype=torch.float32, device=dev)
        M = B * H * BLOCK_K * Vpad
        carry = torch.zeros(M, dtype=torch.float32, device=dev)
        zeros_start = torch.zeros(M, dtype=torch.float32, device=dev)
        scratch = torch.empty(B * H * Cmax * BLOCK_K * Vpad, dtype=torch.float32, device=dev)
        sQB, sQH, sQT, sQK = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
        sKB, sKH, sKT, sKK = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
        sVB, sVH, sVT, sVV = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
        sDB, sDH, sDT, sDV = do.stride(0), do.stride(1), do.stride(2), do.stride(3)
        sGB, sGH, sGT = g.stride(0), g.stride(1), g.stride(2)
        sBB, sBH, sBT = beta.stride(0), beta.stride(1), beta.stride(2)
        sDVB, sDVH, sDVT, sDVV = dv32.stride(0), dv32.stride(1), dv32.stride(2), dv32.stride(3)
        grid = (B * H * NVB,)
        for c in range(N - 1, -1, -1):
            t0 = c * Cmax
            Lc = Cmax if (t0 + Cmax) <= T else (T - t0)
            if c == 0:
                sstart = zeros_start
            else:
                sstart = ckpt[(c - 1) * M:c * M]
            pdq = torch.empty(B * H * NVB * Lc * BLOCK_K, dtype=torch.float32, device=dev)
            pdk = torch.empty(B * H * NVB * Lc * BLOCK_K, dtype=torch.float32, device=dev)
            pdg = torch.empty(B * H * NVB * Lc, dtype=torch.float32, device=dev)
            pdb = torch.empty(B * H * NVB * Lc, dtype=torch.float32, device=dev)
            _bwd_chunk_kernel[grid](
                q, k, v, do, g, beta, sstart, carry, scratch,
                pdq, pdk, pdg, pdb, dv32,
                sQB, sQH, sQT, sQK, sKB, sKH, sKT, sKK,
                sVB, sVH, sVT, sVV, sDB, sDH, sDT, sDV,
                sGB, sGH, sGT, sBB, sBH, sBT,
                sDVB, sDVH, sDVT, sDVV,
                B, H, K, V, NVB, Vpad, Cmax, t0, Lc, scale_f,
                BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, EVEN_K=EVEN_K, EVEN_V=EVEN_V,
                num_warps=4, num_stages=1,
            )
            dq32[:, :, t0:t0 + Lc, :] = pdq.view(B, H, NVB, Lc, BLOCK_K).sum(2)[..., :K]
            dk32[:, :, t0:t0 + Lc, :] = pdk.view(B, H, NVB, Lc, BLOCK_K).sum(2)[..., :K]
            dg32[:, :, t0:t0 + Lc] = pdg.view(B, H, NVB, Lc).sum(2)
            db32[:, :, t0:t0 + Lc] = pdb.view(B, H, NVB, Lc).sum(2)
        return dq32.to(q.dtype), dk32.to(k.dtype), dv32.to(v.dtype), torch.zeros_like(e), dg32.to(g.dtype), db32.to(beta.dtype), torch.zeros_like(gamma), None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _DeltaFunc.apply(q, k, v, e, g, beta, gamma, scale)
