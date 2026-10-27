```python
import torch
import triton
import triton.language as tl


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, g_ptr, beta_ptr, ckpt_ptr,
    sQB, sQH, sQT, sQK,
    sKB, sKH, sKT, sKK,
    sVB, sVH, sVT, sVV,
    sOB, sOH, sOT, sOV,
    sGB, sGH, sGT,
    sBB, sBH, sBT,
    B, H, T, K, V, NVB, Cmax, N,
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
    Vpad = NVB * BLOCK_V
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    k_ptrs = k_ptr + b * sKB + h * sKH + k_offs * sKK
    q_ptrs = q_ptr + b * sQB + h * sQH + k_offs * sQK
    v_ptrs = v_ptr + b * sVB + h * sVH + (v_start + v_offs) * sVV
    o_ptrs = o_ptr + b * sOB + h * sOH + (v_start + v_offs) * sOV
    g_cur = g_ptr + b * sGB + h * sGH
    b_cur = beta_ptr + b * sBB + h * sBH
    k_mask = k_offs < K
    v_mask = (v_start + v_offs) < V
    c_idx = 0
    next_ckpt = Cmax - 1
    for t in range(T):
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
        g_t = tl.load(g_cur).to(tl.float32)
        beta_t = tl.load(b_cur).to(tl.float32)
        d = tl.exp(g_t)
        b2 = beta_t * 2.0
        S = S * d
        k_col = k_vec[:, None]
        pred = tl.sum(S * k_col, 0)
        diff = v_vec - pred
        S = S + (k_col * b2) * diff[None, :]
        out_vec = tl.sum(S * q_vec[:, None], 0) * scale
        if EVEN_V:
            tl.store(o_ptrs, out_vec)
        else:
            tl.store(o_ptrs, out_vec, mask=v_mask)
        is_ckpt = (t == next_ckpt) | (t == T - 1)
        if is_ckpt:
            ckpt_base = ckpt_ptr + c_idx * B * H * Kp * Vpad + b * H * Kp * Vpad + h * Kp * Vpad
            ckpt_ptrs = ckpt_base + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
            tl.store(ckpt_ptrs, S)
            c_idx += 1
            next_ckpt += Cmax
        k_ptrs += sKT
        q_ptrs += sQT
        v_ptrs += sVT
        o_ptrs += sOT
        g_cur += sGT
        b_cur += sBT


@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, g_ptr, beta_ptr,
    ckpt_ptr, carry_ptr, scratch_ptr,
    pdq_ptr, pdk_ptr, pdg_ptr, pdbeta_ptr, dv_ptr,
    sQB, sQH, sQT, sQK,
    sKB, sKH, sKT, sKK,
    sVB, sVH, sVT, sVV,
    sDB, sDH, sDT, sDV,
    sGB, sGH, sGT,
    sBB, sBH, sBT,
    sDVB, sDVH, sDVT, sDVV,
    B, H, T, K, V, NVB, Cmax, N,
    t_start, L, chunk_idx,
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
    Vpad = NVB * BLOCK_V
    k_mask = k_offs < K
    v_mask = (v_start + v_offs) < V
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    if chunk_idx != 0:
        ckpt_base0 = ckpt_ptr + (chunk_idx - 1) * B * H * Kp * Vpad + b * H * Kp * Vpad + h * Kp * Vpad
        ckpt_ptrs0 = ckpt_base0 + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
        S = tl.load(ckpt_ptrs0).to(tl.float32)
    k_fptrs = k_ptr + b * sKB + h * sKH + t_start * sKT + k_offs * sKK
    v_fptrs = v_ptr + b * sVB + h * sVH + t_start * sVT + (v_start + v_offs) * sVV
    g_fcur = g_ptr + b * sGB + h * sGH + t_start * sGT
    b_fcur = beta_ptr + b * sBB + h * sBH + t_start * sBT
    scratch_base = scratch_ptr + b * H * Cmax * Kp * Vpad + h * Cmax * Kp * Vpad
    for i in range(L):
        if EVEN_K:
            k_vec = tl.load(k_fptrs).to(tl.float32)
        else:
            k_vec = tl.load(k_fptrs, mask=k_mask, other=0.0).to(tl.float32)
        if EVEN_V:
            v_vec = tl.load(v_fptrs).to(tl.float32)
        else:
            v_vec = tl.load(v_fptrs, mask=v_mask, other=0.0).to(tl.float32)
        g_t = tl.load(g_fcur).to(tl.float32)
        beta_t = tl.load(b_fcur).to(tl.float32)
        d = tl.exp(g_t)
        b2 = beta_t * 2.0
        S = S * d
        pred = tl.sum(S * k_vec[:, None], 0)
        diff = v_vec - pred
        S = S + (k_vec[:, None] * b2) * diff[None, :]
        scr_ptrs = scratch_base + i * Kp * Vpad + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
        tl.store(scr_ptrs, S)
        k_fptrs += sKT
        v_fptrs += sVT
        g_fcur += sGT
        b_fcur += sBT
    carry_base = carry_ptr + b * H * Kp * Vpad + h * Kp * Vpad
    carry_ptrs = carry_base + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
    F = tl.load(carry_ptrs).to(tl.float32)
    t_last = t_start + L - 1
    q_bptrs = q_ptr + b * sQB + h * sQH + t_last * sQT + k_offs * sQK
    k_bptrs = k_ptr + b * sKB + h * sKH + t_last * sKT + k_offs * sKK
    v_bptrs = v_ptr + b * sVB + h * sVH + t_last * sVT + (v_start + v_offs) * sVV
    do_bptrs = do_ptr + b * sDB + h * sDH + t_last * sDT + (v_start + v_offs) * sDV
    g_bcur = g_ptr + b * sGB + h * sGH + t_last * sGT
    b_bcur = beta_ptr + b * sBB + h * sBH + t_last * sBT
    dv_bptrs = dv_ptr + b * sDVB + h * sDVH + t_last * sDVT + (v_start + v_offs) * sDVV
    pdq_base = pdq_ptr + ((b * H + h) * NVB + vb) * Cmax * BLOCK_K
    pdk_base = pdk_ptr + ((b * H + h) * NVB + vb) * Cmax * BLOCK_K
    pdg_base = pdg_ptr + ((b * H + h) * NVB + vb) * Cmax
    pdbase = pdbeta_ptr + ((b * H + h) * NVB + vb) * Cmax
    cur_i = L - 1
    for _ in range(L):
        if EVEN_K:
            q_vec2 = tl.load(q_bptrs).to(tl.float32)
            k_vec2 = tl.load(k_bptrs).to(tl.float32)
        else:
            q_vec2 = tl.load(q_bptrs, mask=k_mask, other=0.0).to(tl.float32)
            k_vec2 = tl.load(k_bptrs, mask=k_mask, other=0.0).to(tl.float32)
        if EVEN_V:
            v_vec2 = tl.load(v_bptrs).to(tl.float32)
            do_vec = tl.load(do_bptrs).to(tl.float32)
        else:
            v_vec2 = tl.load(v_bptrs, mask=v_mask, other=0.0).to(tl.float32)
            do_vec = tl.load(do_bptrs, mask=v_mask, other=0.0).to(tl.float32)
        g_t2 = tl.load(g_bcur).to(tl.float32)
        beta_t2 = tl.load(b_bcur).to(tl.float32)
        d2 = tl.exp(g_t2)
        b2_2 = beta_t2 * 2.0
        st_ptrs = scratch_base + cur_i * Kp * Vpad + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
        S_t = tl.load(st_ptrs).to(tl.float32)
        if cur_i == 0:
            if chunk_idx == 0:
                S_prev = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
            else:
                ckpt_b = ckpt_ptr + (chunk_idx - 1) * B * H * Kp * Vpad + b * H * Kp * Vpad + h * Kp * Vpad
                ckpt_p = ckpt_b + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
                S_prev = tl.load(ckpt_p).to(tl.float32)
        else:
            sprev_ptrs = scratch_base + (cur_i - 1) * Kp * Vpad + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
            S_prev = tl.load(sprev_ptrs).to(tl.float32)
        G = F + scale * q_vec2[:, None] * do_vec[None, :]
        c_vec = tl.sum(G * k_vec2[:, None], 0)
        a_vec = tl.sum(S_prev * k_vec2[:, None], 0)
        e_vec = v_vec2 - d2 * a_vec
        dv_vec = b2_2 * c_vec
        if EVEN_V:
            tl.store(dv_bptrs, dv_vec)
        else:
            tl.store(dv_bptrs, dv_vec, mask=v_mask)
        pdq_vec = scale * tl.sum(S_t * do_vec[None, :], 1)
        tl.store(pdq_base + cur_i * BLOCK_K + k_offs, pdq_vec)
        tmp_mat = G * e_vec[None, :] - d2 * S_prev * c_vec[None, :]
        pdk_vec = b2_2 * tl.sum(tmp_mat, 1)
        tl.store(pdk_base + cur_i * BLOCK_K + k_offs, pdk_vec)
        sum_gs = tl.sum(G * S_t)
        dot_cv = tl.sum(c_vec * v_vec2)
        tl.store(pdg_base + cur_i, sum_gs - b2_2 * dot_cv)
        tl.store(pdbase + cur_i, 2.0 * tl.sum(c_vec * e_vec))
        F = d2 * G - (d2 * b2_2) * k_vec2[:, None] * c_vec[None, :]
        q_bptrs -= sQT
        k_bptrs -= sKT
        v_bptrs -= sVT
        do_bptrs -= sDT
        g_bcur -= sGT
        b_bcur -= sBT
        dv_bptrs -= sDVT
        cur_i -= 1
    tl.store(carry_ptrs, F)


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
        ctx.scale_f = scale_f
        if T == 0 or B == 0 or H == 0 or K == 0 or V == 0:
            out = torch.empty(B, H, T, V, dtype=torch.float32, device=q.device)
            ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
            ctx.NVB, ctx.Cmax, ctx.N = 1, 1, 0
            ctx.BLOCK_K, ctx.BLOCK_V = 1, 1
            ctx.EVEN_K, ctx.EVEN_V = True, True
            ctx.save_for_backward(q, k, v, e, g, beta, gamma, torch.empty(0, dtype=torch.float32, device=q.device))
            return out
        BLOCK_K = _next_pow2(K)
        target = 8192
        max_bv = target // BLOCK_K if BLOCK_K > 0 else 128
        if max_bv < 16:
            max_bv = 16
        if max_bv > 128:
            max_bv = 128
        if V <= max_bv:
            BLOCK_V = _next_pow2(V)
        else:
            BLOCK_V = _next_pow2(max_bv)
        NVB = (V + BLOCK_V - 1) // BLOCK_V
        EVEN_K = (K == BLOCK_K)
        EVEN_V = (V % BLOCK_V == 0)
        Cmax = 64 if T >= 64 else T
        N = (T + Cmax - 1) // Cmax
        out = torch.empty(B, H, T, V, dtype=torch.float32, device=q.device)
        Kp = BLOCK_K
        Vpad = NVB * BLOCK_V
        ckpt = torch.empty(N * B * H * Kp * Vpad, dtype=torch.float32, device=q.device)
        sQB, sQH, sQT, sQK = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
        sKB, sKH, sKT, sKK = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
        sVB, sVH, sVT, sVV = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
        sOB, sOH, sOT, sOV = out.stride(0), out.stride(1), out.stride(2), out.stride(3)
        sGB, sGH, sGT = g.stride(0), g.stride(1), g.stride(2)
        sBB, sBH, sBT = beta.stride(0), beta.stride(1), beta.stride(2)
        grid = (B * H * NVB,)
        _fwd_kernel[grid](
            q, k, v, out, g, beta, ckpt,
            sQB, sQH, sQT, sQK, sKB, sKH, sKT, sKK,
            sVB, sVH, sVT, sVV, sOB, sOH, sOT, sOV,
            sGB, sGH, sGT, sBB, sBH, sBT,
            B, H, T, K, V, NVB, Cmax, N, scale_f,
            BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, EVEN_K=EVEN_K, EVEN_V=EVEN_V,
            num_warps=4, num_stages=3,
        )
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
        ctx.NVB, ctx.Cmax, ctx.N = NVB, Cmax, N
        ctx.BLOCK_K, ctx.BLOCK_V = BLOCK_K, BLOCK_V
        ctx.EVEN_K, ctx.EVEN_V = EVEN_K, EVEN_V
        ctx.save_for_backward(q, k, v, e, g, beta, gamma, ckpt)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, e, g, beta, gamma, ckpt = ctx.saved_tensors
        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        NVB, Cmax, N = ctx.NVB, ctx.Cmax, ctx.N
        BLOCK_K, BLOCK_V = ctx.BLOCK_K, ctx.BLOCK_V
        EVEN_K, EVEN_V = ctx.EVEN_K, ctx.EVEN_V
        scale_f = ctx.scale_f
        dev = q.device
        if T == 0 or B == 0 or H == 0 or K == 0 or V == 0:
            dq = torch.empty_like(q)
            if dq.numel() > 0:
                dq = torch.zeros_like(q)
            dk = torch.zeros_like(k) if k.numel() > 0 else k.clone()
            dv = torch.zeros_like(v) if v.numel() > 0 else v.clone()
            dg = torch.zeros_like(g) if g.numel() > 0 else g.clone()
            db = torch.zeros_like(beta) if beta.numel() > 0 else beta.clone()
            de = torch.zeros_like(e)
            dgam = torch.zeros_like(gamma)
            try:
                dq = dq.to(q.dtype)
            except Exception:
                pass
            return dq, dk, dv, de, dg, db, dgam, None
        do = grad_out.to(torch.float32) if grad_out.dtype != torch.float32 else grad_out
        dq32 = torch.empty(B, H, T, K, dtype=torch.float32, device=dev)
        dk32 = torch.empty(B, H, T, K, dtype=torch.float32, device=dev)
        dv32 = torch.empty(B, H, T, V, dtype=torch.float32, device=dev)
        dg32 = torch.empty(B, H, T, dtype=torch.float32, device=dev)
        db32 = torch.empty(B, H, T, dtype=torch.float32, device=dev)
        Kp = BLOCK_K
        Vpad = NVB * BLOCK_V
        carry = torch.zeros(B * H * Kp * Vpad, dtype=torch.float32, device=dev)
        scratch = torch.empty(B * H * Cmax * Kp * Vpad, dtype=torch.float32, device=dev)
        pdq = torch.empty(B * H * NVB * Cmax * BLOCK_K, dtype=torch.float32, device=dev)
        pdk = torch.empty(B * H * NVB * Cmax * BLOCK_K, dtype=torch.float32, device=dev)
        pdg = torch.empty(B * H * NVB * Cmax, dtype=torch.float32, device=dev)
        pdb = torch.empty(B * H * NVB * Cmax, dtype=torch.float32, device=dev)
        sQB, sQH, sQT, sQK = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
        sKB, sKH, sKT, sKK = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
        sVB, sVH, sVT, sVV = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
        sDB, sDH, sDT, sDV = do.stride(0), do.stride(1), do.stride(2), do.stride(3)
        sGB, sGH, sGT = g.stride(0), g.stride(1), g.stride(2)
        sBB, sBH, sBT = beta.stride(0), beta.stride(1), beta.stride(2)
        sDVB, sDVH, sDVT, sDVV = dv32.stride(0), dv32.stride(1), dv32.stride(2), dv32.stride(3)
        grid = (B * H * NVB,)
        for c_idx in range(N - 1, -1, -1):
            t_start = c_idx * Cmax
            L = Cmax if (t_start + Cmax) <= T else (T - t_start)
            _bwd_kernel[grid](
                q, k, v, do, g, beta, ckpt, carry, scratch,
                pdq, pdk, pdg, pdb, dv32,
                sQB, sQH, sQT, sQK, sKB, sKH, sKT, sKK,
                sVB, sVH, sVT, sVV, sDB, sDH, sDT, sDV,
                sGB, sGH, sGT, sBB, sBH, sBT,
                sDVB, sDVH, sDVT, sDVV,
                B, H, T, K, V, NVB, Cmax, N, t_start, L, c_idx, scale_f,
                BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, EVEN_K=EVEN_K, EVEN_V=EVEN_V,
                num_warps=8, num_stages=1,
            )
            vpdq = pdq.view(B, H, NVB, Cmax, BLOCK_K)[:, :, :, :L, :].sum(2)[..., :K]
            vpdk = pdk.view(B, H, NVB, Cmax, BLOCK_K)[:, :, :, :L, :].sum(2)[..., :K]
            vpdg = pdg.view(B, H, NVB, Cmax)[:, :, :, :L].sum(2)
            vpdb = pdb.view(B, H, NVB, Cmax)[:, :, :, :L].sum(2)
            dq32[:, :, t_start:t_start + L, :] = vpdq.permute(0, 1, 2, 3) if False else vpdq
            dk32[:, :, t_start:t_start + L, :] = vpdk
            dg32[:, :, t_start:t_start + L] = vpdg
            db32[:, :, t_start:t_start + L] = vpdb
        dq = dq32.to(q.dtype)
        dk = dk32.to(k.dtype)
        dv = dv32.to(v.dtype)
        dg = dg32.to(g.dtype)
        db = db32.to(beta.dtype)
        de = torch.zeros_like(e)
        dgam = torch.zeros_like(gamma)
        return dq, dk, dv, de, dg, db, dgam, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _DeltaFunc.apply(q, k, v, e, g, beta, gamma, scale)
```

My approach fuses the forward recurrence with V-sharded state in registers and saves chunk-boundary states to bound memory, then the backward replays each 64-step chunk forward into scratch and walks it in reverse to form state gradients and input gradients. Cross-shard V reductions for dq/dk/dg/dbeta use per-shard partials plus a small torch sum, while dv and state propagation stay shard-local, and e/gamma get exact zeros. I am unsure about Triton register pressure and chosen warps/stages for the backward tile, exact tl.sum/tl.exp rounding versus torch over long sequences, and masked-tail handling for non-power-of-two K/V.