```python
import torch
import triton
import triton.language as tl


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


@triton.jit
def _fwd_chunk_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, g_ptr, beta_ptr, states_ptr,
    B, H, T, K, V, NVB, Vpad,
    chunk_idx, t_start, L,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    vb = pid % NVB
    tmp = pid // NVB
    h = tmp % H
    b = tmp // H
    bh = b * H + h
    BH = B * H
    Kp = BLOCK_K
    v_start = vb * BLOCK_V
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = (v_start + v_offs) < V
    base0 = (chunk_idx * BH + bh) * Kp * Vpad
    base1 = ((chunk_idx + 1) * BH + bh) * Kp * Vpad
    s_ptrs0 = states_ptr + base0 + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
    S = tl.load(s_ptrs0).to(tl.float32)
    bhT = bh * T
    q_base = q_ptr + bhT * K
    k_base = k_ptr + bhT * K
    v_base = v_ptr + bhT * V
    o_base = o_ptr + bhT * V
    g_base = g_ptr + bhT
    be_base = beta_ptr + bhT
    for t_off in range(L):
        t = t_start + t_off
        k_vec = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        q_vec = tl.load(q_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        v_vec = tl.load(v_base + t * V + (v_start + v_offs), mask=v_mask, other=0.0).to(tl.float32)
        g_t = tl.load(g_base + t).to(tl.float32)
        beta_t = tl.load(be_base + t).to(tl.float32)
        d = tl.exp(g_t)
        b2 = beta_t * 2.0
        S = S * d
        k_col = k_vec[:, None]
        pred = tl.sum(S * k_col, axis=0)
        diff = v_vec - pred
        S = S + (k_col * b2) * diff[None, :]
        out_vec = tl.sum(S * q_vec[:, None], axis=0) * scale
        tl.store(o_base + t * V + (v_start + v_offs), out_vec, mask=v_mask)
    s_ptrs1 = states_ptr + base1 + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
    tl.store(s_ptrs1, S)


@triton.jit
def _bwd_chunk_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, g_ptr, beta_ptr,
    states_ptr, carry_ptr, scratch_ptr,
    pdq_ptr, pdk_ptr, pdg_ptr, pdb_ptr, dv_ptr,
    B, H, T, K, V, NVB, Vpad, Cmax,
    chunk_idx, t_start, L,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    vb = pid % NVB
    tmp = pid // NVB
    h = tmp % H
    b = tmp // H
    bh = b * H + h
    BH = B * H
    Kp = BLOCK_K
    v_start = vb * BLOCK_V
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = (v_start + v_offs) < V
    s0_base = (chunk_idx * BH + bh) * Kp * Vpad
    s0_ptrs = states_ptr + s0_base + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
    S0 = tl.load(s0_ptrs).to(tl.float32)
    carry_base = bh * Kp * Vpad
    carry_ptrs = carry_ptr + carry_base + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
    F = tl.load(carry_ptrs).to(tl.float32)
    S = S0
    bhT = bh * T
    q_base = q_ptr + bhT * K
    k_base = k_ptr + bhT * K
    v_base = v_ptr + bhT * V
    do_base = do_ptr + bhT * V
    g_base = g_ptr + bhT
    be_base = beta_ptr + bhT
    dv_base = dv_ptr + bhT * V
    scr_bh = bh * Cmax * Kp * Vpad
    for i in range(L):
        t = t_start + i
        k_vec = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        v_vec = tl.load(v_base + t * V + (v_start + v_offs), mask=v_mask, other=0.0).to(tl.float32)
        g_t = tl.load(g_base + t).to(tl.float32)
        beta_t = tl.load(be_base + t).to(tl.float32)
        d = tl.exp(g_t)
        b2 = beta_t * 2.0
        S = S * d
        pred = tl.sum(S * k_vec[:, None], axis=0)
        diff = v_vec - pred
        S = S + (k_vec[:, None] * b2) * diff[None, :]
        scr_ptrs = scratch_ptr + scr_bh + i * Kp * Vpad + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
        tl.store(scr_ptrs, S)
    for ii in range(L):
        t_off = L - 1 - ii
        t = t_start + t_off
        q_vec = tl.load(q_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        k_vec2 = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        v_vec2 = tl.load(v_base + t * V + (v_start + v_offs), mask=v_mask, other=0.0).to(tl.float32)
        do_vec = tl.load(do_base + t * V + (v_start + v_offs), mask=v_mask, other=0.0).to(tl.float32)
        g_t2 = tl.load(g_base + t).to(tl.float32)
        beta_t2 = tl.load(be_base + t).to(tl.float32)
        d2 = tl.exp(g_t2)
        b2_2 = beta_t2 * 2.0
        st_ptrs = scratch_ptr + scr_bh + t_off * Kp * Vpad + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
        S_t = tl.load(st_ptrs).to(tl.float32)
        if t_off == 0:
            S_prev = S0
        else:
            sp_ptrs = scratch_ptr + scr_bh + (t_off - 1) * Kp * Vpad + k_offs[:, None] * Vpad + (v_start + v_offs)[None, :]
            S_prev = tl.load(sp_ptrs).to(tl.float32)
        G = F + scale * q_vec[:, None] * do_vec[None, :]
        k_col = k_vec2[:, None]
        c_vec = tl.sum(G * k_col, axis=0)
        a_vec = tl.sum(S_prev * k_col, axis=0)
        e_vec = v_vec2 - d2 * a_vec
        dv_vec = b2_2 * c_vec
        tl.store(dv_base + t * V + (v_start + v_offs), dv_vec, mask=v_mask)
        pdq_vec = scale * tl.sum(S_t * do_vec[None, :], axis=1)
        pdk_vec = b2_2 * tl.sum(G * e_vec[None, :] - d2 * S_prev * c_vec[None, :], axis=1)
        s1 = tl.sum(G * S_t)
        s2 = tl.sum(c_vec * v_vec2)
        s3 = tl.sum(c_vec * e_vec)
        bidx = (bh * NVB + vb) * L + t_off
        qidx = bidx * BLOCK_K
        tl.store(pdq_ptr + qidx + k_offs, pdq_vec)
        tl.store(pdk_ptr + qidx + k_offs, pdk_vec)
        tl.store(pdg_ptr + bidx, s1 - b2_2 * s2)
        tl.store(pdb_ptr + bidx, 2.0 * s3)
        F = d2 * G - (d2 * b2_2) * k_col * c_vec[None, :]
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
        dev = q.device
        if T == 0 or B == 0 or H == 0 or K == 0 or V == 0:
            out = torch.empty(B, H, T, V, dtype=torch.float32, device=dev)
            ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
            ctx.NVB, ctx.Cmax, ctx.N = 1, 32, 0
            ctx.BLOCK_K, ctx.BLOCK_V, ctx.Vpad = 1, 1, 1
            ctx.scale_f = scale_f
            ctx.save_for_backward(q, k, v, e, g, beta, gamma, torch.empty(0, dtype=torch.float32, device=dev))
            return out
        BLOCK_K = _next_pow2(K)
        if V <= 32:
            BLOCK_V = _next_pow2(V)
        else:
            BLOCK_V = 32 if BLOCK_K <= 128 else 16
        NVB = (V + BLOCK_V - 1) // BLOCK_V
        Vpad = NVB * BLOCK_V
        Kp = BLOCK_K
        Cmax = 32
        N = (T + Cmax - 1) // Cmax
        BH = B * H
        M = BH * Kp * Vpad
        qc = q.contiguous()
        kc = k.contiguous()
        vc = v.contiguous()
        gc = g.contiguous()
        bec = beta.contiguous()
        out = torch.empty(B, H, T, V, dtype=torch.float32, device=dev)
        states = torch.zeros((N + 1) * M, dtype=torch.float32, device=dev)
        grid = (BH * NVB,)
        for c in range(N):
            t0 = c * Cmax
            Lc = Cmax if (t0 + Cmax) <= T else (T - t0)
            _fwd_chunk_kernel[grid](
                qc, kc, vc, out, gc, bec, states,
                B, H, T, K, V, NVB, Vpad,
                c, t0, Lc, scale_f,
                BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
                num_warps=4, num_stages=2,
            )
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
        ctx.NVB, ctx.Cmax, ctx.N = NVB, Cmax, N
        ctx.BLOCK_K, ctx.BLOCK_V, ctx.Vpad = BLOCK_K, BLOCK_V, Vpad
        ctx.scale_f = scale_f
        ctx.save_for_backward(qc, kc, vc, e, gc, bec, gamma, states)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        qc, kc, vc, e, gc, bec, gamma, states = ctx.saved_tensors
        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        NVB, Cmax, N = ctx.NVB, ctx.Cmax, ctx.N
        BLOCK_K, BLOCK_V, Vpad = ctx.BLOCK_K, ctx.BLOCK_V, ctx.Vpad
        scale_f = ctx.scale_f
        dev = qc.device
        if N == 0 or T == 0 or B == 0 or H == 0 or K == 0 or V == 0:
            return torch.zeros_like(qc), torch.zeros_like(kc), torch.zeros_like(vc), torch.zeros_like(e), torch.zeros_like(gc), torch.zeros_like(bec), torch.zeros_like(gamma), None
        Kp = BLOCK_K
        BH = B * H
        M = BH * Kp * Vpad
        do_c = grad_out.contiguous().to(torch.float32)
        dq32 = torch.empty(B, H, T, K, dtype=torch.float32, device=dev)
        dk32 = torch.empty(B, H, T, K, dtype=torch.float32, device=dev)
        dv32 = torch.empty(B, H, T, V, dtype=torch.float32, device=dev)
        dg32 = torch.empty(B, H, T, dtype=torch.float32, device=dev)
        db32 = torch.empty(B, H, T, dtype=torch.float32, device=dev)
        carry = torch.zeros(M, dtype=torch.float32, device=dev)
        scratch = torch.empty(BH * Cmax * Kp * Vpad, dtype=torch.float32, device=dev)
        grid = (BH * NVB,)
        for c in range(N - 1, -1, -1):
            t0 = c * Cmax
            Lc = Cmax if (t0 + Cmax) <= T else (T - t0)
            pdq = torch.empty(BH * NVB * Lc * Kp, dtype=torch.float32, device=dev)
            pdk = torch.empty(BH * NVB * Lc * Kp, dtype=torch.float32, device=dev)
            pdg = torch.empty(BH * NVB * Lc, dtype=torch.float32, device=dev)
            pdb = torch.empty(BH * NVB * Lc, dtype=torch.float32, device=dev)
            _bwd_chunk_kernel[grid](
                qc, kc, vc, do_c, gc, bec,
                states, carry, scratch,
                pdq, pdk, pdg, pdb, dv32,
                B, H, T, K, V, NVB, Vpad, Cmax,
                c, t0, Lc, scale_f,
                BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
                num_warps=4, num_stages=1,
            )
            dq32[:, :, t0:t0 + Lc, :] = pdq.view(B, H, NVB, Lc, Kp).sum(2)[..., :K]
            dk32[:, :, t0:t0 + Lc, :] = pdk.view(B, H, NVB, Lc, Kp).sum(2)[..., :K]
            dg32[:, :, t0:t0 + Lc] = pdg.view(B, H, NVB, Lc).sum(2)
            db32[:, :, t0:t0 + Lc] = pdb.view(B, H, NVB, Lc).sum(2)
        return dq32.to(qc.dtype), dk32.to(kc.dtype), dv32.to(vc.dtype), torch.zeros_like(e), dg32.to(gc.dtype), db32.to(bec.dtype), torch.zeros_like(gamma), None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _DeltaFunc.apply(q, k, v, e, g, beta, gamma, scale)
```