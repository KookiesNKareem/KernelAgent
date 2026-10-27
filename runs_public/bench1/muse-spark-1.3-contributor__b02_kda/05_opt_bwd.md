```python
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, o_ptr, ckpt_ptr,
                T, K, V, scale, C, N,
                BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, NUM_VB: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // NUM_VB
    vb = pid % NUM_VB
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = k_offs < K
    mask_v = v_offs < V
    mask2d = tl.expand_dims(mask_k, 1) & tl.expand_dims(mask_v, 0)
    k2 = tl.expand_dims(k_offs, 1)
    v2 = tl.expand_dims(v_offs, 0)
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    for n in range(N):
        for c in range(C):
            t = n * C + c
            if t < T:
                base_qk = (bh * T + t) * K
                base_v = (bh * T + t) * V
                base_b = bh * T + t
                k_t = tl.load(k_ptr + base_qk + k_offs, mask=mask_k, other=0.0)
                q_t = tl.load(q_ptr + base_qk + k_offs, mask=mask_k, other=0.0)
                d_t = tl.load(d_ptr + base_qk + k_offs, mask=mask_k, other=1.0)
                beta_t = tl.load(beta_ptr + base_b)
                v_t = tl.load(v_ptr + base_v + v_offs, mask=mask_v, other=0.0)
                k_f = k_t.to(tl.float32)
                q_f = q_t.to(tl.float32)
                v_f = v_t.to(tl.float32)
                d_f = d_t.to(tl.float32)
                S = S * tl.expand_dims(d_f, 1)
                k_col = tl.expand_dims(k_f, 1)
                pred = tl.sum(S * k_col, axis=0)
                err = (v_f - pred) * beta_t
                S = S + k_col * tl.expand_dims(err, 0)
                q_scaled = q_f * scale
                out_t = tl.sum(S * tl.expand_dims(q_scaled, 1), axis=0)
                tl.store(o_ptr + base_v + v_offs, out_t, mask=mask_v)
        base_ckpt = ckpt_ptr + ((bh * N + n) * K) * V
        ptrs = base_ckpt + k2 * V + v2
        tl.store(ptrs, S, mask=mask2d)


@triton.jit
def _bwd_chunk_kernel(q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, do_ptr,
                      ckpt_ptr, scratch_ptr, gcarry_ptr,
                      dq_ptr, dk_ptr, dg_ptr, dbeta_ptr, dv_ptr,
                      T, K, V, scale, l, L, Cmax, N, n_idx, NUM_VB,
                      BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // NUM_VB
    vb = pid % NUM_VB
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = k_offs < K
    mask_v = v_offs < V
    mask2d = tl.expand_dims(mask_k, 1) & tl.expand_dims(mask_v, 0)
    k2 = tl.expand_dims(k_offs, 1)
    v2 = tl.expand_dims(v_offs, 0)

    S_in = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    if l != 0:
        base_ckpt = ckpt_ptr + ((bh * N + (n_idx - 1)) * K) * V
        ptrs_c = base_ckpt + k2 * V + v2
        S_in = tl.load(ptrs_c, mask=mask2d, other=0.0)

    base_g = gcarry_ptr + (bh * K) * V
    ptrs_g = base_g + k2 * V + v2
    G_in = tl.load(ptrs_g, mask=mask2d, other=0.0)

    S = S_in
    for c in range(L):
        t = l + c
        base_qk = (bh * T + t) * K
        base_v = (bh * T + t) * V
        base_b = bh * T + t
        k_t = tl.load(k_ptr + base_qk + k_offs, mask=mask_k, other=0.0).to(tl.float32)
        d_t = tl.load(d_ptr + base_qk + k_offs, mask=mask_k, other=1.0).to(tl.float32)
        b_t = tl.load(beta_ptr + base_b)
        v_t = tl.load(v_ptr + base_v + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        S = S * tl.expand_dims(d_t, 1)
        k_col = tl.expand_dims(k_t, 1)
        pred = tl.sum(S * k_col, axis=0)
        err = (v_t - pred) * b_t
        S = S + k_col * tl.expand_dims(err, 0)
        base_s = scratch_ptr + ((bh * Cmax + c) * K) * V
        ptrs_s = base_s + k2 * V + v2
        tl.store(ptrs_s, S, mask=mask2d)

    cur = G_in
    r = l + L
    if r < T:
        base_qk_r = (bh * T + r) * K
        k_r = tl.load(k_ptr + base_qk_r + k_offs, mask=mask_k, other=0.0).to(tl.float32)
        d_r = tl.load(d_ptr + base_qk_r + k_offs, mask=mask_k, other=1.0).to(tl.float32)
        b_r = tl.load(beta_ptr + bh * T + r)
        kcol_r = tl.expand_dims(k_r, 1)
        a_r = tl.sum(cur * kcol_r, axis=0)
        cur = (cur - b_r * kcol_r * tl.expand_dims(a_r, 0)) * tl.expand_dims(d_r, 1)

    last_G = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    for c in range(L):
        idx = L - 1 - c
        t = l + idx
        base_qk = (bh * T + t) * K
        base_v = (bh * T + t) * V
        base_b = bh * T + t
        q_t = tl.load(q_ptr + base_qk + k_offs, mask=mask_k, other=0.0).to(tl.float32)
        k_t = tl.load(k_ptr + base_qk + k_offs, mask=mask_k, other=0.0).to(tl.float32)
        d_t = tl.load(d_ptr + base_qk + k_offs, mask=mask_k, other=1.0).to(tl.float32)
        b_t = tl.load(beta_ptr + base_b)
        v_t = tl.load(v_ptr + base_v + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        do_t = tl.load(do_ptr + base_v + v_offs, mask=mask_v, other=0.0).to(tl.float32)
        base_st = scratch_ptr + ((bh * Cmax + idx) * K) * V
        ptrs_st = base_st + k2 * V + v2
        S_t = tl.load(ptrs_st, mask=mask2d, other=0.0)
        S_prev = S_in
        if idx > 0:
            base_sp = scratch_ptr + ((bh * Cmax + (idx - 1)) * K) * V
            ptrs_sp = base_sp + k2 * V + v2
            S_prev = tl.load(ptrs_sp, mask=mask2d, other=0.0)
        S_d = S_prev * tl.expand_dims(d_t, 1)
        k_col = tl.expand_dims(k_t, 1)
        pred = tl.sum(S_d * k_col, axis=0)
        y = v_t - pred
        Q = tl.expand_dims(q_t * scale, 1) * tl.expand_dims(do_t, 0)
        G = Q + cur
        a = tl.sum(G * k_col, axis=0)
        dv = b_t * a
        tl.store(dv_ptr + base_v + v_offs, dv, mask=mask_v)
        dq_part = tl.sum(S_t * tl.expand_dims(do_t, 0), axis=1) * scale
        w_part = tl.sum(G * tl.expand_dims(y, 0), axis=1)
        sda_part = tl.sum(S_d * tl.expand_dims(a, 0), axis=1)
        dot1_part = tl.sum(S_d * G, axis=1)
        dk_part = (w_part - sda_part) * b_t
        dg_part = dot1_part - b_t * k_t * sda_part
        db_part = tl.sum(a * y, axis=0)
        shard_off = ((bh * NUM_VB + vb) * T + t) * K
        tl.store(dq_ptr + shard_off + k_offs, dq_part, mask=mask_k)
        tl.store(dk_ptr + shard_off + k_offs, dk_part, mask=mask_k)
        tl.store(dg_ptr + shard_off + k_offs, dg_part, mask=mask_k)
        tl.store(dbeta_ptr + (bh * NUM_VB + vb) * T + t, db_part)
        cur = (G - b_t * k_col * tl.expand_dims(a, 0)) * tl.expand_dims(d_t, 1)
        last_G = G
    tl.store(ptrs_g, last_G, mask=mask2d)


def _pick_block(K, Vd):
    BK = triton.next_power_of_2(K) if K > 0 else 16
    if BK < 16:
        BK = 16
    BV = 64
    if Vd <= 16:
        BV = 16
    elif Vd <= 32:
        BV = 32
    while BK * BV > 8192 and BV > 16:
        BV //= 2
    if Vd > 0:
        NVB = (Vd + BV - 1) // BV
    else:
        NVB = 1
        BV = 16
    return BK, BV, NVB


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_, k_, v_, e_, g_, beta_, gamma_, scale_f):
        BH, T, K = q_.shape
        Vd = v_.shape[-1]
        dev = q_.device
        BK, BV, NVB = _pick_block(K, Vd)
        if T == 0 or BH == 0 or K == 0 or Vd == 0:
            out = torch.empty((BH, T, Vd), device=dev, dtype=torch.float32)
            ckpt = torch.empty((1,), device=dev, dtype=torch.float32)
            scratch = torch.empty((1,), device=dev, dtype=torch.float32)
            ctx.save_for_backward(q_, k_, v_, e_, g_, beta_, gamma_,
                                   ckpt, scratch,
                                   torch.empty((1,), device=dev, dtype=torch.float32))
            ctx.BH = BH; ctx.T = T; ctx.K = K; ctx.Vd = Vd
            ctx.scale_f = scale_f; ctx.BK = BK; ctx.BV = BV; ctx.NVB = NVB
            ctx.C = 64; ctx.N = 1
            return out
        denom = BH * K * Vd * 4
        C = (32 * 1024 * 1024) // denom if denom > 0 else 64
        if C < 8:
            C = 8
        if C > 128:
            C = 128
        if C > T:
            C = T
        N = (T + C - 1) // C
        d_ = torch.exp(g_) if g_.numel() > 0 else g_.clone()
        out = torch.empty((BH, T, Vd), device=dev, dtype=torch.float32)
        ckpt = torch.empty((BH, N, K, Vd), device=dev, dtype=torch.float32)
        scratch = torch.empty((BH, C, K, Vd), device=dev, dtype=torch.float32)
        grid = (BH * NVB,)
        nw = 8 if BK * BV > 4096 else 4
        _fwd_kernel[grid](q_, k_, v_, d_, beta_, out, ckpt,
                          T, K, Vd, scale_f, C, N,
                          BK, BV, NVB, num_warps=nw)
        ctx.save_for_backward(q_, k_, v_, d_, beta_, ckpt, scratch, e_, gamma_, g_)
        ctx.BH = BH; ctx.T = T; ctx.K = K; ctx.Vd = Vd
        ctx.scale_f = scale_f; ctx.BK = BK; ctx.BV = BV; ctx.NVB = NVB
        ctx.C = C; ctx.N = N
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q_, k_, v_, d_, beta_, ckpt, scratch, e_, gamma_, g_ = ctx.saved_tensors
        BH = ctx.BH; T = ctx.T; K = ctx.K; Vd = ctx.Vd
        scale_f = ctx.scale_f; BK = ctx.BK; BV = ctx.BV; NVB = ctx.NVB
        C = ctx.C; N = ctx.N
        dev = q_.device
        do_ = grad_out.contiguous()
        if T == 0 or BH == 0 or K == 0 or Vd == 0:
            return (torch.zeros_like(q_), torch.zeros_like(k_), torch.zeros_like(v_),
                    torch.zeros_like(e_), torch.zeros_like(g_), torch.zeros_like(beta_),
                    torch.zeros_like(gamma_), None)
        dq_s = torch.empty((BH * NVB, T, K), device=dev, dtype=torch.float32)
        dk_s = torch.empty((BH * NVB, T, K), device=dev, dtype=torch.float32)
        dg_s = torch.empty((BH * NVB, T, K), device=dev, dtype=torch.float32)
        db_s = torch.empty((BH * NVB, T), device=dev, dtype=torch.float32)
        dv = torch.empty((BH, T, Vd), device=dev, dtype=torch.float32)
        gcarry = torch.zeros((BH, K, Vd), device=dev, dtype=torch.float32)
        grid = (BH * NVB,)
        nw = 8 if BK * BV > 4096 else 4
        for n_idx in range(N - 1, -1, -1):
            l = n_idx * C
            L = T - l if l + C > T else C
            if L <= 0:
                continue
            _bwd_chunk_kernel[grid](q_, k_, v_, d_, beta_, do_, ckpt, scratch, gcarry,
                                    dq_s, dk_s, dg_s, db_s, dv,
                                    T, K, Vd, scale_f, l, L, C, N, n_idx, NVB,
                                    BK, BV, num_warps=nw)
        if NVB == 1:
            dq = dq_s.view(BH, T, K).to(q_.dtype)
            dk = dk_s.view(BH, T, K).to(k_.dtype)
            dg = dg_s.view(BH, T, K).to(g_.dtype)
            dbeta = db_s.view(BH, T).to(beta_.dtype)
        else:
            dq = dq_s.view(BH, NVB, T, K).sum(dim=1).to(q_.dtype)
            dk = dk_s.view(BH, NVB, T, K).sum(dim=1).to(k_.dtype)
            dg = dg_s.view(BH, NVB, T, K).sum(dim=1).to(g_.dtype)
            dbeta = db_s.view(BH, NVB, T).sum(dim=1).to(beta_.dtype)
        dv = dv.to(v_.dtype)
        de = torch.zeros_like(e_)
        dgamma = torch.zeros_like(gamma_)
        return (dq, dk, dv, de, dg, dbeta, dgamma, None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    BH = B * H
    scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
    q_ = q.reshape(BH, T, K).contiguous()
    k_ = k.reshape(BH, T, K).contiguous()
    v_ = v.reshape(BH, T, Vd).contiguous()
    e_ = e.reshape(BH, T, K).contiguous()
    g_ = g.reshape(BH, T, K).contiguous()
    beta_ = beta.reshape(BH, T).contiguous()
    gamma_ = gamma.reshape(BH, T).contiguous()
    out_ = _Func.apply(q_, k_, v_, e_, g_, beta_, gamma_, scale_f)
    return out_.view(B, H, T, Vd)
```
