```python
import torch
import triton
import triton.language as tl

@triton.jit
def fwd_chunk_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr, chk_ptr,
    BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    N: tl.constexpr, C: tl.constexpr, NK: tl.constexpr, NV: tl.constexpr,
    SCALE: tl.constexpr, t0: tl.constexpr, cur_T: tl.constexpr, chk_idx: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_k = tl.arange(0, 32)
    offs_v = tl.arange(0, 32)
    v_off = pid_v * 32 + offs_v
    mask_v = v_off < V
    chk_base_start = (pid_bh * (N + 2) + chk_idx) * K * V
    chk_base_work = (pid_bh * (N + 2) + N + 1) * K * V
    # copy checkpoint to working
    for ki in range(NK):
        k_off = ki * 32 + offs_k
        mask2d = (k_off[:, None] < K) & (v_off[None, :] < V)
        s0 = tl.load(chk_ptr + chk_base_start + k_off[:, None] * V + v_off[None, :], mask=mask2d, other=0.0)
        tl.store(chk_ptr + chk_base_work + k_off[:, None] * V + v_off[None, :], s0, mask=mask2d)
    # time loop
    for ii in range(cur_T):
        t = t0 + ii
        beta = tl.load(beta_ptr + pid_bh * T + t).to(tl.float32)
        vt = tl.load(v_ptr + pid_bh * T * V + t * V + v_off, mask=mask_v, other=0.0).to(tl.float32)
        pred = tl.zeros((32,), dtype=tl.float32)
        out_acc = tl.zeros((32,), dtype=tl.float32)
        for ki in range(NK):
            k_off = ki * 32 + offs_k
            mask_k = k_off < K
            mask2d = (k_off[:, None] < K) & (v_off[None, :] < V)
            S = tl.load(chk_ptr + chk_base_work + k_off[:, None] * V + v_off[None, :], mask=mask2d, other=0.0)
            gv = tl.load(g_ptr + pid_bh * T * K + t * K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            d = tl.exp(gv)
            Sd = S * d[:, None]
            tl.store(chk_ptr + chk_base_work + k_off[:, None] * V + v_off[None, :], Sd, mask=mask2d)
            qv = tl.load(q_ptr + pid_bh * T * K + t * K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            kv = tl.load(k_ptr + pid_bh * T * K + t * K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            pred += tl.sum(Sd * kv[:, None], axis=0)
            out_acc += tl.sum(Sd * (qv * SCALE)[:, None], axis=0)
        tl.store(out_ptr + pid_bh * T * V + t * V + v_off, out_acc, mask=mask_v)
        delta = beta * (vt - pred)
        for ki in range(NK):
            k_off = ki * 32 + offs_k
            mask2d = (k_off[:, None] < K) & (v_off[None, :] < V)
            S = tl.load(chk_ptr + chk_base_work + k_off[:, None] * V + v_off[None, :], mask=mask2d, other=0.0)
            kv = tl.load(k_ptr + pid_bh * T * K + t * K + k_off, mask=k_off < K, other=0.0).to(tl.float32)
            S2 = S + kv[:, None] * delta[None, :]
            tl.store(chk_ptr + chk_base_work + k_off[:, None] * V + v_off[None, :], S2, mask=mask2d)
    # copy working to next checkpoint
    chk_base_next = (pid_bh * (N + 2) + chk_idx + 1) * K * V
    for ki in range(NK):
        k_off = ki * 32 + offs_k
        mask2d = (k_off[:, None] < K) & (v_off[None, :] < V)
        S = tl.load(chk_ptr + chk_base_work + k_off[:, None] * V + v_off[None, :], mask=mask2d, other=0.0)
        tl.store(chk_ptr + chk_base_next + k_off[:, None] * V + v_off[None, :], S, mask=mask2d)


@triton.jit
def scratch_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, chk_ptr, scr_ptr,
    BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    N: tl.constexpr, C: tl.constexpr, NK: tl.constexpr,
    t0: tl.constexpr, cur_T: tl.constexpr, chk_idx: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_k = tl.arange(0, 32)
    offs_v = tl.arange(0, 32)
    v_off = pid_v * 32 + offs_v
    chk_base_start = (pid_bh * (N + 2) + chk_idx) * K * V
    chk_base_work = (pid_bh * (N + 2) + N + 1) * K * V
    for ki in range(NK):
        k_off = ki * 32 + offs_k
        mask2d = (k_off[:, None] < K) & (v_off[None, :] < V)
        s0 = tl.load(chk_ptr + chk_base_start + k_off[:, None] * V + v_off[None, :], mask=mask2d, other=0.0)
        tl.store(chk_ptr + chk_base_work + k_off[:, None] * V + v_off[None, :], s0, mask=mask2d)
    for ii in range(cur_T):
        t = t0 + ii
        beta = tl.load(beta_ptr + pid_bh * T + t).to(tl.float32)
        vt = tl.load(v_ptr + pid_bh * T * V + t * V + v_off, mask=v_off < V, other=0.0).to(tl.float32)
        pred = tl.zeros((32,), dtype=tl.float32)
        for ki in range(NK):
            k_off = ki * 32 + offs_k
            mask2d = (k_off[:, None] < K) & (v_off[None, :] < V)
            S = tl.load(chk_ptr + chk_base_work + k_off[:, None] * V + v_off[None, :], mask=mask2d, other=0.0)
            gv = tl.load(g_ptr + pid_bh * T * K + t * K + k_off, mask=k_off < K, other=0.0).to(tl.float32)
            d = tl.exp(gv)
            Sd = S * d[:, None]
            tl.store(chk_ptr + chk_base_work + k_off[:, None] * V + v_off[None, :], Sd, mask=mask2d)
            tl.store(scr_ptr + (pid_bh * C + ii) * K * V + k_off[:, None] * V + v_off[None, :], Sd, mask=mask2d)
            kv = tl.load(k_ptr + pid_bh * T * K + t * K + k_off, mask=k_off < K, other=0.0).to(tl.float32)
            pred += tl.sum(Sd * kv[:, None], axis=0)
        delta = beta * (vt - pred)
        for ki in range(NK):
            k_off = ki * 32 + offs_k
            mask2d = (k_off[:, None] < K) & (v_off[None, :] < V)
            S = tl.load(chk_ptr + chk_base_work + k_off[:, None] * V + v_off[None, :], mask=mask2d, other=0.0)
            kv = tl.load(k_ptr + pid_bh * T * K + t * K + k_off, mask=k_off < K, other=0.0).to(tl.float32)
            S2 = S + kv[:, None] * delta[None, :]
            tl.store(chk_ptr + chk_base_work + k_off[:, None] * V + v_off[None, :], S2, mask=mask2d)


@triton.jit
def bwd_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, do_ptr, scr_ptr, gchk_ptr, dq_ptr, dk_ptr, dg_ptr, dv_ptr, dbeta_ptr,
    BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    N: tl.constexpr, C: tl.constexpr, NK: tl.constexpr, NV: tl.constexpr,
    SCALE: tl.constexpr, t0: tl.constexpr, cur_T: tl.constexpr, chk_idx: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_k = tl.arange(0, 32)
    offs_v = tl.arange(0, 32)
    v_off = pid_v * 32 + offs_v
    mask_v = v_off < V
    gchk_base_next = (pid_bh * (N + 2) + chk_idx + 1) * K * V
    gchk_base_work = (pid_bh * (N + 2) + N + 1) * K * V
    for ki in range(NK):
        k_off = ki * 32 + offs_k
        mask2d = (k_off[:, None] < K) & (v_off[None, :] < V)
        G = tl.load(gchk_ptr + gchk_base_next + k_off[:, None] * V + v_off[None, :], mask=mask2d, other=0.0)
        tl.store(gchk_ptr + gchk_base_work + k_off[:, None] * V + v_off[None, :], G, mask=mask2d)
    for ri in range(cur_T):
        ii = cur_T - 1 - ri
        t = t0 + ii
        do_tile = tl.load(do_ptr + pid_bh * T * V + t * V + v_off, mask=mask_v, other=0.0).to(tl.float32)
        vt = tl.load(v_ptr + pid_bh * T * V + t * V + v_off, mask=mask_v, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + pid_bh * T + t).to(tl.float32)
        # pred and u
        pred = tl.zeros((32,), dtype=tl.float32)
        u = tl.zeros((32,), dtype=tl.float32)
        for ki in range(NK):
            k_off = ki * 32 + offs_k
            Sd = tl.load(scr_ptr + (pid_bh * C + ii) * K * V + k_off[:, None] * V + v_off[None, :], mask=(k_off[:, None] < K) & (v_off[None, :] < V), other=0.0)
            G = tl.load(gchk_ptr + gchk_base_work + k_off[:, None] * V + v_off[None, :], mask=(k_off[:, None] < K) & (v_off[None, :] < V), other=0.0)
            kv = tl.load(k_ptr + pid_bh * T * K + t * K + k_off, mask=k_off < K, other=0.0).to(tl.float32)
            pred += tl.sum(Sd * kv[:, None], axis=0)
            u += tl.sum(G * kv[:, None], axis=0)
        vmp = vt - pred
        dv = beta * u
        tl.store(dv_ptr + pid_bh * T * V + t * V + v_off, dv, mask=mask_v)
        db = tl.sum(u * vmp, axis=0)
        tl.atomic_add(dbeta_ptr + pid_bh * T + t, db)
        for ki in range(NK):
            k_off = ki * 32 + offs_k
            mask_k = k_off < K
            mask2d = (k_off[:, None] < K) & (v_off[None, :] < V)
            Sd = tl.load(scr_ptr + (pid_bh * C + ii) * K * V + k_off[:, None] * V + v_off[None, :], mask=mask2d, other=0.0)
            G = tl.load(gchk_ptr + gchk_base_work + k_off[:, None] * V + v_off[None, :], mask=mask2d, other=0.0)
            qv = tl.load(q_ptr + pid_bh * T * K + t * K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            kv = tl.load(k_ptr + pid_bh * T * K + t * K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            gv = tl.load(g_ptr + pid_bh * T * K + t * K + k_off, mask=mask_k, other=0.0).to(tl.float32)
            d = tl.exp(gv)
            dq_part = SCALE * tl.sum(Sd * do_tile[None, :], axis=1)
            tl.atomic_add(dq_ptr + pid_bh * T * K + k_off, dq_part, mask=mask_k)
            dk_part = beta * (tl.sum(G * vmp[None, :], axis=1) - tl.sum(Sd * u[None, :], axis=1))
            tl.atomic_add(dk_ptr + pid_bh * T * K + k_off, dk_part, mask=mask_k)
            Graw = G + SCALE * qv[:, None] * do_tile[None, :] - beta * kv[:, None] * u[None, :]
            dg_part = tl.sum(Graw * Sd, axis=1)
            tl.atomic_add(dg_ptr + pid_bh * T * K + k_off, dg_part, mask=mask_k)
            Gnext = Graw * d[:, None]
            tl.store(gchk_ptr + gchk_base_work + k_off[:, None] * V + v_off[None, :], Gnext, mask=mask2d)
    # copy working to checkpoint
    gchk_base_cur = (pid_bh * (N + 2) + chk_idx) * K * V
    for ki in range(NK):
        k_off = ki * 32 + offs_k
        mask2d = (k_off[:, None] < K) & (v_off[None, :] < V)
        G = tl.load(gchk_ptr + gchk_base_work + k_off[:, None] * V + v_off[None, :], mask=mask2d, other=0.0)
        tl.store(gchk_ptr + gchk_base_cur + k_off[:, None] * V + v_off[None, :], G, mask=mask2d)


class GatedDeltaFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        device = q.device
        scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        g_c = g.contiguous()
        beta_c = beta.contiguous()
        qf = q_c.view(BH, T, K)
        kf = k_c.view(BH, T, K)
        vf = v_c.view(BH, T, V)
        gf = g_c.view(BH, T, K)
        bf = beta_c.view(BH, T)
        out = torch.empty(B, H, T, V, dtype=torch.float32, device=device)
        outf = out.view(BH, T, V)
        C = 64
        N = (T + C - 1) // C
        NK = (K + 31) // 32
        NV = (V + 31) // 32
        chk = torch.zeros(BH, N + 2, K, V, dtype=torch.float32, device=device)
        for chk_idx in range(N):
            t0 = chk_idx * C
            cur_T = C if t0 + C <= T else T - t0
            grid = (BH, NV)
            fwd_chunk_kernel[grid](qf, kf, vf, gf, bf, outf, chk, BH, T, K, V, N, C, NK, NV, scale_f, t0, cur_T, chk_idx)
        ctx.save_for_backward(qf, kf, vf, gf, bf)
        ctx.chk = chk
        ctx.BH = BH; ctx.B = B; ctx.H = H; ctx.T = T; ctx.K = K; ctx.V = V
        ctx.C = C; ctx.N = N; ctx.NK = NK; ctx.NV = NV; ctx.scale_f = scale_f
        ctx.q_dtype = q.dtype; ctx.k_dtype = k.dtype; ctx.v_dtype = v.dtype; ctx.e_dtype = e.dtype
        ctx.g_dtype = g.dtype; ctx.beta_dtype = beta.dtype; ctx.gamma_dtype = gamma.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        qf, kf, vf, gf, bf = ctx.saved_tensors
        BH, T, K, V, N, C, NK, NV, scale_f = ctx.BH, ctx.T, ctx.K, ctx.V, ctx.N, ctx.C, ctx.NK, ctx.NV, ctx.scale_f
        B, H = ctx.B, ctx.H
        device = grad_out.device
        go = grad_out.contiguous().view(BH, T, V)
        dq = torch.zeros(BH, T, K, dtype=torch.float32, device=device)
        dk = torch.zeros(BH, T, K, dtype=torch.float32, device=device)
        dg = torch.zeros(BH, T, K, dtype=torch.float32, device=device)
        dv = torch.zeros(BH, T, V, dtype=torch.float32, device=device)
        dbeta = torch.zeros(BH, T, dtype=torch.float32, device=device)
        gchk = torch.zeros(BH, N + 2, K, V, dtype=torch.float32, device=device)
        scr = torch.empty(BH, C, K, V, dtype=torch.float32, device=device)
        for chk_idx in range(N - 1, -1, -1):
            t0 = chk_idx * C
            cur_T = C if t0 + C <= T else T - t0
            grid = (BH, NV)
            scratch_kernel[grid](qf, kf, vf, gf, bf, ctx.chk, scr, BH, T, K, V, N, C, NK, t0, cur_T, chk_idx)
            bwd_kernel[grid](qf, kf, vf, gf, bf, go, scr, gchk, dq, dk, dg, dv, dbeta, BH, T, K, V, N, C, NK, NV, scale_f, t0, cur_T, chk_idx)
        dq_out = dq.view(B, H, T, K).to(ctx.q_dtype)
        dk_out = dk.view(B, H, T, K).to(ctx.k_dtype)
        dv_out = dv.view(B, H, T, V).to(ctx.v_dtype)
        dg_out = dg.view(B, H, T, K).to(ctx.g_dtype)
        dbeta_out = dbeta.view(B, H, T).to(ctx.beta_dtype)
        de = torch.zeros(B, H, T, K, dtype=ctx.e_dtype, device=device)
        dgamma = torch.zeros(B, H, T, dtype=ctx.gamma_dtype, device=device)
        return dq_out, dk_out, dv_out, de, dg_out, dbeta_out, dgamma, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return GatedDeltaFunc.apply(q, k, v, e, g, beta, gamma, scale)
```