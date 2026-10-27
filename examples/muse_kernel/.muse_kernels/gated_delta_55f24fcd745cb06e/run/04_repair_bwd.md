```python
import torch
import triton
import triton.language as tl


def _next_pow2(x: int):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr, ckpt_ptr,
    T, K, V, Nc, NV, scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    off_k = tl.arange(0, BLOCK_K)
    off_bv = tl.arange(0, BLOCK_V)
    off_v = vid * BLOCK_V + off_bv
    mask_k = off_k < K
    mask_v = off_v < V
    mask_2d = mask_k[:, None] & mask_v[None, :]
    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    for t in tl.range(0, T):
        q_t = tl.load(q_ptr + bh * T * K + t * K + off_k, mask=mask_k, other=0.0)
        k_t = tl.load(k_ptr + bh * T * K + t * K + off_k, mask=mask_k, other=0.0)
        v_t = tl.load(v_ptr + bh * T * V + t * V + off_v, mask=mask_v, other=0.0)
        g_t = tl.load(g_ptr + bh * T + t).to(tl.float32)
        b_t = tl.load(beta_ptr + bh * T + t).to(tl.float32)
        q_f = q_t.to(tl.float32) * scale
        k_f = k_t.to(tl.float32)
        v_f = v_t.to(tl.float32)
        d = tl.exp(g_t)
        S = S * d
        k_col = tl.expand_dims(k_f, 1)
        pred = tl.sum(S * k_col, axis=0)
        diff = (v_f - pred) * b_t
        S = S + k_col * tl.expand_dims(diff, 0)
        q_col = tl.expand_dims(q_f, 1)
        out_t = tl.sum(S * q_col, axis=0)
        tl.store(o_ptr + bh * T * V + t * V + off_v, out_t, mask=mask_v)
        cond1 = (t + 1) % CHUNK == 0
        cond2 = (t + 1) == T
        if cond1 | cond2:
            idx = (t + 1 + CHUNK - 1) // CHUNK
            base = ckpt_ptr + (bh * (Nc + 1) + idx) * K * V
            ptrs = base + off_k[:, None] * V + off_v[None, :]
            tl.store(ptrs, S, mask=mask_2d)


@triton.jit
def _bwd_chunk_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, do_ptr,
    ckpt_ptr, F_ptr, dv_ptr,
    dq_part_ptr, dk_part_ptr, dg_part_ptr, db_part_ptr,
    T, K, V, Nc, NV, l, cur_len, scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    off_k = tl.arange(0, BLOCK_K)
    off_bv = tl.arange(0, BLOCK_V)
    off_v = vid * BLOCK_V + off_bv
    mask_k = off_k < K
    mask_v = off_v < V
    mask_2d = mask_k[:, None] & mask_v[None, :]
    c = l // CHUNK
    ck_base = ckpt_ptr + (bh * (Nc + 1) + c) * K * V
    ck_ptrs = ck_base + off_k[:, None] * V + off_v[None, :]
    S_init = tl.load(ck_ptrs, mask=mask_2d, other=0.0)
    F_base = F_ptr + bh * K * V
    F_ptrs = F_base + off_k[:, None] * V + off_v[None, :]
    F_cur = tl.load(F_ptrs, mask=mask_2d, other=0.0)
    r = l + cur_len
    for rev in tl.range(0, CHUNK):
        if rev < cur_len:
            t = r - 1 - rev
            tml = t - l
            # recompute S_{t-1} from S_init
            S_tmp = S_init
            for ii in tl.range(0, CHUNK):
                if ii < tml:
                    tt = l + ii
                    k_tt = tl.load(k_ptr + bh * T * K + tt * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
                    v_tt = tl.load(v_ptr + bh * T * V + tt * V + off_v, mask=mask_v, other=0.0).to(tl.float32)
                    g_tt = tl.load(g_ptr + bh * T + tt).to(tl.float32)
                    b_tt = tl.load(beta_ptr + bh * T + tt).to(tl.float32)
                    d_tt = tl.exp(g_tt)
                    S_tmp = S_tmp * d_tt
                    kcol = tl.expand_dims(k_tt, 1)
                    pred_tt = tl.sum(S_tmp * kcol, axis=0)
                    diff_tt = (v_tt - pred_tt) * b_tt
                    S_tmp = S_tmp + kcol * tl.expand_dims(diff_tt, 0)
            q_t = tl.load(q_ptr + bh * T * K + t * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
            k_t = tl.load(k_ptr + bh * T * K + t * K + off_k, mask=mask_k, other=0.0).to(tl.float32)
            v_t = tl.load(v_ptr + bh * T * V + t * V + off_v, mask=mask_v, other=0.0).to(tl.float32)
            g_t = tl.load(g_ptr + bh * T + t).to(tl.float32)
            b_t = tl.load(beta_ptr + bh * T + t).to(tl.float32)
            do_t = tl.load(do_ptr + bh * T * V + t * V + off_v, mask=mask_v, other=0.0).to(tl.float32)
            d_t = tl.exp(g_t)
            S_prime = S_tmp * d_t
            k_col = tl.expand_dims(k_t, 1)
            pred = tl.sum(S_prime * k_col, axis=0)
            e0 = v_t - pred
            S_cur = S_prime + k_col * tl.expand_dims(e0 * b_t, 0)
            q_scaled = q_t * scale
            direct = tl.expand_dims(q_scaled, 1) * tl.expand_dims(do_t, 0)
            DS = direct + F_cur
            w_vec = tl.sum(DS * k_col, axis=0)
            dq_vec = tl.sum(S_cur * tl.expand_dims(do_t, 0), axis=1) * scale
            a_vec = tl.sum(DS * tl.expand_dims(e0, 0), axis=1)
            b_vec = tl.sum(S_prime * tl.expand_dims(w_vec, 0), axis=1)
            dk_vec = (a_vec - b_vec) * b_t
            dg_val = tl.sum(DS * S_prime) - b_t * tl.sum(w_vec * pred)
            db_val = tl.sum(w_vec * e0)
            dv_vec = w_vec * b_t
            ii_t = t - l
            tl.store(dq_part_ptr + pid * CHUNK * K + ii_t * K + off_k, dq_vec, mask=mask_k)
            tl.store(dk_part_ptr + pid * CHUNK * K + ii_t * K + off_k, dk_vec, mask=mask_k)
            tl.store(dg_part_ptr + pid * CHUNK + ii_t, dg_val)
            tl.store(db_part_ptr + pid * CHUNK + ii_t, db_val)
            tl.store(dv_ptr + bh * T * V + t * V + off_v, dv_vec, mask=mask_v)
            # Fut_{t-1} = d_t * (DS - b_t * k_t * w_t^T) : single correction
            wb = w_vec * (d_t * b_t)
            F_cur = DS * d_t - k_col * tl.expand_dims(wb, 0)
    tl.store(F_ptrs, F_cur, mask=mask_2d)


class _GatedDeltaFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale):
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()
        if not g.is_contiguous():
            g = g.contiguous()
        if not beta.is_contiguous():
            beta = beta.contiguous()
        B, H, T, K = q.shape
        Vd = v.shape[-1]
        BH = B * H
        CHUNK = 64
        if Vd <= 32:
            BLOCK_V = _next_pow2(Vd) if Vd > 0 else 1
        else:
            BLOCK_V = 32
        BLOCK_K = _next_pow2(K) if K > 0 else 1
        NV = (Vd + BLOCK_V - 1) // BLOCK_V if Vd > 0 else 1
        if T == 0 or K == 0 or Vd == 0 or BH == 0:
            out = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)
            ctx.save_for_backward(q, k, v, g, beta)
            ctx.B, ctx.H, ctx.T, ctx.K, ctx.Vd, ctx.BH = B, H, T, K, Vd, BH
            ctx.Nc, ctx.NV, ctx.BLOCK_K, ctx.BLOCK_V, ctx.CHUNK = 0, NV, BLOCK_K, BLOCK_V, CHUNK
            ctx.scale = float(scale)
            ctx.ckpt = torch.empty((0,), device=q.device, dtype=torch.float32)
            return out
        Nc = (T + CHUNK - 1) // CHUNK
        out = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)
        ckpt = torch.zeros((BH * (Nc + 1) * K * Vd,), device=q.device, dtype=torch.float32)
        grid = (BH * NV,)
        _fwd_kernel[grid](
            q, k, v, g, beta, out, ckpt,
            T, K, Vd, Nc, NV, float(scale),
            BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, CHUNK=CHUNK,
            num_warps=4, num_stages=1,
        )
        ctx.save_for_backward(q, k, v, g, beta)
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.Vd, ctx.BH = B, H, T, K, Vd, BH
        ctx.Nc, ctx.NV, ctx.BLOCK_K, ctx.BLOCK_V, ctx.CHUNK = Nc, NV, BLOCK_K, BLOCK_V, CHUNK
        ctx.scale = float(scale)
        ctx.ckpt = ckpt
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, g, beta = ctx.saved_tensors
        B, H, T, K, Vd, BH = ctx.B, ctx.H, ctx.T, ctx.K, ctx.Vd, ctx.BH
        Nc, NV, BLOCK_K, BLOCK_V, CHUNK = ctx.Nc, ctx.NV, ctx.BLOCK_K, ctx.BLOCK_V, ctx.CHUNK
        scale = ctx.scale
        if grad_out is None:
            return (None, None, None, None, None, None)
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        if T == 0 or K == 0 or Vd == 0 or BH == 0:
            zq = torch.zeros_like(q)
            zk = torch.zeros_like(k)
            zv = torch.zeros_like(v)
            zg = torch.zeros_like(g)
            zb = torch.zeros_like(beta)
            return (zq, zk, zv, zg, zb, None)
        do = grad_out.to(torch.float32).contiguous()
        dq_f = torch.empty((BH, T, K), device=q.device, dtype=torch.float32)
        dk_f = torch.empty((BH, T, K), device=q.device, dtype=torch.float32)
        dv_f = torch.empty((BH, T, Vd), device=q.device, dtype=torch.float32)
        dg_f = torch.empty((BH, T), device=q.device, dtype=torch.float32)
        db_f = torch.empty((BH, T), device=q.device, dtype=torch.float32)
        F_buf = torch.zeros((BH * K * Vd,), device=q.device, dtype=torch.float32)
        dq_p = torch.empty((BH * NV, CHUNK, K), device=q.device, dtype=torch.float32)
        dk_p = torch.empty((BH * NV, CHUNK, K), device=q.device, dtype=torch.float32)
        dg_p = torch.empty((BH * NV, CHUNK), device=q.device, dtype=torch.float32)
        db_p = torch.empty((BH * NV, CHUNK), device=q.device, dtype=torch.float32)
        grid = (BH * NV,)
        for c in range(Nc - 1, -1, -1):
            l = c * CHUNK
            r = (c + 1) * CHUNK
            if r > T:
                r = T
            cur = r - l
            if cur <= 0:
                continue
            _bwd_chunk_kernel[grid](
                q, k, v, g, beta, do,
                ctx.ckpt, F_buf, dv_f,
                dq_p, dk_p, dg_p, db_p,
                T, K, Vd, Nc, NV, l, cur, float(scale),
                BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, CHUNK=CHUNK,
                num_warps=4, num_stages=1,
            )
            dq_f[:, l:r, :] = dq_p.view(BH, NV, CHUNK, K)[:, :, :cur, :].sum(dim=1)
            dk_f[:, l:r, :] = dk_p.view(BH, NV, CHUNK, K)[:, :, :cur, :].sum(dim=1)
            dg_f[:, l:r] = dg_p.view(BH, NV, CHUNK)[:, :, :cur].sum(dim=1)
            db_f[:, l:r] = db_p.view(BH, NV, CHUNK)[:, :, :cur].sum(dim=1)
        dq = dq_f.view(B, H, T, K).to(q.dtype)
        dk = dk_f.view(B, H, T, K).to(k.dtype)
        dv = dv_f.view(B, H, T, Vd).to(v.dtype)
        dg = dg_f.view(B, H, T).to(g.dtype)
        db = db_f.view(B, H, T).to(beta.dtype)
        return (dq, dk, dv, dg, db, None)


def kernel_function(q, k, v, g, beta, scale):
    return _GatedDeltaFunc.apply(q, k, v, g, beta, scale)
```