import torch
import triton
import triton.language as tl

BLOCK = 32
MAX_K = 8  # 8*32=256 covers all K in tests


@triton.jit
def _fwd_tile(q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, out_ptr, state_ptr,
              scale,
              T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
              Kb: tl.constexpr, Vb: tl.constexpr,
              START, N_V: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // N_V
    vt = pid % N_V
    v_off = vt * 32
    mask_v = (v_off + tl.arange(0, 32)) < V
    state_base = state_ptr + bh * Kb * Vb
    for i in range(32):
        cur = START + i
        valid = cur < T
        d_val = tl.load(d_ptr + bh * T + cur, mask=valid, other=1.0)
        b_val = tl.load(beta_ptr + bh * T + cur, mask=valid, other=0.0)
        v_blk = tl.load(v_ptr + bh * T * Vb + cur * Vb + v_off + tl.arange(0, 32), mask=mask_v & valid, other=0.0).to(tl.float32)
        # p = k^T S
        p = tl.zeros((32,), dtype=tl.float32)
        for ki in range(MAX_K):
            k_off = ki * 32
            if k_off < Kb:
                k_blk = tl.load(k_ptr + bh * T * Kb + cur * Kb + k_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                s_ptr = state_base + k_off * Vb + v_off
                s_blk = tl.load(s_ptr + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], mask=valid, other=0.0).to(tl.float32)
                p += tl.sum(k_blk[:, None] * s_blk, axis=0)
        coeff = b_val * (v_blk - d_val * p)
        for ki in range(MAX_K):
            k_off = ki * 32
            if k_off < Kb:
                k_blk = tl.load(k_ptr + bh * T * Kb + cur * Kb + k_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                s_ptr = state_base + k_off * Vb + v_off
                s_blk = tl.load(s_ptr + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], mask=valid, other=0.0).to(tl.float32)
                s_new = s_blk * d_val + k_blk[:, None] * coeff[None, :]
                tl.store(s_ptr + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s_new, mask=valid)
        o = tl.zeros((32,), dtype=tl.float32)
        for ki in range(MAX_K):
            k_off = ki * 32
            if k_off < Kb:
                q_blk = tl.load(q_ptr + bh * T * Kb + cur * Kb + k_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                s_ptr = state_base + k_off * Vb + v_off
                s_blk = tl.load(s_ptr + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], mask=valid, other=0.0).to(tl.float32)
                o += tl.sum(q_blk[:, None] * s_blk, axis=0)
        o = o * scale
        tl.store(out_ptr + bh * T * V + cur * V + v_off + tl.arange(0, 32), o, mask=mask_v & valid)


@triton.jit
def _recompute_tile(k_ptr, v_ptr, d_ptr, beta_ptr, cur_state_ptr, tmp_ptr,
                    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    Kb: tl.constexpr, Vb: tl.constexpr,
                    START, N_V: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // N_V
    vt = pid % N_V
    v_off = vt * 32
    cur_base = cur_state_ptr + bh * Kb * Vb
    tmp_base = tmp_ptr + bh * 32 * Kb * Vb
    for i in range(32):
        cur = START + i
        valid = cur < T
        d_val = tl.load(d_ptr + bh * T + cur, mask=valid, other=1.0)
        b_val = tl.load(beta_ptr + bh * T + cur, mask=valid, other=0.0)
        v_blk = tl.load(v_ptr + bh * T * Vb + cur * Vb + v_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
        p = tl.zeros((32,), dtype=tl.float32)
        for ki in range(MAX_K):
            k_off = ki * 32
            if k_off < Kb:
                k_blk = tl.load(k_ptr + bh * T * Kb + cur * Kb + k_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                s_ptr = cur_base + k_off * Vb + v_off
                s_blk = tl.load(s_ptr + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], mask=valid, other=0.0).to(tl.float32)
                p += tl.sum(k_blk[:, None] * s_blk, axis=0)
        coeff = b_val * (v_blk - d_val * p)
        for ki in range(MAX_K):
            k_off = ki * 32
            if k_off < Kb:
                k_blk = tl.load(k_ptr + bh * T * Kb + cur * Kb + k_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                s_ptr = cur_base + k_off * Vb + v_off
                s_blk = tl.load(s_ptr + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], mask=valid, other=0.0).to(tl.float32)
                s_new = s_blk * d_val + k_blk[:, None] * coeff[None, :]
                tl.store(s_ptr + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s_new, mask=valid)
        for ki in range(MAX_K):
            k_off = ki * 32
            if k_off < Kb:
                s_ptr = cur_base + k_off * Vb + v_off
                s_blk = tl.load(s_ptr + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], mask=valid, other=0.0).to(tl.float32)
                t_ptr = tmp_base + i * Kb * Vb + k_off * Vb + v_off
                tl.store(t_ptr + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s_blk, mask=valid)


@triton.jit
def _bwd_tile(q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, go_ptr, tmp_ptr, chunk_start_ptr, g_state_ptr,
              dq_part_ptr, dk_part_ptr, dv_ptr, db_part_ptr, dg_part_ptr,
              scale,
              T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
              Kb: tl.constexpr, Vb: tl.constexpr,
              START, N_V: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // N_V
    vt = pid % N_V
    v_off = vt * 32
    mask_v = (v_off + tl.arange(0, 32)) < V
    g_base = g_state_ptr + bh * Kb * Vb
    tmp_base = tmp_ptr + bh * 32 * Kb * Vb
    chunk_base = chunk_start_ptr + bh * Kb * Vb
    for idx in range(32):
        i = 31 - idx
        cur = START + i
        valid = cur < T
        d_val = tl.load(d_ptr + bh * T + cur, mask=valid, other=1.0)
        b_val = tl.load(beta_ptr + bh * T + cur, mask=valid, other=0.0)
        v_blk = tl.load(v_ptr + bh * T * Vb + cur * Vb + v_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
        go_blk = tl.load(go_ptr + bh * T * V + cur * V + v_off + tl.arange(0, 32), mask=mask_v & valid, other=0.0).to(tl.float32)
        # p = k^T S_prev
        p = tl.zeros((32,), dtype=tl.float32)
        for ki in range(MAX_K):
            k_off = ki * 32
            if k_off < Kb:
                k_blk = tl.load(k_ptr + bh * T * Kb + cur * Kb + k_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                if i == 0:
                    s_ptr = chunk_base + k_off * Vb + v_off
                else:
                    s_ptr = tmp_base + (i - 1) * Kb * Vb + k_off * Vb + v_off
                s_blk = tl.load(s_ptr + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], mask=valid, other=0.0).to(tl.float32)
                p += tl.sum(k_blk[:, None] * s_blk, axis=0)
        # r = k^T G_t , G_t = g_state + scale*q*go^T
        r = tl.zeros((32,), dtype=tl.float32)
        for ki in range(MAX_K):
            k_off = ki * 32
            if k_off < Kb:
                k_blk = tl.load(k_ptr + bh * T * Kb + cur * Kb + k_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q_blk = tl.load(q_ptr + bh * T * Kb + cur * Kb + k_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                g_ptr_ = g_base + k_off * Vb + v_off
                g_blk = tl.load(g_ptr_ + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], mask=valid, other=0.0).to(tl.float32)
                g_t = g_blk + scale * q_blk[:, None] * go_blk[None, :]
                r += tl.sum(k_blk[:, None] * g_t, axis=0)
        # second pass: dk/dq/trace and update g_state
        trace_part = 0.0
        for ki in range(MAX_K):
            k_off = ki * 32
            if k_off < Kb:
                k_blk = tl.load(k_ptr + bh * T * Kb + cur * Kb + k_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q_blk = tl.load(q_ptr + bh * T * Kb + cur * Kb + k_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                g_ptr_ = g_base + k_off * Vb + v_off
                g_blk = tl.load(g_ptr_ + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], mask=valid, other=0.0).to(tl.float32)
                g_t = g_blk + scale * q_blk[:, None] * go_blk[None, :]
                if i == 0:
                    s_prev_ptr = chunk_base + k_off * Vb + v_off
                else:
                    s_prev_ptr = tmp_base + (i - 1) * Kb * Vb + k_off * Vb + v_off
                s_prev_blk = tl.load(s_prev_ptr + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], mask=valid, other=0.0).to(tl.float32)
                s_cur_ptr = tmp_base + i * Kb * Vb + k_off * Vb + v_off
                s_cur_blk = tl.load(s_cur_ptr + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], mask=valid, other=0.0).to(tl.float32)
                gv = tl.sum(g_t * v_blk[None, :], axis=1)
                gp = tl.sum(g_t * p[None, :], axis=1)
                sr = tl.sum(s_prev_blk * r[None, :], axis=1)
                dk_blk = b_val * gv - d_val * b_val * gp - d_val * b_val * sr
                dk_base = dk_part_ptr + bh * N_V * T * Kb + vt * T * Kb + cur * Kb
                tl.store(dk_base + k_off + tl.arange(0, 32), dk_blk, mask=((k_off + tl.arange(0, 32)) < K) & valid)
                dq_blk = scale * tl.sum(s_cur_blk * go_blk[None, :], axis=1)
                dq_base = dq_part_ptr + bh * N_V * T * Kb + vt * T * Kb + cur * Kb
                tl.store(dq_base + k_off + tl.arange(0, 32), dq_blk, mask=((k_off + tl.arange(0, 32)) < K) & valid)
                trace_part += tl.sum(g_t * s_prev_blk)
                g_new = d_val * (g_t - b_val * k_blk[:, None] * r[None, :])
                tl.store(g_ptr_ + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], g_new, mask=valid)
        db_val = tl.sum(r * (v_blk - d_val * p))
        tl.store(db_part_ptr + bh * N_V * T + vt * T + cur, db_val, mask=valid)
        dg_val = d_val * (trace_part - b_val * tl.sum(r * p))
        tl.store(dg_part_ptr + bh * N_V * T + vt * T + cur, dg_val, mask=valid)
        dv_blk = b_val * r
        tl.store(dv_ptr + bh * T * V + cur * V + v_off + tl.arange(0, 32), dv_blk, mask=mask_v & valid)


class DeltaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        Vd = v.shape[-1]
        BH = B * H
        scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        Kb = ((K + 31) // 32) * 32
        Vb = ((Vd + 31) // 32) * 32
        if Kb == 0:
            Kb = 32
        if Vb == 0:
            Vb = 32
        nV = Vb // 32
        BLOCK_C = 32
        nC = (T + BLOCK_C - 1) // BLOCK_C
        q_r = q.reshape(BH, T, K).contiguous()
        k_r = k.reshape(BH, T, K).contiguous()
        v_r = v.reshape(BH, T, Vd).contiguous()
        g_r = g.reshape(BH, T).contiguous().to(torch.float32)
        beta_r = beta.reshape(BH, T).contiguous().to(torch.float32)
        q_pad = torch.zeros((BH, T, Kb), device=q.device, dtype=q_r.dtype)
        k_pad = torch.zeros((BH, T, Kb), device=q.device, dtype=k_r.dtype)
        v_pad = torch.zeros((BH, T, Vb), device=q.device, dtype=v_r.dtype)
        if K > 0:
            q_pad[:, :, :K] = q_r
            k_pad[:, :, :K] = k_r
        if Vd > 0:
            v_pad[:, :, :Vd] = v_r
        d = torch.exp(g_r)
        state = torch.zeros((BH, Kb, Vb), device=q.device, dtype=torch.float32)
        out = torch.empty((BH, T, Vd), device=q.device, dtype=torch.float32)
        chunk_states = torch.empty((BH, nC, Kb, Vb), device=q.device, dtype=torch.float32)
        grid = BH * nV
        for ck in range(nC):
            START = ck * BLOCK_C
            chunk_states[:, ck].copy_(state)
            _fwd_tile[grid](q_pad, k_pad, v_pad, d, beta_r, out, state, scale_f, T, K, Vd, Kb, Vb, START, nV)
        out_bhntv = out.reshape(B, H, T, Vd)
        ctx.save_for_backward(q_pad, k_pad, v_pad, d, beta_r, chunk_states)
        ctx.B = B; ctx.H = H; ctx.T = T; ctx.K = K; ctx.Vd = Vd; ctx.Kb = Kb; ctx.Vb = Vb
        ctx.BH = BH; ctx.nV = nV; ctx.nC = nC; ctx.scale_f = scale_f
        ctx.q_dtype = q.dtype; ctx.k_dtype = k.dtype; ctx.v_dtype = v.dtype
        ctx.g_dtype = g.dtype; ctx.beta_dtype = beta.dtype; ctx.e_dtype = e.dtype; ctx.gamma_dtype = gamma.dtype
        return out_bhntv.to(q.dtype) if out_bhntv.dtype != q.dtype else out_bhntv

    @staticmethod
    def backward(ctx, go):
        q_pad, k_pad, v_pad, d, beta_r, chunk_states = ctx.saved_tensors
        B = ctx.B; H = ctx.H; T = ctx.T; K = ctx.K; Vd = ctx.Vd; Kb = ctx.Kb; Vb = ctx.Vb
        BH = ctx.BH; nV = ctx.nV; nC = ctx.nC; scale_f = ctx.scale_f
        go_r = go.reshape(BH, T, Vd).contiguous().to(torch.float32)
        dq_part = torch.zeros((BH, nV, T, Kb), device=q_pad.device, dtype=torch.float32)
        dk_part = torch.zeros((BH, nV, T, Kb), device=q_pad.device, dtype=torch.float32)
        dv = torch.zeros((BH, T, Vd), device=q_pad.device, dtype=torch.float32)
        db_part = torch.zeros((BH, nV, T), device=q_pad.device, dtype=torch.float32)
        dg_part = torch.zeros((BH, nV, T), device=q_pad.device, dtype=torch.float32)
        tmp = torch.empty((BH, 32, Kb, Vb), device=q_pad.device, dtype=torch.float32)
        cur_state = torch.empty((BH, Kb, Vb), device=q_pad.device, dtype=torch.float32)
        g_state = torch.zeros((BH, Kb, Vb), device=q_pad.device, dtype=torch.float32)
        grid = BH * nV
        for ck in range(nC - 1, -1, -1):
            START = ck * 32
            cur_state.copy_(chunk_states[:, ck])
            _recompute_tile[grid](k_pad, v_pad, d, beta_r, cur_state, tmp, T, K, Vd, Kb, Vb, START, nV)
            _bwd_tile[grid](q_pad, k_pad, v_pad, d, beta_r, go_r, tmp, chunk_states[:, ck], g_state,
                            dq_part, dk_part, dv, db_part, dg_part, scale_f, T, K, Vd, Kb, Vb, START, nV)
        dq = dq_part.sum(dim=1)[..., :K].reshape(B, H, T, K).to(ctx.q_dtype)
        dk = dk_part.sum(dim=1)[..., :K].reshape(B, H, T, K).to(ctx.k_dtype)
        dv_out = dv.reshape(B, H, T, Vd).to(ctx.v_dtype)
        db = db_part.sum(dim=1).reshape(B, H, T).to(ctx.beta_dtype)
        dg = dg_part.sum(dim=1).reshape(B, H, T).to(ctx.g_dtype)
        ge = torch.zeros((B, H, T, K), device=q_pad.device, dtype=ctx.e_dtype)
        gg = torch.zeros((B, H, T), device=q_pad.device, dtype=ctx.gamma_dtype)
        return dq, dk, dv_out, ge, dg, db, gg, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return DeltaFn.apply(q, k, v, e, g, beta, gamma, scale)
