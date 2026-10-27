import torch
import triton
import triton.language as tl


def _n2(x):
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


@triton.jit
def _fwd_bounds_kernel(
    q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, out_ptr, bounds_ptr,
    T, K, V, scale, C, Nc,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    EVEN_K: tl.constexpr, EVEN_V: tl.constexpr, NEED_TAIL: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    d_base = d_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    o_base = out_ptr + bh * T * V
    b_base = beta_ptr + bh * T

    for t in range(T):
        if EVEN_K:
            q_t = tl.load(q_base + t * K + k_offs).to(tl.float32)
            k_t = tl.load(k_base + t * K + k_offs).to(tl.float32)
            dec = tl.load(d_base + t * K + k_offs).to(tl.float32)
        else:
            q_t = tl.load(q_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            k_t = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            dec = tl.load(d_base + t * K + k_offs, mask=k_mask, other=1.0).to(tl.float32)
        if EVEN_V:
            v_t = tl.load(v_base + t * V + v_offs).to(tl.float32)
        else:
            v_t = tl.load(v_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
        b_t = tl.load(b_base + t).to(tl.float32)

        S = S * tl.expand_dims(dec, 1)
        q_scaled = q_t * scale
        out_t = tl.sum(S * tl.expand_dims(q_scaled, 1), 0)
        if EVEN_V:
            tl.store(o_base + t * V + v_offs, out_t)
        else:
            tl.store(o_base + t * V + v_offs, out_t, mask=v_mask)

        pred = tl.sum(S * tl.expand_dims(k_t, 1), 0)
        if EVEN_V:
            diff = v_t - pred
            b_diff = b_t * diff
        else:
            pred_m = tl.where(v_mask, pred, 0.0)
            diff = tl.where(v_mask, v_t - pred_m, 0.0)
            b_diff = b_t * diff
        S = S + tl.expand_dims(k_t, 1) * tl.expand_dims(b_diff, 0)

        if (t + 1) % C == 0:
            c = t // C
            base = bounds_ptr + (bh * Nc + c) * K * V
            ptrs = base + tl.expand_dims(k_offs, 1) * V + tl.expand_dims(v_offs, 0)
            m2d = tl.expand_dims(k_mask, 1) & tl.expand_dims(v_mask, 0)
            tl.store(ptrs, S, mask=m2d)

    if NEED_TAIL:
        base = bounds_ptr + (bh * Nc + Nc - 1) * K * V
        ptrs = base + tl.expand_dims(k_offs, 1) * V + tl.expand_dims(v_offs, 0)
        m2d = tl.expand_dims(k_mask, 1) & tl.expand_dims(v_mask, 0)
        tl.store(ptrs, S, mask=m2d)


@triton.jit
def _bwd_chunk_kernel(
    q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, do_ptr,
    bounds_ptr, ds_ptr, scratch_ptr,
    dq_part_ptr, dk_part_ptr, dg_part_ptr, dv_ptr, dbeta_part_ptr,
    T, K, V, scale, chunk_start, Lc, chunk_idx, Nc, C, nV,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, IS_FIRST: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    k_offs = tl.arange(0, BLOCK_K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V
    m2d = tl.expand_dims(k_mask, 1) & tl.expand_dims(v_mask, 0)

    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    if IS_FIRST:
        pass
    else:
        base_b = bounds_ptr + (bh * Nc + (chunk_idx - 1)) * K * V
        bptrs = base_b + tl.expand_dims(k_offs, 1) * V + tl.expand_dims(v_offs, 0)
        S = tl.load(bptrs, mask=m2d, other=0.0).to(tl.float32)

    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    d_base = d_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    do_base = do_ptr + bh * T * V
    b_base = beta_ptr + bh * T
    scr_base = scratch_ptr + bh * C * K * V

    for ii in range(Lc):
        t = chunk_start + ii
        k_t = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        d_t = tl.load(d_base + t * K + k_offs, mask=k_mask, other=1.0).to(tl.float32)
        v_t = tl.load(v_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
        b_t = tl.load(b_base + t).to(tl.float32)
        S = S * tl.expand_dims(d_t, 1)
        sptrs = scr_base + ii * K * V + tl.expand_dims(k_offs, 1) * V + tl.expand_dims(v_offs, 0)
        tl.store(sptrs, S, mask=m2d)
        pred = tl.sum(S * tl.expand_dims(k_t, 1), 0)
        pred = tl.where(v_mask, pred, 0.0)
        b_diff = b_t * tl.where(v_mask, v_t - pred, 0.0)
        S = S + tl.expand_dims(k_t, 1) * tl.expand_dims(b_diff, 0)

    ds_base = ds_ptr + bh * K * V
    ds_ptrs = ds_base + tl.expand_dims(k_offs, 1) * V + tl.expand_dims(v_offs, 0)
    ds = tl.load(ds_ptrs, mask=m2d, other=0.0).to(tl.float32)

    part_base = (bh * nV + pid_v) * T * K
    dq_pb = dq_part_ptr + part_base
    dk_pb = dk_part_ptr + part_base
    dg_pb = dg_part_ptr + part_base
    dv_base = dv_ptr + bh * T * V
    db_pb = dbeta_part_ptr + (bh * nV + pid_v) * T

    for jj in range(Lc):
        ii = Lc - 1 - jj
        t = chunk_start + ii
        sptrs = scr_base + ii * K * V + tl.expand_dims(k_offs, 1) * V + tl.expand_dims(v_offs, 0)
        sdec = tl.load(sptrs, mask=m2d, other=0.0).to(tl.float32)

        q_t = tl.load(q_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        k_t = tl.load(k_base + t * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        d_t = tl.load(d_base + t * K + k_offs, mask=k_mask, other=1.0).to(tl.float32)
        v_t = tl.load(v_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
        do_t = tl.load(do_base + t * V + v_offs, mask=v_mask, other=0.0).to(tl.float32)
        b_t = tl.load(b_base + t).to(tl.float32)

        pred = tl.where(v_mask, tl.sum(sdec * tl.expand_dims(k_t, 1), 0), 0.0)
        u = tl.where(v_mask, tl.sum(ds * tl.expand_dims(k_t, 1), 0), 0.0)
        vmp = tl.where(v_mask, v_t - pred, 0.0)

        tl.store(dv_base + t * V + v_offs, b_t * u, mask=v_mask)

        G = ds + tl.expand_dims(q_t * scale, 1) * tl.expand_dims(do_t, 0) - b_t * tl.expand_dims(k_t, 1) * tl.expand_dims(u, 0)

        pdq = scale * tl.sum(sdec * tl.expand_dims(do_t, 0), 1)
        pdg = tl.sum(G * sdec, 1)
        A = tl.sum(ds * tl.expand_dims(vmp, 0), 1)
        Bb = tl.sum(sdec * tl.expand_dims(u, 0), 1)
        pdk = b_t * (A - Bb)
        pdb = tl.sum(u * vmp, 0)

        tl.store(dq_pb + t * K + k_offs, pdq, mask=k_mask)
        tl.store(dk_pb + t * K + k_offs, pdk, mask=k_mask)
        tl.store(dg_pb + t * K + k_offs, pdg, mask=k_mask)
        tl.store(db_pb + t, pdb)

        ds = G * tl.expand_dims(d_t, 1)

    tl.store(ds_ptrs, ds, mask=m2d)


class _RecFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale_f):
        B, H, T, Kd = q.shape
        Vd = v.shape[-1]
        ctx.B, ctx.H, ctx.T, ctx.Kd, ctx.Vd = B, H, T, Kd, Vd
        ctx.scale_f = float(scale_f) if not isinstance(scale_f, float) else scale_f
        ctx.e_shape, ctx.e_dtype, ctx.e_device = e.shape, e.dtype, e.device
        ctx.gamma_shape, ctx.gamma_dtype, ctx.gamma_device = gamma.shape, gamma.dtype, gamma.device
        ctx.q_dtype, ctx.k_dtype, ctx.v_dtype = q.dtype, k.dtype, v.dtype
        ctx.g_dtype, ctx.beta_dtype = g.dtype, beta.dtype
        if T == 0 or Kd == 0 or Vd == 0 or B == 0 or H == 0:
            ctx.BH = 0
            ctx.Nc = 0
            ctx.C = 1
            ctx.BLOCK_K = 16
            ctx.BLOCK_VF = 32
            ctx.BLOCK_VB = 32
            ctx.nVf = 0
            ctx.nVb = 0
            ctx.EVEN_K = True
            ctx.EVEN_V = True
            return torch.empty(B, H, T, Vd, device=q.device, dtype=torch.float32)
        qc = q.contiguous()
        kc = k.contiguous()
        vc = v.contiguous()
        gc = g.contiguous()
        bc = beta.contiguous()
        d = torch.exp(gc)
        if not d.is_contiguous():
            d = d.contiguous()
        BH = B * H
        C = 64 if T > 64 else T
        Nc = (T + C - 1) // C
        NEED_TAIL = (T % C != 0)
        BLOCK_K = _n2(Kd)
        if BLOCK_K < 16:
            BLOCK_K = 16
        if Vd <= 32:
            BLOCK_VF = _n2(Vd)
            if BLOCK_VF < 16:
                BLOCK_VF = 16
        else:
            BLOCK_VF = 32
        EVEN_K = (Kd == BLOCK_K)
        EVEN_V = (Vd % BLOCK_VF == 0)
        nVf = (Vd + BLOCK_VF - 1) // BLOCK_VF
        if BLOCK_K > 128:
            BLOCK_VB = 16
        else:
            BLOCK_VB = BLOCK_VF
        nVb = (Vd + BLOCK_VB - 1) // BLOCK_VB
        num_warps = 8 if BLOCK_K * BLOCK_VF > 2048 else 4
        qc3 = qc.reshape(BH, T, Kd)
        kc3 = kc.reshape(BH, T, Kd)
        dc3 = d.reshape(BH, T, Kd)
        vc3 = vc.reshape(BH, T, Vd)
        bc2 = bc.reshape(BH, T)
        if not qc3.is_contiguous():
            qc3 = qc3.contiguous()
        if not kc3.is_contiguous():
            kc3 = kc3.contiguous()
        if not dc3.is_contiguous():
            dc3 = dc3.contiguous()
        if not vc3.is_contiguous():
            vc3 = vc3.contiguous()
        if not bc2.is_contiguous():
            bc2 = bc2.contiguous()
        out3 = torch.empty(BH, T, Vd, device=q.device, dtype=torch.float32)
        bounds = torch.empty(BH, Nc, Kd, Vd, device=q.device, dtype=torch.float32)
        _fwd_bounds_kernel[(BH, nVf)](
            qc3, kc3, vc3, dc3, bc2, out3, bounds,
            T, Kd, Vd, ctx.scale_f, C, Nc,
            BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_VF,
            EVEN_K=EVEN_K, EVEN_V=EVEN_V, NEED_TAIL=NEED_TAIL,
            num_warps=num_warps, num_stages=2,
        )
        ctx.BH, ctx.Nc, ctx.C = BH, Nc, C
        ctx.BLOCK_K, ctx.BLOCK_VF, ctx.BLOCK_VB = BLOCK_K, BLOCK_VF, BLOCK_VB
        ctx.nVf, ctx.nVb = nVf, nVb
        ctx.EVEN_K, ctx.EVEN_V = EVEN_K, EVEN_V
        ctx.num_warps = num_warps
        ctx.save_for_backward(qc3, kc3, vc3, dc3, bc2, bounds)
        return out3.reshape(B, H, T, Vd)

    @staticmethod
    def backward(ctx, grad_output):
        B, H, T, Kd, Vd = ctx.B, ctx.H, ctx.T, ctx.Kd, ctx.Vd
        if T == 0 or Kd == 0 or Vd == 0 or B == 0 or H == 0 or ctx.BH == 0:
            return (None, None, None, None, None, None, None, None)
        BH, Nc, C = ctx.BH, ctx.Nc, ctx.C
        BLOCK_K, BLOCK_VB, nVb = ctx.BLOCK_K, ctx.BLOCK_VB, ctx.nVb
        scale_f = ctx.scale_f
        qc3, kc3, vc3, dc3, bc2, bounds = ctx.saved_tensors
        goc = grad_output.contiguous().reshape(BH, T, Vd).to(torch.float32)
        if not goc.is_contiguous():
            goc = goc.contiguous()
        dq_part = torch.empty(BH, nVb, T, Kd, dtype=torch.float32, device=qc3.device)
        dk_part = torch.empty(BH, nVb, T, Kd, dtype=torch.float32, device=qc3.device)
        dg_part = torch.empty(BH, nVb, T, Kd, dtype=torch.float32, device=qc3.device)
        dbeta_part = torch.empty(BH, nVb, T, dtype=torch.float32, device=qc3.device)
        dv = torch.empty(BH, T, Vd, dtype=torch.float32, device=qc3.device)
        ds = torch.zeros(BH, Kd, Vd, dtype=torch.float32, device=qc3.device)
        scratch = torch.empty(BH, C, Kd, Vd, dtype=torch.float32, device=qc3.device)
        num_warps_b = 4 if BLOCK_K * BLOCK_VB <= 2048 else 8
        for chunk_idx in range(Nc - 1, -1, -1):
            chunk_start = chunk_idx * C
            Lc = T - chunk_start if chunk_start + C > T else C
            IS_FIRST = (chunk_idx == 0)
            _bwd_chunk_kernel[(BH, nVb)](
                qc3, kc3, vc3, dc3, bc2, goc,
                bounds, ds, scratch,
                dq_part, dk_part, dg_part, dv, dbeta_part,
                T, Kd, Vd, scale_f,
                chunk_start, Lc, chunk_idx, Nc, C, nVb,
                BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_VB, IS_FIRST=IS_FIRST,
                num_warps=num_warps_b, num_stages=1,
            )
        dq = dq_part.sum(dim=1).reshape(B, H, T, Kd).to(ctx.q_dtype)
        dk = dk_part.sum(dim=1).reshape(B, H, T, Kd).to(ctx.k_dtype)
        dg_ = dg_part.sum(dim=1).reshape(B, H, T, Kd).to(ctx.g_dtype)
        dv_out = dv.reshape(B, H, T, Vd).to(ctx.v_dtype)
        dbeta_out = dbeta_part.sum(dim=1).reshape(B, H, T).to(ctx.beta_dtype)
        e_out = torch.zeros(ctx.e_shape, dtype=ctx.e_dtype, device=ctx.e_device)
        gamma_out = torch.zeros(ctx.gamma_shape, dtype=ctx.gamma_dtype, device=ctx.gamma_device)
        return (dq, dk, dv_out, e_out, dg_, dbeta_out, gamma_out, None)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)
    return _RecFunc.apply(q, k, v, e, g, beta, gamma, scale_f)
