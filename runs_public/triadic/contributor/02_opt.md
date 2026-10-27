```python
import torch
import triton
import triton.language as tl


@triton.jit
def _fast_fwd(
    q_ptr, k_ptr, v_ptr, q2_ptr, k2_ptr, dec_ptr, beta_ptr, out_ptr,
    T, D, E, V, BH, NV,
    scale,
    BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr, EVEN_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    kd = tl.arange(0, BLOCK_D)
    vd = vid * BLOCK_V + tl.arange(0, BLOCK_V)

    S0 = tl.zeros((BLOCK_D, BLOCK_V), dtype=tl.float32)
    S1 = tl.zeros((BLOCK_D, BLOCK_V), dtype=tl.float32)
    S2 = tl.zeros((BLOCK_D, BLOCK_V), dtype=tl.float32)
    S3 = tl.zeros((BLOCK_D, BLOCK_V), dtype=tl.float32)

    q_bh = bh * T * D
    k_bh = bh * T * D
    v_bh = bh * T * V
    q2_bh = bh * T * E
    k2_bh = bh * T * E
    dec_bh = bh * T * E
    b_bh = bh * T
    o_bh = bh * T * V

    for t in range(T):
        k = tl.load(k_ptr + k_bh + t * D + kd).to(tl.float32)
        q = tl.load(q_ptr + q_bh + t * D + kd).to(tl.float32)
        if EVEN_V:
            vv = tl.load(v_ptr + v_bh + t * V + vd).to(tl.float32)
        else:
            m = vd < V
            vv = tl.load(v_ptr + v_bh + t * V + vd, mask=m, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + b_bh + t).to(tl.float32)

        k2b = k2_ptr + k2_bh + t * E
        q2b = q2_ptr + q2_bh + t * E
        db = dec_ptr + dec_bh + t * E
        k20 = tl.load(k2b + 0).to(tl.float32)
        k21 = tl.load(k2b + 1).to(tl.float32)
        k22 = tl.load(k2b + 2).to(tl.float32)
        k23 = tl.load(k2b + 3).to(tl.float32)
        q20 = tl.load(q2b + 0).to(tl.float32)
        q21 = tl.load(q2b + 1).to(tl.float32)
        q22 = tl.load(q2b + 2).to(tl.float32)
        q23 = tl.load(q2b + 3).to(tl.float32)
        d0 = tl.load(db + 0).to(tl.float32)
        d1 = tl.load(db + 1).to(tl.float32)
        d2 = tl.load(db + 2).to(tl.float32)
        d3 = tl.load(db + 3).to(tl.float32)

        kq = tl.sum(k * q)
        k2q2 = k20 * q20 + k21 * q21 + k22 * q22 + k23 * q23
        c_raw = kq * k2q2

        k20d = k20 * d0
        k21d = k21 * d1
        k22d = k22 * d2
        k23d = k23 * d3
        q20d = q20 * d0
        q21d = q21 * d1
        q22d = q22 * d2
        q23d = q23 * d3

        ks0r = k * k20
        ks1r = k * k21
        ks2r = k * k22
        ks3r = k * k23
        ks0e = k * k20d
        ks1e = k * k21d
        ks2e = k * k22d
        ks3e = k * k23d
        qs0e = q * q20d
        qs1e = q * q21d
        qs2e = q * q22d
        qs3e = q * q23d

        p0 = tl.sum(S0 * tl.expand_dims(ks0e, 1), axis=0)
        p1 = tl.sum(S1 * tl.expand_dims(ks1e, 1), axis=0)
        p2 = tl.sum(S2 * tl.expand_dims(ks2e, 1), axis=0)
        p3 = tl.sum(S3 * tl.expand_dims(ks3e, 1), axis=0)
        pred = p0 + p1 + p2 + p3

        o0 = tl.sum(S0 * tl.expand_dims(qs0e, 1), axis=0)
        o1 = tl.sum(S1 * tl.expand_dims(qs1e, 1), axis=0)
        o2 = tl.sum(S2 * tl.expand_dims(qs2e, 1), axis=0)
        o3 = tl.sum(S3 * tl.expand_dims(qs3e, 1), axis=0)
        odec = o0 + o1 + o2 + o3

        delta = (vv - pred) * beta_t
        outv = (odec + c_raw * delta) * scale

        if EVEN_V:
            tl.store(out_ptr + o_bh + t * V + vd, outv)
        else:
            m = vd < V
            tl.store(out_ptr + o_bh + t * V + vd, outv, mask=m)

        de = tl.expand_dims(delta, 0)
        S0 = S0 * d0 + tl.expand_dims(ks0r, 1) * de
        S1 = S1 * d1 + tl.expand_dims(ks1r, 1) * de
        S2 = S2 * d2 + tl.expand_dims(ks2r, 1) * de
        S3 = S3 * d3 + tl.expand_dims(ks3r, 1) * de


@triton.jit
def _generic_fwd(
    q_ptr, k_ptr, v_ptr, q2_ptr, k2_ptr, dec_ptr, beta_ptr, out_ptr,
    T, D, E, V, BH, NV,
    scale,
    BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    vid = pid % NV
    d_idx = tl.arange(0, BLOCK_D)
    e_idx = tl.arange(0, BLOCK_E)
    v_idx = vid * BLOCK_V + tl.arange(0, BLOCK_V)
    d_mask = d_idx < D
    e_mask = e_idx < E
    v_mask = v_idx < V
    S = tl.zeros((BLOCK_D, BLOCK_E, BLOCK_V), dtype=tl.float32)
    for t in range(T):
        k_d = tl.load(k_ptr + bh * T * D + t * D + d_idx, mask=d_mask, other=0.0).to(tl.float32)
        q_d = tl.load(q_ptr + bh * T * D + t * D + d_idx, mask=d_mask, other=0.0).to(tl.float32)
        k2_e = tl.load(k2_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=0.0).to(tl.float32)
        q2_e = tl.load(q2_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=0.0).to(tl.float32)
        dec_e = tl.load(dec_ptr + bh * T * E + t * E + e_idx, mask=e_mask, other=1.0).to(tl.float32)
        beta_t = tl.load(beta_ptr + bh * T + t).to(tl.float32)
        v_v = tl.load(v_ptr + bh * T * V + t * V + v_idx, mask=v_mask, other=0.0).to(tl.float32)
        k_3 = tl.expand_dims(tl.expand_dims(k_d, 1), 2)
        q_3 = tl.expand_dims(tl.expand_dims(q_d, 1), 2)
        k2_3 = tl.expand_dims(tl.expand_dims(k2_e, 0), 2)
        q2_3 = tl.expand_dims(tl.expand_dims(q2_e, 0), 2)
        dec_3 = tl.expand_dims(tl.expand_dims(dec_e, 0), 2)
        S = S * dec_3
        tmp_p = S * k_3 * k2_3
        pred = tl.sum(tl.sum(tmp_p, axis=0), axis=0)
        delta = (v_v - pred) * beta_t
        S = S + k_3 * k2_3 * tl.expand_dims(tl.expand_dims(delta, 0), 0)
        tmp_o = S * q_3 * q2_3
        out_v = tl.sum(tl.sum(tmp_o, axis=0), axis=0) * scale
        tl.store(out_ptr + bh * T * V + t * V + v_idx, out_v, mask=v_mask)


def _np2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def kernel_function(q, k, v, q2, k2, g, beta, scale):
    B, H, T, D = q.shape
    E = q2.shape[-1]
    Vd = v.shape[-1]
    BH = B * H
    q3 = q.contiguous().view(BH, T, D)
    k3 = k.contiguous().view(BH, T, D)
    v3 = v.contiguous().view(BH, T, Vd)
    q23 = q2.contiguous().view(BH, T, E)
    k23 = k2.contiguous().view(BH, T, E)
    dec = torch.exp(g.to(torch.float32)).contiguous().view(BH, T, E)
    b3 = beta.contiguous().view(BH, T)
    out3 = torch.empty((BH, T, Vd), device=q.device, dtype=torch.float32)
    scale_f = float(scale.item()) if torch.is_tensor(scale) else float(scale)
    if D == 128 and E == 4:
        BLOCK_D = 128
        BLOCK_V = 32
        NV = (Vd + BLOCK_V - 1) // BLOCK_V
        grid = (BH * NV,)
        _fast_fwd[grid](
            q3, k3, v3, q23, k23, dec, b3, out3,
            T, D, E, Vd, BH, NV, scale_f,
            BLOCK_D=BLOCK_D, BLOCK_V=BLOCK_V, EVEN_V=(Vd % BLOCK_V == 0),
            num_warps=8, num_stages=2,
        )
    else:
        BLOCK_D = _np2(D)
        BLOCK_E = _np2(E)
        BLOCK_V = 16
        NV = (Vd + BLOCK_V - 1) // BLOCK_V
        grid = (BH * NV,)
        _generic_fwd[grid](
            q3, k3, v3, q23, k23, dec, b3, out3,
            T, D, E, Vd, BH, NV, scale_f,
            BLOCK_D=BLOCK_D, BLOCK_E=BLOCK_E, BLOCK_V=BLOCK_V,
            num_warps=4, num_stages=1,
        )
    return out3.view(B, H, T, Vd)
```