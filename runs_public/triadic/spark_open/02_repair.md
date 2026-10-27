```python
import torch
import triton
import triton.language as tl

@triton.jit
def gated_delta_kernel(
    q_ptr, k_ptr, v_ptr, q2_ptr, k2_ptr, g_ptr, beta_ptr, out_ptr, s_ptr,
    stride_q_bh, stride_q_t,
    stride_k_bh, stride_k_t,
    stride_v_bh, stride_v_t,
    stride_q2_bh, stride_q2_t,
    stride_k2_bh, stride_k2_t,
    stride_g_bh, stride_g_t,
    stride_beta_bh, stride_beta_t,
    stride_out_bh, stride_out_t,
    BH, T, K, E, V,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, NK: tl.constexpr, N_SHARDS: tl.constexpr, SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // N_SHARDS
    shard = pid % N_SHARDS
    v_off = shard * BLOCK_V

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_v = (v_off + offs_v) < V

    # pointers to start of this bh (for S)
    s_base = s_ptr + bh * K * V

    for t in range(T):
        # scalar beta_t (fp32)
        beta_t = tl.load(beta_ptr + bh * stride_beta_bh + t * stride_beta_t)
        v_tile = tl.load(v_ptr + bh * stride_v_bh + t * stride_v_t + v_off + offs_v, mask=mask_v, other=0.0).to(tl.float32)

        q_base = q_ptr + bh * stride_q_bh + t * stride_q_t
        k_base = k_ptr + bh * stride_k_bh + t * stride_k_t
        q2_base = q2_ptr + bh * stride_q2_bh + t * stride_q2_t
        k2_base = k2_ptr + bh * stride_k2_bh + t * stride_k2_t
        g_base = g_ptr + bh * stride_g_bh + t * stride_g_t

        # ---- first pass: decay S and compute pred = sum_k S_decay * kappa ----
        pred = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for kb in range(NK):
            k_off = kb * BLOCK_K
            abs_k = k_off + offs_k              # [BLOCK_K]
            mask_k = abs_k < K
            e_idx = abs_k % E
            i_idx = abs_k // E

            k2v = tl.load(k2_base + e_idx, mask=mask_k, other=0.0).to(tl.float32)
            gv = tl.load(g_base + e_idx, mask=mask_k, other=0.0)
            kv = tl.load(k_base + i_idx, mask=mask_k, other=0.0).to(tl.float32)
            decay = tl.exp(gv)                  # [BLOCK_K]
            kappa = kv * k2v                    # [BLOCK_K]

            # S tile [BLOCK_K, BLOCK_V]
            ptrs = s_base + tl.expand_dims(abs_k, 1) * V + tl.expand_dims(offs_v, 0) + tl.expand_dims(v_off, 0) * 0
            # above is equivalent to s_base + abs_k[:,None]*V + v_off+offs_v
            # use explicit 2D ptrs for masked load
            S_ptrs = s_base + tl.expand_dims(abs_k, 1) * V + tl.expand_dims(v_off + offs_v, 0)
            mask_2d = tl.expand_dims(mask_k, 1) & tl.expand_dims(mask_v, 0)
            S_tile = tl.load(S_ptrs, mask=mask_2d, other=0.0)
            S_tile = S_tile * tl.expand_dims(decay, 1)
            tl.store(S_ptrs, S_tile, mask=mask_2d)
            pred = pred + tl.sum(S_tile * tl.expand_dims(kappa, 1), axis=0)

        err = beta_t * (v_tile - pred)          # [BLOCK_V]

        # ---- second pass: S += kappa*err , accumulate output ----
        out_tile = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for kb in range(NK):
            k_off = kb * BLOCK_K
            abs_k = k_off + offs_k
            mask_k = abs_k < K
            e_idx = abs_k % E
            i_idx = abs_k // E

            k2v = tl.load(k2_base + e_idx, mask=mask_k, other=0.0).to(tl.float32)
            q2v = tl.load(q2_base + e_idx, mask=mask_k, other=0.0).to(tl.float32)
            kv = tl.load(k_base + i_idx, mask=mask_k, other=0.0).to(tl.float32)
            qv = tl.load(q_base + i_idx, mask=mask_k, other=0.0).to(tl.float32)
            kappa = kv * k2v
            qq = qv * q2v * SCALE

            S_ptrs = s_base + tl.expand_dims(abs_k, 1) * V + tl.expand_dims(v_off + offs_v, 0)
            mask_2d = tl.expand_dims(mask_k, 1) & tl.expand_dims(mask_v, 0)
            S_tile = tl.load(S_ptrs, mask=mask_2d, other=0.0)
            S_tile = S_tile + tl.expand_dims(kappa, 1) * tl.expand_dims(err, 0)
            tl.store(S_ptrs, S_tile, mask=mask_2d)
            out_tile = out_tile + tl.sum(S_tile * tl.expand_dims(qq, 1), axis=0)

        tl.store(out_ptr + bh * stride_out_bh + t * stride_out_t + v_off + offs_v, out_tile, mask=mask_v)


def _reference_fallback(q, k, v, q2, k2, g, beta, scale):
    B, H, T, d = q.shape
    nE = k2.shape[-1]
    q, k, v, q2, k2, g, beta = (x.to(torch.promote_types(x.dtype, torch.float32)) for x in (q, k, v, q2, k2, g, beta))
    S = q.new_zeros(B, H, d * nE, v.shape[-1])
    outs = []
    for t in range(T):
        kap = (k[:, :, t, :, None] * k2[:, :, t, None, :]).flatten(-2)
        qq = (q[:, :, t, :, None] * q2[:, :, t, None, :]).flatten(-2)
        decay = g[:, :, t, None, :].exp().expand(B, H, d, nE).flatten(-2)
        S = S * decay[..., None]
        pred = (S * kap[..., None]).sum(-2)
        S = S + (beta[:, :, t][..., None] * (v[:, :, t] - pred))[..., None, :] * kap[..., None]
        outs.append((S * (qq * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)


def kernel_function(q, k, v, q2, k2, g, beta, scale):
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    if not q.is_cuda:
        return _reference_fallback(q, k, v, q2, k2, g, beta, scale)

    B, H, T, d = q.shape
    V = v.shape[-1]
    E = q2.shape[-1]
    K = d * E
    BH = B * H

    # collapse B*H and make contiguous to handle any strides/transposes
    q_ = q.reshape(BH, T, d).contiguous()
    k_ = k.reshape(BH, T, d).contiguous()
    v_ = v.reshape(BH, T, V).contiguous()
    q2_ = q2.reshape(BH, T, E).contiguous()
    k2_ = k2.reshape(BH, T, E).contiguous()
    g_ = g.reshape(BH, T, E).contiguous()
    beta_ = beta.reshape(BH, T).contiguous()

    out_ = torch.empty(BH, T, V, device=q.device, dtype=torch.float32)
    s_buf = torch.zeros(BH * K * V, device=q.device, dtype=torch.float32)

    BLOCK_K = 32
    BLOCK_V = 32
    NK = (K + BLOCK_K - 1) // BLOCK_K
    N_SHARDS = (V + BLOCK_V - 1) // BLOCK_V

    grid = (BH * N_SHARDS,)

    gated_delta_kernel[grid](
        q_, k_, v_, q2_, k2_, g_, beta_, out_, s_buf,
        q_.stride(0), q_.stride(1),
        k_.stride(0), k_.stride(1),
        v_.stride(0), v_.stride(1),
        q2_.stride(0), q2_.stride(1),
        k2_.stride(0), k2_.stride(1),
        g_.stride(0), g_.stride(1),
        beta_.stride(0), beta_.stride(1),
        out_.stride(0), out_.stride(1),
        BH, T, K, E, V,
        BLOCK_K, BLOCK_V, NK, N_SHARDS, float(scale),
    )
    return out_.view(B, H, T, V).to(torch.float32)
```

The previous kernel assumed fixed `d=128,E=4,K=512,V=128` with `BLOCK_K=64,NK=8` and 8 unrolled `S0..S7` registers, so any other `K,V` or `H` (which changes `K=d*E` in the tests) read/wrote out-of-bounds and omitted tail masking, and used `offs_k%E` without `k_off` offset. The fix makes `K=d*E` and `V` fully generic: `S` is kept in HBM, `K` and `V` are tiled with `BLOCK_K/V=32`, `NK/N_SHARDS=ceil`, arbitrary `E` via `abs_k%E`/`abs_k//E`, and all loads/stores are masked. I am unsure whether `tl.exp`/`tl.load(...).to(tl.float32)` spelling and `for t in range(T)` with a runtime `T` constexpr are accepted by Triton 3.5 on sm_89, and whether the chosen `BLOCK_K/V` tuning preserves the required speedup.