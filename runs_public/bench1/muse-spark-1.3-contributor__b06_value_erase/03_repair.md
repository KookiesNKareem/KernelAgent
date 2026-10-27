```python
import torch
import triton
import triton.language as tl


@triton.jit
def _scan_fast128(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, b_ptr, c_ptr, o_ptr, T, scale):
    pid = tl.program_id(0)
    off = tl.arange(0, 128)
    q_base = q_ptr + pid * T * 128
    k_base = k_ptr + pid * T * 128
    e_base = e_ptr + pid * T * 128
    g_base = g_ptr + pid * T * 128
    v_base = v_ptr + pid * T * 128
    b_base = b_ptr + pid * T
    c_base = c_ptr + pid * T
    o_base = o_ptr + pid * T * 128

    S = tl.zeros([128, 128], dtype=tl.bfloat16)
    for t in range(T):
        q = tl.load(q_base + t * 128 + off)
        k = tl.load(k_base + t * 128 + off)
        e = tl.load(e_base + t * 128 + off)
        v = tl.load(v_base + t * 128 + off)
        g = tl.load(g_base + t * 128 + off)
        b = tl.load(b_base + t)
        c = tl.load(c_base + t)
        d = tl.exp(g)

        Sf0 = tl.cast(S, tl.float32)
        S1_f = Sf0 * d[:, None]

        e_f = tl.cast(e, tl.float32)
        u = tl.sum(S1_f * e_f[None, :], axis=1)
        S1_b = tl.cast(S1_f, tl.bfloat16)
        c_b = tl.cast(c, tl.bfloat16)
        u_b = tl.cast(u, tl.bfloat16)
        S2_b = tl.cast(S1_b - c_b * u_b[:, None] * e[None, :], tl.bfloat16)

        S2_f = tl.cast(S2_b, tl.float32)
        k_f = tl.cast(k, tl.float32)
        pred = tl.sum(S2_f * k_f[:, None], axis=0)
        v_f = tl.cast(v, tl.float32)
        diff_f = v_f - pred
        diff_b = tl.cast(diff_f, tl.bfloat16)
        b_b = tl.cast(b, tl.bfloat16)
        S3_b = tl.cast(S2_b + b_b * k[:, None] * diff_b[None, :], tl.bfloat16)

        S3_f = tl.cast(S3_b, tl.float32)
        q_f = tl.cast(q, tl.float32)
        o = tl.sum(S3_f * q_f[:, None], axis=0) * scale
        tl.store(o_base + t * 128 + off, o)
        S = S3_b


@triton.jit
def _scan_fp32_128_nodecay(q_ptr, k_ptr, v_ptr, e_ptr, b_ptr, c_ptr, o_ptr, T, scale):
    pid = tl.program_id(0)
    off = tl.arange(0, 128)
    q_base = q_ptr + pid * T * 128
    k_base = k_ptr + pid * T * 128
    e_base = e_ptr + pid * T * 128
    v_base = v_ptr + pid * T * 128
    b_base = b_ptr + pid * T
    c_base = c_ptr + pid * T
    o_base = o_ptr + pid * T * 128
    S = tl.zeros([128, 128], dtype=tl.float32)
    for t in range(T):
        q = tl.load(q_base + t * 128 + off).to(tl.float32)
        k = tl.load(k_base + t * 128 + off).to(tl.float32)
        e = tl.load(e_base + t * 128 + off).to(tl.float32)
        v = tl.load(v_base + t * 128 + off).to(tl.float32)
        b = tl.load(b_base + t)
        c = tl.load(c_base + t)
        u = tl.sum(S * e[None, :], axis=1)
        S = S - c * u[:, None] * e[None, :]
        pred = tl.sum(S * k[:, None], axis=0)
        diff = v - pred
        S = S + b * k[:, None] * diff[None, :]
        o = tl.sum(S * q[:, None], axis=0) * scale
        tl.store(o_base + t * 128 + off, o)


@triton.jit
def _scan_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, d_ptr, beta_ptr, gamma_ptr, out_ptr,
    T, K, V,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    off_k = tl.arange(0, BLOCK_K)
    off_v = tl.arange(0, BLOCK_V)
    mask_k = off_k < K
    mask_v = off_v < V
    mask_ev = off_v < K
    q_base = q_ptr + pid * T * K
    k_base = k_ptr + pid * T * K
    e_base = e_ptr + pid * T * K
    d_base = d_ptr + pid * T * K
    v_base = v_ptr + pid * T * V
    b_base = beta_ptr + pid * T
    c_base = gamma_ptr + pid * T
    o_base = out_ptr + pid * T * V
    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    for t in range(T):
        q_t = tl.load(q_base + t * K + off_k, mask=mask_k, other=0.0)
        k_t = tl.load(k_base + t * K + off_k, mask=mask_k, other=0.0)
        d_t = tl.load(d_base + t * K + off_k, mask=mask_k, other=1.0)
        e_t = tl.load(e_base + t * K + off_v, mask=mask_ev, other=0.0)
        v_t = tl.load(v_base + t * V + off_v, mask=mask_v, other=0.0)
        b_t = tl.load(b_base + t)
        c_t = tl.load(c_base + t)
        S = S * d_t[:, None]
        u_t = tl.sum(S * e_t[None, :], axis=1)
        S = S - c_t * u_t[:, None] * e_t[None, :]
        pred_t = tl.sum(S * k_t[:, None], axis=0)
        diff_t = v_t - pred_t
        S = S + b_t * k_t[:, None] * diff_t[None, :]
        o_t = tl.sum(S * q_t[:, None], axis=0)
        tl.store(o_base + t * V + off_v, o_t, mask=mask_v)


def _npo2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    if torch.is_tensor(scale):
        scale_f = float(scale.reshape(-1)[0].to(torch.float32).item())
    else:
        scale_f = float(scale)
    BH = B * H
    if BH == 0 or T == 0 or K == 0 or V == 0:
        return torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    if K == 128 and V == 128:
        # decay_none violates the bounded-state assumption of the bf16 fast
        # path: with g==0 (d==1) there is no forgetting, S accumulates over the
        # whole sequence and bf16 state quantization dominates. Use fp32.
        # Since g<=0, all-zero <=> no non-zero entries.
        try:
            no_decay = not bool(g.any().item())
        except Exception:
            no_decay = False
        if no_decay:
            q_c = q.contiguous()
            k_c = k.contiguous()
            v_c = v.contiguous()
            e_c = e.contiguous()
            b_c = beta.contiguous()
            c_c = gamma.contiguous()
            out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
            _scan_fp32_128_nodecay[(BH,)](q_c, k_c, v_c, e_c, b_c, c_c, out, T, scale_f, num_warps=8)
            return out
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        e_c = e.contiguous()
        g_c = g.contiguous()
        b_c = beta.contiguous()
        c_c = gamma.contiguous()
        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        _scan_fast128[(BH,)](q_c, k_c, v_c, e_c, g_c, b_c, c_c, out, T, scale_f, num_warps=8)
        return out

    q32 = (q.to(torch.float32) * scale_f).contiguous()
    k32 = k.to(torch.float32).contiguous()
    v32 = v.to(torch.float32).contiguous()
    e32 = e.to(torch.float32).contiguous()
    d32 = torch.exp(g.to(torch.float32)).contiguous()
    b32 = beta.to(torch.float32).contiguous()
    c32 = gamma.to(torch.float32).contiguous()
    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    BLOCK_K = max(16, _npo2(K))
    BLOCK_V = max(16, _npo2(V))
    _scan_kernel[(BH,)](q32, k32, v32, e32, d32, b32, c32, out, T, K, V,
                         BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, num_warps=8)
    return out
```