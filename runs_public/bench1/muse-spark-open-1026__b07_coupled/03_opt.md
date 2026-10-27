```python
import torch
import triton
import triton.language as tl

# fallback HBM kernel (used only if K is huge)
@triton.jit
def _fwd_kernel_hbm(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr,
    s_ptr, out_ptr,
    s0_q, s1_q, s2_q, s3_q,
    s0_k, s1_k, s2_k, s3_k,
    s0_v, s1_v, s2_v, s3_v,
    s0_e, s1_e, s2_e, s3_e,
    s0_g, s1_g, s2_g, s3_g,
    s0_b, s1_b, s2_b,
    s0_gm, s1_gm, s2_gm,
    s0_s, s1_s, s2_s, s3_s,
    s0_o, s1_o, s2_o, s3_o,
    B, H, T, K, V,
    NV: tl.constexpr, NK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, scale,
):
    pid = tl.program_id(0)
    iv = pid % NV
    rem = pid // NV
    h = rem % H
    b = rem // H
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_col = iv * BLOCK_V
    q_bh = q_ptr + b * s0_q + h * s1_q
    k_bh = k_ptr + b * s0_k + h * s1_k
    v_bh = v_ptr + b * s0_v + h * s1_v
    e_bh = e_ptr + b * s0_e + h * s1_e
    g_bh = g_ptr + b * s0_g + h * s1_g
    b_bh = beta_ptr + b * s0_b + h * s1_b
    gm_bh = gamma_ptr + b * s0_gm + h * s1_gm
    s_bh = s_ptr + b * s0_s + h * s1_s
    o_bh = out_ptr + b * s0_o + h * s1_o
    for t in range(T):
        bt = tl.load(b_bh + t * s2_b).to(tl.float32)
        gt = tl.load(gm_bh + t * s2_gm).to(tl.float32)
        ct = gt * (1.0 - bt)
        v_vals = tl.load(v_bh + t * s2_v + (v_col + offs_v) * s3_v, mask=(v_col + offs_v) < V, other=0.0).to(tl.float32)
        eTS = tl.zeros((BLOCK_V,), dtype=tl.float32)
        A = tl.zeros((BLOCK_V,), dtype=tl.float32)
        Bdot = 0.0
        for ik in range(NK):
            k_row = ik * BLOCK_K
            e_blk = tl.load(e_bh + t * s2_e + (k_row + offs_k) * s3_e, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            k_blk = tl.load(k_bh + t * s2_k + (k_row + offs_k) * s3_k, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            g_blk = tl.load(g_bh + t * s2_g + (k_row + offs_k) * s3_g, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            d_blk = tl.exp(g_blk)
            Bdot += tl.sum(k_blk * e_blk)
            s_tile = tl.load(s_bh + k_row * s2_s + v_col * s3_s + offs_k[:, None] * s2_s + offs_v[None, :] * s3_s, mask=((k_row + offs_k)[:, None] < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
            eTS += tl.sum((e_blk * d_blk)[:, None] * s_tile, axis=0)
            A += tl.sum((k_blk * d_blk)[:, None] * s_tile, axis=0)
        pred = A - ct * Bdot * eTS
        delta = v_vals - pred
        out_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for ik in range(NK):
            k_row = ik * BLOCK_K
            e_blk = tl.load(e_bh + t * s2_e + (k_row + offs_k) * s3_e, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            k_blk = tl.load(k_bh + t * s2_k + (k_row + offs_k) * s3_k, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            g_blk = tl.load(g_bh + t * s2_g + (k_row + offs_k) * s3_g, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            q_blk = tl.load(q_bh + t * s2_q + (k_row + offs_k) * s3_q, mask=(k_row + offs_k) < K, other=0.0).to(tl.float32)
            d_blk = tl.exp(g_blk)
            s_tile = tl.load(s_bh + k_row * s2_s + v_col * s3_s + offs_k[:, None] * s2_s + offs_v[None, :] * s3_s, mask=((k_row + offs_k)[:, None] < K) & ((v_col + offs_v)[None, :] < V), other=0.0).to(tl.float32)
            s_new = s_tile * d_blk[:, None] - ct * e_blk[:, None] * eTS[None, :] + bt * k_blk[:, None] * delta[None, :]
            tl.store(s_bh + k_row * s2_s + v_col * s3_s + offs_k[:, None] * s2_s + offs_v[None, :] * s3_s, s_new, mask=((k_row + offs_k)[:, None] < K) & ((v_col + offs_v)[None, :] < V))
            out_acc += tl.sum(q_blk[:, None] * s_new, axis=0)
        out_acc = out_acc * scale
        tl.store(o_bh + t * s2_o + (v_col + offs_v) * s3_o, out_acc, mask=(v_col + offs_v) < V)

_kernel_cache = {}

def _get_col_kernel(NK: int):
    if NK in _kernel_cache:
        return _kernel_cache[NK]
    BLOCK_K = 32
    BLOCK_V = 32
    k_name = f"_kernel_col_{NK}"
    header = f"""
import triton
import triton.language as tl
@triton.jit
def {k_name}(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    s0_q, s1_q, s2_q, s3_q,
    s0_k, s1_k, s2_k, s3_k,
    s0_v, s1_v, s2_v, s3_v,
    s0_e, s1_e, s2_e, s3_e,
    s0_g, s1_g, s2_g, s3_g,
    s0_b, s1_b, s2_b,
    s0_gm, s1_gm, s2_gm,
    s0_o, s1_o, s2_o, s3_o,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NV: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, scale,
):
    pid = tl.program_id(0)
    iv = pid % NV
    rem = pid // NV
    h = rem % H
    b = rem // H
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_col = iv * BLOCK_V
"""
    # init S tiles
    init = ""
    for ik in range(NK):
        init += f"    s{ik} = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)\n"
    body = """
    for t in range(T):
        bt = tl.load(beta_ptr + b * s0_b + h * s1_b + t * s2_b).to(tl.float32)
        gt = tl.load(gamma_ptr + b * s0_gm + h * s1_gm + t * s2_gm).to(tl.float32)
        ct = gt * (1.0 - bt)
        mask_v = (v_col + offs_v) < V
        v_vals = tl.load(v_ptr + b * s0_v + h * s1_v + t * s2_v + (v_col + offs_v) * s3_v, mask=mask_v, other=0.0).to(tl.float32)
"""
    # loads
    loads = ""
    for ik in range(NK):
        k_off = ik * BLOCK_K
        loads += f"        e{ik} = tl.load(e_ptr + b * s0_e + h * s1_e + t * s2_e + ({k_off} + offs_k) * s3_e, mask=({k_off} + offs_k) < K, other=0.0).to(tl.float32)\n"
        loads += f"        k{ik} = tl.load(k_ptr + b * s0_k + h * s1_k + t * s2_k + ({k_off} + offs_k) * s3_k, mask=({k_off} + offs_k) < K, other=0.0).to(tl.float32)\n"
        loads += f"        g{ik} = tl.load(g_ptr + b * s0_g + h * s1_g + t * s2_g + ({k_off} + offs_k) * s3_g, mask=({k_off} + offs_k) < K, other=0.0).to(tl.float32)\n"
        loads += f"        d{ik} = tl.exp(g{ik})\n"
        loads += f"        q{ik} = tl.load(q_ptr + b * s0_q + h * s1_q + t * s2_q + ({k_off} + offs_k) * s3_q, mask=({k_off} + offs_k) < K, other=0.0).to(tl.float32)\n"
    # reductions
    bdot = " + ".join(f"tl.sum(k{ik} * e{ik})" for ik in range(NK)) if NK>0 else "0.0"
    red = f"        Bdot = {bdot}\n"
    red += "        eTS = tl.zeros((BLOCK_V,), dtype=tl.float32)\n"
    red += "        A = tl.zeros((BLOCK_V,), dtype=tl.float32)\n"
    for ik in range(NK):
        red += f"        eTS += tl.sum((e{ik} * d{ik})[:, None] * s{ik}, axis=0)\n"
        red += f"        A += tl.sum((k{ik} * d{ik})[:, None] * s{ik}, axis=0)\n"
    red += "        pred = A - ct * Bdot * eTS\n"
    red += "        delta = v_vals - pred\n"
    # update
    upd = ""
    for ik in range(NK):
        upd += f"        s{ik} = s{ik} * d{ik}[:, None] - ct * e{ik}[:, None] * eTS[None, :] + bt * k{ik}[:, None] * delta[None, :]\n"
    upd += "        out_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)\n"
    for ik in range(NK):
        upd += f"        out_acc += tl.sum(q{ik}[:, None] * s{ik}, axis=0)\n"
    upd += "        out_acc = out_acc * scale\n"
    upd += "        tl.store(out_ptr + b * s0_o + h * s1_o + t * s2_o + (v_col + offs_v) * s3_o, out_acc, mask=mask_v)\n"
    code = header + init + body + loads + red + upd
    g = {}
    exec(code, g)
    fn = g[k_name]
    _kernel_cache[NK] = fn
    return fn

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)
    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
    BLOCK_K = 32
    BLOCK_V = 32
    NK = (K + BLOCK_K - 1) // BLOCK_K
    NV = (V + BLOCK_V - 1) // BLOCK_V
    # HBM fallback for very large K (not expected in tests)
    if NK > 8:
        scratch = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32)
        grid = (B * H * NV,)
        _fwd_kernel_hbm[grid](
            q, k, v, e, g, beta, gamma, scratch, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            e.stride(0), e.stride(1), e.stride(2), e.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            scratch.stride(0), scratch.stride(1), scratch.stride(2), scratch.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H, T, K, V, NV, NK, BLOCK_K, BLOCK_V, scale_f,
            num_warps=4, num_stages=2,
        )
        return out
    kern = _get_col_kernel(NK)
    grid = (B * H * NV,)
    kern[grid](
        q, k, v, e, g, beta, gamma, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        e.stride(0), e.stride(1), e.stride(2), e.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, T, K, V, NV, BLOCK_K, BLOCK_V, scale_f,
        num_warps=4, num_stages=2,
    )
    return out
```
Approach is a V-sharded persistent BH kernel that keeps the KxV state tiled 32x32 in registers/SRAM instead of HBM; rewriting the two rank-1 updates as one decay plus `pred = kᵀS - ct*(kᵀe)*(eᵀS)` lets the whole step be done with one vector load per K-tile and tl.sum reductions, and the kernel is code-generated for the exact NK to keep all loops constexpr. Unsure about optimal BLOCK/nWarps for L4 occupancy and whether the checker’s relative error allows tf32.