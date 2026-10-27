import torch
import triton
import triton.language as tl


@triton.jit
def _recurrent_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    H, T,
    sqb, sqh, sqt, sqk,
    skb, skh, skt, skk,
    svb, svh, svt, svv,
    seb, seh, set_, sek,
    sgb, sgh, sgt, sgk,
    sbb, sbh, sbt,
    scb, sch, sct,
    sob, soh, sot, sov,
    scale,
    KDIM, VDIM,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < KDIM
    mask_v = offs_v < VDIM

    q_ptrs = q_ptr + b * sqb + h * sqh + offs_k * sqk
    k_ptrs = k_ptr + b * skb + h * skh + offs_k * skk
    v_ptrs = v_ptr + b * svb + h * svh + offs_v * svv
    e_ptrs = e_ptr + b * seb + h * seh + offs_k * sek
    g_ptrs = g_ptr + b * sgb + h * sgh + offs_k * sgk
    out_ptrs = out_ptr + b * sob + h * soh + offs_v * sov

    beta_pos = beta_ptr + b * sbb + h * sbh
    gamma_pos = gamma_ptr + b * scb + h * sch

    # One program owns this [K, V-block] slice of the fp32 state.
    S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for _t in range(T):
        kt = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        gt = tl.load(g_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        vt = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
        et = tl.load(e_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        qt = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        bt = tl.load(beta_pos).to(tl.float32)
        ct = tl.load(gamma_pos).to(tl.float32)

        # Decay, then the delta-rule prediction/write.
        S = S * tl.exp(gt)[:, None]
        pred = tl.sum(S * kt[:, None], axis=0)
        S = S + kt[:, None] * (bt * (vt - pred))[None, :]

        # Second, independent additive write along e.
        S = S + et[:, None] * (ct * vt)[None, :]

        ot = tl.sum(S * (qt * scale)[:, None], axis=0)
        tl.store(out_ptrs, ot, mask=mask_v)

        q_ptrs += sqt
        k_ptrs += skt
        v_ptrs += svt
        e_ptrs += set_
        g_ptrs += sgt
        out_ptrs += sot
        beta_pos += sbt
        gamma_pos += sct


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    if B == 0 or H == 0 or T == 0 or V == 0 or K == 0:
        return torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    BLOCK_K = triton.next_power_of_2(K)
    v_pow2 = triton.next_power_of_2(V)

    # Target a roughly 4096-element state tile; this gives BLOCK_V=32
    # for the important K=V=128 case and therefore 128 programs at
    # B=4, H=8, T=2048.
    BLOCK_V = 16
    while BLOCK_V < v_pow2 and BLOCK_K * BLOCK_V < 4096:
        BLOCK_V *= 2

    tile_elems = BLOCK_K * BLOCK_V
    if tile_elems >= 8192:
        num_warps = 8
    elif tile_elems >= 4096:
        num_warps = 4
    elif tile_elems >= 2048:
        num_warps = 2
    else:
        num_warps = 1

    num_v_blocks = (V + BLOCK_V - 1) // BLOCK_V
    grid = (B * H, num_v_blocks)

    _recurrent_kernel[grid](
        q, k, v, e, g, beta, gamma, out,
        H, T,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        e.stride(0), e.stride(1), e.stride(2), e.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        float(scale),
        K, V,
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        num_warps=num_warps, num_stages=2,
    )
    return out
