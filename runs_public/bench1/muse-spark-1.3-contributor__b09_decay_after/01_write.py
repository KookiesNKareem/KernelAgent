import torch
import triton
import triton.language as tl

@triton.jit
def _gated_delta_fwd(
    Q, Kptr, Vptr, Gptr, Betaptr, Out,
    sQb, sQh, sQt, sQk,
    sKb, sKh, sKt, sKk,
    sVb, sVh, sVt, sVv,
    sGb, sGh, sGt, sGk,
    sBb, sBh, sBt,
    sOb, sOh, sOt, sOv,
    H, T, Kdim, Vdim, NV,
    scale,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NV
    iv = pid % NV
    b = bh // H
    h = bh % H
    offK = tl.arange(0, BLOCK_K)
    offV = tl.arange(0, BLOCK_V)
    maskK = offK < Kdim
    vIdx = iv * BLOCK_V + offV
    maskV = vIdx < Vdim

    qBase = Q + b * sQb + h * sQh
    kBase = Kptr + b * sKb + h * sKh
    vBase = Vptr + b * sVb + h * sVh
    gBase = Gptr + b * sGb + h * sGh
    bBase = Betaptr + b * sBb + h * sBh
    oBase = Out + b * sOb + h * sOh

    S = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    for t in range(T):
        q = tl.load(qBase + t * sQt + offK * sQk, mask=maskK, other=0.0).to(tl.float32)
        k = tl.load(kBase + t * sKt + offK * sKk, mask=maskK, other=0.0).to(tl.float32)
        gg = tl.load(gBase + t * sGt + offK * sGk, mask=maskK, other=0.0)
        vv = tl.load(vBase + t * sVt + vIdx * sVv, mask=maskV, other=0.0).to(tl.float32)
        bt = tl.load(bBase + t * sBt)
        d = tl.exp(gg)
        pred = tl.sum(S * k[:, None], axis=0)
        delta = (vv - pred) * bt
        S = S + k[:, None] * delta[None, :]
        S = S * d[:, None]
        o = tl.sum(S * (q * scale)[:, None], axis=0)
        tl.store(oBase + t * sOt + vIdx * sOv, o, mask=maskV)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e: [B,H,T,K] bf16, v: [B,H,T,V] bf16, g: [B,H,T,K] fp32
    # beta,gamma: [B,H,T] fp32, scale: scalar. e,gamma unused (match reference).
    B, H, T, Kdim = q.shape
    Vdim = v.shape[-1]
    out = torch.empty(B, H, T, Vdim, device=q.device, dtype=torch.float32)
    if out.numel() == 0 or T == 0:
        return out
    BLOCK_K = triton.next_power_of_2(Kdim)
    if BLOCK_K < 16:
        BLOCK_K = 16
    if BLOCK_K > 1024:
        BLOCK_K = 1024
    if Vdim <= 16:
        BLOCK_V = 16
    elif Vdim <= 32:
        BLOCK_V = 32
    else:
        BLOCK_V = 64
    while BLOCK_K * BLOCK_V > 16384 and BLOCK_V > 16:
        BLOCK_V //= 2
    NV = (Vdim + BLOCK_V - 1) // BLOCK_V
    if torch.is_tensor(scale):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)
    sQb, sQh, sQt, sQk = q.stride()
    sKb, sKh, sKt, sKk = k.stride()
    sVb, sVh, sVt, sVv = v.stride()
    sGb, sGh, sGt, sGk = g.stride()
    sBb, sBh, sBt = beta.stride()
    sOb, sOh, sOt, sOv = out.stride()
    grid = (B * H * NV,)
    _gated_delta_fwd[grid](
        q, k, v, g, beta, out,
        sQb, sQh, sQt, sQk,
        sKb, sKh, sKt, sKk,
        sVb, sVh, sVt, sVv,
        sGb, sGh, sGt, sGk,
        sBb, sBh, sBt,
        sOb, sOh, sOt, sOv,
        H, T, Kdim, Vdim, NV,
        scale_f,
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        num_warps=4,
    )
    return out
