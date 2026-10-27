import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_kernel(
    Qptr, Kptr, Vptr, Optr, Gptr, Bptr, Sptr,
    scale,
    BH, T, Kdim, Vdim,
    chunk_start,
    BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_c = tl.arange(0, BLOCK_C)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_cols = pid_v * BLOCK_V + offs_v
    remain = T - chunk_start
    mask_c = offs_c < remain
    mask_k = offs_k < Kdim
    mask_v = v_cols < Vdim

    q_base = Qptr + pid_bh * T * Kdim
    k_base = Kptr + pid_bh * T * Kdim
    v_base = Vptr + pid_bh * T * Vdim
    o_base = Optr + pid_bh * T * Vdim
    g_base = Gptr + pid_bh * T
    b_base = Bptr + pid_bh * T

    q_ptrs = q_base + (chunk_start + offs_c)[:, None] * Kdim + offs_k[None, :]
    k_ptrs = k_base + (chunk_start + offs_c)[:, None] * Kdim + offs_k[None, :]
    v_ptrs = v_base + (chunk_start + offs_c)[:, None] * Vdim + v_cols[None, :]
    g_ptrs = g_base + chunk_start + offs_c
    b_ptrs = b_base + chunk_start + offs_c

    Qc = tl.load(q_ptrs, mask=mask_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
    Kc = tl.load(k_ptrs, mask=mask_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
    Vc = tl.load(v_ptrs, mask=mask_c[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
    gvec = tl.load(g_ptrs, mask=mask_c, other=0.0).to(tl.float32)
    bvec = tl.load(b_ptrs, mask=mask_c, other=0.0).to(tl.float32)

    s_ptrs = Sptr + pid_bh * Kdim * Vdim + offs_k[:, None] * Vdim + v_cols[None, :]
    S = tl.load(s_ptrs, mask=mask_k[:, None] & mask_v[None, :], other=0.0).to(tl.float32)

    lower_inc = offs_c[:, None] >= offs_c[None, :]
    strict_lower = offs_c[:, None] > offs_c[None, :]
    lower_f = lower_inc.to(tl.float32)

    G = tl.sum(lower_f * gvec[None, :], axis=1)
    Ddiff = G[:, None] - G[None, :]
    Ddiff = tl.where(lower_inc, Ddiff, -1e30)
    E = tl.exp(Ddiff)

    KK = tl.dot(Kc, tl.trans(Kc))
    KS0 = tl.dot(Kc, S)
    QS0 = tl.dot(Qc, S)
    decay_row = tl.exp(G)
    RHS = Vc - decay_row[:, None] * KS0

    Lmat = tl.where(strict_lower, KK * bvec[None, :] * E, 0.0)

    U = tl.zeros([BLOCK_C, BLOCK_V], tl.float32)
    for ii in range(BLOCK_C):
        sel = (offs_c == ii)
        sel_f = sel[:, None].to(tl.float32)
        L_row = tl.sum(sel_f * Lmat, axis=0)
        rhs_row = tl.sum(sel_f * RHS, axis=0)
        corr = tl.sum(L_row[:, None] * U, axis=0)
        u_row = rhs_row - corr
        U = U + sel_f * (u_row[None, :] - U)

    QK = tl.dot(Qc, tl.trans(Kc))
    Pmat = tl.where(lower_inc, QK * bvec[None, :] * E, 0.0)
    PU = tl.dot(Pmat, U)
    Ochunk = decay_row[:, None] * QS0 + PU
    Ochunk = Ochunk * scale

    o_ptrs = o_base + (chunk_start + offs_c)[:, None] * Vdim + v_cols[None, :]
    tl.store(o_ptrs, Ochunk, mask=mask_c[:, None] & mask_v[None, :])

    sel_last = (offs_c == (BLOCK_C - 1))
    G_end = tl.sum(sel_last.to(tl.float32) * G)
    decay_end = tl.exp(G_end)
    dec_to_end = tl.exp(G_end - G)
    w = bvec * dec_to_end
    DU = w[:, None] * U
    KTDU = tl.dot(tl.trans(Kc), DU)
    Snew = decay_end * S + KTDU
    tl.store(s_ptrs, Snew, mask=mask_k[:, None] & mask_v[None, :])


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    Vdim = v.shape[-1]
    if T == 0:
        return torch.empty(B, H, 0, Vdim, dtype=torch.float32, device=q.device)
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.detach().to(torch.float32).item())
    else:
        scale_f = float(scale)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    BH = B * H
    q_ = q.reshape(BH, T, K)
    k_ = k.reshape(BH, T, K)
    v_ = v.reshape(BH, T, Vdim)
    g_ = g.reshape(BH, T)
    b_ = beta.reshape(BH, T)

    out_ = torch.empty(BH, T, Vdim, dtype=torch.float32, device=q.device)
    state = torch.zeros(BH, K, Vdim, dtype=torch.float32, device=q.device)

    BLOCK_C = 64
    BLOCK_K = _next_pow2(K)
    if BLOCK_K < 16:
        BLOCK_K = 16
    BLOCK_V = 32
    NV = (Vdim + BLOCK_V - 1) // BLOCK_V
    grid = (BH, NV)

    for cs in range(0, T, BLOCK_C):
        _chunk_kernel[grid](
            q_, k_, v_, out_, g_, b_, state,
            scale_f, BH, T, K, Vdim, cs,
            BLOCK_C, BLOCK_K, BLOCK_V,
            num_warps=4,
        )
    return out_.reshape(B, H, T, Vdim)
