import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,
    scale,
    B: tl.constexpr, H: tl.constexpr, T, K, V,
    Q_B: tl.constexpr, Q_H: tl.constexpr, Q_T: tl.constexpr, Q_K: tl.constexpr,
    K_B: tl.constexpr, K_H: tl.constexpr, K_T: tl.constexpr, K_K: tl.constexpr,
    V_B: tl.constexpr, V_H: tl.constexpr, V_T: tl.constexpr, V_V: tl.constexpr,
    E_B: tl.constexpr, E_H: tl.constexpr, E_T: tl.constexpr, E_K: tl.constexpr,
    G_B: tl.constexpr, G_H: tl.constexpr, G_T: tl.constexpr, G_K: tl.constexpr,
    B_B: tl.constexpr, B_H: tl.constexpr, B_T: tl.constexpr,
    Gm_B: tl.constexpr, Gm_H: tl.constexpr, Gm_T: tl.constexpr,
    O_B: tl.constexpr, O_H: tl.constexpr, O_T: tl.constexpr, O_V: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, NK: tl.constexpr, NV: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # state S = NK x NV tiles of (BLOCK_K,BLOCK_V)  KxV = 16k  (64KB)
    # allocate - python unrolled
    S = []
    for ii in range(NK):
        row = []
        for jj in range(NV):
            row.append(tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32))
        S.append(row)

    # main sequential loop - must be a runtime loop, T is not constexpr
    for t in range(T):
        # ---- loads ----
        kt = []
        et = []
        qt = []
        gt = []
        for ii in range(NK):
            offs = tl.arange(0, BLOCK_K) + ii * BLOCK_K
            mask = offs < K
            q_off = b * Q_B + h * Q_H + t * Q_T
            k_off = b * K_B + h * K_H + t * K_T
            e_off = b * E_B + h * E_H + t * E_T
            g_off = b * G_B + h * G_H + t * G_T
            qv = tl.load(q_ptr + q_off + offs * Q_K, mask=mask, other=0.0)
            kv = tl.load(k_ptr + k_off + offs * K_K, mask=mask, other=0.0)
            ev = tl.load(e_ptr + e_off + offs * E_K, mask=mask, other=0.0)
            gv = tl.load(g_ptr + g_off + offs * G_K, mask=mask, other=0.0)
            # promote bf16 -> fp32
            qt.append(qv.to(tl.float32))
            kt.append(kv.to(tl.float32))
            et.append(ev.to(tl.float32))
            gt.append(gv.to(tl.float32))

        vt = []
        for jj in range(NV):
            offs = tl.arange(0, BLOCK_V) + jj * BLOCK_V
            mask = offs < V
            v_off = b * V_B + h * V_H + t * V_T
            vv = tl.load(v_ptr + v_off + offs * V_V, mask=mask, other=0.0)
            vt.append(vv.to(tl.float32))

        beta_t = tl.load(beta_ptr + b * B_B + h * B_H + t * B_T).to(tl.float32)
        gamma_t = tl.load(gamma_ptr + b * Gm_B + h * Gm_H + t * Gm_T).to(tl.float32)

        # ---- 1) S = diag(exp(g)) * S ----
        for ii in range(NK):
            d = tl.exp(gt[ii])  # (BLOCK_K,)
            d2 = d[:, None]  # (BLOCK_K,1)
            for jj in range(NV):
                S[ii][jj] = S[ii][jj] * d2

        # ---- 2) pred = k^T S ----
        pred = []
        for jj in range(NV):
            pred.append(tl.zeros((BLOCK_V,), dtype=tl.float32))
        for ii in range(NK):
            k2 = kt[ii][:, None]
            for jj in range(NV):
                pred[jj] += tl.sum(k2 * S[ii][jj], axis=0)

        # ---- 3) S += beta * k * (v-pred)^T ----
        w = []
        for jj in range(NV):
            w.append(vt[jj] - pred[jj])
        for ii in range(NK):
            k2 = kt[ii][:, None]
            for jj in range(NV):
                # beta * k * w^T
                S[ii][jj] = S[ii][jj] + beta_t * (k2 * w[jj][None, :])

        # ---- 4) proj = e^T S ----
        proj = []
        for jj in range(NV):
            proj.append(tl.zeros((BLOCK_V,), dtype=tl.float32))
        for ii in range(NK):
            e2 = et[ii][:, None]
            for jj in range(NV):
                proj[jj] += tl.sum(e2 * S[ii][jj], axis=0)

        # ---- 5) S -= gamma * e * proj^T ----
        for ii in range(NK):
            e2 = et[ii][:, None]
            for jj in range(NV):
                S[ii][jj] = S[ii][jj] - gamma_t * (e2 * proj[jj][None, :])

        # ---- 6) o = scale * S^T q ----
        for jj in range(NV):
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for ii in range(NK):
                q2 = qt[ii][:, None]
                acc += tl.sum(q2 * S[ii][jj], axis=0)
            acc = acc * scale
            offs = tl.arange(0, BLOCK_V) + jj * BLOCK_V
            mask = offs < V
            o_off = b * O_B + h * O_H + t * O_T
            tl.store(out_ptr + o_off + offs * O_V, acc, mask=mask)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e: [B,H,T,K] bf16  v:[B,H,T,V] bf16  g:[B,H,T,K] fp32  beta,gamma:[B,H,T] fp32
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    # output allowed to be fp32 (reference is fp32) or q dtype
    out = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)

    BLOCK_K = 32
    BLOCK_V = 32
    NK = (K + BLOCK_K - 1) // BLOCK_K
    NV = (Vd + BLOCK_V - 1) // BLOCK_V

    # strides (elements)
    Q_B, Q_H, Q_T, Q_K = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
    K_B, K_H, K_T, K_K = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
    V_B, V_H, V_T, V_V = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
    E_B, E_H, E_T, E_K = e.stride(0), e.stride(1), e.stride(2), e.stride(3)
    G_B, G_H, G_T, G_K = g.stride(0), g.stride(1), g.stride(2), g.stride(3)
    B_B, B_H, B_T = beta.stride(0), beta.stride(1), beta.stride(2)
    Gm_B, Gm_H, Gm_T = gamma.stride(0), gamma.stride(1), gamma.stride(2)
    O_B, O_H, O_T, O_V = out.stride(0), out.stride(1), out.stride(2), out.stride(3)

    # scale is python float / 0-d tensor
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    scale = float(scale)

    grid = (B * H,)
    _fwd_kernel[grid](
        q, k, v, e, g, beta, gamma, out,
        scale,
        B, H, T, K, Vd,
        Q_B, Q_H, Q_T, Q_K,
        K_B, K_H, K_T, K_K,
        V_B, V_H, V_T, V_V,
        E_B, E_H, E_T, E_K,
        G_B, G_H, G_T, G_K,
        B_B, B_H, B_T,
        Gm_B, Gm_H, Gm_T,
        O_B, O_H, O_T, O_V,
        BLOCK_K, BLOCK_V, NK, NV,
    )
    return out
