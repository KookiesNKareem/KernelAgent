```python
import torch
import triton
import triton.language as tl

@triton.jit
def chunk_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr, state_ptr,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, SCALE: tl.constexpr, CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, 32)
    q_head = q_ptr + pid * T * K
    k_head = k_ptr + pid * T * K
    e_head = e_ptr + pid * T * K
    g_head = g_ptr + pid * T * K
    v_head = v_ptr + pid * T * V
    beta_head = beta_ptr + pid * T
    gamma_head = gamma_ptr + pid * T
    out_head = out_ptr + pid * T * V
    state_head = state_ptr + pid * K * V

    r0 = offs
    r1 = offs + 32
    r2 = offs + 64
    r3 = offs + 96
    c0 = offs
    c1 = offs + 32
    c2 = offs + 64
    c3 = offs + 96

    s00 = tl.load(state_head + r0[:, None] * V + c0[None, :])
    s01 = tl.load(state_head + r0[:, None] * V + c1[None, :])
    s02 = tl.load(state_head + r0[:, None] * V + c2[None, :])
    s03 = tl.load(state_head + r0[:, None] * V + c3[None, :])
    s10 = tl.load(state_head + r1[:, None] * V + c0[None, :])
    s11 = tl.load(state_head + r1[:, None] * V + c1[None, :])
    s12 = tl.load(state_head + r1[:, None] * V + c2[None, :])
    s13 = tl.load(state_head + r1[:, None] * V + c3[None, :])
    s20 = tl.load(state_head + r2[:, None] * V + c0[None, :])
    s21 = tl.load(state_head + r2[:, None] * V + c1[None, :])
    s22 = tl.load(state_head + r2[:, None] * V + c2[None, :])
    s23 = tl.load(state_head + r2[:, None] * V + c3[None, :])
    s30 = tl.load(state_head + r3[:, None] * V + c0[None, :])
    s31 = tl.load(state_head + r3[:, None] * V + c1[None, :])
    s32 = tl.load(state_head + r3[:, None] * V + c2[None, :])
    s33 = tl.load(state_head + r3[:, None] * V + c3[None, :])

    for step in range(32):
        cur_t = CHUNK + step
        q0 = tl.load(q_head + cur_t * K + offs).to(tl.float32)
        q1 = tl.load(q_head + cur_t * K + 32 + offs).to(tl.float32)
        q2 = tl.load(q_head + cur_t * K + 64 + offs).to(tl.float32)
        q3 = tl.load(q_head + cur_t * K + 96 + offs).to(tl.float32)
        k0 = tl.load(k_head + cur_t * K + offs).to(tl.float32)
        k1 = tl.load(k_head + cur_t * K + 32 + offs).to(tl.float32)
        k2 = tl.load(k_head + cur_t * K + 64 + offs).to(tl.float32)
        k3 = tl.load(k_head + cur_t * K + 96 + offs).to(tl.float32)
        e0 = tl.load(e_head + cur_t * K + offs).to(tl.float32)
        e1 = tl.load(e_head + cur_t * K + 32 + offs).to(tl.float32)
        e2 = tl.load(e_head + cur_t * K + 64 + offs).to(tl.float32)
        e3 = tl.load(e_head + cur_t * K + 96 + offs).to(tl.float32)
        g0 = tl.load(g_head + cur_t * K + offs)
        g1 = tl.load(g_head + cur_t * K + 32 + offs)
        g2 = tl.load(g_head + cur_t * K + 64 + offs)
        g3 = tl.load(g_head + cur_t * K + 96 + offs)
        v0 = tl.load(v_head + cur_t * V + offs).to(tl.float32)
        v1 = tl.load(v_head + cur_t * V + 32 + offs).to(tl.float32)
        v2 = tl.load(v_head + cur_t * V + 64 + offs).to(tl.float32)
        v3 = tl.load(v_head + cur_t * V + 96 + offs).to(tl.float32)
        beta_t = tl.load(beta_head + cur_t)
        gamma_t = tl.load(gamma_head + cur_t)

        d0 = tl.exp(g0)
        d1 = tl.exp(g1)
        d2 = tl.exp(g2)
        d3 = tl.exp(g3)

        s00 = s00 * d0[:, None]
        s01 = s01 * d0[:, None]
        s02 = s02 * d0[:, None]
        s03 = s03 * d0[:, None]
        s10 = s10 * d1[:, None]
        s11 = s11 * d1[:, None]
        s12 = s12 * d1[:, None]
        s13 = s13 * d1[:, None]
        s20 = s20 * d2[:, None]
        s21 = s21 * d2[:, None]
        s22 = s22 * d2[:, None]
        s23 = s23 * d2[:, None]
        s30 = s30 * d3[:, None]
        s31 = s31 * d3[:, None]
        s32 = s32 * d3[:, None]
        s33 = s33 * d3[:, None]

        h0 = tl.sum(s00 * e0[None, :], axis=1) + tl.sum(s01 * e1[None, :], axis=1) + tl.sum(s02 * e2[None, :], axis=1) + tl.sum(s03 * e3[None, :], axis=1)
        h1 = tl.sum(s10 * e0[None, :], axis=1) + tl.sum(s11 * e1[None, :], axis=1) + tl.sum(s12 * e2[None, :], axis=1) + tl.sum(s13 * e3[None, :], axis=1)
        h2 = tl.sum(s20 * e0[None, :], axis=1) + tl.sum(s21 * e1[None, :], axis=1) + tl.sum(s22 * e2[None, :], axis=1) + tl.sum(s23 * e3[None, :], axis=1)
        h3 = tl.sum(s30 * e0[None, :], axis=1) + tl.sum(s31 * e1[None, :], axis=1) + tl.sum(s32 * e2[None, :], axis=1) + tl.sum(s33 * e3[None, :], axis=1)

        s00 = s00 - gamma_t * h0[:, None] * e0[None, :]
        s01 = s01 - gamma_t * h0[:, None] * e1[None, :]
        s02 = s02 - gamma_t * h0[:, None] * e2[None, :]
        s03 = s03 - gamma_t * h0[:, None] * e3[None, :]
        s10 = s10 - gamma_t * h1[:, None] * e0[None, :]
        s11 = s11 - gamma_t * h1[:, None] * e1[None, :]
        s12 = s12 - gamma_t * h1[:, None] * e2[None, :]
        s13 = s13 - gamma_t * h1[:, None] * e3[None, :]
        s20 = s20 - gamma_t * h2[:, None] * e0[None, :]
        s21 = s21 - gamma_t * h2[:, None] * e1[None, :]
        s22 = s22 - gamma_t * h2[:, None] * e2[None, :]
        s23 = s23 - gamma_t * h2[:, None] * e3[None, :]
        s30 = s30 - gamma_t * h3[:, None] * e0[None, :]
        s31 = s31 - gamma_t * h3[:, None] * e1[None, :]
        s32 = s32 - gamma_t * h3[:, None] * e2[None, :]
        s33 = s33 - gamma_t * h3[:, None] * e3[None, :]

        pred0 = tl.sum(s00 * k0[:, None], axis=0) + tl.sum(s10 * k1[:, None], axis=0) + tl.sum(s20 * k2[:, None], axis=0) + tl.sum(s30 * k3[:, None], axis=0)
        pred1 = tl.sum(s01 * k0[:, None], axis=0) + tl.sum(s11 * k1[:, None], axis=0) + tl.sum(s21 * k2[:, None], axis=0) + tl.sum(s31 * k3[:, None], axis=0)
        pred2 = tl.sum(s02 * k0[:, None], axis=0) + tl.sum(s12 * k1[:, None], axis=0) + tl.sum(s22 * k2[:, None], axis=0) + tl.sum(s32 * k3[:, None], axis=0)
        pred3 = tl.sum(s03 * k0[:, None], axis=0) + tl.sum(s13 * k1[:, None], axis=0) + tl.sum(s23 * k2[:, None], axis=0) + tl.sum(s33 * k3[:, None], axis=0)

        delt0 = v0 - pred0
        delt1 = v1 - pred1
        delt2 = v2 - pred2
        delt3 = v3 - pred3

        s00 = s00 + beta_t * k0[:, None] * delt0[None, :]
        s01 = s01 + beta_t * k0[:, None] * delt1[None, :]
        s02 = s02 + beta_t * k0[:, None] * delt2[None, :]
        s03 = s03 + beta_t * k0[:, None] * delt3[None, :]
        s10 = s10 + beta_t * k1[:, None] * delt0[None, :]
        s11 = s11 + beta_t * k1[:, None] * delt1[None, :]
        s12 = s12 + beta_t * k1[:, None] * delt2[None, :]
        s13 = s13 + beta_t * k1[:, None] * delt3[None, :]
        s20 = s20 + beta_t * k2[:, None] * delt0[None, :]
        s21 = s21 + beta_t * k2[:, None] * delt1[None, :]
        s22 = s22 + beta_t * k2[:, None] * delt2[None, :]
        s23 = s23 + beta_t * k2[:, None] * delt3[None, :]
        s30 = s30 + beta_t * k3[:, None] * delt0[None, :]
        s31 = s31 + beta_t * k3[:, None] * delt1[None, :]
        s32 = s32 + beta_t * k3[:, None] * delt2[None, :]
        s33 = s33 + beta_t * k3[:, None] * delt3[None, :]

        o0 = tl.sum(s00 * q0[:, None], axis=0) + tl.sum(s10 * q1[:, None], axis=0) + tl.sum(s20 * q2[:, None], axis=0) + tl.sum(s30 * q3[:, None], axis=0)
        o1 = tl.sum(s01 * q0[:, None], axis=0) + tl.sum(s11 * q1[:, None], axis=0) + tl.sum(s21 * q2[:, None], axis=0) + tl.sum(s31 * q3[:, None], axis=0)
        o2 = tl.sum(s02 * q0[:, None], axis=0) + tl.sum(s12 * q1[:, None], axis=0) + tl.sum(s22 * q2[:, None], axis=0) + tl.sum(s32 * q3[:, None], axis=0)
        o3 = tl.sum(s03 * q0[:, None], axis=0) + tl.sum(s13 * q1[:, None], axis=0) + tl.sum(s23 * q2[:, None], axis=0) + tl.sum(s33 * q3[:, None], axis=0)
        o0 = o0 * SCALE
        o1 = o1 * SCALE
        o2 = o2 * SCALE
        o3 = o3 * SCALE
        tl.store(out_head + cur_t * V + offs, o0)
        tl.store(out_head + cur_t * V + 32 + offs, o1)
        tl.store(out_head + cur_t * V + 64 + offs, o2)
        tl.store(out_head + cur_t * V + 96 + offs, o3)

    tl.store(state_head + r0[:, None] * V + c0[None, :], s00)
    tl.store(state_head + r0[:, None] * V + c1[None, :], s01)
    tl.store(state_head + r0[:, None] * V + c2[None, :], s02)
    tl.store(state_head + r0[:, None] * V + c3[None, :], s03)
    tl.store(state_head + r1[:, None] * V + c0[None, :], s10)
    tl.store(state_head + r1[:, None] * V + c1[None, :], s11)
    tl.store(state_head + r1[:, None] * V + c2[None, :], s12)
    tl.store(state_head + r1[:, None] * V + c3[None, :], s13)
    tl.store(state_head + r2[:, None] * V + c0[None, :], s20)
    tl.store(state_head + r2[:, None] * V + c1[None, :], s21)
    tl.store(state_head + r2[:, None] * V + c2[None, :], s22)
    tl.store(state_head + r2[:, None] * V + c3[None, :], s23)
    tl.store(state_head + r3[:, None] * V + c0[None, :], s30)
    tl.store(state_head + r3[:, None] * V + c1[None, :], s31)
    tl.store(state_head + r3[:, None] * V + c2[None, :], s32)
    tl.store(state_head + r3[:, None] * V + c3[None, :], s33)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    device = q.device
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item()) if scale.numel() == 1 else float(scale)
    else:
        scale_f = float(scale)
    if K != 128 or V != 128:
        qf, kf, vf, ef, gf = (x.to(torch.float32) for x in (q, k, v, e, g))
        betaf = beta.to(torch.float32)
        gammaf = gamma.to(torch.float32)
        BH = B * H
        q_ = qf.reshape(BH, T, K)
        k_ = kf.reshape(BH, T, K)
        v_ = vf.reshape(BH, T, V)
        e_ = ef.reshape(BH, T, K)
        g_ = gf.reshape(BH, T, K)
        beta_ = betaf.reshape(BH, T)
        gamma_ = gammaf.reshape(BH, T)
        S = torch.zeros(BH, K, V, device=device, dtype=torch.float32)
        out = torch.empty(BH, T, V, device=device, dtype=torch.float32)
        for t in range(T):
            S = S * torch.exp(g_[:, t, :]).unsqueeze(-1)
            h = (S * e_[:, t, :].unsqueeze(1)).sum(-1)
            S = S - gamma_[:, t, None, None] * h.unsqueeze(-1) * e_[:, t, :].unsqueeze(1)
            pred = (S * k_[:, t, :].unsqueeze(-1)).sum(1)
            S = S + (beta_[:, t, None] * (v_[:, t, :] - pred)).unsqueeze(1) * k_[:, t, :].unsqueeze(-1)
            out[:, t, :] = (S * (q_[:, t, :] * scale_f).unsqueeze(-1)).sum(1)
        return out.reshape(B, H, T, V)
    BH = B * H

    def _prep(x, D):
        y = x.reshape(BH, T, D)
        if not y.is_contiguous():
            y = y.contiguous()
        return y

    q_c = _prep(q, K)
    k_c = _prep(k, K)
    e_c = _prep(e, K)
    g_c = _prep(g, K)
    v_c = _prep(v, V)
    beta_c = beta.reshape(BH, T)
    if not beta_c.is_contiguous():
        beta_c = beta_c.contiguous()
    gamma_c = gamma.reshape(BH, T)
    if not gamma_c.is_contiguous():
        gamma_c = gamma_c.contiguous()

    out_c = torch.empty(BH, T, V, device=device, dtype=torch.float32)
    state = torch.zeros(BH, K, V, device=device, dtype=torch.float32)
    n_full = T // 32
    for ck in range(n_full):
        chunk_start = ck * 32
        chunk_kernel[(BH,)](q_c, k_c, v_c, e_c, g_c, beta_c, gamma_c, out_c, state, T, K, V, scale_f, chunk_start)
    tail_start = n_full * 32
    if tail_start < T:
        S = state
        for t in range(tail_start, T):
            gt = g_c[:, t, :].float()
            et = e_c[:, t, :].float()
            kt = k_c[:, t, :].float()
            vt = v_c[:, t, :].float()
            qt = q_c[:, t, :].float()
            ct = gamma_c[:, t]
            bt = beta_c[:, t]
            S = S * torch.exp(gt).unsqueeze(-1)
            h = (S * et.unsqueeze(1)).sum(-1)
            S = S - ct[:, None, None] * h.unsqueeze(-1) * et.unsqueeze(1)
            pred = (S * kt.unsqueeze(-1)).sum(1)
            S = S + (bt[:, None] * (vt - pred)).unsqueeze(1) * kt.unsqueeze(-1)
            out_c[:, t, :] = (S * (qt * scale_f).unsqueeze(-1)).sum(1)
    return out_c.reshape(B, H, T, V)
```

The 64-step unroll exceeded Triton's IR size and failed to compile, so this keeps the proven 32-step fused kernel with the entire 128x128 state in 16 32x32 registers and state carried in HBM between chunks. The wrapper now avoids the bf16->fp32 `contiguous().to()` conversion kernel by passing bf16 pointers directly to Triton with `.to(tl.float32)` on load and only copying when `is_contiguous()` is false, removing the 0.749ms elementwise overhead.