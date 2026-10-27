import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, o_ptr,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    scale,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, N_VB: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = pid // N_VB
    pid_vb = pid % N_VB
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < V

    q_seq = q_ptr + pid_bh * T * K
    k_seq = k_ptr + pid_bh * T * K
    e_seq = e_ptr + pid_bh * T * K
    g_seq = g_ptr + pid_bh * T * K
    v_seq = v_ptr + pid_bh * T * V
    o_seq = o_ptr + pid_bh * T * V
    beta_seq = beta_ptr + pid_bh * T
    gamma_seq = gamma_ptr + pid_bh * T

    S = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)

    for t in range(T):
        k_vec = tl.load(k_seq + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        e_vec = tl.load(e_seq + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        g_vec = tl.load(g_seq + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        q_vec = tl.load(q_seq + t * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_vec = tl.load(v_seq + t * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        bt = tl.load(beta_seq + t).to(tl.float32)
        ct = tl.load(gamma_seq + t).to(tl.float32)

        d_vec = tl.exp(g_vec)
        S = S * d_vec[:, None]

        pred = tl.sum(S * k_vec[:, None], axis=0)
        delta = v_vec - pred
        S = S + k_vec[:, None] * (bt * delta)[None, :]

        dot_e = tl.sum(S * e_vec[:, None], axis=0)
        S = S - (ct * e_vec)[:, None] * dot_e[None, :]

        o_vec = tl.sum(S * (q_vec * scale)[:, None], axis=0)
        tl.store(o_seq + t * V + offs_v, o_vec, mask=mask_v)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        try:
            scale = float(scale)
        except Exception:
            pass
    if T == 0 or K == 0 or Vd == 0:
        return torch.empty(B, H, T, Vd, device=q.device, dtype=torch.float32)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    e = e.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    gamma = gamma.contiguous()
    out = torch.empty(B, H, T, Vd, device=q.device, dtype=torch.float32)

    BLOCK_K = triton.next_power_of_2(K)
    if BLOCK_K < 16:
        BLOCK_K = 16
    target = 8192 // BLOCK_K
    if target > 64:
        target = 64
    if target < 16:
        target = 16
    BLOCK_V = target
    if Vd < BLOCK_V:
        BLOCK_V = triton.next_power_of_2(Vd)
        if BLOCK_V < 16:
            BLOCK_V = 16
    n_vb = (Vd + BLOCK_V - 1) // BLOCK_V
    num_warps = 8 if BLOCK_K * BLOCK_V > 4096 else 4
    grid = (B * H * n_vb,)
    _fwd_kernel[grid](
        q, k, v, e, g, beta, gamma, out,
        T, K, Vd, scale,
        BLOCK_K, BLOCK_V, n_vb,
        num_warps=num_warps,
    )
    return out
