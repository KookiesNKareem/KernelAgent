import torch
import triton
import triton.language as tl


@triton.jit
def _recurrent_delta_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    e_ptr,
    g_ptr,
    beta_ptr,
    gamma_ptr,
    out_ptr,
    T,
    K,
    V,
    scale,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid_v = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V

    bh = pid_bh.to(tl.int64)
    q_base = q_ptr + bh * T * K
    k_base = k_ptr + bh * T * K
    e_base = e_ptr + bh * T * K
    v_base = v_ptr + bh * T * V
    o_base = out_ptr + bh * T * V
    s_base = g_ptr + bh * T

    # State block for this (batch, head, value-block).
    S = tl.zeros((BK, BV), dtype=tl.float32)

    for t in range(T):
        tk = t * K
        tv = t * V

        kt = tl.load(k_base + tk + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        et = tl.load(e_base + tk + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        qt = tl.load(q_base + tk + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        vt = tl.load(v_base + tv + offs_v, mask=mask_v, other=0.0).to(tl.float32)

        decay = tl.exp(tl.load(s_base + t))
        beta_t = tl.load(beta_ptr + bh * T + t)
        gamma_t = tl.load(gamma_ptr + bh * T + t)

        # S = exp(g_t) * S
        S = S * decay

        # S = (I - gamma_t e_t e_t^T) S
        erased = tl.sum(S * et[:, None], axis=0)
        S = S - et[:, None] * (gamma_t * erased)[None, :]

        # S = (I - beta_t k_t k_t^T) S + beta_t k_t v_t^T
        pred = tl.sum(S * kt[:, None], axis=0)
        S = S + kt[:, None] * (beta_t * (vt - pred))[None, :]

        # o_t = scale * S^T q_t
        ot = tl.sum(S * (qt * scale)[:, None], axis=0)
        tl.store(o_base + tv + offs_v, ot, mask=mask_v)


def _next_power_of_two(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]

    if T == 0 or B == 0 or H == 0 or V == 0:
        return torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    input_device = q.device
    work_device = input_device if input_device.type == "cuda" else torch.device("cuda")

    q_c = q.to(work_device).contiguous()
    k_c = k.to(work_device).contiguous()
    v_c = v.to(work_device).contiguous()
    e_c = e.to(work_device).contiguous()
    g_c = g.to(work_device).contiguous()
    beta_c = beta.to(work_device).contiguous()
    gamma_c = gamma.to(work_device).contiguous()

    BH = B * H
    q_f = q_c.reshape(BH, T, K)
    k_f = k_c.reshape(BH, T, K)
    v_f = v_c.reshape(BH, T, V)
    e_f = e_c.reshape(BH, T, K)
    g_f = g_c.reshape(BH, T)
    beta_f = beta_c.reshape(BH, T)
    gamma_f = gamma_c.reshape(BH, T)

    out = torch.empty((BH, T, V), device=work_device, dtype=torch.float32)

    BK = max(16, _next_power_of_two(K))
    BV = 16 if BK >= 256 else 32
    num_v_blocks = triton.cdiv(V, BV)
    num_warps = 8 if BK * BV > 4096 else 4

    _recurrent_delta_kernel[(num_v_blocks, BH)](
        q_f,
        k_f,
        v_f,
        e_f,
        g_f,
        beta_f,
        gamma_f,
        out,
        T,
        K,
        V,
        float(scale),
        BK=BK,
        BV=BV,
        num_warps=num_warps,
        num_stages=2,
    )

    out = out.reshape(B, H, T, V)
    if out.device != input_device:
        out = out.to(input_device)
    return out
