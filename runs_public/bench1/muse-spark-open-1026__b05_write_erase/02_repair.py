import torch
import triton
import triton.language as tl

_BLOCK_K = 32
_BLOCK_V = 32
_cache = {}

def _get_kernel(NK, NV):
    key = (NK, NV)
    if key in _cache:
        return _cache[key]

    lines = []
    lines.append("import triton")
    lines.append("import triton.language as tl")
    lines.append("@triton.jit")
    lines.append(f"def _fwd_{NK}_{NV}_kernel(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr, scale, B, H, T, K, Vd, Q_B, Q_H, Q_T, Q_K, K_B, K_H, K_T, K_K, V_B, V_H, V_T, V_V, E_B, E_H, E_T, E_K, G_B, G_H, G_T, G_K, Beta_B, Beta_H, Beta_T, Gm_B, Gm_H, Gm_T, O_B, O_H, O_T, O_V, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):")
    lines.append("    pid = tl.program_id(0)")
    lines.append("    b = pid // H")
    lines.append("    h = pid % H")
    for i in range(NK):
        for j in range(NV):
            lines.append(f"    s{i}{j} = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)")
    lines.append("    t = 0")
    lines.append("    while t < T:")
    for i in range(NK):
        base = i * _BLOCK_K
        lines.append(f"        offs_k{i} = tl.arange(0, BLOCK_K) + {base}")
        lines.append(f"        mask_k{i} = offs_k{i} < K")
        lines.append(f"        q{i} = tl.load(q_ptr + b * Q_B + h * Q_H + t * Q_T + offs_k{i} * Q_K, mask=mask_k{i}, other=0.0).to(tl.float32)")
        lines.append(f"        k{i} = tl.load(k_ptr + b * K_B + h * K_H + t * K_T + offs_k{i} * K_K, mask=mask_k{i}, other=0.0).to(tl.float32)")
        lines.append(f"        e{i} = tl.load(e_ptr + b * E_B + h * E_H + t * E_T + offs_k{i} * E_K, mask=mask_k{i}, other=0.0).to(tl.float32)")
        lines.append(f"        g{i} = tl.load(g_ptr + b * G_B + h * G_H + t * G_T + offs_k{i} * G_K, mask=mask_k{i}, other=0.0).to(tl.float32)")
    for j in range(NV):
        base = j * _BLOCK_V
        lines.append(f"        offs_v{j} = tl.arange(0, BLOCK_V) + {base}")
        lines.append(f"        mask_v{j} = offs_v{j} < Vd")
        lines.append(f"        v{j} = tl.load(v_ptr + b * V_B + h * V_H + t * V_T + offs_v{j} * V_V, mask=mask_v{j}, other=0.0).to(tl.float32)")
    lines.append("        beta_t = tl.load(beta_ptr + b * Beta_B + h * Beta_H + t * Beta_T).to(tl.float32)")
    lines.append("        gamma_t = tl.load(gamma_ptr + b * Gm_B + h * Gm_H + t * Gm_T).to(tl.float32)")
    # 1) decay
    for i in range(NK):
        lines.append(f"        d{i} = tl.exp(g{i})")
        for j in range(NV):
            lines.append(f"        s{i}{j} = s{i}{j} * d{i}[:, None]")
    # 2) pred = k^T S
    for j in range(NV):
        lines.append(f"        p{j} = tl.zeros((BLOCK_V,), dtype=tl.float32)")
        for i in range(NK):
            lines.append(f"        p{j} += tl.sum(k{i}[:, None] * s{i}{j}, axis=0)")
    # 3) w = v - pred
    for j in range(NV):
        lines.append(f"        w{j} = v{j} - p{j}")
    # 4) S += beta * k w^T
    for i in range(NK):
        for j in range(NV):
            lines.append(f"        s{i}{j} = s{i}{j} + beta_t * (k{i}[:, None] * w{j}[None, :])")
    # 5) proj = e^T S
    for j in range(NV):
        lines.append(f"        r{j} = tl.zeros((BLOCK_V,), dtype=tl.float32)")
        for i in range(NK):
            lines.append(f"        r{j} += tl.sum(e{i}[:, None] * s{i}{j}, axis=0)")
    # 6) S -= gamma * e proj^T
    for i in range(NK):
        for j in range(NV):
            lines.append(f"        s{i}{j} = s{i}{j} - gamma_t * (e{i}[:, None] * r{j}[None, :])")
    # 7) out = scale * S^T q
    for j in range(NV):
        lines.append(f"        acc{j} = tl.zeros((BLOCK_V,), dtype=tl.float32)")
        for i in range(NK):
            lines.append(f"        acc{j} += tl.sum(q{i}[:, None] * s{i}{j}, axis=0)")
        lines.append(f"        acc{j} = acc{j} * scale")
        lines.append(f"        tl.store(out_ptr + b * O_B + h * O_H + t * O_T + offs_v{j} * O_V, acc{j}, mask=mask_v{j})")
    lines.append("        t += 1")

    code = "\n".join(lines)
    ns = {}
    exec(code, ns)
    ker = ns[f"_fwd_{NK}_{NV}_kernel"]
    _cache[key] = ker
    return ker


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e: [B,H,T,K] bf16, v: [B,H,T,V] bf16, g: [B,H,T,K] fp32, beta,gamma: [B,H,T] fp32
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    out = torch.empty((B, H, T, Vd), device=q.device, dtype=torch.float32)

    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    scale = float(scale)

    NK = (K + _BLOCK_K - 1) // _BLOCK_K
    NV = (Vd + _BLOCK_V - 1) // _BLOCK_V

    ker = _get_kernel(NK, NV)

    Q_B, Q_H, Q_T, Q_K = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
    K_B, K_H, K_T, K_K = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
    V_B, V_H, V_T, V_V = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
    E_B, E_H, E_T, E_K = e.stride(0), e.stride(1), e.stride(2), e.stride(3)
    G_B, G_H, G_T, G_K = g.stride(0), g.stride(1), g.stride(2), g.stride(3)
    Beta_B, Beta_H, Beta_T = beta.stride(0), beta.stride(1), beta.stride(2)
    Gm_B, Gm_H, Gm_T = gamma.stride(0), gamma.stride(1), gamma.stride(2)
    O_B, O_H, O_T, O_V = out.stride(0), out.stride(1), out.stride(2), out.stride(3)

    grid = (B * H,)
    ker[grid](
        q, k, v, e, g, beta, gamma, out, scale,
        B, H, T, K, Vd,
        Q_B, Q_H, Q_T, Q_K,
        K_B, K_H, K_T, K_K,
        V_B, V_H, V_T, V_V,
        E_B, E_H, E_T, E_K,
        G_B, G_H, G_T, G_K,
        Beta_B, Beta_H, Beta_T,
        Gm_B, Gm_H, Gm_T,
        O_B, O_H, O_T, O_V,
        _BLOCK_K, _BLOCK_V,
    )
    return out
