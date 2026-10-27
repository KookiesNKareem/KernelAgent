import torch
import triton
import triton.language as tl

_kernel_cache = {}

def _get_kernel(num_k_blocks: int):
    if num_k_blocks in _kernel_cache:
        return _kernel_cache[num_k_blocks]
    lines = []
    lines.append("@triton.jit")
    lines.append(f"def fwd_kernel_{num_k_blocks}(q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, out_ptr,")
    lines.append("    stride_qb, stride_qh, stride_qt, stride_qk,")
    lines.append("    stride_kb, stride_kh, stride_kt, stride_kk,")
    lines.append("    stride_vb, stride_vh, stride_vt, stride_vk,")
    lines.append("    stride_eb, stride_eh, stride_et, stride_ek,")
    lines.append("    stride_gb, stride_gh, stride_gt, stride_gk,")
    lines.append("    stride_betab, stride_betah, stride_betat,")
    lines.append("    stride_gammab, stride_gammah, stride_gammat,")
    lines.append("    stride_ob, stride_oh, stride_ot, stride_ok,")
    lines.append("    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NUM_V_TILES: tl.constexpr,")
    lines.append("    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, SCALE: tl.constexpr):")
    lines.append("    pid = tl.program_id(0)")
    lines.append("    pid_v = pid % NUM_V_TILES")
    lines.append("    bh = pid // NUM_V_TILES")
    lines.append("    h = bh % H")
    lines.append("    b = bh // H")
    lines.append("    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)")
    lines.append("    mask_v = offs_v < V")
    for i in range(num_k_blocks):
        lines.append(f"    s{i} = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)")
    for i in range(num_k_blocks):
        lines.append(f"    offs_k{i} = tl.arange(0, BLOCK_K) + {i*32}")
    lines.append("    q_base = q_ptr + b*stride_qb + h*stride_qh")
    lines.append("    k_base = k_ptr + b*stride_kb + h*stride_kh")
    lines.append("    e_base = e_ptr + b*stride_eb + h*stride_eh")
    lines.append("    g_base = g_ptr + b*stride_gb + h*stride_gh")
    lines.append("    v_base = v_ptr + b*stride_vb + h*stride_vh")
    lines.append("    beta_base = beta_ptr + b*stride_betab + h*stride_betah")
    lines.append("    gamma_base = gamma_ptr + b*stride_gammab + h*stride_gammah")
    lines.append("    out_base = out_ptr + b*stride_ob + h*stride_oh")
    lines.append("    for t in range(T):")
    for i in range(num_k_blocks):
        lines.append(f"        k{i} = tl.load(k_base + t*stride_kt + offs_k{i}*stride_kk, mask=offs_k{i} < K, other=0.0)")
        lines.append(f"        q{i} = tl.load(q_base + t*stride_qt + offs_k{i}*stride_qk, mask=offs_k{i} < K, other=0.0)")
        lines.append(f"        e{i} = tl.load(e_base + t*stride_et + offs_k{i}*stride_ek, mask=offs_k{i} < K, other=0.0)")
        lines.append(f"        g{i} = tl.load(g_base + t*stride_gt + offs_k{i}*stride_gk, mask=offs_k{i} < K, other=0.0)")
    lines.append("        vt = tl.load(v_base + t*stride_vt + offs_v*stride_vk, mask=mask_v, other=0.0)")
    lines.append("        beta = tl.load(beta_base + t*stride_betat)")
    lines.append("        gamma = tl.load(gamma_base + t*stride_gammat)")
    for i in range(num_k_blocks):
        lines.append(f"        k{i} = k{i}.to(tl.float32)")
        lines.append(f"        q{i} = q{i}.to(tl.float32)")
        lines.append(f"        e{i} = e{i}.to(tl.float32)")
        lines.append(f"        g{i} = g{i}.to(tl.float32)")
    lines.append("        vt = vt.to(tl.float32)")
    for i in range(num_k_blocks):
        lines.append(f"        d{i} = tl.exp(g{i})")
        lines.append(f"        s{i} = s{i} * d{i}[:, None]")
    for i in range(num_k_blocks):
        lines.append(f"        p{i} = tl.sum(s{i} * k{i}[:, None], axis=0)")
    pred_expr = " + ".join([f"p{i}" for i in range(num_k_blocks)])
    lines.append(f"        pred = {pred_expr}")
    lines.append("        delta = beta * (vt - pred)")
    for i in range(num_k_blocks):
        lines.append(f"        s{i} = s{i} + k{i}[:, None] * delta[None, :] + gamma * e{i}[:, None] * vt[None, :]")
    for i in range(num_k_blocks):
        lines.append(f"        o{i} = tl.sum(s{i} * q{i}[:, None], axis=0)")
    out_expr = " + ".join([f"o{i}" for i in range(num_k_blocks)])
    lines.append(f"        out = SCALE * ({out_expr})")
    lines.append("        tl.store(out_base + t*stride_ot + offs_v*stride_ok, out, mask=mask_v)")
    src = "\n".join(lines)
    gdict = {"triton": triton, "tl": tl}
    exec(src, gdict)
    ker = gdict[f"fwd_kernel_{num_k_blocks}"]
    _kernel_cache[num_k_blocks] = ker
    return ker

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    """
    q,k,e: [B,H,T,K]  k,e unit-norm, bf16
    v: [B,H,T,V] bf16
    g: [B,H,T,K] fp32 log-decay <=0
    beta,gamma: [B,H,T] fp32 in (0,1)
    scale: float
    returns [B,H,T,V] float32 (or q dtype) matching reference
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    BLOCK_K = 32
    BLOCK_V = 32
    num_k_blocks = (K + BLOCK_K - 1) // BLOCK_K
    num_v_tiles = (V + BLOCK_V - 1) // BLOCK_V

    out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    stride_qb, stride_qh, stride_qt, stride_qk = q.stride()
    stride_kb, stride_kh, stride_kt, stride_kk = k.stride()
    stride_vb, stride_vh, stride_vt, stride_vk = v.stride()
    stride_eb, stride_eh, stride_et, stride_ek = e.stride()
    stride_gb, stride_gh, stride_gt, stride_gk = g.stride()
    stride_betab, stride_betah, stride_betat = beta.stride()
    stride_gammab, stride_gammah, stride_gammat = gamma.stride()
    stride_ob, stride_oh, stride_ot, stride_ok = out.stride()

    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)

    ker = _get_kernel(num_k_blocks)
    grid = (B * H * num_v_tiles,)
    ker[grid](
        q, k, v, e, g, beta, gamma, out,
        stride_qb, stride_qh, stride_qt, stride_qk,
        stride_kb, stride_kh, stride_kt, stride_kk,
        stride_vb, stride_vh, stride_vt, stride_vk,
        stride_eb, stride_eh, stride_et, stride_ek,
        stride_gb, stride_gh, stride_gt, stride_gk,
        stride_betab, stride_betah, stride_betat,
        stride_gammab, stride_gammah, stride_gammat,
        stride_ob, stride_oh, stride_ot, stride_ok,
        B, H, T, K, V, num_v_tiles,
        BLOCK_K, BLOCK_V, scale_f,
    )
    return out
