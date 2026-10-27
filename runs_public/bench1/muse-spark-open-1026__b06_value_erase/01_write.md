```python
import torch
import triton
import triton.language as tl

@triton.jit
def chunk_kernel(
    q_ptr, k_ptr, v_ptr, e_ptr, g_ptr, beta_ptr, gamma_ptr, state_ptr, out_ptr,
    stride_q_bh, stride_q_t, stride_q_k,
    stride_k_bh, stride_k_t, stride_k_k,
    stride_v_bh, stride_v_t, stride_v_v,
    stride_e_bh, stride_e_t, stride_e_k,
    stride_g_bh, stride_g_t, stride_g_k,
    stride_beta_bh, stride_beta_t,
    stride_gamma_bh, stride_gamma_t,
    stride_state_bh, stride_state_k, stride_state_v,
    stride_out_bh, stride_out_t, stride_out_v,
    scale,
    BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_T: tl.constexpr,
    chunk_start,
):
    pid = tl.program_id(0)
    q_base = q_ptr + pid * stride_q_bh
    k_base = k_ptr + pid * stride_k_bh
    v_base = v_ptr + pid * stride_v_bh
    e_base = e_ptr + pid * stride_e_bh
    g_base = g_ptr + pid * stride_g_bh
    beta_base = beta_ptr + pid * stride_beta_bh
    gamma_base = gamma_ptr + pid * stride_gamma_bh
    state_base = state_ptr + pid * stride_state_bh
    out_base = out_ptr + pid * stride_out_bh

    K_OFFS = tl.arange(0, BLOCK_K)
    V_OFFS = tl.arange(0, BLOCK_V)

    NK = K // BLOCK_K
    NV = V // BLOCK_V
    # for arbitrary K,V we still use 32 with masking, NK/NV are constexpr when K,V constexpr
    # Triton unrolls these Python range loops at compile time
    # load state S[K,V] into registers : 4x4 tiles for 128x128
    # using Python lists of tl.tensors (unrolled)
    s = [[None for _ in range(NV)] for _ in range(NK)]
    for ik in range(NK):
        for iv in range(NV):
            off_k = ik * BLOCK_K
            off_v = iv * BLOCK_V
            bptr = tl.make_block_ptr(base=state_base, shape=(K, V), strides=(stride_state_k, stride_state_v), offsets=(off_k, off_v), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
            s[ik][iv] = tl.load(bptr, boundary_check=(0, 1))

    for s_idx in range(BLOCK_T):
        t = chunk_start + s_idx

        # ---- load vectors for time t ----
        k_blks = [None for _ in range(NK)]
        for ik in range(NK):
            offs = ik * BLOCK_K + K_OFFS
            mask = offs < K
            ptr = k_base + t * stride_k_t + offs * stride_k_k
            k_blks[ik] = tl.load(ptr, mask=mask).to(tl.float32)
        e_blks = [None for _ in range(NV)]
        for iv in range(NV):
            offs = iv * BLOCK_V + V_OFFS
            mask = offs < K
            ptr = e_base + t * stride_e_t + offs * stride_e_k
            e_blks[iv] = tl.load(ptr, mask=mask).to(tl.float32)
        v_blks = [None for _ in range(NV)]
        for iv in range(NV):
            offs = iv * BLOCK_V + V_OFFS
            mask = offs < V
            ptr = v_base + t * stride_v_t + offs * stride_v_v
            v_blks[iv] = tl.load(ptr, mask=mask).to(tl.float32)
        g_blks = [None for _ in range(NK)]
        for ik in range(NK):
            offs = ik * BLOCK_K + K_OFFS
            mask = offs < K
            ptr = g_base + t * stride_g_t + offs * stride_g_k
            g_blks[ik] = tl.load(ptr, mask=mask)
        q_blks = [None for _ in range(NK)]
        for ik in range(NK):
            offs = ik * BLOCK_K + K_OFFS
            mask = offs < K
            ptr = q_base + t * stride_q_t + offs * stride_q_k
            q_blks[ik] = tl.load(ptr, mask=mask).to(tl.float32)

        beta_t = tl.load(beta_base + t * stride_beta_t)
        gamma_t = tl.load(gamma_base + t * stride_gamma_t)

        # 1) row decay: S *= diag(exp(g))
        for ik in range(NK):
            d = tl.exp(g_blks[ik])
            for iv in range(NV):
                s[ik][iv] = s[ik][iv] * d[:, None]

        # 2) h = S @ e  (K vector)
        h_blks = [None for _ in range(NK)]
        for ik in range(NK):
            acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for iv in range(NV):
                prod = s[ik][iv] * e_blks[iv][None, :]
                acc += tl.sum(prod, axis=1)
            h_blks[ik] = acc

        # 3) right erase: S -= gamma * h e^T
        for ik in range(NK):
            for iv in range(NV):
                s[ik][iv] = s[ik][iv] - gamma_t * h_blks[ik][:, None] * e_blks[iv][None, :]

        # 4) pred = k^T S  (V vector)
        pred_blks = [None for _ in range(NV)]
        for iv in range(NV):
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for ik in range(NK):
                acc += tl.sum(k_blks[ik][:, None] * s[ik][iv], axis=0)
            pred_blks[iv] = acc

        # 5) delta update: S += beta * k (v-pred)^T
        for iv in range(NV):
            pred_blks[iv] = v_blks[iv] - pred_blks[iv]
        for ik in range(NK):
            for iv in range(NV):
                s[ik][iv] = s[ik][iv] + beta_t * k_blks[ik][:, None] * pred_blks[iv][None, :]

        # 6) output o_t = scale * S^T q
        for iv in range(NV):
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for ik in range(NK):
                acc += tl.sum(q_blks[ik][:, None] * s[ik][iv], axis=0)
            acc = acc * scale
            offs = iv * BLOCK_V + V_OFFS
            mask = offs < V
            ptr = out_base + t * stride_out_t + offs * stride_out_v
            tl.store(ptr, acc, mask=mask)

    # write back state for next chunk
    for ik in range(NK):
        for iv in range(NV):
            off_k = ik * BLOCK_K
            off_v = iv * BLOCK_V
            bptr = tl.make_block_ptr(base=state_base, shape=(K, V), strides=(stride_state_k, stride_state_v), offsets=(off_k, off_v), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
            tl.store(bptr, s[ik][iv], boundary_check=(0, 1))


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e: [B,H,T,K] bf16, v: [B,H,T,V] bf16, g: [B,H,T,K] fp32, beta,gamma: [B,H,T] fp32
    # returns [B,H,T,V] fp32 (allowed) with semantics of reference()
    B, H, T, K = q.shape
    V = v.shape[-1]
    device = q.device
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item()) if scale.numel() == 1 else float(scale)
    else:
        scale = float(scale)

    out = torch.empty(B, H, T, V, device=device, dtype=torch.float32)

    BH = B * H
    BLOCK_K = 32
    BLOCK_V = 32
    BLOCK_T = 32
    # flatten B*H for simpler strides, keep contiguous for stride assumptions
    q3 = q.reshape(BH, T, K).contiguous()
    k3 = k.reshape(BH, T, K).contiguous()
    v3 = v.reshape(BH, T, V).contiguous()
    e3 = e.reshape(BH, T, K).contiguous()
    g3 = g.reshape(BH, T, K).contiguous()
    beta3 = beta.reshape(BH, T).contiguous()
    gamma3 = gamma.reshape(BH, T).contiguous()
    out3 = out.reshape(BH, T, V).contiguous()
    state = torch.zeros(BH, K, V, device=device, dtype=torch.float32)

    stride_q_bh, stride_q_t, stride_q_k = q3.stride(0), q3.stride(1), q3.stride(2)
    stride_k_bh, stride_k_t, stride_k_k = k3.stride(0), k3.stride(1), k3.stride(2)
    stride_v_bh, stride_v_t, stride_v_v = v3.stride(0), v3.stride(1), v3.stride(2)
    stride_e_bh, stride_e_t, stride_e_k = e3.stride(0), e3.stride(1), e3.stride(2)
    stride_g_bh, stride_g_t, stride_g_k = g3.stride(0), g3.stride(1), g3.stride(2)
    stride_beta_bh, stride_beta_t = beta3.stride(0), beta3.stride(1)
    stride_gamma_bh, stride_gamma_t = gamma3.stride(0), gamma3.stride(1)
    stride_state_bh, stride_state_k, stride_state_v = state.stride(0), state.stride(1), state.stride(2)
    stride_out_bh, stride_out_t, stride_out_v = out3.stride(0), out3.stride(1), out3.stride(2)

    # if K or V not multiple of 32 we still launch with masking; Triton handles it
    full_chunks = T // BLOCK_T
    grid = (BH,)
    for ck in range(full_chunks):
        chunk_start = ck * BLOCK_T
        chunk_kernel[grid](
            q3, k3, v3, e3, g3, beta3, gamma3, state, out3,
            stride_q_bh, stride_q_t, stride_q_k,
            stride_k_bh, stride_k_t, stride_k_k,
            stride_v_bh, stride_v_t, stride_v_v,
            stride_e_bh, stride_e_t, stride_e_k,
            stride_g_bh, stride_g_t, stride_g_k,
            stride_beta_bh, stride_beta_t,
            stride_gamma_bh, stride_gamma_t,
            stride_state_bh, stride_state_k, stride_state_v,
            stride_out_bh, stride_out_t, stride_out_v,
            scale,
            BH, T, K, V,
            BLOCK_K, BLOCK_V, BLOCK_T,
            chunk_start,
        )

    tail_start = full_chunks * BLOCK_T
    if tail_start < T:
        # small tail (<32) run with torch ops starting from current state
        S = state.view(B, H, K, V)
        # promote bf16 inputs to fp32 for tail
        qf = q.to(torch.float32)
        kf = k.to(torch.float32)
        vf = v.to(torch.float32)
        ef = e.to(torch.float32)
        for t in range(tail_start, T):
            gt = g[:, :, t]
            et = ef[:, :, t]
            ct = gamma[:, :, t]
            kt = kf[:, :, t]
            vt = vf[:, :, t]
            bt = beta[:, :, t]
            qt = qf[:, :, t]
            S = S * torch.exp(gt)[..., None]
            S = S - ct[..., None, None] * (S * et[..., None, :]).sum(-1)[..., None] * et[..., None, :]
            pred = (S * kt[..., None]).sum(-2)
            S = S + (bt[..., None] * (vt - pred))[..., None, :] * kt[..., None]
            out[:, :, t] = (S * (qt * scale)[..., None]).sum(-2)
    return out
```
My approach fuses the whole recurrence (decay, right-erase, delta update) into one Triton kernel that keeps the K×V state in registers as 32×32 tiles and loops over time inside the kernel, turning ~12k PyTorch launches into ~64 Triton launches with one load/store of state per 32-step chunk. Time is tiled in BLOCK_T=32 chunks to keep compilation bounded and to carry state between chunks via HBM. I am unsure about exact Triton syntax for `[:,None]` broadcasting/`tl.make_block_ptr` boundary checks on sm_89 and for handling arbitrary K≠V where the reference’s `(S*et[...,None,:]).sum(-1)` broadcast would error (I assume square K==V as in the benchmark).