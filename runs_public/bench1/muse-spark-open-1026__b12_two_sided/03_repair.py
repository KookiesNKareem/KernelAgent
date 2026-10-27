import torch
import triton
import triton.language as tl

@triton.jit
def rec_kernel(
    Q_ptr, K_ptr, V_ptr, E_ptr, G_ptr, Beta_ptr, Gamma_ptr, O_ptr, S_ptr, TmpV_ptr, TmpK_ptr,
    T: tl.constexpr, Kdim: tl.constexpr, Vdim: tl.constexpr,
    scale,
    CHUNK_START: tl.constexpr, CUR_T: tl.constexpr,
    NK: tl.constexpr, NV: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)

    for it in range(CUR_T):
        t = CHUNK_START + it
        beta = tl.load(Beta_ptr + pid * T + t).to(tl.float32)
        gamm = tl.load(Gamma_ptr + pid * T + t).to(tl.float32)

        # ----- decay -----
        for kk in range(NK):
            k_off = kk * BLOCK_K
            k_idx = k_off + offs_k
            mask_k = k_idx < Kdim
            g_vec = tl.load(G_ptr + pid * T * Kdim + t * Kdim + k_idx, mask=mask_k, other=0.0).to(tl.float32)
            eg = tl.exp(g_vec)
            for vv in range(NV):
                v_off = vv * BLOCK_V
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                s = s * eg[:, None]
                tl.store(s_ptr, s, boundary_check=(0, 1))

        # ----- y = e^T S  in TmpV -----
        for vv in range(NV):
            v_off = vv * BLOCK_V
            v_idx = v_off + offs_v
            mask_v = v_idx < Vdim
            tl.store(TmpV_ptr + pid * Vdim + v_idx, 0.0, mask=mask_v)
        for kk in range(NK):
            k_off = kk * BLOCK_K
            k_idx = k_off + offs_k
            mask_k = k_idx < Kdim
            e_vec = tl.load(E_ptr + pid * T * Kdim + t * Kdim + k_idx, mask=mask_k, other=0.0).to(tl.float32)
            for vv in range(NV):
                v_off = vv * BLOCK_V
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                contrib = tl.dot(e_vec, s)  # [BV]
                v_idx = v_off + offs_v
                mask_v = v_idx < Vdim
                cur = tl.load(TmpV_ptr + pid * Vdim + v_idx, mask=mask_v, other=0.0).to(tl.float32)
                cur = cur + contrib
                tl.store(TmpV_ptr + pid * Vdim + v_idx, cur, mask=mask_v)

        # ----- left erase -----
        for kk in range(NK):
            k_off = kk * BLOCK_K
            k_idx = k_off + offs_k
            mask_k = k_idx < Kdim
            e_vec = tl.load(E_ptr + pid * T * Kdim + t * Kdim + k_idx, mask=mask_k, other=0.0).to(tl.float32)
            for vv in range(NV):
                v_off = vv * BLOCK_V
                v_idx = v_off + offs_v
                mask_v = v_idx < Vdim
                y_vec = tl.load(TmpV_ptr + pid * Vdim + v_idx, mask=mask_v, other=0.0).to(tl.float32)
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                s = s - gamm * e_vec[:, None] * y_vec[None, :]
                tl.store(s_ptr, s, boundary_check=(0, 1))

        # ----- z = S e in TmpK -----
        for kk in range(NK):
            k_off = kk * BLOCK_K
            k_idx = k_off + offs_k
            mask_k = k_idx < Kdim
            tl.store(TmpK_ptr + pid * Kdim + k_idx, 0.0, mask=mask_k)
        for kk in range(NK):
            k_off = kk * BLOCK_K
            for vv in range(NV):
                v_off = vv * BLOCK_V
                v_idx = v_off + offs_v
                mask_v = v_idx < Vdim
                e_v = tl.load(E_ptr + pid * T * Kdim + t * Kdim + v_idx, mask=mask_v, other=0.0).to(tl.float32)
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                contrib = tl.dot(s, e_v)  # [BK]
                k_idx = k_off + offs_k
                mask_k = k_idx < Kdim
                cur = tl.load(TmpK_ptr + pid * Kdim + k_idx, mask=mask_k, other=0.0).to(tl.float32)
                cur = cur + contrib
                tl.store(TmpK_ptr + pid * Kdim + k_idx, cur, mask=mask_k)

        # ----- right erase -----
        for kk in range(NK):
            k_off = kk * BLOCK_K
            k_idx = k_off + offs_k
            mask_k = k_idx < Kdim
            z_vec = tl.load(TmpK_ptr + pid * Kdim + k_idx, mask=mask_k, other=0.0).to(tl.float32)
            for vv in range(NV):
                v_off = vv * BLOCK_V
                v_idx = v_off + offs_v
                mask_v = v_idx < Vdim
                e_v = tl.load(E_ptr + pid * T * Kdim + t * Kdim + v_idx, mask=mask_v, other=0.0).to(tl.float32)
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                s = s - gamm * z_vec[:, None] * e_v[None, :]
                tl.store(s_ptr, s, boundary_check=(0, 1))

        # ----- pred = k^T S -> TmpV -----
        for vv in range(NV):
            v_off = vv * BLOCK_V
            v_idx = v_off + offs_v
            mask_v = v_idx < Vdim
            tl.store(TmpV_ptr + pid * Vdim + v_idx, 0.0, mask=mask_v)
        for kk in range(NK):
            k_off = kk * BLOCK_K
            k_idx = k_off + offs_k
            mask_k = k_idx < Kdim
            k_vec = tl.load(K_ptr + pid * T * Kdim + t * Kdim + k_idx, mask=mask_k, other=0.0).to(tl.float32)
            for vv in range(NV):
                v_off = vv * BLOCK_V
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                contrib = tl.dot(k_vec, s)
                v_idx = v_off + offs_v
                mask_v = v_idx < Vdim
                cur = tl.load(TmpV_ptr + pid * Vdim + v_idx, mask=mask_v, other=0.0).to(tl.float32)
                cur = cur + contrib
                tl.store(TmpV_ptr + pid * Vdim + v_idx, cur, mask=mask_v)

        # delta = v - pred
        for vv in range(NV):
            v_off = vv * BLOCK_V
            v_idx = v_off + offs_v
            mask_v = v_idx < Vdim
            v_vec = tl.load(V_ptr + pid * T * Vdim + t * Vdim + v_idx, mask=mask_v, other=0.0).to(tl.float32)
            p_vec = tl.load(TmpV_ptr + pid * Vdim + v_idx, mask=mask_v, other=0.0).to(tl.float32)
            tl.store(TmpV_ptr + pid * Vdim + v_idx, v_vec - p_vec, mask=mask_v)

        # ----- S += beta * k * delta^T -----
        for kk in range(NK):
            k_off = kk * BLOCK_K
            k_idx = k_off + offs_k
            mask_k = k_idx < Kdim
            k_vec = tl.load(K_ptr + pid * T * Kdim + t * Kdim + k_idx, mask=mask_k, other=0.0).to(tl.float32)
            for vv in range(NV):
                v_off = vv * BLOCK_V
                v_idx = v_off + offs_v
                mask_v = v_idx < Vdim
                d_vec = tl.load(TmpV_ptr + pid * Vdim + v_idx, mask=mask_v, other=0.0).to(tl.float32)
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                s = s + beta * k_vec[:, None] * d_vec[None, :]
                tl.store(s_ptr, s, boundary_check=(0, 1))

        # ----- out -----
        for vv in range(NV):
            v_off = vv * BLOCK_V
            v_idx = v_off + offs_v
            mask_v = v_idx < Vdim
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for kk in range(NK):
                k_off = kk * BLOCK_K
                k_idx = k_off + offs_k
                mask_k = k_idx < Kdim
                q_vec = tl.load(Q_ptr + pid * T * Kdim + t * Kdim + k_idx, mask=mask_k, other=0.0).to(tl.float32)
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                contrib = tl.dot(q_vec, s)
                acc = acc + contrib
            acc = acc * scale
            tl.store(O_ptr + pid * T * Vdim + t * Vdim + v_idx, acc, mask=mask_v)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    BH = B * H
    q_ = q.contiguous().reshape(BH, T, K)
    k_ = k.contiguous().reshape(BH, T, K)
    v_ = v.contiguous().reshape(BH, T, V)
    e_ = e.contiguous().reshape(BH, T, K)
    g_ = g.contiguous().reshape(BH, T, K)
    beta_ = beta.contiguous().reshape(BH, T)
    gamma_ = gamma.contiguous().reshape(BH, T)
    out_ = torch.empty(BH, T, V, device=q.device, dtype=torch.float32)
    state = torch.zeros(BH, K, V, device=q.device, dtype=torch.float32)
    tmpV = torch.empty(BH, V, device=q.device, dtype=torch.float32)
    tmpK = torch.empty(BH, K, device=q.device, dtype=torch.float32)
    BLOCK_K = 32
    BLOCK_V = 32
    NK = (K + BLOCK_K - 1) // BLOCK_K
    NV = (V + BLOCK_V - 1) // BLOCK_V
    CHUNK = 32
    grid = (BH,)
    sc = float(scale)
    for start in range(0, T, CHUNK):
        cur = min(CHUNK, T - start)
        rec_kernel[grid](q_, k_, v_, e_, g_, beta_, gamma_, out_, state, tmpV, tmpK, T, K, V, sc, start, cur, NK, NV, BLOCK_K, BLOCK_V)
    return out_.reshape(B, H, T, V)
