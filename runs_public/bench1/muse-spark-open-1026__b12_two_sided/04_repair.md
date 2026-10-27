```python
import torch
import triton
import triton.language as tl

@triton.jit
def rec_kernel(
    Q_ptr, K_ptr, V_ptr, E_ptr, G_ptr, Beta_ptr, Gamma_ptr, O_ptr, S_ptr,
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

        # decay
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

        # y = e^T S  (V)
        y0 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        y1 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        y2 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        y3 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        y4 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for kk in range(NK):
            k_off = kk * BLOCK_K
            k_idx = k_off + offs_k
            mask_k = k_idx < Kdim
            e_vec = tl.load(E_ptr + pid * T * Kdim + t * Kdim + k_idx, mask=mask_k, other=0.0).to(tl.float32)
            for vv in range(NV):
                v_off = vv * BLOCK_V
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                contrib = tl.sum(e_vec[:, None] * s, axis=0)
                if vv == 0:
                    y0 = y0 + contrib
                elif vv == 1:
                    y1 = y1 + contrib
                elif vv == 2:
                    y2 = y2 + contrib
                elif vv == 3:
                    y3 = y3 + contrib
                elif vv == 4:
                    y4 = y4 + contrib

        # left erase
        for kk in range(NK):
            k_off = kk * BLOCK_K
            k_idx = k_off + offs_k
            mask_k = k_idx < Kdim
            e_vec = tl.load(E_ptr + pid * T * Kdim + t * Kdim + k_idx, mask=mask_k, other=0.0).to(tl.float32)
            for vv in range(NV):
                v_off = vv * BLOCK_V
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                if vv == 0:
                    s = s - gamm * e_vec[:, None] * y0[None, :]
                elif vv == 1:
                    s = s - gamm * e_vec[:, None] * y1[None, :]
                elif vv == 2:
                    s = s - gamm * e_vec[:, None] * y2[None, :]
                elif vv == 3:
                    s = s - gamm * e_vec[:, None] * y3[None, :]
                elif vv == 4:
                    s = s - gamm * e_vec[:, None] * y4[None, :]
                tl.store(s_ptr, s, boundary_check=(0, 1))

        # z = S e  (K)
        z0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
        z1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
        z2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
        z3 = tl.zeros((BLOCK_K,), dtype=tl.float32)
        z4 = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for kk in range(NK):
            k_off = kk * BLOCK_K
            for vv in range(NV):
                v_off = vv * BLOCK_V
                v_idx = v_off + offs_v
                mask_v = v_idx < Vdim
                e_v = tl.load(E_ptr + pid * T * Kdim + t * Kdim + v_idx, mask=mask_v, other=0.0).to(tl.float32)
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                contrib = tl.sum(s * e_v[None, :], axis=1)
                if kk == 0:
                    z0 = z0 + contrib
                elif kk == 1:
                    z1 = z1 + contrib
                elif kk == 2:
                    z2 = z2 + contrib
                elif kk == 3:
                    z3 = z3 + contrib
                elif kk == 4:
                    z4 = z4 + contrib

        # right erase
        for kk in range(NK):
            k_off = kk * BLOCK_K
            if kk == 0:
                zv = z0
            elif kk == 1:
                zv = z1
            elif kk == 2:
                zv = z2
            elif kk == 3:
                zv = z3
            elif kk == 4:
                zv = z4
            for vv in range(NV):
                v_off = vv * BLOCK_V
                v_idx = v_off + offs_v
                mask_v = v_idx < Vdim
                e_v = tl.load(E_ptr + pid * T * Kdim + t * Kdim + v_idx, mask=mask_v, other=0.0).to(tl.float32)
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                s = s - gamm * zv[:, None] * e_v[None, :]
                tl.store(s_ptr, s, boundary_check=(0, 1))

        # pred = k^T S
        p0 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        p1 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        p2 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        p3 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        p4 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for kk in range(NK):
            k_off = kk * BLOCK_K
            k_idx = k_off + offs_k
            mask_k = k_idx < Kdim
            k_vec = tl.load(K_ptr + pid * T * Kdim + t * Kdim + k_idx, mask=mask_k, other=0.0).to(tl.float32)
            for vv in range(NV):
                v_off = vv * BLOCK_V
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                contrib = tl.sum(k_vec[:, None] * s, axis=0)
                if vv == 0:
                    p0 = p0 + contrib
                elif vv == 1:
                    p1 = p1 + contrib
                elif vv == 2:
                    p2 = p2 + contrib
                elif vv == 3:
                    p3 = p3 + contrib
                elif vv == 4:
                    p4 = p4 + contrib

        # delta = v - pred
        d0 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        d1 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        d2 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        d3 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        d4 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for vv in range(NV):
            v_off = vv * BLOCK_V
            v_idx = v_off + offs_v
            mask_v = v_idx < Vdim
            v_vec = tl.load(V_ptr + pid * T * Vdim + t * Vdim + v_idx, mask=mask_v, other=0.0).to(tl.float32)
            if vv == 0:
                d0 = v_vec - p0
            elif vv == 1:
                d1 = v_vec - p1
            elif vv == 2:
                d2 = v_vec - p2
            elif vv == 3:
                d3 = v_vec - p3
            elif vv == 4:
                d4 = v_vec - p4

        # S += beta * k * delta^T
        for kk in range(NK):
            k_off = kk * BLOCK_K
            k_idx = k_off + offs_k
            mask_k = k_idx < Kdim
            k_vec = tl.load(K_ptr + pid * T * Kdim + t * Kdim + k_idx, mask=mask_k, other=0.0).to(tl.float32)
            for vv in range(NV):
                v_off = vv * BLOCK_V
                if vv == 0:
                    dv = d0
                elif vv == 1:
                    dv = d1
                elif vv == 2:
                    dv = d2
                elif vv == 3:
                    dv = d3
                elif vv == 4:
                    dv = d4
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                s = s + beta * k_vec[:, None] * dv[None, :]
                tl.store(s_ptr, s, boundary_check=(0, 1))

        # out = scale * S^T q
        o0 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        o1 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        o2 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        o3 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        o4 = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for kk in range(NK):
            k_off = kk * BLOCK_K
            k_idx = k_off + offs_k
            mask_k = k_idx < Kdim
            q_vec = tl.load(Q_ptr + pid * T * Kdim + t * Kdim + k_idx, mask=mask_k, other=0.0).to(tl.float32)
            for vv in range(NV):
                v_off = vv * BLOCK_V
                s_ptr = tl.make_block_ptr(base=S_ptr + pid * Kdim * Vdim, shape=(Kdim, Vdim), strides=(Vdim, 1), offsets=(k_off, v_off), block_shape=(BLOCK_K, BLOCK_V), order=(1, 0))
                s = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)
                contrib = tl.sum(q_vec[:, None] * s, axis=0)
                if vv == 0:
                    o0 = o0 + contrib
                elif vv == 1:
                    o1 = o1 + contrib
                elif vv == 2:
                    o2 = o2 + contrib
                elif vv == 3:
                    o3 = o3 + contrib
                elif vv == 4:
                    o4 = o4 + contrib
        for vv in range(NV):
            v_off = vv * BLOCK_V
            v_idx = v_off + offs_v
            mask_v = v_idx < Vdim
            if vv == 0:
                tl.store(O_ptr + pid * T * Vdim + t * Vdim + v_idx, o0 * scale, mask=mask_v)
            elif vv == 1:
                tl.store(O_ptr + pid * T * Vdim + t * Vdim + v_idx, o1 * scale, mask=mask_v)
            elif vv == 2:
                tl.store(O_ptr + pid * T * Vdim + t * Vdim + v_idx, o2 * scale, mask=mask_v)
            elif vv == 3:
                tl.store(O_ptr + pid * T * Vdim + t * Vdim + v_idx, o3 * scale, mask=mask_v)
            elif vv == 4:
                tl.store(O_ptr + pid * T * Vdim + t * Vdim + v_idx, o4 * scale, mask=mask_v)


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
    BLOCK_K = 32
    BLOCK_V = 32
    NK = (K + BLOCK_K - 1) // BLOCK_K
    NV = (V + BLOCK_V - 1) // BLOCK_V
    CHUNK = 32
    grid = (BH,)
    sc = float(scale)
    for start in range(0, T, CHUNK):
        cur = min(CHUNK, T - start)
        rec_kernel[grid](q_, k_, v_, e_, g_, beta_, gamma_, out_, state, T, K, V, sc, start, cur, NK, NV, BLOCK_K, BLOCK_V)
    return out_.reshape(B, H, T, V)
```
Previous huge errors came from wrong reductions via `tl.dot` and unstable masked HBM temporaries; the compilation errors came from `tl.store` of scalar to vector and non-constexpr arithmetic. This version uses only `tl.sum(...,axis)` reductions, keeps all intermediates in registers with explicit 5-way unrolling, passes NK/NV as constexpr and handles arbitrary strides by contiguous copies.