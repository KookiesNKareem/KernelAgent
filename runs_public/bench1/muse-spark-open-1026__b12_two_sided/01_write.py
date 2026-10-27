import torch
import triton
import triton.language as tl

@triton.jit
def rec_chunk_kernel(
    Q_ptr, K_ptr, V_ptr, E_ptr, G_ptr, Beta_ptr, Gamma_ptr, O_ptr, S_ptr,
    BH, T, K_SZ, V_SZ, scale,
    CHUNK_START: tl.constexpr, CUR_T: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)

    # number of blocks (ceil)
    NK = (K_SZ + BLOCK_K - 1) // BLOCK_K
    NV = (V_SZ + BLOCK_V - 1) // BLOCK_V

    # ---- load state S [K,V] into registers 4x4 tiles ----
    # use Python lists, NK,NV are constexpr -> unrolled
    # we allocate 4x4 max, works for K=V=128 -> NK=4 NV=4
    # for generic NK/NV we allocate with loops
    # to keep IR small we manually handle up to 4 blocks; for larger fall back to masked loops
    # Here we create list-of-lists via Python loops (unrolled at compile time)
    s = [[None for _ in range(4)] for _ in range(4)]
    # actually allocate exact size using Python loops (constexpr)
    # Triton will unroll these loops
    for i in range(NK):
        for j in range(NV):
            off_k = i * BLOCK_K
            off_v = j * BLOCK_V
            s_ptr = tl.make_block_ptr(
                base=S_ptr + pid * K_SZ * V_SZ,
                shape=(K_SZ, V_SZ),
                strides=(V_SZ, 1),
                offsets=(off_k, off_v),
                block_shape=(BLOCK_K, BLOCK_V),
                order=(1, 0),
            )
            s[i][j] = tl.load(s_ptr, boundary_check=(0, 1)).to(tl.float32)

    # ---- time loop inside chunk (unrolled, CUR_T <= BLOCK_T <=64) ----
    for t in range(CUR_T):
        # ---- load vectors for this t ----
        q_blocks = [None for _ in range(4)]
        k_blocks = [None for _ in range(4)]
        e_blocks = [None for _ in range(4)]
        g_blocks = [None for _ in range(4)]
        v_blocks = [None for _ in range(4)]

        for i in range(NK):
            offs = i * BLOCK_K + tl.arange(0, BLOCK_K)
            mask = offs < K_SZ
            base = CHUNK_START + t
            q_ptr = Q_ptr + pid * T * K_SZ + base * K_SZ
            k_ptr = K_ptr + pid * T * K_SZ + base * K_SZ
            e_ptr = E_ptr + pid * T * K_SZ + base * K_SZ
            g_ptr = G_ptr + pid * T * K_SZ + base * K_SZ
            q_blocks[i] = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            k_blocks[i] = tl.load(k_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            e_blocks[i] = tl.load(e_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            g_blocks[i] = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        for j in range(NV):
            offs = j * BLOCK_V + tl.arange(0, BLOCK_V)
            mask = offs < V_SZ
            base = CHUNK_START + t
            v_ptr = V_ptr + pid * T * V_SZ + base * V_SZ
            v_blocks[j] = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        beta_t = tl.load(Beta_ptr + pid * T + CHUNK_START + t).to(tl.float32)
        gamma_t = tl.load(Gamma_ptr + pid * T + CHUNK_START + t).to(tl.float32)

        # ---- 1) decay diag(exp(g)) * S row-wise ----
        for i in range(NK):
            exp_g = tl.exp(g_blocks[i])
            exp_g_2d = exp_g[:, None]
            for j in range(NV):
                s[i][j] = s[i][j] * exp_g_2d

        # ---- 2) left erase: S -= gamma * e * (e^T S) ----
        # y_j = sum_k e_k S_{k,j}
        y_blocks = [None for _ in range(4)]
        for j in range(NV):
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for i in range(NK):
                prod = e_blocks[i][:, None] * s[i][j]
                contrib = tl.sum(prod, axis=0)
                acc = acc + contrib
            y_blocks[j] = acc
        for i in range(NK):
            e_col = e_blocks[i][:, None]
            for j in range(NV):
                y_row = y_blocks[j][None, :]
                s[i][j] = s[i][j] - gamma_t * e_col * y_row

        # ---- 3) right erase: S -= gamma * (S e) e^T ----
        z_blocks = [None for _ in range(4)]
        for i in range(NK):
            acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for j in range(NV):
                # e_j is same vector sliced by V dimension; when K==V this is e_blocks[j]
                # for K!=V we reuse e_blocks with V blocking (zero padded if j>=NK)
                e_j = e_blocks[j] if j < NK else tl.zeros((BLOCK_V,), dtype=tl.float32)
                prod = s[i][j] * e_j[None, :]
                contrib = tl.sum(prod, axis=1)
                acc = acc + contrib
            z_blocks[i] = acc
        for i in range(NK):
            z_col = z_blocks[i][:, None]
            for j in range(NV):
                e_j = e_blocks[j] if j < NK else tl.zeros((BLOCK_V,), dtype=tl.float32)
                e_row = e_j[None, :]
                s[i][j] = s[i][j] - gamma_t * z_col * e_row

        # ---- 4) pred = k^T S ----
        p_blocks = [None for _ in range(4)]
        for j in range(NV):
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for i in range(NK):
                prod = k_blocks[i][:, None] * s[i][j]
                contrib = tl.sum(prod, axis=0)
                acc = acc + contrib
            p_blocks[j] = acc

        delta_blocks = [None for _ in range(4)]
        for j in range(NV):
            delta_blocks[j] = v_blocks[j] - p_blocks[j]

        # ---- 5) S += beta * k * delta^T ----
        for i in range(NK):
            k_col = k_blocks[i][:, None]
            for j in range(NV):
                d_row = delta_blocks[j][None, :]
                s[i][j] = s[i][j] + beta_t * k_col * d_row

        # ---- 6) out = scale * S^T q ----
        for j in range(NV):
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for i in range(NK):
                prod = q_blocks[i][:, None] * s[i][j]
                contrib = tl.sum(prod, axis=0)
                acc = acc + contrib
            acc = acc * scale
            offs = j * BLOCK_V + tl.arange(0, BLOCK_V)
            mask = offs < V_SZ
            o_ptr = O_ptr + pid * T * V_SZ + (CHUNK_START + t) * V_SZ
            tl.store(o_ptr + offs, acc, mask=mask)

    # ---- write back state ----
    for i in range(NK):
        for j in range(NV):
            off_k = i * BLOCK_K
            off_v = j * BLOCK_V
            s_ptr = tl.make_block_ptr(
                base=S_ptr + pid * K_SZ * V_SZ,
                shape=(K_SZ, V_SZ),
                strides=(V_SZ, 1),
                offsets=(off_k, off_v),
                block_shape=(BLOCK_K, BLOCK_V),
                order=(1, 0),
            )
            tl.store(s_ptr, s[i][j], boundary_check=(0, 1))


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    """
    q,k,e : [B,H,T,K] bf16 (k,e unit-norm)
    v     : [B,H,T,V] bf16
    g     : [B,H,T,K] fp32 log decay
    beta,gamma : [B,H,T] fp32
    scale : float
    returns [B,H,T,V] float32 (or q dtype) matching reference
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    BH = B * H

    # flatten B*H for simple contiguous strides, keep dtypes
    q_ = q.reshape(BH, T, K).contiguous()
    k_ = k.reshape(BH, T, K).contiguous()
    v_ = v.reshape(BH, T, V).contiguous()
    e_ = e.reshape(BH, T, K).contiguous()
    g_ = g.reshape(BH, T, K).contiguous()
    beta_ = beta.reshape(BH, T).contiguous()
    gamma_ = gamma.reshape(BH, T).contiguous()

    out_ = torch.empty(BH, T, V, device=q.device, dtype=torch.float32)
    state = torch.zeros(BH, K, V, device=q.device, dtype=torch.float32)

    BLOCK_T = 64
    BLOCK_K = 32
    BLOCK_V = 32

    grid = (BH,)
    # scale as float
    scale_f = float(scale) if not isinstance(scale, float) else scale

    for start in range(0, T, BLOCK_T):
        cur_T = min(BLOCK_T, T - start)
        rec_chunk_kernel[grid](
            q_, k_, v_, e_, g_, beta_, gamma_, out_, state,
            BH, T, K, V, scale_f,
            start, cur_T,
            BLOCK_K, BLOCK_V, BLOCK_T,
        )

    out = out_.reshape(B, H, T, V)
    # reference promotes to float32; we return float32 (allowed) to keep bit-exact
    return out
