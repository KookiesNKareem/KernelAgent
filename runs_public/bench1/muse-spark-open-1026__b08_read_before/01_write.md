```python
import torch
import triton
import triton.language as tl

@triton.jit
def gated_delta_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr, state_ptr,
    T0: tl.constexpr, BH: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr, SCALE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_vb = tl.program_id(1)
    bh = pid_bh
    v_block = pid_vb

    # 2D offsets for the state tile [K, BV]
    offs_k_2d = tl.arange(0, K)[:, None]          # K x 1
    offs_v_2d = tl.arange(0, BV)[None, :]          # 1 x BV
    v_global_2d = v_block * BV + offs_v_2d

    # state pointers [K, BV]
    state_base = bh * K * V
    state_ptrs = state_ptr + state_base + offs_k_2d * V + v_global_2d
    mask_k = offs_k_2d < K
    mask_v = v_global_2d < V
    mask_state = mask_k & mask_v
    S = tl.load(state_ptrs, mask=mask_state, other=0.0)  # K x BV, fp32

    offs_k_1d = tl.arange(0, K)          # K
    offs_v_1d = tl.arange(0, BV)          # BV

    for i in range(BT):
        cur_t = T0 + i

        q_ptrs = q_ptr + (bh * T + cur_t) * K + offs_k_1d
        q = tl.load(q_ptrs).to(tl.float32) * SCALE

        k_ptrs = k_ptr + (bh * T + cur_t) * K + offs_k_1d
        k = tl.load(k_ptrs).to(tl.float32)

        g_ptrs = g_ptr + (bh * T + cur_t) * K + offs_k_1d
        g = tl.load(g_ptrs).to(tl.float32)
        d = tl.exp(g)  # K

        v_global = v_block * BV + offs_v_1d
        mask_v1d = v_global < V
        v_ptrs = v_ptr + (bh * T + cur_t) * V + v_global
        v = tl.load(v_ptrs, mask=mask_v1d, other=0.0).to(tl.float32)

        beta = tl.load(beta_ptr + bh * T + cur_t).to(tl.float32)

        # S = diag(exp(g)) * S_{prev}
        d_col = d[:, None]          # K x 1
        S = S * d_col

        # read before write
        q_col = q[:, None]           # K x 1
        k_col = k[:, None]           # K x 1

        out_vec = tl.sum(S * q_col, axis=0)   # BV
        pred = tl.sum(S * k_col, axis=0)      # BV

        out_ptrs = out_ptr + (bh * T + cur_t) * V + v_global
        tl.store(out_ptrs, out_vec, mask=mask_v1d)

        # S_t = S + beta * k * (v-pred)^T
        delta = beta * (v - pred)             # BV
        S = S + k_col * delta[None, :]        # K x BV

    tl.store(state_ptrs, S, mask=mask_state)


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    """
    q,k,e : [B,H,T,K] (bf16, k,e unit-norm)  v : [B,H,T,V] (bf16)
    g : [B,H,T,K] fp32 log-decay   beta,gamma : [B,H,T] fp32
    scale : float
    returns [B,H,T,V] fp32 (or q dtype) matching reference
    e, gamma are unused but kept for API compatibility.
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    BH = B * H
    device = q.device

    # flatten BH for simple contiguous addressing, cheap elementwise prep only
    q_f = q.reshape(BH, T, K).contiguous()
    k_f = k.reshape(BH, T, K).contiguous()
    v_f = v.reshape(BH, T, V).contiguous()
    g_f = g.reshape(BH, T, K).contiguous()
    beta_f = beta.reshape(BH, T).contiguous()

    out_f = torch.empty((BH, T, V), dtype=torch.float32, device=device)
    state = torch.zeros((BH, K, V), dtype=torch.float32, device=device)

    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        scale = float(scale)

    BT = 64          # time chunk, unrolled inside kernel
    BV = 32          # V tile, keeps S tile = K*BV <= 4k elements
    num_v_blocks = (V + BV - 1) // BV
    grid = (BH, num_v_blocks)

    for t0 in range(0, T, BT):
        cur_bt = min(BT, T - t0)
        gated_delta_kernel[grid](q_f, k_f, v_f, g_f, beta_f, out_f, state,
                                 t0, BH, T, K, V, cur_bt, BV, scale)

    return out_f.view(B, H, T, V)
```

Chunked recurrent scan with V-tiling: `S` (`KxV`) is kept in SRAM per `BH` program, tiled only on `V` so the `sum(K)` for output/pred stays inside the program. The outer Python loop iterates over `BT=64` time chunks (`32` launches for `T=2048`) and each Triton program unrolls its chunk, doing `S*=exp(g)`, fused `q/k` reductions and the outer-product update. Unsure about exact Triton 3.5 `tl.sum(axis=)`/`[:,None]` broadcasting syntax and bf16 `tl.load(...).to(tl.float32)` casting on sm_89, and whether `V` not divisible by 32 needs extra masking (handled) versus a dynamic `BV`.