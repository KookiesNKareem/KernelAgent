```python
import torch
import triton
import triton.language as tl

@triton.jit
def _chunk_opt(q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, out_ptr, state_ptr,
               scale,
               T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
               Kb: tl.constexpr, Vb: tl.constexpr,
               START: tl.constexpr,
               BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
    pid = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    # iterate over V tiles - bounds are constexpr so fully unrolled
    for v_off in range(0, Vb, BLOCK_V):
        offs_v = v_off + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V

        # state tile [BLOCK_K, BLOCK_V]
        state_base = state_ptr + pid * Kb * Vb
        s = tl.load(state_base + offs_k[:, None] * Vb + offs_v[None, :],
                    mask=mask_k[:, None] & mask_v[None, :], other=0.0).to(tl.float32)

        out_base = out_ptr + pid * T * V

        for i in range(0, BLOCK_C):
            cur = START + i
            valid = cur < T
            cur_c = tl.where(valid, cur, 0)

            d_val = tl.load(d_ptr + pid * T + cur_c, mask=valid, other=1.0).to(tl.float32)
            b_val = tl.load(beta_ptr + pid * T + cur_c, mask=valid, other=0.0).to(tl.float32)

            # vectors
            q_base = q_ptr + pid * T * K + cur_c * K
            k_base = k_ptr + pid * T * K + cur_c * K
            v_base = v_ptr + pid * T * V + cur_c * V

            q_i = tl.load(q_base + offs_k, mask=mask_k & valid, other=0.0).to(tl.float32)
            k_i = tl.load(k_base + offs_k, mask=mask_k & valid, other=0.0).to(tl.float32)
            v_i = tl.load(v_base + offs_v, mask=mask_v & valid, other=0.0).to(tl.float32)

            # p = k^T S  [BLOCK_V]
            p = tl.sum(k_i[:, None] * s, axis=0)

            # S = d*S + b*k*(v - d*p)  == d*(I - b*k*k^T)S + b*k*v^T
            # no division, d=exp(g) in (0,1] so stable for strong decay
            coeff = b_val * (v_i - d_val * p)  # [BLOCK_V]
            s = s * d_val + k_i[:, None] * coeff[None, :]

            o_i = tl.sum(q_i[:, None] * s, axis=0) * scale
            tl.store(out_base + cur_c * V + offs_v, o_i, mask=mask_v & valid)

        tl.store(state_base + offs_k[:, None] * Vb + offs_v[None, :], s,
                 mask=mask_k[:, None] & mask_v[None, :])

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    # q,k,e:[B,H,T,K] bf16  v:[B,H,T,V] bf16  g,beta,gamma:[B,H,T] fp32
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    BH = B * H
    scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)

    # keep original layout, make contiguous only if needed (cheap vs DtoD pad copy)
    # flatten B*H for simple pointer arithmetic: contiguous => stride (T*K,K,1)
    q_f = q.detach().reshape(BH, T, K)
    if not q_f.is_contiguous():
        q_f = q_f.contiguous()
    k_f = k.detach().reshape(BH, T, K)
    if not k_f.is_contiguous():
        k_f = k_f.contiguous()
    v_f = v.detach().reshape(BH, T, Vd)
    if not v_f.is_contiguous():
        v_f = v_f.contiguous()
    g_f = g.detach().reshape(BH, T).to(torch.float32)
    if not g_f.is_contiguous():
        g_f = g_f.contiguous()
    beta_f = beta.detach().reshape(BH, T).to(torch.float32)
    if not beta_f.is_contiguous():
        beta_f = beta_f.contiguous()

    Kb = ((K + 31) // 32) * 32
    Vb = ((Vd + 31) // 32) * 32
    if Kb == 0:
        Kb = 32
    if Vb == 0:
        Vb = 32

    # cheap elementwise prep allowed: exp in PyTorch (fast vectorized) instead of tl.exp
    d_f = torch.exp(g_f)

    state = torch.zeros((BH, Kb, Vb), device=q.device, dtype=torch.float32)
    out = torch.empty((BH, T, Vd), device=q.device, dtype=torch.float32)

    BLOCK_C = 64
    # choose BLOCK_V to keep register usage < ~32KB
    if Vb <= 128:
        BLOCK_V = 64
    else:
        BLOCK_V = 32

    grid = (BH,)
    # BLOCK_K == Kb (padded) - 1 tile keeps full K in SRAM, no K-loop overhead
    for start in range(0, T, BLOCK_C):
        _chunk_opt[grid](q_f, k_f, v_f, d_f, beta_f, out, state,
                         scale_f, T, K, Vd, Kb, Vb, start,
                         BLOCK_C, Kb, BLOCK_V)

    out = out.reshape(B, H, T, Vd)
    if q.dtype != torch.float32:
        out = out.to(q.dtype)
    return out
```

Previous failure was `tl.dot` shape and `beta/exp(cumsum)` division causing `inf` for strong decay (`exp(-25)≈1e-11` underflows in `D0`). Fix pads `K,V` to 32 and computes the recurrence directly as `S*=d , S+=b·k·(v-d·kᵀS)` with `d=exp(g)` per step, never dividing. For speed `exp` is hoisted to `torch.exp`, padding copies are removed via masked loads, `BLOCK_C=64/BLOCK_V=64` halves launches and state HBM traffic, and per-row `tl.sum` extraction is replaced by scalar `tl.load` of `q_i/k_i/v_i`.