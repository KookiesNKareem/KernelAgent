```python
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr,
                  scale,
                  BH, T,
                  K: tl.constexpr, V: tl.constexpr, BLOCK_V: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_v  = tl.program_id(1)

    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offs < V
    k_offs = tl.arange(0, K)

    # S : [K, BLOCK_V] kept in registers, fp32
    S = tl.zeros((K, BLOCK_V), dtype=tl.float32)

    t = 0
    while t < T:
        off_qk = (pid_bh * T + t) * K
        off_v  = (pid_bh * T + t) * V
        off_b  = pid_bh * T + t

        q_vec = tl.load(q_ptr + off_qk + k_offs).to(tl.float32)
        k_vec = tl.load(k_ptr + off_qk + k_offs).to(tl.float32)
        g_vec = tl.load(g_ptr + off_qk + k_offs).to(tl.float32)
        v_vec = tl.load(v_ptr + off_v + v_offs, mask=v_mask, other=0.0).to(tl.float32)
        beta_val = tl.load(beta_ptr + off_b).to(tl.float32)

        # 1) decay: S = diag(exp(g)) * S  (g <=0)
        decay = tl.exp(g_vec)
        S = S * tl.expand_dims(decay, 1)

        # 2) pred = S^T k  (sum over K)
        SK = S * tl.expand_dims(k_vec, 1)
        pred = tl.sum(SK, axis=0)          # [BLOCK_V]

        # 3) delta update: S += k * beta*(v-pred)^T
        delta = beta_val * (v_vec - pred)  # [BLOCK_V]
        S = S + tl.expand_dims(k_vec, 1) * tl.expand_dims(delta, 0)

        # 4) output: o = scale * S^T q
        SQ = S * tl.expand_dims(q_vec * scale, 1)
        out_vec = tl.sum(SQ, axis=0)       # [BLOCK_V]
        tl.store(out_ptr + off_v + v_offs, out_vec, mask=v_mask)

        t += 1


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    """
    q,k,e: [B,H,T,K] bf16 (k,e unit-norm) ; v: [B,H,T,V] bf16
    g: [B,H,T,K] fp32 log-decay <=0 ; beta,gamma: [B,H,T] fp32 in (0,1)
    scale: float
    returns [B,H,T,V] fp32 (matches reference which promotes to fp32)
    e,gamma are unused but kept for API compatibility.
    """
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    BH = B * H

    # make contiguous so flat offset = ((b*H+h)*T+t)*K + k
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = g.contiguous()
    beta_c = beta.contiguous()

    out = torch.empty((B, H, T, Vd), dtype=torch.float32, device=q.device)

    # scale may be python float or 0-d tensor
    if isinstance(scale, torch.Tensor):
        scale_f = float(scale.item())
    else:
        scale_f = float(scale)

    BLOCK_V = 32
    n_vblocks = (Vd + BLOCK_V - 1) // BLOCK_V
    grid = (BH, n_vblocks)

    # K and V must be constexpr for tl.arange / tl.zeros shape
    _fused_kernel[grid](q_c, k_c, v_c, g_c, beta_c, out,
                        scale_f,
                        BH, T,
                        K, Vd, BLOCK_V,
                        num_warps=4, num_stages=2)
    return out
```

Approach: the recurrence is linear in the state `S[K,V]` with per-step `A_t= (I-βk kᵀ)diag(e^{g})`; keeping `S` in registers and streaming `q,k,v,g,β` fuses the 2048 Python launches. Splitting only the `V` dimension across programs is sufficient because `pred`, the delta update and the output all reduce over `K` and are independent per `V`-slice, so no cross-program reduction is needed; `S` is stored as `[K,BLOCK_V]` and updated fully in FP32.

Unsure: exact Triton broadcasting/`expand_dims` spelling and whether `while t<T` with `BH,T` non-constexpr is accepted on Triton 3.5/sm89 vs requiring `tl.range`, and register pressure of a `128×32` FP32 tile per program.