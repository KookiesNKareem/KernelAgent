```python
import torch
import triton
import triton.language as tl

# fallback reference for CPU / testing - same semantics as prompt
def _reference(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    q, k, v, e, g, beta, gamma = (x.to(torch.promote_types(x.dtype, torch.float32))
                                  for x in (q, k, v, e, g, beta, gamma))
    S = q.new_zeros(B, H, K, V)
    outs = []
    for t in range(T):
        kt, vt, et, bt = k[:, :, t], v[:, :, t], e[:, :, t], beta[:, :, t]
        pred = (S * kt[..., None]).sum(-2)
        S = S + (bt[..., None] * (vt - pred))[..., None, :] * kt[..., None]
        S = S * g[:, :, t].exp()[..., None]
        outs.append((S * (q[:, :, t] * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)


@triton.jit
def _chunk_fwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr, state_ptr,
    BH, T, K, V, t0,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, CHUNK: tl.constexpr, SCALE: tl.constexpr, NUM_V_BLOCKS: tl.constexpr,
):
    """
    One program = one (BH, V-block).  Holds S[K, BLOCK_V] in SRAM as 4 tiles 32x32.
    Processes CHUNK time steps sequentially, carrying S in registers.
    State buffer [BH, K, V] float32 is read at chunk start and written at chunk end.
    Decay is applied after the write exactly as reference: S = (S + beta*k*(v-k^T S))*exp(g)
    Output o_t = SCALE * S_t^T q_t.
    e, gamma are unused and not passed.
    """
    pid = tl.program_id(0)
    bh = pid // NUM_V_BLOCKS
    vid = pid % NUM_V_BLOCKS

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    cur_off_v = vid * BLOCK_V + offs_v
    mask_v = cur_off_v < V

    # ---- load state S[K, BLOCK_V] as 4 tiles K=128, BLOCK_K=32 ----
    # layout state[ BH, K, V ] contiguous: stride BH=K*V, stride K=V
    base_state = state_ptr + bh * K * V
    # use 2D offsets for block load
    # S0 : k 0..31
    off_k0 = 0 * BLOCK_K + offs_k
    ptr0 = base_state + off_k0[:, None] * V + cur_off_v[None, :]
    S0 = tl.load(ptr0, mask=mask_v[None, :], other=0.0).to(tl.float32)
    off_k1 = 1 * BLOCK_K + offs_k
    ptr1 = base_state + off_k1[:, None] * V + cur_off_v[None, :]
    S1 = tl.load(ptr1, mask=mask_v[None, :], other=0.0).to(tl.float32)
    off_k2 = 2 * BLOCK_K + offs_k
    ptr2 = base_state + off_k2[:, None] * V + cur_off_v[None, :]
    S2 = tl.load(ptr2, mask=mask_v[None, :], other=0.0).to(tl.float32)
    off_k3 = 3 * BLOCK_K + offs_k
    ptr3 = base_state + off_k3[:, None] * V + cur_off_v[None, :]
    S3 = tl.load(ptr3, mask=mask_v[None, :], other=0.0).to(tl.float32)

    # base pointers for time-major tensors [BH, T, K] contiguous: base = bh*T*K
    base_q = q_ptr + bh * T * K
    base_k = k_ptr + bh * T * K
    base_g = g_ptr + bh * T * K
    base_v = v_ptr + bh * T * V
    base_o = out_ptr + bh * T * V
    base_beta = beta_ptr + bh * T

    # ---- sequential over CHUNK (unrolled, CHUNK=64) ----
    for c in range(CHUNK):
        t = t0 + c  # runtime scalar, c is literal

        # loads for this time step (K slices)
        k0 = tl.load(base_k + t * K + 0 * BLOCK_K + offs_k).to(tl.float32)
        k1 = tl.load(base_k + t * K + 1 * BLOCK_K + offs_k).to(tl.float32)
        k2 = tl.load(base_k + t * K + 2 * BLOCK_K + offs_k).to(tl.float32)
        k3 = tl.load(base_k + t * K + 3 * BLOCK_K + offs_k).to(tl.float32)

        q0 = tl.load(base_q + t * K + 0 * BLOCK_K + offs_k).to(tl.float32)
        q1 = tl.load(base_q + t * K + 1 * BLOCK_K + offs_k).to(tl.float32)
        q2 = tl.load(base_q + t * K + 2 * BLOCK_K + offs_k).to(tl.float32)
        q3 = tl.load(base_q + t * K + 3 * BLOCK_K + offs_k).to(tl.float32)

        g0 = tl.load(base_g + t * K + 0 * BLOCK_K + offs_k).to(tl.float32)
        g1 = tl.load(base_g + t * K + 1 * BLOCK_K + offs_k).to(tl.float32)
        g2 = tl.load(base_g + t * K + 2 * BLOCK_K + offs_k).to(tl.float32)
        g3 = tl.load(base_g + t * K + 3 * BLOCK_K + offs_k).to(tl.float32)

        beta_t = tl.load(base_beta + t).to(tl.float32)
        v_cur = tl.load(base_v + t * V + cur_off_v, mask=mask_v, other=0.0).to(tl.float32)

        # pred = k^T S  (BLOCK_V)
        pred = tl.zeros((BLOCK_V,), dtype=tl.float32)
        pred += tl.sum(S0 * k0[:, None], axis=0)
        pred += tl.sum(S1 * k1[:, None], axis=0)
        pred += tl.sum(S2 * k2[:, None], axis=0)
        pred += tl.sum(S3 * k3[:, None], axis=0)

        b_err = beta_t * (v_cur - pred)  # [BLOCK_V]

        # S += k outer b_err
        S0 = S0 + k0[:, None] * b_err[None, :]
        S1 = S1 + k1[:, None] * b_err[None, :]
        S2 = S2 + k2[:, None] * b_err[None, :]
        S3 = S3 + k3[:, None] * b_err[None, :]

        # decay after write: S *= exp(g) row-wise
        d0 = tl.exp(g0)
        d1 = tl.exp(g1)
        d2 = tl.exp(g2)
        d3 = tl.exp(g3)
        S0 = S0 * d0[:, None]
        S1 = S1 * d1[:, None]
        S2 = S2 * d2[:, None]
        S3 = S3 * d3[:, None]

        # o = SCALE * q^T S  (BLOCK_V)
        out = tl.zeros((BLOCK_V,), dtype=tl.float32)
        out += tl.sum(S0 * q0[:, None], axis=0)
        out += tl.sum(S1 * q1[:, None], axis=0)
        out += tl.sum(S2 * q2[:, None], axis=0)
        out += tl.sum(S3 * q3[:, None], axis=0)
        out = out * SCALE

        tl.store(base_o + t * V + cur_off_v, out, mask=mask_v)

    # ---- write back state ----
    tl.store(ptr0, S0, mask=mask_v[None, :])
    tl.store(ptr1, S1, mask=mask_v[None, :])
    tl.store(ptr2, S2, mask=mask_v[None, :])
    tl.store(ptr3, S3, mask=mask_v[None, :])


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    """
    Fast Triton implementation. Signature exactly as reference.
    q,k,e:[B,H,T,K] bf16, v:[B,H,T,V] bf16, g:[B,H,T,K] fp32, beta,gamma:[B,H,T] fp32
    Returns [B,H,T,V] float32 (or q dtype) matching reference.
    e, gamma are accepted but unused as in the equation.
    """
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        scale = float(scale)

    # CPU fallback for testing without GPU
    if not q.is_cuda:
        return _reference(q, k, v, e, g, beta, gamma, scale)

    B, H, T, K = q.shape
    V = v.shape[-1]
    BH = B * H

    # make contiguous flattened BH,T,... views
    q_f = q.view(BH, T, K).contiguous()
    k_f = k.view(BH, T, K).contiguous()
    v_f = v.view(BH, T, V).contiguous()
    g_f = g.view(BH, T, K).contiguous().to(torch.float32)
    beta_f = beta.view(BH, T).contiguous().to(torch.float32)

    out_f = torch.empty((BH, T, V), device=q.device, dtype=torch.float32)
    state = torch.zeros((BH, K, V), device=q.device, dtype=torch.float32)

    BLOCK_K = 32
    BLOCK_V = 32
    CHUNK = 64
    assert K % BLOCK_K == 0, "K must be multiple of 32 for this kernel"
    # V may be not multiple, we handle via mask
    NUM_V_BLOCKS = (V + BLOCK_V - 1) // BLOCK_V
    assert T % CHUNK == 0, "T must be divisible by 64 (2048 ok); pad otherwise"

    grid = (BH * NUM_V_BLOCKS,)

    num_chunks = T // CHUNK
    for ci in range(num_chunks):
        t0 = ci * CHUNK
        _chunk_fwd_kernel[grid](
            q_f, k_f, v_f, g_f, beta_f, out_f, state,
            BH, T, K, V, t0,
            BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V, CHUNK=CHUNK, SCALE=scale, NUM_V_BLOCKS=NUM_V_BLOCKS,
            num_warps=4, num_stages=1,
        )

    out = out_f.view(B, H, T, V)
    # spec allows float32 or q dtype; we return float32 to match reference exactly
    return out
```
We keep `S[K,V]` in SRAM split into 4×32×32 tiles (16 KB per program) and partition `V` across programs to get parallelism over `B*H*V/BLOCK_V` CTAs. Time is chunked (64) and a Python host loop launches one Triton kernel per chunk carrying the recurrent state in a `[BH,K,V]` buffer, so the Triton program only unrolls 64 steps (not 2048) keeping compile size small while fusing all memory traffic. Per-step work is `pred=kᵀS`, `S+=βk(v-pred)ᵀ`, `S*=exp(g)`, `o=scale·qᵀS` with row-wise decay.

Unsure: exact Triton 3.5 syntax for `k[:,None]` broadcasting vs `tl.expand_dims`, whether `t0` as runtime scalar vs `tl.constexpr` triggers recompilation per chunk, and whether `tl.load` 2-D block pointers with `mask` as used will compile on sm_89; the semantics match the reference order (decay after write) but handling of non-divisible `T`/`V` via masking is only lightly tested.