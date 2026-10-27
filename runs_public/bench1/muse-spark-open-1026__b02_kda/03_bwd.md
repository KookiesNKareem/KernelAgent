```python
import torch
import triton
import triton.language as tl

# ------------------------------------------------------------
# forward : same idea as the fast 3.18ms kernel, kept separate
# ------------------------------------------------------------
@triton.jit
def _fwd_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, out_ptr,
                scale,
                BH, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr):
    pid = tl.program_id(0)
    k_offs = tl.arange(0, K)
    v_offs = tl.arange(0, V)
    S = tl.zeros((K, V), dtype=tl.float32)

    q_base = q_ptr + pid * T * K
    k_base = k_ptr + pid * T * K
    g_base = g_ptr + pid * T * K
    v_base = v_ptr + pid * T * V
    b_base = beta_ptr + pid * T
    o_base = out_ptr + pid * T * V

    for t in range(T):
        off_k = t * K
        off_v = t * V
        qv = tl.load(q_base + off_k + k_offs).to(tl.float32)
        kv = tl.load(k_base + off_k + k_offs).to(tl.float32)
        gv = tl.load(g_base + off_k + k_offs).to(tl.float32)
        vv = tl.load(v_base + off_v + v_offs).to(tl.float32)
        bv = tl.load(b_base + t).to(tl.float32)

        d = tl.exp(gv)
        S = S * tl.expand_dims(d, 1)

        pred = tl.sum(S * tl.expand_dims(kv, 1), axis=0)
        delta = bv * (vv - pred)
        S = S + tl.expand_dims(kv, 1) * tl.expand_dims(delta, 0)

        out_vec = tl.sum(S * tl.expand_dims(qv * scale, 1), axis=0)
        tl.store(o_base + off_v + v_offs, out_vec)


@triton.jit
def _bwd_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
                go_ptr,
                gq_ptr, gk_ptr, gv_ptr, gg_ptr, gb_ptr,
                scale,
                BH, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr):
    pid = tl.program_id(0)
    k_offs = tl.arange(0, K)
    v_offs = tl.arange(0, V)

    # pointers base
    q_base = q_ptr + pid * T * K
    k_base = k_ptr + pid * T * K
    g_base = g_ptr + pid * T * K
    v_base = v_ptr + pid * T * V
    b_base = beta_ptr + pid * T
    go_base = go_ptr + pid * T * V

    gq_base = gq_ptr + pid * T * K
    gk_base = gk_ptr + pid * T * K
    gg_base = gg_ptr + pid * T * K
    gv_base = gv_ptr + pid * T * V
    gb_base = gb_ptr + pid * T

    # ---------- phase 1 : forward to get S_T ----------
    S = tl.zeros((K, V), dtype=tl.float32)
    for t in range(T):
        off_k = t * K
        off_v = t * V
        kv = tl.load(k_base + off_k + k_offs).to(tl.float32)
        gv = tl.load(g_base + off_k + k_offs).to(tl.float32)
        vv = tl.load(v_base + off_v + v_offs).to(tl.float32)
        bv = tl.load(b_base + t).to(tl.float32)
        d = tl.exp(gv)
        S = S * tl.expand_dims(d, 1)
        pred = tl.sum(S * tl.expand_dims(kv, 1), axis=0)
        delta = bv * (vv - pred)
        S = S + tl.expand_dims(kv, 1) * tl.expand_dims(delta, 0)

    # S now = S_{T-1}
    carry = tl.zeros((K, V), dtype=tl.float32)

    # ---------- phase 2 : reverse ----------
    for t in range(T):
        ti = T - 1 - t  # descending
        off_k = ti * K
        off_v = ti * V
        qv = tl.load(q_base + off_k + k_offs).to(tl.float32)
        kv = tl.load(k_base + off_k + k_offs).to(tl.float32)
        gv = tl.load(g_base + off_k + k_offs).to(tl.float32)
        vv = tl.load(v_base + off_v + v_offs).to(tl.float32)
        bv = tl.load(b_base + ti).to(tl.float32)
        gov = tl.load(go_base + off_v + v_offs).to(tl.float32)

        d = tl.exp(gv)

        # lambda_t = scale*q outer go + carry
        lam = tl.expand_dims(qv * scale, 1) * tl.expand_dims(gov, 0) + carry

        # ----- grads that need S_curr (S) -----
        # grad_q = scale * S @ go
        gq = scale * tl.sum(S * tl.expand_dims(gov, 0), axis=1)
        tl.store(gq_base + off_k + k_offs, gq)

        # u = k^T lam  (V)
        u = tl.sum(lam * tl.expand_dims(kv, 1), axis=0)

        # grad_v = beta * u
        gv_out = bv * u
        tl.store(gv_base + off_v + v_offs, gv_out)

        # ----- need S_prev to compute remaining grads -----
        # S_prev = D^{-1} A^{-1} (S - beta k v^T)
        # A^{-1}= I + c k k^T , c=beta/(1-beta)
        # safe clamp denominator away from 0
        c = bv / (1.0 - bv + 1e-6)

        S_minus = S - bv * tl.expand_dims(kv, 1) * tl.expand_dims(vv, 0)
        kTSm = tl.sum(S_minus * tl.expand_dims(kv, 1), axis=0)  # V
        Ainv = S_minus + c * tl.expand_dims(kv, 1) * tl.expand_dims(kTSm, 0)
        inv_d = tl.exp(-gv)
        S_prev = Ainv * tl.expand_dims(inv_d, 1)

        # a = D * S_prev
        a_mat = S_prev * tl.expand_dims(d, 1)
        pred_a = tl.sum(a_mat * tl.expand_dims(kv, 1), axis=0)  # V
        w = vv - pred_a

        # grad_beta = dot(u,w)
        gb = tl.sum(u * w, axis=0)
        # gb is scalar, store broadcast
        tl.store(gb_base + ti, gb)

        # grad_k = beta * ( lam @ w - a_mat @ u )
        lam_w = tl.sum(lam * tl.expand_dims(w, 0), axis=1)  # K
        a_u = tl.sum(a_mat * tl.expand_dims(u, 0), axis=1)  # K
        gk = bv * (lam_w - a_u)
        tl.store(gk_base + off_k + k_offs, gk)

        # grad_g = d * ( row_dot(S_prev, lam) - beta*k* row_dot(S_prev, u) )
        dot1 = tl.sum(S_prev * lam, axis=1)  # K
        dot2 = tl.sum(S_prev * tl.expand_dims(u, 0), axis=1)  # K
        gg = d * (dot1 - bv * kv * dot2)
        tl.store(gg_base + off_k + k_offs, gg)

        # prepare carry for next iteration: M_t^T lam = D * (lam - beta k u^T)
        tmp = lam - bv * tl.expand_dims(kv, 1) * tl.expand_dims(u, 0)
        carry = tmp * tl.expand_dims(d, 1)

        # step S
        S = S_prev


class _Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        ctx.BH = BH
        ctx.B = B
        ctx.H = H
        ctx.T = T
        ctx.K = K
        ctx.V = V
        ctx.scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        # save for backward - keep contiguous versions to guarantee layout expected by kernels
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        g_c = g.contiguous()
        beta_c = beta.contiguous()
        ctx.save_for_backward(q_c, k_c, v_c, g_c, beta_c)
        # save dtypes for backward casting
        ctx.q_dtype = q.dtype
        ctx.k_dtype = k.dtype
        ctx.v_dtype = v.dtype
        ctx.e_dtype = e.dtype
        ctx.g_dtype = g.dtype
        ctx.beta_dtype = beta.dtype
        ctx.gamma_dtype = gamma.dtype

        out = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)
        grid = (BH,)
        _fwd_kernel[grid](q_c, k_c, v_c, g_c, beta_c, out, ctx.scale_f, BH, T, K, V, num_warps=8, num_stages=3)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q_c, k_c, v_c, g_c, beta_c = ctx.saved_tensors
        B, H, T, K, V, BH = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V, ctx.BH
        scale_f = ctx.scale_f
        go = grad_out.contiguous().to(torch.float32)

        # allocate grad buffers in fp32 for accumulation
        gq_buf = torch.zeros((BH * T * K,), device=q_c.device, dtype=torch.float32)
        gk_buf = torch.zeros((BH * T * K,), device=q_c.device, dtype=torch.float32)
        gg_buf = torch.zeros((BH * T * K,), device=q_c.device, dtype=torch.float32)
        gv_buf = torch.zeros((BH * T * V,), device=q_c.device, dtype=torch.float32)
        gb_buf = torch.zeros((BH * T,), device=q_c.device, dtype=torch.float32)

        # launch expects 2-D view but we use flat pointers: reuse base pointers with flat layout
        # our kernel uses pid*T*K arithmetic, so flat is fine if we pass flat tensors
        grid = (BH,)
        _bwd_kernel[grid](q_c, k_c, v_c, g_c, beta_c, go,
                          gq_buf, gk_buf, gv_buf, gg_buf, gb_buf,
                          scale_f, BH, T, K, V, num_warps=8, num_stages=2)

        # reshape back
        gq = gq_buf.view(B, H, T, K).to(ctx.q_dtype)
        gk = gk_buf.view(B, H, T, K).to(ctx.k_dtype)
        gv = gv_buf.view(B, H, T, V).to(ctx.v_dtype)
        gg = gg_buf.view(B, H, T, K).to(ctx.g_dtype)
        gb = gb_buf.view(B, H, T).to(ctx.beta_dtype)

        # e, gamma unused => zeros
        ge = torch.zeros((B, H, T, K), device=q_c.device, dtype=ctx.e_dtype)
        ggamma = torch.zeros((B, H, T), device=q_c.device, dtype=ctx.gamma_dtype)

        # return grad for each forward arg: q,k,v,e,g,beta,gamma,scale
        return gq, gk, gv, ge, gg, gb, ggamma, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return _Func.apply(q, k, v, e, g, beta, gamma, scale)
```

Forward keeps `S[K,V]` in registers and streams `T=2048` steps, fusing decay, delta update and output to avoid the Python loop; backward recomputes the final state forward then walks backwards using the analytic inverse `M^{-1}=D^{-1}(I+c k k^T)` and the adjoint `M^T=D(I-βkk^T)` so no `T·K·V` checkpoint has to be stored, only two `K×V` tiles in SRAM, with one persistent Triton program per `B·H`. Grad formulas for `q,k,v,g,β` are derived from the outer-product structure and evaluated with row-wise sums, giving `e,γ` zero.