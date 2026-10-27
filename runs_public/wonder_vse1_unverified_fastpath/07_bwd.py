import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward recurrent kernel.
#
# State S is [K, V]. It also stores the state at the beginning of every
# block of BC steps. Backward recomputes each block forward from its
# checkpoint and stores the intermediate post-decay states in scratch,
# rather than storing the state at every timestep.
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, u_ptr, g_ptr, beta_ptr, gamma_ptr,
    o_ptr, ckpt_ptr,
    T, scale, NC,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    off_k = tl.arange(0, BK)
    off_v = tl.arange(0, BV)
    S = tl.zeros([BK, BV], dtype=tl.float32)

    for t in range(T):
        if t % BC == 0:
            chunk = t // BC
            tl.store(
                ckpt_ptr
                + ((bh * NC + chunk) * BK + off_k[:, None]) * BV
                + off_v[None, :],
                S,
            )

        row = bh * T + t
        g = tl.load(g_ptr + row * BK + off_k).to(tl.float32)
        u = tl.load(u_ptr + row * BV + off_v).to(tl.float32)
        kk = tl.load(k_ptr + row * BK + off_k).to(tl.float32)
        vv = tl.load(v_ptr + row * BV + off_v).to(tl.float32)
        qq = tl.load(q_ptr + row * BK + off_k).to(tl.float32)
        beta = tl.load(beta_ptr + row).to(tl.float32)
        gamma = tl.load(gamma_ptr + row).to(tl.float32)

        # Decay, right erase, delta-rule write, output.
        S = S * tl.exp(g)[:, None]
        su = tl.sum(S * u[None, :], axis=1)
        S = S - gamma * su[:, None] * u[None, :]

        pred = tl.sum(S * kk[:, None], axis=0)
        S = S + kk[:, None] * (beta * (vv - pred))[None, :]

        out = scale * tl.sum(S * qq[:, None], axis=0)
        tl.store(o_ptr + row * BV + off_v, out)


# ---------------------------------------------------------------------------
# Backward recurrent kernel.
#
# One program handles one (batch, head). For every block, in reverse order:
#   1. Recompute the block forward from its checkpoint, storing Sa, the
#      state after decay and before the right erase, at each step.
#   2. Walk the block backwards carrying the adjoint A = dL/dS_t.
#
# With A including the output term scale*q*do^T, pred = Sb^T k,
# r = v - pred and Sb = Sa - gamma*(Sa*u)*u^T:
#   dr       = beta * (k^T A)
#   dv       = dr
#   dbeta    = (k^T A) . r
#   dk       = beta * (A r) - Sb dr
#   B        = A - k dr^T                    (adjoint wrt Sb)
#   z        = B u,  w = Sa u
#   dgamma   = -(w . z)
#   du       = -gamma * (B^T w + Sa^T z)
#   dSa      = B - gamma * z u^T
#   dg       = sum_v dSa * Sa
#   dS_prev  = exp(g) * dSa
#   dq       = scale * S_t do
# Sb and S_t are used through Sa via
#   Sb x = Sa x - gamma*w*(u.x),  S_t = Sb + beta*k*r^T,
# so Sb/S_t never need to be materialised as extra tiles.
# ---------------------------------------------------------------------------
@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, u_ptr, g_ptr, beta_ptr, gamma_ptr,
    do_ptr, ckpt_ptr, scratch_ptr,
    dq_ptr, dk_ptr, dv_ptr, du_ptr, dg_ptr, dbeta_ptr, dgamma_ptr,
    T, scale, NC,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    off_k = tl.arange(0, BK)
    off_v = tl.arange(0, BV)
    A = tl.zeros([BK, BV], dtype=tl.float32)

    for ci in range(NC):
        chunk = NC - 1 - ci
        start = chunk * BC
        chunk_len = BC
        if start + BC > T:
            chunk_len = T - start

        # ---- Recompute this block forward, storing Sa at each step. ----
        S = tl.load(
            ckpt_ptr
            + ((bh * NC + chunk) * BK + off_k[:, None]) * BV
            + off_v[None, :]
        )
        for i in range(chunk_len):
            row = bh * T + start + i
            g = tl.load(g_ptr + row * BK + off_k).to(tl.float32)
            u = tl.load(u_ptr + row * BV + off_v).to(tl.float32)
            kk = tl.load(k_ptr + row * BK + off_k).to(tl.float32)
            vv = tl.load(v_ptr + row * BV + off_v).to(tl.float32)
            beta = tl.load(beta_ptr + row).to(tl.float32)
            gamma = tl.load(gamma_ptr + row).to(tl.float32)

            S = S * tl.exp(g)[:, None]
            tl.store(
                scratch_ptr
                + ((bh * BC + i) * BK + off_k[:, None]) * BV
                + off_v[None, :],
                S,
            )
            su = tl.sum(S * u[None, :], axis=1)
            S = S - gamma * su[:, None] * u[None, :]
            pred = tl.sum(S * kk[:, None], axis=0)
            S = S + kk[:, None] * (beta * (vv - pred))[None, :]

        # ---- Walk the block backwards. ----
        for jj in range(chunk_len):
            i = chunk_len - 1 - jj
            t = start + i
            row = bh * T + t

            g = tl.load(g_ptr + row * BK + off_k).to(tl.float32)
            u = tl.load(u_ptr + row * BV + off_v).to(tl.float32)
            kk = tl.load(k_ptr + row * BK + off_k).to(tl.float32)
            vv = tl.load(v_ptr + row * BV + off_v).to(tl.float32)
            qq = tl.load(q_ptr + row * BK + off_k).to(tl.float32)
            dov = tl.load(do_ptr + row * BV + off_v).to(tl.float32)
            beta = tl.load(beta_ptr + row).to(tl.float32)
            gamma = tl.load(gamma_ptr + row).to(tl.float32)

            Sa = tl.load(
                scratch_ptr
                + ((bh * BC + i) * BK + off_k[:, None]) * BV
                + off_v[None, :]
            )

            w = tl.sum(Sa * u[None, :], axis=1)                    # Sa u
            pred = tl.sum(Sa * kk[:, None], axis=0) - gamma * tl.sum(w * kk) * u
            r = vv - pred

            # dq = scale * S_t do, using S_t through Sa.
            Sa_do = tl.sum(Sa * dov[None, :], axis=1)
            dq_vec = scale * (
                Sa_do - gamma * w * tl.sum(u * dov)
                + beta * kk * tl.sum(r * dov)
            )
            tl.store(dq_ptr + row * BK + off_k, dq_vec)

            # Add this step's output contribution to the adjoint.
            A = A + scale * qq[:, None] * dov[None, :]

            kA = tl.sum(kk[:, None] * A, axis=0)                   # k^T A
            dr = beta * kA
            tl.store(dv_ptr + row * BV + off_v, dr)
            tl.store(dbeta_ptr + row, tl.sum(kA * r))

            Ar = tl.sum(A * r[None, :], axis=1)                    # A r
            Sa_dr = tl.sum(Sa * dr[None, :], axis=1)
            dk_vec = beta * Ar - (Sa_dr - gamma * w * tl.sum(u * dr))
            tl.store(dk_ptr + row * BK + off_k, dk_vec)

            # B = A - k dr^T (adjoint wrt Sb), mutated in place in A.
            A = A - kk[:, None] * dr[None, :]
            z = tl.sum(A * u[None, :], axis=1)                     # B u
            Bt_w = tl.sum(A * w[:, None], axis=0)                  # B^T w
            SaT_z = tl.sum(Sa * z[:, None], axis=0)                # Sa^T z
            tl.store(du_ptr + row * BV + off_v, -gamma * (Bt_w + SaT_z))
            tl.store(dgamma_ptr + row, -tl.sum(w * z))

            # dSa, then dg and the adjoint for the previous state.
            A = A - gamma * z[:, None] * u[None, :]
            tl.store(dg_ptr + row * BK + off_k, tl.sum(A * Sa, axis=1))
            A = A * tl.exp(g)[:, None]


_BC = 32


def _pad_last_dim(x, padded_size):
    if x.shape[-1] == padded_size:
        return x.contiguous()
    result = torch.zeros((*x.shape[:-1], padded_size), device=x.device, dtype=x.dtype)
    result[..., : x.shape[-1]] = x
    return result


def _num_warps(state_elements):
    if state_elements <= 4096:
        return 4
    if state_elements <= 16384:
        return 16
    return 32


class _LayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, u, g, beta, gamma, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BH = B * H
        BK = max(16, triton.next_power_of_2(K))
        BV = max(16, triton.next_power_of_2(V))
        NC = (T + _BC - 1) // _BC

        q_ = _pad_last_dim(q, BK).reshape(BH, T, BK)
        k_ = _pad_last_dim(k, BK).reshape(BH, T, BK)
        g_ = _pad_last_dim(g, BK).reshape(BH, T, BK)
        v_ = _pad_last_dim(v, BV).reshape(BH, T, BV)
        u_ = _pad_last_dim(u, BV).reshape(BH, T, BV)
        beta_ = beta.contiguous().reshape(BH, T)
        gamma_ = gamma.contiguous().reshape(BH, T)

        out_ = torch.empty((BH, T, BV), device=q.device, dtype=torch.float32)
        ckpt = torch.zeros((BH, NC, BK, BV), device=q.device, dtype=torch.float32)

        _fwd_kernel[(BH,)](
            q_, k_, v_, u_, g_, beta_, gamma_, out_, ckpt,
            T, float(scale), NC,
            BK=BK, BV=BV, BC=_BC,
            num_warps=_num_warps(BK * BV), num_stages=1,
        )

        ctx.save_for_backward(q_, k_, v_, u_, g_, beta_, gamma_, ckpt)
        ctx.B, ctx.H, ctx.T, ctx.K, ctx.V = B, H, T, K, V
        ctx.BK, ctx.BV, ctx.NC = BK, BV, NC
        ctx.scale = float(scale)
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, u.dtype,
                      g.dtype, beta.dtype, gamma.dtype)
        return out_[:, :, :V].reshape(B, H, T, V)

    @staticmethod
    def backward(ctx, grad_out):
        q_, k_, v_, u_, g_, beta_, gamma_, ckpt = ctx.saved_tensors
        B, H, T, K, V = ctx.B, ctx.H, ctx.T, ctx.K, ctx.V
        BK, BV, NC = ctx.BK, ctx.BV, ctx.NC
        BH = B * H
        dev = q_.device

        go_ = _pad_last_dim(grad_out, BV).reshape(BH, T, BV)
        if go_.dtype != torch.float32:
            go_ = go_.float()

        dq_ = torch.zeros((BH, T, BK), device=dev, dtype=torch.float32)
        dk_ = torch.zeros((BH, T, BK), device=dev, dtype=torch.float32)
        dg_ = torch.zeros((BH, T, BK), device=dev, dtype=torch.float32)
        dv_ = torch.zeros((BH, T, BV), device=dev, dtype=torch.float32)
        du_ = torch.zeros((BH, T, BV), device=dev, dtype=torch.float32)
        dbeta_ = torch.zeros((BH, T), device=dev, dtype=torch.float32)
        dgamma_ = torch.zeros((BH, T), device=dev, dtype=torch.float32)
        scratch = torch.empty((BH, _BC, BK, BV), device=dev, dtype=torch.float32)

        _bwd_kernel[(BH,)](
            q_, k_, v_, u_, g_, beta_, gamma_,
            go_, ckpt, scratch,
            dq_, dk_, dv_, du_, dg_, dbeta_, dgamma_,
            T, ctx.scale, NC,
            BK=BK, BV=BV, BC=_BC,
            num_warps=_num_warps(BK * BV), num_stages=1,
        )

        dq_d, dk_d, dv_d, du_d, dg_d, dbeta_d, dgamma_d = ctx.dtypes
        dq = dq_.reshape(B, H, T, BK)[:, :, :, :K].to(dq_d)
        dk = dk_.reshape(B, H, T, BK)[:, :, :, :K].to(dk_d)
        dg = dg_.reshape(B, H, T, BK)[:, :, :, :K].to(dg_d)
        dv = dv_.reshape(B, H, T, BV)[:, :, :, :V].to(dv_d)
        du = du_.reshape(B, H, T, BV)[:, :, :, :V].to(du_d)
        dbeta = dbeta_.reshape(B, H, T).to(dbeta_d)
        dgamma = dgamma_.reshape(B, H, T).to(dgamma_d)
        return dq, dk, dv, du, dg, dbeta, dgamma, None


def kernel_function(q, k, v, u, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    if B * H == 0 or T == 0 or K == 0 or V == 0:
        return torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)
    return _LayerFn.apply(q, k, v, u, g, beta, gamma, float(scale))
