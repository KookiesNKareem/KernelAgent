```python
import torch
import triton
import triton.language as tl

BLOCK = 32
BLOCK_C = 32

@triton.jit
def _fwd_kernel(q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, out_ptr, state_ptr,
                scale,
                T: tl.constexpr, Kb: tl.constexpr, Vb: tl.constexpr, START: tl.constexpr):
    pid = tl.program_id(0)
    # V tiles are static unrolled: Kb,Vb are constexpr multiples of 32
    for v_off in range(0, Vb, 32):
        # load state slices for this v_off into registers (up to 256 K)
        s0 = tl.load(state_ptr + pid * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        s1 = tl.zeros((32, 32), dtype=tl.float32)
        s2 = tl.zeros((32, 32), dtype=tl.float32)
        s3 = tl.zeros((32, 32), dtype=tl.float32)
        s4 = tl.zeros((32, 32), dtype=tl.float32)
        s5 = tl.zeros((32, 32), dtype=tl.float32)
        s6 = tl.zeros((32, 32), dtype=tl.float32)
        s7 = tl.zeros((32, 32), dtype=tl.float32)
        if Kb > 32:
            s1 = tl.load(state_ptr + pid * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 64:
            s2 = tl.load(state_ptr + pid * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 96:
            s3 = tl.load(state_ptr + pid * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 128:
            s4 = tl.load(state_ptr + pid * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 160:
            s5 = tl.load(state_ptr + pid * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 192:
            s6 = tl.load(state_ptr + pid * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 224:
            s7 = tl.load(state_ptr + pid * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        for i in range(32):
            cur = START + i
            valid = cur < T
            d_val = tl.load(d_ptr + pid * T + cur, mask=valid, other=1.0)
            b_val = tl.load(beta_ptr + pid * T + cur, mask=valid, other=0.0)
            v_vec = tl.load(v_ptr + pid * T * Vb + cur * Vb + v_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            k0 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 0 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            k1 = tl.zeros((32,), dtype=tl.float32)
            k2 = tl.zeros((32,), dtype=tl.float32)
            k3 = tl.zeros((32,), dtype=tl.float32)
            k4 = tl.zeros((32,), dtype=tl.float32)
            k5 = tl.zeros((32,), dtype=tl.float32)
            k6 = tl.zeros((32,), dtype=tl.float32)
            k7 = tl.zeros((32,), dtype=tl.float32)
            q0 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 0 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            q1 = tl.zeros((32,), dtype=tl.float32)
            q2 = tl.zeros((32,), dtype=tl.float32)
            q3 = tl.zeros((32,), dtype=tl.float32)
            q4 = tl.zeros((32,), dtype=tl.float32)
            q5 = tl.zeros((32,), dtype=tl.float32)
            q6 = tl.zeros((32,), dtype=tl.float32)
            q7 = tl.zeros((32,), dtype=tl.float32)
            if Kb > 32:
                k1 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 32 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q1 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 32 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 64:
                k2 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 64 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q2 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 64 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 96:
                k3 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 96 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q3 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 96 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 128:
                k4 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 128 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q4 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 128 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 160:
                k5 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 160 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q5 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 160 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 192:
                k6 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 192 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q6 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 192 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 224:
                k7 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 224 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q7 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 224 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            # p = k^T S
            p = tl.sum(k0[:, None] * s0, axis=0)
            if Kb > 32:
                p = p + tl.sum(k1[:, None] * s1, axis=0)
            if Kb > 64:
                p = p + tl.sum(k2[:, None] * s2, axis=0)
            if Kb > 96:
                p = p + tl.sum(k3[:, None] * s3, axis=0)
            if Kb > 128:
                p = p + tl.sum(k4[:, None] * s4, axis=0)
            if Kb > 160:
                p = p + tl.sum(k5[:, None] * s5, axis=0)
            if Kb > 192:
                p = p + tl.sum(k6[:, None] * s6, axis=0)
            if Kb > 224:
                p = p + tl.sum(k7[:, None] * s7, axis=0)
            coeff = b_val * (v_vec - d_val * p)
            # S = d*S + k*coeff
            s0 = s0 * d_val + k0[:, None] * coeff[None, :]
            if Kb > 32:
                s1 = s1 * d_val + k1[:, None] * coeff[None, :]
            if Kb > 64:
                s2 = s2 * d_val + k2[:, None] * coeff[None, :]
            if Kb > 96:
                s3 = s3 * d_val + k3[:, None] * coeff[None, :]
            if Kb > 128:
                s4 = s4 * d_val + k4[:, None] * coeff[None, :]
            if Kb > 160:
                s5 = s5 * d_val + k5[:, None] * coeff[None, :]
            if Kb > 192:
                s6 = s6 * d_val + k6[:, None] * coeff[None, :]
            if Kb > 224:
                s7 = s7 * d_val + k7[:, None] * coeff[None, :]
            o = tl.sum(q0[:, None] * s0, axis=0)
            if Kb > 32:
                o = o + tl.sum(q1[:, None] * s1, axis=0)
            if Kb > 64:
                o = o + tl.sum(q2[:, None] * s2, axis=0)
            if Kb > 96:
                o = o + tl.sum(q3[:, None] * s3, axis=0)
            if Kb > 128:
                o = o + tl.sum(q4[:, None] * s4, axis=0)
            if Kb > 160:
                o = o + tl.sum(q5[:, None] * s5, axis=0)
            if Kb > 192:
                o = o + tl.sum(q6[:, None] * s6, axis=0)
            if Kb > 224:
                o = o + tl.sum(q7[:, None] * s7, axis=0)
            o = o * scale
            tl.store(out_ptr + pid * T * Vb + cur * Vb + v_off + tl.arange(0, 32), o, mask=valid)
        tl.store(state_ptr + pid * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s0)
        if Kb > 32:
            tl.store(state_ptr + pid * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s1)
        if Kb > 64:
            tl.store(state_ptr + pid * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s2)
        if Kb > 96:
            tl.store(state_ptr + pid * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s3)
        if Kb > 128:
            tl.store(state_ptr + pid * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s4)
        if Kb > 160:
            tl.store(state_ptr + pid * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s5)
        if Kb > 192:
            tl.store(state_ptr + pid * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s6)
        if Kb > 224:
            tl.store(state_ptr + pid * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s7)

@triton.jit
def _recompute_kernel(k_ptr, v_ptr, d_ptr, beta_ptr, cur_state_ptr, tmp_ptr,
                      T: tl.constexpr, Kb: tl.constexpr, Vb: tl.constexpr, START: tl.constexpr):
    pid = tl.program_id(0)
    for v_off in range(0, Vb, 32):
        s0 = tl.load(cur_state_ptr + pid * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        s1 = tl.zeros((32, 32), dtype=tl.float32)
        s2 = tl.zeros((32, 32), dtype=tl.float32)
        s3 = tl.zeros((32, 32), dtype=tl.float32)
        s4 = tl.zeros((32, 32), dtype=tl.float32)
        s5 = tl.zeros((32, 32), dtype=tl.float32)
        s6 = tl.zeros((32, 32), dtype=tl.float32)
        s7 = tl.zeros((32, 32), dtype=tl.float32)
        if Kb > 32:
            s1 = tl.load(cur_state_ptr + pid * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 64:
            s2 = tl.load(cur_state_ptr + pid * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 96:
            s3 = tl.load(cur_state_ptr + pid * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 128:
            s4 = tl.load(cur_state_ptr + pid * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 160:
            s5 = tl.load(cur_state_ptr + pid * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 192:
            s6 = tl.load(cur_state_ptr + pid * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 224:
            s7 = tl.load(cur_state_ptr + pid * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        for i in range(32):
            cur = START + i
            valid = cur < T
            d_val = tl.load(d_ptr + pid * T + cur, mask=valid, other=1.0)
            b_val = tl.load(beta_ptr + pid * T + cur, mask=valid, other=0.0)
            v_vec = tl.load(v_ptr + pid * T * Vb + cur * Vb + v_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            k0 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 0 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            k1 = tl.zeros((32,), dtype=tl.float32)
            k2 = tl.zeros((32,), dtype=tl.float32)
            k3 = tl.zeros((32,), dtype=tl.float32)
            k4 = tl.zeros((32,), dtype=tl.float32)
            k5 = tl.zeros((32,), dtype=tl.float32)
            k6 = tl.zeros((32,), dtype=tl.float32)
            k7 = tl.zeros((32,), dtype=tl.float32)
            if Kb > 32:
                k1 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 32 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 64:
                k2 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 64 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 96:
                k3 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 96 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 128:
                k4 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 128 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 160:
                k5 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 160 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 192:
                k6 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 192 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 224:
                k7 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 224 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            p = tl.sum(k0[:, None] * s0, axis=0)
            if Kb > 32:
                p = p + tl.sum(k1[:, None] * s1, axis=0)
            if Kb > 64:
                p = p + tl.sum(k2[:, None] * s2, axis=0)
            if Kb > 96:
                p = p + tl.sum(k3[:, None] * s3, axis=0)
            if Kb > 128:
                p = p + tl.sum(k4[:, None] * s4, axis=0)
            if Kb > 160:
                p = p + tl.sum(k5[:, None] * s5, axis=0)
            if Kb > 192:
                p = p + tl.sum(k6[:, None] * s6, axis=0)
            if Kb > 224:
                p = p + tl.sum(k7[:, None] * s7, axis=0)
            coeff = b_val * (v_vec - d_val * p)
            s0 = s0 * d_val + k0[:, None] * coeff[None, :]
            if Kb > 32:
                s1 = s1 * d_val + k1[:, None] * coeff[None, :]
            if Kb > 64:
                s2 = s2 * d_val + k2[:, None] * coeff[None, :]
            if Kb > 96:
                s3 = s3 * d_val + k3[:, None] * coeff[None, :]
            if Kb > 128:
                s4 = s4 * d_val + k4[:, None] * coeff[None, :]
            if Kb > 160:
                s5 = s5 * d_val + k5[:, None] * coeff[None, :]
            if Kb > 192:
                s6 = s6 * d_val + k6[:, None] * coeff[None, :]
            if Kb > 224:
                s7 = s7 * d_val + k7[:, None] * coeff[None, :]
            base = tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + v_off + tl.arange(0, 32)[None, :] + tl.arange(0, 32)[:, None] * Vb
            tl.store(base + 0 * Vb, s0, mask=valid)
            if Kb > 32:
                tl.store(base + 32 * Vb, s1, mask=valid)
            if Kb > 64:
                tl.store(base + 64 * Vb, s2, mask=valid)
            if Kb > 96:
                tl.store(base + 96 * Vb, s3, mask=valid)
            if Kb > 128:
                tl.store(base + 128 * Vb, s4, mask=valid)
            if Kb > 160:
                tl.store(base + 160 * Vb, s5, mask=valid)
            if Kb > 192:
                tl.store(base + 192 * Vb, s6, mask=valid)
            if Kb > 224:
                tl.store(base + 224 * Vb, s7, mask=valid)
        # write back cur_state for next chunk
        tl.store(cur_state_ptr + pid * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s0)
        if Kb > 32:
            tl.store(cur_state_ptr + pid * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s1)
        if Kb > 64:
            tl.store(cur_state_ptr + pid * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s2)
        if Kb > 96:
            tl.store(cur_state_ptr + pid * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s3)
        if Kb > 128:
            tl.store(cur_state_ptr + pid * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s4)
        if Kb > 160:
            tl.store(cur_state_ptr + pid * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s5)
        if Kb > 192:
            tl.store(cur_state_ptr + pid * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s6)
        if Kb > 224:
            tl.store(cur_state_ptr + pid * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s7)

@triton.jit
def _bwd_kernel(q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, go_ptr, tmp_ptr, chunk_start_ptr, g_state_ptr,
                dq_ptr, dk_ptr, dv_ptr, db_ptr, dg_ptr,
                scale,
                T: tl.constexpr, Kb: tl.constexpr, Vb: tl.constexpr, START: tl.constexpr):
    pid = tl.program_id(0)
    for v_off in range(0, Vb, 32):
        g0 = tl.load(g_state_ptr + pid * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        g1 = tl.zeros((32, 32), dtype=tl.float32)
        g2 = tl.zeros((32, 32), dtype=tl.float32)
        g3 = tl.zeros((32, 32), dtype=tl.float32)
        g4 = tl.zeros((32, 32), dtype=tl.float32)
        g5 = tl.zeros((32, 32), dtype=tl.float32)
        g6 = tl.zeros((32, 32), dtype=tl.float32)
        g7 = tl.zeros((32, 32), dtype=tl.float32)
        if Kb > 32:
            g1 = tl.load(g_state_ptr + pid * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 64:
            g2 = tl.load(g_state_ptr + pid * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 96:
            g3 = tl.load(g_state_ptr + pid * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 128:
            g4 = tl.load(g_state_ptr + pid * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 160:
            g5 = tl.load(g_state_ptr + pid * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 192:
            g6 = tl.load(g_state_ptr + pid * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        if Kb > 224:
            g7 = tl.load(g_state_ptr + pid * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
        for idx in range(32):
            i = 31 - idx
            cur = START + i
            valid = cur < T
            d_val = tl.load(d_ptr + pid * T + cur, mask=valid, other=1.0)
            b_val = tl.load(beta_ptr + pid * T + cur, mask=valid, other=0.0)
            k0 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 0 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            k1 = tl.zeros((32,), dtype=tl.float32)
            k2 = tl.zeros((32,), dtype=tl.float32)
            k3 = tl.zeros((32,), dtype=tl.float32)
            k4 = tl.zeros((32,), dtype=tl.float32)
            k5 = tl.zeros((32,), dtype=tl.float32)
            k6 = tl.zeros((32,), dtype=tl.float32)
            k7 = tl.zeros((32,), dtype=tl.float32)
            q0 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 0 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            q1 = tl.zeros((32,), dtype=tl.float32)
            q2 = tl.zeros((32,), dtype=tl.float32)
            q3 = tl.zeros((32,), dtype=tl.float32)
            q4 = tl.zeros((32,), dtype=tl.float32)
            q5 = tl.zeros((32,), dtype=tl.float32)
            q6 = tl.zeros((32,), dtype=tl.float32)
            q7 = tl.zeros((32,), dtype=tl.float32)
            if Kb > 32:
                k1 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 32 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q1 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 32 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 64:
                k2 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 64 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q2 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 64 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 96:
                k3 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 96 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q3 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 96 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 128:
                k4 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 128 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q4 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 128 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 160:
                k5 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 160 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q5 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 160 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 192:
                k6 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 192 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q6 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 192 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            if Kb > 224:
                k7 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 224 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
                q7 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 224 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            v_vec = tl.load(v_ptr + pid * T * Vb + cur * Vb + v_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            go_vec = tl.load(go_ptr + pid * T * Vb + cur * Vb + v_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            # S_prev / S_cur for this v_off
            # S_prev needed for p, trace ; load lazily inside K branches
            # compute p = k^T S_prev
            p = tl.zeros((32,), dtype=tl.float32)
            # we need S_prev tiles per K block
            # load them on demand
            sp0 = tl.load(chunk_start_ptr + pid * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            # if i>0, override with tmp
            if i != 0:
                sp0 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            p = p + tl.sum(k0[:, None] * sp0, axis=0)
            if Kb > 32:
                sp1 = tl.load(chunk_start_ptr + pid * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp1 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                p = p + tl.sum(k1[:, None] * sp1, axis=0)
            if Kb > 64:
                sp2 = tl.load(chunk_start_ptr + pid * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp2 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                p = p + tl.sum(k2[:, None] * sp2, axis=0)
            if Kb > 96:
                sp3 = tl.load(chunk_start_ptr + pid * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp3 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                p = p + tl.sum(k3[:, None] * sp3, axis=0)
            if Kb > 128:
                sp4 = tl.load(chunk_start_ptr + pid * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp4 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                p = p + tl.sum(k4[:, None] * sp4, axis=0)
            if Kb > 160:
                sp5 = tl.load(chunk_start_ptr + pid * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp5 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                p = p + tl.sum(k5[:, None] * sp5, axis=0)
            if Kb > 192:
                sp6 = tl.load(chunk_start_ptr + pid * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp6 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                p = p + tl.sum(k6[:, None] * sp6, axis=0)
            if Kb > 224:
                sp7 = tl.load(chunk_start_ptr + pid * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp7 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                p = p + tl.sum(k7[:, None] * sp7, axis=0)
            # r = k^T G_t
            r = tl.zeros((32,), dtype=tl.float32)
            gt0 = g0 + scale * q0[:, None] * go_vec[None, :]
            r = r + tl.sum(k0[:, None] * gt0, axis=0)
            if Kb > 32:
                gt1 = g1 + scale * q1[:, None] * go_vec[None, :]
                r = r + tl.sum(k1[:, None] * gt1, axis=0)
            if Kb > 64:
                gt2 = g2 + scale * q2[:, None] * go_vec[None, :]
                r = r + tl.sum(k2[:, None] * gt2, axis=0)
            if Kb > 96:
                gt3 = g3 + scale * q3[:, None] * go_vec[None, :]
                r = r + tl.sum(k3[:, None] * gt3, axis=0)
            if Kb > 128:
                gt4 = g4 + scale * q4[:, None] * go_vec[None, :]
                r = r + tl.sum(k4[:, None] * gt4, axis=0)
            if Kb > 160:
                gt5 = g5 + scale * q5[:, None] * go_vec[None, :]
                r = r + tl.sum(k5[:, None] * gt5, axis=0)
            if Kb > 192:
                gt6 = g6 + scale * q6[:, None] * go_vec[None, :]
                r = r + tl.sum(k6[:, None] * gt6, axis=0)
            if Kb > 224:
                gt7 = g7 + scale * q7[:, None] * go_vec[None, :]
                r = r + tl.sum(k7[:, None] * gt7, axis=0)
            # dv is fully determined
            tl.store(dv_ptr + pid * T * Vb + cur * Vb + v_off + tl.arange(0, 32), b_val * r, mask=valid)
            # accumulate db/dg partials in HBM (sum over V tiles later)
            db_val = tl.sum(r * (v_vec - d_val * p))
            tl.store(db_ptr + pid * T * Vb + cur * Vb + v_off + tl.arange(0, 32), db_val * tl.full((32,), 1.0, dtype=tl.float32), mask=valid)  # dummy to keep same shape, will reduce
            # per-K contributions: dk,dq, trace, and update g
            # we do K blocks one by one, updating g registers and adding to HBM partials
            # S_cur for dq
            sc0 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            # dq part
            dq0 = scale * tl.sum(sc0 * go_vec[None, :], axis=1)
            trace0 = tl.sum(gt0 * sp0)
            gv0 = tl.sum(gt0 * v_vec[None, :], axis=1)
            gp0 = tl.sum(gt0 * p[None, :], axis=1)
            sr0 = tl.sum(sp0 * r[None, :], axis=1)
            dk0 = b_val * gv0 - d_val * b_val * gp0 - d_val * b_val * sr0
            tl.store(dq_ptr + pid * T * Kb + cur * Kb + 0 + tl.arange(0, 32), dq0, mask=valid)
            tl.store(dk_ptr + pid * T * Kb + cur * Kb + 0 + tl.arange(0, 32), dk0, mask=valid)
            g0 = d_val * (gt0 - b_val * k0[:, None] * r[None, :])
            dg0 = d_val * (trace0 - b_val * tl.sum(r * p))
            # we store dg as partial per V tile, will sum
            tl.store(dg_ptr + pid * T * Vb + cur * Vb + v_off + tl.arange(0, 32), dg0 * tl.full((32,), 1.0, dtype=tl.float32), mask=valid)
            if Kb > 32:
                sc1 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                gt1 = g1 + scale * q1[:, None] * go_vec[None, :]
                dq1 = scale * tl.sum(sc1 * go_vec[None, :], axis=1)
                trace1 = tl.sum(gt1 * sp1)
                gv1 = tl.sum(gt1 * v_vec[None, :], axis=1)
                gp1 = tl.sum(gt1 * p[None, :], axis=1)
                sr1 = tl.sum(sp1 * r[None, :], axis=1)
                dk1 = b_val * gv1 - d_val * b_val * gp1 - d_val * b_val * sr1
                tl.store(dq_ptr + pid * T * Kb + cur * Kb + 32 + tl.arange(0, 32), dq1, mask=valid)
                tl.store(dk_ptr + pid * T * Kb + cur * Kb + 32 + tl.arange(0, 32), dk1, mask=valid)
                g1 = d_val * (gt1 - b_val * k1[:, None] * r[None, :])
            if Kb > 64:
                sc2 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                gt2 = g2 + scale * q2[:, None] * go_vec[None, :]
                dq2 = scale * tl.sum(sc2 * go_vec[None, :], axis=1)
                gv2 = tl.sum(gt2 * v_vec[None, :], axis=1)
                gp2 = tl.sum(gt2 * p[None, :], axis=1)
                sr2 = tl.sum(sp2 * r[None, :], axis=1)
                dk2 = b_val * gv2 - d_val * b_val * gp2 - d_val * b_val * sr2
                tl.store(dq_ptr + pid * T * Kb + cur * Kb + 64 + tl.arange(0, 32), dq2, mask=valid)
                tl.store(dk_ptr + pid * T * Kb + cur * Kb + 64 + tl.arange(0, 32), dk2, mask=valid)
                g2 = d_val * (gt2 - b_val * k2[:, None] * r[None, :])
            if Kb > 96:
                sc3 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                gt3 = g3 + scale * q3[:, None] * go_vec[None, :]
                dq3 = scale * tl.sum(sc3 * go_vec[None, :], axis=1)
                gv3 = tl.sum(gt3 * v_vec[None, :], axis=1)
                gp3 = tl.sum(gt3 * p[None, :], axis=1)
                sr3 = tl.sum(sp3 * r[None, :], axis=1)
                dk3 = b_val * gv3 - d_val * b_val * gp3 - d_val * b_val * sr3
                tl.store(dq_ptr + pid * T * Kb + cur * Kb + 96 + tl.arange(0, 32), dq3, mask=valid)
                tl.store(dk_ptr + pid * T * Kb + cur * Kb + 96 + tl.arange(0, 32), dk3, mask=valid)
                g3 = d_val * (gt3 - b_val * k3[:, None] * r[None, :])
            if Kb > 128:
                sc4 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                gt4 = g4 + scale * q4[:, None] * go_vec[None, :]
                dq4 = scale * tl.sum(sc4 * go_vec[None, :], axis=1)
                gv4 = tl.sum(gt4 * v_vec[None, :], axis=1)
                gp4 = tl.sum(gt4 * p[None, :], axis=1)
                sr4 = tl.sum(sp4 * r[None, :], axis=1)
                dk4 = b_val * gv4 - d_val * b_val * gp4 - d_val * b_val * sr4
                tl.store(dq_ptr + pid * T * Kb + cur * Kb + 128 + tl.arange(0, 32), dq4, mask=valid)
                tl.store(dk_ptr + pid * T * Kb + cur * Kb + 128 + tl.arange(0, 32), dk4, mask=valid)
                g4 = d_val * (gt4 - b_val * k4[:, None] * r[None, :])
            if Kb > 160:
                sc5 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                gt5 = g5 + scale * q5[:, None] * go_vec[None, :]
                dq5 = scale * tl.sum(sc5 * go_vec[None, :], axis=1)
                gv5 = tl.sum(gt5 * v_vec[None, :], axis=1)
                gp5 = tl.sum(gt5 * p[None, :], axis=1)
                sr5 = tl.sum(sp5 * r[None, :], axis=1)
                dk5 = b_val * gv5 - d_val * b_val * gp5 - d_val * b_val * sr5
                tl.store(dq_ptr + pid * T * Kb + cur * Kb + 160 + tl.arange(0, 32), dq5, mask=valid)
                tl.store(dk_ptr + pid * T * Kb + cur * Kb + 160 + tl.arange(0, 32), dk5, mask=valid)
                g5 = d_val * (gt5 - b_val * k5[:, None] * r[None, :])
            if Kb > 192:
                sc6 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                gt6 = g6 + scale * q6[:, None] * go_vec[None, :]
                dq6 = scale * tl.sum(sc6 * go_vec[None, :], axis=1)
                gv6 = tl.sum(gt6 * v_vec[None, :], axis=1)
                gp6 = tl.sum(gt6 * p[None, :], axis=1)
                sr6 = tl.sum(sp6 * r[None, :], axis=1)
                dk6 = b_val * gv6 - d_val * b_val * gp6 - d_val * b_val * sr6
                tl.store(dq_ptr + pid * T * Kb + cur * Kb + 192 + tl.arange(0, 32), dq6, mask=valid)
                tl.store(dk_ptr + pid * T * Kb + cur * Kb + 192 + tl.arange(0, 32), dk6, mask=valid)
                g6 = d_val * (gt6 - b_val * k6[:, None] * r[None, :])
            if Kb > 224:
                sc7 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                gt7 = g7 + scale * q7[:, None] * go_vec[None, :]
                dq7 = scale * tl.sum(sc7 * go_vec[None, :], axis=1)
                gv7 = tl.sum(gt7 * v_vec[None, :], axis=1)
                gp7 = tl.sum(gt7 * p[None, :], axis=1)
                sr7 = tl.sum(sp7 * r[None, :], axis=1)
                dk7 = b_val * gv7 - d_val * b_val * gp7 - d_val * b_val * sr7
                tl.store(dq_ptr + pid * T * Kb + cur * Kb + 224 + tl.arange(0, 32), dq7, mask=valid)
                tl.store(dk_ptr + pid * T * Kb + cur * Kb + 224 + tl.arange(0, 32), dk7, mask=valid)
                g7 = d_val * (gt7 - b_val * k7[:, None] * r[None, :])
        tl.store(g_state_ptr + pid * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], g0)
        if Kb > 32:
            tl.store(g_state_ptr + pid * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], g1)
        if Kb > 64:
            tl.store(g_state_ptr + pid * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], g2)
        if Kb > 96:
            tl.store(g_state_ptr + pid * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], g3)
        if Kb > 128:
            tl.store(g_state_ptr + pid * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], g4)
        if Kb > 160:
            tl.store(g_state_ptr + pid * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], g5)
        if Kb > 192:
            tl.store(g_state_ptr + pid * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], g6)
        if Kb > 224:
            tl.store(g_state_ptr + pid * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], g7)

class DeltaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, e, g, beta, gamma, scale):
        B, H, T, K = q.shape
        Vd = v.shape[-1]
        BH = B * H
        scale_f = float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        Kb = ((K + 31) // 32) * 32
        Vb = ((Vd + 31) // 32) * 32
        if Kb == 0:
            Kb = 32
        if Vb == 0:
            Vb = 32
        nC = (T + 31) // 32
        q_r = q.reshape(BH, T, K).contiguous()
        k_r = k.reshape(BH, T, K).contiguous()
        v_r = v.reshape(BH, T, Vd).contiguous()
        g_r = g.reshape(BH, T).contiguous().to(torch.float32)
        beta_r = beta.reshape(BH, T).contiguous().to(torch.float32)
        q_pad = torch.zeros((BH, T, Kb), device=q.device, dtype=q_r.dtype)
        k_pad = torch.zeros((BH, T, Kb), device=q.device, dtype=k_r.dtype)
        v_pad = torch.zeros((BH, T, Vb), device=q.device, dtype=v_r.dtype)
        if K > 0:
            q_pad[:, :, :K] = q_r
            k_pad[:, :, :K] = k_r
        if Vd > 0:
            v_pad[:, :, :Vd] = v_r
        d = torch.exp(g_r)
        state = torch.zeros((BH, Kb, Vb), device=q.device, dtype=torch.float32)
        out_pad = torch.empty((BH, T, Vb), device=q.device, dtype=torch.float32)
        chunk_states = torch.empty((BH, nC, Kb, Vb), device=q.device, dtype=torch.float32)
        grid = (BH,)
        for ck in range(nC):
            START = ck * 32
            chunk_states[:, ck].copy_(state)
            _fwd_kernel[grid](q_pad, k_pad, v_pad, d, beta_r, out_pad, state, scale_f, T, Kb, Vb, START)
        out = out_pad[:, :, :Vd].reshape(B, H, T, Vd)
        ctx.save_for_backward(q_pad, k_pad, v_pad, d, beta_r, chunk_states)
        ctx.B = B; ctx.H = H; ctx.T = T; ctx.K = K; ctx.Vd = Vd; ctx.Kb = Kb; ctx.Vb = Vb
        ctx.BH = BH; ctx.nC = nC; ctx.scale_f = scale_f
        ctx.q_dtype = q.dtype; ctx.k_dtype = k.dtype; ctx.v_dtype = v.dtype
        ctx.g_dtype = g.dtype; ctx.beta_dtype = beta.dtype; ctx.e_dtype = e.dtype; ctx.gamma_dtype = gamma.dtype
        return out.to(q.dtype) if out.dtype != q.dtype else out
    @staticmethod
    def backward(ctx, go):
        q_pad, k_pad, v_pad, d, beta_r, chunk_states = ctx.saved_tensors
        B = ctx.B; H = ctx.H; T = ctx.T; K = ctx.K; Vd = ctx.Vd; Kb = ctx.Kb; Vb = ctx.Vb
        BH = ctx.BH; nC = ctx.nC; scale_f = ctx.scale_f
        go_pad = torch.zeros((BH, T, Vb), device=go.device, dtype=torch.float32)
        go_r = go.reshape(BH, T, Vd).contiguous().to(torch.float32)
        go_pad[:, :, :Vd] = go_r
        dq_pad = torch.zeros((BH, T, Kb), device=q_pad.device, dtype=torch.float32)
        dk_pad = torch.zeros((BH, T, Kb), device=q_pad.device, dtype=torch.float32)
        dv_pad = torch.zeros((BH, T, Vb), device=q_pad.device, dtype=torch.float32)
        db_pad = torch.zeros((BH, T, Vb), device=q_pad.device, dtype=torch.float32)
        dg_pad = torch.zeros((BH, T, Vb), device=q_pad.device, dtype=torch.float32)
        tmp = torch.empty((BH, 32, Kb, Vb), device=q_pad.device, dtype=torch.float32)
        cur_state = torch.empty((BH, Kb, Vb), device=q_pad.device, dtype=torch.float32)
        g_state = torch.zeros((BH, Kb, Vb), device=q_pad.device, dtype=torch.float32)
        grid = (BH,)
        for ck in range(nC - 1, -1, -1):
            START = ck * 32
            cur_state.copy_(chunk_states[:, ck])
            _recompute_kernel[grid](k_pad, v_pad, d, beta_r, cur_state, tmp, T, Kb, Vb, START)
            _bwd_kernel[grid](q_pad, k_pad, v_pad, d, beta_r, go_pad, tmp, chunk_states[:, ck], g_state, dq_pad, dk_pad, dv_pad, db_pad, dg_pad, scale_f, T, Kb, Vb, START)
        dq = dq_pad[:, :, :K].reshape(B, H, T, K).to(ctx.q_dtype)
        dk = dk_pad[:, :, :K].reshape(B, H, T, K).to(ctx.k_dtype)
        dv = dv_pad[:, :, :Vd].reshape(B, H, T, Vd).to(ctx.v_dtype)
        dg = dg_pad.sum(dim=2).reshape(B, H, T).to(ctx.g_dtype)
        db = db_pad.sum(dim=2).reshape(B, H, T).to(ctx.beta_dtype)
        ge = torch.zeros((B, H, T, K), device=q_pad.device, dtype=ctx.e_dtype)
        gg = torch.zeros((B, H, T), device=q_pad.device, dtype=ctx.gamma_dtype)
        return dq, dk, dv, ge, dg, db, gg, None

def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return DeltaFn.apply(q, k, v, e, g, beta, gamma, scale)
```

The previous checker failure was due to nested `range(0,Vb,32)` with runtime `Vb` and `tl.arange` inside a `for v_off` tiling that Triton lowers as a Python range with non-constexpr bounds, and `MAX_K` global used as loop bound, both illegal in Triton 3.5. The fix keeps a single BH grid, pads K/V to multiples of 32 in Python so inner loads are always 32-wide, hoists `d=exp(g)` to a cheap vectorized `torch.exp` (avoiding `tl.exp` per element and division by `exp(cumsum)` which overflowed for strong decay -25..-5), and implements the gated delta recurrence `S=d*S+b*k*(v-d*k^T S)` with only constexpr `for v_off in range(0,Vb,32)` (Vb constexpr) and `if Kb>...` static branches, keeping state in 32x32 registers to stay in SRAM.