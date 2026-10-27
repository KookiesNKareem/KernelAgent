import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, out_ptr, state_ptr,
                scale, T: tl.constexpr, Kb: tl.constexpr, Vb: tl.constexpr, START: tl.constexpr):
    pid = tl.program_id(0)
    for v_off in range(0, Vb, 32):
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
            q0 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 0 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            k1 = tl.zeros((32,), dtype=tl.float32)
            k2 = tl.zeros((32,), dtype=tl.float32)
            k3 = tl.zeros((32,), dtype=tl.float32)
            k4 = tl.zeros((32,), dtype=tl.float32)
            k5 = tl.zeros((32,), dtype=tl.float32)
            k6 = tl.zeros((32,), dtype=tl.float32)
            k7 = tl.zeros((32,), dtype=tl.float32)
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
def _recompute_kernel(k_ptr, v_ptr, d_ptr, beta_ptr, cur_state_ptr, tmp_ptr, T: tl.constexpr, Kb: tl.constexpr, Vb: tl.constexpr, START: tl.constexpr):
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
            tl.store(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s0, mask=valid)
            if Kb > 32:
                tl.store(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s1, mask=valid)
            if Kb > 64:
                tl.store(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s2, mask=valid)
            if Kb > 96:
                tl.store(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s3, mask=valid)
            if Kb > 128:
                tl.store(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s4, mask=valid)
            if Kb > 160:
                tl.store(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s5, mask=valid)
            if Kb > 192:
                tl.store(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s6, mask=valid)
            if Kb > 224:
                tl.store(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], s7, mask=valid)
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
def _bwd_kernel(q_ptr, k_ptr, v_ptr, d_ptr, beta_ptr, go_ptr, tmp_ptr, chunk_start_ptr, g_state_ptr, dq_ptr, dk_ptr, dv_ptr, db_ptr, dg_ptr, scale, T: tl.constexpr, Kb: tl.constexpr, Vb: tl.constexpr, START: tl.constexpr):
    pid = tl.program_id(0)
    for idx in range(32):
        i = 31 - idx
        cur = START + i
        valid = cur < T
        d_val = tl.load(d_ptr + pid * T + cur, mask=valid, other=1.0)
        b_val = tl.load(beta_ptr + pid * T + cur, mask=valid, other=0.0)
        k0 = tl.load(k_ptr + pid * T * Kb + cur * Kb + 0 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
        q0 = tl.load(q_ptr + pid * T * Kb + cur * Kb + 0 + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
        k1 = tl.zeros((32,), dtype=tl.float32)
        k2 = tl.zeros((32,), dtype=tl.float32)
        k3 = tl.zeros((32,), dtype=tl.float32)
        k4 = tl.zeros((32,), dtype=tl.float32)
        k5 = tl.zeros((32,), dtype=tl.float32)
        k6 = tl.zeros((32,), dtype=tl.float32)
        k7 = tl.zeros((32,), dtype=tl.float32)
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
        dq0 = tl.zeros((32,), dtype=tl.float32)
        dq1 = tl.zeros((32,), dtype=tl.float32)
        dq2 = tl.zeros((32,), dtype=tl.float32)
        dq3 = tl.zeros((32,), dtype=tl.float32)
        dq4 = tl.zeros((32,), dtype=tl.float32)
        dq5 = tl.zeros((32,), dtype=tl.float32)
        dq6 = tl.zeros((32,), dtype=tl.float32)
        dq7 = tl.zeros((32,), dtype=tl.float32)
        dk0 = tl.zeros((32,), dtype=tl.float32)
        dk1 = tl.zeros((32,), dtype=tl.float32)
        dk2 = tl.zeros((32,), dtype=tl.float32)
        dk3 = tl.zeros((32,), dtype=tl.float32)
        dk4 = tl.zeros((32,), dtype=tl.float32)
        dk5 = tl.zeros((32,), dtype=tl.float32)
        dk6 = tl.zeros((32,), dtype=tl.float32)
        dk7 = tl.zeros((32,), dtype=tl.float32)
        trace_sum = 0.0
        rp_sum = 0.0
        rv_sum = 0.0
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
            v_vec = tl.load(v_ptr + pid * T * Vb + cur * Vb + v_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            go_vec = tl.load(go_ptr + pid * T * Vb + cur * Vb + v_off + tl.arange(0, 32), mask=valid, other=0.0).to(tl.float32)
            sp0 = tl.load(chunk_start_ptr + pid * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            sp1 = tl.zeros((32, 32), dtype=tl.float32)
            sp2 = tl.zeros((32, 32), dtype=tl.float32)
            sp3 = tl.zeros((32, 32), dtype=tl.float32)
            sp4 = tl.zeros((32, 32), dtype=tl.float32)
            sp5 = tl.zeros((32, 32), dtype=tl.float32)
            sp6 = tl.zeros((32, 32), dtype=tl.float32)
            sp7 = tl.zeros((32, 32), dtype=tl.float32)
            if i != 0:
                sp0 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            if Kb > 32:
                sp1 = tl.load(chunk_start_ptr + pid * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp1 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            if Kb > 64:
                sp2 = tl.load(chunk_start_ptr + pid * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp2 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            if Kb > 96:
                sp3 = tl.load(chunk_start_ptr + pid * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp3 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            if Kb > 128:
                sp4 = tl.load(chunk_start_ptr + pid * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp4 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            if Kb > 160:
                sp5 = tl.load(chunk_start_ptr + pid * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp5 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            if Kb > 192:
                sp6 = tl.load(chunk_start_ptr + pid * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp6 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            if Kb > 224:
                sp7 = tl.load(chunk_start_ptr + pid * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
                if i != 0:
                    sp7 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + (i - 1) * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            sc0 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            sc1 = tl.zeros((32, 32), dtype=tl.float32)
            sc2 = tl.zeros((32, 32), dtype=tl.float32)
            sc3 = tl.zeros((32, 32), dtype=tl.float32)
            sc4 = tl.zeros((32, 32), dtype=tl.float32)
            sc5 = tl.zeros((32, 32), dtype=tl.float32)
            sc6 = tl.zeros((32, 32), dtype=tl.float32)
            sc7 = tl.zeros((32, 32), dtype=tl.float32)
            if Kb > 32:
                sc1 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            if Kb > 64:
                sc2 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            if Kb > 96:
                sc3 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            if Kb > 128:
                sc4 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            if Kb > 160:
                sc5 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            if Kb > 192:
                sc6 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            if Kb > 224:
                sc7 = tl.load(tmp_ptr + pid * 32 * Kb * Vb + i * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :])
            gt0 = g0 + scale * q0[:, None] * go_vec[None, :]
            gt1 = tl.zeros((32, 32), dtype=tl.float32)
            gt2 = tl.zeros((32, 32), dtype=tl.float32)
            gt3 = tl.zeros((32, 32), dtype=tl.float32)
            gt4 = tl.zeros((32, 32), dtype=tl.float32)
            gt5 = tl.zeros((32, 32), dtype=tl.float32)
            gt6 = tl.zeros((32, 32), dtype=tl.float32)
            gt7 = tl.zeros((32, 32), dtype=tl.float32)
            if Kb > 32:
                gt1 = g1 + scale * q1[:, None] * go_vec[None, :]
            if Kb > 64:
                gt2 = g2 + scale * q2[:, None] * go_vec[None, :]
            if Kb > 96:
                gt3 = g3 + scale * q3[:, None] * go_vec[None, :]
            if Kb > 128:
                gt4 = g4 + scale * q4[:, None] * go_vec[None, :]
            if Kb > 160:
                gt5 = g5 + scale * q5[:, None] * go_vec[None, :]
            if Kb > 192:
                gt6 = g6 + scale * q6[:, None] * go_vec[None, :]
            if Kb > 224:
                gt7 = g7 + scale * q7[:, None] * go_vec[None, :]
            p = tl.sum(k0[:, None] * sp0, axis=0)
            if Kb > 32:
                p = p + tl.sum(k1[:, None] * sp1, axis=0)
            if Kb > 64:
                p = p + tl.sum(k2[:, None] * sp2, axis=0)
            if Kb > 96:
                p = p + tl.sum(k3[:, None] * sp3, axis=0)
            if Kb > 128:
                p = p + tl.sum(k4[:, None] * sp4, axis=0)
            if Kb > 160:
                p = p + tl.sum(k5[:, None] * sp5, axis=0)
            if Kb > 192:
                p = p + tl.sum(k6[:, None] * sp6, axis=0)
            if Kb > 224:
                p = p + tl.sum(k7[:, None] * sp7, axis=0)
            r = tl.sum(k0[:, None] * gt0, axis=0)
            if Kb > 32:
                r = r + tl.sum(k1[:, None] * gt1, axis=0)
            if Kb > 64:
                r = r + tl.sum(k2[:, None] * gt2, axis=0)
            if Kb > 96:
                r = r + tl.sum(k3[:, None] * gt3, axis=0)
            if Kb > 128:
                r = r + tl.sum(k4[:, None] * gt4, axis=0)
            if Kb > 160:
                r = r + tl.sum(k5[:, None] * gt5, axis=0)
            if Kb > 192:
                r = r + tl.sum(k6[:, None] * gt6, axis=0)
            if Kb > 224:
                r = r + tl.sum(k7[:, None] * gt7, axis=0)
            tl.store(dv_ptr + pid * T * Vb + cur * Vb + v_off + tl.arange(0, 32), b_val * r, mask=valid)
            rp_sum = rp_sum + tl.sum(r * p)
            rv_sum = rv_sum + tl.sum(r * v_vec)
            trace_sum = trace_sum + tl.sum(gt0 * sp0)
            if Kb > 32:
                trace_sum = trace_sum + tl.sum(gt1 * sp1)
            if Kb > 64:
                trace_sum = trace_sum + tl.sum(gt2 * sp2)
            if Kb > 96:
                trace_sum = trace_sum + tl.sum(gt3 * sp3)
            if Kb > 128:
                trace_sum = trace_sum + tl.sum(gt4 * sp4)
            if Kb > 160:
                trace_sum = trace_sum + tl.sum(gt5 * sp5)
            if Kb > 192:
                trace_sum = trace_sum + tl.sum(gt6 * sp6)
            if Kb > 224:
                trace_sum = trace_sum + tl.sum(gt7 * sp7)
            dq0 = dq0 + scale * tl.sum(sc0 * go_vec[None, :], axis=1)
            dk0 = dk0 + b_val * tl.sum(gt0 * v_vec[None, :], axis=1) - d_val * b_val * tl.sum(gt0 * p[None, :], axis=1) - d_val * b_val * tl.sum(sp0 * r[None, :], axis=1)
            if Kb > 32:
                dq1 = dq1 + scale * tl.sum(sc1 * go_vec[None, :], axis=1)
                dk1 = dk1 + b_val * tl.sum(gt1 * v_vec[None, :], axis=1) - d_val * b_val * tl.sum(gt1 * p[None, :], axis=1) - d_val * b_val * tl.sum(sp1 * r[None, :], axis=1)
            if Kb > 64:
                dq2 = dq2 + scale * tl.sum(sc2 * go_vec[None, :], axis=1)
                dk2 = dk2 + b_val * tl.sum(gt2 * v_vec[None, :], axis=1) - d_val * b_val * tl.sum(gt2 * p[None, :], axis=1) - d_val * b_val * tl.sum(sp2 * r[None, :], axis=1)
            if Kb > 96:
                dq3 = dq3 + scale * tl.sum(sc3 * go_vec[None, :], axis=1)
                dk3 = dk3 + b_val * tl.sum(gt3 * v_vec[None, :], axis=1) - d_val * b_val * tl.sum(gt3 * p[None, :], axis=1) - d_val * b_val * tl.sum(sp3 * r[None, :], axis=1)
            if Kb > 128:
                dq4 = dq4 + scale * tl.sum(sc4 * go_vec[None, :], axis=1)
                dk4 = dk4 + b_val * tl.sum(gt4 * v_vec[None, :], axis=1) - d_val * b_val * tl.sum(gt4 * p[None, :], axis=1) - d_val * b_val * tl.sum(sp4 * r[None, :], axis=1)
            if Kb > 160:
                dq5 = dq5 + scale * tl.sum(sc5 * go_vec[None, :], axis=1)
                dk5 = dk5 + b_val * tl.sum(gt5 * v_vec[None, :], axis=1) - d_val * b_val * tl.sum(gt5 * p[None, :], axis=1) - d_val * b_val * tl.sum(sp5 * r[None, :], axis=1)
            if Kb > 192:
                dq6 = dq6 + scale * tl.sum(sc6 * go_vec[None, :], axis=1)
                dk6 = dk6 + b_val * tl.sum(gt6 * v_vec[None, :], axis=1) - d_val * b_val * tl.sum(gt6 * p[None, :], axis=1) - d_val * b_val * tl.sum(sp6 * r[None, :], axis=1)
            if Kb > 224:
                dq7 = dq7 + scale * tl.sum(sc7 * go_vec[None, :], axis=1)
                dk7 = dk7 + b_val * tl.sum(gt7 * v_vec[None, :], axis=1) - d_val * b_val * tl.sum(gt7 * p[None, :], axis=1) - d_val * b_val * tl.sum(sp7 * r[None, :], axis=1)
            ng0 = d_val * (gt0 - b_val * k0[:, None] * r[None, :])
            tl.store(g_state_ptr + pid * Kb * Vb + 0 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], ng0)
            if Kb > 32:
                ng1 = d_val * (gt1 - b_val * k1[:, None] * r[None, :])
                tl.store(g_state_ptr + pid * Kb * Vb + 32 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], ng1)
            if Kb > 64:
                ng2 = d_val * (gt2 - b_val * k2[:, None] * r[None, :])
                tl.store(g_state_ptr + pid * Kb * Vb + 64 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], ng2)
            if Kb > 96:
                ng3 = d_val * (gt3 - b_val * k3[:, None] * r[None, :])
                tl.store(g_state_ptr + pid * Kb * Vb + 96 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], ng3)
            if Kb > 128:
                ng4 = d_val * (gt4 - b_val * k4[:, None] * r[None, :])
                tl.store(g_state_ptr + pid * Kb * Vb + 128 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], ng4)
            if Kb > 160:
                ng5 = d_val * (gt5 - b_val * k5[:, None] * r[None, :])
                tl.store(g_state_ptr + pid * Kb * Vb + 160 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], ng5)
            if Kb > 192:
                ng6 = d_val * (gt6 - b_val * k6[:, None] * r[None, :])
                tl.store(g_state_ptr + pid * Kb * Vb + 192 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], ng6)
            if Kb > 224:
                ng7 = d_val * (gt7 - b_val * k7[:, None] * r[None, :])
                tl.store(g_state_ptr + pid * Kb * Vb + 224 * Vb + v_off + tl.arange(0, 32)[:, None] * Vb + tl.arange(0, 32)[None, :], ng7)
        tl.store(dq_ptr + pid * T * Kb + cur * Kb + 0 + tl.arange(0, 32), dq0, mask=valid)
        if Kb > 32:
            tl.store(dq_ptr + pid * T * Kb + cur * Kb + 32 + tl.arange(0, 32), dq1, mask=valid)
        if Kb > 64:
            tl.store(dq_ptr + pid * T * Kb + cur * Kb + 64 + tl.arange(0, 32), dq2, mask=valid)
        if Kb > 96:
            tl.store(dq_ptr + pid * T * Kb + cur * Kb + 96 + tl.arange(0, 32), dq3, mask=valid)
        if Kb > 128:
            tl.store(dq_ptr + pid * T * Kb + cur * Kb + 128 + tl.arange(0, 32), dq4, mask=valid)
        if Kb > 160:
            tl.store(dq_ptr + pid * T * Kb + cur * Kb + 160 + tl.arange(0, 32), dq5, mask=valid)
        if Kb > 192:
            tl.store(dq_ptr + pid * T * Kb + cur * Kb + 192 + tl.arange(0, 32), dq6, mask=valid)
        if Kb > 224:
            tl.store(dq_ptr + pid * T * Kb + cur * Kb + 224 + tl.arange(0, 32), dq7, mask=valid)
        tl.store(dk_ptr + pid * T * Kb + cur * Kb + 0 + tl.arange(0, 32), dk0, mask=valid)
        if Kb > 32:
            tl.store(dk_ptr + pid * T * Kb + cur * Kb + 32 + tl.arange(0, 32), dk1, mask=valid)
        if Kb > 64:
            tl.store(dk_ptr + pid * T * Kb + cur * Kb + 64 + tl.arange(0, 32), dk2, mask=valid)
        if Kb > 96:
            tl.store(dk_ptr + pid * T * Kb + cur * Kb + 96 + tl.arange(0, 32), dk3, mask=valid)
        if Kb > 128:
            tl.store(dk_ptr + pid * T * Kb + cur * Kb + 128 + tl.arange(0, 32), dk4, mask=valid)
        if Kb > 160:
            tl.store(dk_ptr + pid * T * Kb + cur * Kb + 160 + tl.arange(0, 32), dk5, mask=valid)
        if Kb > 192:
            tl.store(dk_ptr + pid * T * Kb + cur * Kb + 192 + tl.arange(0, 32), dk6, mask=valid)
        if Kb > 224:
            tl.store(dk_ptr + pid * T * Kb + cur * Kb + 224 + tl.arange(0, 32), dk7, mask=valid)
        tl.store(db_ptr + pid * T + cur, rv_sum - d_val * rp_sum, mask=valid)
        tl.store(dg_ptr + pid * T + cur, d_val * (trace_sum - b_val * rp_sum), mask=valid)


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
        db_pad = torch.zeros((BH, T), device=q_pad.device, dtype=torch.float32)
        dg_pad = torch.zeros((BH, T), device=q_pad.device, dtype=torch.float32)
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
        dg = dg_pad.reshape(B, H, T).to(ctx.g_dtype)
        db = db_pad.reshape(B, H, T).to(ctx.beta_dtype)
        ge = torch.zeros((B, H, T, K), device=q_pad.device, dtype=ctx.e_dtype)
        gg = torch.zeros((B, H, T), device=q_pad.device, dtype=ctx.gamma_dtype)
        return dq, dk, dv, ge, dg, db, gg, None


def kernel_function(q, k, v, e, g, beta, gamma, scale):
    return DeltaFn.apply(q, k, v, e, g, beta, gamma, scale)
