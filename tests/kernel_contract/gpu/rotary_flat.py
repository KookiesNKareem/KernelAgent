import torch  # assumes a contiguous input: flat row offsets
import triton
import triton.language as tl


@triton.jit
def _rot(x, c, s, o, T, H, D: tl.constexpr, sxb, sxh, sxt, sxd, sct, scd, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b, rem = row // (H * T), row % (H * T)
    h, t = rem // T, rem % T
    half: tl.constexpr = D // 2
    d = tl.arange(0, BLOCK)
    m = d < D
    base = row * D
    xv = tl.load(x + base + d, mask=m, other=0.0).to(tl.float32)
    partner = tl.where(d < half, d + half, d - half)
    xp = tl.load(x + base + partner, mask=m, other=0.0).to(tl.float32)
    rot = tl.where(d < half, -xp, xp)
    cv = tl.load(c + t * sct + d * scd, mask=m, other=0.0).to(tl.float32)
    sv = tl.load(s + t * sct + d * scd, mask=m, other=0.0).to(tl.float32)
    tl.store(o + row * D + d, (xv * cv + rot * sv).to(o.dtype.element_ty), mask=m)


def kernel_function(x, cos, sin):
    B, H, T, D = x.shape
    out = torch.empty((B, H, T, D), dtype=x.dtype, device=x.device)
    c, s = cos.expand(1, 1, T, D)[0, 0], sin.expand(1, 1, T, D)[0, 0]
    _rot[(B * H * T,)](x, c, s, out, T, H, D, *x.stride(), c.stride(0), c.stride(1),
                       BLOCK=triton.next_power_of_2(D))
    return out
