import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(x_ptr, a_ptr, y_ptr, B, T, D, n_dblocks,
                BLOCK_D: tl.constexpr, EVEN_D: tl.constexpr):
    pid = tl.program_id(0)
    pid_b = pid // n_dblocks
    pid_d = pid % n_dblocks
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    base = pid_b * T * D
    y = tl.zeros([BLOCK_D], dtype=tl.float32)
    if EVEN_D:
        for t in tl.range(0, T):
            off = base + t * D + offs_d
            x = tl.load(x_ptr + off, eviction_policy='evict_first')
            aa = tl.load(a_ptr + off, eviction_policy='evict_first')
            y = aa * y + x
            tl.store(y_ptr + off, y)
    else:
        mask_d = offs_d < D
        for t in tl.range(0, T):
            off = base + t * D + offs_d
            x = tl.load(x_ptr + off, mask=mask_d, other=0.0, eviction_policy='evict_first')
            aa = tl.load(a_ptr + off, mask=mask_d, other=0.0, eviction_policy='evict_first')
            y = aa * y + x
            tl.store(y_ptr + off, y, mask=mask_d)


@triton.jit
def _bwd_kernel(a_ptr, y_ptr, g_ptr, dx_ptr, da_ptr, B, T, D, n_dblocks,
                BLOCK_D: tl.constexpr, EVEN_D: tl.constexpr):
    pid = tl.program_id(0)
    pid_b = pid // n_dblocks
    pid_d = pid % n_dblocks
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    base = pid_b * T * D
    carry = tl.zeros([BLOCK_D], dtype=tl.float32)
    a_next = tl.zeros([BLOCK_D], dtype=tl.float32)
    if EVEN_D:
        for i in tl.range(0, T - 1):
            t = T - 1 - i
            off = base + t * D + offs_d
            g = tl.load(g_ptr + off, eviction_policy='evict_first')
            G = g + a_next * carry
            a_t = tl.load(a_ptr + off, eviction_policy='evict_first')
            y_prev = tl.load(y_ptr + off - D, eviction_policy='evict_first')
            tl.store(dx_ptr + off, G)
            tl.store(da_ptr + off, G * y_prev)
            carry = G
            a_next = a_t
        # t = 0, y_{-1}=0 so da=0
        off0 = base + offs_d
        g0 = tl.load(g_ptr + off0, eviction_policy='evict_first')
        G0 = g0 + a_next * carry
        tl.store(dx_ptr + off0, G0)
        tl.store(da_ptr + off0, tl.zeros([BLOCK_D], dtype=tl.float32))
    else:
        mask_d = offs_d < D
        for i in tl.range(0, T - 1):
            t = T - 1 - i
            off = base + t * D + offs_d
            g = tl.load(g_ptr + off, mask=mask_d, other=0.0, eviction_policy='evict_first')
            G = g + a_next * carry
            a_t = tl.load(a_ptr + off, mask=mask_d, other=0.0, eviction_policy='evict_first')
            y_prev = tl.load(y_ptr + off - D, mask=mask_d, other=0.0, eviction_policy='evict_first')
            tl.store(dx_ptr + off, G, mask=mask_d)
            tl.store(da_ptr + off, G * y_prev, mask=mask_d)
            carry = G
            a_next = a_t
        off0 = base + offs_d
        g0 = tl.load(g_ptr + off0, mask=mask_d, other=0.0, eviction_policy='evict_first')
        G0 = g0 + a_next * carry
        tl.store(dx_ptr + off0, G0, mask=mask_d)
        tl.store(da_ptr + off0, tl.zeros([BLOCK_D], dtype=tl.float32), mask=mask_d)


def _unbroadcast_to(grad, shape):
    if grad.shape == shape:
        return grad
    while grad.dim() > len(shape):
        grad = grad.sum(dim=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(dim=i, keepdim=True)
    if grad.shape != shape:
        grad = grad.reshape(shape)
    return grad


class _DecayedScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        if x.shape != a.shape:
            bshape = torch.broadcast_shapes(x.shape, a.shape)
            x_exp = x.expand(bshape)
            a_exp = a.expand(bshape)
            shape = bshape
        else:
            x_exp = x
            a_exp = a
            shape = x.shape
        B = shape[0]
        T = shape[1]
        numel = x_exp.numel()
        D = numel // (B * T) if (B * T) != 0 else 1
        x_c = x_exp.contiguous().reshape(B, T, D)
        a_c = a_exp.contiguous().reshape(B, T, D)
        y_c = torch.empty((B, T, D), device=x.device, dtype=x.dtype)
        BLOCK = 128
        n_dblocks = (D + BLOCK - 1) // BLOCK if D > 0 else 1
        even = (D % BLOCK == 0) if D > 0 else True
        if B > 0 and T > 0 and D > 0:
            grid = (B * n_dblocks,)
            _fwd_kernel[grid](x_c, a_c, y_c, B, T, D, n_dblocks,
                              BLOCK_D=BLOCK, EVEN_D=even,
                              num_warps=4, num_stages=3)
        y = y_c.reshape(shape)
        ctx.save_for_backward(a_c, y_c)
        ctx.B = B
        ctx.T = T
        ctx.D = D
        ctx.shape = shape
        ctx.x_shape = x.shape
        ctx.a_shape = a.shape
        ctx.BLOCK = BLOCK
        ctx.n_dblocks = n_dblocks
        ctx.even = even
        return y

    @staticmethod
    def backward(ctx, gout):
        a_c, y_c = ctx.saved_tensors
        B, T, D = ctx.B, ctx.T, ctx.D
        g_c = gout.contiguous().reshape(B, T, D)
        dx_c = torch.empty((B, T, D), device=g_c.device, dtype=a_c.dtype)
        da_c = torch.empty((B, T, D), device=g_c.device, dtype=a_c.dtype)
        if B > 0 and T > 0 and D > 0:
            grid = (B * ctx.n_dblocks,)
            _bwd_kernel[grid](a_c, y_c, g_c, dx_c, da_c, B, T, D,
                              ctx.n_dblocks, BLOCK_D=ctx.BLOCK,
                              EVEN_D=ctx.even, num_warps=4, num_stages=3)
        dx = _unbroadcast_to(dx_c.reshape(ctx.shape), ctx.x_shape)
        da = _unbroadcast_to(da_c.reshape(ctx.shape), ctx.a_shape)
        return dx, da


def kernel_function(x, a):
    return _DecayedScanFn.apply(x, a)
