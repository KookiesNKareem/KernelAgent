"""Demo workloads written the way eager PyTorch code usually is, with helpers marked as kernel sites.

Both pass non-contiguous views to their helpers in ordinary use: q and k arrive at the rotary
embedding transposed out of the head split, the SwiGLU halves are slices of one fused projection,
and the ConvNeXt layer norm runs on a permuted tensor. Weights are random; the point is the ops.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .model import kernel_site


@kernel_site
def rms_norm(x, weight, eps: float):
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(v + eps)).to(x.dtype) * weight


@kernel_site
def apply_rotary(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


@kernel_site
def swiglu(gate, up):
    return F.silu(gate) * up


class LlamaBlock(nn.Module):
    def __init__(self, dim=2048, heads=16, hidden=5632, seq=2048):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.n1 = nn.Parameter(torch.ones(dim))
        self.n2 = nn.Parameter(torch.ones(dim))
        pos = torch.arange(seq).float()
        inv = 1.0 / (10000 ** (torch.arange(0, self.hd, 2).float() / self.hd))
        ang = torch.outer(pos, inv)
        emb = torch.cat((ang, ang), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None])
        self.register_buffer("sin", emb.sin()[None, None])

    def forward(self, x):
        b, t, d = x.shape
        h = rms_norm(x, self.n1, 1e-6)
        q, k, v = self.qkv(h).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.heads, self.hd).transpose(1, 2) for z in (q, k, v))
        cos, sin = self.cos[:, :, :t].to(x.dtype), self.sin[:, :, :t].to(x.dtype)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, d))
        h = rms_norm(x, self.n2, 1e-6)
        gate, up = self.gate_up(h).chunk(2, dim=-1)
        return x + self.down(swiglu(gate, up))


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, x):
        h = self.dw(x).permute(0, 2, 3, 1)
        h = self.pw2(F.gelu(self.pw1(self.norm(h))))
        return x + (self.gamma * h).permute(0, 3, 1, 2)


def llama_block(device="cuda", dtype=torch.bfloat16, seq=2048):
    torch.manual_seed(0)
    m = LlamaBlock(seq=seq).to(device, dtype).eval()
    x = torch.randn(1, seq, 2048, device=device, dtype=dtype)
    return m, (x,)


def convnext_block(device="cuda", dtype=torch.bfloat16):
    torch.manual_seed(0)
    m = nn.Sequential(*[ConvNeXtBlock() for _ in range(3)]).to(device, dtype).eval()
    x = torch.randn(16, 256, 56, 56, device=device, dtype=dtype)
    return m, (x,)


def tiny_llama_cpu():
    """For tests on a machine without a GPU."""
    torch.manual_seed(0)
    m = LlamaBlock(dim=64, heads=4, hidden=96, seq=16).eval()
    return m, (torch.randn(1, 16, 64),)
