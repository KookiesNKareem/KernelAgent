"""Shared inputs and contract spec for the benchmark layers: every layer takes (q, k, v, e, g, beta, gamma, scale)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_inputs(channel_decay: bool, B=4, H=8, T=1024, D=128, device="cuda", dtype=torch.bfloat16, grad=False,
                Dv=None):
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1).to(dtype)
    v = torch.randn(B, H, T, Dv or D, device=device, dtype=dtype)
    e = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1).to(dtype)
    gshape = (B, H, T, D) if channel_decay else (B, H, T)
    g = F.logsigmoid(torch.randn(*gshape, device=device) + (3 if channel_decay else 0))
    beta = torch.rand(B, H, T, device=device).sigmoid()
    gamma = torch.rand(B, H, T, device=device).sigmoid()
    xs = [q, k, v, e, g, beta, gamma]
    if grad:
        for x in xs:
            x.requires_grad_(True)
    return xs


def make_spec(module: str, channel_decay: bool, B=2, H=4, T=512, D=128, Dv=None):
    from ..spec import capture, with_domains
    x = make_inputs(channel_decay, B=B, H=H, T=T, D=D, Dv=Dv)
    s = capture(f"{module}.reference", (*x, D ** -0.5), name=module.rsplit(".", 1)[-1])
    s.flags["any_float_output"] = True
    s.flags["normwise_rtol"] = 2e-2
    return with_domains(s, {1: "l2row", 3: "l2row", 4: "logdecay", 5: "unit", 6: "unit"})
