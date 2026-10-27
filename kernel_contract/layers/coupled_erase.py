"""Coupled erase: EDA with per-channel decay, where the erase weakens as the write strengthens.

    S' = diag(exp(g_t)) S_{t-1}
    S'' = (I - gamma_t (1 - beta_t) e_t e_t^T) S'
    S_t = (I - beta_t k_t k_t^T) S'' + beta_t k_t v_t^T,      o_t = scale * S_t^T q_t
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reference(q, k, v, e, g, beta, gamma, scale):
    # q, k, e: [B, H, T, K] (k, e unit-norm); v: [B, H, T, V]; g (log decay, <= 0): [B, H, T, K];
    # beta, gamma in (0, 1): [B, H, T]
    B, H, T, K = q.shape
    V = v.shape[-1]
    q, k, v, e, g, beta, gamma = (x.to(torch.promote_types(x.dtype, torch.float32))
                                  for x in (q, k, v, e, g, beta, gamma))
    S = q.new_zeros(B, H, K, V)
    outs = []
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None]
        et, bt = e[:, :, t], beta[:, :, t]
        erase = gamma[:, :, t] * (1 - bt)
        S = S - erase[..., None, None] * et[..., None] * (S * et[..., None]).sum(-2)[..., None, :]
        kt, vt = k[:, :, t], v[:, :, t]
        pred = (S * kt[..., None]).sum(-2)
        S = S + (bt[..., None] * (vt - pred))[..., None, :] * kt[..., None]
        outs.append((S * (q[:, :, t] * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)


def inputs(B=4, H=8, T=1024, D=128, device="cuda", dtype=torch.bfloat16, grad=False, Dv=None):
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1).to(dtype)
    v = torch.randn(B, H, T, Dv or D, device=device, dtype=dtype)
    e = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1).to(dtype)
    g = F.logsigmoid(torch.randn(B, H, T, D, device=device) + 3)
    beta = torch.rand(B, H, T, device=device).sigmoid()
    gamma = torch.rand(B, H, T, device=device).sigmoid()
    xs = [q, k, v, e, g, beta, gamma]
    if grad:
        for x in xs:
            x.requires_grad_(True)
    return xs


def spec(B=2, H=4, T=512, D=128, Dv=None):
    from ..spec import capture, with_domains
    x = inputs(B=B, H=H, T=T, D=D, Dv=Dv)
    s = capture("kernel_contract.layers.coupled_erase.reference", (*x, D ** -0.5), name="coupled_erase")
    s.flags["any_float_output"] = True
    s.flags["normwise_rtol"] = 2e-2
    return with_domains(s, {1: "l2row", 3: "l2row", 4: "logdecay", 5: "unit", 6: "unit"})
