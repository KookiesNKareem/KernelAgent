from __future__ import annotations

import torch
import time
import torch
import torch.nn.functional as F


def gated_delta(q, k, v, g, beta, scale):
    B, H, T, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1]).float()
    outs = []
    for t in range(T):
        S = S * g[:, :, t].float().exp()[..., None, None]
        kt = k[:, :, t].float()
        pred = (S * kt[..., None]).sum(-2)
        S = S + (beta[:, :, t].float()[..., None] * (v[:, :, t].float() - pred))[..., None, :] * kt[..., None]
        outs.append((S * (q[:, :, t].float() * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)
