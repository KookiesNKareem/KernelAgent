"""Benchmark layer b11_add_write.

    S_t = (I - beta_t k_t k_t^T) diag(exp g_t) S_{t-1} + beta_t k_t v_t^T + gamma_t e_t v_t^T (a second, additive write along e),    o_t = scale * S_t^T q_t unless stated otherwise
"""

from __future__ import annotations

import torch

from ._common import make_inputs, make_spec

CHANNEL_DECAY = True


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
        kt, vt, et, bt = k[:, :, t], v[:, :, t], e[:, :, t], beta[:, :, t]
        ct = gamma[:, :, t]
        S = S * g[:, :, t].exp()[..., None]
        pred = (S * kt[..., None]).sum(-2)
        S = S + (bt[..., None] * (vt - pred))[..., None, :] * kt[..., None]
        S = S + ct[..., None, None] * et[..., None] * vt[..., None, :]
        outs.append((S * (q[:, :, t] * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)


def inputs(**kw):
    return make_inputs(CHANNEL_DECAY, **kw)


def spec(**kw):
    return make_spec(__name__, CHANNEL_DECAY, **kw)


fla_reduction = None  # no reduction to an existing FLA kernel
