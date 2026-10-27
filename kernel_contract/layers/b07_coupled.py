"""Benchmark layer b07_coupled.

    S_t = (I - beta_t k_t k_t^T)(I - gamma_t (1 - beta_t) e_t e_t^T) diag(exp g_t) S_{t-1} + beta_t k_t v_t^T,    o_t = scale * S_t^T q_t unless stated otherwise
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
        ct = gamma[:, :, t] * (1 - bt)
        S = S * g[:, :, t].exp()[..., None]
        S = S - ct[..., None, None] * et[..., None] * (S * et[..., None]).sum(-2)[..., None, :]
        pred = (S * kt[..., None]).sum(-2)
        S = S + (bt[..., None] * (vt - pred))[..., None, :] * kt[..., None]
        outs.append((S * (q[:, :, t] * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)


def inputs(**kw):
    return make_inputs(CHANNEL_DECAY, **kw)


def spec(**kw):
    return make_spec(__name__, CHANNEL_DECAY, **kw)


def fla_reduction(q, k, v, e, g, beta, gamma, scale):
    """Expert route: interleave the erase and write steps into a 2T-long chunk_kda call."""
    from fla.ops.kda import chunk_kda

    def inter(a, b):
        return torch.stack([a, b], dim=3).flatten(2, 3)

    zq, zv, zg = torch.zeros_like(q), torch.zeros_like(v), torch.zeros_like(g)
    qq, kk, vv, gg, bb = inter(zq, q), inter(e, k), inter(zv, v), inter(g, zg), inter(gamma * (1 - beta), beta)
    o, _ = chunk_kda(*(x.transpose(1, 2) for x in (qq, kk, vv, gg, bb)), scale=scale)
    return o.transpose(1, 2)[:, :, 1::2]
