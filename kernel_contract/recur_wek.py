"""A variant invented for this project, so no paper or kernel for it exists: per-channel decay, then the
delta-rule write, then an erase along an independent direction that also cleans up the write.

    S'  = diag(exp(g_t)) S_{t-1}                  g_t in R^K, <= 0
    S'' = (I - beta_t k_t k_t^T) S' + beta_t k_t v_t^T
    S_t = (I - gamma_t e_t e_t^T) S'',            o_t = scale * S_t^T q_t

  python -m kernel_contract.recur_wek     # check the reduction, time eager and reduction [--compile]
"""

from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F

from .recur_bench import timeit


def reference(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    q, k, v, e, g, beta, gamma = (x.float() for x in (q, k, v, e, g, beta, gamma))
    S = q.new_zeros(B, H, K, V)
    outs = []
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None]
        kt, vt = k[:, :, t], v[:, :, t]
        pred = (S * kt[..., None]).sum(-2)
        S = S + (beta[:, :, t][..., None] * (vt - pred))[..., None, :] * kt[..., None]
        et = e[:, :, t]
        S = S - gamma[:, :, t][..., None, None] * et[..., None] * (S * et[..., None]).sum(-2)[..., None, :]
        outs.append((S * (q[:, :, t] * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)


def inputs(B=4, H=8, T=1024, D=128, device="cuda", dtype=torch.bfloat16, grad=False):
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1).to(dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    e = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1).to(dtype)
    g = F.logsigmoid(torch.randn(B, H, T, D, device=device) + 3)
    beta = torch.rand(B, H, T, device=device).sigmoid()
    gamma = torch.rand(B, H, T, device=device).sigmoid()
    xs = [q, k, v, e, g, beta, gamma]
    if grad:
        for x in xs:
            x.requires_grad_(True)
    return xs


def spec(B=2, H=4, T=512, D=128):
    from .spec import capture, with_domains
    x = inputs(B=B, H=H, T=T, D=D)
    s = capture("kernel_contract.recur_wek.reference", (*x, D ** -0.5), name="wek")
    s.flags["any_float_output"] = True
    s.flags["normwise_rtol"] = 2e-2
    return with_domains(s, {1: "l2row", 3: "l2row", 4: "logdecay", 5: "unit", 6: "unit"})


def fla_reduction(q, k, v, e, g, beta, gamma, scale):
    """Expert route: interleave write and erase steps into a 2T-long KDA (per-channel gated delta rule)."""
    from fla.ops.kda import chunk_kda

    def inter(a, b):
        return torch.stack([a, b], dim=3).flatten(2, 3)

    qq = inter(torch.zeros_like(q), q)
    kk = inter(k, e)
    vv = inter(v, torch.zeros_like(v))
    gg = inter(g, torch.zeros_like(g))
    bb = inter(beta, gamma)
    o, _ = chunk_kda(*(x.transpose(1, 2) for x in (qq, kk, vv, gg, bb)), scale=scale)
    return o.transpose(1, 2)[:, :, 1::2]


def main():
    x = inputs(B=2, H=4, T=300)
    ref = reference(*x, 128 ** -0.5)
    red = fla_reduction(*x, 128 ** -0.5).float()
    print(json.dumps({"reduction_rel_err": float((red - ref).norm() / ref.norm())}), flush=True)
    scale = 128 ** -0.5
    x = inputs(B=4, H=8, T=2048)
    xg = inputs(B=4, H=8, T=2048, grad=True)
    r = {"eager_ms": timeit(lambda: reference(*x, scale), reps=2, warm=1),
         "reduction_ms": timeit(lambda: fla_reduction(*x, scale), reps=20),
         "eager_fwd_bwd_ms": timeit(lambda: reference(*xg, scale).sum().backward(), reps=1, warm=1),
         "reduction_fwd_bwd_ms": timeit(lambda: fla_reduction(*xg, scale).float().sum().backward(), reps=10)}
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
