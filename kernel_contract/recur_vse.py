"""A two-sided edit: the state is also erased in value space, from the right, along a unit direction u.

    S_t = (I - beta_t k_t k_t^T) diag(exp(g_t)) S_{t-1} (I - gamma_t u_t u_t^T) + beta_t k_t v_t^T,
    o_t = scale * S_t^T q_t

EDA, DeltaProduct and DPLR transform the state only on the key side, so no existing chunk kernel reduces this.

  python -m kernel_contract.recur_vse     # time eager [--compile]
"""

from __future__ import annotations

import json
import sys
import time

import torch
import torch.nn.functional as F

from .recur_bench import timeit


def reference(q, k, v, u, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    q, k, v, u, g, beta, gamma = (x.to(torch.promote_types(x.dtype, torch.float32))
                                  for x in (q, k, v, u, g, beta, gamma))
    S = q.new_zeros(B, H, K, V)
    outs = []
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None]
        ut = u[:, :, t]
        S = S - gamma[:, :, t][..., None, None] * (S * ut[..., None, :]).sum(-1)[..., None] * ut[..., None, :]
        kt, vt = k[:, :, t], v[:, :, t]
        pred = (S * kt[..., None]).sum(-2)
        S = S + (beta[:, :, t][..., None] * (vt - pred))[..., None, :] * kt[..., None]
        outs.append((S * (q[:, :, t] * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)


def inputs(B=4, H=8, T=1024, D=128, device="cuda", dtype=torch.bfloat16, grad=False, Dv=None):
    torch.manual_seed(0)
    Dv = Dv or D
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1).to(dtype)
    v = torch.randn(B, H, T, Dv, device=device, dtype=dtype)
    u = F.normalize(torch.randn(B, H, T, Dv, device=device), dim=-1).to(dtype)
    g = F.logsigmoid(torch.randn(B, H, T, D, device=device) + 3)
    beta = torch.rand(B, H, T, device=device).sigmoid()
    gamma = torch.rand(B, H, T, device=device).sigmoid()
    xs = [q, k, v, u, g, beta, gamma]
    if grad:
        for x in xs:
            x.requires_grad_(True)
    return xs


def spec(B=2, H=4, T=512, D=128, Dv=None):
    from .spec import capture, with_domains
    x = inputs(B=B, H=H, T=T, D=D, Dv=Dv)
    s = capture("kernel_contract.recur_vse.reference", (*x, D ** -0.5), name="vse")
    s.flags["any_float_output"] = True
    s.flags["normwise_rtol"] = 2e-2
    return with_domains(s, {1: "l2row", 3: "l2row", 4: "logdecay", 5: "unit", 6: "unit"})


fla_reduction = None


def main():
    scale = 128 ** -0.5
    x = inputs(B=4, H=8, T=2048)
    xg = inputs(B=4, H=8, T=2048, grad=True)
    r = {"eager_ms": timeit(lambda: reference(*x, scale), reps=2, warm=1),
         "eager_fwd_bwd_ms": timeit(lambda: reference(*xg, scale).sum().backward(), reps=1, warm=1)}
    print(json.dumps(r), flush=True)
    if "--compile" in sys.argv:
        c = torch.compile(reference)
        t0 = time.perf_counter()
        c(*x, scale)
        r["compile_fwd_s"] = time.perf_counter() - t0
        r["compiled_fwd_ms"] = timeit(lambda: c(*x, scale), reps=3, warm=1)
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
