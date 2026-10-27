"""The edited layer: erase-then-delta (EDA, arXiv 2606.26560). Before the delta-rule write along k, the state is
erased along an independent direction e:

    S_t = (I - beta_t k_t k_t^T)(I - gamma_t e_t e_t^T) exp(g_t) S_{t-1} + beta_t k_t v_t^T,   o_t = scale * S_t^T q_t

  python -m kernel_contract.recur_eda     # check the reduction, time eager and reduction [--compile]
"""

from __future__ import annotations

import json
import sys
import time

import torch
import torch.nn.functional as F

from .recur_bench import timeit


def reference(q, k, v, e, g, beta, gamma, scale):
    B, H, T, K = q.shape
    V = v.shape[-1]
    q, k, v, e, g, beta, gamma = (x.to(torch.promote_types(x.dtype, torch.float32)) for x in (q, k, v, e, g, beta, gamma))
    S = q.new_zeros(B, H, K, V)
    outs = []
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]
        et = e[:, :, t]
        S = S - gamma[:, :, t][..., None, None] * et[..., None] * (S * et[..., None]).sum(-2)[..., None, :]
        kt, vt = k[:, :, t], v[:, :, t]
        pred = (S * kt[..., None]).sum(-2)
        S = S + (beta[:, :, t][..., None] * (vt - pred))[..., None, :] * kt[..., None]
        outs.append((S * (q[:, :, t] * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)


def inputs(B=4, H=8, T=1024, D=128, device="cuda", dtype=torch.bfloat16, grad=False, Dv=None):
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1).to(dtype)
    v = torch.randn(B, H, T, Dv or D, device=device, dtype=dtype)
    e = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1).to(dtype)
    g = F.logsigmoid(torch.randn(B, H, T, device=device))
    beta = torch.rand(B, H, T, device=device).sigmoid()
    gamma = torch.rand(B, H, T, device=device).sigmoid()
    xs = [q, k, v, e, g, beta, gamma]
    if grad:
        for x in xs:
            x.requires_grad_(True)
    return xs


def spec(B=2, H=4, T=512, D=128, Dv=None):
    from .spec import capture, with_domains
    q, k, v, e, g, beta, gamma = inputs(B=B, H=H, T=T, D=D, Dv=Dv)
    s = capture("kernel_contract.recur_eda.reference", (q, k, v, e, g, beta, gamma, D ** -0.5), name="eda")
    s.flags["any_float_output"] = True
    s.flags["normwise_rtol"] = 2e-2
    return with_domains(s, {1: "l2row", 3: "l2row", 4: "logdecay", 5: "unit", 6: "unit"})


def fla_reduction(q, k, v, e, g, beta, gamma, scale):
    """The paper's Section 3.4 trick: interleave erase and write steps into a 2T-long gated delta rule."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    B, H, T, K = q.shape

    def inter(a, b):
        return torch.stack([a, b], dim=3).flatten(2, 3)

    qq = inter(torch.zeros_like(q), q)
    kk = inter(e, k)
    vv = inter(torch.zeros_like(v), v)
    gg = inter(g, torch.zeros_like(g))
    bb = inter(gamma, beta)
    o, _ = chunk_gated_delta_rule(*(x.transpose(1, 2) for x in (qq, kk, vv, gg, bb)), scale=scale,
                                  use_qk_l2norm_in_kernel=False)
    return o.transpose(1, 2)[:, :, 1::2]


def main():
    x = inputs(B=2, H=4, T=300)
    ref = reference(*x, 128 ** -0.5)
    red = fla_reduction(*x, 128 ** -0.5).float()
    print(json.dumps({"reduction_rel_err": float((red - ref).norm() / ref.norm())}), flush=True)

    scale = 128 ** -0.5
    x = inputs(B=4, H=8, T=2048)
    r = {"eager_ms": timeit(lambda: reference(*x, scale), reps=2, warm=1),
         "reduction_ms": timeit(lambda: fla_reduction(*x, scale), reps=20)}
    xg = inputs(B=4, H=8, T=2048, grad=True)
    r["eager_fwd_bwd_ms"] = timeit(lambda: reference(*xg, scale).sum().backward(), reps=1, warm=1)
    r["reduction_fwd_bwd_ms"] = timeit(lambda: fla_reduction(*xg, scale).float().sum().backward(), reps=10)
    print(json.dumps(r), flush=True)
    if "--compile" not in sys.argv:
        return
    compiled = torch.compile(reference)
    t0 = time.perf_counter()
    compiled(*x, scale)
    r["compile_s"] = time.perf_counter() - t0
    r["compiled_ms"] = timeit(lambda: compiled(*x, scale), reps=3, warm=1)
    print(json.dumps(r), flush=True)
    t0 = time.perf_counter()
    compiled(*xg, scale).sum().backward()
    r["compile_bwd_s"] = time.perf_counter() - t0
    r["compiled_fwd_bwd_ms"] = timeit(lambda: compiled(*xg, scale).sum().backward(), reps=2, warm=1)
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
