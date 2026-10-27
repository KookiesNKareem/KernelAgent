"""Kill test for the paper-to-kernel idea: how far is torch.compile from a hand-written chunked
kernel on a recurrent layer whose reference is a loop over time?

Gated delta rule (DeltaNet family): S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T, o_t = S_t q_t.
Compares the plain-PyTorch reference (eager and compiled) with flash-linear-attention's chunk kernel,
forward and forward+backward.
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F


def reference(q, k, v, g, beta, scale):
    """The kind of code a paper ships: a Python loop over time, fp32 state."""
    B, H, T, K = q.shape
    V = v.shape[-1]
    q, k, v, g, beta = (x.to(torch.promote_types(x.dtype, torch.float32)) for x in (q, k, v, g, beta))
    S = q.new_zeros(B, H, K, V)
    outs = []
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]
        kt, vt = k[:, :, t], v[:, :, t]
        pred = (S * kt[..., None]).sum(-2)
        S = S + (beta[:, :, t][..., None] * (vt - pred))[..., None, :] * kt[..., None]
        outs.append((S * (q[:, :, t] * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)


def inputs(B=4, H=8, T=1024, D=128, device="cuda", dtype=torch.bfloat16, grad=False):
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = F.normalize(torch.randn(B, H, T, D, device=device, dtype=torch.float32), dim=-1).to(dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, H, T, device=device, dtype=torch.float32))
    beta = torch.rand(B, H, T, device=device, dtype=torch.float32).sigmoid()
    xs = [q, k, v, g, beta]
    if grad:
        for x in xs:
            x.requires_grad_(True)
    return xs


def timeit(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / reps


def main():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    res = {}
    for T in (512, 2048):
        q, k, v, g, beta = inputs(T=T)
        scale = q.shape[-1] ** -0.5

        def fla_fwd():
            o, _ = chunk_gated_delta_rule(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                          g.transpose(1, 2), beta.transpose(1, 2), scale=scale,
                                          use_qk_l2norm_in_kernel=False)
            return o.transpose(1, 2)

        ref = reference(q, k, v, g, beta, scale)
        err = float((fla_fwd().float() - ref).abs().max())
        compiled = torch.compile(reference)
        t_c0 = time.perf_counter()
        compiled(q, k, v, g, beta, scale)
        compile_s = time.perf_counter() - t_c0
        r = {"eager_ms": timeit(lambda: reference(q, k, v, g, beta, scale), reps=2, warm=1),
             "compiled_ms": timeit(lambda: compiled(q, k, v, g, beta, scale), reps=3, warm=1),
             "compile_s": compile_s,
             "fla_ms": timeit(fla_fwd, reps=20),
             "fla_vs_ref_maxerr": err, "ref_scale": float(ref.abs().max())}

        qg, kg, vg, gg, bg = inputs(T=T, grad=True)
        r["eager_fwd_bwd_ms"] = timeit(lambda: reference(qg, kg, vg, gg, bg, scale).sum().backward(), reps=1, warm=1)
        r["fla_fwd_bwd_ms"] = timeit(lambda: chunk_gated_delta_rule(
            qg.transpose(1, 2), kg.transpose(1, 2), vg.transpose(1, 2), gg.transpose(1, 2), bg.transpose(1, 2),
            scale=scale, use_qk_l2norm_in_kernel=False)[0].sum().backward(), reps=10)
        res[f"T={T}"] = r
        print(json.dumps({f"T={T}": r}), flush=True)


if __name__ == "__main__":
    main()
