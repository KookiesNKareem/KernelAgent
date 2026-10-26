"""Check a candidate gated-delta-rule kernel against the paper's reference loop, then time it.

  python -m kernel_contract.recur_check CANDIDATE.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch

from .model import load_kernel
from .recur_bench import inputs, timeit
from .runner import run_contract
from .spec import capture, with_domains


def gdr_spec(B=2, H=4, T=512, D=128):
    q, k, v, g, beta = inputs(B=B, H=H, T=T, D=D)
    spec = capture("kernel_contract.recur_bench.reference", (q, k, v, g, beta, D ** -0.5), name="gated_delta_rule")
    spec.flags["any_float_output"] = True
    spec.flags["normwise_rtol"] = 2e-2
    return with_domains(spec, {1: "l2row", 3: "logdecay", 4: "unit"})


def main(path: str, grads: bool = False):
    spec = gdr_spec()
    if grads:
        spec.flags["check_grads"] = True
    v = run_contract(spec, Path(path), timeout=900)
    print(json.dumps({"verdict": v.outcome, "by_outcome": Counter(r["outcome"] for r in v.results),
                      "failures": [(r["name"], r["outcome"], r.get("reason", "")[:160] or round(r.get("err", 0), 4))
                                   for r in v.failures]}, default=str), flush=True)
    if v.outcome not in ("accept",) and any(r["name"] == "observed" and r["outcome"] != "accept" for r in v.results):
        return
    kern = load_kernel(Path(path))
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    q, k, vv, g, beta = inputs(B=4, H=8, T=2048, D=128)
    scale = 128 ** -0.5
    cand_ms = timeit(lambda: kern(q, k, vv, g, beta, scale), reps=20)
    fla_ms = timeit(lambda: chunk_gated_delta_rule(q.transpose(1, 2), k.transpose(1, 2), vv.transpose(1, 2),
                                                   g.transpose(1, 2), beta.transpose(1, 2), scale=scale,
                                                   use_qk_l2norm_in_kernel=False), reps=20)
    print(json.dumps({"candidate_ms": cand_ms, "fla_ms": fla_ms, "eager_ref_ms": 361.0, "compiled_ref_ms": 74.0,
                      "speedup_vs_compiled": 74.0 / cand_ms}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], grads="--grads" in sys.argv)
