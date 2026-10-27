"""Pod side of the paper agent: check a recurrent-layer candidate, then time and profile it. One JSON line out.

  python -m kernel_contract.recur_eval CANDIDATE.py [--grads] [--layer gdr|eda]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from .model import load_kernel
from . import recur_bench, recur_eda, recur_vse, recur_wek
from .recur_bench import timeit
from .recur_check import gdr_spec
from .cases import build_cases
from .runner import run_contract

# Shapes the kernel is timed and trained at, plus K != V and B=1 with a ragged T; seed cases only.
WORKLOADS = {"timed": dict(B=4, H=8, T=2048), "train": dict(B=8, H=4, T=512),
             "b1_tail": dict(B=1, H=4, T=300), "k_ne_v": dict(B=2, H=4, T=256, D=128, Dv=64)}

LAYERS = {"gdr": (gdr_spec, recur_bench.inputs), "eda": (recur_eda.spec, recur_eda.inputs),
          "wek": (recur_wek.spec, recur_wek.inputs),
          "vse": (recur_vse.spec, recur_vse.inputs)}

SCALE = 128 ** -0.5


def profile(kern, grads: bool, inputs) -> list[dict]:
    from torch.profiler import ProfilerActivity, profile as prof
    x = inputs(B=4, H=8, T=2048, D=128, grad=grads)
    step = (lambda: kern(*x, SCALE).float().sum().backward()) if grads else (lambda: kern(*x, SCALE))
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    with prof(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5):
            step()
        torch.cuda.synchronize()
    rows = sorted(p.key_averages(), key=lambda e: -e.device_time_total)[:8]
    return [{"kernel": e.key[:60], "ms_per_step": e.device_time_total / 5 / 1000} for e in rows]


def main(path: str, grads: bool, layer: str = "gdr"):
    make_spec, inputs = LAYERS[layer]
    spec = make_spec()
    spec.flags["check_grads"] = grads
    cases = build_cases(spec)
    t_sym = next(k for k, val in spec.sizes.items() if val == 512)
    cases.append({"name": "compound", "axis": "values", "seed": 3, "sizes": {t_sym: 257},
                  "layout": "transposed", "stress": "decay_strong"})
    v = run_contract(spec, Path(path), timeout=900, cases=cases)
    results = list(v.results)
    for wname, shape in WORKLOADS.items():
        if grads and wname == "timed":
            # fp64 autograd through 2048 reference steps at B=4, H=8 needs ~50 GB; keep T, shrink B and H
            wname, shape = "timed_b1h2", dict(B=1, H=2, T=2048)
        ws = make_spec(**shape)
        ws.flags["check_grads"] = grads
        seeds = [c for c in build_cases(ws) if c["name"] in ("observed", "seed1")]
        for r in run_contract(ws, Path(path), timeout=900, cases=seeds).results:
            results.append({**r, "name": f"{wname}/{r['name']}", "axis": "workload"})
    worst = max(results, key=lambda r: {"wrong": 3, "crash": 2, "refuse": 1}.get(r["outcome"], 0))["outcome"]
    out = {"outcome": worst if worst in ("wrong", "crash", "refuse") else "accept", "results": results}
    if all(r["outcome"] not in ("wrong", "crash") for r in results) and \
            any(r["name"] == "observed" and r["outcome"] == "accept" for r in results):
        kern = load_kernel(Path(path))
        if grads:
            x = inputs(B=4, H=8, T=2048, D=128, grad=True)
            out["ms"] = timeit(lambda: kern(*x, SCALE).float().sum().backward(), reps=10)
        else:
            x = inputs(B=4, H=8, T=2048, D=128)
            out["ms"] = timeit(lambda: kern(*x, SCALE), reps=20)
        out["profile"] = profile(kern, grads, inputs)
    print("RESULT " + json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], grads="--grads" in sys.argv,
         layer=sys.argv[sys.argv.index("--layer") + 1] if "--layer" in sys.argv else "gdr")
