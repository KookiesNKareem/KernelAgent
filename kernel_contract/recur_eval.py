"""Pod side of the paper agent: check a recurrent-layer candidate, then time and profile it. One JSON line out.

  python -m kernel_contract.recur_eval CANDIDATE.py [--grads] [--layer gdr|eda]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from .model import load_kernel
from . import recur_bench, recur_eda
from .recur_bench import timeit
from .recur_check import gdr_spec
from .runner import run_contract

LAYERS = {"gdr": (gdr_spec, recur_bench.inputs), "eda": (recur_eda.spec, recur_eda.inputs)}

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
    v = run_contract(spec, Path(path), timeout=900)
    out = {"outcome": v.outcome, "results": v.results}
    if all(r["outcome"] not in ("wrong", "crash") for r in v.results) and \
            any(r["name"] == "observed" and r["outcome"] == "accept" for r in v.results):
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
