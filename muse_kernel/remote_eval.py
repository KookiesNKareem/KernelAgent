"""GPU-host side of a build: judge a candidate against the user's own function on its observed contract, with
gradients, then time candidate and reference on the exact observed call. One JSON line out.

  python -m muse_kernel.remote_eval JOB_DIR CANDIDATE.py [--grads]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from kernel_contract.cases import build_cases
from kernel_contract.model import load_kernel
from kernel_contract.recur_bench import timeit
from kernel_contract.runner import run_contract
from kernel_contract.spec import OpSpec, materialize


def observed_inputs(spec: OpSpec, grads: bool):
    args, kwargs = materialize(spec, spec.sizes, 0, "cuda")
    if grads:
        args = [a.detach().requires_grad_(True) if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                for a in args]
    return args, kwargs


def step(fn, args, kwargs, grads: bool):
    def run():
        out = fn(*args, **kwargs)
        if grads:
            out.float().sum().backward()
    return run


def main(job: str, path: str, grads: bool, holdout: str | None = None):
    job = Path(job)
    spec = OpSpec.from_json(json.loads((job / "spec.json").read_text()))
    spec.flags["check_grads"] = grads
    cases = json.loads(Path(holdout).read_text())["cases"] if holdout else build_cases(spec)
    domains = {a["tensor"].get("domain") for a in spec.args if isinstance(a, dict) and "tensor" in a}
    if not holdout and "logdecay" in domains:
        cases.append({"name": "compound", "axis": "values", "seed": 3, "layout": "transposed", "stress": "decay_strong"})
    results = list(run_contract(spec, Path(path), timeout=900, cases=cases).results)
    for i, extra in enumerate(json.loads((job / "extra_specs.json").read_text()) if (job / "extra_specs.json").exists() else []):
        es = OpSpec.from_json(extra)
        es.flags["check_grads"] = grads
        seeds = cases if holdout else [c for c in build_cases(es) if c["name"] in ("observed", "seed1", "gate_typical")]
        for r in run_contract(es, Path(path), timeout=900, cases=seeds).results:
            results.append({**r, "name": f"call{i + 2}/{r['name']}", "axis": "workload"})
    worst = max(results, key=lambda r: {"wrong": 3, "crash": 2, "refuse": 1}.get(r["outcome"], 0))["outcome"]
    out = {"outcome": worst if worst in ("wrong", "crash", "refuse") else "accept", "results": results}
    if all(r["outcome"] not in ("wrong", "crash") for r in results) and \
            any(r["name"] == "observed" and r["outcome"] == "accept" for r in results):
        try:
            kern = load_kernel(Path(path))
            args, kwargs = observed_inputs(spec, grads)
            out["ms"] = timeit(step(kern, args, kwargs, grads), reps=10, warm=2)
            ref_ms = job / ("ref_fwd_bwd_ms" if grads else "ref_fwd_ms")
            if not ref_ms.exists():
                ref_ms.write_text(str(timeit(step(spec.reference(), *observed_inputs(spec, grads), grads), reps=1, warm=1)))
            out["reference_ms"] = float(ref_ms.read_text())
        except Exception as e:
            results.append({"name": "observed/benchmark", "axis": "workload", "outcome": "refuse",
                            "reason": f"{type(e).__name__}: {e}"[:300]})
            out["outcome"] = "refuse"
    print("RESULT " + json.dumps(out, default=str), flush=True)


def time_reference(job: str):
    spec = OpSpec.from_json(json.loads((Path(job) / "spec.json").read_text()))
    args, kwargs = observed_inputs(spec, False)
    print(f"REFERENCE_MS {timeit(step(spec.reference(), args, kwargs, False), reps=1, warm=1)}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--reference":
        time_reference(sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], grads="--grads" in sys.argv,
             holdout=sys.argv[sys.argv.index("--holdout")+1] if "--holdout" in sys.argv else None)
