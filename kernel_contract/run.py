"""End to end: trace a model, write and verify kernels for its hottest ops, install them behind
the guard, and compare the model.

  python -m kernel_contract.run --model demo_models:llama_block --model-id muse-spark-1.2 --top 4 --out runs/x

Every step is appended to OUT/events.jsonl, which the dashboard tails.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import torch

from .agent import solve
from .model import Installed, compare_model, load_kernel, trace
from .runner import run_contract


class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: dict):
        event = {"t": time.time(), **event}
        with open(self.path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
        if event["event"] not in ("case",):
            print(json.dumps({k: v for k, v in event.items() if k not in ("usage",)}, default=str)[:300], flush=True)


def load_model(ref: str):
    mod, _, fn = ref.partition(":")
    return getattr(importlib.import_module(mod), fn)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="module:function returning (model, inputs)")
    ap.add_argument("--model-id", default="muse-spark-1.2")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--repairs", type=int, default=3)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    log = Log(a.out / "events.jsonl")
    model, inputs = load_model(a.model)
    log({"event": "trace_start", "model": a.model})
    sites = trace(model, inputs, device=a.device)
    total = sum(s.ms_total for s in sites)
    log({"event": "trace_done", "sites": [{"op": s.spec.name, "call": s.spec.op, "calls": s.calls,
                                           "ms_total": s.ms_total, "share": s.ms_total / total} for s in sites[:20]]})
    installed = []
    llm_kwargs = {"reasoning_effort": a.reasoning_effort} if a.reasoning_effort else {}
    for site in sites[: a.top]:
        res = solve(site.spec, a.model_id, a.out / "kernels", max_repairs=a.repairs, device=a.device,
                    emit=log, **llm_kwargs)
        final = res.final
        if final is None:
            continue
        path = a.out / "kernels" / f"{site.spec.name}_final.py"
        path.write_text(final.source)
        if final.verdict.outcome == "accept":
            installed.append(Installed(site.spec, load_kernel(path), strided_ok=True))
            log({"event": "install", "op": site.spec.name, "contract": "strided"})
            continue
        # a kernel that only refuses strided views is still safe behind the guard if it passes the dense contract
        dense = run_contract(site.spec, path, contract="dense", device=a.device)
        if dense.outcome == "accept":
            installed.append(Installed(site.spec, load_kernel(path), strided_ok=False))
            log({"event": "install", "op": site.spec.name, "contract": "dense"})
        else:
            log({"event": "reject", "op": site.spec.name, "outcome": final.verdict.outcome})
    report = compare_model(model, inputs, installed)
    log({"event": "model_compare", **report})
    (a.out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
