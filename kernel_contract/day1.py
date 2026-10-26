"""Day-one measurement: how often does a Muse model's first kernel fail the contract on the demo sites?

  python -m kernel_contract.day1 --model-id muse-spark-1.2 --samples 3 --out runs/day1_spark

Writes one record per (site, sample): verdict of the first write, failing cases by axis and outcome,
and whether one repair fixed it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .agent import solve
from .demo_models import convnext_block, llama_block
from .model import trace


def demo_sites(n_per_model: int = 4):
    out = []
    for build in (llama_block, convnext_block):
        model, inputs = build()
        sites = [s for s in trace(model, inputs) if "linear" not in s.spec.op and "attention" not in s.spec.op
                 and "conv" not in s.spec.op]
        out += sites[:n_per_model]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="muse-spark-1.2")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--repairs", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rec_path = a.out / "records.jsonl"
    for site in demo_sites():
        for k in range(a.samples):
            res = solve(site.spec, a.model_id, a.out / site.spec.name / f"s{k}", max_repairs=a.repairs)
            first, last = res.attempts[0].verdict, res.attempts[-1].verdict
            rec = {"site": site.spec.name, "op": site.spec.op, "sample": k, "first": first.outcome,
                   "first_failures": [(f["name"], f["axis"], f["outcome"]) for f in first.failures],
                   "after_repair": last.outcome, "attempts": len(res.attempts)}
            with open(rec_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    recs = [json.loads(x) for x in rec_path.read_text().splitlines()]
    print("first-write outcomes:", Counter(r["first"] for r in recs))
    print("failing axes:", Counter(ax for r in recs for _, ax, _ in r["first_failures"]))
    print("after repair:", Counter(r["after_repair"] for r in recs))


if __name__ == "__main__":
    main()
