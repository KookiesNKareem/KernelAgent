"""Summarize a benchmark directory: per model and layer, whether forward and forward+backward were verified,
their times against the eager loop and the expert route, and the exact model cost.

  python -m kernel_contract.bench_report runs/bench1 runs/bench_baselines.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from .bench import LAYERS
from .paper_agent import blocking, usage_cost

MODELS = {"muse-wonder-1.0": "Wonder", "muse-spark-1.3-contributor": "Spark 1.3", "muse-spark-open-1026": "Spark Open"}
FULL_ID = {"muse-spark-open-1026": "meta-models/muse-spark-open-1026"}


def summarize(run: Path, model: str) -> dict:
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    calls = [e for e in events if e["event"] == "llm_done"]
    best = {False: None, True: None}
    for e in events:
        if e["event"] != "eval" or blocking(e["results"]) or not e.get("ms"):
            continue
        g = bool(e["grads"])
        if best[g] is None or e["ms"] < best[g]:
            best[g] = e["ms"]
    return {"fwd_ms": best[False], "fwd_bwd_ms": best[True], "calls": len(calls),
            "usd": sum(usage_cost(FULL_ID.get(model, model), e.get("usage")) for e in calls)}


def main(bench: str, base: str):
    bench, base = Path(bench), json.loads(Path(base).read_text())
    table = defaultdict(dict)
    for d in sorted(bench.iterdir()):
        if d.is_dir() and (d / "events.jsonl").exists() and "__" in d.name:
            model, layer = d.name.split("__", 1)
            table[layer][model] = summarize(d, model)
    head = "| Layer | Expert route fwd+bwd | " + " | ".join(f"{m} fwd / fwd+bwd, cost" for m in MODELS.values()) + " |"
    print(head)
    print("|" + "---|" * (2 + len(MODELS)))
    totals = {m: {"fwd": 0, "bwd": 0, "usd": 0.0, "n": 0} for m in MODELS}
    for layer in LAYERS:
        b = base[layer]
        cells = []
        for m in MODELS:
            r = table[layer].get(m)
            if not r:
                cells.append("not run")
                continue
            t = totals[m]
            t["n"] += 1
            t["usd"] += r["usd"]
            t["fwd"] += r["fwd_ms"] is not None
            t["bwd"] += r["fwd_bwd_ms"] is not None
            f = f"{r['fwd_ms']:.1f}" if r["fwd_ms"] else "✗"
            fb = f"{r['fwd_bwd_ms']:.0f} ms ({b['eager_fwd_bwd_ms'] / r['fwd_bwd_ms']:.0f}× eager)" if r["fwd_bwd_ms"] else "✗"
            cells.append(f"{f} / {fb}, ${r['usd']:.2f}")
        expert = f"{b['expert_fwd_bwd_ms']:.1f} ms" if "expert_fwd_bwd_ms" in b else "none"
        print(f"| {layer} | {expert} | " + " | ".join(cells) + " |")
    print()
    for m, name in MODELS.items():
        t = totals[m]
        if t["n"]:
            print(f"{name}: verified forward {t['fwd']}/{t['n']}, verified forward+backward {t['bwd']}/{t['n']}, "
                  f"${t['usd']:.2f} total, ${t['usd'] / t['n']:.2f} per layer")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
