"""Parent side: run every case against a kernel in subprocesses and summarize the verdict."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .cases import build_cases
from .spec import OpSpec

SEVERITY = {"wrong": 3, "crash": 2, "refuse": 1, "accept": 0, "skip": 0}
REPO = Path(__file__).resolve().parents[1]


@dataclass
class Verdict:
    outcome: str
    results: list[dict] = field(default_factory=list)

    @property
    def failures(self) -> list[dict]:
        return [r for r in self.results if r["outcome"] in ("wrong", "crash", "refuse")]

    def to_json(self) -> dict:
        return {"outcome": self.outcome, "results": self.results}


def run_contract(spec: OpSpec, kernel: Path, contract: str = "strided", device: str = "cuda",
                 timeout: int = 300, cases: list[dict] | None = None, on_case=None) -> Verdict:
    cases = cases if cases is not None else build_cases(spec, contract)
    with tempfile.TemporaryDirectory(prefix="contract_") as d:
        d = Path(d)
        (d / "spec.json").write_text(json.dumps(spec.to_json()))
        (d / "cases.json").write_text(json.dumps(cases))
        out = d / "out.jsonl"
        out.touch()
        seen = 0
        while seen < len(cases):
            try:
                subprocess.run([sys.executable, "-m", "kernel_contract.worker", str(d / "spec.json"),
                                str(kernel.resolve()), str(d / "cases.json"), str(out), str(seen), device],
                               cwd=REPO, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
            lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
            for r in lines[seen:]:
                if on_case:
                    on_case(r)
            if len(lines) == seen:
                # the process died or hung before finishing this case
                c = cases[seen]
                r = {"name": c["name"], "axis": c["axis"], "outcome": "crash", "reason": "process died or timed out"}
                with open(out, "a") as f:
                    f.write(json.dumps(r) + "\n")
                if on_case:
                    on_case(r)
                lines.append(r)
            seen = len(lines)
        results = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    worst = max(results, key=lambda r: SEVERITY[r["outcome"]])["outcome"] if results else "accept"
    return Verdict("accept" if worst == "skip" else worst, results)
