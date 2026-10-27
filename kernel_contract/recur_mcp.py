"""MCP server for "change the equation, keep training": Muse Code writes the Triton kernel for a recurrent
layer, and these tools are the part it cannot fake. Checks and training run on a GPU host over `--remote`;
a verdict is bound to the sha256 of the exact source it judged.

  python -m kernel_contract.recur_mcp --remote "ssh gpu-host" --remote-dir /workspace/ka

Muse Code settings (mcp_servers block):
  {"recur": {"transport": "stdio", "command": "python3",
             "args": ["-m", "kernel_contract.recur_mcp", "--remote", "ssh gpu-host"]}}
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import shlex
import subprocess
import time
from pathlib import Path

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # mcp < 2
    from mcp.server.fastmcp import FastMCP as MCPServer

from .paper_agent import blocking, describe

mcp = MCPServer("recur-contract")
STATE: dict = {"remote": None, "remote_dir": "/workspace/ka", "events": None, "passed": {}}
BUILTIN = {"gdr": "kernel_contract.recur_bench", "eda": "kernel_contract.recur_eda",
           "wek": "kernel_contract.recur_wek", "vse": "kernel_contract.recur_vse"}


def emit(**e):
    if STATE["events"]:
        with open(STATE["events"], "a") as f:
            f.write(json.dumps({"t": time.time(), **e}, default=str) + "\n")


def remote(cmd: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(shlex.split(STATE["remote"]) + [cmd], input=stdin, capture_output=True)


def upload(local: Path, rel: str):
    remote(f"mkdir -p $(dirname {STATE['remote_dir']}/{rel}) && cat > {STATE['remote_dir']}/{rel}", local.read_bytes())


def module_path(layer: str) -> str:
    return BUILTIN.get(layer, layer)


@mcp.tool()
def describe_layer(layer: str) -> str:
    """The layer's reference loop (the ground truth every kernel is judged against), its module docstring
    with the equation, and the kernel_function signature to implement. `layer` is gdr, eda, wek, vse,
    or a module path such as kernel_contract.layers.my_edit."""
    mod = importlib.import_module(module_path(layer))
    sig = inspect.signature(mod.reference)
    return json.dumps({"equation": (mod.__doc__ or "").strip(), "reference": inspect.getsource(mod.reference),
                       "implement": f"def kernel_function{sig}:", "inputs": inspect.getsource(mod.inputs)}, indent=1)


@mcp.tool()
def check_recurrence(layer: str, kernel_path: str, grads: bool = False) -> str:
    """Judge a kernel module against the layer's reference in float64 on the GPU: 41 cases (seeds, value
    scales, sizes, layouts, decay and gate ranges, a compound case, the timed and training shapes, K != V),
    plus every gradient when grads=True. Returns the verdict, each failing case in plain language, the
    forward or forward+backward time at B=4 H=8 T=2048 K=V=128, and a GPU profile."""
    src = Path(kernel_path).read_bytes()
    digest = hashlib.sha256(src).hexdigest()[:16]
    rel = f"runs_remote/mcp/{digest}.py"
    if module_path(layer).startswith("kernel_contract.layers."):
        lp = Path(importlib.import_module(module_path(layer)).__file__)
        upload(lp, f"kernel_contract/layers/{lp.name}")
    upload(Path(kernel_path), rel)
    emit(event="check_start", layer=layer, kernel=kernel_path, sha=digest, grads=grads)
    p = remote(f"cd {STATE['remote_dir']} && python -m kernel_contract.recur_eval {rel} --layer {module_path(layer)}"
               + (" --grads" if grads else ""))
    out = p.stdout.decode()
    line = next((x for x in out.splitlines() if x.startswith("RESULT ")), None)
    if not line:
        return json.dumps({"error": "the checker itself failed; this is not a verdict on the kernel",
                           "detail": (p.stderr.decode() or out)[-1500:]})
    res = json.loads(line[7:])
    block = blocking(res["results"])
    if not block:
        STATE["passed"][(digest, grads)] = kernel_path
    fails = [{"case": r["name"], "means": describe(r["name"]), "outcome": r["outcome"],
              "detail": (r.get("reason") or f"error {r.get('err', 0):.3g} relative to output scale")[:500]}
             for r in res["results"] if r["outcome"] not in ("accept", "skip")]
    verdict = {"sha256": digest, "verified": not block, "blocking": [r["name"] for r in block], "failures": fails,
               "passed": sum(r["outcome"] == "accept" for r in res["results"]), "cases": len(res["results"]),
               "ms": res.get("ms"), "profile": res.get("profile")}
    emit(event="check_done", layer=layer, results=[[r["name"], r["outcome"]] for r in res["results"]], **verdict)
    return json.dumps(verdict, indent=1)


@mcp.tool()
def train_compare(layer: str, kernel_path: str, steps: int = 200) -> str:
    """Train a small byte-level language model with this layer as its sequence mixer, using the kernel,
    and report step time, tokens/s, loss and validation loss. Refuses unless this exact source (by sha256)
    passed check_recurrence with grads=True."""
    digest = hashlib.sha256(Path(kernel_path).read_bytes()).hexdigest()[:16]
    if (digest, True) not in STATE["passed"]:
        return json.dumps({"refused": "this exact source has not passed check_recurrence with grads=True",
                           "sha256": digest})
    rel = f"runs_remote/mcp/{digest}.py"
    emit(event="train_start", layer=layer, sha=digest, steps=steps)
    p = remote(f"cd {STATE['remote_dir']} && python -m kernel_contract.train_demo --layer {module_path(layer)} "
               f"--mixer {rel} --steps {steps} --out runs_remote/mcp/{digest}_train.jsonl > /dev/null 2>&1; "
               f"cat runs_remote/mcp/{digest}_train.jsonl")
    recs = [json.loads(x) for x in p.stdout.decode().splitlines() if x.startswith("{")]
    if not recs:
        return json.dumps({"error": "training produced no steps", "detail": p.stderr.decode()[-1500:]})
    warm = sorted(r["step_s"] for r in recs[5:]) or [recs[-1]["step_s"]]
    val = [r for r in recs if "val_loss" in r]
    summary = {"sha256": digest, "steps": len(recs), "median_step_ms": 1000 * warm[len(warm) // 2],
               "first_loss": recs[0]["loss"], "final_loss": recs[-1]["loss"],
               "final_val_loss": val[-1]["val_loss"] if val else None, "peak_mem_gb": val[-1].get("peak_mem_gb") if val else None,
               "loss_curve": [round(r["loss"], 4) for r in recs[:: max(1, len(recs) // 20)]]}
    emit(event="train_done", layer=layer, **summary)
    return json.dumps(summary, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    ap.add_argument("--remote-dir", default="/workspace/ka")
    ap.add_argument("--events", help="append tool events here for the dashboard")
    a = ap.parse_args()
    STATE.update(remote=a.remote, remote_dir=a.remote_dir, events=a.events)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
