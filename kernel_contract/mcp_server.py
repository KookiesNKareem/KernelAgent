"""MCP server exposing the contract checker to a coding agent (Muse Code).

The agent writes kernels itself; the server is the part it cannot fake: it traces the model,
checks a kernel against the contract in isolated subprocesses, and installs verified kernels
behind the guard to compare the whole model.

  python -m kernel_contract.mcp_server --model kernel_contract.demo_models:llama_block

Muse Code settings.json:
  {"mcpServers": {"kernel-contract": {"command": "python", "args": ["-m", "kernel_contract.mcp_server",
   "--model", "kernel_contract.demo_models:llama_block"]}}}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # mcp < 2
    from mcp.server.fastmcp import FastMCP as MCPServer

from .agent import describe_call, describe_case
from .model import Installed, compare_model, envelope, load_kernel, trace
from .run import load_model
from .runner import run_contract

mcp = MCPServer("kernel-contract")
STATE: dict = {"model": None, "inputs": None, "sites": {}, "verdicts": {}, "device": "cuda"}


@mcp.tool()
def list_sites(top: int = 10) -> str:
    """The model's op calls, hottest first: name, share of op time, the exact call (shapes, dtypes,
    strides as the model passes them), and the signature a kernel_function must have."""
    sites = list(STATE["sites"].values())
    total = sum(s.ms_total for s in sites) or 1.0
    rows = [{"site": s.spec.name, "share_of_op_time": round(s.ms_total / total, 3), "calls": s.calls,
             "call": describe_call(s.spec)} for s in sites[:top]]
    return json.dumps(rows, indent=1)


@mcp.tool()
def check_kernel(site: str, kernel_path: str) -> str:
    """Run the contract on a kernel module that defines kernel_function with the site's arguments.
    Returns the verdict, every failing case in plain language with its outcome, and the deployable
    envelope (which layouts and sizes the guard may route to it)."""
    s = STATE["sites"][site]
    v = run_contract(s.spec, Path(kernel_path), device=STATE["device"])
    STATE["verdicts"][site] = (kernel_path, v)
    fails = [{"case": r["name"], "means": describe_case(r["name"], s.spec), "outcome": r["outcome"],
              "detail": r.get("reason") or {k: r.get(k) for k in ("err", "torch_err", "scale")}}
             for r in v.failures]
    return json.dumps({"verdict": v.outcome, "failures": fails,
                       "passed": [r["name"] for r in v.results if r["outcome"] == "accept"],
                       "deployable": envelope(s.spec, v.results)}, indent=1)


@mcp.tool()
def install_and_compare() -> str:
    """Install every checked kernel whose envelope is deployable behind the guard, run the model,
    and report max output difference, eager vs guarded time, and how many calls each kernel took
    versus fell back to PyTorch (with reasons)."""
    installed = []
    for site, (path, v) in STATE["verdicts"].items():
        env = envelope(STATE["sites"][site].spec, v.results)
        if env:
            installed.append(Installed(STATE["sites"][site].spec, load_kernel(Path(path)), env["strided_ok"],
                                       frozenset(env["pinned"])))
    report = compare_model(STATE["model"], STATE["inputs"], installed)
    report["installed"] = [i.spec.name for i in installed]
    return json.dumps(report, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    a = ap.parse_args()
    STATE["device"] = a.device
    STATE["model"], STATE["inputs"] = load_model(a.model)
    STATE["sites"] = {s.spec.name: s for s in trace(STATE["model"], STATE["inputs"], device=a.device)}
    mcp.run(transport=a.transport)


if __name__ == "__main__":
    main()
