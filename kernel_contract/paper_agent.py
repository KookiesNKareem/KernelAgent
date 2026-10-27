"""Paper to kernel, unattended: a Muse model writes a chunked Triton kernel for a recurrent layer from its
reference loop, the contract checker on a GPU host judges it, and the checker's report is the only feedback.

  python -m kernel_contract.paper_agent --model muse-wonder-1.0 --remote ./pod.sh --out runs/wonder1

Model calls run here, so the Meta spend ledger stays on this machine; `--remote` is a command prefix that
runs a shell command on the GPU host, where the repo lives at --remote-dir.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

from utils.providers import get_model_provider
from utils.providers.meta_provider import spent

from .agent import CASE_TEXT, extract_code

OBSERVED = "B=2, H=4, T=512, K=V=128"
SYMBOLS = {"s0": "H", "s1": "T", "s2": "K=V"}

REFERENCE = '''def reference(q, k, v, g, beta, scale):
    # q, k: [B, H, T, K]; v: [B, H, T, V]; g (log decay, <= 0): [B, H, T]; beta in (0,1): [B, H, T]
    B, H, T, K = q.shape
    V = v.shape[-1]
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    S = q.new_zeros(B, H, K, V)
    outs = []
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]
        kt, vt = k[:, :, t], v[:, :, t]
        pred = (S * kt[..., None]).sum(-2)                      # S^T k_t, shape [B,H,V]
        S = S + (beta[:, :, t][..., None] * (vt - pred))[..., None, :] * kt[..., None]
        outs.append((S * (q[:, :, t] * scale)[..., None]).sum(-2))
    return torch.stack(outs, dim=2)                            # [B, H, T, V], float32'''

WRITE = f"""A research paper introduces this recurrent sequence layer. Its reference implementation is a Python loop
over time:

```python
{REFERENCE}
```

In matrix form, with state S of shape [K, V]: S_t = exp(g_t) * (I - beta_t k_t k_t^T) S_{{t-1}} + beta_t k_t v_t^T,
and o_t = scale * S_t^T q_t.

The loop takes 361 ms forward at B=4, H=8, T=2048, K=V=128 on an NVIDIA L4, too slow to train with. Write a
fast Triton implementation for the L4 (Ada, sm_89, Triton 3.5).

Requirements:
- Define `def kernel_function(q, k, v, g, beta, scale):` with the reference's argument order, shapes and
  semantics, returning a [B, H, T, V] tensor (float32 or q's dtype). q, k, v arrive as bf16, g and beta as fp32.
- It must match the reference on any inputs the reference accepts.
- The computation must happen in your own @triton.jit kernels; the Python wrapper may only allocate, reshape,
  launch and do cheap elementwise prep. Do not import or copy any existing implementation of this layer.

You cannot run anything. Return the complete module in one ```python block, then 2-4 sentences on your
approach and anything you are unsure about."""

BACKWARD = """Next, make it trainable. Write a new self-contained module whose kernel_function(q, k, v, g, beta, scale)
returns the output through a torch.autograd.Function, so that out.backward(grad) yields correct gradients for
q, k, v, g and beta, each in its input's dtype. The backward must also run in your own Triton kernels; Python
may only allocate, launch, and do cheap elementwise or reduction glue. You cannot run anything. Return the
complete module in one ```python block, then 3-5 sentences on your approach."""


def describe(name: str) -> str:
    if name in CASE_TEXT:
        return CASE_TEXT[name]
    if "=" in name:
        sym, val = name.split("=")
        return f"{SYMBOLS.get(sym, sym)}={val} (the observed call is {OBSERVED})"
    return name


def blocking(results: list[dict]) -> list[dict]:
    """Wrong answers and crashes always block; a refusal blocks unless it is a size the kernel declined."""
    return [r for r in results if r["outcome"] in ("wrong", "crash")
            or (r["outcome"] == "refuse" and r.get("axis") != "sizes")]


def report(results: list[dict]) -> str:
    lines = []
    for r in results:
        if r["outcome"] in ("accept", "skip"):
            continue
        if r["outcome"] == "wrong":
            how = r.get("reason") or f"returned wrong values (error {r.get('err', 0):.3g} relative to the output scale)"
        elif r["outcome"] == "refuse":
            how = f"raised {r.get('reason', '')}"
        else:
            how = f"crashed the process ({r.get('reason', '')})"
        lines.append(f"- {r['name']}: {describe(r['name'])}. The kernel {how[:600]}")
    passed = ", ".join(r["name"] for r in results if r["outcome"] == "accept") or "none"
    return "\n".join(lines) + f"\n\nIt passed: {passed}."


def profile_text(res: dict) -> str:
    return "\n".join(f"- {p['kernel']}: {p['ms_per_step']:.3f} ms" for p in res.get("profile", []) if p["ms_per_step"] > 0)


class Run:
    def __init__(self, a):
        self.a, self.out = a, Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.provider = get_model_provider(a.model)
        self.messages: list[dict] = []
        self.start_usd = spent()
        self.n = 0

    def log(self, **e):
        e["t"] = time.time()
        with open(self.out / "events.jsonl", "a") as f:
            f.write(json.dumps(e, default=str) + "\n")
        print(json.dumps({k: v for k, v in e.items() if k not in ("results", "profile")}, default=str), flush=True)

    def ask(self, text: str, tag: str) -> Path:
        if spent() - self.start_usd >= self.a.max_usd:
            raise SystemExit(f"run budget ${self.a.max_usd} reached")
        self.messages.append({"role": "user", "content": text})
        self.log(event="llm_start", tag=tag, prompt=text)
        t0 = time.time()
        resp = self.provider.get_response(self.a.model, self.messages, max_tokens=32000,
                                          reasoning_effort=self.a.effort)
        self.messages.append({"role": "assistant", "content": resp.content})
        self.n += 1
        path = self.out / f"{self.n:02d}_{tag}.py"
        try:
            path.write_text(extract_code(resp.content))
        except ValueError:
            path.write_text("def kernel_function(*a, **k):\n    raise RuntimeError('model returned no code')\n")
        (self.out / f"{self.n:02d}_{tag}.md").write_text(resp.content)
        self.log(event="llm_done", tag=tag, path=str(path), seconds=time.time() - t0, usage=resp.usage,
                 run_usd=spent() - self.start_usd)
        return path

    def evaluate(self, path: Path, grads: bool) -> dict:
        dest = f"{self.a.remote_dir}/runs_remote/{self.out.name}/{path.name}"
        pre = shlex.split(self.a.remote)
        subprocess.run(pre + [f"mkdir -p $(dirname {dest}) && cat > {dest}"], input=path.read_bytes(), check=True)
        cmd = f"cd {self.a.remote_dir} && python -m kernel_contract.recur_eval {dest}" + (" --grads" if grads else "")
        p = subprocess.run(pre + [cmd], capture_output=True, text=True)
        line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
        res = json.loads(line[7:]) if line else {"outcome": "crash", "results": [
            {"name": "observed", "axis": "seed", "outcome": "crash", "reason": (p.stderr or p.stdout)[-600:]}]}
        if re.search(r"^\s*(import|from)\s+fla\b", path.read_text(), re.M):
            res = {"outcome": "refuse", "results": [{"name": "observed", "axis": "seed", "outcome": "refuse",
                                                     "reason": "the module imports an existing implementation"}]}
        res["blocking"] = [r["name"] for r in blocking(res["results"])]
        self.log(event="eval", path=str(path), grads=grads, outcome=res["outcome"], blocking=res["blocking"],
                 ms=res.get("ms"), results=res["results"], profile=res.get("profile"))
        return res

    def until_correct(self, path: Path, res: dict, grads: bool, repairs: int, what: str):
        for _ in range(repairs):
            if not res["blocking"]:
                break
            path = self.ask(f"A contract checker compiled and ran your {what} on an NVIDIA L4 against the reference "
                            f"loop. It failed these cases:\n\n{report(res['results'])}\n\nDiagnose which assumption "
                            f"the failing inputs violate and fix exactly that. Return the complete fixed module.",
                            "repair_bwd" if grads else "repair")
            res = self.evaluate(path, grads)
        return path, res

    def optimize(self, path: Path, res: dict, grads: bool, rounds: int, expert_ms: float):
        best = (path, res)
        for _ in range(rounds):
            p, r = best
            new = self.ask(f"That passed the checker. It runs in {r['ms']:.2f} ms ({'forward+backward' if grads else 'forward'}, "
                           f"B=4, H=8, T=2048, K=V=128 on the L4). "
                           f"GPU profile per step:\n{profile_text(r)}\n\nOptimize for speed while keeping it correct. "
                           f"The checker bounds error relative to the output scale at 2e-2. Return the complete module.",
                           "opt_bwd" if grads else "opt")
            nres = self.evaluate(new, grads)
            new, nres = self.until_correct(new, nres, grads, 1, "optimized kernel")
            if not nres["blocking"] and nres.get("ms") and nres["ms"] < r["ms"]:
                best = (new, nres)
        return best

    def go(self):
        path = self.ask(WRITE, "write")
        res = self.evaluate(path, False)
        path, res = self.until_correct(path, res, False, self.a.repairs, "kernel")
        if res["blocking"]:
            return self.log(event="done", stage="forward", ok=False, run_usd=spent() - self.start_usd)
        path, res = self.optimize(path, res, False, self.a.opt_rounds, 1.54)
        self.log(event="forward_final", path=str(path), ms=res["ms"])
        bpath = self.ask(BACKWARD, "bwd")
        bres = self.evaluate(bpath, True)
        bpath, bres = self.until_correct(bpath, bres, True, self.a.repairs, "backward")
        if not bres["blocking"]:
            bpath, bres = self.optimize(bpath, bres, True, self.a.bwd_opt_rounds, 5.36)
        self.log(event="done", stage="backward", ok=not bres["blocking"], forward=str(path), forward_ms=res["ms"],
                 backward=str(bpath), fwd_bwd_ms=bres.get("ms"), calls=self.n, run_usd=spent() - self.start_usd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="muse-wonder-1.0")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--remote", required=True)
    ap.add_argument("--remote-dir", default="/workspace/ka")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repairs", type=int, default=3)
    ap.add_argument("--opt-rounds", type=int, default=2)
    ap.add_argument("--bwd-opt-rounds", type=int, default=1)
    ap.add_argument("--max-usd", type=float, default=15.0)
    Run(ap.parse_args()).go()


if __name__ == "__main__":
    main()
