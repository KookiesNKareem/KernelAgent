"""Write a kernel for an op with a Muse model, check it against the contract, repair from counterexamples."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from utils.providers import get_model_provider

from .runner import Verdict, run_contract
from .spec import OpSpec

CASE_TEXT = {
    "observed": "the exact inputs the model passes this op",
    "seed1": "a fresh random draw at the observed shapes",
    "seed2": "another fresh random draw at the observed shapes",
    "scale_up": "every floating input multiplied by 1000",
    "scale_down": "every floating input multiplied by 0.001",
    "shift": "every floating input shifted by +100",
    "negative": "every floating input replaced by -|x| - 1 (all strictly negative)",
    "half_zero": "every floating input clamped at zero (about half the values are exactly 0)",
    "constant": "every floating input set to the constant 0.5",
    "scale_mid": "every floating input multiplied by 37",
    "shift_neg": "every floating input shifted by -100",
    "contiguous": "the same values in a fresh contiguous tensor",
    "strided2": "each tensor is a view with its last-dimension stride doubled (x.stride(-1) == 2)",
    "strided3_off1": "each tensor is a view with its last-dimension stride tripled and a storage offset of 1",
    "transposed": "each tensor has its last two dimensions stored transposed in memory (same logical values)",
    "leading_swap": "each tensor has its first two dimensions stored swapped in memory (same logical values)",
    "decay_none": "every log-decay input is exactly 0 (no decay: the state accumulates over the whole sequence)",
    "decay_strong": "every log-decay input is between -25 and -5 (the state is almost erased every step)",
    "gate_edges": "every gate input is within 1e-4 of 0 or of 1",
    "gate_typical": "gates drawn as sigmoid(uniform(0,1)), about 0.5 to 0.73, and slow decay log(sigmoid(N(3,1)))",
    "gate_high": "every gate input is between 0.9 and 0.99",
    "gate_mid": "every gate input is between 0.3 and 0.95",
}


def describe_case(name: str, spec: OpSpec) -> str:
    if name in CASE_TEXT:
        return CASE_TEXT[name]
    if "=" in name:
        sym, val = name.split("=")
        return f"a size that is {spec.sizes.get(sym)} in the observed call changed to {val}"
    return name


def describe_call(spec: OpSpec) -> str:
    def fmt(a) -> str:
        if isinstance(a, dict) and "tensor" in a:
            t = a["tensor"]
            shape = [spec.sizes[s["sym"]] if isinstance(s, dict) else s for s in t["shape"]]
            lay = "contiguous" if t.get("contiguous", True) else f"non-contiguous, strides {t['stride']}"
            return f"Tensor(shape={shape}, dtype={t['dtype']}, {lay})"
        if isinstance(a, dict) and "sym" in a:
            return str(spec.sizes[a["sym"]])
        if isinstance(a, dict) and "seq" in a:
            inner = ", ".join(fmt(x) for x in a["seq"])
            return f"({inner},)" if a.get("tuple") and len(a["seq"]) == 1 else (f"({inner})" if a.get("tuple") else f"[{inner}]")
        if isinstance(a, dict) and "dtype" in a:
            return f"torch.{a['dtype']}"
        return repr(a)

    parts = [fmt(a) for a in spec.args] + [f"{k}={fmt(v)}" for k, v in spec.kwargs.items()]
    return f"{spec.op}({', '.join(parts)})"


WRITE_PROMPT = """You are writing a Triton GPU kernel that replaces one PyTorch call inside a real model.

The model makes this call:

    {call}

Write a Python module that defines

    def kernel_function({params}):

with the same arguments as the call above, returning the same result as the PyTorch call. The
heavy computation must happen in one or more @triton.jit kernels; the Python wrapper may only
allocate outputs, compute launch parameters and launch. Do not call the PyTorch op itself or
any equivalent high-level torch function for the computation.

The kernel will be dropped into the model and must return the same values as PyTorch for any
inputs the PyTorch call accepts. It should be fast on an NVIDIA L4 (Ada, sm_89).

Return the complete module in one ```python block.
"""

REPAIR_PROMPT = """You wrote this Triton kernel to replace a PyTorch call inside a real model:

    {call}

```python
{source}
```

A contract checker ran it against the PyTorch reference (evaluated in float64) on generated inputs.
It failed these cases:

{failures}

It passed these cases: {passes}.

Diagnose which assumption in the kernel the failing inputs violate, fix exactly that, and keep the
algorithm and launch configuration otherwise. The kernel must still do its computation in Triton.
Return the complete fixed module in one ```python block.
"""


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    if not blocks:
        raise ValueError("no python block in the response")
    return max(blocks, key=len)


def param_names(spec: OpSpec) -> str:
    names = [f"arg{i}" for i in range(len(spec.args))] + [f"{k}=None" for k in spec.kwargs]
    return ", ".join(names)


def failure_text(spec: OpSpec, verdict: Verdict) -> str:
    lines = []
    for r in verdict.failures:
        what = describe_case(r["name"], spec)
        if r["outcome"] == "wrong":
            how = r.get("reason") or (f"returned wrong values: max error {r['err']:.3g} where PyTorch's own "
                                      f"error is {r['torch_err']:.3g}, on outputs up to {r['scale']:.3g}")
        elif r["outcome"] == "refuse":
            how = f"raised {r['reason']}"
        else:
            how = f"crashed the process ({r['reason']})"
        lines.append(f"- {r['name']}: {what}. The kernel {how}.")
    return "\n".join(lines)


@dataclass
class Attempt:
    kind: str
    source: str
    verdict: Verdict
    seconds: float
    usage: dict | None = None


@dataclass
class OpResult:
    spec: OpSpec
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def final(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts else None

    @property
    def accepted(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].verdict.outcome == "accept"


def solve(spec: OpSpec, model: str, workdir: Path, max_repairs: int = 3, contract: str = "strided",
          device: str = "cuda", emit: Callable[[dict], None] | None = None, **llm_kwargs) -> OpResult:
    emit = emit or (lambda e: None)
    provider = get_model_provider(model)
    workdir.mkdir(parents=True, exist_ok=True)
    result = OpResult(spec)
    call = describe_call(spec)
    prompt = WRITE_PROMPT.format(call=call, params=param_names(spec))
    kind = "write"
    for i in range(max_repairs + 1):
        emit({"event": "llm_start", "op": spec.name, "kind": kind, "attempt": i, "model": model, "call": call})
        t0 = time.time()
        resp = provider.get_response(model, [{"role": "user", "content": prompt}], **llm_kwargs)
        seconds = time.time() - t0
        try:
            source = extract_code(resp.content)
        except ValueError:
            source = "def kernel_function(*a, **k):\n    raise RuntimeError('model returned no code')\n"
        path = workdir / f"{spec.name}_{i}_{kind}.py"
        path.write_text(source)
        emit({"event": "llm_done", "op": spec.name, "kind": kind, "attempt": i, "seconds": seconds,
              "path": str(path), "usage": resp.usage})
        verdict = run_contract(spec, path, contract=contract, device=device,
                               on_case=lambda r: emit({"event": "case", "op": spec.name, "attempt": i,
                                                       "means": describe_case(r["name"], spec), **r}))
        result.attempts.append(Attempt(kind, source, verdict, seconds, resp.usage))
        emit({"event": "verdict", "op": spec.name, "attempt": i, "outcome": verdict.outcome,
              "failures": [f["name"] for f in verdict.failures],
              "failing_axes": sorted({f["axis"] for f in verdict.failures})})
        if verdict.outcome == "accept":
            break
        passes = ", ".join(r["name"] for r in verdict.results if r["outcome"] == "accept") or "none"
        prompt = REPAIR_PROMPT.format(call=call, source=source, failures=failure_text(spec, verdict), passes=passes)
        kind = "repair"
    return result
