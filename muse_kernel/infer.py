"""Turn a few real calls of a PyTorch function into a contract: shapes, dtypes and value domains, observed
rather than declared. Also extract the function into a standalone module the GPU host can import."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap

import torch

# a domain is claimed only if every observed value satisfies it; the runtime guard re-checks it per call
NORM_TOL = 1e-2


def tensor_domain(x: torch.Tensor) -> str | None:
    if not x.is_floating_point() or x.numel() == 0:
        return None
    x = x.detach().float()
    lo, hi = float(x.min()), float(x.max())
    if x.dim() >= 1 and x.shape[-1] > 1 and float((x.norm(dim=-1) - 1).abs().max()) < NORM_TOL:
        return "l2row"
    if 0 <= lo and hi <= 1:
        return "unit"
    if hi <= 0:
        return "logdecay"
    if lo > 0:
        return "positive"
    return None


def merge_domains(a: dict[int, str | None], b: dict[int, str | None]) -> dict[int, str | None]:
    """A domain survives only if every observed call agrees on it."""
    return {i: (a.get(i) if a.get(i) == b.get(i) else None) for i in set(a) | set(b)}


def observe(args: tuple) -> dict:
    return {"domains": {i: tensor_domain(a) for i, a in enumerate(args) if isinstance(a, torch.Tensor)},
            "shapes": [list(a.shape) if isinstance(a, torch.Tensor) else None for a in args],
            "dtypes": [str(a.dtype) if isinstance(a, torch.Tensor) else None for a in args],
            "scalars": [a if isinstance(a, (int, float, bool)) and not isinstance(a, torch.Tensor) else None
                        for a in args]}


def source_hash(fn) -> str:
    return hashlib.sha256(standalone_module(fn).encode()).hexdigest()[:24]


def standalone_module(fn) -> str:
    """Capture referenced helpers and immutable globals, or reject an unsupported closure explicitly."""
    import json
    import types
    chunks, imports, seen = [], set(), set()

    def visit(f, name):
        if name in seen:
            return
        seen.add(name)
        cv = inspect.getclosurevars(f)
        for key, value in {**cv.globals, **cv.nonlocals}.items():
            if isinstance(value, types.ModuleType):
                imports.add(f"import {value.__name__} as {key}")
            elif inspect.isfunction(value) and value.__module__ == f.__module__:
                visit(value, key)
            elif inspect.isfunction(value) or inspect.isclass(value) or inspect.isbuiltin(value):
                if '<locals>' in value.__qualname__:
                    raise ValueError(f"Cannot capture local dependency {key}")
                imports.add(f"from {value.__module__} import {value.__name__} as {key}")
            elif isinstance(value, (str, int, float, bool, tuple, list, dict, type(None))):
                json.dumps(value, allow_nan=False)
                chunks.append(f"{key} = {value!r}")
            else:
                raise ValueError(f"Unsupported captured dependency: {key} ({type(value).__name__})")
        node = ast.parse(textwrap.dedent(inspect.getsource(f))).body[0]
        node.decorator_list = []
        node.name = name
        chunks.append(ast.unparse(node))
    visit(fn, fn.__name__)
    return "from __future__ import annotations\n" + "\n".join(sorted(imports)) + "\n\n" + "\n\n".join(chunks) + "\n"


def describe_domains(domains: dict[int, str | None], names: list[str]) -> list[str]:
    text = {"l2row": "has unit-norm rows (L2 over the last dimension)", "unit": "lies in [0, 1]",
            "logdecay": "is <= 0 (a log decay)", "positive": "is > 0"}
    return [f"{names[i]} {text[d]}" for i, d in sorted(domains.items()) if d]
