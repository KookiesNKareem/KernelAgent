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
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]


def standalone_module(fn) -> str:
    """The function's own source, without decorators, plus its module's top-level imports."""
    mod = inspect.getmodule(fn)
    imports = []
    if mod is not None:
        try:
            tree = ast.parse(inspect.getsource(mod))
            imports = [ast.get_source_segment(inspect.getsource(mod), n) for n in tree.body
                       if isinstance(n, (ast.Import, ast.ImportFrom)) and "muse_kernel" not in ast.dump(n)]
        except (OSError, TypeError):
            pass
    src = textwrap.dedent(inspect.getsource(fn))
    fn_tree = ast.parse(src).body[0]
    lines = src.splitlines()
    body = "\n".join(lines[fn_tree.lineno - 1:] if not fn_tree.decorator_list else lines[fn_tree.decorator_list[-1].end_lineno:])
    head = "\n".join(i for i in imports if i and "__future__" not in i)
    return f"from __future__ import annotations\n\nimport torch\n{head}\n\n\n{body}\n"


def describe_domains(domains: dict[int, str | None], names: list[str]) -> list[str]:
    text = {"l2row": "has unit-norm rows (L2 over the last dimension)", "unit": "lies in [0, 1]",
            "logdecay": "is <= 0 (a log decay)", "positive": "is > 0"}
    return [f"{names[i]} {text[d]}" for i, d in sorted(domains.items()) if d]
