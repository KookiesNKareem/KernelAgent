"""Model mode: record the op calls a model makes, rank them by time, and run the model with
verified kernels installed behind a guard that sends any call outside the verified contract back
to PyTorch."""

from __future__ import annotations

import importlib.util
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode

from .spec import OpSpec, capture, materialize

SKIP = {"size", "dim", "numel", "view", "reshape", "permute", "transpose", "t", "contiguous", "to",
        "__getitem__", "__setitem__", "unsqueeze", "squeeze", "expand", "flatten", "unflatten",
        "chunk", "split", "narrow", "select", "detach", "clone", "float", "half", "bfloat16", "type_as",
        "item", "tolist", "__len__", "__iter__", "__format__", "__repr__", "is_contiguous", "stride",
        "storage_offset", "data_ptr", "dtype", "device", "shape", "requires_grad_", "__get__",
        "new_empty", "new_zeros", "new_ones", "empty_like", "zeros_like", "ones_like", "arange", "full",
        "zeros", "ones", "empty", "cat", "stack", "copy_", "fill_", "masked_fill_", "__bool__", "eq", "ne"}


def op_path(func) -> str | None:
    name = getattr(func, "__name__", None)
    if name is None:
        return None
    for prefix, ns in (("torch.nn.functional", torch.nn.functional), ("torch", torch)):
        if getattr(ns, name, None) is func:
            return f"{prefix}.{name}"
    qual = getattr(func, "__qualname__", "")
    if qual.startswith(("TensorBase.", "Tensor.")) and hasattr(torch.Tensor, name):
        return f"torch.Tensor.{name}"
    mod = getattr(func, "__module__", None)
    return f"{mod}.{name}" if mod else None


class _Recorder(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.calls: Counter = Counter()
        self.specs: dict[str, OpSpec] = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = getattr(func, "__name__", "")
        path = op_path(func)
        if path and name not in SKIP and isinstance(out, torch.Tensor) and out.is_floating_point():
            try:
                spec = capture(path, tuple(args), kwargs)
            except TypeError:
                return out
            key = json.dumps(spec.to_json(), sort_keys=True)
            self.calls[key] += 1
            self.specs.setdefault(key, spec)
        return out


@dataclass
class OpSite:
    spec: OpSpec
    calls: int
    ms_per_call: float

    @property
    def ms_total(self) -> float:
        return self.calls * self.ms_per_call


def _time(fn, reps: int = 20) -> float:
    cuda = torch.cuda.is_available()
    for _ in range(3):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / reps


@torch.no_grad()
def trace(model: torch.nn.Module, inputs: tuple, device: str = "cuda") -> list[OpSite]:
    """Every distinct op call in one forward pass, timed by replay, hottest first."""
    rec = _Recorder()
    with rec:
        model(*inputs)
    sites = []
    for key, n in rec.calls.items():
        spec = rec.specs[key]
        args, kwargs = materialize(spec, spec.sizes, 0, device)
        fn = spec.reference()
        try:
            ms = _time(lambda: fn(*args, **kwargs))
        except Exception:
            continue
        spec.name = f"{spec.op.rsplit('.', 1)[-1]}_{len(sites)}"
        sites.append(OpSite(spec, n, ms))
    return sorted(sites, key=lambda s: -s.ms_total)


def load_kernel(path: Path):
    mspec = importlib.util.spec_from_file_location(f"k_{path.stem}", path)
    mod = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(mod)
    return mod.kernel_function


@dataclass
class Installed:
    spec: OpSpec
    kernel: object
    strided_ok: bool


@dataclass
class GuardStats:
    routed: Counter = field(default_factory=Counter)
    fallback: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)


def _signature(spec_json: dict) -> tuple:
    """What must match exactly for a call to be inside a verified contract: the op, every tensor's
    dtype and rank, and every non-size argument. Sizes may vary (the fuzzer perturbed them)."""
    def sig(a):
        if isinstance(a, dict) and "tensor" in a:
            return ("T", a["tensor"]["dtype"], len(a["tensor"]["shape"]))
        if isinstance(a, dict) and "sym" in a:
            return ("size",)
        if isinstance(a, dict) and "seq" in a:
            return ("seq",) + tuple(sig(x) for x in a["seq"])
        if isinstance(a, dict) and "dtype" in a:
            return ("dtype", a["dtype"])
        return ("lit", repr(a))
    return (spec_json["op"], tuple(sig(a) for a in spec_json["args"]),
            tuple(sorted((k, sig(v)) for k, v in spec_json.get("kwargs", {}).items())))


class GuardedModel(TorchFunctionMode):
    """Route in-contract calls to verified kernels; everything else runs in PyTorch."""

    def __init__(self, installed: list[Installed]):
        super().__init__()
        self.table: dict[tuple, Installed] = {}
        for inst in installed:
            self.table[_signature(inst.spec.to_json())] = inst
        self.ops = {inst.spec.op for inst in installed}
        self.stats = GuardStats()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        path = op_path(func)
        if path in self.ops:
            try:
                spec = capture(path, tuple(args), kwargs)
            except TypeError:
                spec = None
            inst = self.table.get(_signature(spec.to_json())) if spec else None
            if inst is None:
                self.stats.fallback[path] += 1
                self.stats.reasons["signature outside contract"] += 1
            elif not inst.strided_ok and not all(
                    a.is_contiguous() for a in args if isinstance(a, torch.Tensor)):
                self.stats.fallback[path] += 1
                self.stats.reasons["non-contiguous input, kernel verified dense only"] += 1
            else:
                self.stats.routed[path] += 1
                return inst.kernel(*args, **kwargs)
        return func(*args, **kwargs)


@torch.no_grad()
def compare_model(model: torch.nn.Module, inputs: tuple, installed: list[Installed], reps: int = 20) -> dict:
    ref = model(*inputs)
    guard = GuardedModel(installed)
    with guard:
        out = model(*inputs)
    ref_t = ref if isinstance(ref, torch.Tensor) else ref[0]
    out_t = out if isinstance(out, torch.Tensor) else out[0]
    diff = float((ref_t.float() - out_t.float()).abs().max())
    scale = float(ref_t.float().abs().max())
    eager_ms = _time(lambda: model(*inputs), reps)

    def guarded():
        with GuardedModel(installed):
            model(*inputs)
    guarded_ms = _time(guarded, reps)
    return {"max_abs_diff": diff, "output_scale": scale, "eager_ms": eager_ms, "guarded_ms": guarded_ms,
            "speedup": eager_ms / guarded_ms, "routed": dict(guard.stats.routed),
            "fallback": dict(guard.stats.fallback), "fallback_reasons": dict(guard.stats.reasons)}
