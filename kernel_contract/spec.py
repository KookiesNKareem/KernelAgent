"""An op call as data: the reference callable and the arguments a model actually passed it.

A spec is JSON so it can cross into the subprocess that runs untrusted kernels. Tensor
arguments are described, never stored; integers that equal a tensor dimension are bound to
the same size symbol, so perturbing a symbol keeps shapes and size arguments consistent.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from typing import Any

import torch

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
          "float64": torch.float64, "int64": torch.int64, "int32": torch.int32, "bool": torch.bool}


def dtype_name(dt: torch.dtype) -> str:
    return next(k for k, v in DTYPES.items() if v == dt)


def resolve(path: str):
    parts = path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(path)


@dataclass
class OpSpec:
    op: str
    args: list[Any]
    kwargs: dict[str, Any] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)
    name: str = ""
    flags: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"op": self.op, "args": self.args, "kwargs": self.kwargs, "sizes": self.sizes, "name": self.name,
                "flags": self.flags}

    @classmethod
    def from_json(cls, d: dict) -> "OpSpec":
        return cls(d["op"], d["args"], d.get("kwargs", {}), d.get("sizes", {}), d.get("name", ""), d.get("flags", {}))

    def reference(self):
        return resolve(self.op)


def _tensor_desc(t: torch.Tensor) -> dict:
    return {"tensor": {"shape": list(t.shape), "obs_shape": list(t.shape), "dtype": dtype_name(t.dtype),
                       "stride": list(t.stride()), "contiguous": t.is_contiguous()}}


def _flat(xs):
    for x in xs:
        if isinstance(x, (list, tuple)):
            yield from _flat(x)
        else:
            yield x


def capture(op: str, args: tuple, kwargs: dict | None = None, name: str = "") -> OpSpec:
    """Describe a concrete call. Dimensions and integer arguments with equal values share a symbol."""
    kwargs = kwargs or {}
    sizes: dict[str, int] = {}
    by_value: dict[int, str] = {}
    tensor_dims = {s for a in _flat(list(args) + list(kwargs.values())) if isinstance(a, torch.Tensor) for s in a.shape}

    def sym(v: int) -> Any:
        if v < 3:
            return v
        if v not in by_value:
            by_value[v] = f"s{len(by_value)}"
            sizes[by_value[v]] = v
        return {"sym": by_value[v]}

    def enc(a: Any) -> Any:
        if isinstance(a, torch.Tensor):
            d = _tensor_desc(a)
            d["tensor"]["shape"] = [sym(s) for s in a.shape]
            return d
        if isinstance(a, bool) or a is None or isinstance(a, (float, str)):
            return a
        if isinstance(a, int):
            # an integer that matches no tensor dimension is configuration (dim, groups), not a size
            return sym(a) if a in tensor_dims else a
        if isinstance(a, (list, tuple)):
            return {"seq": [enc(x) for x in a], "tuple": isinstance(a, tuple)}
        if isinstance(a, torch.dtype):
            return {"dtype": dtype_name(a)}
        raise TypeError(f"cannot describe argument of type {type(a).__name__}")

    encoded_args = [enc(a) for a in args]
    encoded_kwargs = {k: enc(v) for k, v in kwargs.items()}
    return OpSpec(op, encoded_args, encoded_kwargs, sizes, name or op.rsplit(".", 1)[-1])


DOMAINS = ("logdecay", "unit", "l2row", "positive")


def draw_domain(dom: str, shape: list[int], g: torch.Generator, stress: str | None) -> torch.Tensor:
    """Inputs a layer defines only on a constrained set. Value transforms never touch these;
    stress cases push them to the edges of their domain instead."""
    u = torch.rand(shape, generator=g, dtype=torch.float32)
    if dom == "logdecay":  # log of a decay factor in (0, 1]
        if stress == "decay_none":
            return torch.zeros(shape)
        if stress == "decay_strong":
            return -20 * u - 5
        if stress == "gate_typical":  # slow decay, as in the timing and training inputs
            return torch.nn.functional.logsigmoid(torch.randn(shape, generator=g) + 3)
        return torch.nn.functional.logsigmoid(torch.randn(shape, generator=g))
    if dom == "unit":  # a gate in (0, 1)
        if stress == "gate_edges":
            return torch.where(u > 0.5, 1 - 1e-4 * u, 1e-4 * u)
        # kernels may branch on the gate range, so every range a real input can occupy gets its own case
        if stress == "gate_typical":
            return u.sigmoid()
        if stress == "gate_high":
            return 0.9 + 0.09 * u
        if stress == "gate_mid":
            return 0.3 + 0.65 * u
        return u.clamp(1e-3, 1 - 1e-3)
    if dom == "l2row":  # rows of unit L2 norm along the last dim
        return torch.nn.functional.normalize(torch.randn(shape, generator=g), dim=-1)
    if dom == "positive":
        return u * 2 + 1e-3
    raise ValueError(f"unknown domain {dom}")


def with_domains(spec: "OpSpec", domains: dict[int, str]) -> "OpSpec":
    """Mark positional tensor arguments with value domains."""
    for i, dom in domains.items():
        assert dom in DOMAINS, dom
        spec.args[i]["tensor"]["domain"] = dom
    return spec


def materialize(spec: OpSpec, sizes: dict[str, int], seed: int, device: str,
                layout: str | None = None, value_fn=None, stress: str | None = None) -> tuple[list, dict]:
    """Build concrete arguments. `layout` overrides the observed strides of every tensor of rank >= 1."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def dim(x: Any) -> int:
        return sizes[x["sym"]] if isinstance(x, dict) else x

    def dec(a: Any) -> Any:
        if isinstance(a, dict) and "tensor" in a:
            t = a["tensor"]
            shape = [dim(s) for s in t["shape"]]
            dt = DTYPES[t["dtype"]]
            if dt.is_floating_point:
                dom = t.get("domain")
                if dom:
                    x = draw_domain(dom, shape, g, stress)
                else:
                    x = torch.rand(shape, generator=g, dtype=torch.float32) * 2 - 1
                    if value_fn is not None:
                        x = value_fn(x)
                x = x.to(dt)
            elif dt == torch.bool:
                x = torch.rand(shape, generator=g) > 0.5
            else:
                x = torch.randint(0, max(2, min(shape) if shape else 2), shape, generator=g, dtype=dt)
            x = x.to(device)
            mode = layout if layout is not None else ("observed" if not t.get("contiguous", True) else None)
            exact = t.get("stride") if shape == t.get("obs_shape") else None
            return apply_layout(x, mode, t.get("stride"), exact)
        if isinstance(a, dict) and "sym" in a:
            return sizes[a["sym"]]
        if isinstance(a, dict) and "seq" in a:
            items = [dec(x) for x in a["seq"]]
            return tuple(items) if a.get("tuple") else items
        if isinstance(a, dict) and "dtype" in a:
            return DTYPES[a["dtype"]]
        return a

    return [dec(a) for a in spec.args], {k: dec(v) for k, v in spec.kwargs.items()}


def apply_layout(x: torch.Tensor, mode: str | None, observed_stride: list[int] | None = None,
                 exact_stride: list[int] | None = None) -> torch.Tensor:
    """Same logical values, different memory layout."""
    if mode is None or x.dim() == 0:
        return x
    if mode == "contiguous":
        return x.contiguous()
    if mode == "observed":
        if observed_stride is None or list(x.stride()) == observed_stride:
            return x
        if exact_stride is not None and all(st >= 0 for st in exact_stride):
            # the model's exact strides, gaps included, inside a buffer just large enough
            span = sum((n - 1) * st for n, st in zip(x.shape, exact_stride)) + 1
            buf = torch.zeros(span, dtype=x.dtype, device=x.device)
            view = torch.as_strided(buf, x.shape, exact_stride)
            view.copy_(x)
            return view
        order = sorted(range(x.dim()), key=lambda i: -observed_stride[i])
        y = x.permute(order).contiguous()
        inv = [order.index(i) for i in range(x.dim())]
        return y.permute(inv)
    if mode in ("strided2", "strided3_off1"):
        k, off = (2, 0) if mode == "strided2" else (3, 1)
        buf = torch.zeros((*x.shape[:-1], x.shape[-1] * k + off), dtype=x.dtype, device=x.device)
        view = buf[..., off::k][..., : x.shape[-1]]
        view.copy_(x)
        return view
    if mode == "transposed":
        if x.dim() < 2:
            return x
        y = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        return y
    if mode == "leading_swap":
        if x.dim() < 3:
            return apply_layout(x, "transposed")
        return x.transpose(0, 1).contiguous().transpose(0, 1)
    raise ValueError(f"unknown layout {mode}")


def numel(spec: OpSpec, sizes: dict[str, int]) -> int:
    total = 0
    for a in spec.args:
        if isinstance(a, dict) and "tensor" in a:
            total += math.prod(sizes[s["sym"]] if isinstance(s, dict) else s for s in a["tensor"]["shape"])
    return total
