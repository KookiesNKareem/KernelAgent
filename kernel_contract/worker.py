"""Subprocess side: run cases against one kernel, one JSON line per case.

  python -m kernel_contract.worker SPEC_JSON KERNEL_PY CASES_JSON OUT_JSONL START DEVICE

A wrong kernel can poison the CUDA context, so on a device error the worker writes a crash
line and exits; the parent restarts it at the next case.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback

import torch

from .cases import VALUE_TRANSFORMS
from .spec import OpSpec, materialize

TOL = {torch.bfloat16: (1e-2, 1e-2), torch.float16: (1e-3, 1e-3), torch.float32: (1e-4, 1e-4),
       torch.float64: (1e-7, 1e-7)}
DEVICE_ERRORS = ("CUDA error", "illegal memory access", "device-side assert", "misaligned address",
                 "unspecified launch failure")


def load_kernel(path: str):
    spec = importlib.util.spec_from_file_location("candidate_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernel_function


def first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"kernel returned {type(out).__name__}, expected a tensor")


def to_f64(x):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.double()
    if isinstance(x, (list, tuple)):
        return type(x)(to_f64(v) for v in x)
    return x


def compare(out: torch.Tensor, ref: torch.Tensor, truth: torch.Tensor, any_float: bool = False) -> dict:
    if tuple(out.shape) != tuple(ref.shape):
        return {"outcome": "wrong", "reason": f"shape {tuple(out.shape)} != {tuple(ref.shape)}"}
    if out.dtype != ref.dtype and not (any_float and out.is_floating_point() and ref.is_floating_point()):
        return {"outcome": "wrong", "reason": f"dtype {out.dtype} != {ref.dtype}"}
    tol_dtype = out.dtype if TOL.get(out.dtype, (0,))[0] > TOL.get(ref.dtype, (0,))[0] else ref.dtype
    o, r, t = out.double(), ref.double(), truth.double().to(out.device)
    fin_t = torch.isfinite(t)
    if not torch.equal(torch.isfinite(o), fin_t):
        bad = int((torch.isfinite(o) != fin_t).sum())
        return {"outcome": "wrong", "reason": f"{bad} non-finite mismatches"}
    if not fin_t.any():
        return {"outcome": "accept", "err": 0.0, "torch_err": 0.0, "scale": 0.0}
    o, r, t = o[fin_t], r[fin_t], t[fin_t]
    atol, rtol = TOL.get(tol_dtype, (1e-2, 1e-2))
    scale = float(t.abs().max())
    err = float((o - t).abs().max())
    torch_err = float((r - t).abs().max())
    within = bool(((o - t).abs() <= atol * min(1.0, scale) + rtol * t.abs()).all())
    ok = within or (torch_err > 0 and err <= 2 * torch_err)
    return {"outcome": "accept" if ok else "wrong", "err": err, "torch_err": torch_err, "scale": scale}


def run_case(spec: OpSpec, kernel, case: dict, device: str) -> dict:
    sizes = {**spec.sizes, **case.get("sizes", {})}
    vf = VALUE_TRANSFORMS.get(case.get("transform", ""))

    def fresh():
        # rebuilt from the seed for every call: a clone would not keep a gapped view's strides
        return materialize(spec, sizes, case["seed"], device, layout=case.get("layout"), value_fn=vf,
                           stress=case.get("stress"))

    args, kwargs = fresh()
    fn = spec.reference()
    try:
        ref = first_tensor(fn(*args, **kwargs))
    except Exception as e:
        return {"outcome": "skip", "reason": f"reference raised {type(e).__name__}: {e}"[:300]}
    args, kwargs = fresh()
    try:
        truth = first_tensor(fn(*to_f64(args), **to_f64(kwargs)))
    except Exception:
        truth = first_tensor(fn(*[a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                                  for a in args], **kwargs))
    try:
        kargs, kkwargs = fresh()
        out = first_tensor(kernel(*kargs, **kkwargs))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if any(s in msg for s in DEVICE_ERRORS):
            return {"outcome": "crash", "reason": msg[:300]}
        return {"outcome": "refuse", "reason": msg[:300]}
    return compare(out, ref, truth, any_float=spec.flags.get("any_float_output", False))


def main():
    spec_path, kernel_path, cases_path, out_path, start, device = sys.argv[1:7]
    spec = OpSpec.from_json(json.load(open(spec_path)))
    cases = json.load(open(cases_path))
    try:
        kernel = load_kernel(kernel_path)
    except Exception:
        with open(out_path, "a") as f:
            for c in cases[int(start):]:
                f.write(json.dumps({"name": c["name"], "axis": c["axis"], "outcome": "refuse",
                                    "reason": "kernel failed to import: " + traceback.format_exc()[-300:]}) + "\n")
        return
    for c in cases[int(start):]:
        r = run_case(spec, kernel, c, device)
        with open(out_path, "a") as f:
            f.write(json.dumps({"name": c["name"], "axis": c["axis"], **r}) + "\n")
        if r["outcome"] == "crash":
            sys.exit(3)


if __name__ == "__main__":
    main()
