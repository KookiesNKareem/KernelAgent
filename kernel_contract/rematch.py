"""Same-GPU, back-to-back timing of candidate kernels against FLA and torch.compile, repeated.

  python -m kernel_contract.rematch --layer gdr --fwd A.py --fwdbwd B.py [--compile]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from . import recur_bench, recur_eda
from .model import load_kernel
from .recur_bench import timeit

SCALE = 128 ** -0.5


def fla_gdr(q, k, v, g, beta, scale):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    return chunk_gated_delta_rule(*(x.transpose(1, 2) for x in (q, k, v, g, beta)), scale=scale,
                                  use_qk_l2norm_in_kernel=False)[0].transpose(1, 2)


LAYERS = {"gdr": (recur_bench.inputs, recur_bench.reference, fla_gdr),
          "eda": (recur_eda.inputs, recur_eda.reference, recur_eda.fla_reduction)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="gdr")
    ap.add_argument("--fwd")
    ap.add_argument("--fwdbwd")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    inputs, reference, expert = LAYERS[a.layer]
    x = inputs(B=4, H=8, T=2048)
    xg = inputs(B=4, H=8, T=2048, grad=True)
    fwd = {"expert": lambda: expert(*x, SCALE)}
    fb = {"expert": lambda: expert(*xg, SCALE).float().sum().backward()}
    if a.fwd:
        kf = load_kernel(Path(a.fwd))
        fwd["candidate"] = lambda: kf(*x, SCALE)
    if a.fwdbwd:
        kb = load_kernel(Path(a.fwdbwd))
        fwd["candidate_autograd_fwd"] = lambda: kb(*x, SCALE)
        fb["candidate"] = lambda: kb(*xg, SCALE).float().sum().backward()
    runs = {f"fwd/{n}": [] for n in fwd} | {f"fwdbwd/{n}": [] for n in fb}
    for _ in range(a.rounds):
        for n, f in fwd.items():
            runs[f"fwd/{n}"].append(timeit(f, reps=20, warm=3))
        for n, f in fb.items():
            runs[f"fwdbwd/{n}"].append(timeit(f, reps=10, warm=2))
    out = {k: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)} for k, v in runs.items()}
    out["gpu"] = torch.cuda.get_device_name()
    print(json.dumps(out), flush=True)
    if a.compile:
        c = torch.compile(reference)
        t0 = time.perf_counter()
        c(*x, SCALE)
        out["compile_fwd_s"] = time.perf_counter() - t0
        out["compiled_fwd_ms"] = timeit(lambda: c(*x, SCALE), reps=3, warm=1)
        print(json.dumps(out), flush=True)
        t0 = time.perf_counter()
        c(*xg, SCALE).sum().backward()
        out["compile_fwdbwd_s"] = time.perf_counter() - t0
        out["compiled_fwdbwd_ms"] = timeit(lambda: c(*xg, SCALE).sum().backward(), reps=2, warm=1)
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
