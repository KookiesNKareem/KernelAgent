"""One decorator on a slow PyTorch function; a Muse model writes a verified Triton kernel for it in the background.

    import muse_kernel

    @muse_kernel.fast
    def my_layer(q, k, v, g, beta, scale):
        ...  # the reference loop, unchanged

Training runs on the function as usual. After a few calls the decorator has seen the real shapes, dtypes and
value ranges, and starts a background build. Once a forward+backward kernel passes the checker, calls whose
shapes, dtypes and values match what was verified go to the kernel; everything else still runs the function.
Builds are cached by the function's source, so editing the equation triggers a new build automatically.

Settings (environment): MUSE_KERNEL_MODEL (default muse-wonder-1.0), MUSE_KERNEL_MAX_USD (default 8),
MUSE_KERNEL_CACHE (default .muse_kernels/ next to the source file; commit it to share kernels), MUSE_KERNEL_REMOTE / MUSE_KERNEL_REMOTE_DIR to check
kernels on another GPU host (default: this machine).
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import torch

from .infer import NORM_TOL, describe_domains, merge_domains, observe, source_hash, standalone_module

__all__ = ["fast"]


def cache_root(fn) -> Path:
    """Next to the function's source by default, so a verified kernel can be committed and reused anywhere."""
    if "MUSE_KERNEL_CACHE" in os.environ:
        return Path(os.environ["MUSE_KERNEL_CACHE"])
    src = inspect.getsourcefile(fn)
    return Path(src).resolve().parent / ".muse_kernels" if src else Path.home() / ".cache" / "muse_kernel"


class Fast:
    def __init__(self, fn, observe_calls: int):
        functools.update_wrapper(self, fn)
        self.fn, self.n_observe = fn, observe_calls
        self.hash = source_hash(fn)
        self.job = cache_root(fn) / f"{fn.__name__}_{self.hash}"
        self.seen: list[dict] = []
        self.domains: dict[int, str | None] = {}
        self.kernel = None
        self.verified: list[tuple] = []
        self.stats = Counter()
        self.lock = threading.Lock()
        self._last_check = 0.0
        self._load_if_ready()

    # ---- the guard -------------------------------------------------------------------------------
    def _signature(self, args) -> tuple:
        return tuple((tuple(a.shape), str(a.dtype)) if isinstance(a, torch.Tensor) else a for a in args)

    def _values_ok(self, args) -> bool:
        checks = []
        for i, d in self.domains.items():
            if not d:
                continue
            x = args[i].detach()
            if d == "l2row":
                checks.append((x.float().norm(dim=-1) - 1).abs().max() < NORM_TOL)
            elif d == "unit":
                checks.append((x.min() >= 0) & (x.max() <= 1))
            elif d == "logdecay":
                checks.append(x.max() <= 0)
            elif d == "positive":
                checks.append(x.min() > 0)
        return bool(torch.stack(checks).all()) if checks else True

    def __call__(self, *args, **kwargs):
        self._poll()
        if self.kernel is not None and not kwargs and self._signature(args) in self.verified:
            if self._values_ok(args):
                try:
                    out = self.kernel(*args)
                    self.stats["kernel"] += 1
                    return out
                except Exception as e:  # never let a kernel failure stop training
                    self.kernel = None
                    self.stats[f"disabled: {type(e).__name__}"] += 1
            else:
                self.stats["fallback: values outside the verified range"] += 1
        elif self.kernel is not None:
            self.stats["fallback: shape or dtype not verified"] += 1
        if self.kernel is None and not kwargs:
            self._observe(args)
        self.stats["reference"] += 1
        return self.fn(*args, **kwargs)

    # ---- observe and build ------------------------------------------------------------------------
    def _observe(self, args):
        if self.job.joinpath("status.json").exists() or len(self.seen) >= self.n_observe:
            return
        obs = observe(args)
        self.domains = obs["domains"] if not self.seen else merge_domains(self.domains, obs["domains"])
        self.seen.append({"args": tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in args), **obs})
        if len(self.seen) == self.n_observe:
            self._start_build()

    def _start_build(self):
        from kernel_contract.agent import describe_call
        from kernel_contract.spec import capture, with_domains
        self.job.mkdir(parents=True, exist_ok=True)
        (self.job / "reference.py").write_text(standalone_module(self.fn))
        op = f"muse_jobs.j_{self.hash}.{self.fn.__name__}"
        names = list(inspect.signature(self.fn).parameters)
        specs, sigs = [], []
        for s in self.seen:
            sig = self._signature(s["args"])
            if sig in sigs:
                continue
            sigs.append(sig)
            spec = capture(op, tuple(s["args"]), name=self.fn.__name__)
            spec.flags.update(any_float_output=True, normwise_rtol=2e-2)
            specs.append(with_domains(spec, {i: d for i, d in self.domains.items() if d}))
        (self.job / "spec.json").write_text(json.dumps(specs[0].to_json()))
        (self.job / "extra_specs.json").write_text(json.dumps([s.to_json() for s in specs[1:]]))
        job = {"hash": self.hash, "name": self.fn.__name__, "source": inspect.getsource(self.fn),
               "signature": ", ".join(names), "call": describe_call(specs[0]), "sizes": specs[0].sizes,
               "domains": describe_domains(self.domains, names), "verified_signatures": [repr(s) for s in sigs],
               "model": os.environ.get("MUSE_KERNEL_MODEL", "muse-wonder-1.0"),
               "max_usd": float(os.environ.get("MUSE_KERNEL_MAX_USD", 8))}
        (self.job / "job.json").write_text(json.dumps(job, indent=1))
        (self.job / "status.json").write_text(json.dumps({"state": "queued", "updated": time.time()}))
        with open(self.job / "builder.log", "w") as log:
            subprocess.Popen([sys.executable, "-m", "muse_kernel.build", str(self.job)], stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True, cwd=Path(__file__).resolve().parents[1])
        print(f"[muse_kernel] {self.fn.__name__}: building a kernel in the background ({self.job})", flush=True)

    def _poll(self):
        if self.kernel is not None or time.time() - self._last_check < 5:
            return
        self._last_check = time.time()
        self._load_if_ready()

    def _load_if_ready(self):
        st = self.job / "status.json"
        if not st.exists():
            return
        s = json.loads(st.read_text())
        if s.get("state") != "ready":
            return
        job = json.loads((self.job / "job.json").read_text())
        spec_mod = importlib.util.spec_from_file_location(f"muse_kernel_{self.hash}", s["kernel"])
        mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(mod)
        self.kernel = mod.kernel_function
        self.verified = [eval(x, {"torch": torch}) for x in job["verified_signatures"]]
        self.domains = {int(i): d for i, d in json.loads((self.job / "domains.json").read_text()).items()} \
            if (self.job / "domains.json").exists() else self._domains_from_spec()
        print(f"[muse_kernel] {self.fn.__name__}: verified kernel in use "
              f"({s.get('verified_cases')}/{s.get('cases')} cases, {s.get('fwd_bwd_ms', 0):.1f} ms fwd+bwd "
              f"vs {s.get('reference_ms', 0):.0f} ms reference forward, ${s.get('usd', 0):.2f})", flush=True)

    def _domains_from_spec(self) -> dict[int, str | None]:
        spec = json.loads((self.job / "spec.json").read_text())
        return {i: a["tensor"].get("domain") for i, a in enumerate(spec["args"]) if isinstance(a, dict) and "tensor" in a}

    def status(self) -> dict:
        st = self.job / "status.json"
        return {**(json.loads(st.read_text()) if st.exists() else {"state": "observing"}), "calls": dict(self.stats)}


def fast(fn=None, *, observe_calls: int = 2):
    """Decorate a slow PyTorch function; see the module docstring."""
    if fn is None:
        return lambda f: Fast(f, observe_calls)
    return Fast(fn, observe_calls)
