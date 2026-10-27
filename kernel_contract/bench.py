"""Run the paper agent over every (model, layer) pair with a fixed budget, one GPU host per worker.

  python -m kernel_contract.bench --hosts hosts.txt --models muse-wonder-1.0,muse-spark-1.3-contributor \\
      --out runs/bench1

hosts.txt holds one remote command prefix per line (e.g. "ssh -p 1234 root@1.2.3.4"). Baselines (eager time
per layer) come from baselines.json, written by `python -m kernel_contract.bench --baselines` on a host.
"""

from __future__ import annotations

import argparse
import json
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

LAYERS = ["b01_gdr", "b02_kda", "b03_eda", "b04_eda_channel", "b05_write_erase", "b06_value_erase", "b07_coupled",
          "b08_read_before", "b09_decay_after", "b10_neg_eig", "b11_add_write", "b12_two_sided"]
USD = {"muse-wonder-1.0": 6.0, "muse-spark-1.3-contributor": 1.0, "muse-spark-1.3": 3.0,
       "meta-models/muse-spark-open-1026": 4.0}


def baselines():
    """On a GPU host: eager forward and forward+backward of every reference, and the expert route if any."""
    import importlib

    from .recur_bench import timeit
    out = {}
    for name in LAYERS:
        m = importlib.import_module(f"kernel_contract.layers.{name}")
        x, xg = m.inputs(B=4, H=8, T=2048), m.inputs(B=4, H=8, T=2048, grad=True)
        s = 128 ** -0.5
        r = {"eager_ms": timeit(lambda: m.reference(*x, s), reps=1, warm=1),
             "eager_fwd_bwd_ms": timeit(lambda: m.reference(*xg, s).sum().backward(), reps=1, warm=1)}
        if m.fla_reduction:
            red = m.fla_reduction(*m.inputs(B=2, H=4, T=300), s).float()
            ref = m.reference(*m.inputs(B=2, H=4, T=300), s)
            r["expert_rel_err"] = float((red - ref).norm() / ref.norm())
            r["expert_ms"] = timeit(lambda: m.fla_reduction(*x, s), reps=20)
            r["expert_fwd_bwd_ms"] = timeit(lambda: m.fla_reduction(*xg, s).float().sum().backward(), reps=10)
        out[name] = r
        print(json.dumps({name: r}), flush=True)
    Path("baselines.json").write_text(json.dumps(out, indent=1))


def worker(host: str, jobs: queue.Queue, out: Path, base: dict, log):
    while True:
        try:
            model, layer = jobs.get_nowait()
        except queue.Empty:
            return
        tag = f"{model.split('/')[-1]}__{layer}"
        cmd = [sys.executable, "-m", "kernel_contract.paper_agent", "--model", model,
               "--layer", f"kernel_contract.layers.{layer}", "--eager-ms", str(round(base[layer]["eager_ms"])),
               "--remote", host, "--out", str(out / tag), "--max-usd", str(USD.get(model, 3.0)),
               "--opt-rounds", "1", "--bwd-opt-rounds", "1"]
        t0 = time.time()
        with open(out / f"{tag}.log", "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        log(json.dumps({"job": tag, "host": host.split()[-1], "rc": rc, "minutes": round((time.time() - t0) / 60, 1)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--hosts")
    ap.add_argument("--models", default="muse-wonder-1.0,muse-spark-1.3-contributor,meta-models/muse-spark-open-1026")
    ap.add_argument("--layers", default=",".join(LAYERS))
    ap.add_argument("--base", default="runs/bench_baselines.json")
    ap.add_argument("--out", default="runs/bench1")
    a = ap.parse_args()
    if a.baselines:
        return baselines()
    hosts = [h.strip() for h in Path(a.hosts).read_text().splitlines() if h.strip()]
    base = json.loads(Path(a.base).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs: queue.Queue = queue.Queue()
    for layer in a.layers.split(","):  # interleave models so every layer gets each model early
        for model in a.models.split(","):
            if not (out / f"{model.split('/')[-1]}__{layer}" / "events.jsonl").exists():
                jobs.put((model, layer))
    lock = threading.Lock()

    def log(line):
        with lock, open(out / "bench.log", "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    threads = [threading.Thread(target=worker, args=(h, jobs, out, base, log)) for h in hosts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
