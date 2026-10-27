"""Background build for one @fast function: stage it on the GPU host, have a Muse model write and repair a
forward+backward kernel against the checker, and record the verified result in status.json.

  python -m muse_kernel.build JOB_DIR
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def status(job: Path, **kw):
    cur = json.loads((job / "status.json").read_text()) if (job / "status.json").exists() else {}
    cur.update(kw, updated=time.time())
    tmp = job / "status.json.tmp"
    tmp.write_text(json.dumps(cur, indent=1))
    tmp.replace(job / "status.json")


def remote_prefix() -> list[str]:
    return shlex.split(os.environ.get("MUSE_KERNEL_REMOTE", "bash -c"))


def remote_dir() -> str:
    return os.environ.get("MUSE_KERNEL_REMOTE_DIR", str(ROOT))


def run_remote(cmd: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(remote_prefix() + [cmd], input=stdin, capture_output=True)


def stage(job: Path, J: dict) -> str:
    """Put the function's module and the contract where the checker imports them; return the remote job dir."""
    rd, h = remote_dir(), J["hash"]
    rjob = f"{rd}/muse_jobs/{h}"
    run_remote(f"mkdir -p {rjob} && touch {rd}/muse_jobs/__init__.py")
    for local, dest in ((job / "reference.py", f"{rd}/muse_jobs/j_{h}.py"), (job / "spec.json", f"{rjob}/spec.json"),
                        (job / "extra_specs.json", f"{rjob}/extra_specs.json")):
        run_remote(f"cat > {dest}", local.read_bytes())
    return rjob


def main(job: str):
    job = Path(job)
    J = json.loads((job / "job.json").read_text())
    status(job, state="building", started=time.time())
    try:
        rjob = stage(job, J)
        p = run_remote(f"cd {remote_dir()} && python -m muse_kernel.remote_eval --reference {rjob}")
        J["eager_ms"] = float(next(x for x in p.stdout.decode().splitlines() if x.startswith("REFERENCE_MS")).split()[1])
        (job / "job.json").write_text(json.dumps(J, indent=1))
        status(job, reference_ms=J["eager_ms"])
        cmd = [sys.executable, "-m", "kernel_contract.paper_agent", "--job", str(job), "--remote-job", rjob,
               "--model", J["model"], "--max-usd", str(J["max_usd"]), "--out", str(job / "run"),
               "--remote", " ".join(shlex.quote(x) for x in remote_prefix()), "--remote-dir", remote_dir(),
               "--opt-rounds", "1", "--bwd-opt-rounds", "1"]
        with open(job / "build.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        events = [json.loads(x) for x in (job / "run" / "events.jsonl").read_text().splitlines()]
        done = next((e for e in reversed(events) if e["event"] == "done"), None)
        usd = sum(e.get("usd", 0) for e in events if e["event"] == "llm_done")
        if done and done.get("ok") and done.get("backward"):
            kernel = job / "kernel.py"
            kernel.write_text(Path(done["backward"]).read_text())
            last = next(e for e in reversed(events) if e["event"] == "eval" and e["path"] == done["backward"])
            status(job, state="ready", kernel=str(kernel), fwd_bwd_ms=done.get("fwd_bwd_ms"), usd=usd,
                   verified_cases=sum(r["outcome"] == "accept" for r in last["results"]), cases=len(last["results"]))
        else:
            status(job, state="failed", usd=usd, reason="no kernel passed the checker within the budget")
    except Exception as e:  # the training run must never be affected by a failed build
        status(job, state="failed", reason=f"{type(e).__name__}: {e}"[:500])


if __name__ == "__main__":
    main(sys.argv[1])
