"""Live view of a run: tails OUT/events.jsonl and streams it to the page over server-sent events.

  python -m kernel_contract.dashboard.server runs/x/events.jsonl --port 8765
  python -m kernel_contract.dashboard.server runs/x/events.jsonl --replay 4   # recorded run at 4x speed
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("index.html")
RECUR_PAGE = Path(__file__).with_name("recur.html")


def preamble(layer: str | None, baseline: Path | None, kernel_train: Path | None = None) -> list[str]:
    """Events the recurrence page needs before the run's own: the equation and the reference training curve."""
    out = []
    if layer:
        import importlib
        mod = importlib.import_module(layer)
        out.append(json.dumps({"event": "layer", "name": layer.rsplit(".", 1)[-1], "doc": (mod.__doc__ or "").strip()}))
    if baseline:
        recs = [json.loads(x) for x in baseline.read_text().splitlines() if x.strip()]
        warm = sorted(r["step_s"] for r in recs[5:])
        out.append(json.dumps({"event": "baseline", "loss": [r["loss"] for r in recs],
                               "median_step_ms": 1000 * warm[len(warm) // 2]}))
    if kernel_train:  # replaying a run whose training happened outside the MCP tools
        recs = [json.loads(x) for x in kernel_train.read_text().splitlines() if x.strip()]
        warm = sorted(r["step_s"] for r in recs[5:])
        val = [r for r in recs if "val_loss" in r]
        out.append(json.dumps({"event": "train_done", "steps": len(recs), "loss_curve": [r["loss"] for r in recs],
                               "median_step_ms": 1000 * warm[len(warm) // 2], "final_val_loss": val[-1]["val_loss"]}))
    return out


def make_handler(events: Path, replay: float | None, pre: list[str] = (), model: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?") or self.path.startswith("/recur"):
                body = (RECUR_PAGE if self.path.startswith("/recur") else PAGE).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/events":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                self._stream()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send(self, line: str):
            if model and '"llm_done"' in line:  # older logs lack per-call cost: price their usage
                from ..paper_agent import usage_cost
                e = json.loads(line)
                e.setdefault("usd", usage_cost(model, e.get("usage")))
                line = json.dumps(e)
            self.wfile.write(f"data: {line}\n\n".encode())
            self.wfile.flush()

        def _stream(self):
            self._send(json.dumps({"event": "run_metadata", "mode": "recorded" if replay else "live",
                                   "replay_speed": replay}))
            deferred = [x for x in pre if json.loads(x).get("event") == "train_done"]
            for x in pre:
                if x not in deferred:
                    self._send(x)
            if replay:
                lines = [x for x in events.read_text().splitlines() if x.strip()]
                prev = None
                for x in lines:
                    t = json.loads(x).get("t")
                    if prev is not None and t is not None:
                        time.sleep(min(max(t - prev, 0) / replay, 3.0))
                    prev = t
                    self._send(x)
                for x in deferred:
                    self._send(x)
                while True:  # keep the stream open so the browser does not reconnect and replay again
                    time.sleep(15)
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            pos = 0
            while True:
                if events.exists():
                    with open(events) as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                    for x in chunk.splitlines():
                        if x.strip():
                            self._send(x)
                time.sleep(0.25)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", type=Path)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--replay", type=float, default=None, help="replay a finished run at this speed")
    ap.add_argument("--layer", help="layer module whose equation the recurrence page shows")
    ap.add_argument("--baseline", type=Path, help="reference-loop training log (jsonl) to plot against")
    ap.add_argument("--model", help="model id, to price the usage in older logs")
    ap.add_argument("--kernel-train", type=Path, help="the kernel's training log, when replaying a paper-agent run")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.events, a.replay, preamble(a.layer, a.baseline, a.kernel_train), a.model))
    print(f"http://127.0.0.1:{a.port}" + ("/recur" if a.layer else ""), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
