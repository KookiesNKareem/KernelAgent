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


def make_handler(events: Path, replay: float | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                body = PAGE.read_bytes()
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
            self.wfile.write(f"data: {line}\n\n".encode())
            self.wfile.flush()

        def _stream(self):
            if replay:
                lines = [x for x in events.read_text().splitlines() if x.strip()]
                prev = None
                for x in lines:
                    t = json.loads(x).get("t")
                    if prev is not None and t is not None:
                        time.sleep(min(max(t - prev, 0) / replay, 3.0))
                    prev = t
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
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.events, a.replay))
    print(f"http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
