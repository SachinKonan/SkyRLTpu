#!/usr/bin/env python
"""Prove `rs_generate.py`'s client really runs N requests concurrently.

The previous attempt at this experiment produced

    16/1060 ok=16 err=0 tok=174025 172 tok/s 1012s elapsed

which is exactly what four concurrent streams look like, and nobody could tell
until 1012 s of slice time had been spent.  The client's concurrency is a
property of the client, so it is checked here on CPU, against a mock HTTP
server that answers after a fixed delay, before any TPU is booked.

PASS criteria, with 64 requested workers and a 0.5 s server:
  * peak in-flight >= 64
  * wall clock < 4 x the per-request delay (i.e. genuinely overlapped, not
    serialised -- serialised would be 64 x 0.5 = 32 s)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rs_generate  # noqa: E402

DELAY = 0.5
NTOK = 8


class Server(ThreadingHTTPServer):
    # socketserver's default listen backlog is 5, so 64 simultaneous connects
    # get RST by the MOCK and the test fails for a reason that has nothing to
    # do with the client. uvicorn (what vLLM serves behind) defaults to 2048.
    request_queue_size = 256
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        time.sleep(DELAY)
        body = json.dumps(
            {"choices": [{"token_ids": list(range(NTOK)), "text": "x",
                          "finish_reason": "length"}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Tok:
    def decode(self, ids):
        return "x" * len(ids)

    def encode(self, s, add_special_tokens=False):
        return [0] * len(s)


def main() -> int:
    want = 64
    srv = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    gauge = rs_generate.Gauge()
    eng = rs_generate.Engine([base], "m", _Tok(), gauge)

    ths = [
        threading.Thread(
            target=lambda: eng.complete([1, 2, 3], NTOK, 1.0, [], False),
            daemon=True,
        )
        for _ in range(want)
    ]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    el = time.time() - t0
    srv.shutdown()

    ok = gauge.peak >= want and el < DELAY * 4
    print(
        f"CONC-SELFTEST requested={want} peak_inflight={gauge.peak} "
        f"wall={el:.2f}s (serial would be {want * DELAY:.0f}s) -> "
        + ("PASS" if ok else "FAIL"),
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
