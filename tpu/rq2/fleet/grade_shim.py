"""Per-slice grading ingress: HTTP on worker 0 port 8002 -> Ray remotes across the slice.

Neuronic clients cannot be Ray drivers (the firewall admits only tcp 8001-8012 and Ray's
driver protocol needs its own port suite), but intra-slice VPC traffic is unrestricted. So the
league's proven design runs here unchanged -- Ray head on worker 0, minimal grader venvs on
workers 1-7, the league evaluator's payload mode shipping code by value -- and this shim is the
only new piece: a thin stdlib HTTP server that turns a grade request into a Ray task.

  GET  /health          -> {"outstanding", "done", "slots", "ray_alive"}
  POST /grade           <- {"problem", "solution", "fast", "fast_budget", "base_construction"}
                        -> {"score", "valid", "detail", "secs"}

Saturation: clients read `outstanding`/`slots` and route to the least-loaded slice; Ray then
load-balances across that slice's workers natively. Runs ON worker 0 in the league discover
venv with GRADER_HOME pointing at the synced bundle, RAY_ADDRESS=127.0.0.1:6379,
TTD_EVAL_BACKEND handled per request via grade_core(backend="ray"), TTD_RAY_PAYLOAD=1.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import grade_core

SLOTS = int(os.environ.get("GRADE_SLOTS", "600"))       # ray tasks in flight across the slice
POOL = ThreadPoolExecutor(max_workers=SLOTS)            # threads just await blocking get_reward
STATE = {"outstanding": 0, "done": 0}
LOCK = threading.Lock()


def _ray_alive():
    try:
        import ray
        return ray.is_initialized()
    except Exception:
        return False


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):                          # quiet; health is the interface
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            with LOCK:
                st = dict(STATE)
            self._send(200, {**st, "slots": SLOTS, "ray_alive": _ray_alive(),
                             "ts": round(time.time(), 1)})
        else:
            self._send(404, {"error": "unknown path"})

    def do_POST(self):
        if not self.path.startswith("/grade"):
            self._send(404, {"error": "unknown path"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return
        prob = req.get("problem")
        if prob not in grade_core.PROBLEMS:
            self._send(400, {"error": f"unknown problem {prob!r}"})
            return
        with LOCK:
            STATE["outstanding"] += 1
        fut = POOL.submit(
            grade_core.grade, prob, req.get("solution") or "",
            fast=bool(req.get("fast", True)), fast_budget=req.get("fast_budget"),
            base_construction=req.get("base_construction"),
            logdir=os.environ.get("GRADE_LOGDIR", "/tmp/grade_shim"), backend="ray")
        try:
            res = fut.result()                          # holds THIS request's thread only
        finally:
            with LOCK:
                STATE["outstanding"] -= 1
                STATE["done"] += 1
        self._send(200, res)


def main():
    import ray
    addr = os.environ.get("RAY_ADDRESS", "127.0.0.1:6379")
    ray.init(address=addr, ignore_reinit_error=True, log_to_driver=False)
    os.environ["TTD_RAY_PAYLOAD"] = "1"
    port = int(os.environ.get("GRADE_PORT", "8002"))
    print(f"[shim] ray={addr} alive={_ray_alive()} slots={SLOTS} serving 0.0.0.0:{port}",
          flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
