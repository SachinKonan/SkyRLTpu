"""Pull-queue for arena grading (user-settled architecture, phase 3).

The queue LIVES WITH THE TRAINING CLIENT (v5p host in prod, the sbatch
driver in sim/phase-4); judges are pure stateless pollers:

    client  ->  POST /submit  {problem, code, ...}      -> {work_id}
    judge   ->  GET  /work?worker_id=w                  -> lease or 204
    judge   ->  POST /heartbeat {lease_id}              (extends the lease)
    judge   ->  POST /result   {lease_id, work_id, result}
    client  ->  GET  /result/{work_id}                  (poll until done)

Semantics:
  * strict FIFO dispatch from a single thread-locked in-memory queue;
  * every lease has an expiry (~2x max grade time, launch flag) refreshed
    by heartbeats; expired leases are swept lazily on every /work//status
    call and the item is REQUEUED (attempts += 1);
  * double grading is harmless by design: grading is idempotent (hidden
    seeds live judge-side, hash->reward cache) and /result accepts a late
    worker's post for an already-completed item as a counted duplicate;
  * the queue is deliberately memoryless across restarts — the CLIENT is
    the source of truth and resubmits whatever it has not seen complete
    (ArenaQueueClient.wait does this automatically). No GCS registry, no
    client-side judge routing; a keeper's only job is fleet size.

Run:  python -m pallas_arena.judge.queue --port 8770 --lease-timeout 120
"""

from __future__ import annotations

import argparse
import itertools
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel


class SubmitRequest(BaseModel):
    problem: str
    code: str
    mode: str = "full"
    smoke: bool = False
    cases: list[str] | None = None
    enforce_pallas: bool | None = None
    tag: str | None = None  # client-side correlation id


class HeartbeatRequest(BaseModel):
    lease_id: str


class ResultRequest(BaseModel):
    lease_id: str
    work_id: str
    result: dict


@dataclass
class _Item:
    work_id: str
    payload: dict
    state: str = "queued"  # queued | leased | done
    lease_id: str | None = None
    lease_expiry: float = 0.0
    worker_id: str | None = None
    attempts: int = 0
    result: dict | None = None
    enqueued_at: float = field(default_factory=time.time)
    completed_at: float | None = None


def create_queue_app(lease_timeout_s: float = 120.0) -> FastAPI:
    lock = threading.Lock()
    items: dict[str, _Item] = {}
    fifo: deque[str] = deque()
    seq = itertools.count()
    stats = {
        "submitted": 0,
        "completed": 0,
        "requeues": 0,
        "duplicates": 0,
        "expired_leases": 0,
    }
    workers_last_poll: dict[str, float] = {}

    def _sweep_expired_locked(now: float) -> None:
        # lazy sweep: any leased item past expiry goes back to the FIFO head
        # region (front, preserving age order among requeues)
        expired = [it for it in items.values() if it.state == "leased" and it.lease_expiry <= now]
        for it in sorted(expired, key=lambda x: x.enqueued_at):
            it.state = "queued"
            it.lease_id = None
            it.worker_id = None
            stats["expired_leases"] += 1
            stats["requeues"] += 1
            fifo.appendleft(it.work_id)

    app = FastAPI(title="pallas-arena work queue")

    @app.post("/submit")
    async def submit(req: SubmitRequest):
        work_id = f"w{next(seq):06d}-{uuid.uuid4().hex[:8]}"
        payload = req.model_dump()
        with lock:
            items[work_id] = _Item(work_id=work_id, payload=payload)
            fifo.append(work_id)
            stats["submitted"] += 1
        return {"work_id": work_id}

    @app.get("/work")
    async def get_work(worker_id: str = "anonymous"):
        now = time.time()
        with lock:
            workers_last_poll[worker_id] = now
            _sweep_expired_locked(now)
            while fifo:
                work_id = fifo.popleft()
                it = items.get(work_id)
                if it is None or it.state != "queued":
                    continue  # stale pointer (completed while requeued etc.)
                it.state = "leased"
                it.lease_id = uuid.uuid4().hex
                it.lease_expiry = now + lease_timeout_s
                it.worker_id = worker_id
                it.attempts += 1
                return {
                    "work_id": it.work_id,
                    "lease_id": it.lease_id,
                    "lease_timeout_s": lease_timeout_s,
                    "attempt": it.attempts,
                    "payload": it.payload,
                }
        return Response(status_code=204)  # no body: nothing to grade

    @app.post("/heartbeat")
    async def heartbeat(req: HeartbeatRequest):
        now = time.time()
        with lock:
            for it in items.values():
                if it.lease_id == req.lease_id and it.state == "leased":
                    if it.lease_expiry <= now:
                        break  # already expired: worker must not keep it
                    it.lease_expiry = now + lease_timeout_s
                    return {"ok": True, "lease_expiry": it.lease_expiry}
        raise HTTPException(status_code=404, detail="unknown or expired lease")

    @app.post("/result")
    async def post_result(req: ResultRequest):
        now = time.time()
        with lock:
            it = items.get(req.work_id)
            if it is None:
                raise HTTPException(status_code=404, detail="unknown work_id")
            if it.state == "done":
                stats["duplicates"] += 1
                return {"ok": True, "duplicate": True}
            # accept results even from an expired lease (double-grades are
            # harmless and the item may not have been re-leased yet)
            it.result = req.result
            it.state = "done"
            it.completed_at = now
            it.lease_id = None
            stats["completed"] += 1
            return {"ok": True, "duplicate": False}

    @app.get("/result/{work_id}")
    async def get_result(work_id: str):
        with lock:
            it = items.get(work_id)
            if it is None:
                raise HTTPException(status_code=404, detail="unknown work_id")
            if it.state != "done":
                return {"done": False, "state": it.state, "attempts": it.attempts}
            return {"done": True, "result": it.result, "attempts": it.attempts}

    @app.get("/status")
    async def status():
        now = time.time()
        with lock:
            _sweep_expired_locked(now)
            depth = sum(1 for it in items.values() if it.state == "queued")
            leased = sum(1 for it in items.values() if it.state == "leased")
            return {
                "ok": True,
                "queue_depth": depth,
                "leased": leased,
                "lease_timeout_s": lease_timeout_s,
                "workers_seen": {w: now - t for w, t in workers_last_poll.items()},
                **stats,
            }

    return app


# ------------------------------------------------------------------ client
class ArenaQueueClient:
    """Client-side helper: submit + wait with automatic resubmission.

    The client keeps every payload it submitted; if the queue restarts (it
    is memoryless by design) /result returns 404 and the item is silently
    resubmitted under a fresh work_id. Safe because grading is idempotent."""

    def __init__(self, base_url: str, timeout_s: float = 30.0):
        self.base = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._payloads: dict[str, dict] = {}

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=self.timeout_s) as resp:
            return json.loads(resp.read())

    def submit(self, problem: str, code: str, **kw) -> str:
        payload = {"problem": problem, "code": code, **kw}
        work_id = self._post("/submit", payload)["work_id"]
        self._payloads[work_id] = payload
        return work_id

    def status(self) -> dict:
        return self._get("/status")

    def wait(self, work_ids: list[str], timeout_s: float = 600.0, poll_s: float = 1.0) -> dict[str, dict]:
        """Wait for all ids; returns {original_work_id: result}. Resubmits
        through queue restarts and maps the fresh id back to the original."""
        alias = {wid: wid for wid in work_ids}  # original -> current
        results: dict[str, dict] = {}
        deadline = time.time() + timeout_s
        while time.time() < deadline and len(results) < len(work_ids):
            for orig in work_ids:
                if orig in results:
                    continue
                cur = alias[orig]
                try:
                    r = self._get(f"/result/{cur}")
                except urllib.error.HTTPError as e:
                    if e.code == 404 and cur in self._payloads:
                        # queue lost the item (restart): resubmit
                        new_id = self._post("/submit", self._payloads[cur])["work_id"]
                        self._payloads[new_id] = self._payloads[cur]
                        alias[orig] = new_id
                    continue
                except Exception:
                    continue  # queue briefly down: retry next poll
                if r.get("done"):
                    results[orig] = r["result"]
            if len(results) < len(work_ids):
                time.sleep(poll_s)
        return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--lease-timeout", type=float, default=120.0)
    args = ap.parse_args()

    import uvicorn

    uvicorn.run(create_queue_app(args.lease_timeout), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
