"""FastAPI FIFO judge server (~the 150-line judge from DESIGN.md).

One judge serves ONE problem type, fixed at launch (`--problem`); requests
for any other problem are rejected with 400 — refusing to grade anything
else is itself an anti-cheat property. One worker per chip; each worker
pulls from a single FIFO queue and grades candidates through
grader.grade() (fork-per-candidate, timeout, RLIMIT_AS, hidden stdin seed).

At boot the judge measures the per-chip noise floor (ref-vs-ref through the
identical interleaved protocol) and every grade is gated on it. A GCS (or
local-dir) hash->reward cache makes repeat kernels instant and consistent.

Launch (phase-2 judge host, from tmux — the lora-prune-daemon pattern):
  python -m pallas_arena.judge.server --problem rmsnorm --port 8765 \
      --cache gs://sk7524-pallas-arena-us-east5/reward-cache
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pallas_arena.judge import grader
from pallas_arena.judge.cache import RewardCache

VALID_MODES = ("full", "gates", "pregate")


class GradeRequest(BaseModel):
    problem: str
    code: str
    mode: str = "full"
    smoke: bool = False
    cases: list[str] | None = None
    timeout_s: float | None = None
    enforce_pallas: bool | None = None


def create_app(
    problem_name: str,
    *,
    workers: int = 1,
    cache_root: str | None = None,
    smoke: bool = False,
    measure_floor_on_boot: bool = True,
    worker_envs: list[dict] | None = None,
    grade_overrides: dict | None = None,
) -> FastAPI:
    cache = RewardCache(cache_root) if cache_root else None
    worker_envs = worker_envs or [{} for _ in range(workers)]
    assert len(worker_envs) == workers, "need one env per worker (chip)"
    grade_overrides = grade_overrides or {}

    state: dict = {
        "problem": problem_name,
        "queue": None,
        "seq": itertools.count(),
        "graded_total": 0,
        "completed_order": [],
        "noise_floors": [None] * workers,
        "boot_ref_vs_ref": [None] * workers,
        "started_at": time.time(),
    }

    async def _worker(idx: int, queue: asyncio.Queue):
        env = worker_envs[idx]
        while True:
            req_id, req, fut = await queue.get()
            try:
                kwargs = dict(
                    mode=req.mode,
                    smoke=req.smoke or smoke,
                    cases=req.cases,
                    enforce_pallas=req.enforce_pallas,
                    noise_floor=state["noise_floors"][idx],
                    cache=cache,
                    child_env=env,
                )
                if req.timeout_s:
                    kwargs["timeout_s"] = min(req.timeout_s, 900.0)
                kwargs.update(grade_overrides)
                result = await asyncio.to_thread(grader.grade, problem_name, req.code, **kwargs)
                result["request_id"] = req_id
                result["worker"] = idx
                state["graded_total"] += 1
                state["completed_order"].append(req_id)
                if not fut.done():
                    fut.set_result(result)
            except Exception as e:  # harness bug — surface, don't die
                if not fut.done():
                    fut.set_exception(e)
            finally:
                queue.task_done()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        queue: asyncio.Queue = asyncio.Queue()
        state["queue"] = queue
        if measure_floor_on_boot:
            for idx in range(workers):
                floor_res = await asyncio.to_thread(
                    grader.measure_noise_floor, problem_name, smoke=smoke, child_env=worker_envs[idx]
                )
                if floor_res.get("ok"):
                    state["noise_floors"][idx] = floor_res.get("noise_floor")
                    state["boot_ref_vs_ref"][idx] = floor_res.get("ref_vs_ref_scores")
        tasks = [asyncio.create_task(_worker(i, queue)) for i in range(workers)]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()

    app = FastAPI(title=f"pallas-arena judge [{problem_name}]", lifespan=lifespan)
    app.state.arena = state

    @app.post("/grade")
    async def grade_endpoint(req: GradeRequest):
        if req.problem != problem_name:
            raise HTTPException(
                status_code=400,
                detail=f"this judge grades only {problem_name!r} " f"(launch flag); got {req.problem!r}",
            )
        if req.mode not in VALID_MODES:
            raise HTTPException(status_code=400, detail=f"mode must be one of {VALID_MODES}")
        req_id = next(state["seq"])
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await state["queue"].put((req_id, req, fut))
        return await fut

    @app.get("/healthz")
    async def healthz():
        return {
            "ok": True,
            "problem": problem_name,
            "queue_depth": state["queue"].qsize() if state["queue"] else -1,
            "graded_total": state["graded_total"],
            "workers": workers,
            "noise_floors": state["noise_floors"],
            "boot_ref_vs_ref": state["boot_ref_vs_ref"],
            "uptime_s": time.time() - state["started_at"],
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--workers", type=int, default=1, help="one grading worker per chip")
    ap.add_argument("--cache", default=None, help="gs://... prefix or local dir for the reward cache")
    ap.add_argument("--smoke", action="store_true", help="grade tiny smoke shapes (CPU battery)")
    ap.add_argument("--no-boot-floor", action="store_true")
    ap.add_argument("--worker-envs", default=None, help="JSON list of env dicts, one per worker/chip")
    args = ap.parse_args()

    import uvicorn

    app = create_app(
        args.problem,
        workers=args.workers,
        cache_root=args.cache,
        smoke=args.smoke,
        measure_floor_on_boot=not args.no_boot_floor,
        worker_envs=json.loads(args.worker_envs) if args.worker_envs else None,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
