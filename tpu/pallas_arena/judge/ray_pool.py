"""Ray grading pool for the cell's dedicated grading host.

TOPOLOGY. In the v5p-32 RL cell the grading host (w3) is preempted with the
rest of the slice -- shared fate, so there is never a warm trainer waiting on
dead graders. But one judge process grades ONE candidate on ONE chip, leaving
three of the host's four chips idle: 6% of the cell's compute doing the work
that gates every RL step. This module runs N judge ACTORS on that host, one
per chip, fed by the same queue.

WHY RAY. Ray schedules TPU chips as a resource and sets TPU_VISIBLE_CHIPS for
each worker, so a test that needs 4 chips is `resources={"TPU": 4}` rather
than a hand-rolled chip-set mutex. Tensor-parallel cases (splash's tp4-*/
tp8-*) are declared shape cases, so variable width is a real requirement --
this is the mechanism for it. (Their TIMINGS are not yet trustworthy: open
task #12, impossible scores on v6e-8. Scheduling them is ready before
believing them, which is fine -- TP cases are phase 2.)

WHY ACTORS, NOT max_calls=1 TASKS. A PersistentWorker's boot elects baselines
and calibrates the noise floor -- minutes of work amortized over every item
it then grades. A fresh process per grade would re-pay that each time. Crash
isolation does not need it either: `grader.py` already forks a plain
subprocess per candidate, so a core-halting kernel dies in the child. Ray's
actor restart covers the rarer case where the actor process itself dies.

INGRESS STAYS HTTP. The queue keeps the durable lease/requeue semantics that
survived 1699 items with zero loss; Ray is only the intra-host scheduler.

Run (on the grading host, after `ray start --head --resources='{"TPU": 4}'`):
    python -m pallas_arena.judge.ray_pool --queue http://127.0.0.1:8791 \\
      --problems rg_lru --actors 4 --cache gs://...
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request


def _http_json(url: str, payload: dict | None = None, timeout: float = 30.0):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status == 204:
            return None
        return json.loads(resp.read())


class JudgeActorImpl:
    """Boots one PersistentWorker and grades items on its own chip(s).

    Kept as a plain class so it is importable/testable without Ray; the Ray
    decoration happens in `make_actor_cls`.
    """

    def __init__(self, problem: str, cfg: dict):
        # PIN THIS ACTOR TO ITS CHIPS. Ray schedules the actor against the
        # "TPU" resource, but whether it also exports TPU_VISIBLE_CHIPS
        # varies by version -- measured on ray 2.58: the actor came up with
        # it UNSET. On a 1-chip judge that is harmless; on a 4-chip grading
        # host every actor would initialise against all four and collide.
        # Ray always knows the assignment, so derive it and set it here
        # BEFORE jax is imported (libtpu reads it at init).
        try:
            import ray as _ray
            ids = (_ray.get_runtime_context().get_accelerator_ids() or {}).get("TPU") or []
            if ids and not os.environ.get("TPU_VISIBLE_CHIPS"):
                os.environ["TPU_VISIBLE_CHIPS"] = ",".join(str(i) for i in ids)
        except Exception:
            pass
        # Every actor on this host shares ONE XLA/JAX compile cache: the same
        # baselines and reference are compiled by each of them otherwise.
        cache_dir = cfg.get("compile_cache_dir")
        if cache_dir:
            os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
        # Candidate export children must not touch this actor's chip.
        os.environ.setdefault("ARENA_CHILD_JAX_PLATFORMS", "tpu")
        from pallas_arena.judge import cache as cache_mod
        from pallas_arena.judge.worker import PersistentWorker

        cache = cache_mod.RewardCache(cfg["cache"]) if cfg.get("cache") else None
        self.problem = problem
        self.worker = PersistentWorker(
            problem,
            smoke=cfg.get("smoke", False),
            timing_pairs=cfg.get("timing_pairs", 20),
            compile_budget_s=cfg.get("compile_budget_s", 90.0),
            grade_budget_s=cfg.get("grade_budget_s", 900.0),
            cache=cache,
            worker_id=cfg.get("worker_id", "ray-actor"),
        )
        self.boot_report = self.worker.boot()

    def ready(self) -> dict:
        return {
            "problem": self.problem,
            "visible_chips": os.environ.get("TPU_VISIBLE_CHIPS", "?"),
            "jax_devices": self._device_count(),
            "noise_floor": self.boot_report.get("noise_floor"),
            "boot_s": self.boot_report.get("boot_s"),
        }

    def _device_count(self) -> int:
        """How many chips this actor actually sees. On a multi-chip host a
        count > width means the pinning did not take and actors are sharing
        silicon -- which would corrupt every timing on the host."""
        try:
            import jax
            return len(jax.local_devices())
        except Exception:
            return -1

    def grade(self, payload: dict) -> dict:
        return self.worker.grade_code(payload.get("code", ""), tag=payload.get("tag"))


def make_actor_cls(width: int):
    """Ray actor class reserving `width` TPU chips (Ray sets
    TPU_VISIBLE_CHIPS accordingly)."""
    import ray

    return ray.remote(
        num_cpus=cpu_per_actor(),
        resources={"TPU": width},
        max_restarts=5,
        max_task_retries=0,
    )(JudgeActorImpl)


def cpu_per_actor() -> int:
    """Mosaic compiles are CPU-hungry and run concurrently now. Leave the
    host room for the queue and the trainer's own sidecars."""
    try:
        return max(2, (os.cpu_count() or 8) // 8)
    except Exception:
        return 2


def _gsutil_rsync(src: str, dst: str, label: str) -> None:
    import subprocess
    try:
        r = subprocess.run(["gsutil", "-m", "-q", "rsync", "-r", src, dst],
                           capture_output=True, timeout=900)
        print(f"[cache] {label}: rc={r.returncode} {src} -> {dst}", flush=True)
    except Exception as e:  # never let cache plumbing break grading
        print(f"[cache] {label} failed: {type(e).__name__}: {e}", flush=True)


def _cache_file_count(path: str) -> int:
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += len(files)
    return n


def run_pool(
    queue_url: str,
    problems: list[str],
    *,
    actors: int,
    cfg: dict,
    poll_s: float = 1.0,
    heartbeat_frac: float = 3.0,
    max_items: int | None = None,
    idle_exit_s: float | None = None,
) -> int:
    import ray

    base = queue_url.rstrip("/")
    ray.init(address=cfg.get("ray_address", "auto"), ignore_reinit_error=True)

    # GROUND-TRUTH COMPILE CACHE. Every judge boot compiles the production
    # baseline and the fp32 reference before it can grade anything -- the
    # bulk of a ~119 s actor boot, paid again on every preemption. Restore
    # the banked cache first; snapshot it back ONCE after boot, before any
    # candidate compiles, so the bucket holds baseline entries and not
    # thousands of throwaway candidate kernels.
    # The path is keyed by chip: an XLA cache is compiled FOR a target, so a
    # v6e cache cannot serve a v5p judge (same rule as the vLLM caches).
    jax_cache_gcs = cfg.get("jax_cache_gcs")
    local_cache = os.environ.get("JAX_COMPILATION_CACHE_DIR",
                                 os.path.expanduser("~/jax-compile-cache"))
    os.makedirs(local_cache, exist_ok=True)
    if jax_cache_gcs:
        _gsutil_rsync(jax_cache_gcs, local_cache, "restore")
        print(f"[cache] {_cache_file_count(local_cache)} local entries after restore", flush=True)

    width = int(cfg.get("width", 1))
    cls = make_actor_cls(width)
    pool = []
    for i in range(actors):
        problem = problems[i % len(problems)]
        a = cls.remote(problem, {**cfg, "worker_id": f"ray-{problem}-{i}"})
        pool.append({"actor": a, "problem": problem, "busy": None})
    print(f"[pool] booting {actors} actors ({width} chip each) ...", flush=True)
    for i, slot in enumerate(pool):
        info = ray.get(slot["actor"].ready.remote())
        print(f"[pool] actor {i} up: {info}", flush=True)

    if jax_cache_gcs:
        # Boot is done; the cache now holds exactly the baseline/reference
        # entries. Snapshot before the first candidate touches it.
        n = _cache_file_count(local_cache)
        print(f"[cache] snapshotting {n} entries after boot", flush=True)
        _gsutil_rsync(local_cache, jax_cache_gcs, "save")

    inflight: dict = {}  # ObjectRef -> lease info
    stop = threading.Event()

    def beat():
        """One thread beats EVERY in-flight lease. Per-item threads would
        multiply with concurrency for no benefit."""
        while not stop.wait(2.0):
            for meta in list(inflight.values()):
                try:
                    _http_json(f"{base}/heartbeat", {"lease_id": meta["lease_id"]})
                except Exception:
                    pass

    hb = threading.Thread(target=beat, daemon=True)
    hb.start()

    done_count = 0
    # idle_exit_s: leave when the queue has been empty this long AND nothing
    # is in flight. None = never (the RL cell's pool must outlive lulls
    # between steps); the offline fleet sets it so a finished corpus releases
    # the host instead of holding a spot VM idle.
    last_work = time.time()
    try:
        while max_items is None or done_count < max_items:
            if (idle_exit_s is not None and not inflight
                    and time.time() - last_work > idle_exit_s):
                print(f"[pool] idle {idle_exit_s:.0f}s with an empty queue; exiting", flush=True)
                break
            # 1. fill idle actors
            progressed = False
            for slot in pool:
                if slot["busy"] is not None:
                    continue
                if max_items is not None and done_count + len(inflight) >= max_items:
                    break
                try:
                    item = _http_json(f"{base}/work?worker_id=ray-pool")
                except Exception:
                    item = None
                if not item:
                    break
                payload = item.get("payload") or {}
                want = payload.get("problem")
                if want and want != slot["problem"]:
                    # This actor is booted for another problem. Post a judge
                    # fault (excluded from reward) rather than sit on a lease
                    # we cannot serve; loud, because it means misconfiguration.
                    print(f"[pool] WARNING no actor for problem={want}; faulting {item['work_id']}", flush=True)
                    last_work = time.time()
                    try:
                        _http_json(f"{base}/result", {
                            "lease_id": item["lease_id"], "work_id": item["work_id"],
                            "result": {"ok": False, "gate": "judge_fault",
                                       "violations": [f"no judge actor booted for problem {want}"]},
                        })
                    except Exception:
                        pass
                    continue
                last_work = time.time()
                ref = slot["actor"].grade.remote(payload)
                inflight[ref] = {
                    "lease_id": item["lease_id"], "work_id": item["work_id"],
                    "slot": slot, "t0": time.time(),
                }
                slot["busy"] = ref
                progressed = True

            if not inflight:
                time.sleep(poll_s)
                continue

            # 2. collect whatever finished
            ready, _ = ray.wait(list(inflight), num_returns=1, timeout=poll_s)
            for ref in ready:
                meta = inflight.pop(ref)
                meta["slot"]["busy"] = None
                try:
                    result = ray.get(ref)
                except Exception as e:  # actor died mid-grade: let the lease
                    # expire so the queue requeues it on another actor.
                    print(f"[pool] grade failed {meta['work_id']}: {type(e).__name__}: {e}", flush=True)
                    continue
                result["item_wall_s"] = time.time() - meta["t0"]
                # CORE HALT RECOVERY. A candidate can halt a TPU core at
                # RUNTIME (past Mosaic): measured 2026-08-26, one kernel
                # halted a SparseCoreSequencer and every later grade on that
                # chip returned "the program continuator has halted
                # unexpectedly" -- 102 halts, all subsequent verdicts junk.
                # Forking per candidate does not help: the DEVICE is
                # poisoned, not the process. Replace the actor so the next
                # item meets a freshly initialised chip. In RL this is the
                # difference between one bad rollout and a whole step of
                # falsely-zero rewards.
                blob = (str(result.get("violations")) + str(result.get("observation")))
                if any(sig in blob for sig in (
                        "halted unexpectedly", "CoreHalt", "continuator has halted")):
                    slot = meta["slot"]
                    print(f"[pool] CORE HALT on {meta['work_id']}; replacing actor", flush=True)
                    try:
                        ray.kill(slot["actor"], no_restart=True)
                    except Exception:
                        pass
                    slot["actor"] = cls.remote(slot["problem"],
                                               {**cfg, "worker_id": f"ray-{slot['problem']}-r"})
                    try:
                        print(f"[pool] replacement actor up: "
                              f"{ray.get(slot['actor'].ready.remote())}", flush=True)
                    except Exception as e:
                        print(f"[pool] replacement actor FAILED: {e!r}", flush=True)
                try:
                    _http_json(f"{base}/result", {
                        "lease_id": meta["lease_id"], "work_id": meta["work_id"], "result": result,
                    })
                    done_count += 1
                    # WHY it failed, not just that it did: without the gate and
                    # first violation here, a systematic rejection (every
                    # candidate dying at the same pregate) is invisible until
                    # someone reads the caller's results file.
                    why = ""
                    if not result.get("passed"):
                        v = result.get("violations") or []
                        why = f" gate={result.get('gate')} {str(v[0])[:120] if v else ''}"
                    print(f"[pool] {meta['work_id']} done in {result['item_wall_s']:.1f}s "
                          f"passed={result.get('passed')}{why} ({done_count} total)", flush=True)
                except Exception as e:
                    print(f"[pool] result post failed for {meta['work_id']}: {e!r}", flush=True)
            if not ready and not progressed:
                time.sleep(poll_s)
    finally:
        stop.set()
    return done_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--problems", default="rg_lru", help="comma list; actors are assigned round-robin")
    ap.add_argument("--actors", type=int, default=4, help="one per chip on a v5p-8 grading host")
    ap.add_argument("--width", type=int, default=1, help="chips per actor (TP cases: >1)")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--compile-cache-dir", default=os.path.expanduser("~/jax-compile-cache"))
    ap.add_argument("--jax-cache-gcs", default=os.environ.get("ARENA_JAX_CACHE_GCS", ""),
                    help="GCS prefix for the baseline compile cache; MUST be keyed by chip kind")
    ap.add_argument("--timing-pairs", type=int, default=20)
    ap.add_argument("--compile-budget-s", type=float, default=90.0)
    ap.add_argument("--grade-budget-s", type=float, default=900.0)
    ap.add_argument("--ray-address", default="auto")
    ap.add_argument("--poll-s", type=float, default=1.0)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--idle-exit-s", type=float, default=None,
                    help="exit after this many idle seconds (offline fleets); default never")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = {
        "cache": args.cache,
        "compile_cache_dir": args.compile_cache_dir,
        "timing_pairs": args.timing_pairs,
        "compile_budget_s": args.compile_budget_s,
        "grade_budget_s": args.grade_budget_s,
        "ray_address": args.ray_address,
        "jax_cache_gcs": args.jax_cache_gcs or None,
        "width": args.width,
        "smoke": args.smoke,
    }
    n = run_pool(
        args.queue,
        [p.strip() for p in args.problems.split(",") if p.strip()],
        actors=args.actors,
        cfg=cfg,
        poll_s=args.poll_s,
        max_items=args.max_items,
        idle_exit_s=args.idle_exit_s,
    )
    print(f"[pool] exiting after {n} items")


if __name__ == "__main__":
    main()
