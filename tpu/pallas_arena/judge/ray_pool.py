"""Ray grading pool for the cell's dedicated grading host.

TOPOLOGY. In the v5p-32 RL cell the grading host (w3) is preempted with the
rest of the slice -- shared fate, so there is never a warm trainer waiting on
dead graders. But ONE judge process grades ONE candidate on ONE chip, leaving
three of the host's four chips idle: 6% of the cell's compute doing the work
that gates every RL step.

DESIGN: ONE RAY HEAD, ONE TASK PER TEST, CHIPS AS A RESOURCE.

    grade.options(resources={"TPU": width}).remote(...)

Ray schedules TPU chips as a resource, so a test that needs 4 chips is
`width=4` and a normal test is `width=1` -- no hand-rolled chip mutex. This
is what makes TENSOR-PARALLEL tests work: a task holds its chips only while
it runs and releases them on return, so a TP-4 test is scheduled as soon as
four chips free up. (Long-lived width-1 ACTORS cannot do this: they hold
every chip permanently and Ray will not preempt them, so a {"TPU": 4}
request could never be satisfied on the same host.) splash declares
tp4-*/tp8-* shape cases, so variable width is a real requirement; whether
their TIMINGS are trustworthy is open task #12, but scheduling them
correctly is this module's job.

max_calls=1 (a FRESH PROCESS per task) is load-bearing, not caution: JAX and
libtpu initialise once per process and bind to TPU_VISIBLE_CHIPS at init, so
a REUSED Ray worker could carry a stale chip pin into a task scheduled on
different chips -- silent co-tenancy, and since the reward is a latency
ratio, wrong numbers look plausible rather than broken. A fresh process pins
cleanly every time; it also means a core-halting candidate (measured: one
halted a SparseCoreSequencer and junked 102 subsequent verdicts) poisons at
most its own process, with libtpu re-initialising on the next task.

What makes fresh processes affordable is the banked GROUND-TRUTH COMPILE
CACHE: the baseline/reference compiles dominate a judge's warmup and are
restored from GCS instead of recompiled. Every verdict carries task_boot_s
so the true per-task overhead stays measured, not assumed.

INGRESS STAYS HTTP. The queue keeps the durable lease/requeue semantics that
survived 1699 items with zero loss; Ray is only the intra-host scheduler.

Run (on the grading host, after ray_start_tpu.sh):
    python -m pallas_arena.judge.ray_pool --queue http://127.0.0.1:8791 \\
      --problems rg_lru --chips 4 --cache gs://... --jax-cache-gcs gs://...
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


def _pin_chips() -> str:
    """Bind this process to the chips Ray assigned to this task.

    Ray publishes the assignment via the runtime context; whether it also
    exports TPU_VISIBLE_CHIPS varies by version (measured on ray 2.58: it
    did NOT). libtpu reads the variable at process init, so this must run
    before jax is imported. Only plain chip indices are usable; anything
    else would break init for the task, which is worse than not pinning.
    """
    try:
        import ray

        ids = (ray.get_runtime_context().get_accelerator_ids() or {}).get("TPU") or []
        idx = [str(int(i)) for i in ids if str(i).strip().lstrip("-").isdigit()]
        if idx and not os.environ.get("TPU_VISIBLE_CHIPS"):
            os.environ["TPU_VISIBLE_CHIPS"] = ",".join(idx)
            return os.environ["TPU_VISIBLE_CHIPS"]
    except Exception:
        pass
    return os.environ.get("TPU_VISIBLE_CHIPS", "?")


def grade_one(problem: str, payload: dict, cfg: dict) -> dict:
    """Grade ONE candidate in this task's own fresh process."""
    pinned = _pin_chips()
    cache_dir = cfg.get("compile_cache_dir")
    if cache_dir:
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    os.environ.setdefault("ARENA_CHILD_JAX_PLATFORMS", "tpu")

    from pallas_arena.judge import cache as cache_mod
    from pallas_arena.judge.worker import PersistentWorker

    t0 = time.time()
    rc = cache_mod.RewardCache(cfg["cache"]) if cfg.get("cache") else None
    # EXPLICIT CASE LISTS. The worker's default (declared non-probe cases) is
    # wrong for fleet grading: splash's declared shapes cannot be graded on
    # one chip at all (the fp32 reference materialises 10.9-43.5 GB), and the
    # prompt declares the PROBE set. tp* cases only run when listed here AND
    # the task was scheduled wide enough to see the chips.
    cases = (cfg.get("cases_by_problem") or {}).get(problem)
    worker = PersistentWorker(
        problem,
        cases=cases,
        smoke=cfg.get("smoke", False),
        timing_pairs=cfg.get("timing_pairs", 20),
        compile_budget_s=cfg.get("compile_budget_s", 90.0),
        grade_budget_s=cfg.get("grade_budget_s", 900.0),
        cache=rc,
        worker_id=f"ray-task-{pinned}",
    )
    boot = worker.boot()
    t_boot = time.time() - t0
    result = worker.grade_code(payload.get("code", ""), tag=payload.get("tag"))
    # The number that keeps this design honest: if warm boots stop being
    # cheap, it shows up on every verdict rather than in a forgotten note.
    result["task_boot_s"] = round(t_boot, 1)
    result["task_chips"] = pinned
    result["task_noise_floor"] = boot.get("noise_floor")
    return result


def cpu_per_task() -> int:
    """Mosaic compiles are CPU-hungry and now run concurrently; leave the
    host room for the queue and any trainer sidecars."""
    try:
        return max(2, (os.cpu_count() or 8) // 8)
    except Exception:
        return 2


def item_width(payload: dict, default: int = 1) -> int:
    """Chips this test needs: explicit width/tp/chips in the payload wins."""
    for key in ("width", "tp", "chips"):
        v = payload.get(key)
        if v:
            try:
                return max(1, int(v))
            except (TypeError, ValueError):
                continue
    return default


def _gsutil_rsync(src: str, dst: str, label: str) -> None:
    import subprocess

    try:
        r = subprocess.run(["gsutil", "-m", "-q", "rsync", "-r", src, dst],
                           capture_output=True, timeout=900)
        print(f"[cache] {label}: rc={r.returncode} {src} -> {dst}", flush=True)
    except Exception as e:  # cache plumbing must never break grading
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
    chips: int,
    cfg: dict,
    poll_s: float = 1.0,
    max_items: int | None = None,
    idle_exit_s: float | None = None,
) -> int:
    import ray

    base = queue_url.rstrip("/")
    ray.init(address=cfg.get("ray_address", "auto"), ignore_reinit_error=True)

    # GROUND-TRUTH COMPILE CACHE: restore before any task runs; snapshot back
    # after the first grade completes (the cache then holds baseline entries,
    # not thousands of throwaway candidate kernels).
    jax_cache_gcs = cfg.get("jax_cache_gcs")
    local_cache = cfg.get("compile_cache_dir") or os.path.expanduser("~/jax-compile-cache")
    os.makedirs(local_cache, exist_ok=True)
    cfg = {**cfg, "compile_cache_dir": local_cache}
    if jax_cache_gcs:
        _gsutil_rsync(jax_cache_gcs, local_cache, "restore")
        print(f"[cache] {_cache_file_count(local_cache)} entries after restore", flush=True)
    banked = not bool(jax_cache_gcs)

    cpus = cpu_per_task()
    default_width = int(cfg.get("width", 1))
    grade = ray.remote(num_cpus=cpus, max_calls=1)(grade_one)
    print(f"[pool] task mode: {chips} chips, {cpus} cpus/task, default width "
          f"{default_width}, problems {problems}", flush=True)

    inflight: dict = {}  # ObjectRef -> lease meta
    chips_used = 0
    stop = threading.Event()

    def beat():
        """One thread beats EVERY in-flight lease; per-item threads would
        multiply with concurrency for no benefit."""
        while not stop.wait(2.0):
            for meta in list(inflight.values()):
                try:
                    _http_json(f"{base}/heartbeat", {"lease_id": meta["lease_id"]})
                except Exception:
                    pass

    threading.Thread(target=beat, daemon=True).start()

    done_count = 0
    last_work = time.time()
    try:
        while max_items is None or done_count < max_items:
            if (idle_exit_s is not None and not inflight
                    and time.time() - last_work > idle_exit_s):
                print(f"[pool] idle {idle_exit_s:.0f}s with an empty queue; exiting", flush=True)
                break

            # 1. lease work while chips remain. Leasing past capacity would
            #    park items in Ray's queue burning their lease timeout, so we
            #    stop at the chip budget; the next pass picks up more.
            progressed = False
            while chips_used < chips and (max_items is None
                                          or done_count + len(inflight) < max_items):
                try:
                    item = _http_json(f"{base}/work?worker_id=ray-pool")
                except Exception:
                    item = None
                if not item:
                    break
                payload = item.get("payload") or {}
                want = payload.get("problem") or problems[0]
                if want not in problems:
                    print(f"[pool] WARNING problem={want} not served by this pool; "
                          f"faulting {item['work_id']}", flush=True)
                    last_work = time.time()
                    try:
                        _http_json(f"{base}/result", {
                            "lease_id": item["lease_id"], "work_id": item["work_id"],
                            "result": {"ok": False, "gate": "judge_fault",
                                       "violations": [f"pool serves {problems}, not {want}"]},
                        })
                    except Exception:
                        pass
                    continue
                w = min(item_width(payload, default_width), chips)
                ref = grade.options(resources={"TPU": w}).remote(want, payload, cfg)
                inflight[ref] = {"lease_id": item["lease_id"], "work_id": item["work_id"],
                                 "t0": time.time(), "width": w}
                chips_used += w
                last_work = time.time()
                progressed = True

            if not inflight:
                time.sleep(poll_s)
                continue

            # 2. collect whatever finished
            ready, _ = ray.wait(list(inflight), num_returns=1, timeout=poll_s)
            for ref in ready:
                meta = inflight.pop(ref)
                chips_used -= meta["width"]
                try:
                    result = ray.get(ref)
                except Exception as e:
                    # Task process died (OOM, segfault): post NOTHING -- the
                    # lease expires and the queue requeues the item.
                    print(f"[pool] task died {meta['work_id']}: {type(e).__name__}: "
                          f"{str(e)[:160]}", flush=True)
                    continue
                result["item_wall_s"] = time.time() - meta["t0"]
                blob = str(result.get("violations")) + str(result.get("observation"))
                if any(sig in blob for sig in ("halted unexpectedly", "CoreHalt",
                                               "continuator has halted")):
                    # Runtime core halt: the chip is re-initialised by the next
                    # task's fresh process, but flag it loudly -- verdicts that
                    # shared this window are suspect.
                    print(f"[pool] CORE HALT on {meta['work_id']}; chip re-inits on "
                          f"next task, nearby verdicts suspect", flush=True)
                why = ""
                if not result.get("passed"):
                    v = result.get("violations") or []
                    why = f" gate={result.get('gate')} {str(v[0])[:120] if v else ''}"
                try:
                    _http_json(f"{base}/result", {
                        "lease_id": meta["lease_id"], "work_id": meta["work_id"],
                        "result": result,
                    })
                    done_count += 1
                    print(f"[pool] {meta['work_id']} w={meta['width']} "
                          f"boot={result.get('task_boot_s')}s wall={result['item_wall_s']:.1f}s "
                          f"passed={result.get('passed')}{why} ({done_count} total)", flush=True)
                except Exception as e:
                    print(f"[pool] result post failed for {meta['work_id']}: {e!r}", flush=True)

                if not banked:
                    banked = True
                    n = _cache_file_count(local_cache)
                    print(f"[cache] snapshotting {n} entries after first grade", flush=True)
                    _gsutil_rsync(local_cache, jax_cache_gcs, "save")

            if not ready and not progressed:
                time.sleep(poll_s)
    finally:
        stop.set()
    return done_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--problems", default="rg_lru", help="comma list this pool serves")
    ap.add_argument("--cases", action="append", default=[],
                    help="problem=case1,case2 (repeatable); REQUIRED in practice -- "
                         "the worker default grades declared shapes, which for splash "
                         "cannot run on one chip")
    ap.add_argument("--chips", type=int, default=4, help="TPU chips on this host")
    ap.add_argument("--width", type=int, default=1, help="default chips per test")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--compile-cache-dir", default=os.path.expanduser("~/jax-compile-cache"))
    ap.add_argument("--jax-cache-gcs", default=os.environ.get("ARENA_JAX_CACHE_GCS", ""),
                    help="GCS prefix for the baseline compile cache; keyed by chip kind")
    ap.add_argument("--timing-pairs", type=int, default=20)
    ap.add_argument("--compile-budget-s", type=float, default=90.0)
    ap.add_argument("--grade-budget-s", type=float, default=900.0)
    ap.add_argument("--ray-address", default="auto")
    ap.add_argument("--poll-s", type=float, default=1.0)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--idle-exit-s", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    # Accepted for launcher compatibility: --actors N now means "N chips".
    ap.add_argument("--actors", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    cases_by_problem = {}
    for spec in args.cases:
        prob, _, names = spec.partition("=")
        cases_by_problem[prob.strip()] = [c.strip() for c in names.split(",") if c.strip()]
    cfg = {
        "cases_by_problem": cases_by_problem,
        "cache": args.cache,
        "compile_cache_dir": args.compile_cache_dir,
        "jax_cache_gcs": args.jax_cache_gcs or None,
        "timing_pairs": args.timing_pairs,
        "compile_budget_s": args.compile_budget_s,
        "grade_budget_s": args.grade_budget_s,
        "ray_address": args.ray_address,
        "width": args.width,
        "smoke": args.smoke,
    }
    chips = args.chips if args.actors is None else max(args.chips, args.actors)
    n = run_pool(
        args.queue,
        [p.strip() for p in args.problems.split(",") if p.strip()],
        chips=chips,
        cfg=cfg,
        poll_s=args.poll_s,
        max_items=args.max_items,
        idle_exit_s=args.idle_exit_s,
    )
    print(f"[pool] exiting after {n} items")


if __name__ == "__main__":
    main()
