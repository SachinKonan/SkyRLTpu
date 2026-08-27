"""Ray grading pool: PER-TEST dispatch over the host's chips.

THE UNIT OF SCHEDULING IS (candidate, shape case), not the candidate:

    llm program -> cheap error check (CPU, no chips)
                -> one Ray task PER TEST, resources={"TPU": 4 if tp4 else 1}
                -> each task compares fwd AND bwd against ground truth on its
                   own chip(s), runs the 20 counterbalanced pairs, reports
                   its medians
                -> the collector applies the global rules (correct-everywhere,
                   geomean over all fwd+bwd factors, per-case noise-floor
                   gating) and posts ONE verdict to the queue.

Why per-test:
  * A candidate's wall time becomes max-over-tests instead of sum: each test
    is its own Mosaic compile (the measured dominant cost), so tests of one
    candidate parallelise across chips.
  * TP composes with throughput: a tp4 test holds 4 chips only while it
    runs; single-chip tests pack around it. Per-candidate width forced the
    whole grade to the widest case.
  * Failure blast radius is one test: a core-halting shape no longer takes
    the whole grade's evidence with it (though it still zeroes the verdict
    -- a halt is the candidate's fault).

Fresh process per task (max_calls=1) stays load-bearing: JAX/libtpu bind to
TPU_VISIBLE_CHIPS at process init, so reuse could carry a stale pin (silent
co-tenancy -- ratio rewards make wrong numbers look plausible), and a halt
poisons at most its own process. Affordable because the GROUND-TRUTH COMPILE
CACHE is restored from GCS; every verdict carries per-case task_boot_s so
the overhead stays measured, not assumed.

STAGE 0 runs in the driver on CPU (grader child, JAX_PLATFORMS=cpu): a
syntax/dialect-broken candidate costs ~1.5 s and ZERO chip tasks.

INGRESS STAYS HTTP: the queue keeps lease/requeue semantics (1699 items,
zero lost). Ray is only the intra-host scheduler.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request

from pallas_arena.judge import collect

MAX_PENDING_FACTOR = 1   # never queue more tasks than chips: a dying
                         # process must release its chip before the next
                         # task lands on it (device-busy class)


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


_TP_BOUNDS = {1: "1,1,1", 2: "2,1,1", 4: "2,2,1", 8: "4,2,1"}


def _pin_chips(width: int = 1) -> str:
    """Bind this task's process to ``width`` of the chips Ray assigned it.

    Ray publishes the assignment via the runtime context; whether it also
    exports TPU_VISIBLE_CHIPS varies by version (measured on ray 2.58: it
    did NOT). libtpu reads the variable at init, so this runs before jax is
    imported. Only plain chip indices are usable; anything else would break
    init, which is worse than not pinning.

    A TP task RESERVES the whole host (slice-level libtpu init cannot share a
    host with single-chip processes) but must still be pinned to exactly the
    ``width`` chips its case declares, with a matching process-bounds mesh --
    otherwise a tp4 case would come up on 8 chips and shard the wrong way.
    """
    try:
        import ray

        ids = (ray.get_runtime_context().get_accelerator_ids() or {}).get("TPU") or []
        idx = [str(int(i)) for i in ids if str(i).strip().lstrip("-").isdigit()]
        idx = idx[:width] if width and len(idx) >= width else idx
        if idx and not os.environ.get("TPU_VISIBLE_CHIPS"):
            os.environ["TPU_VISIBLE_CHIPS"] = ",".join(idx)
            os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
            os.environ.setdefault(
                "TPU_CHIPS_PER_PROCESS_BOUNDS", _TP_BOUNDS.get(len(idx), "1,1,1"))
            return os.environ["TPU_VISIBLE_CHIPS"]
    except Exception:
        pass
    return os.environ.get("TPU_VISIBLE_CHIPS", "?")


def grade_case(problem: str, case: str, payload: dict, cfg: dict) -> dict:
    """ONE test: grade the candidate at ONE shape case, fwd + bwd, in this
    task's own fresh process on its own chip(s)."""
    # The driver runs with JAX_PLATFORMS=cpu (stage 0); tasks must not
    # inherit that even if the runtime propagates env.
    if os.environ.get("JAX_PLATFORMS") == "cpu":
        del os.environ["JAX_PLATFORMS"]
    from pallas_arena.judge import collect as _collect

    pinned = _pin_chips(_collect.case_width(case))
    cache_dir = cfg.get("compile_cache_dir")
    if cache_dir:
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    os.environ.setdefault("ARENA_CHILD_JAX_PLATFORMS", "tpu")

    from pallas_arena.judge import cache as cache_mod
    from pallas_arena.judge.worker import PersistentWorker

    t0 = time.time()
    rc = cache_mod.RewardCache(cfg["cache"]) if cfg.get("cache") else None
    # DEVICE-BUSY RETRY: after a driver restart (or a fast chip turnover) the
    # previous task's process may still hold libtpu while dying -- measured:
    # 5 of 8 tasks died at jax xla_bridge init with device-busy, and their
    # exclusion left a 3-case verdict with thrashed timings. Give the chip up
    # to a minute to free before failing the task.
    import jax  # noqa: F401 -- imported HERE so init happens under the retry
    last = None
    for attempt in range(4):
        try:
            import jax as _j
            _j.local_devices()
            last = None
            break
        except Exception as e:
            last = e
            print(f"[task {case}] device init attempt {attempt + 1} failed "
                  f"({str(e)[:80]}); retrying in 15s", flush=True)
            time.sleep(15)
    if last is not None:
        raise RuntimeError(f"device never freed: {last}")
    worker = PersistentWorker(
        problem,
        cases=[case],
        smoke=cfg.get("smoke", False),
        timing_pairs=cfg.get("timing_pairs", 20),
        compile_budget_s=cfg.get("compile_budget_s", 90.0),
        grade_budget_s=cfg.get("grade_budget_s", 900.0),
        cache=rc,
        worker_id=f"ray-{case}-{pinned}",
    )
    boot = worker.boot()
    t_boot = time.time() - t0
    result = worker.grade_code(payload.get("code", ""), tag=payload.get("tag"))
    result["task_boot_s"] = round(t_boot, 1)
    result["task_chips"] = pinned
    result["task_noise_floor"] = boot.get("noise_floor")
    result["task_case"] = case
    return result


def stage0_pregate(problem: str, code: str) -> dict:
    """Cheap error check, CPU only: AST/dialect/trace gates via the grader
    child with JAX_PLATFORMS=cpu. A broken candidate never costs a chip."""
    from pallas_arena.judge import grader

    return grader.grade(
        problem, code, mode="pregate", timeout_s=120.0,
        child_env={"JAX_PLATFORMS": "cpu", "PALLAS_INTERPRET": "1"},
    )


def cpu_per_task() -> int:
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
    except Exception as e:  # cache plumbing must never break grading
        print(f"[cache] {label} failed: {type(e).__name__}: {e}", flush=True)


def _cache_file_count(path: str) -> int:
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += len(files)
    return n


class _Candidate:
    def __init__(self, item: dict, problem: str, cases: list[str]):
        self.work_id = item["work_id"]
        self.lease_id = item["lease_id"]
        self.payload = item.get("payload") or {}
        self.problem = problem
        self.cases = cases
        self.t0 = time.time()
        self.pending: dict = {}        # ObjectRef -> case name
        self.entries: dict = {}        # case -> collect entry
        self.done = False


def run_pool(
    queue_url: str,
    problems: list[str],
    *,
    chips: int,
    cfg: dict,
    poll_s: float = 1.0,
    max_items: int | None = None,
    idle_exit_s: float | None = None,
    max_tp_width: int = 4,
) -> int:
    import ray

    base = queue_url.rstrip("/")
    ray.init(address=cfg.get("ray_address", "auto"), ignore_reinit_error=True)

    # GROUND-TRUTH COMPILE CACHE: restore before any task; snapshot after the
    # first completed candidate (baseline entries only, no throwaways).
    jax_cache_gcs = cfg.get("jax_cache_gcs")
    local_cache = cfg.get("compile_cache_dir") or os.path.expanduser("~/jax-compile-cache")
    os.makedirs(local_cache, exist_ok=True)
    cfg = {**cfg, "compile_cache_dir": local_cache}
    if jax_cache_gcs:
        _gsutil_rsync(jax_cache_gcs, local_cache, "restore")
        print(f"[cache] {_cache_file_count(local_cache)} entries after restore", flush=True)
    banked = not bool(jax_cache_gcs)

    cases_by_problem = cfg.get("cases_by_problem") or {}
    cpus = cpu_per_task()
    grade = ray.remote(num_cpus=cpus, max_calls=1)(grade_case)
    print(f"[pool] per-test dispatch: {chips} chips, {cpus} cpus/task, "
          f"max tp width {max_tp_width}, problems {problems}", flush=True)
    for prob in problems:
        print(f"[pool]   {prob}: cases {cases_by_problem.get(prob) or '(worker default!)'}",
              flush=True)

    cands: list[_Candidate] = []
    stop = threading.Event()

    def beat():
        while not stop.wait(2.0):
            for cand in list(cands):
                if not cand.done:
                    try:
                        _http_json(f"{base}/heartbeat", {"lease_id": cand.lease_id})
                    except Exception:
                        pass

    threading.Thread(target=beat, daemon=True).start()

    done_count = 0
    last_work = time.time()

    def finish(cand: _Candidate):
        nonlocal done_count, banked
        merged = collect.merge_case_results(
            cand.problem, cand.entries,
            general_mode=True, has_bwd=cand.problem in ("rg_lru", "splash_attention"),
            default_floor=0.05,
        )
        merged["item_wall_s"] = round(time.time() - cand.t0, 1)
        merged["tag"] = cand.payload.get("tag")
        try:
            from pallas_arena.judge.observation import attach_observation

            attach_observation(merged)
        except Exception:
            pass
        try:
            _http_json(f"{base}/result", {
                "lease_id": cand.lease_id, "work_id": cand.work_id, "result": merged,
            })
            done_count += 1
            why = "" if merged.get("passed") else (
                f" gate={merged.get('gate')} {str((merged.get('violations') or ['?'])[0])[:100]}")
            print(f"[pool] {cand.work_id} done in {merged['item_wall_s']}s "
                  f"passed={merged.get('passed')} reward={merged.get('reward_with_bwd')}"
                  f"{why} boots={merged.get('case_boot_s')} ({done_count} total)", flush=True)
        except Exception as e:
            print(f"[pool] result post failed for {cand.work_id}: {e!r}", flush=True)
        cand.done = True
        if not banked:
            banked = True
            n = _cache_file_count(local_cache)
            print(f"[cache] snapshotting {n} entries after first candidate", flush=True)
            _gsutil_rsync(local_cache, jax_cache_gcs, "save")

    try:
        while max_items is None or done_count < max_items:
            cands = [c for c in cands if not c.done]
            n_pending = sum(len(c.pending) for c in cands)
            if (idle_exit_s is not None and not n_pending
                    and time.time() - last_work > idle_exit_s):
                print(f"[pool] idle {idle_exit_s:.0f}s with an empty queue; exiting", flush=True)
                break

            # 1. lease new candidates while the task backlog is bounded
            while (n_pending < chips * MAX_PENDING_FACTOR
                   and (max_items is None or done_count + len(cands) < max_items)):
                try:
                    item = _http_json(f"{base}/work?worker_id=ray-pool")
                except Exception:
                    item = None
                if not item:
                    break
                payload = item.get("payload") or {}
                problem = payload.get("problem") or problems[0]
                if problem not in problems:
                    # MULTI-POOL semantic: two pool drivers may share one queue
                    # (e.g. a per-problem driver added beside the fleet's).
                    # Faulting a foreign item would destroy work another pool
                    # can grade -- drop the lease silently instead; it expires
                    # (240 s) and requeues for the pool that serves it.
                    print(f"[pool] problem={problem} not served here; releasing "
                          f"{item['work_id']} for another pool", flush=True)
                    last_work = time.time()
                    continue
                cases = cases_by_problem.get(problem)
                cand = _Candidate(item, problem, cases or [])
                cands.append(cand)
                last_work = time.time()

                # STAGE 0: cheap error check, CPU, before any chip task.
                try:
                    pre = stage0_pregate(problem, payload.get("code", ""))
                except Exception as e:
                    pre = {"passed": True, "note": f"pregate driver error: {e!r}"}
                if pre.get("passed") is False:
                    cand.entries["__pregate__"] = {"fatal": (
                        "pregate", str((pre.get("violations") or ["pregate failed"])[0])[:300])}
                    finish(cand)
                    continue
                if not cases:
                    cand.entries["__cases__"] = {"fatal": (
                        "judge_fault", f"no case list configured for {problem}")}
                    finish(cand)
                    continue

                # FAN OUT: one task per test, width from the case name.
                for case in cases:
                    w = collect.case_width(case)
                    if w > min(chips, max_tp_width):
                        cand.entries[case] = {"skipped":
                                              f"needs {w} chips (host {chips}, max tp {max_tp_width})"}
                        continue
                    # A MULTI-CHIP TASK MUST OWN THE HOST. Chips are isolated
                    # for single-chip processes (one vfio device each), but a
                    # TP task opens a SLICE-level libtpu session, and that
                    # cannot be built while other processes hold the host's
                    # other chips: measured 2026-08-27, every tp4 case died
                    # with "Cancel TPU slice due to HAL init error" /
                    # TPU_RET while single-chip cases ran beside them, and a
                    # tp4 case that ran alone elected its baseline fine.
                    # Reserving all chips makes Ray serialize it against
                    # every sibling; TPU_VISIBLE_CHIPS still pins it to w.
                    ask = chips if w > 1 else 1
                    ref = grade.options(resources={"TPU": ask}).remote(
                        problem, case, payload, cfg)
                    cand.pending[ref] = case
                n_pending = sum(len(c.pending) for c in cands)
                if not cand.pending:
                    finish(cand)

            # 2. collect finished tests
            all_refs = [r for c in cands for r in c.pending]
            if not all_refs:
                time.sleep(poll_s)
                continue
            ready, _ = ray.wait(all_refs, num_returns=1, timeout=poll_s)
            for ref in ready:
                cand = next(c for c in cands if ref in c.pending)
                case = cand.pending.pop(ref)
                try:
                    result = ray.get(ref)
                    cand.entries[case] = {"result": result}
                    fatal_now = (not result.get("passed")
                                 and result.get("gate") not in collect.JUDGE_FAULT_GATES
                                 and case not in (result.get("skipped_tp") or {}))
                except Exception as e:
                    kind = collect.classify_task_error(f"{type(e).__name__}: {e}")
                    if kind == "fatal":
                        cand.entries[case] = {"fatal": ("runtime_halt",
                                                        f"task died: {str(e)[:200]}")}
                        fatal_now = True
                    else:
                        cand.entries[case] = {"judge_fault": f"task died: {str(e)[:200]}"}
                        fatal_now = False
                if fatal_now and cand.pending:
                    # correct-everywhere: the candidate is already zeroed --
                    # cancel its remaining tests instead of burning chips.
                    print(f"[pool] {cand.work_id}: {case} is candidate-fatal; "
                          f"cancelling {len(cand.pending)} sibling tests", flush=True)
                    for sib in list(cand.pending):
                        try:
                            ray.cancel(sib, force=True)
                        except Exception:
                            pass
                        cand.entries[cand.pending.pop(sib)] = {
                            "judge_fault": "cancelled after sibling candidate-fault"}
                if not cand.pending:
                    finish(cand)
    finally:
        stop.set()
    return done_count


def main() -> None:
    # The DRIVER never touches a chip: stage-0 pregates and problem-metadata
    # imports run on CPU jax. Tasks re-enable TPU for themselves.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--problems", default="rg_lru", help="comma list this pool serves")
    ap.add_argument("--cases", action="append", default=[],
                    help="problem=case1,case2 (repeatable); REQUIRED in practice -- "
                         "the worker default grades declared shapes, which for splash "
                         "cannot run on one chip")
    ap.add_argument("--chips", type=int, default=4, help="TPU chips on this host")
    ap.add_argument("--max-tp-width", type=int, default=4,
                    help="largest tp case width to run (tp8 excluded by default)")
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
    # accepted for launcher compatibility; --actors N historically meant chips
    ap.add_argument("--actors", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--width", type=int, default=None, help=argparse.SUPPRESS)
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
        max_tp_width=args.max_tp_width,
    )
    print(f"[pool] exiting after {n} items")


if __name__ == "__main__":
    main()
