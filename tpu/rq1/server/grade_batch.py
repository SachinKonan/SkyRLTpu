"""Offline batch grader for RQ1 run directories. NEURONIC-ONLY (discover venv, compute node).

Consumes the collector contract (submissions.jsonl + solutions/<hash>.txt), grades every unique
program, writes result.json. Agents never saw a grader; this is the only scoring pass.

  srun ... $PY grade_batch.py --problem fc46 --run-dir runs/fc46_B [--concurrency 24]

Staging:
  cpp     stage 1 compile+1-case smoke (TTD_FCALGO_MAX_CASES=1, cheap) -> stage 2 full grade of
          the survivors. Uses a PROCESS pool: _grade mutates os.environ[TTD_FCALGO_MAX_CASES],
          which is process-global and unsafe to interleave fast/full in threads.
  python  full grade directly (invalid programs fail fast on their own; a fast pre-stage would
          mis-kill budget-honoring programs, and ud has no budget_s knob to rewrite at all).
          Thread pool (the evaluator subprocesses the candidate anyway).

Also re-grades the pack seed (--with-baseline, default on) as a determinism sanity check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "/n/fs/vision-mix/sk7524/SkyRLTpu"
AB = f"{REPO}/tpu/distill_ablation"
sys.path.insert(0, AB)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grading_mcp import _grade  # noqa: E402
from make_problem_pack import PROBLEMS  # noqa: E402

CLIENT_DATA = Path(__file__).resolve().parent.parent / "client" / "data"


def _payload(code, lang, fence):
    return f"```{fence}\n{code}\n```" if lang == "python" else code


def _job(task):
    """Top-level so ProcessPoolExecutor can pickle it."""
    (root, mod, cls, ptype, lang, constr, payload, fast, logdir) = task
    t0 = time.time()
    r = _grade(root, mod, cls, ptype, lang, constr, payload, fast, logdir)
    r["secs"] = round(time.time() - t0, 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=list(PROBLEMS))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="grade only the first K (smoke)")
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    root, mod, cls, ptype, lang, maximize = PROBLEMS[args.problem]
    d = CLIENT_DATA / args.problem
    meta = json.loads((d / "meta.json").read_text())
    fence = meta["fence"]
    constr = None
    cj = d / "seed_construction.json"
    if cj.exists():
        constr = json.loads(cj.read_text())

    run = Path(args.run_dir).resolve()
    logdir = run / "grade_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    subs = [json.loads(l) for l in (run / "submissions.jsonl").read_text().splitlines()
            if l.strip()]
    by_hash = {}
    for s in subs:
        by_hash.setdefault(s["sol_hash"], []).append(s.get("session"))
    hashes = list(by_hash)
    if args.limit:
        hashes = hashes[:args.limit]
    progs = {}
    for h in hashes:
        f = run / "solutions" / f"{h}.txt"
        if f.exists():
            progs[h] = f.read_text()
    print(f"[grade] {args.problem} {run.name}: {len(subs)} submissions, "
          f"{len(progs)} unique programs, lang={lang}", flush=True)

    results = {h: {"sol_hash": h, "sessions": by_hash[h]} for h in progs}
    Pool = ProcessPoolExecutor if lang == "cpp" else ThreadPoolExecutor

    def run_stage(items, fast, tag):
        tasks = {h: (root, mod, cls, ptype, lang, constr,
                     _payload(progs[h], lang, fence), fast, str(logdir)) for h in items}
        out = {}
        with Pool(max_workers=args.concurrency) as ex:
            futs = {ex.submit(_job, t): h for h, t in tasks.items()}
            n = 0
            for fut in as_completed(futs):
                h = futs[fut]
                try:
                    out[h] = fut.result()
                except Exception as e:
                    out[h] = {"score": None, "valid": False,
                              "detail": f"grader crashed: {e}"[:200], "secs": None}
                n += 1
                if n % 10 == 0:
                    print(f"[grade] {tag}: {n}/{len(tasks)}", flush=True)
        return out

    t0 = time.time()
    if lang == "cpp":
        fast = run_stage(list(progs), True, "check")
        for h, r in fast.items():
            results[h]["check"] = r
        survivors = [h for h, r in fast.items() if r.get("valid")]
        print(f"[grade] check: {len(survivors)}/{len(progs)} valid; full-grading survivors",
              flush=True)
        full = run_stage(survivors, False, "full")
    else:
        full = run_stage(list(progs), False, "full")
    for h, r in full.items():
        results[h]["full"] = r

    baseline = None
    if not args.no_baseline:
        seedf = d / ("seed.py" if lang == "python" else "seed.cpp")
        b = _job((root, mod, cls, ptype, lang, constr,
                  _payload(seedf.read_text(), lang, fence), False, str(logdir)))
        baseline = b.get("score")
        drift = (baseline is not None and meta.get("seed_score") is not None
                 and abs(baseline - meta["seed_score"]) > max(1e-9, abs(baseline) * 1e-3))
        print(f"[grade] baseline re-grade: {baseline} (pack says {meta.get('seed_score')})"
              + (" DRIFT!" if drift else ""), flush=True)

    scored = [(h, r["full"]["score"]) for h, r in results.items()
              if r.get("full", {}).get("valid")]
    scored.sort(key=lambda x: x[1], reverse=maximize)
    manifest = {}
    mf = run / "manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text())
    out = {"problem": args.problem, "run_dir": str(run), "cell": manifest.get("cell"),
           "model": manifest.get("model"), "maximize": maximize,
           "seed_score": meta.get("seed_score"), "baseline_regrade": baseline,
           "n_submissions": len(subs), "n_unique": len(progs),
           "n_valid": len(scored),
           "best": {"sol_hash": scored[0][0], "score": scored[0][1]} if scored else None,
           "scores": {h: s for h, s in scored},
           "results": results, "graded_secs": int(time.time() - t0),
           "graded_at": time.strftime("%F %T")}
    (run / "result.json").write_text(json.dumps(out, indent=2))
    print(f"[grade] DONE in {out['graded_secs']}s: valid {len(scored)}/{len(progs)}, "
          f"best={out['best']} -> {run}/result.json", flush=True)


if __name__ == "__main__":
    main()
