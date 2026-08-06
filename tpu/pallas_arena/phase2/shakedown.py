"""Phase-2 RMSNorm end-to-end shakedown. Runs ON the v6e-1 judge host,
inside the arena venv, with:

    JAX_PLATFORMS=cpu ARENA_CHILD_JAX_PLATFORMS=tpu \
        python pallas_arena/phase2/shakedown.py --out ~/shakedown-results.json

(the parent stays off the exclusive TPU chip; every grading child claims it).

What it measures / asserts (PLAN.md phase 2):
  * boot noise floor per chip, ref-vs-ref grade == 1.00 +/- 0.02 per case;
  * on-chip goldens: honest XLA + three real Pallas kernels PASS,
    wrapped/aliased reference AST-rejected, subtly-wrong (eps) fails
    numerics, the whole cheater battery fails on silicon;
  * determinism N=5 bitwise (inside every full grade);
  * same-kernel regrade stability (honest-xla vs honest-xla-b) +/- 3%;
  * peak HBM reported per grade; %-of-speed-of-light for the memory-bound
    RMSNorm; true per-candidate wall cost (the throughput-table input);
  * POST /grade acceptance through the real FIFO server (launch-flag lock),
    incl. an instant cache-hit regrade.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ARENA_IMPORT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ARENA_IMPORT_ROOT))

from pallas_arena.judge import grader  # noqa: E402
from pallas_arena.phase2.variants import (  # noqa: E402
    EXPECT_PASS,
    EXPECT_SLOW,
    shakedown_variants,
)

RLIMIT_GB = 512.0  # RLIMIT_AS must clear libtpu's giant VA reservations
TIMEOUT_S = 900.0
PROD_CASES = None  # None -> full declared+holdout non-smoke shape set


def _post(url: str, payload: dict, timeout: float = 1200.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/shakedown-results.json"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--skip-server", action="store_true")
    args = ap.parse_args()

    results: dict = {"invariants": {}, "grades": {}, "meta": {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    hard_fail = []

    def invariant(name: str, ok: bool, detail):
        results["invariants"][name] = {"ok": bool(ok), "detail": detail}
        print(f"[inv] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            hard_fail.append(name)

    # ---- 1. boot noise floor + ref-vs-ref invariant (test layer 3)
    t0 = time.perf_counter()
    floor_res = grader.measure_noise_floor("rmsnorm", smoke=False, timing_pairs=20, timeout_s=TIMEOUT_S)
    results["noise_floor_run"] = floor_res
    if not floor_res.get("ok"):
        invariant("noise-floor-measured", False, floor_res.get("error"))
        Path(args.out).write_text(json.dumps(results, indent=1, default=str))
        return 1
    noise_floor = floor_res["noise_floor"]
    invariant(
        "noise-floor-measured",
        True,
        {"floor": noise_floor, "wall_s": time.perf_counter() - t0, "per_case": floor_res["noise_floors"]},
    )
    ref_scores = floor_res["ref_vs_ref_scores"]
    bad = {c: s for c, s in ref_scores.items() if abs(s - 1.0) > 0.02}
    invariant("ref-vs-ref-1.00±2%", not bad, {"scores": ref_scores, "violations": bad})

    # ---- 2. grade every variant through the direct grader
    per_candidate_walls = []
    for name, code in shakedown_variants().items():
        t0 = time.perf_counter()
        r = grader.grade(
            "rmsnorm",
            code,
            mode="full",
            smoke=False,
            cases=PROD_CASES,
            noise_floor=noise_floor,
            timeout_s=TIMEOUT_S,
            rlimit_gb=RLIMIT_GB,
            cache=None,
        )
        wall = time.perf_counter() - t0
        per_candidate_walls.append(wall)
        results["grades"][name] = {
            k: r.get(k)
            for k in (
                "passed",
                "gate",
                "score",
                "reward",
                "violations",
                "noise_floor",
                "peak_hbm_bytes",
                "speed_of_light_fracs",
                "per_case",
                "holdout",
                "latencies",
                "wall_s",
                "first_call_plus_correctness_s",
                "device_kind",
                "backend",
            )
        }
        print(
            f"[grade] {name}: passed={r.get('passed')} gate={r.get('gate')} "
            f"score={r.get('score')} reward={r.get('reward')} "
            f"wall={wall:.1f}s peak_hbm={r.get('peak_hbm_bytes')}",
            flush=True,
        )
        want_pass = name in EXPECT_PASS
        if bool(r.get("passed")) != want_pass:
            invariant(
                f"verdict:{name}",
                False,
                {"expected_pass": want_pass, "got": r.get("gate"), "violations": r.get("violations")},
            )
        else:
            invariant(f"verdict:{name}", True, {"expected_pass": want_pass, "gate": r.get("gate")})
        if name in EXPECT_SLOW and r.get("passed"):
            invariant(f"slowdown-measured:{name}", (r.get("score") or 9) < 0.95, {"score": r.get("score")})

    # backend really was the TPU
    backends = {g.get("backend") for g in results["grades"].values() if g.get("backend")}
    invariant("children-on-tpu", backends == {"tpu"}, sorted(backends))
    hbm = [g["peak_hbm_bytes"] for g in results["grades"].values() if g.get("peak_hbm_bytes")]
    invariant("peak-hbm-reported", len(hbm) > 0, {"max_gb": max(hbm) / 2**30 if hbm else None})

    # ---- 3. regrade stability (independent code copies) +/- 3%
    a = results["grades"]["honest-xla"].get("score")
    b = results["grades"]["honest-xla-b"].get("score")
    if a and b:
        invariant("same-kernel-regrade-±3%", abs(a / b - 1.0) < 0.03, {"a": a, "b": b, "ratio": a / b})
    else:
        invariant("same-kernel-regrade-±3%", False, {"a": a, "b": b})

    results["per_candidate_cost_s"] = {
        "mean": sum(per_candidate_walls) / len(per_candidate_walls),
        "min": min(per_candidate_walls),
        "max": max(per_candidate_walls),
        "all": per_candidate_walls,
    }

    # ---- 4. POST /grade acceptance through the real FIFO server
    if not args.skip_server:
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["ARENA_CHILD_JAX_PLATFORMS"] = "tpu"
        env["ARENA_RLIMIT_GB"] = str(int(RLIMIT_GB))
        env["PYTHONPATH"] = str(ARENA_IMPORT_ROOT)
        cache_dir = os.path.expanduser("~/arena-reward-cache")
        srv = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pallas_arena.judge.server",
                "--problem",
                "rmsnorm",
                "--port",
                str(args.port),
                "--host",
                "127.0.0.1",
                "--workers",
                "1",
                "--cache",
                cache_dir,
            ],
            env=env,
            stdout=open(os.path.expanduser("~/judge-server.log"), "wb"),
            stderr=subprocess.STDOUT,
        )
        base = f"http://127.0.0.1:{args.port}"
        try:
            for _ in range(600):  # boot incl. its own noise-floor pass
                try:
                    h = json.loads(urllib.request.urlopen(base + "/healthz", timeout=5).read())
                    if h.get("ok"):
                        break
                except Exception:
                    time.sleep(2)
            else:
                raise RuntimeError("judge server never became healthy")
            invariant("server-boot-floor", h["noise_floors"][0] is not None, h["noise_floors"])

            wrong = _post(base + "/grade", {"problem": "splash_attention", "code": "x"})
            invariant("server-problem-lock", False, wrong)  # must 400 above
        except urllib.error.HTTPError as e:
            invariant("server-problem-lock", e.code == 400, e.code)
        except Exception as e:
            invariant("server-http", False, repr(e))
        try:
            t0 = time.perf_counter()
            r1 = _post(
                base + "/grade",
                {"problem": "rmsnorm", "code": shakedown_variants()["honest-xla"], "timeout_s": TIMEOUT_S},
            )
            first = time.perf_counter() - t0
            t0 = time.perf_counter()
            r2 = _post(
                base + "/grade",
                {"problem": "rmsnorm", "code": shakedown_variants()["honest-xla"], "timeout_s": TIMEOUT_S},
            )
            second = time.perf_counter() - t0
            invariant("server-grade-passes", bool(r1.get("passed")), {"reward": r1.get("reward"), "wall_s": first})
            invariant(
                "server-cache-hit-instant",
                bool(r2.get("cache_hit")) and second < 5.0,
                {"cache_hit": r2.get("cache_hit"), "wall_s": second},
            )
            invariant(
                "server-cache-consistent",
                r2.get("reward") == r1.get("reward"),
                {"r1": r1.get("reward"), "r2": r2.get("reward")},
            )
            results["server"] = {"first_wall_s": first, "cached_wall_s": second}
        except Exception as e:
            invariant("server-grade", False, repr(e))
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=10)
            except subprocess.TimeoutExpired:
                srv.kill()

    results["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["hard_failures"] = hard_fail
    Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    print(f"[shakedown] done; {len(hard_fail)} hard failures -> {args.out}", flush=True)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
