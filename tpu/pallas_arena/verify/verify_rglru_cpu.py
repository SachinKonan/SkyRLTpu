"""Adversarial CPU verification of the arena's first above-1.0 candidate.

The seam probe (job 3651278) scored `gemma4-31b | seam | rg_lru` at 1.1941 --
the first candidate in this arena to beat its baseline. The probe ran with
`--no-adversarial`, so that kernel has NEVER faced the adversarial vector
library. Everything in this file is correctness, so it is CPU work:

  1. the full judge path (`mode="full"`, smoke shapes, adversarial ON) on the
     exact composed program the judge graded;
  2. the adversarial vectors re-run by hand at the PRODUCTION width (d=2560)
     and at a T long enough to cross several chunk boundaries -- the library's
     own `tiny` case is t=64, which with CHUNK=256 is a SINGLE chunk, so the
     library as configured cannot see the carry the seam exists to test;
  3. exactness checks the tolerance machinery does not make: a==0 passthrough,
     reset rows, finiteness at a -> 1-1e-6;
  4. the non-divisible holdout T=1500 and a T that is prime to the chunk.

Usage:
  JAX_PLATFORMS=cpu python -m pallas_arena.verify.verify_rglru_cpu \
      --results runs/pallas_arena/seam-results-3651278.jsonl --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from pallas_arena.judge import grader
from pallas_arena.judge.problems import get_problem
from pallas_arena.judge.problems.base import ShapeCase, check_tolerance, error_stats


def load_candidates(path: str, task: str, min_reward: float) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    sel = [r for r in rows if r["task"] == task and (r.get("reward") or 0.0) >= min_reward]
    sel.sort(key=lambda r: -(r.get("reward") or 0.0))
    return sel


def _exec_kernel(code: str):
    ns: dict = {}
    exec(compile(code, "<candidate>", "exec"), ns)  # noqa: S102 -- our own cached text
    return ns["kernel"], ns


def _err(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    denom = np.maximum(np.abs(b), 1e-30)
    return float(np.max(np.abs(a - b))), float(np.max(np.abs(a - b) / denom))


def chunk_structure(ns, t: int) -> dict:
    """How many chunks does this candidate's own CHUNK produce at length t?"""
    ch = int(ns.get("CHUNK", 0)) if "CHUNK" in ns else None
    if not ch:
        return {"CHUNK": None}
    eff = max(1, min(ch, t))
    return {"CHUNK": ch, "effective_chunk": eff, "n_chunks": -(-t // eff)}


def hand_adversarial(kernel, problem, b: int, t: int, d: int, seed: int = 0) -> list[dict]:
    """The three library vectors, at an arbitrary (here: production) shape."""
    case = ShapeCase(f"hand-{b}x{t}x{d}", {"b": b, "t": t, "d": d})
    key = jax.random.PRNGKey(seed)
    x, a, reset = problem.make_inputs(key, case)
    out = []

    def run(label, xx, aa, rr, check):
        ref = problem.reference(xx, aa, rr)
        got = kernel(xx, aa, rr)
        jax.block_until_ready(got)
        stats = error_stats(got, ref)
        tol = problem.calibrated_tolerance((xx, aa, rr), ref)
        ok, why = check_tolerance(stats, tol)
        amax, rmax = _err(got, ref)
        rec = {
            "vector": label,
            "shape": f"{b}x{t}x{d}",
            "judge_verdict": "PASS" if ok else f"FAIL: {why}",
            "max_abs_err": amax,
            "max_rel_err": rmax,
            "tol_max": float(tol["max"]),
            "finite": bool(np.isfinite(np.asarray(got, np.float64)).all()),
        }
        try:
            check(got, (xx, aa, rr))
            rec["invariant"] = "OK"
        except AssertionError as e:
            rec["invariant"] = f"FAIL: {str(e)[:300]}"
        out.append(rec)
        return rec

    # a -> 1 - 1e-6: near-perfect memory; the fp32-drift vector
    run(
        "a-to-one-long-memory",
        x,
        jnp.full_like(a, 1.0 - 1e-6),
        reset,
        lambda h, ins: np.testing.assert_(np.isfinite(np.asarray(h, np.float64)).all()),
    )
    # dense reset boundaries every 7 steps
    rr = (jnp.arange(t) % 7 == 0)[None, :].repeat(b, axis=0)

    def _reset_rows(h, ins):
        rmask = np.asarray(ins[2])
        xx = np.asarray(ins[0], np.float32)
        np.testing.assert_allclose(np.asarray(h)[rmask], xx[rmask], rtol=1e-5, atol=1e-5)

    run("dense-reset-boundaries", x, a, rr, _reset_rows)

    # a == 0 -> h == x EXACTLY
    def _passthrough(h, ins):
        xx = np.asarray(ins[0], np.float32)
        np.testing.assert_allclose(np.asarray(h), xx, rtol=1e-5, atol=1e-5)

    run("a-zero-passthrough", x, jnp.zeros_like(a), reset, _passthrough)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/pallas_arena/seam-results-3651278.jsonl")
    ap.add_argument("--task", default="rg_lru")
    ap.add_argument("--min-reward", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--full-grade", action="store_true", help="run grader.grade(mode=full, smoke) too")
    args = ap.parse_args()

    problem = get_problem(args.task)
    cands = load_candidates(args.results, args.task, args.min_reward)
    print(f"backend={jax.default_backend()}  candidates>={args.min_reward}: {len(cands)}")

    report = {"backend": jax.default_backend(), "candidates": []}

    # ---- structural fact about the library itself, independent of candidates
    lib_case = problem.case_by_name(problem.adversarial_case_name)
    report["adversarial_library_case"] = {
        "name": lib_case.name,
        "dims": dict(lib_case.dims),
    }
    print(f"[library] adversarial vectors are built on case {lib_case.name} {dict(lib_case.dims)}")

    for r in cands:
        tag = f"{r['model']}|{r['variant']}|idx{r['idx']}|reward={r['reward']:.4f}"
        print(f"\n================ {tag}")
        rec = {
            "model": r["model"],
            "variant": r["variant"],
            "idx": r["idx"],
            "probe_reward": r["reward"],
            "code_sha": r.get("code_sha"),
            "observation": r.get("observation"),
        }
        code = r["code"]
        try:
            kernel, ns = _exec_kernel(code)
        except Exception as e:
            rec["exec_error"] = f"{type(e).__name__}: {e}"
            report["candidates"].append(rec)
            continue

        rec["chunk_at_library_shape"] = chunk_structure(ns, lib_case.dims["t"])
        rec["chunk_at_probe_2048"] = chunk_structure(ns, 2048)
        rec["chunk_at_holdout_1500"] = chunk_structure(ns, 1500)
        print(f"  chunking: library-shape {rec['chunk_at_library_shape']}  probe2048 {rec['chunk_at_probe_2048']}")

        # ---- 1. adversarial vectors at PRODUCTION width and multi-chunk T
        rec["adversarial_production"] = hand_adversarial(kernel, problem, b=2, t=1024, d=2560)
        # ---- 1b. and at the library's own tiny shape, for the comparison
        rec["adversarial_tiny"] = hand_adversarial(
            kernel, problem, b=lib_case.dims["b"], t=lib_case.dims["t"], d=lib_case.dims["d"]
        )
        for a in rec["adversarial_production"] + rec["adversarial_tiny"]:
            print(f"  adv {a['shape']:14s} {a['vector']:24s} absErr={a['max_abs_err']:.3e} {a['invariant']}")

        # ---- 2. plain correctness at the declared + holdout shapes
        shapes = [
            ("probe-4x2048x2560", 4, 2048, 2560),
            ("probe-2x1024x2560", 2, 1024, 2560),
            ("probe-holdout-2x1500x2560", 2, 1500, 2560),
            ("prime-T-1x1021x2560", 1, 1021, 2560),
            ("T-eq-1", 1, 1, 2560),
        ]
        rec["correctness"] = []
        for name, b, t, d in shapes:
            case = ShapeCase(name, {"b": b, "t": t, "d": d})
            inputs = problem.make_inputs(jax.random.PRNGKey(7), case)
            ref = problem.reference(*inputs)
            try:
                got = kernel(*inputs)
                jax.block_until_ready(got)
            except Exception as e:
                rec["correctness"].append({"case": name, "error": f"{type(e).__name__}: {e}"})
                print(f"  corr {name:28s} ERROR {type(e).__name__}: {e}")
                continue
            amax, rmax = _err(got, ref)
            tol = problem.calibrated_tolerance(inputs, ref)
            ok, why = check_tolerance(error_stats(got, ref), tol)
            rec["correctness"].append(
                {"case": name, "max_abs_err": amax, "max_rel_err": rmax,
                 "tol_max": float(tol["max"]), "tol_q99": float(tol["q99"]),
                 "judge_verdict": "PASS" if ok else f"FAIL: {why}"}
            )
            print(f"  corr {name:28s} absErr={amax:.3e} tol_max={tol['max']:.3e} -> {'PASS' if ok else why}")

        # ---- 3. determinism, CPU, N=8 bitwise
        case = ShapeCase("det", {"b": 2, "t": 1024, "d": 2560})
        inputs = problem.make_inputs(jax.random.PRNGKey(3), case)
        outs = [np.asarray(jax.block_until_ready(kernel(*inputs))) for _ in range(8)]
        bitwise = all(np.array_equal(outs[0], o) for o in outs[1:])
        rec["cpu_determinism_n8_bitwise"] = bool(bitwise)
        print(f"  determinism (CPU, N=8): {'BITWISE IDENTICAL' if bitwise else 'DIVERGED'}")

        report["candidates"].append(rec)

    # ---- 4. optional: the real judge path, smoke shapes, adversarial ON
    if args.full_grade and cands:
        r = cands[0]
        wd = tempfile.mkdtemp(prefix="verify-rglru-")
        res = grader.grade(
            args.task, r["code"], mode="full", smoke=True, timeout_s=900.0,
            timing_pairs=3, determinism_runs=5, workdir=wd,
            child_env={"JAX_PLATFORMS": "cpu"},
        )
        report["full_grade_smoke_cpu"] = {
            k: res.get(k) for k in ("passed", "gate", "reward", "score", "violations", "per_case")
        }
        print("\n[full grade, smoke, CPU, adversarial ON]", json.dumps(report["full_grade_smoke_cpu"], default=str))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1, default=str)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
