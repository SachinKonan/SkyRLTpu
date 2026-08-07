"""CPU proof that the proposed prompts are SATISFIABLE.

A prompt fix is only a fix if a plausible model answer under it actually clears
the gate. Each fill in `proposed.py` is pushed through the identical path a
generation takes -- compose with the real scaffold, then the SAME sandbox AOT
export child the judge runs as step 1 -- at every declared probe shape
including the non-divisible holdout. The narrowed RPA seam is additionally
checked for numerics under `interpret=True` against the problem's own fp32
reference, because a scaffold change can be a correctness change.

Usage: JAX_PLATFORMS=cpu python -m pallas_arena.verify.prove_fixes --out x.json
"""

from __future__ import annotations

import argparse
import json
import sys

from pallas_arena.probe import configs, pregate, seam
from pallas_arena.verify import proposed


def _pregate(task, code, sigs, timeout_s):
    res = pregate.pregate_one(task, code, sigs, timeout_s=timeout_s)
    return {
        "passed": bool(res.get("passed")),
        "gate": res.get("gate"),
        "observation": (res.get("observation") or "").replace("\n", " | ")[:300],
        "violations": (res.get("violations") or [])[:2],
    }


def numerics(task: str, code: str, cases: list[str]) -> dict:
    """Interpreted CPU numerics for a composed program, vs the fp32 reference."""
    import jax

    from pallas_arena.judge.problems import get_problem
    from pallas_arena.judge.problems.base import check_tolerance, error_stats

    problem = get_problem(task)
    ns: dict = {}
    src = code.replace("_pl.pallas_call(", "_pl.pallas_call(interpret=True, ") if "_pl.pallas_call(" in code else code
    try:
        exec(compile(src, "<proposed>", "exec"), ns)  # noqa: S102
    except Exception as e:
        return {"error": f"exec: {type(e).__name__}: {e}"}
    kernel = ns["kernel"]
    out = {}
    for name in cases:
        case = problem.case_by_name(name)
        try:
            inputs = problem.make_inputs(jax.random.PRNGKey(11), case)
            ref = problem.reference(*inputs)
            got = kernel(*inputs)
            jax.block_until_ready(got)
            ok, why = check_tolerance(error_stats(got, ref), problem.calibrated_tolerance(inputs, ref))
            out[name] = {"verdict": "PASS" if ok else f"FAIL: {why}"}
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:250]}"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--numerics-case", default="tiny", help="smoke case for the interpreted check")
    ap.add_argument("--skip-numerics", action="store_true")
    args = ap.parse_args()

    report = {"proposed_fills": [], "narrowed_seams": []}

    # ---- 1. plausible model answers under the improved prompt, current seams
    for task, fills in proposed.PROPOSED_FILLS.items():
        sigs = pregate.probe_signatures(task, configs.TASK_CASES[task])
        for label, fill in fills:
            code = seam.compose(task, fill)
            r = _pregate(task, code, sigs, args.timeout_s)
            r.update(task=task, label=label, fill_chars=len(fill),
                     fill_approx_tokens=len(fill) // 4)
            report["proposed_fills"].append(r)
            print(f"[proposed] {task:24s} {label:32s} "
                  f"{'EXPORTS' if r['passed'] else 'FAIL ' + str(r['gate'])}  {r['observation'][:150]}")

    # ---- 2. the narrowed seams: a NEW scaffold, so compose by hand
    for task, (required, scaffold, fill) in proposed.NARROWED.items():
        sigs = pregate.probe_signatures(task, configs.TASK_CASES[task])
        code = fill.rstrip() + "\n\n" + scaffold.strip() + "\n"
        assert f"def {required}" in fill, f"{task}: the narrowed fill must define {required}"
        r = _pregate(task, code, sigs, args.timeout_s)
        r.update(task=task, label=f"narrowed-seam({required})", fill_chars=len(fill),
                 fill_approx_tokens=len(fill) // 4)
        print(f"[narrowed] {task:24s} {required:32s} "
              f"{'EXPORTS' if r['passed'] else 'FAIL ' + str(r['gate'])}  {r['observation'][:200]}")
        if not args.skip_numerics:
            r["numerics_interpreted"] = numerics(task, code, [args.numerics_case])
            print(f"[narrowed] {task}: interpreted numerics {r['numerics_interpreted']}")
        report["narrowed_seams"].append(r)

    n_ok = sum(1 for r in report["proposed_fills"] + report["narrowed_seams"] if r["passed"])
    n = len(report["proposed_fills"]) + len(report["narrowed_seams"])
    print(f"\n=== {n_ok}/{n} proposed answers clear the CPU pre-gate ===")
    report["summary"] = {"passed": n_ok, "total": n}
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
