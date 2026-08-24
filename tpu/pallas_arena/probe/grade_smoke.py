"""Offline CPU grading for the rf3 smoke: validity + correctness, no TPU.

Per completion: extract the program, run the AOT pre-gate (AST incl. the
pallas_call requirement, poison stubs, export at every declared shape --
the exact validity the judge enforces), then check correctness against the
reference at the tiny smoke cases. Reports per-cell:

  validity %          (pre-gate pass rate -- gate a)
  gate histogram      (where the failures die)
  correctness %       (of valid programs, tiny-case tolerance pass -- gate b)
  truncation %        (finish_reason != stop: the thinking-budget tax)
  distinct programs   (spread proxy: 32 identical kernels have no ranking
                       gradient regardless of validity -- the tailored-prompt
                       lesson)
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, "tpu")
    import os

    os.environ["PALLAS_INTERPRET"] = "1"  # scaffold candidates: CPU interpret
    os.environ["ARENA_RLIMIT_GB"] = "64"  # Mosaic exports abort under the 16GB default
    import jax

    from pallas_arena.judge import grader
    from pallas_arena.judge.problems import get_problem
    from pallas_arena.judge.problems.base import check_tolerance, error_stats
    from pallas_arena.judge.worker import build_signatures
    from pallas_arena.probe.repair_driver import extract_program

    # Validity = the judge's REAL gate: jax.export targeting TPU from a CPU
    # child under the device-kind shim. NOT the CPU-backend pregate, which
    # SIGABRTs on any real pallas_call -- with it, every VALID pallas
    # candidate would have been scored as a crash and the validity metric
    # would have been garbage.
    _sig_cache: dict = {}

    def validity(task: str, program: str):
        if task not in _sig_cache:
            pr = get_problem(task)
            scored = [c for c in pr.shape_cases() if c.smoke and not c.holdout]
            hold = [c for c in pr.shape_cases() if c.smoke and c.holdout]
            _sig_cache[task] = build_signatures(pr, scored, hold, [])[0]
        r = grader.grade(
            task, program, mode="aot_export", smoke=True,
            export_signatures=_sig_cache[task], export_platforms=["tpu"],
            child_env={"JAX_PLATFORMS": "cpu", "PALLAS_INTERPRET": "0"},
        )
        why = "; ".join(str(v) for v in (r.get("violations") or [])) or r.get("gate", "?")
        return bool(r.get("passed")), why

    cells: dict = collections.defaultdict(lambda: {
        "n": 0, "truncated": 0, "gen_error": 0, "no_program": 0,
        "valid": 0, "correct": 0, "gates": collections.Counter(),
        "program_hashes": set(), "rows": [],
    })

    for line in open(args.gens):
        row = json.loads(line)
        cell = cells[(row["task"], row["variant"])]
        cell["n"] += 1
        verdict = {"idx": row["idx"]}
        if "error" in row:
            cell["gen_error"] += 1
            verdict["outcome"] = f"gen_error: {row['error'][:80]}"
            cell["rows"].append(verdict)
            continue
        if row.get("finish_reason") != "stop":
            cell["truncated"] += 1
        text = row.get("text") or ""
        # Shared with the generator (incl. the repair round's re-extraction):
        # forcing-cue-first, then last kernel-bearing fenced block.
        from pallas_arena.probe.gen_smoke import extract_completion
        program = extract_completion(text) or extract_program(text)
        if not program:
            cell["no_program"] += 1
            verdict["outcome"] = "no_program"
            cell["rows"].append(verdict)
            continue
        cell["program_hashes"].add(hashlib.sha256(program.encode()).hexdigest()[:12])

        if str(row.get("variant", "")).startswith("rf3c"):
            # FIXED-OUTPUT CONTRACT: the block is defs-only; assemble the
            # scaffold machinery before grading. A ContractError is the gate.
            from pallas_arena.probe.contract_compose import ContractError, compose_contract
            from pallas_arena.probe.seam_scaffolds import RGLRU_SCAFFOLD, SPLASH_SCAFFOLD
            scaf = {"rg_lru": RGLRU_SCAFFOLD, "splash_attention": SPLASH_SCAFFOLD}[row["task"]]
            try:
                program = compose_contract(scaf, program)
            except ContractError as ce:
                gate = "contract violation"
                cell["gates"][gate] += 1
                verdict["outcome"] = f"pregate: contract violation: {str(ce)[:150]}"
                cell["rows"].append(verdict)
                continue

        ok, why = validity(row["task"], program)
        if not ok:
            gate = why.split(":", 1)[0][:40]
            cell["gates"][gate] += 1
            verdict["outcome"] = f"pregate: {why[:120]}"
            cell["rows"].append(verdict)
            continue
        cell["valid"] += 1

        # correctness at the tiny smoke cases (CPU-affordable)
        p = get_problem(row["task"])
        ns: dict = {}
        try:
            exec(compile(program, "<cand>", "exec"), ns)
            kfn = ns["kernel"]
            good = True
            for case in [c for c in p.shape_cases() if c.smoke and not c.holdout][:2]:
                ins = p.make_inputs(jax.random.PRNGKey(row["idx"]), case)
                ref = p.reference(*ins)
                tol = p.calibrated_tolerance(ins, ref)
                okc, whyc = check_tolerance(error_stats(kfn(*ins), ref), tol)
                if not okc:
                    good = False
                    verdict["outcome"] = f"incorrect: {case.name}: {whyc[:100]}"
                    break
            if good:
                cell["correct"] += 1
                verdict["outcome"] = "correct"
        except Exception as e:  # noqa: BLE001
            good = False
            verdict["outcome"] = f"runtime: {type(e).__name__}: {str(e)[:100]}"
        cell["rows"].append(verdict)

    report = {}
    for (task, variant), c in sorted(cells.items()):
        report[f"{task}:{variant}"] = {
            "n": c["n"],
            "validity_pct": round(100 * c["valid"] / max(c["n"], 1), 1),
            "correct_pct_of_all": round(100 * c["correct"] / max(c["n"], 1), 1),
            "truncated_pct": round(100 * c["truncated"] / max(c["n"], 1), 1),
            "gen_errors": c["gen_error"],
            "no_program": c["no_program"],
            "distinct_programs": len(c["program_hashes"]),
            "gate_histogram": dict(c["gates"]),
            "rows": c["rows"],
        }
        r = report[f"{task}:{variant}"]
        print(f"{task}:{variant:5s} n={r['n']:3d} valid={r['validity_pct']:5.1f}% "
              f"correct={r['correct_pct_of_all']:5.1f}% trunc={r['truncated_pct']:5.1f}% "
              f"distinct={r['distinct_programs']:3d} gates={r['gate_histogram']}", flush=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
