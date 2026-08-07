"""Client-side AOT pre-gate for the probe, run in parallel on CPU.

Phase 5 measured a single judge lane at ~5.6 s median. A 2400-candidate probe
on ONE v6e-1 would therefore cost >3.5 h of chip time even if every candidate
were honest — and most of a base model's first attempts do not survive the
sandbox child at all. So the *same* sandbox child the judge would run
(``mode="aot_export"``, AST gate + poison stubs + timeout + RLIMIT, no
device, exporting FOR the tpu platform from a CPU host) runs first on the
neuronic compute node, 16-way parallel, at zero chip cost.

That is not a shortcut: it is literally step 1 of ``PersistentWorker.
grade_code``, using ``build_signatures`` so the exported contract is
identical. A candidate that fails here would have failed on the judge at the
same gate; only survivors are worth a chip.

What the pre-gate CANNOT see is anything that happens after StableHLO: the
Mosaic/XLA backend compile (where ``CompileTimeScopedVmemOom`` lives),
correctness, determinism, gradients and timing. Those are the judge's.
"""

from __future__ import annotations

import shutil
import tempfile

from pallas_arena.judge import grader
from pallas_arena.judge.observation import attach_observation


def probe_signatures(problem_name: str, case_names: list[str]):
    """The exact signature set the judge will demand for this case set, with
    the adversarial library OFF (its shapes are not in any prompt)."""
    from pallas_arena.judge.problems import get_problem
    from pallas_arena.judge.worker import build_signatures

    problem = get_problem(problem_name)
    cases = [problem.case_by_name(n) for n in case_names]
    scored = [c for c in cases if not c.holdout]
    holdout = [c for c in cases if c.holdout]
    return build_signatures(problem, scored, holdout, [])


def pregate_one(
    problem_name: str,
    code: str,
    signatures: list[dict],
    *,
    timeout_s: float = 180.0,
    rlimit_gb: float = 24.0,
) -> dict:
    """Run the sandbox export child once. Returns a compact verdict dict."""
    if not code.strip():
        return attach_observation(
            {
                "ok": True,
                "passed": False,
                "gate": "no_code",
                "reward": 0.0,
                "problem": problem_name,
                "violations": ["model produced no extractable python program"],
            }
        )
    wd = tempfile.mkdtemp(prefix="probe-pregate-")
    try:
        res = grader.grade(
            problem_name,
            code,
            mode="aot_export",
            enforce_pallas=None,  # the problem's own require_pallas decides
            timeout_s=timeout_s,
            rlimit_gb=rlimit_gb,
            cache=None,
            workdir=wd,
            export_signatures=signatures,
            export_platforms=["tpu"],
            child_env={"JAX_PLATFORMS": "cpu"},
        )
    except Exception as e:
        res = {
            "ok": False,
            "passed": False,
            "gate": "harness",
            "reward": 0.0,
            "violations": [f"{type(e).__name__}: {e}"],
        }
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    res.pop("artifacts", None)
    res["problem"] = problem_name
    if not res.get("passed"):
        attach_observation(res)
    return res
