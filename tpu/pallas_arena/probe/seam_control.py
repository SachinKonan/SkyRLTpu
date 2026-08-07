"""CONTROL EXPERIMENT for the seam: known-good fills through the EXACT path.

MANDATORY before any chip is spent. A grid of zeros is evidence about models
only if the path is known to pass a correct kernel, and the previous probe cost
itself two of its own bugs that both wore a candidate's clothes (a 4-tuple
handed to the export child; `flatbuffers` missing from the pre-gate venv).

For every one of the five seams this runs, in order:

    known-good fill
      -> wrapped in a model-STYLE response (prose + a DECOY block first)
      -> extract_fill          (does the seam extractor find the real code?)
      -> compose               (does fill + scaffold make ONE valid program?)
      -> CPU AOT pre-gate      (does it export at EVERY declared shape,
                                including the non-divisible holdout, for the
                                TPU platform, from a CPU host?)
      -> queue -> judge        (does it compile, run and SCORE on silicon?)

`--interpret` additionally runs the composed FLCE and splash programs on CPU
against the problem's own fp32 reference at tiny shapes, so the numerics of the
scaffold -- not just its syntax -- are checked before a chip is booked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pallas_arena.probe.pregate import pregate_one, probe_signatures  # noqa: E402
from pallas_arena.probe.seam import SEAMS, compose, extract_fill  # noqa: E402
from pallas_arena.probe.seam_fills import CONTROL_FILLS  # noqa: E402

# The one-chip PROBE case sets, per task. One list, used by the prompt, the
# pre-gate and the judge, so a candidate is never graded against a contract it
# was not shown.
TASK_CASES: dict[str, list[str]] = {
    "splash_attention": ["probe-h8-s4096", "probe-h4-s2048", "probe-holdout-h4-s2049"],
    "flce": ["probe-4096x2880x151936", "probe-2048x2880x151936", "probe-holdout-3000x2880x151936"],
    "ragged_paged_attention": ["probe-b16-len1024", "probe-b8-len512", "probe-holdout-b17-len512"],
    "megablox_gmm": ["probe-m4096-e4-uniform", "probe-m2048-e4-zipf", "probe-holdout-m3000-e4-zipf"],
    "rg_lru": ["probe-4x2048x2560", "probe-2x1024x2560", "probe-holdout-2x1500x2560"],
}

# Tiny CPU-battery cases used by --interpret (real data, real reference).
SMOKE_CASES = {
    "flce": ["tiny", "tiny-ragged"],
    "rg_lru": ["tiny", "tiny-ragged"],
    "splash_attention": ["tiny", "tiny-ragged"],
    "megablox_gmm": ["tiny", "tiny-zipf"],
    "ragged_paged_attention": ["tiny"],
}

DECOY = (
    "Before I commit, here is the shape of the thing I am NOT going to do -- it\n"
    "materializes the whole thing and would blow the memory budget:\n\n"
    "```python\n# rejected\ndef tile_forward(*a):\n    raise NotImplementedError\n```\n\n"
    "Now the actual fill-ins:\n\n"
)


def model_style_response(fill: str) -> str:
    """Wrap a fill the way a model actually answers, so the control exercises
    EXTRACTION too -- including the decoy block models routinely emit first."""
    return DECOY + "```python\n" + fill.strip() + "\n```\n"


def run_one(task: str, label: str, fill: str, queue_url: str | None, timeout_s: float) -> dict:
    seam = SEAMS[task]
    resp = model_style_response(fill)
    extracted, how, missing = extract_fill(resp, seam.required)
    program = compose(task, extracted)
    out = {
        "task": task,
        "label": label,
        "extraction": how,
        "fill_chars": len(extracted),
        "fill_approx_tokens": len(extracted) // 4,
        "missing_names": missing,
        "took_the_decoy": "raise NotImplementedError" in extracted,
        "program_chars": len(program),
        "program_defines_kernel": "\ndef kernel(" in program,
    }
    if missing or out["took_the_decoy"]:
        print(f"[control] {task}/{label}: EXTRACTION BROKEN missing={missing} decoy={out['took_the_decoy']}", flush=True)
        return out

    sigs, case_sig, adv_sig, grad_sig = probe_signatures(task, TASK_CASES[task])
    out["n_signatures"] = len(sigs)
    out["signature_labels"] = [s["label"] for s in sigs]
    t0 = time.time()
    v = pregate_one(task, program, sigs, timeout_s=timeout_s)
    out["pregate_s"] = round(time.time() - t0, 2)
    out["pregate_passed"] = bool(v.get("passed"))
    out["pregate_gate"] = v.get("gate")
    out["pregate_observation"] = v.get("observation")
    print(
        f"[control] {task}/{label}: extraction={how} fill={len(extracted)}ch "
        f"(~{len(extracted)//4} tok) pregate={out['pregate_passed']} "
        f"gate={v.get('gate')} ({out['pregate_s']}s)",
        flush=True,
    )
    if not out["pregate_passed"]:
        print("           " + (v.get("observation") or "").replace("\n", "\n           "), flush=True)
        return out
    if not queue_url:
        return out

    from pallas_arena.judge.queue import ArenaQueueClient

    cl = ArenaQueueClient(queue_url, timeout_s=120.0)
    wid = cl.submit(task, program, tag=f"seam-control:{label}")
    print(f"[control] {task}/{label}: submitted {wid}, awaiting the judge...", flush=True)
    res = cl.wait([wid], timeout_s=1200.0, poll_s=5.0)
    r = res.get(wid) or {}
    out["judge_gate"] = r.get("gate")
    out["judge_passed"] = bool(r.get("passed"))
    out["judge_reward"] = r.get("reward")
    out["judge_score"] = r.get("score")
    out["judge_observation"] = r.get("observation")
    print(
        f"[control] {task}/{label}: JUDGE passed={out['judge_passed']} "
        f"gate={out['judge_gate']} reward={out['judge_reward']}",
        flush=True,
    )
    print("           " + (out["judge_observation"] or "").replace("\n", "\n           "), flush=True)
    return out


# ------------------------------------------------------------------ numerics
def interpret_check(task: str, label: str, fill: str) -> dict:
    """Run the COMPOSED program on CPU against the problem's own fp32 reference
    at tiny shapes. Pallas kernels run under `interpret=True`.

    This is the check the previous probe's tailored scaffolds got right and is
    worth keeping: `compile()` proves syntax, this proves the scaffold's tiling,
    padding, masking and (for FLCE) its custom_vjp actually compute the task.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from pallas_arena.judge.problems import get_problem

    problem = get_problem(task)
    seam = SEAMS[task]
    extracted, _how, missing = extract_fill(model_style_response(fill), seam.required)
    program = compose(task, extracted)
    if task in ("splash_attention", "megablox_gmm", "ragged_paged_attention"):
        # Make every pallas_call interpretable on a CPU host. Inject at
        # `out_shape=` and NOT right after the open paren: the kernel body is a
        # POSITIONAL argument, so a keyword inserted before it is a SyntaxError
        # (which is how the first control run failed).
        program = program.replace("        out_shape=", "        interpret=True,\n        out_shape=")
    ns: dict = {}
    out = {"task": task, "label": label, "missing_names": missing}
    try:
        exec(compile(program, f"<seam:{task}>", "exec"), ns)  # noqa: S102 - our own control code
    except Exception as e:
        out["error"] = f"exec: {type(e).__name__}: {e}"
        return out
    fn = ns.get("kernel")
    if fn is None:
        out["error"] = "composed program defines no `kernel`"
        return out

    worst_fwd, worst_grad = 0.0, None
    for name in SMOKE_CASES[task]:
        case = problem.case_by_name(name)
        inputs = problem.make_inputs(jax.random.PRNGKey(7), case)
        try:
            got = jax.jit(fn)(*inputs)
        except Exception as e:
            out["error"] = f"{name}: {type(e).__name__}: {e}"
            return out
        ref = problem.reference(*inputs)
        err = float(jnp.max(jnp.abs(got.astype(jnp.float32) - ref) / (jnp.abs(ref) + 1.0)))
        worst_fwd = max(worst_fwd, err)
        if not bool(jnp.isfinite(got).all()):
            out["error"] = f"{name}: non-finite output"
            return out
        if problem.has_bwd:
            try:
                g_cand = problem.grad_outputs(fn, *inputs)
                g_ref = problem.grad_outputs(problem.reference, *inputs)
                ge = float(jnp.max(jnp.abs(g_cand - g_ref) / (jnp.abs(g_ref) + 1.0)))
                worst_grad = ge if worst_grad is None else max(worst_grad, ge)
            except Exception as e:
                out["error"] = f"{name} grad: {type(e).__name__}: {e}"
                return out
    out["max_fwd_err"] = worst_fwd
    out["max_grad_err"] = worst_grad
    out["ok"] = worst_fwd < 5e-2 and (worst_grad is None or worst_grad < 5e-2)
    np.seterr(all="ignore")
    print(
        f"[interpret] {task}/{label}: max fwd err {worst_fwd:.3e}"
        + (f", max grad err {worst_grad:.3e}" if worst_grad is not None else "")
        + f"  -> {'OK' if out['ok'] else 'OUT OF TOLERANCE'}",
        flush=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tasks", default=None, help="comma list; default all five")
    ap.add_argument("--interpret", action="store_true", help="also run CPU numerics")
    ap.add_argument("--no-pregate", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=900.0)
    args = ap.parse_args()

    tasks = args.tasks.split(",") if args.tasks else list(SEAMS)
    results: list[dict] = []
    interp: list[dict] = []
    for task in tasks:
        for label, fill in CONTROL_FILLS[task]:
            if args.interpret:
                interp.append(interpret_check(task, label, fill))
            if not args.no_pregate:
                results.append(run_one(task, label, fill, args.queue, args.timeout_s))

    payload = {"pregate": results, "interpret": interp}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1, default=str))

    bad_i = [r for r in interp if not r.get("ok")]
    bad_p = [r for r in results if not r.get("pregate_passed")]
    print(
        f"\n[control] interpret {len(interp) - len(bad_i)}/{len(interp)} OK | "
        f"pre-gate {len(results) - len(bad_p)}/{len(results)} PASS",
        flush=True,
    )
    for r in bad_i:
        print(f"  INTERPRET FAIL {r['task']}/{r['label']}: {r.get('error') or r}", flush=True)
    for r in bad_p:
        print(f"  PREGATE FAIL {r['task']}/{r['label']}: {r.get('pregate_gate')}", flush=True)
    return 1 if (bad_i or bad_p) else 0


if __name__ == "__main__":
    raise SystemExit(main())
