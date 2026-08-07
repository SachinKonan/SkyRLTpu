"""CPU dry run for the seam run: every moving part except the chips.

Validates, without spending a single TPU-second:
  * every task has both a `reference` and a `seam` prompt, and both declare the
    numbers of every shape the judge will grade;
  * the scaffold shown in the prompt is BYTE-IDENTICAL to the one the harness
    appends (they are the same object, and this proves it stayed that way);
  * every signature in the API block still matches `inspect.signature` at the
    installed jax, and every name the block says does NOT exist really does
    not -- prose about an API is worth nothing if it has drifted;
  * `extract_fill` survives the ways a model actually answers: one block, one
    block per function, a rejected decoy first, an unterminated fence, prose;
  * `compose` yields a program whose `kernel` is the harness's;
  * `--problem` parses, every probe case name resolves, and no case list is
    holdout-only (which would make `final_reward` raise and every verdict come
    back `gate=worker`);
  * the headline metric returns the right answer on a synthetic record set
    with a KNOWN answer, including the case the last run got wrong: a group
    that is non-uniform but entirely below the noise floor.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pallas_arena.probe import configs as C  # noqa: E402
from pallas_arena.probe.metrics import summarize  # noqa: E402
from pallas_arena.probe.prompt_seam import API_BLOCK  # noqa: E402
from pallas_arena.probe.prompts import PROMPTS, prompt_table  # noqa: E402
from pallas_arena.probe.seam import SEAMS, compose, extract_fill, scaffold_of  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILURES.append(f"{name}: {detail}")


# --------------------------------------------------------------- shape decls
_SHAPE_KEYS = {
    "splash_attention": ("heads", "seq", "d"),
    "flce": ("n", "h", "v"),
    "ragged_paged_attention": ("batch", "max_len"),
    "megablox_gmm": ("m", "k", "n"),
    "rg_lru": ("b", "t", "d"),
}


def _shape_mentioned(task: str, case_name: str, prompt: str) -> bool:
    from pallas_arena.judge.problems import get_problem

    dims = get_problem(task).case_by_name(case_name).dims
    return all(str(dims[k]) in prompt for k in _SHAPE_KEYS[task] if k in dims)


# ---------------------------------------------------------------- API block
def check_api_block():
    """Every signature the prompt states must still be the real one."""
    import inspect

    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    # (label, object, a few parameter names the block claims, in order)
    claims = [
        ("pl.pallas_call", pl.pallas_call, ["kernel", "out_shape", "grid_spec", "grid", "in_specs",
                                            "out_specs", "scratch_shapes", "compiler_params"]),
        ("pl.BlockSpec", pl.BlockSpec, ["block_shape", "index_map", "pipeline_mode", "memory_space"]),
        ("pl.GridSpec", pl.GridSpec, ["grid", "in_specs", "out_specs", "scratch_shapes"]),
        ("pltpu.PrefetchScalarGridSpec", pltpu.PrefetchScalarGridSpec,
         ["num_scalar_prefetch", "grid", "in_specs", "out_specs", "scratch_shapes"]),
        ("pltpu.CompilerParams", pltpu.CompilerParams,
         ["dimension_semantics", "allow_input_fusion", "vmem_limit_bytes", "collective_id"]),
        ("jax.ShapeDtypeStruct", jax.ShapeDtypeStruct, ["shape", "dtype", "sharding", "weak_type"]),
        ("pl.program_id", pl.program_id, ["axis"]),
        ("pl.num_programs", pl.num_programs, ["axis"]),
        ("pl.ds", pl.ds, ["start", "size", "stride"]),
    ]
    for label, obj, params in claims:
        try:
            got = list(inspect.signature(obj).parameters)
        except (TypeError, ValueError) as e:
            check(f"API block: {label} introspectable", False, repr(e))
            continue
        got = [p for p in got if p != "self"]
        missing = [p for p in params if p not in got]
        check(f"API block: {label} params", not missing, f"missing {missing}" if missing else "")
        check(f"API block: {label} named in the prompt", label.split(".")[-1] in API_BLOCK)
        if label == "pl.BlockSpec":
            # the block asserts POSITIONAL (block_shape, index_map)
            check("API block: BlockSpec positional order", got[:2] == ["block_shape", "index_map"], got[:2])

    # the block also asserts a list of names that must NOT exist
    absent = [
        ("pl.load", pl, "load"),
        ("pl.store", pl, "store"),
        ("pltpu.ANY", pltpu, "ANY"),
        ("pltpu.TPUCompilerParams", pltpu, "TPUCompilerParams"),
        ("jax.Shape", jax, "Shape"),
    ]
    for label, mod, attr in absent:
        check(f"API block: {label} really is absent", not hasattr(mod, attr))
        check(f"API block: {label} listed as absent", label in API_BLOCK)

    # and that pallas_call really rejects the invented kwargs
    for bad in ("out_spec", "out_dtype", "out_shapes", "block_shapes"):
        check(
            f"API block: pallas_call has no `{bad}`",
            bad not in inspect.signature(pl.pallas_call).parameters and bad in API_BLOCK,
        )


# ---------------------------------------------------------------- extraction
def check_extraction():
    req = SEAMS["flce"].required  # ("TILE", "tile_forward", "tile_backward")
    one = (
        "Here is my answer.\n\n```python\nTILE = 256\n\ndef tile_forward(h, w, t):\n    return h, None\n\n"
        "def tile_backward(h, w, t, g, c):\n    return h\n```\n"
    )
    src, how, missing = extract_fill(one, req)
    check("extract: one complete block", not missing and how.endswith("single"), (how, missing))

    per_fn = (
        "First the constant:\n```python\nTILE = 256\n```\nthen the forward:\n"
        "```python\ndef tile_forward(h, w, t):\n    return h, None\n```\nand the backward:\n"
        "```python\ndef tile_backward(h, w, t, g, c):\n    return h\n```\n"
    )
    src, how, missing = extract_fill(per_fn, req)
    check("extract: one block per function", not missing and "multi" in how, (how, missing))

    decoy = (
        "A sketch I am rejecting:\n```python\ndef tile_forward(*a):\n    raise NotImplementedError\n```\n"
        "The real thing:\n```python\nTILE = 512\n\ndef tile_forward(h, w, t):\n    return h, None\n\n"
        "def tile_backward(h, w, t, g, c):\n    return h\n```\n"
    )
    src, how, missing = extract_fill(decoy, req)
    check(
        "extract: rejects a decoy block when a complete one exists",
        not missing and "raise NotImplementedError" not in src,
        (how, missing),
    )

    src, how, missing = extract_fill("I cannot help with that.", req)
    check("extract: prose only", how == "none" and sorted(missing) == sorted(req), (how, missing))

    partial = "```python\nTILE = 8\n\ndef tile_forward(h, w, t):\n    return h, None\n```\n"
    src, how, missing = extract_fill(partial, req)
    check("extract: reports what is missing", missing == ["tile_backward"], missing)

    unterm = "```python\ndef scan_chunk(x, a, r, h):\n    return x, h\nCHUNK = 4\n"
    src, how, missing = extract_fill(unterm, SEAMS["rg_lru"].required)
    check("extract: unterminated fence", not missing, (how, missing))


def check_composition():
    for task, seam in SEAMS.items():
        prog = compose(task, "# nothing\n")
        check(f"compose/{task}: defines kernel", "\ndef kernel(" in prog)
        check(f"compose/{task}: scaffold is last", prog.rstrip().endswith(scaffold_of(task).rstrip()[-60:]))
        # a model that pastes a whole program back must NOT win the entrypoint
        prog2 = compose(task, "def kernel(*a):\n    raise SystemExit\n")
        check(
            f"compose/{task}: harness kernel overrides the model's",
            prog2.index("def kernel(*a)") < prog2.rindex("def kernel("),
        )


# ------------------------------------------------------------------- judge
def check_case_sets():
    from pallas_arena.judge.problems import get_problem

    arg = C.problem_arg()
    check("problem arg has no spaces", " " not in arg, arg[:120])
    for task, names in C.TASK_CASES.items():
        problem = get_problem(task)
        cases = []
        for n in names:
            try:
                cases.append(problem.case_by_name(n))
            except KeyError as e:
                check(f"{task}: case {n} resolves", False, repr(e))
        scored = [c for c in cases if not c.holdout]
        check(f"{task}: at least one SCORED case", len(scored) >= 1, [c.name for c in cases])
        check(f"{task}: exactly one holdout", sum(1 for c in cases if c.holdout) == 1)
        check(f"{task}: all probe cases", all(c.probe for c in cases))


def check_baselines_resolve():
    """Every problem must have a baseline that RETURNS on this host. On the
    judge the persistent worker treats BaselineUnavailable as 'problem not
    served' -- it does not fall back, it silently stops serving the task."""
    import jax

    from pallas_arena.judge.problems import get_problem

    for task in C.TASK_CASES:
        problem = get_problem(task)
        case = next(c for c in problem.shape_cases() if c.smoke)
        try:
            problem.baseline(*problem.make_inputs(jax.random.PRNGKey(0), case))
            impl = getattr(problem, "baseline_impl", None)
            check(f"{task}: baseline resolves on CPU", True, f"impl={impl}")
        except Exception as e:
            # splash gates its whole baseline (production kernel AND the XLA
            # fallback) behind `default_backend() == "tpu"` by design, so on a
            # CPU host it is EXPECTED to raise. It booted fine on the v6e-1
            # last run. Every other task must resolve here, because the
            # persistent worker silently stops serving a problem whose
            # baseline raises.
            expected = task == "splash_attention" and jax.default_backend() != "tpu"
            check(
                f"{task}: baseline resolves on CPU",
                expected,
                f"{type(e).__name__}: {e}" + (" (expected: TPU-only by design)" if expected else ""),
            )


# ----------------------------------------------------------------- metrics
def check_metrics():
    recs = []
    # g0: all zero -> uniform, no signal
    for _ in range(16):
        recs.append({"config": "m|seam|flce", "group": "g0", "stage": "pregate", "gate": "ast",
                     "reward": 0.0, "variant": "seam", "model": "m", "task": "flce"})
    # g1: mixed pass/fail -> spread 0.71, way above any floor
    for i in range(16):
        recs.append({"config": "m|seam|splash_attention", "group": "g1", "stage": "judge",
                     "gate": "all" if i == 3 else "correctness", "reward": 0.71 if i == 3 else 0.0,
                     "variant": "seam", "model": "m", "task": "splash_attention"})
    # g2: THE tailored trap -- 16/16 pass, spread 0.0042, floor 0.0158.
    # Non-uniform, and NOT signal.
    for i in range(16):
        recs.append({"config": "m|tailored|flce", "group": "g2", "stage": "judge", "gate": "all",
                     "reward": 0.6966 + 0.0042 * (i / 15), "variant": "tailored", "model": "m",
                     "task": "flce"})
    floors = {"flce": 0.0158, "splash_attention": 0.0284}
    s = summarize(recs, group_size=16, noise_floors=floors)
    h = s["headline"]
    check("metrics: 3 complete groups, 2 non-uniform", h["overall_nonuniform_groups"] == "2/3",
          h["overall_nonuniform_groups"])
    check("metrics: only ONE carries signal above the floor", h["overall_signal_groups"] == "1/3",
          h["overall_signal_groups"])
    check("metrics: the sub-noise group is NOT counted as signal",
          s["per_config"]["m|tailored|flce"]["signal_groups"] == 0)
    check("metrics: the sub-noise group IS still counted as non-uniform",
          s["per_config"]["m|tailored|flce"]["nonuniform_groups"] == 1)
    check("metrics: uniform-zero config has no signal",
          s["per_config"]["m|seam|flce"]["signal_group_frac"] == 0.0)
    check("metrics: mixed config has signal",
          s["per_config"]["m|seam|splash_attention"]["signal_group_frac"] == 1.0)
    check("metrics: per-task rollup present", "flce|seam" in s["by_task_variant"])


def main() -> int:
    print("=== prompts ===", flush=True)
    for row in sorted(prompt_table(), key=lambda r: (r["task"], r["variant"])):
        print(f"  {row['task']:24s} {row['variant']:10s} {row['chars']:6d} chars "
              f"(~{row['approx_tokens']} tokens)", flush=True)

    for task in C.TASKS:
        vs = PROMPTS.get(task, {})
        check(f"{task}: has reference + seam", {"reference", "seam"} <= set(vs), sorted(vs))
        for v in ("reference", "seam"):
            if v not in vs:
                continue
            check(f"{task}/{v}: declares every probe shape",
                  all(_shape_mentioned(task, s, vs[v]) for s in C.TASK_CASES[task]))
            # gemma is served at 16384; a prompt that leaves under 4k of room
            # cannot land a kernel behind it
            approx = len(vs[v]) // 4
            check(f"{task}/{v}: leaves >= 6k tokens of room at 16384", approx <= 10000, f"~{approx} tokens")
        check(f"{task}/seam: embeds the scaffold VERBATIM", scaffold_of(task) in vs["seam"])
        check(f"{task}/seam: names every required fill",
              all(n in vs["seam"] for n in SEAMS[task].required), SEAMS[task].required)
        check(f"{task}/seam: carries the API block", "Real JAX 0.10.2 API" in vs["seam"])

    print("=== extraction ===", flush=True)
    check_extraction()
    print("=== composition ===", flush=True)
    check_composition()
    print("=== case sets / judge arg ===", flush=True)
    check_case_sets()
    print("=== baselines ===", flush=True)
    check_baselines_resolve()
    print("=== api block ===", flush=True)
    check_api_block()
    print("=== metrics ===", flush=True)
    check_metrics()

    print("\n=== seam dry run: " + (f"{len(FAILURES)} FAILURES" if FAILURES else "ALL GREEN") + " ===", flush=True)
    for f in FAILURES:
        print("  " + f, flush=True)
    Path("/tmp/seam-dryrun.json").write_text(json.dumps({"failures": FAILURES}, indent=1))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
