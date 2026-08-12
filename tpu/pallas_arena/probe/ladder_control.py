"""CONTROL for the prompt-ladder run: prove the PATH before spending a chip.

Three earlier runs in this arena produced confident all-zero grids from
infrastructure faults (a token budget that 400'd, a signature-tuple bug, dead
engines). The rule that came out of it is that nothing goes to silicon until a
known-good answer has been pushed through the *identical* extraction ->
compose -> pre-gate path and come out PASS.

What this checks, in order:

  1. **Prompt budget.** Every rung of every task, tokenized the cheap way
     (chars/3.2, deliberately pessimistic) against gemma's served 16384
     window: a prompt that leaves under 11k tokens for the completion is a
     cell that will truncate mid-think and read as a model failure.
  2. **Whole-program control.** For each task, a KNOWN-GOOD complete program
     (the verified seam fill composed with its scaffold -- 8/8 PASS in job
     3650852) wrapped in a model-style response *with a decoy code block
     first*, then extracted with the real `extract_program`, composed with the
     real `compose_ladder`, and pushed through the real sandbox AOT export
     child at every declared probe shape. Run at rung p1 (no prelude) and at
     rung p4 (prelude prepended) so the prelude is proven additive.
  3. **Primitive numerics.** Every helper in `ladder.PRELUDES` checked against
     its own reference semantics on CPU, including the two cases the prompt
     promises: a fully-masked row out of `online_softmax` is exactly 0 and
     never NaN, and `chunk_scan` reproduces the rg_lru recurrence.
  4. **A primitives-based whole program**, end to end: an rg_lru kernel
     written the way a model would write it under P4 (a call to `chunk_scan`
     and a chunk length), pre-gated at every declared shape AND checked
     numerically against the problem's own fp32 reference.

Usage:
    JAX_PLATFORMS=cpu python -m pallas_arena.probe.ladder_control --out x.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pallas_arena.probe import configs as C  # noqa: E402
from pallas_arena.probe import ladder, prompt_ladder  # noqa: E402
from pallas_arena.probe.pregate import pregate_one, probe_signatures  # noqa: E402
from pallas_arena.probe.sampler import extract_program  # noqa: E402
from pallas_arena.probe.seam import SEAMS, compose  # noqa: E402
from pallas_arena.probe.seam_fills import CONTROL_FILLS  # noqa: E402

# --------------------------------------------------------------------------
# A model-style response: prose, a REJECTED sketch in its own fenced block,
# more prose, then the real answer. `extract_program` must take the last one.

_DECOY = '''Let me think about the tiling first.

```python
# first idea -- rejected, this materializes the whole thing
def kernel(*args):
    raise NotImplementedError("too much memory")
```

That will not fit. Here is the real answer.

```python
{code}
```

I chose the block size to stay under the scoped VMEM limit.
'''


def _wrap(code: str) -> str:
    return _DECOY.format(code=code)


# --------------------------------------------------------------------------
# A P4-style whole program for rg_lru: exactly what the primitives prompt asks
# for -- a chunk length, a call to `chunk_scan`, and the carry between chunks.
# Deliberately plain: if an answer this obvious cannot pass, the rung is wrong.

RGLRU_P4_PROGRAM = '''
import jax
import jax.numpy as jnp

CHUNK = 256


def kernel(x, a, reset):
    b, t, d = x.shape
    chunk = min(CHUNK, t)
    nch = -(-t // chunk)
    pad = nch * chunk - t
    if pad:
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0)))
        a = jnp.pad(a, ((0, 0), (0, pad), (0, 0)))
        reset = jnp.pad(reset, ((0, 0), (0, pad)))
    a32 = a.astype(jnp.float32) * (1.0 - reset[..., None].astype(jnp.float32))
    gx = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a32), 0.0)) * x.astype(jnp.float32)

    ac = jnp.moveaxis(a32.reshape(b, nch, chunk, d), 1, 0)
    gc = jnp.moveaxis(gx.reshape(b, nch, chunk, d), 1, 0)

    def body(h_prev, ins):
        a_c, g_c = ins
        # fold the incoming carry into the first timestep, then one scan
        g_c = g_c.at[:, 0, :].add(a_c[:, 0, :] * h_prev)
        h = chunk_scan(a_c, g_c)
        return h[:, -1, :], h

    _, hs = jax.lax.scan(body, jnp.zeros((b, d), jnp.float32), (ac, gc))
    return jnp.moveaxis(hs, 0, 1).reshape(b, nch * chunk, d)[:, :t, :]
'''


# --------------------------------------------------------------------------
def check_prompt_budget(max_model_len: int, reserve: int, want_new: int) -> list[dict]:
    rows = []
    for task in prompt_ladder.TASKS:
        for rung in prompt_ladder.RUNGS:
            p = prompt_ladder.build(task, rung)
            # pessimistic: real tokenizers give ~3.6-4.0 chars/token on this text
            approx = len(p) // 3
            room = max_model_len - approx - reserve
            rows.append(
                {
                    "task": task,
                    "rung": rung,
                    "chars": len(p),
                    "approx_tokens_pessimistic": approx,
                    "room_for_completion": room,
                    "ok": room >= want_new,
                }
            )
    return rows


def check_programs(timeout_s: float) -> list[dict]:
    out = []
    for task in prompt_ladder.TASKS:
        sigs = probe_signatures(task, C.TASK_CASES[task])
        label, fill = CONTROL_FILLS[task][0]
        program = compose(task, fill)  # a complete, verified module
        text = _wrap(program)
        code, how = extract_program(text)
        for rung in ("p1", "p4"):
            composed = ladder.compose_ladder(task, rung, code)
            res = pregate_one(task, composed, sigs, timeout_s=timeout_s)
            out.append(
                {
                    "task": task,
                    "rung": rung,
                    "fill": label,
                    "extraction": how,
                    "decoy_dropped": "NotImplementedError" not in code,
                    "prelude_prepended": composed != code,
                    "passed": bool(res.get("passed")),
                    "gate": res.get("gate"),
                    "violations": (res.get("violations") or [])[:2],
                    "observation": (res.get("observation") or "").replace("\n", " | ")[:240],
                }
            )
    return out


def check_primitives() -> dict:
    """Every helper against its own reference semantics, on CPU."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    out: dict = {}

    def run(task: str):
        ns: dict = {}
        exec(compile(ladder.PRELUDES[task], "<prelude>", "exec"), ns)  # noqa: S102
        return ns

    # dot_f32 -------------------------------------------------------------
    ns = run("splash_attention")
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(8, 16)), jnp.float32)
    y = jnp.asarray(rng.normal(size=(12, 16)), jnp.float32)
    got = ns["dot_f32"](x, y, 1, 1)
    want = jnp.einsum("mk,nk->mn", x, y)
    out["dot_f32"] = {
        "shape_ok": got.shape == (8, 12),
        "max_err": float(jnp.max(jnp.abs(got - want))),
        "hi_max_err": float(jnp.max(jnp.abs(ns["dot_f32"](x, y, 1, 1, hi=True) - want))),
    }

    # iota2 ---------------------------------------------------------------
    out["iota2"] = {
        "rows": np.asarray(ns["iota2"]((4, 1), 0)).ravel().tolist(),
        "cols": np.asarray(ns["iota2"]((1, 4), 1)).ravel().tolist(),
    }

    # online_softmax: streamed in 3 blocks == one dense masked softmax ------
    R_, K_, D_ = 6, 24, 8
    s_all = jnp.asarray(rng.normal(size=(R_, K_)) * 4.0, jnp.float32)
    v_all = jnp.asarray(rng.normal(size=(K_, D_)), jnp.float32)
    live_all = jnp.asarray(rng.random((R_, K_)) > 0.3)
    live_all = live_all.at[2, :].set(False)  # a FULLY masked row
    neg = ns["NEG"]
    m = jnp.full((R_, 1), neg, jnp.float32)
    lsum = jnp.zeros((R_, 1), jnp.float32)
    o = jnp.zeros((R_, D_), jnp.float32)
    for b0 in range(0, K_, 8):
        m, lsum, o = ns["online_softmax"](
            m, lsum, o, s_all[:, b0 : b0 + 8], v_all[b0 : b0 + 8], live_all[:, b0 : b0 + 8]
        )
    got = jnp.where(lsum > 0, o / jnp.maximum(lsum, 1e-30), 0.0)
    sm = jnp.where(live_all, s_all, -1e30)
    p = jnp.where(live_all, jnp.exp(sm - jnp.max(sm, 1, keepdims=True)), 0.0)
    rowlive = jnp.any(live_all, axis=1, keepdims=True)
    p = jnp.where(rowlive, p / jnp.maximum(jnp.sum(p, 1, keepdims=True), 1e-30), 0.0)
    want = p @ v_all
    out["online_softmax"] = {
        "max_err": float(jnp.max(jnp.abs(got - want))),
        "masked_row_is_exactly_zero": bool(jnp.all(got[2] == 0.0)),
        "finite": bool(jnp.all(jnp.isfinite(got))),
    }

    # chunk_scan == the rg_lru recurrence ----------------------------------
    nsr = run("rg_lru")
    B_, T_, Dd = 3, 17, 5
    a = jnp.asarray(rng.random((B_, T_, Dd)), jnp.float32)
    g = jnp.asarray(rng.normal(size=(B_, T_, Dd)), jnp.float32)
    got = nsr["chunk_scan"](a, g)
    h = np.zeros((B_, Dd), np.float32)
    ref = np.zeros((B_, T_, Dd), np.float32)
    for t in range(T_):
        h = np.asarray(a)[:, t] * h + np.asarray(g)[:, t]
        ref[:, t] = h
    out["chunk_scan"] = {
        "layout_ok": got.shape == (B_, T_, Dd),
        "max_err": float(np.max(np.abs(np.asarray(got) - ref))),
    }

    # fill_ref, inside a real pallas kernel body ---------------------------
    nsp = run("ragged_paged_attention")
    from jax.experimental import pallas as pl

    def body(o_ref):
        nsp["fill_ref"](o_ref, -1e30)

    got = pl.pallas_call(
        body, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32), interpret=True
    )()
    out["fill_ref"] = {"all_set": bool(jnp.all(got == -1e30)), "shape": list(got.shape)}
    return out


def check_p4_program(timeout_s: float) -> dict:
    """The plain P4-style rg_lru answer: pre-gate AND numerics."""
    import numpy as np

    task = "rg_lru"
    sigs = probe_signatures(task, C.TASK_CASES[task])
    code, how = extract_program(_wrap(RGLRU_P4_PROGRAM.strip()))
    composed = ladder.compose_ladder(task, "p4", code)
    res = pregate_one(task, composed, sigs, timeout_s=timeout_s)

    from pallas_arena.judge.problems import get_problem

    problem = get_problem(task)
    case = problem.case_by_name("tiny-ragged")
    import jax

    inputs = problem.make_inputs(jax.random.PRNGKey(0), case)
    ns: dict = {}
    err = None
    try:
        exec(compile(composed, "<p4>", "exec"), ns)  # noqa: S102
        got = np.asarray(ns["kernel"](*inputs), np.float64)
        want = np.asarray(problem.reference(*inputs), np.float64)
        rel = float(np.max(np.abs(got - want) / (np.abs(want) + 1.0)))
    except Exception as e:  # pragma: no cover - reported, not raised
        rel, err = None, f"{type(e).__name__}: {e}"
    return {
        "extraction": how,
        "uses_primitive": "chunk_scan(" in code,
        "prelude_prepended": composed != code,
        "passed": bool(res.get("passed")),
        "gate": res.get("gate"),
        "observation": (res.get("observation") or "").replace("\n", " | ")[:240],
        "numerics_max_rel_err": rel,
        "numerics_error": err,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--want-new-tokens", type=int, default=11000)
    args = ap.parse_args()

    spec = C.MODELS["gemma4-31b"]
    report: dict = {
        "prompt_budget": check_prompt_budget(spec.max_model_len, spec.reserve_tokens, args.want_new_tokens),
    }
    report["primitives"] = check_primitives()
    report["programs"] = check_programs(args.timeout_s)
    report["p4_program"] = check_p4_program(args.timeout_s)

    budget_ok = all(r["ok"] for r in report["prompt_budget"])
    prog_ok = all(r["passed"] and r["decoy_dropped"] for r in report["programs"])
    prel_ok = all(r["prelude_prepended"] for r in report["programs"] if r["rung"] == "p4")
    prim = report["primitives"]
    prim_ok = (
        prim["dot_f32"]["max_err"] < 1e-4
        and prim["online_softmax"]["max_err"] < 1e-5
        and prim["online_softmax"]["masked_row_is_exactly_zero"]
        and prim["online_softmax"]["finite"]
        and prim["chunk_scan"]["layout_ok"]
        and prim["chunk_scan"]["max_err"] < 1e-4
        and prim["fill_ref"]["all_set"]
    )
    p4 = report["p4_program"]
    p4_ok = bool(p4["passed"] and p4["uses_primitive"] and p4["prelude_prepended"]
                 and p4["numerics_max_rel_err"] is not None and p4["numerics_max_rel_err"] < 1e-3)
    report["summary"] = {
        "prompt_budget_ok": budget_ok,
        "control_programs_ok": prog_ok,
        "prelude_applied_ok": prel_ok,
        "primitives_ok": prim_ok,
        "p4_program_ok": p4_ok,
        "GREEN": bool(budget_ok and prog_ok and prel_ok and prim_ok and p4_ok),
    }
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["summary"], indent=1))
    for r in report["prompt_budget"]:
        print(f"  budget {r['task']:24s} {r['rung']}: ~{r['approx_tokens_pessimistic']:5d} tok, "
              f"room {r['room_for_completion']:6d} {'OK' if r['ok'] else 'TOO LONG'}")
    for r in report["programs"]:
        print(f"  control {r['task']:24s} {r['rung']}: {'PASS' if r['passed'] else 'FAIL'} "
              f"gate={r['gate']} {r['observation'][:110]}")
    print(f"  p4 program: {report['p4_program']}")
    return 0 if report["summary"]["GREEN"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
