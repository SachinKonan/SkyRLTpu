"""Generate the TP bring-up VALIDATION payload: candidates designed to pass.

Why not real winners (learned on job 3715217): the sd-era winners predate the
generalized contract, and the tp8-gqa/mqa signatures hand them per-shard
shapes (q 4 heads over k 1 head) they were never written for. Best case they
fail aot_export; the case actually observed was worse -- a Mosaic compile of
one splash winner HALTED THE TPU CORE, and everything after it (the other
winner's fixtures, RPA's whole boot) died of the wedged chip.

The bring-up's job is to validate the TP grading path, not to score old code.
So each program here is our own XLA formulation wrapped as `kernel`:
  * traces at EVERY declared shape (GQA/MQA/asymmetric-dv included),
  * pure XLA -- cannot core-halt Mosaic,
  * differentiable by autodiff -- exercises the SCORED-backward path against
    the production backward (grad_reward + grad_baseline_impl for real),
  * runs under --no-enforce-pallas, which the bring-up passes explicitly;
    the AST pallas requirement stays on for actual RL grading.

Task order in the payload is deliberate: splash LAST, so if anything does
wedge the chip the other three tasks have already reported.

Each program is exec'd and checked against its task's reference on a smoke
case BEFORE being written -- this generator refuses to emit a payload that
would fail on the QR.
"""

from __future__ import annotations

import argparse
import inspect
import json
import textwrap


def _splash_program() -> str:
    from pallas_arena.judge.problems import splash_attention as m

    return (
        "import jax\nimport jax.numpy as jnp\nimport numpy as np\n\n"
        f"NEG_INF = {m.NEG_INF!r}\n_FALLBACK_BLOCK_Q = {m._FALLBACK_BLOCK_Q}\n\n"
        + textwrap.dedent(inspect.getsource(m._xla_grouped_attention))
        + "\n\ndef kernel(q, k, v, segment_ids):\n"
        "    return _xla_grouped_attention(q, k, v, segment_ids)\n"
    )


def _megablox_program() -> str:
    return (
        "import jax\nimport jax.numpy as jnp\n\n"
        "def kernel(lhs, rhs, group_sizes):\n"
        "    # the design-floor formulation: fp32 ragged dot (explicitly legal)\n"
        "    return jax.lax.ragged_dot(\n"
        "        lhs.astype(jnp.float32), rhs.astype(jnp.float32), group_sizes\n"
        "    )\n"
    )


def _rpa_program() -> str:
    from pallas_arena.judge.problems import ragged_paged_attention as m

    return (
        "import jax\nimport jax.numpy as jnp\nimport numpy as np\n\n"
        + textwrap.dedent(inspect.getsource(m._xla_paged_decode))
        + "\n\ndef kernel(q, k_pages, v_pages, page_tables, seq_lens):\n"
        "    return _xla_paged_decode(q, k_pages, v_pages, page_tables, seq_lens)\n"
    )


def _rg_lru_program() -> str:
    from pallas_arena.judge.problems import rg_lru as m

    return (
        "import jax\nimport jax.numpy as jnp\n\n"
        + textwrap.dedent(inspect.getsource(m._apply_reset))
        + "\n"
        + textwrap.dedent(inspect.getsource(m.rg_lru_associative))
        + "\n\ndef kernel(x, a, reset):\n"
        "    return rg_lru_associative(x, a, reset)\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tpu/pallas_arena/verify/tp-validation-codes.json")
    args = ap.parse_args()

    import jax

    from pallas_arena.judge.problems import get_problem
    from pallas_arena.judge.problems.base import check_tolerance, error_stats

    programs = {
        # splash deliberately LAST (chip-wedge blast-radius control)
        "megablox_gmm": {"val-ragged-dot": _megablox_program()},
        "rg_lru": {"val-associative-scan": _rg_lru_program()},
        "ragged_paged_attention": {"val-xla-paged": _rpa_program()},
        "splash_attention": {"val-xla-grouped": _splash_program()},
    }

    # refuse to ship a program that fails its own smoke check
    for task, entries in programs.items():
        p = get_problem(task)
        smoke = next(c for c in p.shape_cases() if c.smoke and not c.holdout)
        inputs = p.make_inputs(jax.random.PRNGKey(0), smoke)
        ref = p.reference(*inputs)
        tol = p.calibrated_tolerance(inputs, ref)
        for name, src in entries.items():
            ns: dict = {}
            exec(compile(src, f"<{name}>", "exec"), ns)
            out = ns["kernel"](*inputs)
            ok, why = check_tolerance(error_stats(out, ref), tol)
            assert ok, f"{task}/{name} fails its own smoke check: {why}"
            print(f"validated {task}/{name} on {smoke.name}", flush=True)

    with open(args.out, "w") as f:
        json.dump(programs, f, indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
