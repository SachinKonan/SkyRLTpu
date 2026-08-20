"""Reference-first prompts (variants rf1 / rf2) -- the first-principles restart.

The sd prompts grew defensively: the seam, the ten-bullet dialect list with
kill counts, and the declared/holdout vocabulary all patch measured
gemma-4-31B failures, and a fresh model has to decode that scar tissue before
it can start. This generator follows ttt-discover's TriMul register instead
(third_party/discover/examples/gpu_mode/prompt.py): a clean task statement.

Principles:
  * SEMANTICS AS CODE. The task's own fp32 reference is inlined verbatim
    (inspect.getsource, so prompt and judge cannot drift) and framed as
    editable starter code: correct, slow, restructure anything.
  * SHAPES PLAINLY. Tile sizes derive from shapes, so the graded case list is
    stated as a table, non-divisible cases called out in one sentence.
  * ERROR KNOWLEDGE LIVES IN THE LOOP. The repair driver feeds back verbatim
    tracebacks and judge verdicts; the prompt keeps only three platform notes
    that are physics rather than model pathology.

rf1 = the plain prompt. rf2 = rf1 + a minimal pallas_call example at the
TASK'S OWN operand rank (the earlier worked example backfired precisely
because it was a foreign kernel at a foreign rank).
"""

from __future__ import annotations

import inspect

from pallas_arena.judge.problems import (
    megablox_gmm,
    ragged_paged_attention,
    rg_lru,
    splash_attention,
)
from pallas_arena.judge.problems.base import Problem

# ---------------------------------------------------------------- per-task data

_TASKS: dict[str, dict] = {
    "splash_attention": {
        "title": "causal segment-masked multi-head attention",
        "ref_fn": splash_attention.causal_segment_attention,
        "problem": splash_attention.PROBLEM,
        "entry": ("def kernel(q, k, v, segment_ids, *, window=None, soft_cap=None, "
                  "sinks=None):  # -> o"),
        "io": (
            "q: [q_heads, seq, d] bfloat16 (pre-scaled by 1/sqrt(d))\n"
            "k: [kv_heads, seq, d] bfloat16; v: [kv_heads, seq, d_v] bfloat16\n"
            "  (q_heads = kv_heads * group; kv_heads=1 is MQA; d_v may differ from d)\n"
            "segment_ids: [seq] int32; 0 marks padding\n"
            "o: [q_heads, seq, d_v] float32"
        ),
        "baseline": "Google's Pallas splash-attention kernel (tuned block sizes)",
        "shape_note": "seq=2049 is deliberately not divisible by any tile size; your kernel must still trace and run there.",
        "example_shape": ("x", "[heads, seq, head_dim]"),
    },
    "megablox_gmm": {
        "title": "grouped matrix multiply (MoE expert dispatch)",
        "ref_fn": megablox_gmm.gmm_reference,
        "problem": megablox_gmm.PROBLEM,
        "entry": "def kernel(lhs, rhs, group_sizes):  # -> out",
        "io": (
            "lhs: [m, k] bfloat16 (rows grouped contiguously by expert)\n"
            "rhs: [num_groups, k, n] bfloat16\n"
            "group_sizes: [num_groups] int32, sums to m; groups may be size 0\n"
            "out: [m, n] float32, out[rows of group g] = lhs[rows] @ rhs[g]"
        ),
        "baseline": "the Pallas megablox GMM kernel (tuned tiling)",
        "shape_note": "m=3000 is deliberately not a multiple of any reasonable row tile; zipf group sizes are heavily imbalanced and can include empty groups.",
        "example_shape": ("x", "[m, n]"),
    },
    "ragged_paged_attention": {
        "title": "paged decode attention (one query token per sequence)",
        "ref_fn": ragged_paged_attention.paged_decode_attention_reference,
        "problem": ragged_paged_attention.PROBLEM,
        "entry": "def kernel(q, k_pages, v_pages, page_tables, seq_lens):  # -> o",
        "io": (
            "q: [batch, q_heads, head_dim] bfloat16 (pre-scaled; one decode token per sequence)\n"
            "k_pages, v_pages: [num_pages, page_size, kv_heads, head_dim] bfloat16\n"
            "page_tables: [batch, max_pages] int32; seq_lens: [batch] int32\n"
            "o: [batch, q_heads, head_dim] float32   (q_heads = kv_heads * group)"
        ),
        "baseline": "the Pallas ragged-paged-attention kernel that ships in jax",
        "shape_note": "batch=17 is deliberately odd; sequence lengths are ragged, so most pages of most sequences are partially or fully dead.",
        "example_shape": ("x", "[batch, heads, head_dim]"),
    },
    "rg_lru": {
        "title": "gated linear recurrence (RG-LRU, the Griffin recurrence)",
        "ref_fn": rg_lru.rg_lru_scan_reference,
        "problem": rg_lru.PROBLEM,
        "entry": "def kernel(x, a, reset):  # -> h",
        "io": (
            "x: [b, t, d] bfloat16\n"
            "a: [b, t, d] float32 in [0, 1); reset: [b, t] bool\n"
            "h: [b, t, d] float32"
        ),
        "baseline": "DeepMind's recurrentgemma Pallas scan",
        "shape_note": "t=1500 is deliberately not a multiple of any chunk size.",
        "example_shape": ("x", "[b, t, d]"),
    },
}

# rf3-only per-task fields (backward description; which tasks have one)
_RF3_BWD = {
    "splash_attention": {
        "grad_inputs": "(q, k, v)",
        "bwd_baseline": "the production splash kernel's own fused backward",
    },
    "rg_lru": {
        "grad_inputs": "(x, a)  -- the gradient must flow BACK THROUGH the recurrence, including the gate path",
        "bwd_baseline": "the differentiated production scan",
    },
}

# rf3: the EVOLUTION-grade contract additions. Everything here states a fact
# the judge already enforces; rf1/rf2 predate three contract changes and are
# kept frozen as historical controls.
#   1. BACKWARD IS SCORED: the judge differentiates the kernel and times the
#      backward separately vs the production kernel's own backward. A raw
#      pallas_call is NOT differentiable (generic autodiff cannot re-enter a
#      Pallas grid), so without jax.custom_vjp the backward component is
#      forfeited -- the forward reward is kept either way. The observation
#      hint that used to teach this fired only on a gate that scored mode
#      never reaches, so the PROMPT must say it.
#   2. FEATURE KWARGS: featured cases bind window/soft_cap/sinks as STATIC
#      kwargs at export; a kernel without those parameters TypeErrors at
#      export on every featured case.
#   3. THE PROMPT RENDERS FROM THE GRADED CASE LIST: prompt/contract drift
#      (stale names, unstated shapes) has cost real judge runs; build3 takes
#      the exact case names the driver grades.
_BWD_SECTION = """
## Backward pass (separately scored)

Your kernel is also differentiated: the judge runs `jax.grad` through it and
checks d/d{grad_inputs} against the reference's gradients (per-input
tolerances), then times your backward against {bwd_baseline}. This is a
SEPARATE reward component -- a forward-only kernel keeps its forward reward
and simply earns 0 for the backward.

A raw `pl.pallas_call` is NOT differentiable: generic autodiff cannot trace
into a Pallas grid. To earn the backward component, wrap your kernel in
`jax.custom_vjp` and write the backward as its own Pallas kernel (save what
the backward needs -- e.g. row max/denominator statistics -- as residuals).
"""

_FEATURE_SECTION = """
## Feature cases (static kwargs)

Some graded cases bind extra STATIC keyword arguments into your kernel at
compile time -- your `kernel` must accept them (defaults shown in the
signature above), and because they are static Python values you can and
should SPECIALIZE the kernel on them:

{feature_lines}
"""

_FEATURE_DOCS = {
    "window": ("window=W: sliding-window attention -- a query at position i "
               "attends only to keys in (i-W, i]. A static window lets you "
               "SKIP whole KV blocks, which is the entire speedup."),
    "soft_cap": ("soft_cap=C: logit soft-cap -- apply C * tanh(logits / C) "
                 "BEFORE masking and the softmax max-shift."),
    "sinks": ("sinks=(s_0..s_{H-1}): attention sinks, one scalar per query "
              "head -- the sink joins the softmax max (m = max(row_max, s_h)) "
              "and adds exp(s_h - m) to the denominator, but contributes no "
              "value row."),
}

_HEADER = """You are an expert JAX/Pallas TPU engineer. Below is a correct but slow \
implementation of {title}. Rewrite it as a fast TPU kernel.

## Starter code (correct, slow -- restructure anything you like)

This is the exact code your output will be checked against for correctness. It is \
slow because it materializes everything at fp32 and fuses nothing. Your job is to \
make it fast on a TPU while producing the same numbers.

```python
{ref_source}```

## What you must define

```python
{entry}
```

{io}

Your program must contain a real `pl.pallas_call` -- a submission without one \
(e.g. pure `jax.numpy` / `lax.associative_scan` formulations) is rejected before \
grading. Do not import this problem's production kernel or any library \
attention/GMM/scan entry point; write the kernel yourself.

## Test shapes

One program must trace and run at ALL of these:

{shapes}

{shape_note} Choose your block/tile sizes with these shapes in mind.

## Scoring

Correctness first: per-element error against the starter code above on hidden random \
seeds (max and 99th-percentile tail, tolerance calibrated to what honest \
implementations achieve), plus a few adversarial inputs (saturating values, fully \
masked rows, empty groups). Then speed: your kernel is timed against {baseline}; \
reward = its median time / yours, so 1.0 is parity and above 1.0 beats it.

## Three TPU notes

1. `jnp.dot`/`jnp.einsum` on two bfloat16 arrays RETURNS bfloat16, silently rounding \
away the MXU's float32 accumulator. Pass `preferred_element_type=jnp.float32` on \
every matmul and keep running max/sum/accumulators float32; use bfloat16 only as \
matmul inputs. Measured here: kernels that do this sit comfortably inside the \
correctness tolerance; kernels that don't fail it.
2. A kernel invocation has roughly 32 MB of VMEM for all its Refs and scratch; \
budget your block sizes against it. Last-dimension tiles want multiples of 128, \
second-to-last multiples of 8.
3. Imports: `from jax.experimental import pallas as pl` and \
`from jax.experimental.pallas import tpu as pltpu`. Refs are read and written by \
indexing (`ref[...]`, `ref[i, :] = v`); there is no pl.load/pl.store.

## Output

One fenced ```python block containing the complete program. No prose after it.
"""

_EXAMPLE = """
## A minimal complete Pallas kernel (shape pattern only -- not this task)

Doubles an array of the same rank as this task's operands, one block at a time:

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def _double_block(x_ref, o_ref):
    o_ref[...] = x_ref[...].astype(jnp.float32) * 2.0

def double({var}):  # {var}: {shape}
    n0 = {var}.shape[-2]
    blk = 256 if n0 % 256 == 0 else n0
    grid = (n0 // blk,)
    spec = pl.BlockSpec({var}.shape[:-2] + (blk, {var}.shape[-1]),
                        lambda i: (0,) * (len({var}.shape) - 2) + (i, 0))
    return pl.pallas_call(
        _double_block,
        grid=grid,
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct({var}.shape, jnp.float32),
    )({var})
```
"""


def _shapes_table(problem: Problem) -> str:
    rows = []
    for c in problem.shape_cases():
        if not c.probe:
            continue
        dims = ", ".join(f"{k}={v}" for k, v in c.dims.items())
        rows.append(f"  * {dims}")
    return "\n".join(rows)


def build(task: str, example: bool = False) -> str:
    t = _TASKS[task]
    ref_source = inspect.getsource(t["ref_fn"])
    prompt = _HEADER.format(
        title=t["title"],
        ref_source="import jax\nimport jax.numpy as jnp\n\n" + ref_source,
        entry=t["entry"],
        io=t["io"],
        shapes=_shapes_table(t["problem"]),
        shape_note=t["shape_note"],
        baseline=t["baseline"],
    )
    if example:
        var, shape = t["example_shape"]
        head, _, tail = prompt.rpartition("## Output")
        prompt = head + _EXAMPLE.format(var=var, shape=shape) + "\n## Output" + tail
    return prompt


def _shapes_table3(problem: Problem, case_names: list[str]) -> tuple[str, str]:
    """Shapes rendered from the EXACT case list the driver grades.

    Returns (plain_rows, feature_lines). TP cases render at the PER-SHARD
    shape the kernel is actually traced at, stated as such -- a model
    reasoning about tile sizes from the full shape designs for the wrong
    problem."""
    rows, feats_seen = [], {}
    for name in case_names:
        c = problem.case_by_name(name)
        w = problem.tp_declared_width(c)
        if w:
            per_shard = ", ".join(
                f"{tuple(a.shape)} {a.dtype}" for a in problem.abstract_inputs_tp(c, w)
            )
            rows.append(f"  * sharded over {w} devices -- your kernel sees per-shard inputs {per_shard}")
        else:
            dims = ", ".join(f"{k}={v}" for k, v in c.dims.items())
            feat = dict(c.features)
            if feat:
                fstr = ", ".join(f"{k}={v}" for k, v in feat.items())
                rows.append(f"  * {dims}   [static kwargs: {fstr}]")
                for k in feat:
                    feats_seen[k] = _FEATURE_DOCS[k]
            else:
                rows.append(f"  * {dims}")
    feature_lines = "\n".join(f"  * {doc}" for doc in feats_seen.values())
    return "\n".join(rows), feature_lines


def build3(task: str, case_names: list[str], example: bool = False) -> str:
    """The rf3 evolution prompt: rf1's register + the scored-backward section
    + feature-kwarg docs, rendered from the graded case list."""
    t = _TASKS[task]
    ref_source = inspect.getsource(t["ref_fn"])
    shapes, feature_lines = _shapes_table3(t["problem"], case_names)
    prompt = _HEADER.format(
        title=t["title"],
        ref_source="import jax\nimport jax.numpy as jnp\n\n" + ref_source,
        entry=t["entry"],
        io=t["io"],
        shapes=shapes,
        shape_note=t["shape_note"],
        baseline=t["baseline"],
    )
    inserts = ""
    if task in _RF3_BWD:
        inserts += _BWD_SECTION.format(**_RF3_BWD[task])
    if feature_lines:
        inserts += _FEATURE_SECTION.format(feature_lines=feature_lines)
    if inserts:
        head, _, tail = prompt.rpartition("## Output")
        prompt = head + inserts + "\n## Output" + tail
    if example:
        var, shape = t["example_shape"]
        head, _, tail = prompt.rpartition("## Output")
        prompt = head + _EXAMPLE.format(var=var, shape=shape) + "\n## Output" + tail
    return prompt


_SCAFFOLD_SECTION = """
## Working scaffold (recommended starting point)

The program below has ALL the plumbing written and verified: `pallas_call`
grids, BlockSpecs (including the GQA head mapping and padding for
non-divisible sequences), and the complete `jax.custom_vjp` wiring with the
correct cotangent structure. The bodies marked `NotImplementedError` are
yours: fill them (and tune the tile constants) or restructure anything --
it is a starting point, not a cage. With only the forward body filled the
kernel already grades; the backward bodies earn the separate backward
reward.

```python
{scaffold}```
"""


def build3s(task: str, case_names: list[str]) -> str:
    """rf3s: the rf3 contract prompt + the verified seam scaffold.

    Measured poles this variant navigates (PROBE-REPORT): whole-program
    splash exported 0/96 (the plumbing kills everyone), while the
    fully-tailored scaffold passed 16/16 with within-group spread BELOW the
    judge noise floor (nothing to rank). The scaffold hands over the
    dialect; the bodies keep the algorithm -- and the spread -- open.
    """
    from pallas_arena.probe.seam_scaffolds import RGLRU_SCAFFOLD, SPLASH_SCAFFOLD

    scaffold = {"splash_attention": SPLASH_SCAFFOLD, "rg_lru": RGLRU_SCAFFOLD}[task]
    prompt = build3(task, case_names, example=False)
    head, _, tail = prompt.rpartition("## Output")
    return head + _SCAFFOLD_SECTION.format(scaffold=scaffold) + "\n## Output" + tail


IMPROVE_TEMPLATE = """{base}

## Your previous attempt (reward: {reward})

```python
{program}
```

## Judge feedback

{observation}

Improve on your previous attempt: keep what works, fix what the feedback
names, and make it faster. Output one fenced ```python block containing the
complete improved program. No prose after it.
"""


REF_FIRST_PROMPTS: dict[str, dict[str, str]] = {
    task: {"rf1": build(task), "rf2": build(task, example=True)} for task in _TASKS
}

# ------------------------------------------------------------- import-time asserts
for _task, _t in _TASKS.items():
    _p1 = REF_FIRST_PROMPTS[_task]["rf1"]
    assert "def " in inspect.getsource(_t["ref_fn"]), _task
    assert _shapes_table(_t["problem"]), f"{_task}: no probe shapes"
    assert len(_p1) // 4 < 3200, f"{_task}: rf1 estimate {len(_p1) // 4} tokens > 3200"
    assert "pallas_call" in REF_FIRST_PROMPTS[_task]["rf2"], _task
