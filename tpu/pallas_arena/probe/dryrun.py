"""CPU dry run for the probe: every moving part except the chips.

Validates, without spending a single TPU-second:
  * the prompt modules import and the three variants are strictly nested;
  * `probe_signatures` builds the same signature set the judge will demand,
    for both probe tasks, at the probe shapes;
  * the sandbox export child reaches the RIGHT gate on hand-written
    candidates covering ast / exec / poison_stub / aot_export / pass;
  * every verdict carries a curated observation under 500 chars;
  * `metrics.summarize` produces the headline non-uniform-group number from
    a synthetic record set with a known answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pallas_arena.probe import configs as C  # noqa: E402
from pallas_arena.probe.metrics import summarize  # noqa: E402
from pallas_arena.probe.pregate import pregate_one, probe_signatures  # noqa: E402
from pallas_arena.probe.prompts import PROMPTS, prompt_table  # noqa: E402
from pallas_arena.probe.sampler import extract_program  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILURES.append(f"{name}: {detail}")


# --------------------------------------------------------------- candidates
NO_KERNEL = "import jax\nx = 1\n"
SYNTAX_BOOM = "def kernel(q, k, v, segment_ids)\n    return q\n"
BANNED = (
    "import jax\n"
    "from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sak\n"
    "def kernel(q, k, v, segment_ids):\n    return sak\n"
)
NO_PALLAS_SPLASH = (
    "import jax\nimport jax.numpy as jnp\n"
    "def kernel(q, k, v, segment_ids):\n"
    "    return jnp.einsum('hqd,hkd->hqd', q.astype(jnp.float32), k.astype(jnp.float32))\n"
)
WRONG_SHAPE_FLCE = (
    "import jax.numpy as jnp\n"
    "def kernel(hidden, w, targets):\n"
    "    return jnp.zeros((hidden.shape[0] + 1,), jnp.float32)\n"
)
HONEST_FLCE = '''
import jax
import jax.numpy as jnp

_TILE = 512

def _tile_lp(h_tile, t_tile, w):
    logits = (h_tile.astype(jnp.float32) @ w.astype(jnp.float32))
    tl = jnp.take_along_axis(logits, t_tile[:, None], axis=1)[:, 0]
    return tl - jax.nn.logsumexp(logits, axis=-1)

def kernel(hidden, w, targets):
    n, hdim = hidden.shape
    tile = min(_TILE, n)
    num_tiles = -(-n // tile)
    pad = num_tiles * tile - n
    flat = jnp.pad(hidden, ((0, pad), (0, 0))) if pad else hidden
    tgt = jnp.pad(targets, ((0, pad),)) if pad else targets
    hf = flat.reshape(num_tiles, tile, hdim)
    tf = tgt.reshape(num_tiles, tile)

    def body(c, xs):
        return c, _tile_lp(xs[0], xs[1], w)

    @jax.custom_vjp
    def _tiled(hf):
        _, lp = jax.lax.scan(body, None, (hf, tf))
        return lp

    def _f(hf):
        _, lp = jax.lax.scan(body, None, (hf, tf))
        return lp, hf

    def _b(res, cot):
        def bb(c, xs):
            h_t, t_t, c_t = xs
            _, vjp = jax.vjp(lambda h: _tile_lp(h, t_t, w), h_t)
            return c, vjp(c_t)[0]
        _, g = jax.lax.scan(bb, None, (res, tf, cot))
        return (g,)

    _tiled.defvjp(_f, _b)
    return _tiled(hf).reshape(num_tiles * tile)[:n]
'''

EXPECT = {
    "splash_attention": [
        ("no-kernel", NO_KERNEL, {"exec"}),
        ("syntax-error", SYNTAX_BOOM, {"exec"}),
        ("banned-import", BANNED, {"ast"}),
        ("no-pallas-call", NO_PALLAS_SPLASH, {"ast"}),
    ],
    "flce": [
        ("no-kernel", NO_KERNEL, {"exec"}),
        ("wrong-output-shape", WRONG_SHAPE_FLCE, {"aot_export", "pass"}),
        ("honest-tiled-custom-vjp", HONEST_FLCE, {"pass"}),
    ],
}


def main() -> int:
    print("=== prompts ===", flush=True)
    for row in prompt_table():
        print(f"  {row['task']:18s} {row['variant']:10s} {row['chars']:6d} chars "
              f"(~{row['approx_tokens']} tokens)", flush=True)
    for task, vs in PROMPTS.items():
        check(f"{task}: three variants present", set(vs) == set(C.VARIANTS), sorted(vs))
        check(f"{task}: reference nests minimal", vs["reference"].startswith(vs["minimal"][:400]))
        check(f"{task}: tailored nests reference", vs["tailored"].startswith(vs["reference"][:400]))
        check(f"{task}: minimal < reference < tailored",
              len(vs["minimal"]) < len(vs["reference"]) < len(vs["tailored"]),
              (len(vs["minimal"]), len(vs["reference"]), len(vs["tailored"])))
        for v, p in vs.items():
            check(f"{task}/{v}: declares every probe shape",
                  all(_shape_mentioned(task, s, p) for s in C.TASK_CASES[task]))

    print("=== code extraction ===", flush=True)
    code, how = extract_program("blah\n```python\nimport jax\ndef kernel(): pass\n```\ntrailing")
    check("extract fenced", how == "fenced" and code.startswith("import jax"), how)
    code, how = extract_program("```python\nfirst = 1\n```\nthen\n```python\ndef kernel(): pass\n```")
    check("extract takes the LAST block", "def kernel" in code, how)
    check("extract on prose only", extract_program("I cannot help with that.")[1] == "none")

    print("=== signatures ===", flush=True)
    sigs = {}
    for task in C.TASKS:
        s, case_sig, adv_sig, grad_sig = probe_signatures(task, C.TASK_CASES[task])
        sigs[task] = s
        print(f"  {task}: {len(s)} signatures, grad={grad_sig}", flush=True)
        for sg in s:
            print(f"    {sg['name']:8s} {sg['label']:32s} "
                  f"{[tuple(a['shape']) for a in sg['args']]}", flush=True)
        check(f"{task}: one signature per declared case", len(case_sig) == len(C.TASK_CASES[task]))
        check(f"{task}: adversarial shapes excluded", not adv_sig)
    check("flce exports a gradient signature",
          any(sg["kind"] == "grad" for sg in sigs["flce"]))
    check("splash exports no gradient signature (fwd-only contract)",
          not any(sg["kind"] == "grad" for sg in sigs["splash_attention"]))

    print("=== pre-gate on hand-written candidates ===", flush=True)
    for task, cases in EXPECT.items():
        for name, code, allowed in cases:
            res = pregate_one(task, code, sigs[task], timeout_s=600.0)
            gate = "pass" if res.get("passed") else res.get("gate")
            obs = res.get("observation") or ""
            check(f"{task}/{name} gate", gate in allowed, f"got {gate!r} (allowed {sorted(allowed)})")
            if not res.get("passed"):
                check(f"{task}/{name} observation <=500 chars", len(obs) <= 500, len(obs))
                print(f"    observation:\n      " + obs.replace("\n", "\n      "), flush=True)

    print("=== metrics ===", flush=True)
    recs = []
    for i in range(16):  # a uniformly-zero group
        recs.append({"config": "m|minimal|flce", "group": "g0", "stage": "pregate", "gate": "ast", "reward": 0.0,
                     "variant": "minimal", "model": "m", "task": "flce", "pregate_passed": False})
    for i in range(16):  # a group with one live candidate
        recs.append({"config": "m|tailored|flce", "group": "g1", "stage": "judge",
                     "gate": "all" if i == 3 else "correctness", "reward": 0.71 if i == 3 else 0.0,
                     "variant": "tailored", "model": "m", "task": "flce", "pregate_passed": True})
    s = summarize(recs, group_size=16)
    check("headline counts complete groups", s["headline"]["overall_nonuniform_groups"] == "1/2",
          s["headline"]["overall_nonuniform_groups"])
    check("uniform-zero config is flagged untrainable",
          s["per_config"]["m|minimal|flce"]["nonuniform_group_frac"] == 0.0)
    check("mixed config is flagged trainable",
          s["per_config"]["m|tailored|flce"]["nonuniform_group_frac"] == 1.0)
    check("correctness rate is 1/16", abs(s["per_config"]["m|tailored|flce"]["correctness_rate"] - 1 / 16) < 1e-9)
    check("score histogram separates the zeros",
          s["per_config"]["m|tailored|flce"]["score_histogram"]["=0"] == 15)

    print("\n=== dry run: " + (f"{len(FAILURES)} FAILURES" if FAILURES else "ALL GREEN") + " ===", flush=True)
    for f in FAILURES:
        print("  " + f, flush=True)
    Path("/tmp/probe-dryrun.json").write_text(json.dumps({"failures": FAILURES}, indent=1))
    return 1 if FAILURES else 0


def _shape_mentioned(task: str, case_name: str, prompt: str) -> bool:
    """The prompt must state the numbers of every declared case."""
    from pallas_arena.judge.problems import get_problem

    dims = get_problem(task).case_by_name(case_name).dims
    keys = ("heads", "seq", "d") if task == "splash_attention" else ("n", "h", "v")
    return all(str(dims[k]) in prompt for k in keys if k in dims)


if __name__ == "__main__":
    raise SystemExit(main())
