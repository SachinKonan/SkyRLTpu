"""CONTROL EXPERIMENT: push KNOWN-GOOD kernels through the EXACT probe path.

A grid of zeros is only evidence about models if the path itself is known to
pass a correct kernel. So this takes kernels we already trust, wraps each in a
model-STYLE response (prose, then a fenced block, exactly what a generation
looks like), and runs the identical pipeline the generations run:

    extract_program  ->  pregate_one  ->  queue.submit  ->  judge verdict

If a known-good kernel fails here, the grid is measuring a harness bug. If it
passes, the zeros are about the models.

Includes `rmsnorm/pallas-br256` — proven on silicon in phases 2, 4 and 5 — as
a plumbing control that is independent of the two probe tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pallas_arena.probe import configs as C  # noqa: E402
from pallas_arena.probe.pregate import pregate_one, probe_signatures  # noqa: E402
from pallas_arena.probe.sampler import extract_program  # noqa: E402

# --------------------------------------------------------------- FLCE (good)
FLCE_GOOD = '''
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

# -------------------------------------------------------------- splash (good)
# The prompt's own tailored scaffold with the ONE blank filled in: a standard
# online-softmax (flash) step over key blocks.
SPLASH_GOOD = '''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_LANES = 128
_SUBLANES = 8
_BQ = 512
_BK = 512
_VMEM_BUDGET = 24 * 1024 * 1024
_BYTES_PER_ELT = 24
_NEG = -1e30


def _scoped_bytes(bq, bk, d):
    io = 2 * (bq * d * 2 + 2 * bk * d * 2 + bq * d * 4 + bq * _LANES * 4 + _SUBLANES * bk * 4)
    scratch = 2 * bq * _LANES * 4
    return io + scratch + bq * bk * _BYTES_PER_ELT


def _pick_blocks(d):
    bq, bk = _BQ, _BK
    while bk > 128 and _scoped_bytes(bq, bk, d) > _VMEM_BUDGET:
        bk //= 2
    while bq > 128 and _scoped_bytes(bq, bk, d) > _VMEM_BUDGET:
        bq //= 2
    return bq, bk


def _make_kernel(bq, bk, n_k):
    def _kern(q_ref, k_ref, v_ref, sq_ref, sk_ref, o_ref, m_ref, l_ref):
        i = pl.program_id(1)
        j = pl.program_id(2)

        @pl.when(j == 0)
        def _init():
            m_ref[...] = jnp.full_like(m_ref, _NEG)
            l_ref[...] = jnp.zeros_like(l_ref)
            o_ref[...] = jnp.zeros_like(o_ref)

        q = q_ref[0].astype(jnp.float32)
        k = k_ref[0].astype(jnp.float32)
        v = v_ref[0].astype(jnp.float32)
        s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())))

        seg_q = sq_ref[:, :1]
        seg_k = sk_ref[:1, :]
        pos_q = i * bq + jax.lax.broadcasted_iota(jnp.int32, (bq, 1), 0)
        pos_k = j * bk + jax.lax.broadcasted_iota(jnp.int32, (1, bk), 1)
        vis = (pos_k <= pos_q) & (seg_q == seg_k) & (seg_q != 0)
        s = jnp.where(vis, s, _NEG)

        m_old = m_ref[:, :1]
        l_old = l_ref[:, :1]
        m_new = jnp.maximum(m_old, jnp.max(s, axis=1, keepdims=True))
        p = jnp.where(vis, jnp.exp(s - m_new), 0.0)
        corr = jnp.exp(m_old - m_new)
        l_new = corr * l_old + jnp.sum(p, axis=1, keepdims=True)
        o_new = corr * o_ref[0] + jax.lax.dot_general(p, v, (((1,), (0,)), ((), ())))
        m_ref[...] = jnp.broadcast_to(m_new, m_ref.shape)
        l_ref[...] = jnp.broadcast_to(l_new, l_ref.shape)
        o_ref[0] = o_new

        @pl.when(j == n_k - 1)
        def _fin():
            ll = l_ref[:, :1]
            o_ref[0] = jnp.where(ll > 0, o_ref[0] / jnp.maximum(ll, 1e-30), 0.0)

    return _kern


@jax.jit
def _fwd(q, k, v, segment_ids):
    h, s, d = q.shape
    dp = ((d + _LANES - 1) // _LANES) * _LANES
    bq, bk = _pick_blocks(dp)
    blk = max(bq, bk)
    sp = ((s + blk - 1) // blk) * blk
    pad_s, pad_d = sp - s, dp - d

    def _pad(x):
        if pad_s or pad_d:
            return jnp.pad(x, ((0, 0), (0, pad_s), (0, pad_d)))
        return x

    qp, kp, vp = _pad(q), _pad(k), _pad(v)
    segp = jnp.pad(segment_ids, (0, pad_s)) if pad_s else segment_ids
    seg_q = jnp.broadcast_to(segp[:, None], (sp, _LANES))
    seg_k = jnp.broadcast_to(segp[None, :], (_SUBLANES, sp))

    n_q, n_k = sp // bq, sp // bk
    out = pl.pallas_call(
        _make_kernel(bq, bk, n_k),
        grid=(h, n_q, n_k),
        in_specs=[
            pl.BlockSpec((1, bq, dp), lambda hh, i, j: (hh, i, 0)),
            pl.BlockSpec((1, bk, dp), lambda hh, i, j: (hh, j, 0)),
            pl.BlockSpec((1, bk, dp), lambda hh, i, j: (hh, j, 0)),
            pl.BlockSpec((bq, _LANES), lambda hh, i, j: (i, 0)),
            pl.BlockSpec((_SUBLANES, bk), lambda hh, i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((1, bq, dp), lambda hh, i, j: (hh, i, 0)),
        out_shape=jax.ShapeDtypeStruct((h, sp, dp), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((bq, _LANES), jnp.float32),
            pltpu.VMEM((bq, _LANES), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(qp, kp, vp, seg_q, seg_k)
    return out[:, :s, :d]


def kernel(q, k, v, segment_ids):
    return _fwd(q, k, v, segment_ids)
'''


def model_style_response(code: str) -> str:
    """Wrap a program the way a model actually answers, so the control
    exercises extraction too — not just the grader."""
    return (
        "Let me think about the block sizes first.\n\n"
        "Here is a sketch I rejected:\n\n"
        "```python\n# rejected: materializes the whole score matrix\n"
        "def kernel(*a):\n    raise NotImplementedError\n```\n\n"
        "Now the real implementation:\n\n"
        "```python\n" + code.strip() + "\n```\n"
    )


def run(task: str, code: str, cases: list[str], queue_url: str | None, label: str) -> dict:
    resp = model_style_response(code)
    extracted, how = extract_program(resp)
    out = {
        "label": label,
        "task": task,
        "extraction": how,
        "extracted_chars": len(extracted),
        "extracted_is_the_real_one": "raise NotImplementedError" not in extracted,
        "has_def_kernel": "def kernel" in extracted,
    }
    sigs, *_ = probe_signatures(task, cases)
    out["n_signatures"] = len(sigs)
    t0 = time.time()
    v = pregate_one(task, extracted, sigs, timeout_s=600.0)
    out["pregate_s"] = round(time.time() - t0, 2)
    out["pregate_passed"] = bool(v.get("passed"))
    out["pregate_gate"] = v.get("gate")
    out["pregate_observation"] = v.get("observation")
    print(f"[control] {label}: extraction={how} real_block={out['extracted_is_the_real_one']} "
          f"pregate_passed={out['pregate_passed']} gate={v.get('gate')} ({out['pregate_s']}s)", flush=True)
    if not out["pregate_passed"]:
        print("           " + (v.get("observation") or "").replace("\n", "\n           "), flush=True)
        return out
    if not queue_url:
        return out
    from pallas_arena.judge.queue import ArenaQueueClient

    cl = ArenaQueueClient(queue_url, timeout_s=120.0)
    wid = cl.submit(task, extracted, tag=f"control:{label}")
    print(f"[control] {label}: submitted {wid}, awaiting the judge...", flush=True)
    res = cl.wait([wid], timeout_s=900.0, poll_s=5.0)
    r = res.get(wid) or {}
    out["judge_gate"] = r.get("gate")
    out["judge_passed"] = bool(r.get("passed"))
    out["judge_reward"] = r.get("reward")
    out["judge_observation"] = r.get("observation")
    out["judge_violations"] = r.get("violations")
    print(f"[control] {label}: JUDGE passed={out['judge_passed']} gate={out['judge_gate']} "
          f"reward={out['judge_reward']}", flush=True)
    print("           " + (out["judge_observation"] or "").replace("\n", "\n           "), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma list of labels")
    args = ap.parse_args()

    plan = [
        # plumbing control, independent of the two probe tasks: this exact
        # kernel passed on silicon in phases 2, 4 and 5.
        ("rmsnorm-pallas-br256", "rmsnorm", None, ["tiny"], True),
        ("flce-known-good", "flce", FLCE_GOOD, C.TASK_CASES["flce"], False),
        ("splash-scaffold-filled", "splash_attention", SPLASH_GOOD, C.TASK_CASES["splash_attention"], False),
    ]
    only = set(args.only.split(",")) if args.only else None
    results = []
    for label, task, code, cases, is_rmsnorm in plan:
        if only and label not in only:
            continue
        if is_rmsnorm:
            from pallas_arena.phase2.variants import PALLAS_BR256

            code = PALLAS_BR256
        # rmsnorm is not served by the probe judge, so it is pre-gate only
        results.append(run(task, code, cases, None if is_rmsnorm else args.queue, label))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    bad = [r for r in results if not r["pregate_passed"]]
    print(f"\n[control] {len(results) - len(bad)}/{len(results)} passed the pre-gate", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
