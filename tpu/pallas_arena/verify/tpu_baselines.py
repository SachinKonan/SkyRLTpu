"""On-chip: settle the three things CPU cannot answer about the arena's denominators.

Runs on ONE TPU judge host. No generation, no RL, no candidate code -- only the
arena's own references, baselines and honest variants, plus tokamax's production
kernels.

A. BAND ON SILICON. `preferred_element_type` is a no-op inside a fused op on CPU,
   so the CPU tolerance probe proved nothing for megablox (its "sloppy" variant
   read 0.67x and its faithful one exactly 0.0). Here the same three
   implementations run on a real MXU:
     A  reference_bf16      -- what the band was calibrated from before
     B  production-faithful -- bf16 matmul INPUTS, fp32 accumulators
     C  sloppy bf16         -- no preferred_element_type anywhere
   The band (now including honest_variants) must admit B and reject C. On CPU
   that holds for splash/RPA/flce/rg_lru; megablox is the open question.

B. WHICH BASELINE ACTUALLY RAN. Four of five tasks fall back to a labelled
   non-production denominator when the real kernel refuses a shape or is absent
   from the host. `baseline_impl` is recorded per task so a score is never
   reported as "beat the production kernel" when it beat `lax.ragged_dot`.

C. TOKAMAX AS A REAL DENOMINATOR. openxla/tokamax ships production TPU Pallas
   kernels and pins jax>=0.9.2, which resolves to the arena's own 0.10.2:
     megablox -> tokamax.ragged_dot(implementation='mosaic')
     splash   -> tokamax.dot_product_attention(implementation='mosaic')
     flce     -> tokamax.linear_softmax_cross_entropy_loss(
                     implementation='mosaic_tpu')   [TIMING ONLY -- its public
                 API offers reduction sum/mean but not 'none', so it returns a
                 scalar where our contract returns per-token logprobs]
   For each: does it run at our shapes, does it agree with our fp32 reference
   inside the calibrated band, and how fast is it against the current baseline.
   Anything that fails is REPORTED, never silently skipped -- a missing
   denominator is the finding.

Nothing here changes a task definition. It produces the evidence for that call.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from pallas_arena.judge.problems import get_problem
from pallas_arena.judge.problems.base import BaselineUnavailable, error_stats

F32 = jnp.float32
NEG = -0.7 * float(np.finfo(np.float32).max)


# ----------------------------------------------------------------- A: variants
def _splash_mask(seg, seq):
    idx = jnp.arange(seq)
    live = seg != 0
    return (idx[:, None] >= idx[None, :]) & (seg[:, None] == seg[None, :]) & live[:, None] & live[None, :]


def splash_faithful(q, k, v, seg):
    m = _splash_mask(seg, q.shape[1])
    logits = jnp.einsum("hqd,hkd->hqk", q, k, preferred_element_type=F32)
    logits = jnp.where(m[None], logits, NEG)
    row_live = m.any(-1)
    p = jnp.where(m[None], jnp.exp(logits - logits.max(-1, keepdims=True)), 0.0)
    p = jnp.where(row_live[None, :, None], p / jnp.maximum(p.sum(-1, keepdims=True), 1e-30), 0.0)
    return jnp.einsum("hqk,hkd->hqd", p.astype(jnp.bfloat16), v, preferred_element_type=F32)


def splash_sloppy(q, k, v, seg):
    m = _splash_mask(seg, q.shape[1])
    logits = jnp.einsum("hqd,hkd->hqk", q, k).astype(jnp.bfloat16)
    logits = jnp.where(m[None], logits, jnp.bfloat16(-3e38))
    row_live = m.any(-1)
    p = jnp.where(m[None], jnp.exp((logits - logits.max(-1, keepdims=True)).astype(jnp.bfloat16)), jnp.bfloat16(0))
    den = jnp.maximum(p.sum(-1, keepdims=True, dtype=jnp.bfloat16), jnp.bfloat16(1e-30))
    p = jnp.where(row_live[None, :, None], p / den, jnp.bfloat16(0))
    return jnp.einsum("hqk,hkd->hqd", p, v).astype(F32)


def gmm_faithful(lhs, rhs, gs):
    return jax.lax.ragged_dot(lhs, rhs, gs, preferred_element_type=F32)


def gmm_sloppy(lhs, rhs, gs):
    return jax.lax.ragged_dot(lhs, rhs, gs, preferred_element_type=jnp.bfloat16).astype(F32)


def _rpa(q, kp, vp, pt, sl, accum):
    b, qh, d = q.shape
    _, ps, kvh, _ = kp.shape
    group = qh // kvh
    ml = pt.shape[1] * ps
    k = kp[pt].reshape(b, ml, kvh, d)
    v = vp[pt].reshape(b, ml, kvh, d)
    logits = jnp.einsum("bhgd,bthd->bhgt", q.reshape(b, kvh, group, d), k, preferred_element_type=accum)
    live = jnp.arange(ml)[None, :] < sl[:, None]
    big = NEG if accum == F32 else jnp.bfloat16(-3e38)
    logits = jnp.where(live[:, None, None, :], logits, big)
    p = jnp.exp((logits - logits.max(-1, keepdims=True)).astype(accum))
    p = jnp.where(live[:, None, None, :], p, jnp.asarray(0, accum))
    p = p / jnp.maximum(p.sum(-1, keepdims=True, dtype=accum), jnp.asarray(1e-30, accum))
    o = jnp.einsum("bhgt,bthd->bhgd", p.astype(jnp.bfloat16), v, preferred_element_type=accum)
    return o.reshape(b, qh, d).astype(F32)


def _flce(hidden, w, targets, tile, accum):
    n = hidden.shape[0]
    nt = -(-n // tile)
    pad = nt * tile - n
    hf = jnp.pad(hidden, ((0, pad), (0, 0))) if pad else hidden
    tf = jnp.pad(targets, ((0, pad),)) if pad else targets
    hf, tf = hf.reshape(nt, tile, -1), tf.reshape(nt, tile)

    def body(_c, xs):
        h_t, t_t = xs
        lg = jnp.dot(h_t, w, preferred_element_type=accum)
        tl = jnp.take_along_axis(lg, t_t[:, None], axis=1)[:, 0]
        return None, (tl - jax.nn.logsumexp(lg, axis=-1)).astype(F32)

    _, lp = jax.lax.scan(body, None, (hf, tf))
    return lp.reshape(nt * tile)[:n]


def _rglru_sloppy(x, a, reset):
    from pallas_arena.judge.problems.rg_lru import _apply_reset

    a32 = _apply_reset(a.astype(F32), reset)
    gx = (jnp.sqrt(jnp.maximum(1.0 - jnp.square(a32), 0.0)).astype(jnp.bfloat16) * x.astype(jnp.bfloat16))
    a16 = a32.astype(jnp.bfloat16)

    def step(h, xs):
        a_t, gx_t = xs
        nh = (a_t * h + gx_t).astype(jnp.bfloat16)
        return nh, nh

    h0 = jnp.zeros((x.shape[0], x.shape[2]), jnp.bfloat16)
    _, hs = jax.lax.scan(step, h0, (jnp.moveaxis(a16, 1, 0), jnp.moveaxis(gx, 1, 0)))
    return jnp.moveaxis(hs, 0, 1).astype(F32)


def _rglru_faithful(x, a, reset):
    from pallas_arena.judge.problems.rg_lru import rg_lru_associative

    return rg_lru_associative(x, a, reset)


VARIANTS = {
    "splash_attention": (splash_faithful, splash_sloppy, None),
    "megablox_gmm": (gmm_faithful, gmm_sloppy, None),
    "ragged_paged_attention": (lambda *a: _rpa(*a, accum=F32), lambda *a: _rpa(*a, accum=jnp.bfloat16), None),
    "flce": (lambda *a: _flce(*a, accum=F32), lambda *a: _flce(*a, accum=jnp.bfloat16), "tile"),
    "rg_lru": (_rglru_faithful, _rglru_sloppy, None),
}


def band_check(problem, case, faithful, sloppy, extra):
    key = jax.random.PRNGKey(0)
    inputs = problem.make_inputs(key, case)
    jax.block_until_ready(inputs)
    ref32 = problem.reference(*inputs)
    jax.block_until_ready(ref32)
    tol = problem.calibrated_tolerance(inputs, ref32)
    args = list(inputs) + ([case.dims[extra]] if extra else [])

    out = {"case": case.name, "dims": {k: str(v) for k, v in case.dims.items()},
           "tol_max": float(tol["max"]), "tol_q99": float(tol["q99"])}
    for tag, fn in (("A_reference_bf16", lambda: problem.reference_bf16(*inputs)),
                    ("B_faithful", lambda: faithful(*args)),
                    ("C_sloppy", lambda: sloppy(*args))):
        try:
            s = error_stats(jax.block_until_ready(fn()), ref32)
            out[tag] = {
                "max": float(s["max"]), "q99": float(s["q99"]),
                "ratio_max": float(s["max"] / tol["max"]) if tol["max"] else None,
                "pass": bool(s["max"] <= tol["max"] and s["q99"] <= tol["q99"]),
            }
        except Exception as e:  # noqa: BLE001
            out[tag] = {"error": f"{type(e).__name__}: {e}"}
    a, b, c = out.get("A_reference_bf16", {}), out.get("B_faithful", {}), out.get("C_sloppy", {})
    verdict = "inconclusive"
    if b.get("pass") and c.get("pass") is False:
        verdict = "BAND CORRECT (admits faithful, rejects sloppy)"
    elif b.get("pass") is False:
        verdict = "BAND TOO TIGHT (rejects the production numeric path)"
    elif c.get("pass"):
        # Before calling this a discrimination failure, check whether there is
        # anything to discriminate. A task with a SINGLE reduction (a fused
        # matmul) has no intermediate to lose: the MXU accumulates in fp32
        # regardless, so `preferred_element_type=bfloat16` costs exactly ONE
        # output rounding -- which is precisely what reference_bf16 already
        # models and deliberately allows. Measured on run 3689350: megablox's
        # sloppy error equals its reference_bf16 error to 4 significant
        # figures, and its faithful error is 0.0 (bit-identical to the fp32
        # reference). That is not a loose band, it is a task where the
        # accumulate-in-fp32 rule is a no-op. The rule still matters wherever
        # reductions CHAIN: attention (matmul->softmax->matmul), FLCE
        # (matmul->LSE), rg_lru (sequential scan).
        am, cm = a.get("max"), c.get("max")
        chained = not (am and cm and abs(cm - am) <= 0.01 * max(am, cm))
        verdict = (
            "BAND TOO LOOSE (cannot tell sloppy from faithful)"
            if chained
            else "NO ACCUMULATION CHAIN (single reduction; sloppy == one output rounding, already allowed)"
        )
    out["verdict"] = verdict
    return out


# ------------------------------------------------------------- B/C: baselines
def _import_tokamax():
    """tokamax pulls in absl and parses sys.argv LAZILY on first use, so an
    argparse flag of ours ('--out') reaches it and raises UnrecognizedFlagError
    at call time -- which reads exactly like "the kernel does not work here".
    Run 3689350 lost every tokamax measurement to this. Consume the flags with
    a clean argv first, then import."""
    import sys

    saved = sys.argv
    try:
        sys.argv = saved[:1]
        try:
            from absl import flags

            if not flags.FLAGS.is_parsed():
                flags.FLAGS(saved[:1], known_only=True)
        except Exception:  # absl absent or already parsed: fine
            pass
        import tokamax

        return tokamax
    finally:
        sys.argv = saved


def _time(fn, *args, pairs=10, warmup=2):
    f = jax.jit(fn)
    for _ in range(warmup):
        jax.block_until_ready(f(*args))
    ts = []
    for _ in range(pairs):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def tokamax_probe(name, problem, case):
    """Can tokamax's production kernel stand in as this task's denominator?"""
    res = {"task": name, "case": case.name}
    try:
        tokamax = _import_tokamax()
    except Exception as e:  # noqa: BLE001
        return {**res, "error": f"tokamax import failed: {type(e).__name__}: {e}"}

    key = jax.random.PRNGKey(0)
    inputs = problem.make_inputs(key, case)
    jax.block_until_ready(inputs)
    ref32 = problem.reference(*inputs)
    tol = problem.calibrated_tolerance(inputs, ref32)

    def attempt(label, fn, compare=None):
        try:
            out = jax.block_until_ready(jax.jit(fn)())
            entry = {"ran": True}
            cmp_ref = ref32 if compare is None else compare
            try:
                s = error_stats(out, cmp_ref)
                entry.update(max=float(s["max"]), q99=float(s["q99"]),
                             within_band=bool(s["max"] <= tol["max"] and s["q99"] <= tol["q99"]))
            except Exception as e:  # noqa: BLE001
                entry["agreement_error"] = f"{type(e).__name__}: {e}"
            entry["median_s"] = _time(fn)
            return entry
        except Exception as e:  # noqa: BLE001
            return {"ran": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}

    if name == "megablox_gmm":
        lhs, rhs, gs = inputs
        res["tokamax_mosaic"] = attempt(
            "ragged_dot",
            lambda: tokamax.ragged_dot(lhs, rhs, gs, preferred_element_type=F32, implementation="mosaic"),
        )
        res["tokamax_xla"] = attempt(
            "ragged_dot_xla",
            lambda: tokamax.ragged_dot(lhs, rhs, gs, preferred_element_type=F32, implementation="xla"),
        )
    elif name == "splash_attention":
        q, k, v, seg = inputs
        # ours is [heads, seq, d] (MHA, per-shard); tokamax wants [*B, T, N, h]
        qt, kt, vt = (jnp.swapaxes(x, 0, 1)[None] for x in (q, k, v))

        def _dpa(impl):
            def go():
                o = tokamax.dot_product_attention(qt, kt, vt, is_causal=True, implementation=impl)
                return jnp.swapaxes(o[0], 0, 1).astype(F32)

            return go

        # NOTE: causal only -- tokamax's public DPA has no segment_ids argument,
        # so this is NOT our contract (we additionally mask across segments and
        # require padded rows to be exactly 0). Recorded as such: agreement is
        # expected to FAIL, and the useful number here is the timing.
        res["note"] = "dot_product_attention has no segment_ids; causal-only, so disagreement is expected"
        res["tokamax_mosaic"] = attempt("dpa_mosaic", _dpa("mosaic"))
        res["tokamax_xla"] = attempt("dpa_xla", _dpa("xla"))
    elif name == "flce":
        hidden, w, targets = inputs
        # theirs returns a SCALAR loss (no reduction='none'); ours returns
        # per-token logprobs. The comparable quantity is -sum(logprobs).
        ours_sum = -jnp.sum(ref32)

        def _lsce(impl):
            return lambda: tokamax.linear_softmax_cross_entropy_loss(
                hidden, targets, w, reduction="sum", implementation=impl
            ).astype(F32)

        res["note"] = "timing-only: tokamax returns a scalar loss, our contract is per-token logprobs"
        res["ours_negsum"] = float(ours_sum)
        res["tokamax_mosaic_tpu"] = attempt("lsce_mosaic", _lsce("mosaic_tpu"), compare=ours_sum)
        res["tokamax_xla"] = attempt("lsce_xla", _lsce("xla"), compare=ours_sum)
    else:
        res["error"] = "no tokamax counterpart (paged attention / RG-LRU are not in the public API)"
    return res


def baseline_identity(name, problem, case):
    """Which denominator binds here -- and does it AGREE with the fp32
    reference. The agreement check is what catches a wrong layout adapter
    (e.g. a transposed K/V interleave in the RPA binding) before a single
    candidate is graded against it: an adapter bug produces garbage error,
    not a subtle bias, so within-band agreement is decisive."""
    out = {"task": name, "case": case.name}
    try:
        inputs = problem.make_inputs(jax.random.PRNGKey(0), case)
        jax.block_until_ready(inputs)
        o = jax.block_until_ready(problem.baseline(*inputs))
        try:
            ref32 = problem.reference(*inputs)
            tol = problem.calibrated_tolerance(inputs, ref32)
            s = error_stats(o, ref32)
            out["agrees_with_reference"] = bool(s["max"] <= tol["max"] and s["q99"] <= tol["q99"])
            out["agreement_max_err"] = float(s["max"])
            out["agreement_tol"] = float(tol["max"])
        except Exception as e:  # noqa: BLE001
            out["agreement_error"] = f"{type(e).__name__}: {str(e)[:150]}"
        # Tasks that can fall back record which denominator ran. FLCE defines
        # no fallback at all -- its baseline IS our production custom_vjp
        # kernel -- so a bare "?" there means something quite different from a
        # missing measurement, and must not read as "unknown".
        out["baseline_impl"] = getattr(type(problem), "baseline_impl", None) or (
            "task-own-production-kernel (no fallback path defined)"
        )
        out["median_s"] = _time(problem.baseline, *inputs)
        out["shape"] = str(np.asarray(o).shape) if not isinstance(o, (tuple, list)) else "tuple"
    except BaselineUnavailable as e:
        out["error"] = f"BaselineUnavailable: {e}"
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["baseline_impl"] = getattr(type(problem), "baseline_impl", "?")
    return out


def dispatch_overhead_probe(name, problem, case):
    """How much of our wallclock timing is Python dispatch, not kernel?

    Our reward is a RATIO measured with perf_counter + block_until_ready, so a
    constant per-call overhead c turns a true a/b into (a+c)/(b+c) -- every
    score compressed toward 1.0. tokamax defaults to device-level profiling on
    TPU for exactly this reason (wallclock is only their fallback).

    Measure it directly: time the baseline at k=1 and at k=8 chained calls
    (barrier-separated so XLA cannot collapse them). If per-call time falls as
    k rises, the difference IS the dispatch overhead, and the ratio distortion
    follows from it.
    """
    from pallas_arena.judge import timing as T

    out = {"task": name, "case": case.name}
    try:
        inputs = problem.make_inputs(jax.random.PRNGKey(0), case)
        jax.block_until_ready(inputs)

        def med(fn, n=7):
            f = jax.jit(fn)
            for _ in range(2):
                jax.block_until_ready(f(*inputs))
            ts = []
            for _ in range(n):
                t0 = time.perf_counter()
                jax.block_until_ready(f(*inputs))
                ts.append(time.perf_counter() - t0)
            return float(np.median(ts))

        t1 = med(problem.baseline)
        k = 8
        tk = med(T.amortized_call(problem.baseline, inputs, k))
        per_call_k = tk / k
        overhead = max(0.0, t1 - per_call_k)
        out.update(k1_s=t1, k8_total_s=tk, per_call_k8_s=per_call_k,
                   dispatch_overhead_s=overhead,
                   overhead_frac_of_k1=overhead / t1 if t1 else None)
        # what that overhead does to a TRUE 1.5x speedup measured our way
        a, b = per_call_k, per_call_k / 1.5
        out["true_1.5x_measures_as"] = (a + overhead) / (b + overhead)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="splash_attention,megablox_gmm,ragged_paged_attention,flce,rg_lru")
    ap.add_argument("--probe-cases", action="store_true", default=True,
                    help="use the one-chip PROBE cases (the fp32 reference cannot run at production shapes)")
    args = ap.parse_args()

    report = {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
              "backend": jax.default_backend(), "bands": [], "baselines": [], "tokamax": []}
    try:
        report["tokamax_version"] = getattr(_import_tokamax(), "__version__", "?")
    except Exception as e:  # noqa: BLE001
        report["tokamax_version"] = f"IMPORT FAILED: {type(e).__name__}: {e}"

    for name in args.tasks.split(","):
        name = name.strip()
        if not name:
            continue
        print(f"\n{'=' * 72}\n{name}", flush=True)
        try:
            problem = get_problem(name)
        except Exception as e:  # noqa: BLE001
            report["bands"].append({"task": name, "error": f"get_problem: {e}"})
            continue

        cases = [c for c in problem.shape_cases() if c.probe] or [c for c in problem.shape_cases() if c.smoke]
        faithful, sloppy, extra = VARIANTS[name]

        for case in cases:
            print(f"  [band] {case.name} ...", flush=True)
            try:
                row = band_check(problem, case, faithful, sloppy, extra)
                row["task"] = name
                report["bands"].append(row)
                print(f"  band {case.name}: {row['verdict']}", flush=True)
            except Exception:
                report["bands"].append({"task": name, "case": case.name, "error": traceback.format_exc()[-400:]})
                print(f"  band {case.name}: EXCEPTION", flush=True)

        # GENERAL mode elects a denominator PER SHAPE at boot; report the
        # election for every case, not just the first, since the whole point is
        # that the winner differs by shape.
        for probe_case in cases:
            b = baseline_identity(name, problem, probe_case)
            report["baselines"].append(b)
        probe_case = cases[0]
        b = report["baselines"][-len(cases)]
        print(f"  baseline: {b.get('baseline_impl', b.get('error'))} {b.get('median_s')}", flush=True)

        t = tokamax_probe(name, problem, probe_case)
        report["tokamax"].append(t)

        do = dispatch_overhead_probe(name, problem, probe_case)
        report.setdefault("dispatch", []).append(do)
        if "error" not in do:
            print(f"  dispatch: k1={do['k1_s'] * 1e3:.3f}ms per-call@k8={do['per_call_k8_s'] * 1e3:.3f}ms "
                  f"overhead={do['dispatch_overhead_s'] * 1e3:.3f}ms "
                  f"({(do['overhead_frac_of_k1'] or 0) * 100:.0f}%) -> a true 1.5x reads as "
                  f"{do['true_1.5x_measures_as']:.3f}", flush=True)
        print(f"  tokamax: {json.dumps({k: v for k, v in t.items() if k != 'case'})[:300]}", flush=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {args.out}", flush=True)

    print("\n=== SUMMARY ===")
    for r in report["bands"]:
        if "verdict" in r:
            print(f"  {r['task']:24s} {r['case']:26s} {r['verdict']}")
    for r in report["baselines"]:
        print(f"  {r['task']:24s} baseline={r.get('baseline_impl', 'ERR')}")


if __name__ == "__main__":
    main()
