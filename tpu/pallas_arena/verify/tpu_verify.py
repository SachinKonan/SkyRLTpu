"""On-chip adversarial verification of the arena's first above-1.0 candidate.

Runs on the v6e-1 judge host. Four independent questions, one boot:

  1. REPRODUCIBILITY. A fresh `PersistentWorker` boot -- its OWN noise floor and
     ref-vs-ref for these shapes -- then N repeat grades of each above-1.0
     candidate with the reward cache OFF. 8 samples with a 1.19 max could be an
     outlier; 5 repeats of the same kernel cannot be.
  2. GATE STRENGTH. The probe ran `--no-adversarial`. Here the adversarial
     library is ON, built on a shape long enough to cross chunk boundaries
     (the library's default `tiny` is t=64, which with CHUNK=256 is a SINGLE
     chunk -- the carry the seam exists to test is never exercised).
  3. WORKSPACE. `peak_bytes_in_use` on the judge is a PROCESS-cumulative peak,
     so the observation's "peak HBM 28.63GB" cannot discriminate candidates.
     Each kernel is therefore measured alone, in a fresh child process.
  4. CEILING + ROOFLINE. A CHUNK sweep against the baseline and the fp32
     sequential reference, plus the fraction of v6e HBM speed-of-light both
     numerator and denominator achieve. If both are 10% of roofline, "1.19x" is
     a race between two slow programs and the task's real headroom is elsewhere.

Usage (on the judge host):
    python -m pallas_arena.verify.tpu_verify --codes codes.json --out out.json
    python -m pallas_arena.verify.tpu_verify --mem-only <name> --codes codes.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

CASES = ["probe-4x2048x2560", "probe-2x1024x2560", "probe-holdout-2x1500x2560"]
SCORED = CASES[:2]


def set_cases(names: list[str]) -> None:
    """CPU smoke path: run the whole script against the tiny battery shapes so
    every code path is exercised before a chip is spent."""
    global CASES, SCORED
    from pallas_arena.judge.problems import get_problem

    problem = get_problem("rg_lru")
    CASES = names
    SCORED = [n for n in names if not problem.case_by_name(n).holdout]


def _now():
    return time.strftime("%H:%M:%S")


# --------------------------------------------------------------- 3. workspace
def mem_only(name: str, codes: dict) -> None:
    """Run ONE kernel, alone, in this process, and print its peak HBM."""
    import jax
    import jax.numpy as jnp  # noqa: F401

    from pallas_arena.judge.problems import get_problem

    problem = get_problem("rg_lru")
    dev = jax.devices()[0]
    if name == "__baseline__":
        fn = problem.baseline
    elif name == "__reference__":
        fn = problem.reference
    else:
        ns: dict = {}
        exec(compile(codes[name], "<candidate>", "exec"), ns)  # noqa: S102
        fn = ns["kernel"]
    def stats():  # memory_stats() is None on CPU and on some backends
        try:
            return dev.memory_stats() or {}
        except Exception:
            return {}

    jfn = jax.jit(fn)
    peaks = {}
    for case_name in SCORED:
        case = problem.case_by_name(case_name)
        inputs = problem.make_inputs(jax.random.PRNGKey(0), case)
        jax.block_until_ready(inputs)
        base = stats().get("peak_bytes_in_use", 0)
        out = jfn(*inputs)
        jax.block_until_ready(out)
        peaks[case_name] = {
            "peak_bytes_in_use": int(stats().get("peak_bytes_in_use", 0)),
            "peak_before_run": int(base),
            "bytes_in_use": int(stats().get("bytes_in_use", 0)),
            "input_bytes": int(sum(int(a.size) * a.dtype.itemsize for a in inputs)),
        }
        del out
    print("MEMJSON " + json.dumps({"name": name, "peaks": peaks}))


# ------------------------------------------------------- 4. ceiling, roofline
def _chunked(chunk: int):
    import jax
    import jax.numpy as jnp

    def kernel(x, a, reset):
        b, t, d = x.shape
        ch = max(1, min(int(chunk), t))
        nch = -(-t // ch)
        pad = nch * ch - t
        if pad:
            x = jnp.pad(x, ((0, 0), (0, pad), (0, 0)))
            a = jnp.pad(a, ((0, 0), (0, pad), (0, 0)))
            reset = jnp.pad(reset, ((0, 0), (0, pad)))
        a_eff = a * (1.0 - reset[..., None].astype(a.dtype))
        xs = jnp.moveaxis(x.reshape(b, nch, ch, d), 1, 0)
        as_ = jnp.moveaxis(a_eff.reshape(b, nch, ch, d), 1, 0)

        def combine(l, r):
            return l[0] * r[0], l[1] * r[0] + r[1]

        def body(h_prev, ins):
            xc, ac = ins
            g = jnp.sqrt(jnp.maximum(1.0 - jnp.square(ac), 0.0)) * xc.astype(jnp.float32)
            A, G = jax.lax.associative_scan(combine, (ac, g), axis=1)
            hc = A * h_prev[:, None, :] + G
            return hc[:, -1, :], hc

        _, hs = jax.lax.scan(body, jnp.zeros((b, d), jnp.float32), (xs, as_))
        return jnp.moveaxis(hs, 0, 1).reshape(b, nch * ch, d)[:, :t, :]

    return kernel


def bench(fn, inputs, iters=12, warmup=3):
    import jax

    jfn = jax.jit(fn)
    for _ in range(warmup):
        jax.block_until_ready(jfn(*inputs))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(jfn(*inputs))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def ceiling_sweep(codes: dict) -> dict:
    import jax

    from pallas_arena.judge.problems import get_problem
    from pallas_arena.judge.problems.rg_lru import rg_lru_associative, rg_lru_scan_reference

    problem = get_problem("rg_lru")
    out = {}
    for case_name in SCORED:
        case = problem.case_by_name(case_name)
        inputs = problem.make_inputs(jax.random.PRNGKey(0), case)
        jax.block_until_ready(inputs)
        bm = problem.bytes_moved(case)
        row = {"bytes_moved": bm, "impls": {}}

        def rec(label, fn):
            try:
                lat = bench(fn, inputs)
            except Exception as e:
                row["impls"][label] = {"error": f"{type(e).__name__}: {e}"}
                return
            row["impls"][label] = {
                "median_s": lat,
                "ms": lat * 1e3,
                "sol_frac": (bm / lat) / 1.6e12 if bm else None,
            }

        rec("baseline(lax.associative_scan)", problem.baseline)
        rec("fp32-sequential-lax.scan", rg_lru_scan_reference)
        rec("assoc-full", rg_lru_associative)
        for ch in (64, 128, 256, 512, 1024, 2048):
            rec(f"chunked-assoc-{ch}", _chunked(ch))
        for name, code in codes.items():
            ns: dict = {}
            try:
                exec(compile(code, f"<{name}>", "exec"), ns)  # noqa: S102
            except Exception as e:
                row["impls"][f"cand:{name}"] = {"error": f"{type(e).__name__}: {e}"}
                continue
            rec(f"cand:{name}", ns["kernel"])
        base = row["impls"].get("baseline(lax.associative_scan)", {}).get("median_s")
        if base:
            for k, v in row["impls"].items():
                if "median_s" in v:
                    v["speedup_vs_baseline"] = base / v["median_s"]
        out[case_name] = row
    return out


# ------------------------------------------- 1 + 2. repeat grades, adversarial
def repeat_grades(codes: dict, repeats: int, adversarial_case: str, timing_pairs: int) -> dict:
    from pallas_arena.judge.worker import PersistentWorker

    res = {"boots": {}, "grades": {}}
    for adv_on, adv_case in ((False, None), (True, adversarial_case)):
        tag = "adversarial-on" if adv_on else "adversarial-off"
        print(f"\n=== [{_now()}] fresh judge boot, {tag} ===", flush=True)
        w = PersistentWorker(
            "rg_lru",
            cases=CASES,
            use_adversarial=adv_on,
            adversarial_case=adv_case,
            timing_pairs=timing_pairs,
            grade_budget_s=900.0,
            compile_budget_s=600.0,
            cache=None,  # NEVER cache: repeats must be independent measurements
        )
        boot = w.boot()
        res["boots"][tag] = boot
        print(f"[boot {tag}] {json.dumps(boot, default=str)[:700]}", flush=True)
        if not boot.get("ok"):
            continue
        for name, code in codes.items():
            key = f"{tag}|{name}"
            res["grades"][key] = []
            n = repeats if not adv_on else max(1, repeats // 2)
            for i in range(n):
                t0 = time.perf_counter()
                r = w.grade_code(code, tag=f"verify-{name}-{i}")
                rec = {
                    "i": i,
                    "passed": bool(r.get("passed")),
                    "gate": r.get("gate"),
                    "reward": r.get("reward"),
                    "score": r.get("score"),
                    "per_case": r.get("per_case"),
                    "holdout": r.get("holdout"),
                    "sol": r.get("speed_of_light_fracs"),
                    "latencies": r.get("latencies"),
                    "violations": (r.get("violations") or [])[:2],
                    "wall_s": time.perf_counter() - t0,
                }
                res["grades"][key].append(rec)
                print(f"[{tag}] {name} #{i}: gate={rec['gate']} reward={rec['reward']} "
                      f"per_case={rec['per_case']} viol={rec['violations']}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, help="json {name: source}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mem-only", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--timing-pairs", type=int, default=20)
    # The library's default is `tiny` (t=64): with any CHUNK >= 64 that is a
    # SINGLE chunk, so the carry between chunks -- the one thing the seam lets
    # the model decide -- is never exercised by any adversarial vector. Rebuild
    # the library on a DECLARED probe shape instead: 4 chunks at CHUNK=256, and
    # no new exported signature, because the shape is already in the contract.
    ap.add_argument("--adversarial-case", default="probe-2x1024x2560")
    ap.add_argument("--skip", default="", help="comma list of: grades,ceiling,mem")
    ap.add_argument("--cases", default=None, help="comma list; CPU smoke uses the tiny battery")
    args = ap.parse_args()

    if args.cases:
        set_cases(args.cases.split(","))
    codes = json.load(open(args.codes))
    if args.mem_only:
        mem_only(args.mem_only, codes)
        return 0

    skip = set(x for x in args.skip.split(",") if x)
    report = {"host": os.uname().nodename, "started": _now()}
    import jax

    report["device_kind"] = jax.devices()[0].device_kind
    report["backend"] = jax.default_backend()
    print(f"device: {report['device_kind']} backend={report['backend']}", flush=True)

    if "ceiling" not in skip:
        print(f"\n=== [{_now()}] ceiling sweep + roofline ===", flush=True)
        report["ceiling"] = ceiling_sweep(codes)
        for case, row in report["ceiling"].items():
            print(f"-- {case} (bytes_moved={row['bytes_moved']})")
            for k, v in sorted(row["impls"].items(), key=lambda kv: kv[1].get("median_s", 9e9)):
                if "error" in v:
                    print(f"   {k:34s} ERROR {v['error'][:80]}")
                else:
                    print(f"   {k:34s} {v['ms']:8.3f} ms  x{v.get('speedup_vs_baseline', 0):5.3f}  "
                          f"SoL {100 * (v['sol_frac'] or 0):5.1f}%")

    if "mem" not in skip:
        print(f"\n=== [{_now()}] per-kernel peak HBM, each alone in a fresh process ===", flush=True)
        report["memory"] = {}
        for name in ["__baseline__", "__reference__", *codes]:
            cmd = [sys.executable, "-m", "pallas_arena.verify.tpu_verify",
                   "--codes", args.codes, "--mem-only", name]
            if args.cases:
                cmd += ["--cases", args.cases]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            line = next((l for l in p.stdout.splitlines() if l.startswith("MEMJSON ")), None)
            if line:
                d = json.loads(line[len("MEMJSON "):])
                report["memory"][name] = d["peaks"]
                for c, v in d["peaks"].items():
                    print(f"   {name:44s} {c:26s} peak {v['peak_bytes_in_use'] / 2**30:7.3f} GB "
                          f"(inputs {v['input_bytes'] / 2**20:.0f} MB)")
            else:
                report["memory"][name] = {"error": (p.stderr or p.stdout)[-500:]}
                print(f"   {name}: FAILED {(p.stderr or p.stdout)[-200:]}")

    if "grades" not in skip:
        report["verification"] = repeat_grades(codes, args.repeats, args.adversarial_case, args.timing_pairs)
        for key, rows in report["verification"]["grades"].items():
            rewards = [r["reward"] for r in rows if r.get("reward") is not None]
            if rewards:
                print(f"[summary] {key}: n={len(rewards)} rewards={['%.4f' % x for x in rewards]} "
                      f"median={statistics.median(rewards):.4f}")

    report["finished"] = _now()
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1, default=str)
        print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
