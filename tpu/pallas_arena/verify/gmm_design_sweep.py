"""Is the megablox denominator sound when the kernel is used AS DESIGNED?

Retraction this exists to test. Run 3691370 reported `pallas-megablox-gmm` at
30-59x slower than XLA `ragged_dot` at every shape, and I concluded the kernel
was mistuned for v6e and the arena should swap its denominator. That conclusion
was measured with `jax.experimental...megablox.gmm(lhs, rhs, group_sizes)` --
i.e. the LIBRARY DEFAULT `tiling=(128, 128, 128)`. At our k=4096/n=14336/m=4096
that is a 32x32x112 grid, ~115k tile iterations of 4.2 MFLOP each. Nobody runs
megablox that way; the parameter even accepts a callable `(m,k,n) -> tiling` so
callers can supply a tuned choice, and tokamax ships a top-level `autotune` API
for exactly this reason. So the measurement said "I misconfigured the kernel",
not "the kernel is slow".

Two things are varied here, because BOTH were wrong before:

  * SHAPES -- tokamax's own canonical specs, not ours. Their arg_specs put real
    MoE at g=128-256 with m in the hundreds of thousands, plus named anchors
    (`compute_bound` 8/4096/4096/4096 and `8x7b` 8/8192/14336/4096, which is the
    TRANSPOSE of the orientation the arena declares). The arena's g=4 with
    n=14336 matches nothing they tune for.
  * TILING -- a real grid, not the default, reporting best-configured.

The question this answers: at the shapes and configuration the Google kernel was
designed for, does it beat XLA? If yes, a GMM env is defensible at those shapes
and the arena should move there. If a properly-configured megablox at its own
design shapes STILL loses to XLA, only then is swapping the denominator honest.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

F32 = jnp.float32
HBM_BUDGET_GB = 24.0


def group_sizes(key, g, m, dist):
    w = jnp.ones((g,)) if dist == "uniform" else jax.random.permutation(
        key, 1.0 / jnp.arange(1, g + 1, dtype=F32)
    )
    probs = w / jnp.sum(w)
    counts = jnp.floor(probs * m).astype(jnp.int32)
    return counts.at[jnp.argmax(probs)].add(m - jnp.sum(counts)).astype(jnp.int32)


def est_gb(m, g, k, n):
    return 2 * (m * k * 2 + g * k * n * 2 + m * n * 4) / 1e9


def timeit(fn, pairs=5, warmup=2):
    f = jax.jit(fn)
    for _ in range(warmup):
        jax.block_until_ready(f())
    ts = []
    for _ in range(pairs):
        t0 = time.perf_counter()
        jax.block_until_ready(f())
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# (m, k, n) tile candidates. The default is included so the retraction is
# visible in the same table as the tuned numbers.
TILINGS = [
    (128, 128, 128),  # library default -- what run 3691370 actually measured
    (256, 512, 512),
    (512, 512, 512),
    (512, 1024, 1024),
    (1024, 1024, 1024),
    (256, 1024, 2048),
]

# tokamax's canonical GMM specs (tokamax/_src/ops/ragged_dot/arg_specs.py),
# plus the arena's current declared orientation for direct comparison.
SHAPES = [
    ("tokamax compute_bound", 4096, 8, 4096, 4096, "uniform"),
    ("tokamax memory_bound", 8, 8, 4096, 4096, "uniform"),
    ("tokamax 8x7b", 8192, 8, 14336, 4096, "uniform"),
    ("tokamax 8x7b-zipf", 8192, 8, 14336, 4096, "zipf"),
    ("deepseek-ish g128", 65536, 128, 2048, 2048, "zipf"),
    ("deepseek-ish g256", 65536, 256, 1024, 1024, "zipf"),
    ("ARENA probe (ours)", 4096, 4, 4096, 14336, "uniform"),
    ("ARENA prod 32k-e8 (ours)", 32768, 8, 4096, 14336, "uniform"),
]


def run(label, m, g, k, n, dist):
    row = {"label": label, "m": m, "g": g, "k": k, "n": n, "dist": dist,
           "est_gb": round(est_gb(m, g, k, n), 2), "rows_per_group": m // g}
    if row["est_gb"] > HBM_BUDGET_GB:
        row["skipped"] = f"est {row['est_gb']} GB > {HBM_BUDGET_GB} GB"
        return row

    kl, kr, kg = jax.random.split(jax.random.PRNGKey(0), 3)
    lhs = jax.random.normal(kl, (m, k), F32).astype(jnp.bfloat16)
    rhs = jax.random.normal(kr, (g, k, n), F32).astype(jnp.bfloat16)
    gs = group_sizes(kg, g, m, dist)
    jax.block_until_ready((lhs, rhs, gs))
    flop = 2.0 * m * k * n

    try:
        s = timeit(lambda: jax.lax.ragged_dot(lhs, rhs, gs, preferred_element_type=F32))
        row["xla"] = {"median_s": s, "tflops": flop / s / 1e12}
    except Exception as e:  # noqa: BLE001
        row["xla"] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}

    from jax.experimental.pallas.ops.tpu.megablox import gmm

    row["megablox"] = {}
    for t in TILINGS:
        if t[0] > m or t[1] > k or t[2] > n:
            row["megablox"][str(t)] = {"skipped": "tile exceeds a dim"}
            continue
        try:
            s = timeit(lambda t=t: gmm(lhs, rhs, gs, tiling=t).astype(F32))
            row["megablox"][str(t)] = {"median_s": s, "tflops": flop / s / 1e12}
        except Exception as e:  # noqa: BLE001
            row["megablox"][str(t)] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}

    ok = {k2: v for k2, v in row["megablox"].items() if "median_s" in v}
    if ok and "median_s" in row.get("xla", {}):
        best_t = min(ok, key=lambda k2: ok[k2]["median_s"])
        row["best_tiling"] = best_t
        row["best_megablox_s"] = ok[best_t]["median_s"]
        row["best_over_xla"] = ok[best_t]["median_s"] / row["xla"]["median_s"]
        d = row["megablox"].get("(128, 128, 128)", {})
        if "median_s" in d:
            row["default_over_xla"] = d["median_s"] / row["xla"]["median_s"]
            row["tuning_speedup"] = d["median_s"] / ok[best_t]["median_s"]
    del lhs, rhs, gs
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {"device": [str(d) for d in jax.devices()], "rows": []}
    for spec in SHAPES:
        print(f"--- {spec[0]}", flush=True)
        try:
            r = run(*spec)
        except Exception as e:  # noqa: BLE001
            r = {"label": spec[0], "error": f"{type(e).__name__}: {str(e)[:200]}"}
        out["rows"].append(r)
        if "best_over_xla" in r:
            print(f"    xla={r['xla']['median_s'] * 1e3:8.2f} ms ({r['xla']['tflops']:6.1f} TF/s) | "
                  f"megablox best={r['best_megablox_s'] * 1e3:8.2f} ms @ {r['best_tiling']} "
                  f"({r['best_over_xla']:.2f}x xla) | default was {r.get('default_over_xla', float('nan')):.1f}x, "
                  f"tuning won {r.get('tuning_speedup', float('nan')):.1f}x", flush=True)
        else:
            print(f"    {str(r.get('skipped') or r.get('error'))[:150]}", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== SUMMARY: megablox/xla, <1.0 means the Pallas kernel WINS ===")
    print(f"{'shape':28s} {'g':>4s} {'xla ms':>9s} {'mblox ms':>9s} {'best tiling':>18s} {'ratio':>7s} {'dflt':>7s}")
    for r in out["rows"]:
        if "best_over_xla" in r:
            print(f"{r['label']:28s} {r['g']:4d} {r['xla']['median_s'] * 1e3:9.2f} "
                  f"{r['best_megablox_s'] * 1e3:9.2f} {r['best_tiling']:>18s} "
                  f"{r['best_over_xla']:6.2f}x {r.get('default_over_xla', 0):6.1f}x")
        else:
            print(f"{r['label']:28s} {'--':>4s} {str(r.get('skipped') or r.get('error'))[:60]}")


if __name__ == "__main__":
    main()
