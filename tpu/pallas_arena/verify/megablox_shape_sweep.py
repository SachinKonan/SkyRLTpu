"""Where does the megablox denominator stop being pathological?

Measured on v6e-1 (the grading hardware) at the probe shapes the arena actually
grades on, our bound `pallas-megablox-gmm` baseline runs 49.49 ms against
tokamax's XLA `ragged_dot` at 1.297 ms -- 38x slower -- and tokamax's OWN mosaic
GMM lands at 49.63 ms, i.e. two independently tuned production Pallas
grouped-matmul kernels collapse identically. A reward >1.0 against a denominator
running at ~4% of peak does not mean the candidate beat a competitive kernel.

The obvious hypothesis -- "too few rows per group" -- does not survive contact
with the shape table: probe is m=4096/g=4 = 1024 rows/group, while production
`32k-e64` is 32768/64 = 512, i.e. FEWER. So this sweeps the (m, g) plane and
lets the measurement pick the variable.

The reason this can cover PRODUCTION shapes on a 32 GB judge at all: timing the
baseline needs only bf16 operands and the f32 output. It is the fp32 REFERENCE
(rhs cast to f32, 15 GB at g=64) that forced the one-chip probe set into
existence -- and correctness is not what is in question here.

Emits, per shape: rows/group, both implementations' median wall time, achieved
TFLOP/s, and the megablox/XLA ratio. A shape where that ratio is near 1 is a
defensible place to grade; a shape where it is 38 is not.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

F32 = jnp.float32
HBM_BUDGET_GB = 26.0  # of a 32 GB v6e-1, leaving headroom for XLA scratch


def sample_group_sizes(key, g, m, dist):
    if dist == "uniform":
        w = jnp.ones((g,))
    else:
        w = 1.0 / jnp.arange(1, g + 1, dtype=F32)
        w = jax.random.permutation(key, w)
    probs = w / jnp.sum(w)
    counts = jnp.floor(probs * m).astype(jnp.int32)
    counts = counts.at[jnp.argmax(probs)].add(m - jnp.sum(counts))
    return counts.astype(jnp.int32)


def est_gb(m, g, k, n):
    """bf16 lhs + bf16 rhs + f32 out, x2 for transient copies."""
    return 2 * (m * k * 2 + g * k * n * 2 + m * n * 4) / 1e9


def timeit(fn, pairs=7, warmup=2):
    f = jax.jit(fn)
    for _ in range(warmup):
        jax.block_until_ready(f())
    ts = []
    for _ in range(pairs):
        t0 = time.perf_counter()
        jax.block_until_ready(f())
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def run_shape(m, g, k, n, dist, label):
    row = {"label": label, "m": m, "g": g, "k": k, "n": n, "dist": dist,
           "rows_per_group": m // g, "est_gb": round(est_gb(m, g, k, n), 2)}
    if row["est_gb"] > HBM_BUDGET_GB:
        row["skipped"] = f"est {row['est_gb']} GB > {HBM_BUDGET_GB} GB budget"
        return row

    key = jax.random.PRNGKey(0)
    kl, kr, kg = jax.random.split(key, 3)
    lhs = jax.random.normal(kl, (m, k), F32).astype(jnp.bfloat16)
    rhs = jax.random.normal(kr, (g, k, n), F32).astype(jnp.bfloat16)
    gs = sample_group_sizes(kg, g, m, dist)
    jax.block_until_ready((lhs, rhs, gs))
    flop = 2.0 * m * k * n

    def xla():
        return jax.lax.ragged_dot(lhs, rhs, gs, preferred_element_type=F32)

    def pallas():
        from jax.experimental.pallas.ops.tpu.megablox import gmm

        return gmm(lhs, rhs, gs).astype(F32)

    for tag, fn in (("xla_ragged_dot", xla), ("pallas_megablox", pallas)):
        try:
            s = timeit(fn)
            row[tag] = {"median_s": s, "tflops": flop / s / 1e12}
        except Exception as e:  # noqa: BLE001
            row[tag] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    p, x = row.get("pallas_megablox", {}), row.get("xla_ragged_dot", {})
    if "median_s" in p and "median_s" in x:
        row["megablox_over_xla"] = p["median_s"] / x["median_s"]
    del lhs, rhs, gs
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    K, N = 4096, 14336
    shapes = []
    # what the arena grades today
    for m, dist in ((4096, "uniform"), (2048, "zipf"), (3000, "zipf")):
        shapes.append((m, 4, K, N, dist, f"PROBE m{m}-g4-{dist}"))
    # the declared production set (timing needs no fp32 reference)
    for g in (8, 64):
        for dist in ("uniform", "zipf"):
            shapes.append((32768, g, K, N, dist, f"PROD 32k-e{g}-{dist}"))
    shapes.append((16384, 32, 2048, 7168, "zipf", "PROD holdout-16k-e32-zipf"))
    # the (m, g) plane: does raising tokens at fixed g rescue it?
    for g in (4, 8):
        for m in (8192, 16384, 32768):
            shapes.append((m, g, K, N, "uniform", f"SWEEP m{m}-g{g}-uniform"))

    out = {"device": [str(d) for d in jax.devices()], "backend": jax.default_backend(), "rows": []}
    for spec in shapes:
        print(f"--- {spec[-1]}", flush=True)
        try:
            r = run_shape(*spec)
        except Exception as e:  # noqa: BLE001
            r = {"label": spec[-1], "error": f"{type(e).__name__}: {str(e)[:200]}"}
        out["rows"].append(r)
        if "skipped" in r:
            print(f"    skipped: {r['skipped']}", flush=True)
        elif "megablox_over_xla" in r:
            print(f"    rows/grp={r['rows_per_group']:6d}  megablox={r['pallas_megablox']['median_s'] * 1e3:8.2f} ms "
                  f"({r['pallas_megablox']['tflops']:6.1f} TF/s)  xla={r['xla_ragged_dot']['median_s'] * 1e3:8.2f} ms "
                  f"({r['xla_ragged_dot']['tflops']:6.1f} TF/s)  ratio={r['megablox_over_xla']:.1f}x", flush=True)
        else:
            print(f"    {json.dumps({k: v for k, v in r.items() if k not in ('label',)})[:220]}", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== SUMMARY (ratio = megablox / xla; near 1.0 is a defensible denominator) ===")
    print(f"{'shape':34s} {'rows/grp':>9s} {'megablox ms':>12s} {'xla ms':>9s} {'ratio':>7s}")
    for r in out["rows"]:
        if "megablox_over_xla" in r:
            print(f"{r['label']:34s} {r['rows_per_group']:9d} "
                  f"{r['pallas_megablox']['median_s'] * 1e3:12.2f} {r['xla_ragged_dot']['median_s'] * 1e3:9.2f} "
                  f"{r['megablox_over_xla']:6.1f}x")
        else:
            print(f"{r['label']:34s} {'--':>9s} {str(r.get('skipped') or r.get('error'))[:52]}")


if __name__ == "__main__":
    main()
