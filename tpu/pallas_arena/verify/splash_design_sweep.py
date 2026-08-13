"""Is the splash denominator sound when the kernel is used AS DESIGNED?

Same audit that overturned the megablox conclusion (job 3691513: tuned megablox
BEATS XLA 4.1x at Mixtral 8x7b; the library default cost 13-55x). splash has the
identical setup, and JAX says so in its own source:

    @classmethod
    def get_default(cls):
      # TODO(apaszke,sharadmv): Select better parameters based on a heuristic.
      return BlockSizes(block_q=128, block_kv=128, ...)

`make_splash_mha(..., block_sizes=None)` falls back to that all-128 placeholder,
and the arena's `baseline()` passes no `block_sizes` at all. At our probe
seq=4096 that is a 32x32 block grid -- the same "thousands of tiny tiles"
pathology class that cost megablox 38x. So the one denominator still described
as competitive has never been measured in a tuned configuration.

Two axes, mirroring the GMM sweep:
  * SHAPES  -- tokamax's canonical attention sweep (arg_specs.py): seq_len over
    (512..16384) x head_dim over (64,128,256) at a FIXED 16384-token budget, so
    total work is comparable while every blocking decision changes. Plus the
    arena's own probe shapes for direct comparison.
  * BLOCKS  -- a real (block_q, block_kv) grid, reporting best-configured
    alongside the default so the correction is visible in one table.

CAUSAL ONLY. The arena additionally imposes segment-id masking, which splash
supports but which is our requirement, not the regime it is tuned against;
measuring design intent means measuring the kernel's own mode first.

Baseline to beat: a query-blocked fp32 XLA attention (never materializes more
than [heads, block_q, seq], so it runs at seq=16384 where the closed form would
need 8.6 GB).
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

F32 = jnp.float32
NEG = -0.7 * float(np.finfo(np.float32).max)
TOKENS = 16384  # tokamax's fixed token budget: batch = TOKENS // seq


def xla_causal_attention(q, k, v, block_q: int = 512):
    """Query-blocked fp32 causal attention -- the honest floor."""
    h, s, d = q.shape
    pad = (-s) % block_q
    qp = jnp.pad(q, ((0, 0), (0, pad), (0, 0))) if pad else q
    idx_k = jnp.arange(s)
    k32, v32 = k.astype(F32), v.astype(F32)

    def blk(_c, i):
        start = i * block_q
        qb = jax.lax.dynamic_slice(qp, (0, start, 0), (h, block_q, d)).astype(F32)
        pos_q = start + jnp.arange(block_q)
        logits = jnp.einsum("hqd,hkd->hqk", qb, k32)
        m = (pos_q[:, None] >= idx_k[None, :]) & (pos_q[:, None] < s)
        logits = jnp.where(m[None], logits, NEG)
        row_live = m.any(-1)
        p = jnp.where(m[None], jnp.exp(logits - logits.max(-1, keepdims=True)), 0.0)
        p = jnp.where(row_live[None, :, None], p / jnp.maximum(p.sum(-1, keepdims=True), 1e-30), 0.0)
        return None, jnp.einsum("hqk,hkd->hqd", p, v32)

    _, blocks = jax.lax.scan(blk, None, jnp.arange((s + pad) // block_q))
    out = jnp.transpose(blocks, (1, 0, 2, 3)).reshape(h, s + pad, d)
    return out[:, :s, :]


BLOCKS = [(128, 128), (256, 256), (512, 512), (512, 1024), (1024, 512), (1024, 1024), (2048, 512)]


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


def run(label, heads, seq, d):
    row = {"label": label, "heads": heads, "seq": seq, "d": d}
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)
    scale = 1.0 / np.sqrt(d)
    q = (jax.random.normal(kq, (heads, seq, d), F32) * scale).astype(jnp.bfloat16)
    k = jax.random.normal(kk, (heads, seq, d), F32).astype(jnp.bfloat16)
    v = jax.random.normal(kv, (heads, seq, d), F32).astype(jnp.bfloat16)
    jax.block_until_ready((q, k, v))
    # causal attention flops: ~ heads * seq^2 * d (halved by causality) * 2 matmuls * 2
    flop = 2.0 * 2.0 * heads * (seq * seq / 2) * d

    try:
        s = timeit(lambda: xla_causal_attention(q, k, v))
        row["xla"] = {"median_s": s, "tflops": flop / s / 1e12}
    except Exception as e:  # noqa: BLE001
        row["xla"] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}

    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sak
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sam

    row["splash"] = {}
    for bq, bkv in BLOCKS:
        if bq > seq or bkv > seq:
            row["splash"][f"{bq}x{bkv}"] = {"skipped": "block exceeds seq"}
            continue
        try:
            mask = sam.MultiHeadMask([sam.CausalMask(shape=(seq, seq)) for _ in range(heads)])
            bs = sak.BlockSizes(block_q=bq, block_kv=bkv, block_kv_compute=bkv,
                                block_q_dkv=bq, block_kv_dkv=bkv, block_kv_dkv_compute=bkv,
                                block_q_dq=bq, block_kv_dq=bkv)
            kern = sak.make_splash_mha(mask=mask, block_sizes=bs, head_shards=1, q_seq_shards=1)
            s = timeit(lambda kern=kern: kern(q, k, v).astype(F32))
            row["splash"][f"{bq}x{bkv}"] = {"median_s": s, "tflops": flop / s / 1e12}
        except Exception as e:  # noqa: BLE001
            row["splash"][f"{bq}x{bkv}"] = {"error": f"{type(e).__name__}: {str(e)[:110]}"}

    ok = {kk2: vv for kk2, vv in row["splash"].items() if "median_s" in vv}
    if ok and "median_s" in row.get("xla", {}):
        best = min(ok, key=lambda z: ok[z]["median_s"])
        row["best_blocks"] = best
        row["best_splash_s"] = ok[best]["median_s"]
        row["best_over_xla"] = ok[best]["median_s"] / row["xla"]["median_s"]
        dflt = row["splash"].get("128x128", {})
        if "median_s" in dflt:
            row["default_over_xla"] = dflt["median_s"] / row["xla"]["median_s"]
            row["tuning_speedup"] = dflt["median_s"] / ok[best]["median_s"]
    del q, k, v
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    shapes = []
    # tokamax's canonical sweep at a fixed token budget (heads folded from batch)
    for seq in (1024, 2048, 4096, 8192, 16384):
        for d in (64, 128):
            heads = max(1, min(32, TOKENS // seq * 2))
            shapes.append((f"tokamax s{seq}-d{d}", heads, seq, d))
    # the arena's own declared probe shapes
    shapes += [("ARENA probe-h8-s4096", 8, 4096, 128), ("ARENA probe-h4-s2048", 4, 2048, 128)]

    out = {"device": [str(x) for x in jax.devices()], "rows": []}
    for spec in shapes:
        print(f"--- {spec[0]}", flush=True)
        try:
            r = run(*spec)
        except Exception as e:  # noqa: BLE001
            r = {"label": spec[0], "error": f"{type(e).__name__}: {str(e)[:200]}"}
        out["rows"].append(r)
        if "best_over_xla" in r:
            print(f"    xla={r['xla']['median_s'] * 1e3:8.2f} ms | splash best={r['best_splash_s'] * 1e3:8.2f} ms "
                  f"@ {r['best_blocks']} ({r['best_over_xla']:.2f}x xla) | default {r.get('default_over_xla', float('nan')):.2f}x, "
                  f"tuning won {r.get('tuning_speedup', float('nan')):.1f}x", flush=True)
        else:
            print(f"    {str(r.get('error'))[:160]}", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== SUMMARY: splash/xla, <1.0 means the Pallas kernel WINS ===")
    print(f"{'shape':26s} {'xla ms':>9s} {'splash ms':>10s} {'best blk':>10s} {'ratio':>7s} {'dflt':>7s} {'tune':>6s}")
    for r in out["rows"]:
        if "best_over_xla" in r:
            print(f"{r['label']:26s} {r['xla']['median_s'] * 1e3:9.2f} {r['best_splash_s'] * 1e3:10.2f} "
                  f"{r['best_blocks']:>10s} {r['best_over_xla']:6.2f}x {r.get('default_over_xla', 0):6.2f}x "
                  f"{r.get('tuning_speedup', 0):5.1f}x")
        else:
            print(f"{r['label']:26s} {str(r.get('error'))[:60]}")


if __name__ == "__main__":
    main()
