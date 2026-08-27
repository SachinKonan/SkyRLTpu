"""CPU check of the grouped _honest_online_softmax: correctness vs the fp32
reference and the calibrated band it produces at GQA / d_v!=d shapes."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu")

import jax
import jax.numpy as jnp
import numpy as np

from pallas_arena.judge.problems.splash_attention import (
    PROBLEM,
    _honest_online_softmax,
    causal_segment_attention,
)

SHAPES = [
    ("mha-h4-s2048-d128", 4, 4, 2048, 128, 128),
    ("mixtral-like-gqa8x2-s1024-d64", 8, 2, 1024, 64, 64),
    ("deepseek-like-h2-s1024-d192-dv128", 2, 2, 1024, 192, 128),
]

bad = 0
for name, qh, kvh, s, d, dv in SHAPES:
    kq, kk, kv_, _ = jax.random.split(jax.random.PRNGKey(3), 4)
    q = (jax.random.normal(kq, (qh, s, d)) / np.sqrt(d)).astype(jnp.bfloat16)
    k = jax.random.normal(kk, (kvh, s, d)).astype(jnp.bfloat16)
    v = jax.random.normal(kv_, (kvh, s, dv)).astype(jnp.bfloat16)
    seg = jnp.where(jnp.arange(s) < s - 64,
                    jnp.where(jnp.arange(s) < s // 2, 1, 2), 0).astype(jnp.int32)
    ref = causal_segment_attention(q, k, v, seg)
    var = _honest_online_softmax(q, k, v, seg)
    err = float(jnp.max(jnp.abs(var - ref) / (jnp.abs(ref) + 1.0)))
    pad_zero = float(jnp.max(jnp.abs(var[:, seg == 0, :]))) == 0.0
    ok = err < 5e-2 and pad_zero and bool(jnp.isfinite(var).all())
    bad += 0 if ok else 1
    print(f"[varcheck] {name}: var_vs_ref_max_err={err:.3e} pad_zero={pad_zero}"
          f" -> {'OK' if ok else 'FAIL'}", flush=True)
    tol = PROBLEM.calibrated_tolerance((q, k, v, seg), ref)
    print(f"[varcheck] {name}: calibrated band now max={tol['max']:.3e} "
          f"q99={tol['q99']:.3e}", flush=True)

print(f"[varcheck] {'ALL OK' if bad == 0 else f'{bad} FAILURES'}", flush=True)
sys.exit(1 if bad else 0)
