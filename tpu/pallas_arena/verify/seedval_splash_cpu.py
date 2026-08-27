"""CPU-interpret parity check for the splash seed (fwd + grads) against the
judge's fp32 reference, at MHA / GQA / asymmetric-d_v shapes.

    PALLAS_INTERPRET=1 JAX_PLATFORMS=cpu python3 seedval_splash_cpu.py
"""
import os
import sys

os.environ.setdefault("PALLAS_INTERPRET", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/pallas_arena")
sys.path.insert(0, "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/pallas_arena/probe")

import jax
import jax.numpy as jnp
import numpy as np

from judge.problems.splash_attention import causal_segment_attention
import seed_splash_flash as seed

SHAPES = [
    ("mha-h4-s2048-d128", 4, 4, 2048, 128, 128),
    ("gqa-8x2-s1024-d64", 8, 2, 1024, 64, 64),
    ("dv-asym-h2-s1024-d192-dv128", 2, 2, 1024, 192, 128),
]

fails = 0
for name, qh, kvh, s, d, dv in SHAPES:
    key = jax.random.PRNGKey(7)
    kq, kk, kv_, ks = jax.random.split(key, 4)
    q = (jax.random.normal(kq, (qh, s, d), jnp.float32) / np.sqrt(d)).astype(jnp.bfloat16)
    k = jax.random.normal(kk, (kvh, s, d), jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(kv_, (kvh, s, dv), jnp.float32).astype(jnp.bfloat16)
    seg = jnp.where(jnp.arange(s) < s - 64,
                    jnp.where(jnp.arange(s) < s // 2, 1, 2), 0).astype(jnp.int32)

    ref = causal_segment_attention(q, k, v, seg)
    out = seed.kernel(q, k, v, seg)
    err = float(jnp.max(jnp.abs(out - ref) / (jnp.abs(ref) + 1.0)))
    pad_ok = float(jnp.max(jnp.abs(out[:, seg == 0, :]))) == 0.0

    def scalar_seed(q, k, v):
        return jnp.sum(jnp.sin(seed.kernel(q, k, v, seg)))

    def scalar_ref(q, k, v):
        return jnp.sum(jnp.sin(causal_segment_attention(q, k, v, seg)))

    gs = jax.grad(scalar_seed, argnums=(0, 1, 2))(q, k, v)
    gr = jax.grad(scalar_ref, argnums=(0, 1, 2))(q, k, v)
    gerr = max(
        float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))
                      / (jnp.abs(b.astype(jnp.float32)) + 1.0)))
        for a, b in zip(gs, gr)
    )
    ok = err < 2e-2 and gerr < 2e-2 and pad_ok
    fails += 0 if ok else 1
    print(f"[seedval-splash] {name}: fwd_err={err:.2e} grad_err={gerr:.2e} "
          f"pad_zero={pad_ok} -> {'OK' if ok else 'FAIL'}", flush=True)

print(f"[seedval-splash] {'ALL OK' if fails == 0 else f'{fails} FAILURES'}", flush=True)
sys.exit(1 if fails else 0)
