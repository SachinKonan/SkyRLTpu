"""Small pure-JAX physics-aware autoencoder example; intentionally easy to replace.

No Flax/Optax needed. Uses automatic XLA backend selection (CPU/GPU/TPU).
This is a FUNCTIONAL STARTER, not a trained or competition-winning baseline.
Contract: fit(train, val, config, out_dir); load_model(out_dir, config);
          predict(model, maps, mask, config) -> (batch,) anomaly scores.
"""
from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np
import jax
import jax.numpy as jnp


def _features(maps, mask, cfg):
    # maps: B,H,W in physical convergence units, with zeros off-footprint.
    x = jnp.asarray(maps, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    b = x.shape[0]
    h, w = int(cfg["grid_height"]), int(cfg["grid_width"])
    # Mask-corrected resampling suppresses bias near survey footprint boundaries.
    pooled = jax.image.resize((x * mask)[..., None], (b, h, w, 1), method="linear")
    coverage = jax.image.resize(mask[None, ..., None], (1, h, w, 1), method="linear")
    grid = jnp.where(coverage > 0.02, pooled / jnp.maximum(coverage, 0.02), 0)
    return jnp.reshape(jnp.clip(grid / .02582, -15., 15.), (b, h * w))


def _init(rng, cfg):
    d = int(cfg["grid_height"]) * int(cfg["grid_width"])
    h = int(cfg["hidden_dim"])
    z = int(cfg["latent_dim"])
    dims = [(d, h, "e1"), (h, z, "e2"), (z, h, "d1"),
            (h, d, "d2"), (z, 2, "physics")]
    keys = jax.random.split(rng, len(dims))
    result = {}
    for (din, dout, name), key in zip(dims, keys):
        result[name + "_w"] = jax.random.normal(key, (din, dout)) * jnp.sqrt(2. / din)
        result[name + "_b"] = jnp.zeros((dout,), dtype=jnp.float32)
    return result


def _forward(params, f):
    h = jax.nn.gelu(f @ params["e1_w"] + params["e1_b"])
    z = h @ params["e2_w"] + params["e2_b"]
    h2 = jax.nn.gelu(z @ params["d1_w"] + params["d1_b"])
    recon = h2 @ params["d2_w"] + params["d2_b"]
    theta = z @ params["physics_w"] + params["physics_b"]
    return z, recon, theta


def _train_step(params, m, v, t, features, targets, cfg):
    @jax.value_and_grad
    def objective(p):
        _, reconstructed, predicted = _forward(p, features)
        rec = jnp.mean((features - reconstructed)**2)
        supervised = jnp.mean((predicted - targets)**2)
        regularization = sum(jnp.sum(x**2) for k, x in p.items() if k.endswith("_w"))
        return (rec + cfg["physics_weight"] * supervised +
                cfg["weight_decay"] * regularization)
    loss, g = objective(params)
    lr = float(cfg["learning_rate"])
    m = jax.tree.map(lambda a,b: .9*a + .1*b, m, g)
    v = jax.tree.map(lambda a,b: .999*a + .001*b*b, v, g)
    corrected_m = jax.tree.map(lambda x: x / (1-.9**t), m)
    corrected_v = jax.tree.map(lambda x: x / (1-.999**t), v)
    params = jax.tree.map(lambda p,a,b: p - lr*a/(jnp.sqrt(b)+1e-8),
                          params, corrected_m, corrected_v)
    return params, m, v, loss


def fit(train, val, config, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = _init(jax.random.PRNGKey(int(config["seed"])), config)
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    # Fixed compile shape for trial loops; final partial batches are dropped.
    # Python-level config is closed over, keeping JAX JIT inputs simple.
    @jax.jit
    def update(p, mm, vv, t, x, y):
        return _train_step(p, mm, vv, t, x, y, config)

    batch_size = int(config["batch_size"])
    steps = int(config["steps"])
    if len(train) < batch_size:
        batch_size = max(1, len(train))
    stream = train.iter_batches(batch_size=batch_size, epochs=max(2, steps + 1),
                                shuffle=True, drop_last=True,
                                seed=int(config["seed"]), add_noise=True)
    for t, batch in zip(range(1, steps + 1), stream):
        x = _features(batch["maps"], batch["mask"], config)
        # Only physically meaningful training labels; not used for evaluation scoring.
        y = (jnp.asarray(batch["labels"][:, :2]) - jnp.array([.3, .8])) / jnp.array([.15, .10])
        params, m, v, loss = update(params, m, v, t, x, y)
        if not bool(jnp.isfinite(loss)):
            raise FloatingPointError("Training produced NaN/Inf loss")
    arrays = {key: np.asarray(value) for key, value in params.items()}
    np.savez_compressed(out / "model.npz", **arrays)
    (out / "train_info.txt").write_text(f"steps={steps} loss={float(loss):.6g}\n")
    # InD-only validation for monitoring. Do not use private proxy reward examples.
    return {"loss": float(loss), "steps": steps}


def load_model(out_dir, config):
    with np.load(Path(out_dir) / "model.npz") as z:
        return {key: jnp.asarray(z[key]) for key in z.files}


def predict(model, maps, mask, config):
    features = _features(maps, mask, config)
    _, reconstructed, _ = _forward(model, features)
    # Reconstruction-based outlier score; higher => more anomalous.
    return np.asarray(jnp.mean((features - reconstructed) ** 2, axis=1))
