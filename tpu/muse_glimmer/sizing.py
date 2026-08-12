#!/usr/bin/env python3
"""Memory sizing for muse-glimmer-30b at our real training shape.

Builds the ABSTRACT MaxText parameter tree (no weights materialised, so this runs
on a CPU login-class node) for the production config and reports:
  * exact parameter count and bf16 footprint, total and per chip under the mesh;
  * the per-parameter logical sharding annotations, so an unsharded parameter is
    impossible to miss;
  * an activation estimate at the sequence length the sweep actually trains at.

Run inside the MaxText venv. No TPU is touched.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Fake CHIPS CPU devices so the real 4-way FSDP mesh can actually be built and the
# reported sharding is the one training would use, not a 1-device degenerate mesh.
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"

import contextlib
import tempfile
from pathlib import Path

import jax
import numpy as np
from flax import linen as nn


@contextlib.contextmanager
def maxtext_config_cwd():
  import maxtext

  configs_dir = Path(maxtext.__file__).parent / "configs"
  prev = os.getcwd()
  with tempfile.TemporaryDirectory() as td:
    # pyconfig resolves a bare `base.yml` against <cwd>/src/, and derives the
    # models/ directory from the *parent* of the base config's directory. Lay the
    # temp dir out so both land inside the installed package's configs/.
    src = Path(td) / "src"
    src.mkdir()
    (src / "base.yml").symlink_to(configs_dir / "base.yml")
    (src / "models").symlink_to(configs_dir / "models")
    (src / "maxtext").symlink_to(configs_dir.parent)
    os.chdir(td)
    try:
      yield
    finally:
      os.chdir(prev)


# The sweep's shape: one v5p host, 4 chips, pure FSDP, 24576-token rows.
CHIPS = 4
SEQ = 24576


def main():
  import maxtext.configs.pyconfig as pyconfig
  from maxtext.layers import quantizations
  from maxtext.models import models
  from maxtext.utils import maxtext_utils

  overrides = dict(
      model_name="muse-glimmer-30b",
      per_device_batch_size=1,
      max_target_length=SEQ,
      scan_layers=True,
      pure_nnx=True,
      pure_nnx_decoder=True,
      dtype="bfloat16",
      weight_dtype="bfloat16",
      ici_fsdp_parallelism=CHIPS,
      skip_jax_distributed_system=True,
      enable_checkpointing=False,
  )
  argv = ["", "base.yml"] + [
      f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in overrides.items()
  ]
  with maxtext_config_cwd():
    cfg = pyconfig.initialize(argv)

  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = jax.sharding.Mesh(devices_array, cfg.mesh_axes)
  quant = quantizations.configure_quantization(cfg)
  model = models.transformer_as_linen(cfg, mesh, quant=quant)
  abstract = maxtext_utils.get_abstract_param(model, cfg)["params"]

  rows = []
  total = 0
  unannotated = []
  for path, leaf in jax.tree_util.tree_flatten_with_path(
      abstract, is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned)
  )[0]:
    key = "/".join(k.key for k in path if hasattr(k, "key"))
    shape = tuple(getattr(leaf, "value", leaf).shape)
    names = getattr(leaf, "names", None)
    n = int(np.prod(shape))
    total += n
    rows.append((key, shape, names, n))
    if not names or all(x is None for x in names):
      unannotated.append(key)

  print(f"model_name       : {cfg.model_name}")
  print(f"decoder_block    : {cfg.decoder_block}")
  print(f"layers           : {cfg.num_decoder_layers} "
        f"(scan_layers={cfg.scan_layers}, cycle={cfg.inhomogeneous_layer_cycle_interval})")
  print(f"emb/mlp/heads/kv : {cfg.emb_dim}/{cfg.mlp_dim}/{cfg.num_query_heads}/{cfg.num_kv_heads}"
        f" head_dim={cfg.head_dim}")
  print(f"mesh axes        : {cfg.mesh_axes}")
  print(f"logical rules hit: fsdp={CHIPS}")
  print()
  print(f"PARAMETERS       : {total:,}  ({total / 1e9:.3f} B)")
  print(f"bf16 total       : {total * 2 / 2**30:.2f} GiB")
  print(f"bf16 per chip /{CHIPS}  : {total * 2 / 2**30 / CHIPS:.2f} GiB   (pure FSDP, all axes shardable)")
  print()

  print("largest parameters:")
  for key, shape, names, n in sorted(rows, key=lambda r: -r[3])[:12]:
    print(f"  {n:>14,}  {str(shape):>28}  {names}  {key}")
  print()

  # The attention output gate is an extra full-size projection this architecture
  # has and most do not -- call it out, it is 5% of the model.
  gate = sum(n for key, _, _, n in rows if "attn_gate" in key)
  print(f"attention output gate (self_attn.gate_proj): {gate:,} params "
        f"= {100 * gate / total:.1f}% of the model, {gate * 2 / 2**30:.2f} GiB bf16")
  print()

  if unannotated:
    print(f"!! {len(unannotated)} PARAMETERS WITH NO LOGICAL SHARDING ANNOTATION:")
    for k in unannotated:
      print("   " + k)
  else:
    print("all parameters carry a logical sharding annotation")

  # Activation estimate under remat: with `remat_policy=full` only the per-block
  # input is kept, so the saved activations are (num scanned blocks) x [B, S, E].
  blocks = cfg.num_decoder_layers // cfg.inhomogeneous_layer_cycle_interval
  act = blocks * SEQ * cfg.emb_dim * 2
  print()
  print(f"activation floor at S={SEQ} (remat_policy=full, per-block input only):")
  print(f"  {blocks} blocks x {SEQ} x {cfg.emb_dim} x 2B = {act / 2**30:.2f} GiB "
        f"({act / 2**30 / CHIPS:.2f} GiB/chip when the batch axis is sharded)")
  print("  NB: this is a floor. The peak also holds one block's recomputed"
        " activations plus the [B,S,V] logits unless FLCE tiling is on;")
  print(f"  [B,S,V] in f32 at S={SEQ}, V={cfg.vocab_size} = "
        f"{SEQ * cfg.vocab_size * 4 / 2**30:.2f} GiB -- use TUNIX_FLCE_TILE_SIZE.")


if __name__ == "__main__":
  main()
