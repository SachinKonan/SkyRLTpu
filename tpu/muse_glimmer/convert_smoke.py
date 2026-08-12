#!/usr/bin/env python3
"""End-to-end smoke of the REAL HF -> orbax converter CLI for muse-glimmer.

`parity_maxtext_check.py` exercises the mapping/hook functions in process. This
script exercises the thing the tunix backend actually shells out to:

    python -m maxtext.checkpoint_conversion.to_maxtext \
        model_name=muse-glimmer-tiny hf_model_path=<dir> \
        base_output_directory=<out> scan_layers=True use_multimodal=false \
        skip_jax_distributed_system=True --lazy_load_tensors=True \
        checkpoint_storage_use_ocdbt=True checkpoint_storage_use_zarr3=True

(the exact argv from `skyrl/backends/tunix_backend.py::_ensure_maxtext_orbax_checkpoint`,
plus `hf_model_path` so nothing is downloaded), then loads the produced orbax
checkpoint back through `model_creation_utils.from_pretrained` and re-checks the
logits against the HF reference. CPU only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity_maxtext_check import errs, maxtext_config_cwd, run_model  # noqa: E402


def write_hf_dir(dump_dir: str, out_dir: str):
  """Materialise stage 1's tensors as a real HF checkpoint directory."""
  from safetensors.numpy import save_file

  os.makedirs(out_dir, exist_ok=True)
  weights = dict(np.load(os.path.join(dump_dir, "weights.npz")))
  meta = json.load(open(os.path.join(dump_dir, "meta.json")))

  # to_maxtext reads tensors by the names in the index; keep them verbatim.
  save_file({k: v.astype(np.float32) for k, v in weights.items()},
            os.path.join(out_dir, "model.safetensors"))

  text_cfg = meta["config"]
  cfg = {
      "architectures": ["MuseGlimmerForConditionalGeneration"],
      "model_type": "muse_glimmer",
      "dtype": "float32",
      "text_config": text_cfg,
  }
  with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2, default=str)
  return out_dir


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dump", required=True)
  ap.add_argument("--work", required=True)
  ap.add_argument("--model_name", default="muse-glimmer-tiny")
  args = ap.parse_args()

  hf_dir = write_hf_dir(args.dump, os.path.join(args.work, "hf"))
  out_dir = os.path.join(args.work, "orbax")
  os.makedirs(out_dir, exist_ok=True)
  print(f"[convert] HF dir  : {hf_dir}")
  print(f"[convert] orbax to: {out_dir}")

  cmd = [
      sys.executable,
      "-m",
      "maxtext.checkpoint_conversion.to_maxtext",
      f"model_name={args.model_name}",
      f"base_output_directory={out_dir}",
      "scan_layers=True",
      "use_multimodal=false",
      "skip_jax_distributed_system=True",
      "--lazy_load_tensors=True",
      # argparse flag, NOT a pyconfig key -- without the `--` pyconfig rejects it.
      "--hf_model_path",
      hf_dir,
      "checkpoint_storage_use_ocdbt=True",
      "checkpoint_storage_use_zarr3=True",
  ]
  env = os.environ.copy()
  env["JAX_PLATFORMS"] = "cpu"
  env["HF_HUB_OFFLINE"] = "1"
  with maxtext_config_cwd():
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
  tail = (r.stdout + r.stderr)[-4000:]
  print(tail)
  if r.returncode != 0:
    print(f"\n[convert] FAILED (exit {r.returncode})")
    sys.exit(1)

  items = os.path.join(out_dir, "0", "items")
  if not os.path.isdir(items):
    print(f"\n[convert] FAILED: {items} does not exist")
    sys.exit(1)
  print(f"\n[convert] orbax checkpoint at {items}")

  # ---- load it back and re-verify the logits --------------------------------
  import maxtext.configs.pyconfig as pyconfig
  from maxtext.utils import model_creation_utils

  ref = dict(np.load(os.path.join(args.dump, "ref.npz")))
  seq = 37
  overrides = dict(
      model_name=args.model_name,
      load_parameters_path=items,
      per_device_batch_size=1,
      max_target_length=seq,
      enable_checkpointing=True,
      pure_nnx=True,
      pure_nnx_decoder=True,
      dtype="float32",
      weight_dtype="float32",
      matmul_precision="highest",
      attention="dot_product",
      skip_jax_distributed_system=True,
      scan_layers=True,
  )
  argv = ["", "base.yml"] + [
      f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in overrides.items()
  ]
  with maxtext_config_cwd():
    cfg = pyconfig.initialize(argv)
    model = model_creation_utils.from_pretrained(cfg, mesh=None)
  if isinstance(model, tuple):
    model, _mesh = model

  from maxtext.common.common_types import MODEL_MODE_TRAIN

  ids = jnp.asarray(ref[f"ids_{seq}"], jnp.int32)
  pos = jnp.arange(seq, dtype=jnp.int32)[None, :]
  seg = jnp.ones((1, seq), jnp.int32)
  out = model(ids, pos, decoder_segment_ids=seg, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)
  logits = np.asarray(out[0] if isinstance(out, tuple) else out)
  a, r, rs = errs(logits, ref[f"logits_{seq}"])
  print(f"\n[convert] round-tripped orbax logits vs HF: max_abs={a:.4e} rel_to_scale={r:.4e} rel_sig={rs:.4e}")
  ok = a < 1e-4 and r < 1e-3
  print("[convert] " + ("PASS" if ok else "FAIL"))
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
