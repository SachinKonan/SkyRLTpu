#!/usr/bin/env python3
"""Convert the REAL Muse-Glimmer-30B HF checkpoint to a MaxText orbax one.

Same argv the tunix backend shells out to
(`skyrl/backends/tunix_backend.py::_ensure_maxtext_orbax_checkpoint`), plus
`--hf_model_path` so nothing is downloaded, exactly as
`tpu/muse_glimmer/convert_smoke.py` does on tiny weights.

Run on a CPU node, not on the TPU host: the v5p-8 boot disk is 97 GB and the
55.5 GiB HF checkpoint already lives there, so a second ~52 GiB artefact does
not fit.  Converting here and shipping the orbax directory through GCS costs
the slice ~90 s of download instead of a conversion it has no room for.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity_maxtext_check import maxtext_config_cwd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="muse-glimmer-30b")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "maxtext.checkpoint_conversion.to_maxtext",
        f"model_name={args.model_name}",
        f"base_output_directory={args.out}",
        "scan_layers=True",
        "use_multimodal=false",
        "skip_jax_distributed_system=True",
        "--lazy_load_tensors=True",
        "--hf_model_path",
        args.hf_dir,
        "checkpoint_storage_use_ocdbt=True",
        "checkpoint_storage_use_zarr3=True",
    ]
    print("[convert] " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["HF_HUB_OFFLINE"] = "1"
    t0 = time.time()
    with maxtext_config_cwd():
        r = subprocess.run(cmd, env=env)
    print(f"[convert] exit {r.returncode} in {time.time() - t0:.0f}s",
          flush=True)
    if r.returncode != 0:
        return 1
    items = os.path.join(args.out, "0", "items")
    if not os.path.isdir(items):
        print(f"[convert] FAILED: {items} missing")
        return 1
    n = sum(len(f) for _, _, f in os.walk(items))
    sz = sum(
        os.path.getsize(os.path.join(d, f))
        for d, _, fs in os.walk(items) for f in fs)
    print(f"[convert] OK: {items}  {n} files  {sz / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
