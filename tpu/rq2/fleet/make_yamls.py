"""Emit jobman YAMLs for the llama-farm slices.

Clones the proven v5p-64 config (tpu/jobman_configs/ttd_erdos_v5p64_split_spot_us_east5a.yaml,
whose bring-up command is worker-agnostic and scales as-is) and changes only the names. The
farm is a SERVING fleet -- no trainer, no SkyRL venv build -- so the base command is trimmed to
the packages serve_vllm.sh needs, and the reservation is held open with `sleep infinity` exactly
as the template does.

  python3 make_yamls.py --out tpu/rq2/fleet/yamls --slices a b c
"""
from __future__ import annotations

import argparse
from pathlib import Path

TEMPLATE = """job:
  name: {name}
  env_type: null
  loop: true
  remote_user: sk7524_princeton_edu
  worker_num: spot

tpu:
  allocation_mode: "queued-resources"
  accelerator: v5p-64
  name: {name}
  zone: {zone}
  version: v2-alpha-tpuv5
  pricing: spot
  flags:
    - "--project=vision-mix"

gcsfuse:
  bucket_name: sk7524-tinker-tpu-us-east5
  mount_path: /home/sk7524_princeton_edu/gcs

ssh:
  private_key: ~/.ssh/jobman_tpu_ed25519
  identities:
    - private_key: ~/.ssh/jobman_tpu_ed25519
      public_key: ~/.ssh/jobman_tpu_ed25519.pub
      config_entry: |
        Host 10.* 34.* 35.* 104.* 136.* 8.*
          IdentityFile ~/.ssh/jobman_tpu_ed25519
          IdentitiesOnly yes
          StrictHostKeyChecking no
          UserKnownHostsFile /dev/null

command:
  cmd: |
    set -euxo pipefail
    # Serving fleet only: no trainer, no SkyRL venv. serve_vllm.sh builds its own
    # ~/.venvs/vllm-tpu on first use and is idempotent thereafter.
    sudo apt-get update -y
    sudo apt-get install -y git curl tmux python3 python3-venv python3-pip
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
    mkdir -p /home/sk7524_princeton_edu/gcs/hf-cache
    echo "llama-farm host ready on $(hostname)."
    sleep infinity
  workers: "all"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "yamls"))
    ap.add_argument("--slices", nargs="+", default=["a", "b", "c"])
    ap.add_argument("--zone", default="us-east5-a")
    ap.add_argument("--prefix", default="sk7524-llamafarm")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in args.slices:
        name = f"{args.prefix}-{s}-v5p64-east5a"
        p = out / f"{name}.yaml"
        p.write_text(TEMPLATE.format(name=name, zone=args.zone))
        print(f"[yaml] {name} -> {p}")
    print(f"\nQR names will be '<name>_spot' (jobman appends the pricing suffix).")


if __name__ == "__main__":
    main()
