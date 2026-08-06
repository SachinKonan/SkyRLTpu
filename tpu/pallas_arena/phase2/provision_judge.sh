#!/usr/bin/env bash
# Provision a fresh v6e-1 judge host for the phase-2 shakedown. Runs ON the
# TPU VM (idempotent). Mirrors tpu/provision_tpu_worker.sh conventions.
# Pin: jax[tpu]==0.10.2 (the arena-wide pin from PHASE0-REPORT.md).
set -euo pipefail

sudo apt-get update -y -qq
sudo apt-get install -y -qq git curl tmux python3 python3-venv rsync

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -x "$HOME/arena-venv/bin/python" ]; then
  uv venv "$HOME/arena-venv" --python 3.12 --seed
fi
"$HOME/arena-venv/bin/pip" install --quiet \
  "jax[tpu]==0.10.2" numpy fastapi uvicorn pydantic

# TPU sanity: the chip must enumerate
JAX_PLATFORMS=tpu "$HOME/arena-venv/bin/python" - <<'EOF'
import jax
devs = jax.devices()
print("tpu-sanity:", devs)
assert devs and devs[0].platform == "tpu", devs
EOF

echo "provision_judge: OK on $(hostname)"
