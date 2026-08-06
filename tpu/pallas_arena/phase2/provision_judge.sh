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
  "jax[tpu]==0.10.2" numpy fastapi uvicorn pydantic flatbuffers google-cloud-storage

# TPU sanity: the chip must enumerate
JAX_PLATFORMS=tpu "$HOME/arena-venv/bin/python" - <<'EOF'
import jax
devs = jax.devices()
print("tpu-sanity:", devs)
assert devs and devs[0].platform == "tpu", devs
EOF

# Shared reward-cache reachability. A silently-unreachable cache degrades to
# a permanent miss, which looks exactly like "everything works" until the
# byte-identical-repeat-reward property is quietly gone -- so prove RW here,
# at provision time, where the failure is attributable.
if [ -n "${ARENA_CACHE_PREFIX:-}" ]; then
  ARENA_CACHE_PREFIX="$ARENA_CACHE_PREFIX" "$HOME/arena-venv/bin/python" - <<'EOF' || echo "arena-cache: UNAVAILABLE"
import os, socket, sys
sys.path.insert(0, os.path.expanduser("~/arena"))
from pallas_arena.judge.cache import RewardCache
c = RewardCache(os.environ["ARENA_CACHE_PREFIX"])
key = ("ab" * 32)[:64]
probe = {"probe": socket.gethostname()}
c.put(key + "-probe", probe)
got = c.get(key + "-probe")
print("arena-cache:", "RW OK" if got else "UNAVAILABLE", got)
assert got, "reward cache round-trip failed"
EOF
fi

echo "provision_judge: OK on $(hostname)"
