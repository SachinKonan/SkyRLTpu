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

# Real GENERAL-mode denominators on top of the pin, never instead of it:
#   recurrentgemma+flax -- DeepMind's Pallas LRU scan (rg_lru baseline)
#   tokamax             -- linear_softmax_cross_entropy_loss mosaic_tpu (lsce)
# Both declare jax floors that 0.10.2 satisfies; the assert below makes a
# silent jax upgrade a hard provision failure, because it would invalidate
# every timing this arena has recorded.
"$HOME/arena-venv/bin/pip" install --quiet recurrentgemma flax tokamax
"$HOME/arena-venv/bin/python" - <<'EOF'
import jax
assert jax.__version__ == "0.10.2", f"ARENA JAX PIN MOVED: {jax.__version__}"
from recurrentgemma.jax.pallas import lru_pallas_scan  # noqa: F401
import tokamax  # noqa: F401
print("denominator deps: OK (jax pin intact)")
EOF

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
