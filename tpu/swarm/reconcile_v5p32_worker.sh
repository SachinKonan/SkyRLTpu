#!/usr/bin/env bash
# Make a reused pool worker safe for THIS cell before bring-up.
#
# Pool workers outlive jobs, so a host that served muse in the last job may host
# a gemma cell now with 55 GB of muse HF weights and 40 GB of muse orbax still on
# its 97 GB boot disk. The gemma orbax restore (44 GB) then cannot fit and the
# cell sits in bring-up on "No space left on device" (jobs 200/212, 2026-09-05).
#
# Everything removed here is a cache re-derivable from GCS. Nothing under
# ~/skyrl-runs or the checkpoint root is touched: that is the run's durable
# state and has its own publish/prune path (cell_sync.sh, prune_local_checkpoints.sh).
#
# Runs on EVERY rank (SkyPilot runs the task on all four TPU VMs). Role by rank:
#   rank 0    trainer + client: keeps its own model's orbax checkpoint and its
#             own HF hub dir (tokenizer/config for the client; ensure_orbax_ckpt
#             drops the weights itself if the checkpoint would not fit).
#   rank 1-3  vLLM engines: keep their own model's HF hub dir; never use orbax.
#
#   CELL=g-meta SKYPILOT_NODE_RANK=1 reconcile_v5p32_worker.sh
#   RECONCILE_WIPE_ALL=1 CELL=x reconcile_v5p32_worker.sh   # manual: drop every model cache
set -uo pipefail
: "${CELL:?CELL selects the model (g-* gemma, m-* muse, else qwen)}"
rank="${SKYPILOT_NODE_RANK:-${JOBMAN_WORKER_ID:-0}}"
# RECONCILE_WIPE_ALL=1 keeps nothing model-specific (used for the one-time sweep
# of hosts left behind by pre-reconcile jobs); the next cell restores what it needs.
wipe_all="${RECONCILE_WIPE_ALL:-0}"

case "$CELL" in
  g-*) MT=gemma4-31b;       HF=models--google--gemma-4-31B-it ;;
  m-*) MT=muse-glimmer-30b; HF=models--meta-models--Muse-Glimmer-30B ;;
  *)   MT=qwen3.5-27b;      HF=models--Qwen--Qwen3.5-27B ;;
esac
MT="${TUNIX_MAXTEXT_MODEL_NAME:-$MT}"
HUB="${REMOTE_HF_HOME:-$HOME/.cache/huggingface}/hub"
ORBAX="${TUNIX_MAXTEXT_CKPT_CACHE:-$HOME/skyrl-maxtext-ckpts-local}"
BUNDLES="$HOME/.cache/tpuswarm/bundles"
current_bundle="$(readlink -f "${SKYRL_REPO_DIR:-/nonexistent}" 2>/dev/null || true)"

free_gb() { df -Pk "$HOME" | awk 'NR==2 {printf "%d", $4 / 1024 / 1024}'; }
before="$(free_gb)"

evict() {  # $1 = reason, $2 = path
  [ -e "$2" ] || return 0
  echo "reconcile[rank $rank]: $1 -- removing $2 ($(du -sh "$2" 2>/dev/null | cut -f1))"
  rm -rf -- "$2"
}

# HF snapshots of OTHER models (every rank).
for d in "$HUB"/models--*; do
  [ -d "$d" ] || continue
  if [ "$wipe_all" = 1 ]; then evict "HF snapshot (wipe-all)" "$d"
  elif [ "$(basename "$d")" != "$HF" ]; then evict "foreign HF snapshot" "$d"; fi
done

# The kept snapshot may still carry the blob/snapshot duplicate a GCS restore
# materializes (gemma: 11.9 GB twice). Collapse it to hardlinks.
if [ "$wipe_all" != 1 ] && [ -d "$HUB/$HF" ]; then
  bash "$(dirname "${BASH_SOURCE[0]}")/../dedupe_hf_snapshot.sh" "$HUB/$HF" 2>/dev/null | tail -1 | sed "s/^/reconcile[rank $rank]: /"
fi

# Orbax checkpoints: other models on the trainer host, every model on engines.
for d in "$ORBAX"/*/; do
  d="${d%/}"
  [ -d "$d" ] || continue
  if [ "$wipe_all" = 1 ]; then
    evict "orbax checkpoint (wipe-all)" "$d"
  elif [ "$rank" != "0" ]; then
    evict "orbax checkpoint on an engine host" "$d"
  elif [ "$(basename "$d")" != "$MT" ]; then
    evict "foreign orbax checkpoint" "$d"
  fi
done

# Compile caches are keyed by hash, not by model, so they cannot be pruned per
# entry: a host that served three models carries three models' compiles (8 GB,
# 233 entries seen on one engine). The current model's cache is restored from
# GCS at start (VLLM_XLA_CACHE_GCS / the trainer's jax-compile-cache bucket), so
# on a model switch the whole local dir goes. A marker records which model the
# local cache was last used for.
for cache in "${VLLM_XLA_CACHE_PATH:-$HOME/vllm-xla-cache-local}" "$HOME/jax_cache"; do
  [ -d "$cache" ] || continue
  marker="$cache/.tpuswarm-model"
  last="$(cat "$marker" 2>/dev/null || true)"
  if [ "$wipe_all" = 1 ] || [ "$last" != "$MT" ]; then
    evict "compile cache last used for ${last:-an unrecorded model}" "$cache"
  fi
done
if [ "$wipe_all" != 1 ]; then
  for cache in "${VLLM_XLA_CACHE_PATH:-$HOME/vllm-xla-cache-local}" "$HOME/jax_cache"; do
    mkdir -p "$cache" && printf '%s\n' "$MT" > "$cache/.tpuswarm-model"
  done
fi

# Uploaded LoRA adapters belong to a run's tinker model id; a new cell never
# reads another run's adapter (vllm_tpu_server keeps only the newest per run).
for d in "$HOME"/skyrl-local-loras/*/; do
  d="${d%/}"
  [ -d "$d" ] && evict "stale uploaded LoRA adapter" "$d"
done

# Superseded bundle generations (each carries a built discover venv, ~7 GB).
for d in "$BUNDLES"/*/; do
  d="${d%/}"
  [ -d "$d" ] || continue
  [ "$(readlink -f "$d")" = "$current_bundle" ] || evict "superseded bundle generation" "$d"
done

# Serving-benchmark XLA caches (tpu/swarm/bench writes one per model/setting,
# 5-6 GB each). A gemma engine host already sits near 90 GB of 97 with just its
# own model and runtime, so these were exactly what tipped node 7b to 0 bytes.
for d in "$HOME"/vllm-xla-cache-bench-*; do
  [ -d "$d" ] && evict "benchmark XLA cache" "$d"
done

# uv keeps downloaded wheel archives after the venvs are built (~1 GB reclaimable;
# the venvs hardlink into the cache, so this never breaks an installed venv).
[ -x "$HOME/.local/bin/uv" ] && "$HOME/.local/bin/uv" cache prune -q >/dev/null 2>&1 || true

# The TPU VM image ships docker with ~2 GB of images no container references
# (5.8 GB under /var/lib/docker on a 97 GB disk). Nothing here uses docker.
if command -v docker >/dev/null 2>&1; then
  sudo -n docker image prune -af >/dev/null 2>&1 && echo "reconcile[rank $rank]: pruned unreferenced docker images" || true
fi

echo "reconcile[rank $rank]: cell $CELL keeps $MT / $HF; free ${before} GB -> $(free_gb) GB"
