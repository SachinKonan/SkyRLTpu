#!/usr/bin/env bash
# Base provisioning for a fresh (re)created TPU VM worker — replicates the
# jobman YAML's setup for environments without jobman: system packages, uv,
# gcsfuse mount of the shared bucket, and dirs. Run ON the TPU worker (any
# user with sudo). Idempotent.
set -euo pipefail

BUCKET="${BUCKET:-sk7524-tinker-tpu-us-east5}"
MOUNT="$HOME/gcs"

sudo apt-get update -y -qq
sudo apt-get install -y -qq git curl tmux python3 python3-venv python3-pip rsync

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if ! command -v gcsfuse >/dev/null 2>&1; then
  export GCSFUSE_REPO="gcsfuse-$(lsb_release -c -s)"
  echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get update -y -qq && sudo apt-get install -y -qq gcsfuse
fi

mkdir -p "$MOUNT"
if ! mountpoint -q "$MOUNT"; then
  gcsfuse --implicit-dirs --dir-mode 777 --file-mode 777 "$BUCKET" "$MOUNT"
fi
mkdir -p "$MOUNT"/{hf-cache,skyrl-logs,skyrl-checkpoints,skyrl-lora-models,skyrl-maxtext-ckpts,vllm-xla-cache,skyrl-runs} "$HOME/skyrl-logs"

echo "provisioned: $(hostname) uv=$(uv --version 2>/dev/null | head -1) gcsfuse-mounted=$(mountpoint -q "$MOUNT" && echo yes || echo no)"
