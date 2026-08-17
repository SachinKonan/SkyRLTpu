#!/usr/bin/env bash
# =============================================================================
# Muse-Glimmer-30B RL launch path: self-hosted SkyRL Tinker server on ONE
# existing v5p-32 spot slice, mirroring the qwen35 RL-server pattern
# (tpu/start_ttd_split_qwen35_9b.sh worker split + tpu/runs/qwen35-27b.env
# tunix recipe), launched through tpu/start_colocated_vllm_tinker.sh.
#
# This script does NOT create TPU resources. It assumes TPU_NAME already
# exists (jobman / queued-resource flow, as for qwen35) and only provisions
# software on it. The ttt-discover client then runs on neuronic against the
# Tinker API on train worker 0 (see CLIENT section at the bottom).
#
# ---------------------------------------------------------------------------
# WORKER SPLIT (v5p-32 = 4 hosts x 4 chips)
#
#   worker 0      tunix/MaxText trainer + Tinker API   (4 chips, 4-way FSDP)
#   workers 1,2   vLLM serving, 2 engines each, TP=2   (4 engines total)
#   worker 3      idle by default
#
# Why: the tunix backend supports a single train host (enforced by the
# launcher's backend-config build), so training is worker 0 — the same
# train/serve split qwen35-27b runs on its v5p-16 (train 0, vllm 1). Two
# serving hosts x 2 engines gives the validated 4-engine shape and a 4-URL
# CSV. Worker 3 is left unassigned on purpose: the 4-engine shape is what the
# 22k benchmark validated; grow with VLLM_WORKERS=1,2,3 (6 engines) once
# sustained serving is proven.
#
# ---------------------------------------------------------------------------
# WHY 2 ENGINES x TP=2 PER HOST (not 1 x TP=4)
#
# Muse-Glimmer has num_key_value_heads=2 and tpu-inference pads KV heads up
# to the TP size, so per-token KV doubles at TP=4 for zero benefit. Measured
# head-to-head at max-model-len 22528 on real erdos prompts, warmed and
# differenced (tpu/muse_glimmer/TP-BENCHMARK-22K.md, job 3714197):
#   - throughput at the RL profile (~64-96 concurrent, long generations):
#     2xTP=2 = 2570-2736 tok/s/host at 96-128 offered vs 1xTP=4's best
#     1620-1700 at 64-96 -> the split wins ~1.5-1.6x;
#   - KV capacity: 222.4 vs 134.3 max sequences @22528 (1.656x);
#   - batch-1 latency still belongs to the wide engine (92.5 vs 67.9 tok/s);
#     we are serving throughput, not latency.
#
# WHY THE TORCH MODEL (VLLM_MODEL_IMPL_TYPE=vllm, not the flax_nnx default)
#
# RL needs --enable-lora, which the JAX path cannot do (get_flax_model
# returns lora_manager=None). The torch path is TPU-proven at TP=2 AND TP=4
# on real weights: greedy 4/5 token-exact + the known 0.0-gap tie, 260
# LoRA-wrapped modules (5/layer incl. attn_gate_proj), full adapter
# load/swap/unload round trip (VLLM-IMPL.md sections 10-11). NOTE the model
# renamed the attention gate to attn_gate_proj precisely because vLLM matches
# LoRA target_modules on the LAST dotted component and self_attn.gate_proj
# collided with the MLP gate_proj; nothing here may reintroduce plain
# gate_proj for the attention gate.
#
# THE CLIENT CONSUMES THE MULTI-URL CSV NATIVELY (no client changes):
# skyrl/backends/vllm_sampling.py splits the comma-separated URL list at
# line 41 (_normalize_vllm_url_list), round-robins completions across the
# URLs at line 116 (_completion_url_for_index), and broadcasts every adapter
# load/upload to every server, verifying each (lines 167/210/218 + the
# sequential push_adapter loop). The launcher builds the CSV with BOTH ports
# of every serving host (base_urls_for_workers, VLLM_ENGINES_PER_HOST=2).
#
# ---------------------------------------------------------------------------
# WHAT REMAINS UNPROVEN (do not read this launch as fully de-risked)
#
#   1. Cold prefill at 22528: the benchmark's prefill probes hit vLLM's
#      prefix cache (TP-BENCHMARK-22K.md caveat 1). Unmeasured.
#   2. k/v-targeting adapter deltas under KV-head replication at TP>=4
#      (VLLM-IMPL.md section 9 residual) — MOOT at TP=2 (no replication).
#   3. Sustained serving over hours/days (README "Not proven").
#   4. An adapter trained by a REAL RL step through the tunix export path:
#      the on-slice LoRA round trips used synthetic PEFT adapters built from
#      the model dir. The tunix PEFT export writes module paths under
#      model.layers.* (tunix_backend.py _MAXTEXT_PROJ_TO_HF) while muse's HF
#      tree nests under model.language_model.layers.* — first bring-up must
#      verify the first uploaded adapter actually loads AND changes outputs.
#      (Exported targets are q/k/v/o_proj + gate/up/down_proj; the attention
#      gate is NOT adapted by tunix — MAXTEXT.md "LoRA targeting".)
#   5. Train-side sizing at 24576/vocab 202048: micro-batch and token budget
#      below follow the qwen35-27b precedent, not muse measurements. The
#      qwen35 deploy also found FLCE did NOT actually stop [N*S, V] logits
#      from materializing on TPU (memory: ttd-qwen35-tpu-server); watch the
#      first fb step's HBM before committing to a long run.
# =============================================================================
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

# --- slice / transport (existing slice; nothing is created here) -------------
export PROJECT="${PROJECT:-vision-mix}"
export ZONE="${ZONE:-us-east5-a}"
export TPU_NAME="${TPU_NAME:-sk7524-tunix-muse-v5p32-r1-east5a_spot}"
export REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
export SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/home/${REMOTE_USER}/.cache/huggingface}"
export REMOTE_HF_HOME

# --- worker split (see header) ----------------------------------------------
export TRAIN_WORKERS="${TRAIN_WORKERS:-0}"
export VLLM_WORKERS="${VLLM_WORKERS:-1,2}"

# --- model -------------------------------------------------------------------
# HF weights live in the shared GCS hub-format cache (refs/main -> a4e59da5,
# verified): gs://sk7524-tinker-tpu-us-east5/hf-cache/models--meta-models--
# Muse-Glimmer-30B. Staged scoped (NOT the whole hf-cache prefix — that also
# holds Qwen3.5-27B and would blow the boot disk) onto each serving host
# below, so vllm's --download-dir finds them locally.
export MODEL_NAME="${MODEL_NAME:-meta-models/Muse-Glimmer-30B}"
HF_HUB_ENTRY="models--meta-models--Muse-Glimmer-30B"
HF_ENTRY_GCS="${HF_ENTRY_GCS:-gs://sk7524-tinker-tpu-us-east5/hf-cache/${HF_HUB_ENTRY}}"

# --- serving: the validated 2xTP=2 torch-path shape --------------------------
export VLLM_MODEL_IMPL_TYPE="${VLLM_MODEL_IMPL_TYPE:-vllm}"       # torch model (LoRA)
export VLLM_TPU_BACKEND_TYPE="${VLLM_TPU_BACKEND_TYPE:-jax}"      # bench: vllm_tp4_bench_tpu.sh
export VLLM_SKIP_JAX_PRECOMPILE="${VLLM_SKIP_JAX_PRECOMPILE:-1}"  # bench: boots ~100-115s
export VLLM_ENGINES_PER_HOST="${VLLM_ENGINES_PER_HOST:-2}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-22528}"
# Pool supports ~111 seqs/engine at 22528; vLLM caps scheduling at the pool,
# exactly how prod qwen cells run 128. Keep 128 (TP-BENCHMARK-22K.md config).
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
# Power-of-2 batched-tokens: multimodal-arch models on TPU otherwise mismatch
# the LoRA punica warmup buckets (qwen35-27b.env note; muse IS registered via
# MuseGlimmerForConditionalGeneration). 4096 is also the exact shape the muse
# --enable-lora on-slice smoke booted with (VLLM-IMPL.md section 10.3).
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---max-num-batched-tokens 4096}"
export VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-32}"             # >=8 required; ttd uses 32
export MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
# Shared LOCAL XLA cache: both engines point at the same dir on purpose (2nd
# engine boots in ~79s off the 1st's compile — VLLM-IMPL.md 10.5). Local SSD,
# not the gcsfuse mount: compile-time writes to fuse have flaked before.
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/home/${REMOTE_USER}/vllm-xla-cache-muse}"
# Serving code: the FIXED tpu-inference build (stock qkv geometry + width
# assert + LoRA-bypass seam fix). Parent repo pins the submodule at this SHA.
export TPU_INFERENCE_FORK_URL="${TPU_INFERENCE_FORK_URL:-https://github.com/SachinKonan/tpu-inference.git}"
export TPU_INFERENCE_FORK_REF="${TPU_INFERENCE_FORK_REF:-afe0cb9e9bf259a072242c6f3279d92b702f9f2a}"
# muse_glimmer is unreleased in stable transformers: the serving venv needs
# transformers@main, tokenizers pinned alongside because @main is a moving
# target whose floor once broke a landed slice (vllm_tp4_bench_tpu.sh, the
# proven venv recipe; installed --no-deps --force-reinstall, LAST).
export VLLM_EXTRA_PIP_SPECS="${VLLM_EXTRA_PIP_SPECS:-'transformers @ git+https://github.com/huggingface/transformers@main' 'tokenizers>=0.23.1,<0.24.0'}"

# --- training: tunix/MaxText fork with muse support --------------------------
export TINKER_BACKEND="${TINKER_BACKEND:-tunix}"
export TUNIX_MODEL_SOURCE="${TUNIX_MODEL_SOURCE:-maxtext}"
export TUNIX_MAXTEXT_MODEL_NAME="${TUNIX_MAXTEXT_MODEL_NAME:-muse-glimmer-30b}"
# MAXTEXT.md section 5.1: branch muse-glimmer pushed to SachinKonan/maxtext
# at 4f65ba509 (verified reachable). Parity PASS at this commit (job 3686711).
export TUNIX_MAXTEXT_PIP_SPEC="${TUNIX_MAXTEXT_PIP_SPEC:-maxtext @ git+https://github.com/SachinKonan/maxtext.git@4f65ba509}"
# MAXTEXT.md section 3 launch spec: remat full + 4-way FSDP on the one train
# host (sizing: 12.97 GiB/chip weights). num_vocab_tiling>1 is REQUIRED for
# FLCE (decoder must return hidden — tunix_backend.flce_tile_size doc + the
# launcher's models.py patch); value 8 is the qwen35-27b finish-line
# precedent. FLCE tile 2048 per MAXTEXT.md (the [B,S,V] f32 logits at
# S=24576, V=202048 are the binding 18.50 GiB term).
export TUNIX_MAXTEXT_KWARGS="${TUNIX_MAXTEXT_KWARGS:-{\"remat_policy\":\"full\",\"ici_fsdp_parallelism\":4,\"num_vocab_tiling\":8}}"
export TUNIX_MAX_TARGET_LENGTH="${TUNIX_MAX_TARGET_LENGTH:-24576}"
export TUNIX_FLCE_TILE_SIZE="${TUNIX_FLCE_TILE_SIZE:-2048}"
# Sizing precedent, NOT muse-measured (unproven item 5): micro-batch from
# qwen35-27b.env; token budget = the qwen35 finish-line 45056 (2 x 22528).
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-8}"
export TUNIX_TRAIN_TOKEN_BUDGET="${TUNIX_TRAIN_TOKEN_BUDGET:-45056}"
# Converted orbax weights are already staged under the launcher's default
# TUNIX_MAXTEXT_CKPT_CACHE_GCS prefix: gs://sk7524-tinker-tpu-us-east5/
# skyrl-maxtext-ckpts/muse-glimmer-30b/0/items (verified). The launcher
# restores that subdir to local SSD before the engine loads, so no HF->orbax
# conversion runs at bring-up.

export VLLM_MAX_CONCURRENT_REQUESTS="${VLLM_MAX_CONCURRENT_REQUESTS:-256}"

# =============================================================================
# Preflights (CPU-only, fail loud before touching the slice)
# =============================================================================
if [[ "${MUSE_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  # TODO(blocking): the fixed tpu-inference commit afe0cb9e9 exists only in
  # the LOCAL submodule as of authoring (remote agent/muse-glimmer-text tip
  # is behind it; every on-slice run so far shipped it by git-archive+rsync,
  # not pip). The serving bootstrap installs git+URL@REF, so the SHA must be
  # reachable on the fork remote first:
  #     git -C third_party/tpu-inference push origin agent/muse-glimmer-text
  # This preflight verifies exactly that and aborts otherwise.
  sub="${repo_root}/third_party/tpu-inference"
  if git -C "$sub" fetch -q origin 2>/dev/null; then
    if ! git -C "$sub" branch -r --contains "$TPU_INFERENCE_FORK_REF" 2>/dev/null | grep -q .; then
      echo "PREFLIGHT FAIL: ${TPU_INFERENCE_FORK_REF} is not reachable on ${TPU_INFERENCE_FORK_URL}." >&2
      echo "Push the submodule branch first: git -C third_party/tpu-inference push origin agent/muse-glimmer-text" >&2
      exit 1
    fi
    echo "preflight: tpu-inference ${TPU_INFERENCE_FORK_REF:0:9} reachable on fork remote"
  else
    echo "preflight WARNING: could not fetch ${TPU_INFERENCE_FORK_URL}; cannot verify the pinned SHA is pushed." >&2
  fi
fi

# =============================================================================
# Scoped HF weight staging on the serving hosts (hub layout, this model only)
# =============================================================================
# shellcheck source=../tpu_ssh_lib.sh
source "${repo_root}/tpu/tpu_ssh_lib.sh"

if [[ "${MUSE_STAGE_WEIGHTS:-1}" == "1" ]]; then
  # NOTE this string is remote-executed via tpu_vm_ssh: keep it free of
  # command substitution and backticks (they would expand locally).
  stage_cmd="set -e
GCLOUD_BIN=\$HOME/google-cloud-sdk/bin/gcloud
if ! [ -x \$GCLOUD_BIN ]; then GCLOUD_BIN=gcloud; fi
mkdir -p ${REMOTE_HF_HOME}/hub/${HF_HUB_ENTRY}
\$GCLOUD_BIN storage rsync -r ${HF_ENTRY_GCS} ${REMOTE_HF_HOME}/hub/${HF_HUB_ENTRY}
test -s ${REMOTE_HF_HOME}/hub/${HF_HUB_ENTRY}/refs/main
echo WEIGHTS-STAGED
"
  IFS=',' read -r -a _muse_vllm_workers <<< "$VLLM_WORKERS"
  for worker in "${_muse_vllm_workers[@]}"; do
    echo "Staging ${HF_HUB_ENTRY} on serving worker ${worker} (scoped, hub layout)"
    tpu_vm_ssh "$worker" "$stage_cmd"
  done
fi

# =============================================================================
# Launch: colocated tunix trainer + 4 serving engines, 4-URL CSV
# =============================================================================
exec bash "${repo_root}/tpu/start_colocated_vllm_tinker.sh"

# =============================================================================
# CLIENT (runs on neuronic, NOT here) — qwen35 pattern, muse knobs:
#
#   sbatch --export=ALL,\
#     TPU_NAME=sk7524-tunix-muse-v5p32-r1-east5a_spot,\
#     MODEL_NAME=meta-models/Muse-Glimmer-30B,\
#     RENDERER_NAME=muse_glimmer,\
#     CONTEXT_WINDOW=22528,\
#     EXPERIMENT_NAME=erdos-muse30b,\
#     TTD_RUN_DIR=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_muse30b \
#     tpu/run_ttd_qwen35_neuronic.sbatch
#
#   - RENDERER_NAME=muse_glimmer (registry in third_party/discover
#     ttt_discover/tinker_utils/renderers.py, merged at 6f0c076; xhigh
#     variant exists but check the phase-1 token budget first).
#   - Stop ids: generation_config.json lists TWO eos ids, 200001
#     <|end_of_text|> and 200008 <|eot|> (verified against the shipped
#     files). The renderer's get_stop_sequences returns BOTH — treating only
#     one as stop silently drops rollouts as truncated/format errors. The
#     server needs no extra stop plumbing; stops ride each sampling request.
#   - CONTEXT_WINDOW must not exceed the serving max-model-len 22528, or
#     vLLM 400s context-overflow requests.
#   - The TPU-side worktree sync EXCLUDES third_party/discover (launcher tar
#     excludes; verified) — the discover submodule the client needs lives on
#     neuronic only.
# =============================================================================
