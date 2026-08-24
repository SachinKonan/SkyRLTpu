#!/usr/bin/env bash
# jobman command.cmd for a Stage-A cell (runs on ALL workers, branches on
# JOBMAN_WORKER_ID). w0 orchestrates: client venv, engines (tinker trainer on
# itself + vLLM on w1-3, via the PROVEN start_colocated_vllm_tinker.sh invoked
# from w0 over the slice's internal IPs with the jobman ssh key), and the Ray
# grading cluster. Other workers no-op -- w0 reaches them over ssh exactly the
# way the login-node bring-up did, so the engine recipe stays byte-identical.
#
# Idempotent: healthy engines are detected and left alone, so a jobman loop
# iteration after a mere monitor-ssh hiccup does not restart a working slice.
set -euo pipefail
: "${JOBMAN_WORKER_ID:?}"; : "${JOBMAN_TPU_INTERNAL_IPS:?}"; : "${CELL:?}"
[ "$JOBMAN_WORKER_ID" = "0" ] || { echo "worker $JOBMAN_WORKER_ID: engines are driven from w0"; exit 0; }

export PATH="$HOME/.local/bin:$PATH"
REPO="$HOME/SkyRLTpu-league"
KEY="$HOME/.ssh/jobman_tpu_ed25519"
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
INT="$JOBMAN_TPU_INTERNAL_IPS"
W0INT=$(echo "$INT" | cut -d, -f1)
# Host count from the IP list itself: a stage cell passes 4 IPs (unchanged
# behavior); a meta member passes its 4 + any spare hosts, which become extra
# vLLM replicas + ray workers with no other change.
NHOSTS=$(echo "$INT" | awk -F, '{print NF}')
VLLM_IDXS=$(seq -s, 1 $((NHOSTS-1)))
ln -sfn "$REPO" "$HOME/ttd-client"

# --- client venv (idempotent) ------------------------------------------------
if [ ! -x "$REPO/third_party/discover/.venv-ttd-discover/bin/python" ]; then
  ( cd "$REPO/third_party/discover" && uv sync --extra math --python 3.11 > ~/venv-build.log 2>&1 \
      && ln -sfn .venv .venv-ttd-discover )
  "$REPO/third_party/discover/.venv-ttd-discover/bin/python" -c "import tinker,numpy,wandb" \
    || { echo "client venv build FAILED"; tail -5 ~/venv-build.log; exit 1; }
fi
echo "client venv OK"

# --- engines: skip when healthy ---------------------------------------------
engines_healthy() {
  curl -fsS -m6 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1 || return 1
  vllm_healthy
}
tinker_healthy() { curl -fsS -m6 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1; }
vllm_healthy() {
  # EVERY engine, not just the first: with ENGINES_PER_HOST=2 each host serves
  # 8001 AND 8002 (engine e listens on VLLM_PORT+e). Probing only 8001 would
  # call a host healthy with a dead second engine -- bring-up would be skipped
  # and the client, which round-robins across all 6 URLs, would sample against
  # a dead endpoint every other request.
  local ip e port
  for ip in $(echo "$INT" | cut -d, -f2-$NHOSTS | tr ',' ' '); do
    for (( e=0; e<${ENGINES_PER_HOST:-1}; e++ )); do
      port=$(( 8001 + e ))
      curl -fsS -m6 "http://$ip:$port/v1/models" >/dev/null 2>&1 || return 1
    done
  done
  return 0
}
# --- model dimension (cell prefix g- = gemma-4-31B, else qwen3.5-27B) --------
# Gemma values are the league-validated uniform-10240 config (bringup_v5p64
# step 4b): tile 1024 / nvt 32 / budget 40960 / vLLM 16k with its own caches.
# Qwen cells keep the Stage-A per-suffix tile logic below, byte-identical.
PIP="maxtext @ git+https://github.com/SachinKonan/maxtext.git@skyrl/qwen35-dense"
VLLM_IMPL=vllm
TPUINF_REF=skyrl/v0.23.0-lora
TF_VERSION=5.8.0
TP_SIZE=4; ENGINES_PER_HOST=1
MAX_NUM_SEQS=128
SKIP_PRECOMPILE=0
EXTRA_PIP=""
LORA_RETRIES=3; LORA_RETRY_SLEEP=2
REQ_TIMEOUT=300
TPU_BACKEND=torchax
FREE_BASE_STATE=0
ROUTE_PREFIX=0
UNSET_PLUGINS=0
VLLM_XARGS="--max-num-batched-tokens 8192 --gpu-memory-utilization 0.85"
HF_OFFLINE=0
case "$CELL" in
  g-*)
    MODEL_NAME=google/gemma-4-31B-it; MAXTEXT_MODEL=gemma4-31b
    MAXTGT=10240; BUDGET=40960; UNIFORM=10240
    VLLM_LEN=16384
    XLA_GCS="gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-gemma4-31b-16k"
    JAX_CACHE_GCS="gs://sk7524-tinker-tpu-us-east5/jax-compile-cache-gemma4-10k"
    HF_GCS="gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4"
    ;;
  m-*)
    # Muse-Glimmer-30B. Serving = the TORCH implementation (tpu-inference vllm
    # wrapper path), which is what makes RL possible at all: the JAX/flax_nnx
    # path hardcodes lora_manager=None, so LoRA weight sync -- our whole
    # train->sample loop -- cannot work there. Deployment shape is the validated
    # 2xTP=2: muse has 2 KV heads, so TP=2 is the zero-padding point (TP=4 pads
    # 2->4 and stores every KV twice), and each non-train host runs 2 engines =>
    # 6 serving engines per v5p-32. See tpu/muse_glimmer/{VLLM-IMPL,README}.md.
    # transformers: NOT pinned -- vllm-tpu 0.23.0 pulls 5.15.0, the first
    # release carrying muse_glimmer's modeling code (no .py in the checkpoint,
    # no trust_remote_code path). VLLM_EXTRA_PIP_SPECS overlays @main with the
    # tokenizers co-pin, per run_muse_rl.sh.
    MODEL_NAME=meta-models/Muse-Glimmer-30B; MAXTEXT_MODEL=muse-glimmer-30b
    PIP="maxtext @ git+https://github.com/SachinKonan/maxtext.git@4f65ba509"
    # Sizing per run_muse_rl.sh (the validated RL spec), not the older
    # MAXTEXT.md training-half doc: target 22528 = the serving ctx.
    # BUDGET: the spec started at 45056 (2 x uniform) with the explicit note
    # "raise only after watching the first fb step's HBM (unproven item 5)".
    # The first real fb step (2026-08-21, attempt 8) answered it: the program
    # asked to reserve 60.47G with 55.48G free -- over by 9%. One uniform seq
    # per fb call halves the arena (~30G, fits with headroom); costs 2x fb
    # calls per train step, changes nothing about what is trained.
    # 18432 == qwen's exact train shape. The 22528 fb program asks 60.47G
    # regardless of token budget (verified twice live: identical ask at
    # budget 45056 and 22528) -- the arena belongs to the [1, 22528] pass
    # itself, dominated by the 13 full-attention layers' S^2 backward.
    # 18432 scales it to ~40G (fits, ~16G headroom) and, paired with
    # CTX/PHASE1 18432/13824 in launch_cell.sh, nothing is dropped from
    # training -- rollouts simply cap at qwen's budget, which also removes
    # the longer-leash confound from the A/B.
    # BUDGET = 4 x UNIFORM, i.e. qwen's packing. Measured on-slice at steady
    # state (3 fb calls, first discarded as JIT): 1 datum 27.3s, 4 datums 27.4s
    # -- FOUR DATUMS COST THE SAME AS ONE. The fb is not compute-bound; it is
    # bound by a fixed per-microbatch cost, almost certainly the FSDP all-gather
    # of 55.7GB of weights. So packing is ~4x free throughput: 27.3 -> 6.85 s
    # per datum. At ~296 trained datums/step that is train 7983s -> ~2030s, and
    # since 2030 < the 6972s sampling window it should finally disappear under
    # the pipeline instead of serialising after it.
    #
    # remat_policy stays "full": measured 27.4s (none) vs 27.5s (full) at 4
    # datums, i.e. free in time, and it saves a large amount of HBM. Dropping it
    # would be a pure memory regression for no speed.
    #
    # Verified to FIT at 4 x 18432 with free_base_state, which is production's
    # worst case since uniform mode pads every datum to 18432.
    MAXTGT=18432; BUDGET=73728; UNIFORM=18432
    VLLM_LEN=22528
    VLLM_IMPL=vllm
    # The exact SHA run_muse_rl.sh pins: carries the torch muse model AND the
    # stock-qkv-geometry fix (first-request empty-v crash) + width assert.
    # skyrl/v0.23.0-lora predates all of it -- that ref has no torch muse model.
    TPUINF_REF=afe0cb9e9bf259a072242c6f3279d92b702f9f2a
    TF_VERSION=""
    # jax, NOT the torchax default (run_muse_rl.sh:134, the validated spec).
    # Under torchax the plugin defers to vLLM's own model registry, which has
    # no native muse -- it silently serves the generic transformers-backend
    # fallback, which boots, lists models, even loads LoRA adapters, and then
    # kills EngineCore on the FIRST generate (NonConcreteBooleanIndexError:
    # modeling_muse_glimmer.py boolean-mask __setitem__ cannot trace). The
    # fork's torch muse model (tpu_inference/models/vllm/muse_glimmer.py) is
    # only reachable through the jax backend wrapper.
    TPU_BACKEND=jax
    # VLLM_PLUGINS is an allow-list; pinning it to the resolver excluded the
    # fork's vllm.general_plugins registration (register_layers), so vLLM's
    # registry never learned MuseGlimmerForCausalLM and silently served the
    # transformers fallback -- EngineCore-fatal on the first generate. The
    # validated smoke ran UNSET (all plugins load, resolver included).
    UNSET_PLUGINS=1
    # Reclaims 35.5 GiB. Measured on this cell: create_model took HBM in_use from
    # 12.97 -> 49.79 GiB, rank-independent (rank 4 within 1.4 GiB of rank 32), i.e.
    # duplicated base weights, not adapters -- qwix's wrap does not share the merged
    # arrays and _init_lora_state builds a SECOND whole model to read 0.77 GiB of
    # seeds. That is why the [1, 18432] fb asked 56.99G against 56.45G free and no
    # budget/tiling/remat knob ever moved it. With the release: delta 1.33 GiB, and
    # the same fb COMPLETES in 46.9s (loss -0.00215, peak 60.9 of 95.7 GiB).
    FREE_BASE_STATE=1
    # REVERTED, and left here as the record. The theory was that phase 2
    # re-sends prompt + ALL phase-1 tokens (~13.8k) and index round-robin lands
    # it on the engine holding that KV only 1 time in 6. Pinning by prompt
    # prefix was measured and did NOT work: windowed prefix-cache hit rate was
    # 36.1% (queries +666988, hits +240992 over a live 10-min sampling window)
    # against a 49.6% lifetime baseline -- i.e. worse, not better -- while step
    # 2's sampling ran 7060s vs step 1's 5949s on FEWER generated tokens
    # (throughput -25%). Load balance was fine (six engines within 2%), so the
    # allocator worked; the affinity itself did not take. Most likely the
    # phase-2 request never reaches the patched router with its prompt_ids and
    # silently falls back to index routing -- worth confirming offline before
    # anyone tries this again. Re-enable with ROUTE_PREFIX=1 only with a
    # windowed hit-rate measurement to prove it.
    ROUTE_PREFIX=0
    VLLM_XARGS="--max-num-batched-tokens 8192 --gpu-memory-utilization 0.85"
    TP_SIZE=2; ENGINES_PER_HOST=2
    # 64/engine x 6 engines = 384 pooled scheduler slots, under the measured
    # 222-sequence KV capacity per host pair and matched to the RL burst:
    # GRPO 16x32 = 512 rollouts => ~85/engine offered, so the scheduler stays
    # the limiter rather than the KV pool.
    MAX_NUM_SEQS=64
    # Both from run_muse_rl.sh, and the first is load-bearing: with precompile
    # ON, capture_model() -> maybe_select_dummy_loras -> _set_active_loras leaks
    # a JAX tracer through torchax and every engine dies with
    # UnexpectedTracerError (observed live, both engines, deterministic). The
    # validated path never runs that code. Cost: first requests compile lazily.
    SKIP_PRECOMPILE=1
    # Skipping precompile moves compilation into the first requests, so an
    # engine can be mid-compile (minutes) when the client pushes its LoRA
    # adapter. The default 3 retries x 2s = ~4s of tolerance, so the upload
    # fails, the client raises, and jobman relaunches it -- observed as a new
    # adapter every ~3 min, 35 restarts, never reaching the first sampling
    # burst. qwen/gemma never hit this because they precompile at boot and are
    # responsive by the time their client connects.
    LORA_RETRIES=20; LORA_RETRY_SLEEP=30
    # The first adapter LOAD on an engine compiles the LoRA graphs (minutes).
    # At the default 300s socket timeout the client hangs up first, the engine
    # finishes and logs 200 to nobody, and every retry hits the extracted-dir
    # path -- observed live: /v1/models listed all three "failed" adapters.
    # 1800s lets the honest first attempt win; the /v1/models fallback in
    # push_adapter covers anything longer.
    REQ_TIMEOUT=1800
    EXTRA_PIP="'transformers @ git+https://github.com/huggingface/transformers@main' 'tokenizers>=0.23.1,<0.24.0'"
    XLA_GCS="gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-mg-22k-tp2"
    JAX_CACHE_GCS="gs://sk7524-tinker-tpu-us-east5/jax-compile-cache-muse-22k"
    HF_GCS="gs://sk7524-tinker-tpu-us-east5/hf-cache"
    HF_OFFLINE=0
    ;;
  *)
    MODEL_NAME=Qwen/Qwen3.5-27B; MAXTEXT_MODEL=qwen3.5-27b
    MAXTGT=22528; BUDGET=73728; UNIFORM=18432
    # Serve just above the client's context, not 4k above it. TTD_M0_CONTEXT_WINDOW
    # is 18432 and the two-phase completer budgets every phase against that, so
    # prompt+generated can never exceed it; 22528 was reserving 22% more KV
    # envelope per sequence than any rollout can use. 1024 tokens of margin kept
    # deliberately -- serving exactly at the client ceiling turns any off-by-one
    # into a context-overflow 400, which is a known failure mode here.
    VLLM_LEN=19456
    # 16384: qwen's phase-2 re-prefill is prompt(3085) + phase1(13824) ~= 17k
    # tokens, which at 8192 was split into 3 chunks. 0.90: qwen's weights are
    # 13.5 GiB/chip at TP=4, so there is room for more KV blocks. A too-high
    # value fails loudly at boot rather than corrupting a run.
    VLLM_XARGS="--max-num-batched-tokens 16384 --gpu-memory-utilization 0.90"
    # New prefix: max-model-len is part of the compiled shape key, so the 22k
    # cache cannot be reused. First bring-up after this pays a cold compile.
    XLA_GCS="gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-qwen35-19k"
    JAX_CACHE_GCS="gs://sk7524-tinker-tpu-us-east5/jax-compile-cache-qwen35-18k"
    HF_GCS="gs://sk7524-tinker-tpu-us-east5/hf-cache"
    ;;
esac
pick_tiles() {
  case "$MAXTEXT_MODEL" in
    gemma4-31b)
      FLCE_TILE=1024; VOCAB_TILING=32
      MT_KWARGS="{\"num_vocab_tiling\": $VOCAB_TILING}" ;;
    muse-glimmer-30b)
      # nvt 32, not the RL spec's 8: the fb arena measured a ~41G
      # seq-independent constant (60.47G @22528 vs 56.99G @18432 -- only the
      # ~850KB/token slope moved), i.e. vocab-sized workspace dominates, and
      # V/nvt is its divisor. 32 is the value muse's own MAXTEXT.md
      # training-half config validated (gemma runs 32, qwen 64; 8 was the
      # outlier). FLCE tile stays 2048 -- both validated configs agree, and
      # smaller FLCE tiles grow the scorer (the known trap).
      # FLCE tile 1024 (gemma's value, larger-vocab precedent): at 18432 the
      # fb ask is 56.99G vs 56.45G free -- short by 0.54G. Measured knob
      # response: budget none, uniform 850KB/token, nvt none; the FLCE
      # working buffers are the next term with a validated smaller setting.
      FLCE_TILE=1024; VOCAB_TILING=32
      MT_KWARGS="{\"remat_policy\": \"full\", \"ici_fsdp_parallelism\": 4, \"num_vocab_tiling\": $VOCAB_TILING, \"parameter_memory_host_offload\": true}" ;;
    *)
      case "$CELL" in
        *k-j) FLCE_TILE=512; VOCAB_TILING=64 ;;
        *-j)  FLCE_TILE=2048; VOCAB_TILING=8 ;;
        *)    FLCE_TILE=512; VOCAB_TILING=64 ;;
      esac
      MT_KWARGS="{\"num_vocab_tiling\": $VOCAB_TILING}" ;;
  esac
  # ONE compiled shape, always. The scorer is pinned to the same uniform length
  # the fb path uses (TUNIX_UNIFORM_SEQ_LEN), so a run ever holds exactly two
  # pinned programs -- fb + scorer -- instead of growing a new jit_fwd per
  # 8192-bucket as sequences lengthen. Not cell-specific: any cell that scores
  # (KL penalty OR the measure-only pass) must never build a bucket ladder.
  SCORE_FIXED=$UNIFORM
}

# NO w0 HF weight staging. w0 runs the trainer (which loads MaxText/orbax, not
# safetensors) plus the client (which needs only tokenizer/config). Rsyncing the
# whole model dir here put 71G of gemma-4 safetensors on a 97G boot disk and left
# no room for the tunix install -- `uv pip install ... aqtp` died with
# "No space left on device", the trainer never started, and the cell looped.
# Tokenizer/config resolve from the HF hub (all our models are public repos).

# Trainer JAX compile cache: restore before bring-up, then publish on a cadence
# so a preempted node still leaves its compiles behind (same reasoning as the
# vLLM cache seed-back; keyed by HLO hash, so a miss just recompiles).
# MaxText owns this path: base.yml sets `jax_cache_dir: "~/jax_cache"` and calls
# JAX's cache init with it, which OVERRIDES the JAX_COMPILATION_CACHE_DIR we
# export. We synced ~/jax-compile-cache for weeks while every compile landed in
# ~/jax_cache -- all three per-model GCS caches sat at 0 MB while the trainer
# host held 120 MB of real entries including jit_forward_backward_fn. Net effect:
# every fresh VM recompiled the fb from scratch (~25 min for muse), on a zone
# that preempts every few hours. Sync the directory MaxText actually writes.
JAX_CACHE_LOCAL="${TUNIX_JAX_CACHE_LOCAL:-$HOME/jax_cache}"
mkdir -p "$JAX_CACHE_LOCAL"
if [ -n "${JAX_CACHE_GCS:-}" ]; then
  gcloud storage rsync -r "$JAX_CACHE_GCS" "$JAX_CACHE_LOCAL" >/dev/null 2>&1 \
    && echo "trainer JAX cache restored from $JAX_CACHE_GCS" \
    || echo "trainer JAX cache empty/miss (will compile)"
  # Publish cadence: a preemption between "compile finished" and "next tick"
  # loses the whole compile (bit muse live: ~25min fb compile, 10min tick,
  # slice died in the gap -- cache stayed at 0 after the attempt). Short spot
  # windows need a tight cadence; the rsync is checksum-additive so an
  # empty-delta tick is nearly free. 180s x 480 spans the same 24h.
  ( for _i in $(seq 1 480); do
      sleep "${JAX_CACHE_PUBLISH_SECS:-180}"
      gcloud storage rsync -r "$JAX_CACHE_LOCAL" "$JAX_CACHE_GCS" >/dev/null 2>&1
    done ) >/dev/null 2>&1 &
fi
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_LOCAL"

if engines_healthy; then
  echo "engines already healthy -- skipping bring-up"
elif ! tinker_healthy && vllm_healthy; then
  # Surgical recovery for a wedged/dead TRAINER with healthy samplers (the
  # ENGINE-SICK case): restart only the tinker server -- a process restart is
  # the only defragmentation the TPU runtime has, and bouncing three healthy
  # vLLM workers would waste ~20 min of cache reloads for nothing. The client
  # re-registers against the fresh registry at its next launch.
  echo "trainer down, vLLM healthy -- surgical tinker-only restart"
  tmux kill-session -t skyrl-tinker 2>/dev/null || true; sleep 3
  pick_tiles
  env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS="$INT" TPU_INTERNAL_IPS="$INT" TPU_NAME="stagea-$CELL" \
    PROJECT=vision-mix ZONE=us-east5-a REMOTE_USER=sk7524_princeton_edu SSH_KEY_FILE="$KEY" \
    TINKER_BACKEND=tunix TRAIN_WORKERS=0 VLLM_WORKERS=$VLLM_IDXS VLLM_RAY_EXECUTOR=0 VLLM_CLIENT_SIDE_ROUND_ROBIN=1 \
    VLLM_MODEL_IMPL_TYPE="$VLLM_IMPL" TPU_INFERENCE_FORK_REF="$TPUINF_REF" HF_HUB_OFFLINE="$HF_OFFLINE" \
    VLLM_TRANSFORMERS_VERSION="$TF_VERSION" VLLM_TP_SIZE="$TP_SIZE" VLLM_ENGINES_PER_HOST="$ENGINES_PER_HOST"  \
    VLLM_SKIP_JAX_PRECOMPILE="$SKIP_PRECOMPILE" VLLM_EXTRA_PIP_SPECS="$EXTRA_PIP"  \
    VLLM_LORA_LOAD_RETRIES="$LORA_RETRIES" VLLM_LORA_LOAD_RETRY_SLEEP_SEC="$LORA_RETRY_SLEEP" \
    VLLM_REQUEST_TIMEOUT_SEC="$REQ_TIMEOUT" VLLM_TPU_BACKEND_TYPE="$TPU_BACKEND" VLLM_UNSET_PLUGINS="$UNSET_PLUGINS" \
    TUNIX_FREE_BASE_STATE="$FREE_BASE_STATE" VLLM_ROUTE_BY_PROMPT_PREFIX="$ROUTE_PREFIX" \
    MODEL_NAME="$MODEL_NAME" TUNIX_MAXTEXT_MODEL_NAME="$MAXTEXT_MODEL" TUNIX_MAXTEXT_PIP_SPEC="$PIP" \
    TUNIX_MAXTEXT_KWARGS="$MT_KWARGS" \
    TUNIX_MAX_TARGET_LENGTH=$MAXTGT TUNIX_TRAIN_TOKEN_BUDGET=$BUDGET TUNIX_FLCE_TILE_SIZE=$FLCE_TILE TRAIN_MICRO_BATCH_SIZE=1 \
    TUNIX_UNIFORM_SEQ_LEN=$UNIFORM TUNIX_SEQ_BUCKETS="4096,8192,12288,16384,20480" TUNIX_MINIMAL_FB_OUTPUT=1 \
    SKYRL_SCORE_FIXED_LEN=$SCORE_FIXED \
    READY_ATTEMPTS=900 SYNC_SKYRL=0 START_VLLM=0 START_TINKER=1 \
    bash "$REPO/tpu/start_colocated_vllm_tinker.sh" > ~/tinker-restart.log 2>&1 || true
  if ! tinker_healthy; then
    # ESCALATE, do not just exit: the next attempt would see vLLM still healthy,
    # take this same surgical branch, and fail identically -- an infinite loop
    # (observed live: 96 retries on g-ttd-n-j, ~15h and six slices burned, zero
    # rows). Tear down vLLM so the next attempt is forced through a FULL
    # bring-up, which is the only path that rebuilds trainer state from scratch.
    echo "tinker-only restart FAILED -- tearing down vLLM to force a full rebuild next attempt"
    tail -6 ~/tinker-restart.log 2>/dev/null || true
    for ip in $(echo "$INT" | cut -d, -f2-$NHOSTS | tr ',' ' '); do
      timeout 60 ssh $SSHO sk7524_princeton_edu@"$ip" \
        "tmux kill-session -t skyrl-vllm 2>/dev/null; pkill -f '[v]llm serve' 2>/dev/null; true" 2>/dev/null || true
    done
    exit 1
  fi
  echo "trainer restarted (vLLM untouched)"
else
  echo "engine bring-up ($MAXTEXT_MODEL uniform=$UNIFORM budget=$BUDGET)..."
  # Erdos cells (long sequences, 18432-class fb buckets) OOM'd at compile with
  # the league tiles on these builds: HLO temporaries 111G vs 95.7G/chip, every
  # train step, silently caught by the ensemble guard -- the cells sampled
  # without training. Heavier tiling (the values the gemma engine has always
  # used) shrinks the fb program. JSSP sequences land in small buckets and
  # trained fine, so -j cells keep the faster original tiles.
  # K arms get small tiles even on JSSP: the penalty pass pins its own ~16G
  # scoring arena beside the fb arena, and grpo-k-j proved 2048/8 + penalty
  # does not fit (1/9 steps trained). Non-K JSSP keeps the faster tiles.
  pick_tiles
  env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS="$INT" TPU_INTERNAL_IPS="$INT" TPU_NAME="stagea-$CELL" \
    PROJECT=vision-mix ZONE=us-east5-a REMOTE_USER=sk7524_princeton_edu SSH_KEY_FILE="$KEY" \
    TINKER_BACKEND=tunix TRAIN_WORKERS=0 VLLM_WORKERS=$VLLM_IDXS VLLM_RAY_EXECUTOR=0 VLLM_CLIENT_SIDE_ROUND_ROBIN=1 \
    VLLM_MODEL_IMPL_TYPE="$VLLM_IMPL" TPU_INFERENCE_FORK_REF="$TPUINF_REF" HF_HUB_OFFLINE="$HF_OFFLINE" \
    VLLM_TRANSFORMERS_VERSION="$TF_VERSION" VLLM_TP_SIZE="$TP_SIZE" VLLM_ENGINES_PER_HOST="$ENGINES_PER_HOST"  \
    VLLM_SKIP_JAX_PRECOMPILE="$SKIP_PRECOMPILE" VLLM_EXTRA_PIP_SPECS="$EXTRA_PIP"  \
    VLLM_LORA_LOAD_RETRIES="$LORA_RETRIES" VLLM_LORA_LOAD_RETRY_SLEEP_SEC="$LORA_RETRY_SLEEP" \
    VLLM_REQUEST_TIMEOUT_SEC="$REQ_TIMEOUT" VLLM_TPU_BACKEND_TYPE="$TPU_BACKEND" VLLM_UNSET_PLUGINS="$UNSET_PLUGINS" \
    TUNIX_FREE_BASE_STATE="$FREE_BASE_STATE" VLLM_ROUTE_BY_PROMPT_PREFIX="$ROUTE_PREFIX" \
    MODEL_NAME="$MODEL_NAME" TUNIX_MAXTEXT_MODEL_NAME="$MAXTEXT_MODEL" TUNIX_MAXTEXT_PIP_SPEC="$PIP" \
    TUNIX_MAXTEXT_KWARGS="$MT_KWARGS" \
    TUNIX_MAX_TARGET_LENGTH=$MAXTGT TUNIX_TRAIN_TOKEN_BUDGET=$BUDGET TUNIX_FLCE_TILE_SIZE=$FLCE_TILE TRAIN_MICRO_BATCH_SIZE=1 \
    TUNIX_UNIFORM_SEQ_LEN=$UNIFORM TUNIX_SEQ_BUCKETS="4096,8192,12288,16384,20480" TUNIX_MINIMAL_FB_OUTPUT=1 \
    SKYRL_SCORE_FIXED_LEN=$SCORE_FIXED \
    VLLM_MAX_MODEL_LEN=$VLLM_LEN VLLM_MAX_NUM_SEQS=$MAX_NUM_SEQS VLLM_XLA_CACHE_PATH=/home/sk7524_princeton_edu/vllm-xla-cache-local \
    VLLM_XLA_CACHE_GCS="$XLA_GCS" \
    HF_CACHE_GCS="$HF_GCS" \
    VLLM_EXTRA_ARGS="$VLLM_XARGS" \
    READY_ATTEMPTS=900 SYNC_SKYRL=1 START_VLLM=1 START_TINKER=1 \
    bash "$REPO/tpu/start_colocated_vllm_tinker.sh" > ~/engine-bringup.log 2>&1 || true
  curl -fsS -m8 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1 \
    || { echo "engine bring-up FAILED"; tail -8 ~/engine-bringup.log; exit 1; }
  echo "engines UP"
fi

# --- ray grading cluster (idempotent) ---------------------------------------
RAYBIN="$REPO/third_party/discover/.venv-ttd-discover/bin/ray"
if ! "$RAYBIN" status >/dev/null 2>&1; then
  pkill -f "ray/core" 2>/dev/null || true; sleep 2
  "$RAYBIN" start --head --port=6379 --num-cpus=0 --disable-usage-stats >/tmp/ray-head.log 2>&1
  echo "ray head started"
fi
RAYV=$("$REPO/third_party/discover/.venv-ttd-discover/bin/python" -c "import ray; print(ray.__version__)")
for ip in $(echo "$INT" | cut -d, -f2-$NHOSTS | tr ',' ' '); do
  timeout 900 ssh $SSHO sk7524_princeton_edu@"$ip" "
    export PATH=\$HOME/.local/bin:\$PATH
    pgrep -f '[r]ay/core' >/dev/null && { echo \"ray already on \$(hostname)\"; exit 0; }
    [ -x ~/.venvs/grader/bin/ray ] || {
      uv venv ~/.venvs/grader --python 3.11 >/dev/null 2>&1
      uv pip install --python ~/.venvs/grader/bin/python 'ray==$RAYV' numpy scipy shapely numba scikit-learn psutil >/dev/null 2>&1
    }
    ~/.venvs/grader/bin/ray start --address=$W0INT:6379 --num-cpus=150 --disable-usage-stats >/tmp/ray-worker.log 2>&1 && echo \"ray worker \$(hostname)\"
  " 2>/dev/null || echo "ray worker $ip FAILED (grading degrades, not fatal)"
done
echo "cell worker 0 ready ($CELL)"
