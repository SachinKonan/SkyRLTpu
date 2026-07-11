# Citation v4 on TPU

This directory launches Qwen3.5-9B LoRA SFT and citation-v4 policy RL on a
two-host `v5p-16` using the EasyDeL Tinker backend. Model and dataset archives
are staged once from a same-zone GCS bucket to each TPU VM's local disk.

## SFT

The full conservative dataset is the 6,962-example, 60K-context tokenized file:

```text
/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v4/tpu_stage/conservative_qwen35_ctx60k_train.jsonl.gz
```

Launch the full run from the repository root:

```bash
MODE=full \
RUN_ID=citation-sft-qwen35-9b-conservative-$(date -u +%Y%m%d-%H%M%S) \
./tpu/citation_v4/submit_citation_sft.sh
```

The defaults are LoRA rank 32, learning rate `5e-6`, token-mean loss, batch
size 2, TP4/SP2, and a 60K sequence limit. State checkpoints are named
`examples_002000`, `examples_004000`, `examples_006000`, and
`final_006962`. The launcher mirrors checkpoint archives, client state, and a
consistent Tinker SQLite snapshot to GCS every five minutes, so a spot
reallocation can reuse the same `RUN_ID`.

Use `MODE=canary` for the five-example real-data long-context canary.

## Citation RL

The TPU policy path uses the same citation-v4 text environment as GPU RL:

- raw `<question>`, `<search>`, `<information>`, `<citation>`, and `<done>` protocol;
- at most four question sections and four searches per section;
- retrieve 50 candidates, rank-normalized reranking with semantic weight 0.1
  and citation weight 0.9, then expose top 5;
- citation-count beta 0.5 and at most 12 displayed authors;
- recall reward, with protocol violations and over-citation receiving zero;
- temperature 0.6, token-mean policy loss, prompt-group GRPO standardization,
  CISPO ratio clipping `[1, 6]`, and token TIS capped at 2.

The retriever remains a gpu-test sidecar. Model generation, log-prob scoring,
forward/backward, AdamW, and checkpointing run on TPU.

The EasyDeL adapter submits all pending prompts for one model/checkpoint to
eSurge as one batch. Per-request seeds, remaining-token budgets, stop rules,
logprobs, and output order are preserved through eSurge's sampling-parameter
callback. This is required for production `Bp x n` throughput; the original
adapter collected requests together but generated them serially.

The Tinker clip thresholds `[1, 6]` are the direct ratio form of the flat
SkyRL run's CISPO epsilon settings `[0, 5]`: `[1-eps_low, 1+eps_high]`.

### Easy launch

Prepare a sanitized RL seed from the selected SFT checkpoint as described
below, then provision the TPU server, retriever chain, and RL head together:

```bash
RESTORE_RESULT_PREFIX=gs://<bucket>/<sanitized-rl-seed> \
TINKER_INITIAL_STATE_PATH='tinker://<sft-model-id>/weights/final_006962' \
RUN_ID=citation-rl-qwen35-9b-$(date -u +%Y%m%d-%H%M%S) \
SWEEP_BP=40 \
SWEEP_N=30 \
SWEEP_LR=3e-6 \
MAX_STEPS=100 \
./tpu/citation_v4/submit_citation_rl.sh
```

The wrapper creates the two-host `v5p-16`, restores the clean Tinker
database/checkpoint, waits through allocation and API startup, chains
`gpu-test` retrievers, runs the head client, waits for the final 60-second
checkpoint mirror, uploads the head checkpoint metadata, signals clean
shutdown, and explicitly removes its TPU VM, queued resource, and retriever.
A run-directory completion sentinel makes resumed controllers idempotent rather
than launching a duplicate RL head. The defaults match the current flat v4 setup where backend
semantics overlap: top-5 after 50-candidate reranking, `Bp=40`, `n=30`,
LR `3e-6`, GRPO standardization, CISPO, TIS, and token-mean reduction.

### Start the TPU API

For RL initialized from an SFT run, restore that run's result prefix into a
new, separate RL result prefix:

```bash
SFT_RESULTS=gs://hk4638-autoresearch-tpu-us-east5/skyrl-tpu/citation-v4/sft/\
<sft-run-id>/results

MODE=canary \
WORKLOAD_MODE=server \
RESTORE_RESULT_PREFIX="$SFT_RESULTS" \
RUN_ID=citation-rl-qwen35-9b-$(date -u +%Y%m%d-%H%M%S) \
./tpu/citation_v4/submit_citation_sft.sh
```

Read the desired SFT `state_path` from its uploaded
`client-output/<sft-run-id>/checkpoints.jsonl`. Pass that URI as
`TINKER_INITIAL_STATE_PATH`; it initializes RL step 0 without importing the
SFT optimizer-step counter.

For a cross-workload handoff, do not restore an unsanitized database captured
from a live SFT process: it can contain pending SFT futures. Run
`prepare_tinker_rl_seed.py` next to the source database, upload the compact
database plus the selected learner archive as a separate RL seed prefix, and
use that prefix as `RESTORE_RESULT_PREFIX`. The utility retains only the source
model creation record and selected completed training checkpoint, marks the
source model unloaded, removes stale futures/sampling sessions, and vacuums the
database. A completed, cleanly stopped SFT database can also use this path to
avoid transferring historical future payloads.

### Run the head client

`run_citation_rl_head.sh` maintains the TPU SSH tunnel, launches or follows a
gpu-test retriever chain, and runs the unmodified Tinker client API:

```bash
TINKER_TPU_NAME=hk4638-<rl-run-id>_1 \
TINKER_INITIAL_STATE_PATH='tinker://<sft-model-id>/weights/final_006962' \
MODEL_KEY=qwen35_9b \
SWEEP_BP=40 \
SWEEP_N=30 \
SWEEP_LR=3e-6 \
MAX_STEPS=100 \
SAVE_EVERY=10 \
EVAL_EVERY=10 \
./tpu/citation_v4/run_citation_rl_head.sh
```

For a functional smoke, use `SWEEP_BP=2`, `SWEEP_N=4`, `MAX_STEPS=1`,
`SAVE_EVERY=1`, `EVAL_EVERY=0`, and
`TINKER_AGENT_MAX_PROMPT_LENGTH=12000`. Production keeps the 45K trajectory
budget. A 4K cap is useful only for testing zero-reward overlength handling;
it is too short to expect a representative citation reward.

Use the learner-only gate before paying for a full rollout batch. It runs two
nonzero CISPO/TIS updates on Qwen3.5-9B, restores the second client from the
first saved state, and samples from the post-update adapter:

```bash
uv run --isolated --extra tinker python \
  tpu/citation_v4/tinker_qwen35_rl_learner_smoke.py \
  --base-url http://127.0.0.1:<tpu-api-tunnel-port> \
  --tokenizer-path /scratch/gpfs/ZHUANGL/hk4638/huggingface/easydel/qwen35_9b \
  --output /scratch/gpfs/ZHUANGL/hk4638/tinker_outputs/citation_prediction_v4/learner-smoke.json
```

### Verified runtime state (2026-07-11)

- The real-data SFT canary passed at sequence lengths 2,999, 8,000, 31,996,
  45,022, and 59,916 tokens and wrote a final state archive.
- The full 6,962-example conservative SFT is active as Jobman job `000278`;
  it saved `examples_002000` at exactly 2,000 examples and continued training.
  The 391.8 MB state archive was mirrored to the same-zone GCS result prefix;
  the 4,000/6,000/final triggers remain enabled.
- A live raw-protocol citation batch issued `<search>` actions, received
  `/retrieve` HTTP 200 responses, and treated malformed function-call syntax
  and 4K overlength trajectories as zero reward without crashing. It started
  from the untouched base model, so that batch had no final citations and is
  not counted as a nonzero-reward acceptance pass.
- On a clean server, the learner-only Qwen3.5-9B gate passed with pre-clip
  gradient norms `37.5` and `43.0` before and after restoring into a fresh
  training client. Post-update sampling returned four tokens with four finite
  logprobs. Learner, sampler, resumed learner, and validation artifacts are
  mirrored under
  `citation-rl-qwen35-9b-clean-smoke-20260711-110520/results`.
- A server that had accumulated multiple 9B model runtimes crashed its eSurge
  background process during a later sampler refresh. Keep one training client
  per production run; the reusable smoke samples before creating its temporary
  resumed client so its lifecycle matches that constraint.
- The SFT-initialized raw citation smoke restored
  `interim_examples_001022_smoke`, ran `Bp=2, n=4`, and completed one
  12K-context CISPO/TIS update. Six trajectories finished with citations and
  two overlength trajectories were correctly zero-masked. Mean/min/max recall
  reward was `0.05729 / 0 / 0.16667`; advantage standard deviation was
  `0.86601`; entropy was `0.41685`; pre-clip grad norm was `1.25781`;
  and sample/train time was `1831.45 / 371.99` seconds before batched eSurge
  generation was enabled.
- The smoke's final learner and sampler states are
  `tinker://model_2528c961/weights/final` and
  `tinker://model_2528c961/final`. Both archives and the consistent SQLite
  state are durable under
  `citation-rl-qwen35-9b-sft1022-clean-20260711-120845/results`.
- A second live smoke exercised true eSurge batching: all eight initial sample
  requests were queued together, then a surviving trajectory completed six
  search turns against the real top-50 retriever/top-5 reranker. Rollout and
  train time were `2502.73` and `406.44` seconds. This stochastic batch had
  zero recall and therefore zero gradient norm, but finite entropy (`1.35912`)
  and sampling/training KL (`-0.00438` / `0.00431`). Its final learner and
  sampler are `tinker://model_242e52a3/weights/final` and
  `tinker://model_242e52a3/final`; archives, SQLite state, checkpoint metadata,
  and runtime task config are under
  `citrl9b-batchsmoke-0711-0925-server/results`.
- The batch exposed a policy-dependent straggler cost: three interim-SFT
  trajectories reached the 12K context cap, so the eight-way call did not
  return shorter members until the longest member completed. This is not an
  inference deadlock, but production throughput must be measured again using
  the later SFT checkpoint before scaling the rollout count.

For the first production RL run, wait for an SFT state checkpoint, launch the
server with `RESTORE_RESULT_PREFIX` pointing to that SFT run, and pass the
matching `state_path` as `TINKER_INITIAL_STATE_PATH`. Do not start production
RL from the untouched EasyDeL base checkpoint.

## Acceptance Gates

Before scaling a new image or checkpoint, require all of the following:

1. Raw citation trajectories perform a real retriever call and receive a
   user-role `<information>` observation.
2. Rewards and prompt-group advantages are finite; at least one smoke batch
   has nonzero advantage variance.
3. EasyDeL reports a finite, nonzero pre-clip gradient norm.
4. A learner-state and sampler checkpoint are written and mirrored to GCS.
5. Loading that state continues at the next RL step and post-update sampling
   succeeds.
