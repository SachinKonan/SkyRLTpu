# Running RL cells: Muse-Glimmer, Qwen3.5, Gemma-4 — deployment & efficiency runbook

Everything here was learned the hard way on the 2026-08 muse bring-up (~20 failed
attempts, five stacked bugs) and the qwen/gemma Stage-A/B campaigns. Numbers are
measured, not estimated; where something is a projection it says so.

## 1. What a cell is

One experiment arm = one spot **v5p-32** (4 hosts × 4 chips) driven by a jobman
controller on the login node:

- **worker 0** — tinker trainer (`skyrl.tinker.engine`, MaxText backend) + the
  RL client (`tpu/run_ttd_ensemble.py`) + Ray grading head
- **workers 1–3** — vLLM engines + Ray grading workers

Lifecycle (`jobman run <id>`, `loop: true`): request → prepare → restore →
lease → command (`cell_worker.sh`) → monitor (`cell_monitor.sh`) → sync
(`cell_sync.sh`) → completion_probe (`cell_probe.sh`). Preemption recycles the
QR and re-runs everything from scratch; the ONLY durable state is GCS (code
bundle, caches, run dir) and the controller process.

## 2. Quick start

```bash
# 1. Config: clone an existing one from tpu/jobman/configs/
#    (see §8 "cloning a config" — two real incidents came from doing this wrong)
# 2. Ship code: the bundle is the ONLY code path to the TPUs
SKYRL_ROOT=$PWD bash /n/fs/vision-mix/sk7524/SkyRLTpu/third_party/jobman/scripts/build_skyrl_code_bundle.sh \
  gs://sk7524-tinker-tpu-us-east5/code-bundles/stagea-league.tar.gz
# 3. Launch
jobman create tpu/jobman/configs/<cell>.yaml     # prints job id, starts the controller
# 4. Watch
tail -f third_party/jobman/jobs/sk7524/<id>/logs/job.log
gsutil cat gs://.../skyrl-runs/<RUN_DIR>/tinker_log/<EXP>/metrics.jsonl | tail -1
```

Cell prefix selects the model in `cell_worker.sh`: `g-*` → gemma-4-31B,
`m-*` → Muse-Glimmer-30B, anything else → qwen3.5-27B. The prefix is
load-bearing — everything in §3 hangs off it.

## 3. Per-model settings (all live in `tpu/jobman/cell_worker.sh` + `tpu/launch_cell.sh`)

| knob | qwen3.5-27B (default) | gemma-4-31B (`g-*`) | muse-30B (`m-*`) |
|---|---|---|---|
| serving impl | torch/vLLM | torch/vLLM | torch/vLLM (fork model) |
| `TPU_BACKEND_TYPE` | torchax | torchax | **jax** (mandatory, §4.1) |
| `VLLM_PLUGINS` | pinned to resolver | pinned | **UNSET** (mandatory, §4.2) |
| tpu-inference ref | `skyrl/v0.23.0-lora` | same | SHA `afe0cb9e` |
| engine layout | 1×TP=4/host (3 engines) | 1×TP=4 | **2×TP=2/host (6 engines)** |
| `max_num_seqs` | 128 | 128 | 64 |
| serve len | 19456 (staged, unmeasured) | 16384 | 22528 |
| client CTX / PHASE1 | 18432 / 13824 | 10240 / 6656 | 18432 / 13824 (qwen-matched) |
| train UNIFORM / BUDGET | 18432 / 73728 (4 seq) | 10240 / 40960 (4 seq) | 18432 / **73728 (4 seq)** |
| FLCE tile / vocab tiling | 512/64 (Erdős), 2048/8 (JSSP) | 1024/32 | 1024/32 |
| maxtext kwargs | nvt only | nvt only | + `remat_policy: full`, `ici_fsdp: 4` |
| precompile at boot | on | on | **skipped** (tracer bug, §4.3) |
| adapter-push retries / timeout | 3×2s / 300s | same | 20×30s / **1800s** |
| `free_base_state` | off (could be on) | off | **on** (§4.4) |
| two-phase completer | Qwen (`</think>`) | Gemma (`<channel|>`) | **Muse** (`to=user`, §4.5) |
| vocab | 152k | 256k | 202k |
| caches (GCS prefixes) | `vllm-xla-cache-qwen35-19k`, `jax-compile-cache-qwen35-18k` | `...gemma4-31b-16k`, `...gemma4-10k` | `vllm-xla-cache-mg-22k-tp2`, `jax-compile-cache-muse-22k` |

Batch shape (GRPO 16×32 = 512 rollouts/step), LoRA rank 32, temp 1.0 are common.
LR is per-arm via `LEARNING_RATE` in the jobman config env (default 4e-5).

## 4. Muse: the five things that MUST be right (each cost a live failure)

1. **`TPU_BACKEND_TYPE=jax`.** Under torchax, vLLM's registry has no muse and
   silently serves the generic transformers fallback — it boots, health-checks,
   even loads LoRA, then kills EngineCore on the FIRST generate
   (`NonConcreteBooleanIndexError`). Discriminator in the engine log:
   `LORA-WRAPPED-MODULES total=260` = real model; `721` = fallback.
2. **`VLLM_PLUGINS` unset.** It is an ALLOW-LIST: naming any plugin excludes the
   fork's `vllm.general_plugins` registration that injects
   `MuseGlimmerForCausalLM`. Unset loads everything, resolver included (its real
   on-switch is `VLLM_LORA_RESOLVER_CACHE_DIR`).
3. **`VLLM_SKIP_JAX_PRECOMPILE=1`** (boot-time capture leaks a tracer through
   torchax). Consequence: the FIRST LoRA load per engine compiles for minutes →
   needs the 1800s push timeout AND the client's `/v1/models` fallback
   (an upload is not failed until the server says the adapter is absent —
   we once burned 206 relaunches on lost ACKs while every push had landed).
4. **`free_base_state_after_template=1`.** `create_model` otherwise duplicates
   ~35 GiB of base weights (qwix wrap doesn't share arrays; `_init_lora_state`
   builds a second whole model). Measured: in_use 12.97→49.79 GiB without it,
   →14.31 GiB with it; the fb literally does not fit without this.
5. **`MuseTwoPhaseTokenCompleter`** (dispatch BEFORE the `TTD_QWEN_TWO_PHASE`
   branch, which is on for every cell). Muse's renderer strictly splits
   `to=self`/`to=user`; qwen's `</think>` cue lands the forced answer in the
   reasoning channel and the parser discards it — 344/512 rollouts scored
   "Invalid code" with the programs generated and thrown away.
   NOTE: the two-phase logic is DUPLICATED in `completers.py` (`__call__` and
   the coalesced `sample_group`); edit both or the edit silently doesn't apply.

## 5. Efficiency (measured)

- **fb is bound by a fixed per-microbatch cost** (FSDP all-gather of the 55.7 GB
  weights), not tokens: 1 datum 27.3 s, 4 datums 27.4 s steady-state. So
  **`BUDGET = 4×UNIFORM` is ~4× free training throughput**. Keep
  `remat_policy: full` — 27.4 s (none) vs 27.5 s (full): free in time, big HBM
  saving. Beware: single-call timings include ~8 s JIT; use `--repeat`.
- **Trainer JAX cache: MaxText writes `~/jax_cache`**, NOT the
  `JAX_COMPILATION_CACHE_DIR` you export (base.yml `jax_cache_dir` wins). We
  synced the wrong dir for weeks; every fresh VM re-paid a ~34 min fb compile.
  `cell_worker` now syncs `~/jax_cache` on a 3-min cadence (short spot windows
  lose whole compiles at 10 min).
- **vLLM XLA cache** is keyed by (model, max-model-len, TP): one GCS prefix per
  shape. Muse's ladder is 13 `jit_step_fun_impl` programs ≈ 1.76 GB; a warm boot
  skips ~60 min of first-burst compile.
- **Engine layout**: 2×TP=2 per host beats 1×TP=4 by ~1.66× capacity (measured
  on muse; also the fix for muse's 2 KV heads padding at TP=4). Qwen still runs
  1×TP=4 — switching it is the biggest unstaged qwen win, needs its own
  before/after and a fresh cache prefix.
- **Qwen serving knobs staged, unmeasured** (`62e12d06`): serve 19456 (client
  ctx is 18432; 22528 wasted 22% of KV envelope), batched-tokens 16384 (phase-2
  re-prefill ~17k), mem-util 0.90. Compare `sampling` vs the ~4.5 h baseline on
  first use.
- **Prefix-affinity routing: tried and REVERTED** (`e30d8a53`). Windowed hit
  rate 36% vs 49.6% baseline; sampling got slower. Load balance was fine — the
  affinity never took. Do not retry without a windowed
  `vllm:prefix_cache_{queries,hits}_total` measurement (cumulative log lines
  cannot show recent behaviour).
- Grading env has **numba** (`numba 0.67`, ~0.4 s JIT then 400× loops) — muse
  exploits it in 81% of its programs; qwen/gemma never do.

## 6. Diagnostics toolbox

- **`tpu/muse_glimmer/fb_mem_probe.py`** — standalone fb HBM/speed probe
  (~40 s backend init). Flags: `--length --budget --datums --repeat --remat
  --nvt --flce --rank --free-base --skip-fb`. MUST run under tmux with the
  trainer's TPU env or it hangs forever at zero CPU:
  `TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
  TPU_PROCESS_ADDRESSES=<w0int>:8477 CLOUD_TPU_TASK_ID=0` (copy from the
  generated `~/start_colocated_skyrl_api.sh`). Kill orphans + `rm
  /tmp/libtpu_lockfile` first — a dead probe still holds the TPU.
- **Trajectory archives**: `<run>/tinker_log/<exp>/member_<tag>/trajectories/
  <tag>_step_NNNNNN.jsonl.gz`, 512 rows/step, GCS-synced. `response_raw` holds
  reasoning+answer for every model; split post hoc with the family regex
  (`rosetta_stone`). Never compare raw `response` lengths across models.
- **Two-phase telemetry**: `two_phase outcome=natural|injected|continued|wall
  [unterminated] phase1=N answer=M` per rollout in the client log. CAUTION:
  partial-burst counts are ordering-biased (naturals finish first) — only read
  complete 512-bursts.
- **Adapter pushes**: `/v1/models` on each engine is ground truth.
- **ssh to cell hosts**: `ssh -i ~/.ssh/jobman_tpu_ed25519
  sk7524_princeton_edu@<external-ip>` (IPs change every recreation:
  `gcloud compute tpus tpu-vm describe <qr> --zone=us-east5-a`).

## 7. The three config layers (the #1 recurring trap)

A fix "shipped" is not a fix "running". Config is FROZEN at three layers:

1. **GCS bundle** — updated by `build_skyrl_code_bundle.sh`; reaches a host only
   at its next `prepare` (attempt recycle).
2. **jobman config snapshot** (`jobs/sk7524/<id>/config.yaml`) — env for hooks;
   edit it AND restart the controller for changes mid-job.
3. **Running processes** — the generated `~/start_colocated_skyrl_api.sh` and
   `~/run_vllm_tpu_server.sh` bake values at generation time; the tinker
   `--backend-config` JSON freezes them again at process start.

To force a change NOW: kill `cell` + `skyrl-tinker` tmux + `pkill -9 -f
skyrl.tinker.engine` (it survives tmux kills and holds the TPU) → the jobman
loop re-prepares (new bundle) and regenerates everything. To hand-patch
instead: edit the generated script on the host, then restart only the affected
process. Verify against the RUNNING process (`ps` args, `/proc/<pid>/environ`),
never against the file you edited.

To truly RERUN a cell from step 0: archive the run dir **both** locally and in
GCS (`gsutil mv gs://.../skyrl-runs/<RUN> ...-archived`) — `launch_cell.sh`
restores from GCS on every relaunch and will resurrect the old tree/weights.

## 8. Cloning a config for a new arm (two real incidents)

- Rename ALL identity fields: both `name:` fields, **`run_id_prefix`** (no
  `sk7524-` prefix, easy to miss — a stale one deadlocks the new arm on the old
  arm's local lease, silently, forever), `CELL`, `RUN_DIR_NAME`,
  `EXPERIMENT_NAME`.
- Verify with an **unfiltered** `diff old.yaml new.yaml` — filtering the diff
  through `grep -v <names>` once hid exactly the broken field.
- When inserting env lines by script, **assert on the substitution count**, not
  just preconditions — a quote-style mismatch (`'0'` vs `"0"`) once made an
  insert silently no-op and the "LR ablation" arm ran at the control LR.
- After launch, verify the value in the CLIENT process:
  `tr '\0' '\n' < /proc/<pid>/environ | grep LEARNING_RATE` — and pick the pid
  of `run_ttd_ensemble.py`, not a Ray worker from the same venv.

## 9. File map

| file | role |
|---|---|
| `tpu/jobman/gen_configs.sh` | generates the per-cell yaml grid (prepare hook incl. bundle unpack + `ensure_orbax_ckpt.sh`) |
| `tpu/jobman/configs/*.yaml` | one jobman config per arm |
| `tpu/jobman/cell_worker.sh` | per-model settings + orchestration (venv, caches, engines, Ray) |
| `tpu/jobman/cell_monitor.sh` / `cell_sync.sh` / `cell_probe.sh` | client relaunch loop / GCS sync / completion |
| `tpu/jobman/ensure_orbax_ckpt.sh` | pre-engine checkpoint restore (gcloud storage rsync, purge-on-torn) |
| `tpu/launch_cell.sh` | client env (CTX/PHASE1/LR/rank) + run-dir restore + client start |
| `tpu/start_colocated_vllm_tinker.sh` | trainer + engines bring-up; generates the frozen host scripts |
| `tpu/start_vllm_tpu.sh` | engine venv + runner generation (backend/plugins/caches baked here) |
| `tpu/vllm_tpu_server.py` | vLLM + `/skyrl/v1/upload_lora_adapter` (drains body on skip path) |
| `skyrl/backends/tunix_backend.py` | trainer backend: fb path, FLCE, `free_base_state_after_template` |
| `skyrl/backends/vllm_sampling.py` | push/sample client, `/v1/models` fallback, (disabled) prefix routing |
| `third_party/discover/ttt_discover/tinker_utils/completers.py` | two-phase completers (Qwen/Gemma/**Muse**) + outcome telemetry |
| `third_party/discover/ttt_discover/tinker_utils/renderers.py` | per-model renderers (muse = strict channel split) |
| `third_party/discover/ttt_discover/rl/ensemble.py` | RL loop, pipelined fb, trajectory archiving |
| `tpu/muse_glimmer/fb_mem_probe.py` | fb memory/speed probe (§6) |
| `tpu/muse_glimmer/{README,E2E,VLLM-IMPL,MAXTEXT}.md` | the muse port's own record |
