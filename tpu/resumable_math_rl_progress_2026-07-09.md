# Resumable MathRL Progress - 2026-07-09

## Scope

This tracks the Qwen3.5-9B MathRL autoresume work in the `skyrltpu-resumable`
worktree. The working colocated `p1tpufork` TPU session was not touched.

## Code Delta

Pushed commits:

- `SkyRLTpu`: `abac0277` - `Update jobman autoresume checkpoint path`
- `third_party/jobman`: `c4b7395` - `Use shared checkpoints for SkyRL autoresume`

Uncommitted follow-up test coverage:

- `third_party/tinker-cookbook/tinker_cookbook/rl/train_resume_test.py`
  - Verifies `checkpoints.jsonl` with `batch=N` makes `main()` restore via
    `create_training_client_from_state_with_optimizer_async(...)`.
  - Verifies the training loop receives `start_batch=N`, so the completed batch
    is not replayed.
  - Verifies no checkpoint starts from `start_batch=0` with a fresh LoRA
    training client.
- `tests/tinker/test_preempt_endpoint.py`
  - Verifies the simulated preemption route is registered at `/api/v1/preempt`
    and `/preempt`.
  - Verifies the endpoint is disabled unless `SKYRL_ENABLE_PREEMPT_ENDPOINT`
    is truthy.
  - Verifies remote clients are blocked unless `SKYRL_PREEMPT_ALLOW_REMOTE`
    is explicitly enabled.
  - Verifies a local confirmed request schedules `SIGTERM` for the API process.

Effective Jobman changes:

- `scripts/skyrl_math_rl_autoresume_worker.sh`
  - Changed the default `SKYRL_LOCAL_CHECKPOINTS` from `/tmp/skyrl-checkpoints/${JOBMAN_RUN_ID}` to `${HOME}/gcs/skyrl-checkpoints/${JOBMAN_RUN_ID}`.
  - Reason: multi-host Orbax checkpoint saves require a filesystem visible from all TPU workers.
- `scripts/skyrl_resumable_restore.sh`
  - Uses the shared GCSFuse checkpoint path by default.
  - Skips checkpoint copy when the local and GCS checkpoint paths are the same.
- `scripts/skyrl_resumable_sync.sh`
  - Uses the shared GCSFuse checkpoint path by default.
  - Skips checkpoint copy when the local and GCS checkpoint paths are the same.
- `config/skyrl_math_rl_qwen35_9b_v5p16_autoresume_east5a.yaml`
  - Clears the one-off `SKYRLTPU_BUNDLE_PATH` in the reusable config so future launches pull the pushed branch/submodules.
  - Enables `SKYRL_ENABLE_PREEMPT_ENDPOINT` while leaving
    `SKYRL_PREEMPT_ALLOW_REMOTE=false` for local-only live resume testing.

New simulated preemption path:

- `skyrl/tinker/api.py`
  - Adds opt-in POST routes `/api/v1/preempt` and `/preempt`.
  - Requires JSON body `{"confirm": "preempt"}`.
  - Rejects non-local clients by default.
  - Schedules `SIGTERM` for the API process after a short delay so the caller can
    receive a response before `skyrl-tinker` exits.

Live-test command, once the running TPU has code containing this endpoint:

```bash
curl -fsS -X POST http://localhost:8000/api/v1/preempt \
  -H 'Content-Type: application/json' \
  -d '{"confirm":"preempt"}'
```

Expected outcome: the API process exits, Jobman's monitor sees `skyrl-tinker`
die, the sync hook runs, the controller loop reacquires/restarts, and MathRL
resumes from the latest `checkpoints.jsonl` batch.

The live `000001` job used a one-off uploaded code bundle. That bundle object
was overwritten at `2026-07-09T15:33:00Z` with parent commit `85aeada4`, so the
already-running Jobman controller can keep the same snapshot config and its next
retry will unpack code containing the simulated preemption endpoint.

## Live Run State

Active MathRL job:

- Jobman ID: `000001`
- TPU: `v5p-16`
- Zone: `us-east5-a`
- Run ID: `math-qwen35-9b-180step-100605b06fb4df06`
- GCS run path: `gs://sk7524-tinker-tpu-us-east5/jobman-runs/math-qwen35-9b-180step-100605b06fb4df06/`

Confirmed behavior after restart:

- Tinker API launched with `--checkpoints-base /home/sk7524_princeton_edu/gcs/skyrl-checkpoints/math-qwen35-9b-180step-100605b06fb4df06`.
- JAX backend initialized as `process_id=0/1`, `num_processes=2`, `total devices=8`.
- MathRL client initialized and began `180` batches.
- Batch 0 completed.
- Batch 3 completed.
- Latest durable checkpoint observed: batch 4.

Fresh batch-0 metrics:

- `time/total`: `1429.671s`
- `time/sampling`: `489.443s`
- `time/train_step`: `572.281s`
- `time/save_checkpoint`: `4.106s`
- `time/save_checkpoint_async`: `27.261s`
- `env/all/correct`: `0.176758`
- `test/env/all/correct`: `0.142`

## Resume Boundary

The first durable checkpoint exists:

```json
{"name": "000001", "batch": 1, "state_path": "tinker://model_be40affe/weights/000001", "rolling": true}
```

The latest observed durable checkpoint is:

```json
{"name": "000004", "batch": 4, "state_path": "tinker://model_be40affe/weights/000004", "rolling": true}
```

`latest_good.json` also reports:

```json
{
  "latest_checkpoint": {
    "batch": 4,
    "name": "000004",
    "rolling": true,
    "state_path": "tinker://model_be40affe/weights/000004"
  },
  "local_checkpoints": "/home/sk7524_princeton_edu/gcs/skyrl-checkpoints/math-qwen35-9b-180step-100605b06fb4df06",
  "gcs_checkpoints": "/home/sk7524_princeton_edu/gcs/skyrl-checkpoints/math-qwen35-9b-180step-100605b06fb4df06"
}
```

This confirms the old `/tmp/skyrl-checkpoints/...` multi-host checkpoint issue
was bypassed for the current attempt. If the TPU is preempted now, resume should
start from the checkpoint recorded in `checkpoints.jsonl`.

## Local Verification

- `uv run --with pytest --with pytest-asyncio --with pygments python -m pytest tinker_cookbook/rl/train_resume_test.py -q`: passed, 2 tests.
- `uv run --extra tinker --extra dev python -m pytest tests/tinker/test_preempt_endpoint.py -q`: passed, 5 tests.
- `uv run --with ruff ruff check skyrl/tinker/api.py tests/tinker/test_preempt_endpoint.py`: passed.
- `python3 -m compileall -q skyrl/tinker/api.py`: passed.
- `bash -n third_party/jobman/scripts/skyrl_math_rl_autoresume_worker.sh`: passed.

## Useful Checks

```bash
jobman list
tail -n 80 third_party/jobman/jobs/sk7524/000001/logs/job.log
gsutil cat gs://sk7524-tinker-tpu-us-east5/jobman-runs/math-qwen35-9b-180step-100605b06fb4df06/latest_good.json
gsutil cat gs://sk7524-tinker-tpu-us-east5/jobman-runs/math-qwen35-9b-180step-100605b06fb4df06/math_rl/checkpoints.jsonl
gsutil cat gs://sk7524-tinker-tpu-us-east5/jobman-runs/math-qwen35-9b-180step-100605b06fb4df06/math_rl/metrics.jsonl | tail -n 2
cd third_party/tinker-cookbook
uv run --with pytest --with pytest-asyncio --with pygments python -m pytest tinker_cookbook/rl/train_resume_test.py -q
cd /scratch/gpfs/ZHUANGL/sk7524/skyrltpu-resumable
uv run --extra tinker --extra dev python -m pytest tests/tinker/test_preempt_endpoint.py -q
bash -n third_party/jobman/scripts/skyrl_math_rl_autoresume_worker.sh
```
