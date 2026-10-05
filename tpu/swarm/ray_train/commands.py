"""Pure launch contracts shared by host actors and their regression tests."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .config import Config


def trainer_environment(config: Config, root: Path, run: Path, train_ips, process_id):
    t, p = config.trainer, config.ports
    env = dict(os.environ)
    for key in ("JAX_COORDINATOR_ADDRESS", "TPU_MULTIHOST_BACKEND", "TPU_MULTIPROCESS_DP"):
        env.pop(key, None)
    env.update(
        JAX_PLATFORMS="tpu,cpu", HF_HOME=str(root / "ram/hf"), HF_HUB_OFFLINE="1",
        JAX_COMPILATION_CACHE_DIR=str(root / "ram/compile"),
        JAX_ENABLE_COMPILATION_CACHE="true",
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
        TUNIX_UNIFORM_SEQ_LEN=str(t.sequence_length), TUNIX_ROW_SHARD=str(t.fsdp),
        SKYRL_TRAIN_PROCESS_ID=str(process_id), SKYRL_TINKER_ENGINE_DIRECT_PYTHON="1",
        TPU_PROCESS_BOUNDS=t.process_bounds, TPU_CHIPS_PER_PROCESS_BOUNDS=t.chip_bounds,
        TPU_PROCESS_ADDRESSES=",".join(f"{ip}:{p.trainer_tpu}" for ip in train_ips),
        TPU_PROCESS_PORT=str(p.trainer_tpu), CLOUD_TPU_TASK_ID=str(process_id),
        TPU_VISIBLE_CHIPS="0,1,2,3", TINKER_API_KEY="tml-local-skyrl-no-auth",
        SKYRL_DATABASE_URL="sqlite:///" + str(run / "tinker.db"),
        TPUSWARM_BUNDLE_ID=config.base_bundle_sha256,
        SKYRL_EXTERNAL_WATCHDOG_INFLIGHT_SEC="0", SKYRL_EXTERNAL_WATCHDOG_ABANDON_SEC="28800",
        SKYRL_EXTERNAL_WATCHDOG_STALE_SEC="30", SKYRL_EXTERNAL_WATCHDOG_MAX_REDISPATCH="4",
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env.update(config.trainer_env)
    return env


def trainer_backend_config(config, root, head, train_ips):
    t, p = config.trainer, config.ports
    kwargs = dict(ici_tensor_parallelism=t.tp, ici_fsdp_parallelism=t.fsdp,
                  ici_context_parallelism=1, remat_policy=t.remat, num_vocab_tiling=64,
                  attention="autoselected", use_tokamax_splash=True,
                  jax_cache_dir=str(root / "ram/compile"))
    if t.tp == 8:
        kwargs.update(allow_split_physical_axes=True, override_model_config=True,
                      base_num_kv_heads=t.logical_kv_heads)
    backend = dict(
        model_source="maxtext", maxtext_model_name=t.maxtext_model,
        maxtext_max_target_length=t.sequence_length, train_token_budget=t.token_budget,
        flce_tile_size=t.flce_tile, max_lora_rank=t.effective_max_lora_rank,
        train_micro_batch_size=1, sample_max_num_sequences=256,
        param_dtype="bfloat16", free_base_state_after_template=True,
        maxtext_ckpt_cache_dir=str(root / "ram/orbax"), maxtext_kwargs=kwargs,
        inference_backend="vllm", vllm_base_url=f"http://{head}:{p.inference}",
        vllm_model_name=config.model, vllm_lora_base_dir=str(root / "runs" / config.run_id / "loras"),
        vllm_lora_upload_endpoint="/skyrl/v1/upload_lora_adapter",
        vllm_client_side_round_robin=False, vllm_request_timeout_sec=config.inference.request_timeout,
        vllm_lora_load_retries=3, vllm_lora_load_retry_sleep_sec=10,
        checkpoint_mirror_gcs=config.run_gcs + "/checkpoints")
    if len(train_ips) > 1:
        backend.update(coordinator_address=f"{train_ips[0]}:{p.trainer_jax}", num_processes=len(train_ips))
    return backend


def trainer_command(config, root, source, head, train_ips, process_id):
    python = str(root / "envs/trainer/bin/python")
    p = config.ports
    if process_id:
        return [python, "-m", "skyrl.backends.rpc", "--backend", "tunix",
                "--coordinator-address", f"{train_ips[0]}:{p.trainer_jax}",
                "--num-processes", str(len(train_ips)), "--process-id", str(process_id)]
    return [python, "-m", "skyrl.tinker.api", "--base-model", config.model,
            "--host", "0.0.0.0", "--port", str(p.trainer), "--backend", "tunix",
            "--session-timeout-sec", "1800",
            # The synchronous mirror requires a local staging file. It uploads
            # and verifies this file in run_gcs before acknowledging the save.
            "--checkpoints-base", str(root / "runs" / config.run_id / "checkpoints"),
            "--external-inference-url", f"http://{head}:{p.inference}",
            "--external-inference-lora-base", str(root / "runs" / config.run_id / "loras"),
            "--external-inference-timeout-sec", str(config.inference.request_timeout),
            "--backend-config", json.dumps(trainer_backend_config(config, root, head, train_ips))]


def inference_environment(config, root, run):
    env = dict(os.environ)
    for key in ("JAX_COORDINATOR_ADDRESS", "TPU_MULTIHOST_BACKEND", "TPU_MULTIPROCESS_DP"):
        env.pop(key, None)
    env.update(
        JAX_PLATFORMS="tpu,cpu", HF_HOME=str(root / "ram/hf"), HF_HUB_OFFLINE="1",
        TPU_PROCESS_BOUNDS="1,1,1", TPU_CHIPS_PER_PROCESS_BOUNDS="2,2,1",
        TPU_PROCESS_ADDRESSES=f"localhost:{config.ports.inference_tpu}",
        TPU_PROCESS_PORT=str(config.ports.inference_tpu), CLOUD_TPU_TASK_ID="0",
        TPU_VISIBLE_CHIPS="0,1,2,3", TPU_BACKEND_TYPE="torchax", MODEL_IMPL_TYPE="vllm",
        VLLM_ALLOW_RUNTIME_LORA_UPDATING="True", VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_XLA_CACHE_PATH=str(root / "ram/compile"), JAX_COMPILATION_CACHE_DIR=str(root / "ram/compile"),
        JAX_ENABLE_COMPILATION_CACHE="true",
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0", JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
        SKIP_JAX_PRECOMPILE="1", USE_BATCHED_RPA_KERNEL="1", USE_JAX_RAGGED_CONV1D="1",
        VLLM_PLUGINS="lora_filesystem_resolver", VLLM_LORA_RESOLVER_CACHE_DIR=str(run / "loras"),
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    return env


def inference_command(config, root, source, snapshot, run):
    v = config.inference
    return [str(root / "envs/serving/bin/python"), str(source / "tpu/vllm_tpu_server.py"),
            str(snapshot), "--served-model-name", config.model, "--skyrl-lora-dir", str(run / "loras"),
            "--host", "0.0.0.0", "--port", str(config.ports.engine),
            "--tensor-parallel-size", str(v.tp), "--max-model-len", str(v.max_model_length),
            "--max-num-seqs", str(v.max_sequences), "--max-num-batched-tokens", str(v.chunk_tokens),
            "--enable-chunked-prefill", "--gpu-memory-utilization", str(v.memory_utilization),
            "--limit-mm-per-prompt", '{"image":0,"video":0}',
            "--enable-lora", "--max-loras", str(v.max_loras), "--max-lora-rank", str(v.max_lora_rank)]


def client_environment(config, root, head):
    config.validate()
    env = dict(os.environ)
    t = config.trainer
    defaults = dict(
        TINKER_API_KEY="tml-local-skyrl-no-auth", TINKER_BASE_URL=f"http://{head}:{config.ports.trainer}",
        TTD_M0_BASE_URL=f"http://{head}:{config.ports.trainer}",
        HF_HOME=str(root / "ram/hf"), HF_HUB_OFFLINE="1", JAX_PLATFORMS="cpu",
        TTD_RUN_DIR=str(root / "runs" / config.run_id / "client"), EXPERIMENT_NAME=config.run_id,
        TTD_ENV="erdos_min_overlap", TTD_ENSEMBLE_MODELS=f"{config.model}:qwen3:qwen",
        TTD_ALLOW_SINGLE_MEMBER="1", TTD_QWEN_TWO_PHASE="1", TTD_DISABLE_WANDB_TABLES="1",
        TTD_CROSS_WEIGHT="0", TTD_ADV_ESTIMATOR="mean_baseline", TTD_ELITE_SLOTS="2",
        TTD_REJECT_TRUNCATED="1", TTD_RESTART_RATIO="0", TTD_KL_MEASURE_EVERY="0",
        TTD_RESUME_STRICT="1", TTD_SAMPLING_PROGRESS_TIMEOUT="0", TTD_MAX_CONSEC_TRAIN_ERR="1",
        TTD_EVAL_BACKEND="ray", TTD_RAY_PAYLOAD="1", TTD_LEAGUE_PIPELINE="1",
        RAY_ADDRESS=f"{head}:{config.ports.ray}", RAY_NAMESPACE=config.run_id,
        NUM_CPUS_PER_TASK="1", GROUPS_PER_BATCH="16", GROUP_SIZE="32", NUM_EPOCHS="15",
        LEARNING_RATE="1.5e-4", LORA_RANK=str(t.lora_rank), KL_PENALTY_COEF="0", TEMPERATURE="1.0",
        EVAL_TIMEOUT="1100", SAVE_EVERY="1", WANDB_MODE="offline", WANDB_PROJECT="tpu-tinker-exps",
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    defaults.update(config.client_sampling_environment())
    defaults.update(config.client_env)
    env.update(defaults)
    return env
