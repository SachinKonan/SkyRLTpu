"""Pure launch contracts shared by host actors and their regression tests.

The defaults mirror the legacy v5p-32 cell launcher for the configured model
preset (see config.PRESETS); tests/tpu_swarm/test_ray_train_commands.py pins
the qwen command line and environment against tpu/start_vllm_tpu.sh,
tpu/start_colocated_vllm_tinker.sh, tpu/jobman/cell_worker.sh and
tpu/launch_cell.sh.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .config import Config


def _flag(value):
    return "1" if value else "0"


def inference_urls(config: Config, head, inference_ips=None):
    """Where the trainer sends completions and adapters.

    direct: the legacy contract -- every engine URL, comma-separated; the
    backend round-robins requests and pushes each adapter to every engine.
    ingress: the single Ray Serve endpoint on the head.
    """
    if config.inference.routing == "direct" and inference_ips:
        return ",".join(f"http://{ip}:{config.ports.engine}" for ip in inference_ips)
    return f"http://{head}:{config.ports.inference}"


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
        # Legacy cell_worker.sh forwards both to every trainer.
        TUNIX_SEQ_BUCKETS=t.seq_buckets, TUNIX_MINIMAL_FB_OUTPUT=_flag(t.minimal_fb_output),
        SKYRL_TRAIN_PROCESS_ID=str(process_id), SKYRL_TINKER_ENGINE_DIRECT_PYTHON="1",
        TPU_PROCESS_BOUNDS=t.process_bounds, TPU_CHIPS_PER_PROCESS_BOUNDS=t.chip_bounds,
        TPU_PROCESS_ADDRESSES=",".join(f"{ip}:{p.trainer_tpu}" for ip in train_ips),
        TPU_PROCESS_PORT=str(p.trainer_tpu), CLOUD_TPU_TASK_ID=str(process_id),
        TPU_VISIBLE_CHIPS="0,1,2,3", TINKER_API_KEY="tml-local-skyrl-no-auth",
        SKYRL_DATABASE_URL="sqlite:///" + str(run / "tinker.db"),
        TPUSWARM_BUNDLE_ID=config.base_bundle_sha256,
        # No SKYRL_EXTERNAL_WATCHDOG_* overrides: the legacy v5p-32 cell runs the
        # dispatch defaults (stale 300 s, inflight 3600 s, 2 redispatches,
        # abandon 7200 s); the 0/28800/30/4 set belonged to the v4-64 launcher.
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env.update(config.trainer_env)
    return env


def maxtext_kwargs(config: Config, root: Path):
    """cell_worker.sh pick_tiles + the FSDP/TP injection of start_colocated."""
    t = config.trainer
    kwargs = dict(ici_tensor_parallelism=t.tp, ici_fsdp_parallelism=t.fsdp,
                  ici_context_parallelism=1, remat_policy=t.remat, num_vocab_tiling=t.num_vocab_tiling,
                  attention="autoselected")
    if t.tokamax_splash:
        kwargs["use_tokamax_splash"] = True
    if t.tp == 8:
        kwargs.update(allow_split_physical_axes=True, override_model_config=True,
                      base_num_kv_heads=t.logical_kv_heads)
    kwargs.update(t.maxtext_kwargs)
    kwargs["jax_cache_dir"] = str(root / "ram/compile")
    return kwargs


def trainer_backend_config(config, root, head, train_ips, inference_ips=None):
    t, p = config.trainer, config.ports
    direct = config.inference.routing == "direct" and bool(inference_ips)
    backend = dict(
        model_source="maxtext", maxtext_model_name=t.maxtext_model,
        maxtext_max_target_length=t.effective_max_target_length, train_token_budget=t.token_budget,
        flce_tile_size=t.flce_tile, max_lora_rank=t.effective_max_lora_rank,
        train_micro_batch_size=1, sample_max_num_sequences=256,
        param_dtype="bfloat16", free_base_state_after_template=t.free_base_state,
        maxtext_ckpt_cache_dir=str(root / "ram/orbax"), maxtext_kwargs=maxtext_kwargs(config, root),
        inference_backend="vllm", vllm_base_url=inference_urls(config, head, inference_ips),
        vllm_model_name=config.model, vllm_lora_base_dir=str(root / "runs" / config.run_id / "loras"),
        vllm_lora_load_endpoint="/v1/load_lora_adapter",
        vllm_lora_unload_endpoint="/v1/unload_lora_adapter",
        vllm_lora_upload_endpoint="/skyrl/v1/upload_lora_adapter",
        vllm_client_side_round_robin=direct,
        vllm_route_by_prompt_prefix=False,
        vllm_max_concurrent_requests=t.max_concurrent_requests,
        vllm_request_timeout_sec=t.request_timeout,
        vllm_lora_load_retries=t.lora_load_retries, vllm_lora_load_retry_sleep_sec=t.lora_load_retry_sleep,
        checkpoint_mirror_gcs=config.run_gcs + "/checkpoints")
    if len(train_ips) > 1:
        backend.update(coordinator_address=f"{train_ips[0]}:{p.trainer_jax}", num_processes=len(train_ips))
    return backend


def trainer_command(config, root, source, head, train_ips, process_id, inference_ips=None):
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
            "--external-inference-url", inference_urls(config, head, inference_ips),
            "--external-inference-lora-base", str(root / "runs" / config.run_id / "loras"),
            "--external-inference-timeout-sec", str(config.inference.request_timeout),
            "--backend-config", json.dumps(trainer_backend_config(config, root, head, train_ips, inference_ips))]


def inference_environment(config, root, run, head=None, group=None):
    """Engine process environment. group = the engine's host IPs (pairs run
    pipeline-parallel over vLLM's Ray executor on the executor's own Ray
    cluster at head:ports.ray; the worker for the second host runs under the
    serving venv via the job runtime_env set in tpu/vllm_tpu_server.py)."""
    v = config.inference
    env = dict(os.environ)
    for key in ("JAX_COORDINATOR_ADDRESS", "TPU_MULTIHOST_BACKEND", "TPU_MULTIPROCESS_DP"):
        env.pop(key, None)
    pair = bool(group) and len(group) > 1
    env.update(
        JAX_PLATFORMS="tpu,cpu", HF_HOME=str(root / "ram/hf"), HF_HUB_OFFLINE="1",
        TPU_PROCESS_BOUNDS="1,1,1", TPU_CHIPS_PER_PROCESS_BOUNDS="2,2,1",
        TPU_PROCESS_ADDRESSES=f"localhost:{config.ports.inference_tpu}",
        TPU_PROCESS_PORT=str(config.ports.inference_tpu), CLOUD_TPU_TASK_ID="0",
        TPU_VISIBLE_CHIPS="0,1,2,3", TPU_BACKEND_TYPE=v.tpu_backend, MODEL_IMPL_TYPE=v.model_impl,
        VLLM_ALLOW_RUNTIME_LORA_UPDATING="True",
        VLLM_XLA_CACHE_PATH=str(root / "ram/compile"), JAX_COMPILATION_CACHE_DIR=str(root / "ram/compile"),
        JAX_ENABLE_COMPILATION_CACHE="true",
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0", JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
        # Legacy start_vllm_tpu.sh exports (per-model values from the preset).
        VLLM_USE_RAY_EXECUTOR="0",
        SKIP_JAX_PRECOMPILE=_flag(v.skip_precompile), USE_BATCHED_RPA_KERNEL=_flag(v.batched_rpa_kernel),
        USE_JAX_RAGGED_CONV1D=_flag(v.ragged_conv1d),
        VLLM_PLUGINS="lora_filesystem_resolver", VLLM_LORA_RESOLVER_CACHE_DIR=str(run / "loras"),
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env.update(v.engine_env)
    if pair:
        if not head:
            raise ValueError("pipeline-parallel engines need the executor Ray head address")
        # tpu-inference (multihost=ray) isolates each host as its own JAX
        # cluster and sets the TPU process variables itself; leave them unset
        # so they do not describe a single-host slice to a two-host engine.
        for key in ("TPU_PROCESS_BOUNDS", "TPU_CHIPS_PER_PROCESS_BOUNDS", "TPU_PROCESS_ADDRESSES",
                    "TPU_PROCESS_PORT", "CLOUD_TPU_TASK_ID", "TPU_VISIBLE_CHIPS"):
            env.pop(key, None)
        env.update(TPU_MULTIHOST_BACKEND="ray", VLLM_USE_RAY_EXECUTOR="1",
                   RAY_ADDRESS=f"{head}:{config.ports.ray}",
                   SKYRL_RAY_PLACEMENT_HOSTS=",".join(group),
                   # The server connects to Ray as a driver before vLLM starts
                   # its engine core; a forked core deadlocks (job 523).
                   VLLM_WORKER_MULTIPROC_METHOD="spawn")
    return env


def inference_command(config, root, source, snapshot, run, group=None):
    v = config.inference
    command = [str(root / "envs/serving/bin/python"), str(source / "tpu/vllm_tpu_server.py"),
               str(snapshot), "--served-model-name", config.model, "--skyrl-lora-dir", str(run / "loras"),
               "--host", "0.0.0.0", "--port", str(config.ports.engine),
               "--tensor-parallel-size", str(v.tp), "--max-model-len", str(v.max_model_length),
               "--max-num-seqs", str(v.max_sequences)]
    if v.prefix_caching:
        command.append("--enable-prefix-caching")
    command += ["--enable-lora", "--max-loras", str(v.max_loras), "--max-lora-rank", str(v.max_lora_rank),
                "--download-dir", str(root / "ram/hf/hub")]
    if v.limit_mm_per_prompt:
        command += ["--limit-mm-per-prompt", v.limit_mm_per_prompt]
    # Legacy VLLM_EXTRA_ARGS: batched tokens + memory utilization (+ model extras).
    command += ["--max-num-batched-tokens", str(v.chunk_tokens),
                "--gpu-memory-utilization", str(v.memory_utilization)]
    if v.chunked_prefill:
        command.append("--enable-chunked-prefill")
    if group and len(group) > 1:
        command += ["--pipeline-parallel-size", str(len(group)), "--distributed-executor-backend", "ray",
                    "--skyrl-ray-placement-hosts", ",".join(group)]
    command += list(v.extra_args)
    return command


def client_environment(config, root, head, inference_ips=None, api_host=None):
    """`head` is the executor Ray head; `api_host` is where the trainer API
    (trainer process 0) listens, which differs from the head whenever the
    trainer block does not start at Sky rank 0."""
    config.validate()
    env = dict(os.environ)
    t = config.trainer
    api_host = api_host or head
    defaults = dict(
        TINKER_API_KEY="tml-local-skyrl-no-auth", TINKER_BASE_URL=f"http://{api_host}:{config.ports.trainer}",
        TTD_M0_BASE_URL=f"http://{api_host}:{config.ports.trainer}",
        HF_HOME=str(root / "ram/hf"), HF_HUB_OFFLINE=_flag(config.client_hf_offline), JAX_PLATFORMS="cpu",
        TTD_RUN_DIR=str(root / "runs" / config.run_id / "client"), EXPERIMENT_NAME=config.run_id,
        TTD_ENV="erdos_min_overlap", TTD_PROBLEM_TYPE="", TTD_FCALGO_MAX_CASES="0",
        TTD_ENSEMBLE_MODELS=config.client_member_spec,
        TTD_ALLOW_SINGLE_MEMBER="1", TTD_QWEN_TWO_PHASE="1", TTD_DISABLE_WANDB_TABLES="1",
        TTD_CROSS_WEIGHT="0", TTD_ADV_ESTIMATOR="mean_baseline", TTD_ELITE_SLOTS="2",
        TTD_REJECT_TRUNCATED="1", TTD_RESTART_RATIO="0", TTD_KL_MEASURE_EVERY="0",
        TTD_RESUME_STRICT="1", TTD_SAMPLING_PROGRESS_TIMEOUT="0", TTD_MAX_CONSEC_TRAIN_ERR="1",
        TTD_EVAL_BACKEND="ray", TTD_RAY_PAYLOAD="1", TTD_LEAGUE_PIPELINE="1",
        RAY_ADDRESS=f"{head}:{config.ports.ray}", RAY_NAMESPACE=config.run_id,
        NUM_CPUS_PER_TASK="1", GROUPS_PER_BATCH="16", GROUP_SIZE="32", NUM_EPOCHS="15",
        LEARNING_RATE=config.client_learning_rate, LORA_RANK=str(t.lora_rank), KL_PENALTY_COEF="0",
        TEMPERATURE="1.0", EVAL_TIMEOUT="1100", SAVE_EVERY="1", WANDB_MODE="offline",
        WANDB_PROJECT="tpu-tinker-exps", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    defaults.update(config.client_sampling_environment())
    defaults.update(config.client_env)
    env.update(defaults)
    return env
