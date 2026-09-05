from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]


def test_vllm_rejects_partial_qwen_hf_snapshot():
    source = (REPO / "tpu/start_vllm_tpu.sh").read_text()

    assert 'index_path = snapshot / "model.safetensors.index.json"' in source
    assert 'not (snapshot / name).is_file() for name in required' in source
    assert '("preprocessor_config.json", "video_preprocessor_config.json")' in source


def test_vllm_hf_cache_restore_uses_resumable_sequential_gcloud_copy():
    source = (REPO / "tpu/start_vllm_tpu.sh").read_text()

    assert "export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=0" in source
    assert "export CLOUDSDK_STORAGE_PROCESS_COUNT=1" in source
    assert "export CLOUDSDK_STORAGE_THREAD_COUNT=1" in source
    assert 'gcloud storage cp --recursive --no-clobber' in source
    assert 'gsutil -m -q rsync -r "${HF_CACHE_GCS}/${HF_MODEL_DIR}"' not in source


def test_vllm_hf_cache_restore_is_serialized_across_engines_on_a_host():
    # VLLM_ENGINES_PER_HOST>1 runners share one HF hub. Unlocked, two engines
    # raced the same 55 GB copy on a fresh v5p-32 pool host; the loser saw the
    # winner's partials, purged them, and exited before its log opened.
    source = (REPO / "tpu/start_vllm_tpu.sh").read_text()

    lock_open = source.index('exec 9>"${REMOTE_HF_HOME}/hub/.${HF_MODEL_DIR}.restore.lock"')
    lock_take = source.index("flock 9\n", lock_open)
    restore = source.index("gcloud storage cp --recursive --no-clobber", lock_take)
    lock_release = source.index("flock -u 9\nexec 9>&-", restore)
    assert lock_open < lock_take < restore < lock_release
    # The lock must be released before the engine process replaces this shell.
    assert lock_release < source.index("exec ", lock_release + len("flock -u 9\nexec 9>&-"))


def test_qwen_engines_default_to_the_complete_offline_hf_cache():
    # gs://…/hf-cache/models--Qwen--Qwen3.5-27B holds ~4 GB (one shard + metadata);
    # jobman engines fetched the rest from HuggingFace. Pool workers are offline,
    # so the runner refused the incomplete snapshot and every qwen engine died
    # before its log opened (job 190, 2026-09-05).
    source = (REPO / "tpu/jobman/cell_worker.sh").read_text()
    qwen = source.index("MODEL_NAME=Qwen/Qwen3.5-27B")
    block = source[qwen:qwen + 2500]
    assert 'HF_GCS="gs://sk7524-tinker-tpu-us-east5/hf-cache-qwen35-v1"' in block
    assert 'HF_GCS="gs://sk7524-tinker-tpu-us-east5/hf-cache"\n' not in block

    yaml_text = (REPO / "tpu/swarm/examples/v5p32-cells/meta-wt16-fresh-g0-qwen.yaml").read_text()
    assert "HF_CACHE_GCS: gs://sk7524-tinker-tpu-us-east5/hf-cache-qwen35-v1" in yaml_text


def test_tinker_readiness_ignores_stale_vllm_failure_log():
    source = (REPO / "tpu/start_colocated_vllm_tinker.sh").read_text()

    assert 'local inspect_vllm_log="${4:-0}"' in source
    assert "if [ '${inspect_vllm_log}' = '1' ]" in source
    assert '"$engine_label" 1' in source
    assert '"Tinker API" 1' not in source


def test_missing_no_ray_vllm_session_is_restarted_once():
    source = (REPO / "tpu/start_colocated_vllm_tinker.sh").read_text()

    assert "restart_attempted=0" in source
    assert "detached vLLM session is absent; retrying its bootstrap once" in source
    assert "VLLM_RELATIVE_WORKER_ID=0 VLLM_USE_RAY_EXECUTOR=0" in source


def test_v4_inference_smoke_fails_when_detached_server_exits():
    task_path = REPO / "tpu/swarm/examples/v4-32-qwen35-inference-smoke-32k.yaml"
    task = yaml.safe_load(task_path.read_text())

    assert "tmux has-session -t vllm-tpu" in task["run"]
    assert task["envs"]["TPUSWARM_SKYRL_BUNDLE_URL"].endswith("v17.tar.gz")


def test_v4_inference_smoke_clears_stale_trainer_without_stopping_ray():
    task_path = REPO / "tpu/swarm/examples/v4-32-qwen35-inference-smoke-32k.yaml"
    task = yaml.safe_load(task_path.read_text())
    run = task["run"]

    assert "tmux kill-session -t skyrl-tinker" in run
    assert "[s]kyrl\\.tinker\\.(api|engine)" in run
    assert "[s]tart_vllm_tpu\\.sh" in run
    assert "[s]tart_vllm_tpu_bootstrap\\.sh" in run
    assert "[g]sutil.*hf-cache-" in run
    assert "[g]cloud storage cp.*hf-cache-" in run
    assert 'rm -f -- "$HOME/skyrl-logs/vllm-tpu.log"' in run
    assert "ray stop" not in run
    assert "--gpu-memory-utilization 0.85" in task["envs"]["VLLM_EXTRA_ARGS"]


def test_vllm_passes_multimodal_limits_as_one_typed_argument():
    source = (REPO / "tpu/start_vllm_tpu.sh").read_text()

    assert "printf -v vllm_limit_mm_per_prompt_q '%q'" in source
    assert (
        r'limit_mm_args+=(--limit-mm-per-prompt "\${VLLM_LIMIT_MM_PER_PROMPT}")'
        in source
    )
    assert r'"\${limit_mm_args[@]}"' in source
