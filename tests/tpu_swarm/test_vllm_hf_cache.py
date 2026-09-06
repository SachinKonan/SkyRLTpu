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

    assert '${CACHE_DOWNLOAD_SLICED_THRESHOLD:-0}' in source
    assert '${CACHE_DOWNLOAD_PROCESSES:-1}' in source
    assert '${CACHE_DOWNLOAD_THREADS:-1}' in source
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
    # The lock must be released before the engine process starts.
    server_start = source.index('"\\${server_cmd[@]}"', lock_release)
    assert lock_release < server_start


def test_vllm_runner_preserves_restart_logs_and_exit_codes():
    source = (REPO / "tpu/start_vllm_tpu.sh").read_text()

    assert 'runner_history_dir="\\$HOME/skyrl-logs/vllm-history"' in source
    assert 'mv "\\$runner_log_path"' in source
    assert "NR > 8" in source
    assert "exit_code=%s" in source
    assert "server_rc=\\$?" in source
    assert 'exit "\\$server_rc"' in source


def test_hf_tree_manifest_materialization_is_best_effort():
    # gemma4's trees/<commit>.json references blobs never uploaded; a fatal
    # SystemExit killed every engine runner pre-log on a warm host (job 194).
    # Readiness is decided by hf_snapshot_ready, which runs right after.
    source = (REPO / "tpu/start_vllm_tpu.sh").read_text()
    call = source.index("    materialize_hf_tree_manifest \\\\\n      || echo")
    ready_check = source.index('if ! hf_snapshot_ready && [[ "\\${HF_HUB_OFFLINE}" == "1" ]]', call)
    assert call < ready_check
    assert "\n    materialize_hf_tree_manifest\n" not in source


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


def test_multi_engine_runner_defines_its_log_suffix_before_the_log_setup():
    # feca01c4 moved the runner's log/history setup to the top of the template,
    # ahead of the engine env block that defined engine_log_suffix. With set -u
    # the two-engines-per-host runner (muse) died on line 4 before any log
    # existed; every muse cell from bundle v5 on failed bring-up (jobs 241/254).
    source = (REPO / "tpu/start_vllm_tpu.sh").read_text()
    template = source.index('cat > "$runner_script" <<EOF')
    define = source.index('engine_log_suffix=""\nif [ "\\${VLLM_ENGINE_INDEX:-0}" != "0" ]', template)
    log_path = source.index('runner_log_path="\\$HOME/skyrl-logs/${runner_log_name}"', template)
    env_block = source.index("${engine_env_block}", template)
    assert template < define < log_path < env_block
    # the multi-engine log name really does depend on that variable
    assert "runner_log_name='vllm-tpu${engine_log_suffix}.log'" in source
