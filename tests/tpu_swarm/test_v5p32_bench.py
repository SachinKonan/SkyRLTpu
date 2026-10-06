import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "tpu/swarm/bench"
YAMLS = REPO / "tpu/swarm/examples/v5p32-bench"
MODELS = ("qwen35", "gemma4", "muse")


def _yaml_files():
    return sorted(p for p in YAMLS.glob("*.yaml"))


def test_generator_emits_every_model_setting_and_capacity_probe(tmp_path):
    subprocess.run([sys.executable, str(YAMLS / "gen_bench_yamls.py"), "--out-dir", str(tmp_path),
                    "--max-num-seqs", "qwen35=320,gemma4=384,muse=192"], check=True, capture_output=True)
    names = sorted(p.stem for p in tmp_path.glob("*.yaml"))
    assert len(names) == 15
    for model in MODELS:
        for setting in ("S1", "S2", "S3", "S4"):
            assert f"bench-{model}-{setting}" in names
        assert f"capacity-{model}" in names
    # MAX/2 resolves per model; engines x TP = 4 chips on every host
    for path in tmp_path.glob("*.yaml"):
        doc = yaml.safe_load(path.read_text())
        env = doc["envs"]
        engines = int(env["BENCH_ENGINES_PER_HOST"])
        assert engines in (1, 2)
        assert env["BENCH_GPU_UTIL"] == "0.92"
        assert env["TPUSWARM_BUNDLE_ID"] == f"bench-{env['BENCH_MODEL']}-{path.stem.split('-')[-1] if path.stem.startswith('bench-') else 'capacity'}"
        assert doc["resources"]["accelerators"] == "tpu-v5p-32"
        assert doc["resources"]["job_recovery"]["max_restarts_on_errors"] == 1
        assert 'run_v5p32_bench.sh"' in doc["run"]
    qwen_s4 = yaml.safe_load((tmp_path / "bench-qwen35-S4.yaml").read_text())["envs"]
    assert qwen_s4["BENCH_MAX_NUM_SEQS"] == "160" and qwen_s4["BENCH_ENGINES_PER_HOST"] == "2"
    assert yaml.safe_load((tmp_path / "capacity-muse.yaml").read_text())["envs"]["BENCH_MODE"] == "capacity"


def test_checked_in_yamls_are_consistent():
    files = _yaml_files()
    assert len(files) == 15
    for path in files:
        doc = yaml.safe_load(path.read_text())
        env = doc["envs"]
        assert env["BENCH_NAME"] == path.stem == doc["name"]
        assert env["BENCH_MODEL"] in MODELS
        assert env["BENCH_STATES_URL"].startswith("gs://sk7524-tinker-tpu-us-east5/v5p-bench/")
        assert env["BENCH_RESULT_GCS_PREFIX"].startswith("gs://sk7524-tinker-tpu-us-east5/v5p-bench-results")


def test_runner_uses_rank0_as_driver_and_three_engine_hosts():
    source = (BENCH / "run_v5p32_bench.sh").read_text()
    assert '[ "$node_count" -ne 4 ]' in source
    assert 'if [ "$SKYPILOT_NODE_RANK" != "0" ]; then' in source
    assert "VLLM_WORKERS=1,2,3" in source
    assert "engine_hosts=(\"${ordered[@]:1}\")" in source
    assert "TP_SIZE=$(( 4 / BENCH_ENGINES_PER_HOST ))" in source
    # no trainer, no tinker
    assert "start_colocated_vllm_tinker" not in source
    assert "skyrl.tinker" not in source
    # per-model blocks mirror cell_worker.sh
    assert "TPU_BACKEND=jax; UNSET_PLUGINS=1; SKIP_PRECOMPILE=1; REQ_TIMEOUT=1800" in source
    assert "hf-cache-qwen35-v1" in source and "hf-cache-gemma4" in source
    assert "vllm-xla-cache-bench-" in source
    # engine facts + result publication
    assert "Maximum concurrency for" in source and "KV cache size" in source
    assert 'gcloud storage cp "$HOME/bench/${BENCH_NAME}.json" "$BENCH_RESULT_GCS_PREFIX/${BENCH_NAME}.json"' in source
    assert 'tmux kill-session -t "=$s"' in source  # exact tmux targets (prefix-match bug)


def test_bench_script_supports_multi_engine_percentiles_and_capacity_mode():
    source = (BENCH / "realistic_bench.py").read_text()
    assert '"--bases"' in source
    assert "endpoints[gi % len(endpoints)]" in source  # round-robin groups over engines
    assert "def percentiles(" in source and '"p95"' in source
    assert "sequence_wall_seconds" in source
    assert "def scrape_metrics(" in source and "vllm:prefix_cache_hits_total" in source
    assert "class GaugeSampler" in source and "vllm:num_requests_waiting" in source
    assert 'choices=["realistic", "capacity"]' in source
    assert "steady_decode_tok_s" in source and "ignore_eos=True" in source


def test_bundle_lists_the_bench_runner():
    source = (REPO / "tpu/swarm/build_skyrl_bundle.sh").read_text()
    assert "  tpu/swarm/bench/run_v5p32_bench.sh\n" in source
    assert "  tpu/swarm/bench/realistic_bench.py\n" in source


def test_bench_runner_supplies_start_vllm_generation_inputs():
    # start_vllm_tpu.sh reads REMOTE_SKYRL_DIR at generation time (tpu-inference
    # fork overlay path); the cells get it from start_colocated_vllm_tinker.sh.
    # Calling start_vllm_tpu.sh directly without it died before any engine
    # started (bench jobs 201/203, 2026-09-05).
    source = (REPO / "tpu/swarm/bench/run_v5p32_bench.sh").read_text()
    call = source.index('bash "$REPO/tpu/start_vllm_tpu.sh"')
    block = source[max(0, call - 2500):call]
    assert 'REMOTE_SKYRL_DIR="$REPO"' in block
    assert 'VLLM_TPU_PROCESS_PORT=8476' in block


def test_bench_engines_are_independent_single_host_processes():
    # Without explicit single-host TPU bounds, libtpu discovers the 4-host pod
    # from VM metadata and every engine tries to join one slice coordinated by
    # host 1 ("Failed to establish SliceBuilder grpc channel to ...:8471") —
    # jobs 208/209, 2026-09-05. The cells pass exactly these values.
    source = (REPO / "tpu/swarm/bench/run_v5p32_bench.sh").read_text()
    call = source.index('bash "$REPO/tpu/start_vllm_tpu.sh"')
    block = source[max(0, call - 2500):call]
    assert "VLLM_TPU_PROCESS_BOUNDS=1,1,1" in block
    assert "VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1" in block
    assert 'VLLM_TPU_PROCESS_ADDRESSES=""' in block


def test_bench_runner_reconciles_reused_worker_caches_on_every_rank():
    # A reused pool worker may hold another model's 55 GB HF snapshot; the
    # cells drop it via reconcile_v5p32_worker.sh before bring-up, and the
    # bench must do the same, on every rank, before the rank-0 gate.
    source = (REPO / "tpu/swarm/bench/run_v5p32_bench.sh").read_text()
    reconcile = source.index('bash "$REPO/tpu/swarm/reconcile_v5p32_worker.sh"')
    gate = source.index('if [ "$SKYPILOT_NODE_RANK" != "0" ]; then')
    assert reconcile < gate
    assert 'gemma4) reconcile_cell="g-bench"' in source
    assert 'muse)   reconcile_cell="m-bench"' in source
    assert (REPO / "tpu/swarm/reconcile_v5p32_worker.sh").exists()


def test_bench_runner_publishes_and_drops_its_xla_caches_on_exit():
    # The local bench XLA caches (5-6 GB per model/TP on every engine host) are
    # what tipped reused pool workers to 0 bytes free for the training cells
    # (2026-09-05). GCS is the durable copy (start_vllm_tpu.sh restores it), so
    # every exit path must publish then delete the local copy.
    source = (REPO / "tpu/swarm/bench/run_v5p32_bench.sh").read_text()
    assert "trap bench_teardown EXIT" in source
    fn = source[source.index("publish_and_drop_xla_caches() {"):]
    fn = fn[: fn.index("\n}\n")]
    assert "gcs_rsync.sh -r '$XLA_LOCAL' '$XLA_GCS'" in fn
    assert "rm -rf -- '$XLA_LOCAL'" in fn
    assert fn.index("gcs_rsync.sh") < fn.index("rm -rf")
    teardown = source[source.index("bench_teardown() {"):]
    teardown = teardown[: teardown.index("\n}\n")]
    assert "stop_engines" in teardown and "publish_and_drop_xla_caches" in teardown
    assert 'BENCH_KEEP_ENGINES" != "1"' in teardown
