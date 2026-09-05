from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import yaml


def _load(name: str) -> dict:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "tpu/swarm/examples" / name
    return yaml.safe_load(path.read_text())


def test_v4_64_pool_is_separate_and_fixed_at_seven_workers():
    config = _load("v4-64-qwen35-grpo-erdos-pool.yaml")

    assert config["pool"]["workers"] == 7
    assert config["pool"]["min_workers"] == 7
    assert config["pool"]["max_workers"] == 7
    assert config["resources"]["accelerators"] == "tpu-v4-64"
    assert config["resources"]["zone"] == "us-central2-b"
    assert "run_qwen35_v4_64_grpo.sh" in config["setup"]
    assert "jax==0.11.1" in config["setup"]
    assert "libtpu==0.0.46" in config["setup"]
    assert config["setup"].count("UV_NO_CONFIG=1 uv") == 2
    assert 'test -r "$staging/tpu/probe_topology.py"' in config["setup"]
    assert "TPUSWARM_SETUP_RETRY_SECONDS:-60" in config["setup"]
    assert "retaining the TPU and retrying" in config["setup"]


def test_v4_64_grpo_contract_has_valid_mesh_and_grader_isolation():
    config = _load("v4-64-qwen35-grpo-erdos.yaml")
    env = config["envs"]

    assert config["name"] == "qwen35-v4-64-grpo-erdos-005"
    assert env["TINKER_API_KEY"] == "tml-local-skyrl-no-auth"
    assert env["TINKER_BASE_URL"] == "http://127.0.0.1:8000"
    assert env["EXTRA_TTD_ENV"] == "TINKER_API_KEY=tml-local-skyrl-no-auth"
    assert config["resources"]["accelerators"] == "tpu-v4-64"
    assert env["CELL"] == "grpo-n"
    assert env["TTD_ENV"] == "erdos_min_overlap"
    assert env["TTD_ADV_ESTIMATOR"] == "mean_baseline"
    assert env["V4_64_AUTO_TOPOLOGY"] == "1"
    assert "tp4-fsdp4" in env["TPUSWARM_BUNDLE_ID"]
    assert "TRAIN_WORKERS" not in env
    assert "VLLM_WORKERS" not in env
    assert "gcloud storage cp" in config["run"]
    assert "jax==0.11.1" in config["run"]
    assert config["run"].count("UV_NO_CONFIG=1 uv") == 2
    assert "reconcile_v4_64_role_caches.sh" in config["run"]

    tp = int(env["TRAIN_TP_SIZE"])
    fsdp = int(env["TRAIN_FSDP_SIZE"])
    rows = int(env["TUNIX_ROW_SHARD"])
    uniform = int(env["TUNIX_UNIFORM_SEQ_LEN"])
    budget = int(env["TUNIX_TRAIN_TOKEN_BUDGET"])
    assert tp * fsdp == 16
    assert rows == fsdp
    assert budget == rows * uniform

    assert env["VLLM_TP_SIZE"] == "4"
    assert env["VLLM_ENGINES_PER_HOST"] == "1"
    assert env["VLLM_RAY_EXECUTOR"] == "0"
    assert env["VLLM_MAX_NUM_SEQS"] == "8"
    assert json.loads(env["VLLM_LIMIT_MM_PER_PROMPT"]) == {"image": 0, "video": 0}
    assert env["GRADER_RAY_PORT"] == "6379"
    assert env["EVAL_TIMEOUT"] == "1100"
    assert env["EXTERNAL_INFERENCE_TIMEOUT_SEC"] == "21600"
    assert env["SKYRL_EXTERNAL_WATCHDOG_INFLIGHT_SEC"] == "0"
    assert env["SKYRL_EXTERNAL_WATCHDOG_ABANDON_SEC"] == "28800"
    assert env["SKYRL_EXTERNAL_WATCHDOG_STALE_SEC"] == "30"
    assert env["SKYRL_EXTERNAL_WATCHDOG_MAX_REDISPATCH"] == "4"

    worker = (Path(__file__).resolve().parents[2] / "tpu/jobman/cell_worker.sh").read_text()
    qwen_arm = re.search(
        r'pick_tiles\(\) \{.*?^    \*\)\n(?P<body>.*?)^  esac$',
        worker,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert qwen_arm is not None
    assert '\\"attention\\": \\"autoselected\\"' in qwen_arm.group("body")
    assert '\\"use_tokamax_splash\\": true' in qwen_arm.group("body")
    assert '\\"remat_policy\\": \\"full\\"' in qwen_arm.group("body")
    assert '\\"base_num_kv_heads\\": 8' in qwen_arm.group("body")

    server = (Path(__file__).resolve().parents[2] / "tpu/vllm_tpu_server.py").read_text()
    assert '.skyrl-latest-lora' in server
    assert 'latest_marker.read_text().strip()' in server


def test_v4_64_tp8_fsdp2_grpo_contract():
    config = _load("v4-64-qwen35-grpo-erdos-tp8-fsdp2.yaml")
    env = config["envs"]

    assert config["name"] == "qwen35-v4-64-grpo-erdos-tp8-fsdp2-005"
    assert int(env["TRAIN_TP_SIZE"]) == 8
    assert int(env["TRAIN_FSDP_SIZE"]) == 2
    assert "tp8-fsdp2" in env["TPUSWARM_BUNDLE_ID"]
    assert int(env["TUNIX_ROW_SHARD"]) == 2
    assert int(env["TUNIX_TRAIN_TOKEN_BUDGET"]) == (
        int(env["TUNIX_ROW_SHARD"]) * int(env["TUNIX_UNIFORM_SEQ_LEN"])
    )
    assert env["TUNIX_JAX_CACHE_GCS"].endswith(
        "/jax-compile-cache-v4-qwen35-tp8-fsdp2-r32-s22528-b45056-v1"
    )
    assert env["GCS_RUN"].endswith(
        "/skyrl-runs/v4-64-qwen35-grpo-erdos-tp8-fsdp2-005"
    )
    assert env["VLLM_SKIP_JAX_PRECOMPILE"] == "0"
    assert json.loads(env["VLLM_LIMIT_MM_PER_PROMPT"]) == {"image": 0, "video": 0}
    assert env["EXTERNAL_INFERENCE_TIMEOUT_SEC"] == "21600"
    assert env["SKYRL_EXTERNAL_WATCHDOG_INFLIGHT_SEC"] == "0"
    assert env["SKYRL_EXTERNAL_WATCHDOG_ABANDON_SEC"] == "28800"
    assert env["SKYRL_EXTERNAL_WATCHDOG_STALE_SEC"] == "30"
    assert env["SKYRL_EXTERNAL_WATCHDOG_MAX_REDISPATCH"] == "4"
    assert config["run"].count("UV_NO_CONFIG=1 uv") == 2


def test_cell_launcher_passes_external_inference_timeout():
    repo = Path(__file__).resolve().parents[2]
    launcher = (repo / "tpu/start_colocated_vllm_tinker.sh").read_text()

    assert 'EXTERNAL_INFERENCE_TIMEOUT_SEC="${EXTERNAL_INFERENCE_TIMEOUT_SEC:-7200}"' in launcher
    assert '--external-inference-timeout-sec "${EXTERNAL_INFERENCE_TIMEOUT_SEC}"' in launcher


def test_vllm_bundle_identity_is_propagated_and_checked():
    repo = Path(__file__).resolve().parents[2]
    worker = (repo / "tpu/jobman/cell_worker.sh").read_text()
    launcher = (repo / "tpu/start_vllm_tpu.sh").read_text()

    assert "vLLM bundle mismatch" in worker
    assert "TPUSWARM_BUNDLE_ID=" in launcher


def test_cell_launcher_passes_local_tinker_key_via_env():
    repo = Path(__file__).resolve().parents[2]
    launcher = (repo / "tpu/launch_cell.sh").read_text()

    assert "cd $CLIENT_ROOT && \\\n  env \\" in launcher
    assert "TINKER_API_KEY=${TINKER_API_KEY:-tml-local-skyrl-no-auth}" in launcher
    assert "TINKER_BASE_URL=${TINKER_BASE_URL:-http://127.0.0.1:8000}" in launcher
    assert 'CLIENT_ROOT=$(readlink -f "${CLIENT_ROOT:-$HOME/ttd-client}")' in launcher


def test_v4_64_topology_selector_uses_rank_zero_physical_row():
    repo = Path(__file__).resolve().parents[2]
    selector = repo / "tpu/swarm/select_v4_64_topology.py"
    rank_coords = {
        0: [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        1: [[0, 2, 2], [1, 2, 2], [0, 3, 2], [1, 3, 2]],
        2: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        3: [[0, 0, 3], [1, 0, 3], [0, 1, 3], [1, 1, 3]],
        4: [[0, 2, 1], [1, 2, 1], [0, 3, 1], [1, 3, 1]],
        5: [[0, 2, 0], [1, 2, 0], [0, 3, 0], [1, 3, 0]],
        6: [[0, 2, 3], [1, 2, 3], [0, 3, 3], [1, 3, 3]],
        7: [[0, 0, 2], [1, 0, 2], [0, 1, 2], [1, 1, 2]],
    }
    payload = "".join(
        json.dumps({"process_id": rank, "coords": coords}) + "\n"
        for rank, coords in rank_coords.items()
    )

    result = subprocess.run(
        [str(selector)], input=payload, text=True, capture_output=True, check=True
    )

    assert result.stdout.splitlines() == [
        "TRAIN_WORKERS=0,7,3,2",
        "VLLM_WORKERS=1,4,5,6",
    ]


def test_hf_pruner_removes_weights_but_preserves_tokenizer(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    model_dir = tmp_path / "models--Qwen--Qwen3.5-27B"
    blobs = model_dir / "blobs"
    snapshots = model_dir / "snapshots" / "revision"
    trees = model_dir / "trees"
    blobs.mkdir(parents=True)
    snapshots.mkdir(parents=True)
    trees.mkdir(parents=True)

    weight_blob = blobs / "weight-blob"
    tokenizer_blob = blobs / "tokenizer-blob"
    weight_blob.write_bytes(b"weights")
    tokenizer_blob.write_bytes(b"tokenizer")
    (snapshots / "model-00001-of-00001.safetensors").symlink_to(
        os.path.relpath(weight_blob, snapshots)
    )
    (snapshots / "tokenizer.json").symlink_to(
        os.path.relpath(tokenizer_blob, snapshots)
    )
    manifest = {
        "files": {
            "model-00001-of-00001.safetensors": {
                "lfs_sha256": weight_blob.name,
                "size": weight_blob.stat().st_size,
            },
            "tokenizer.json": {
                "blob_id": tokenizer_blob.name,
                "size": tokenizer_blob.stat().st_size,
            },
        }
    }
    (trees / "revision.json").write_text(json.dumps(manifest))

    subprocess.run(
        [str(repo / "tpu/swarm/prune_hf_weight_cache.py"), str(model_dir)],
        check=True,
    )

    assert not weight_blob.exists()
    assert not (snapshots / "model-00001-of-00001.safetensors").exists()
    assert tokenizer_blob.exists()
    assert (snapshots / "tokenizer.json").exists()
    assert (trees / "revision.json").exists()


def test_hf_metadata_stager_excludes_safetensors(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    revision = "abc123"
    (source / "refs").mkdir(parents=True)
    (source / "trees").mkdir()
    (source / "blobs").mkdir()
    (source / "refs/main").write_text(f"{revision}\n")
    (source / "blobs/tokenizer-blob").write_bytes(b"tokenizer")
    (source / "blobs/weight-blob").write_bytes(b"weights")
    manifest = {
        "files": {
            "tokenizer.json": {"blob_id": "tokenizer-blob", "size": 9},
            "model-00001-of-00001.safetensors": {
                "lfs_sha256": "weight-blob",
                "size": 7,
            },
        }
    }
    (source / f"trees/{revision}.json").write_text(json.dumps(manifest))

    subprocess.run(
        [
            str(repo / "tpu/swarm/stage_hf_metadata_cache.py"),
            str(source),
            str(destination),
        ],
        check=True,
    )

    tokenizer = destination / f"snapshots/{revision}/tokenizer.json"
    assert (destination / "refs/main").read_text() == revision
    assert tokenizer.is_symlink()
    assert tokenizer.read_bytes() == b"tokenizer"
    assert not (destination / "blobs/weight-blob").exists()
    assert not (destination / f"snapshots/{revision}/model-00001-of-00001.safetensors").exists()


def test_v4_64_launcher_reconciles_roles_before_cell_worker():
    repo = Path(__file__).resolve().parents[2]
    wrapper = (repo / "tpu/swarm/run_qwen35_v4_64_grpo.sh").read_text()

    reconcile = wrapper.index("reconcile_v4_64_role_caches.sh")
    bringup = wrapper.index("tpu/jobman/cell_worker.sh")
    monitor = wrapper.index("tpu/jobman/cell_monitor.sh")
    assert reconcile < bringup < monitor
    assert "V4_64_TOPOLOGY_FINGERPRINT" in wrapper
    assert "reusing cached v4-64 topology" in wrapper
    assert 'REPO=$(readlink -f "${SKYRL_REPO_DIR:-$PWD}")' in wrapper
    assert 'export TPUSWARM_BUNDLE_ID=' in wrapper

    worker = (repo / "tpu/jobman/cell_worker.sh").read_text()
    assert "trainer bundle mismatch" in worker
    assert "import jax.scipy.linalg" in worker

    monitor = (repo / "tpu/jobman/cell_monitor.sh").read_text()
    assert 'OWNER_TOKEN="$RUN:${TPUSWARM_BUNDLE_ID:-unversioned}:${SKYPILOT_INTERNAL_JOB_ID:-standalone}"' in monitor
    assert "full engine readiness check failed" in monitor
    assert 'engines_healthy 1' in monitor

    colocated = (repo / "tpu/start_colocated_vllm_tinker.sh").read_text()
    cleanup = colocated.index("Remove an old trainer before the potentially long vLLM startup")
    start_vllm = colocated.index('if [[ "$START_VLLM" == "1" ]]; then')
    assert cleanup < start_vllm

    worker = (repo / "tpu/jobman/cell_worker.sh").read_text()
    assert "launcher_rc=$bringup_rc" in worker
    assert "|| bringup_rc=$?" in worker

    host_reconcile = (repo / "tpu/swarm/reconcile_v4_64_host_role.sh").read_text()
    assert "http://127.0.0.1:8001/v1/models" in host_reconcile
    assert "[g]cloud\\.py storage cp" in host_reconcile
    assert "*_.gstmp" in host_reconcile


def test_v4_64_host_reconcile_uses_runtime_compatible_awk(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name, body in {
        "ps": "#!/bin/sh\nexit 0\n",
        "tmux": "#!/bin/sh\nexit 0\n",
        "python3": "#!/bin/sh\nexit 0\n",
    }.items():
        executable = fake_bin / name
        executable.write_text(body)
        executable.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "V4_64_HOST_ROLE": "trainer",
            "SKYRL_REPO_DIR": str(repo),
            "HF_MODEL_CACHE_DIR": str(tmp_path / "hf"),
            "HF_MODEL_CACHE_GCS": str(tmp_path / "gcs"),
            "MAXTEXT_MODEL_CACHE_DIR": str(tmp_path / "orbax"),
        }
    )
    result = subprocess.run(
        [str(repo / "tpu/swarm/reconcile_v4_64_host_role.sh")],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "role=trainer" in result.stdout


def test_v4_64_tasks_use_checkpoint_durable_bundle():
    repo = Path(__file__).resolve().parents[2]
    task_paths = (
        repo / "tpu/swarm/examples/v4-64-qwen35-grpo-erdos.yaml",
        repo / "tpu/swarm/examples/v4-64-qwen35-grpo-erdos-tp8-fsdp2.yaml",
    )

    for task_path in task_paths:
        source = task_path.read_text()
        assert "tpuswarm-skyrl-v4-mixed-v35.tar.gz" in source
        assert "TPUSWARM_BUNDLE_ID: v35-" in source
        assert 'VLLM_INPLACE_RESTART_LIMIT: "2"' in source
        assert (
            "SKYRL_CKPT_GCS: "
            "gs://sk7524-tinker-tpu-us-central2/skyrl-checkpoints"
        ) in source
        for required in (
            "tpu/gcs_rsync.sh",
            "tpu/launch_cell.sh",
            "tpu/jobman/cell_sync.sh",
            "tpu/jobman/cell_monitor.sh",
        ):
            assert f'test -r "$staging/{required}"' in source
