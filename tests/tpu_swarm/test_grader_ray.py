from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(textwrap.dedent(contents))
    path.chmod(0o755)


def test_grader_ray_uses_separate_ports_and_exact_membership(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    helper = repo / "tpu/jobman/grader_ray.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    state = tmp_path / "state"
    calls = tmp_path / "calls"

    _write_executable(
        fake_bin / "ray",
        """\
        #!/bin/sh
        printf '%s\\n' "$*" >> "$CALLS"
        if [ "$1" = health-check ]; then
          [ "$(cat "$STATE" 2>/dev/null)" = head ]
          exit
        fi
        if [ "$1" = start ]; then
          case " $* " in
            *" --head "*) printf 'head' > "$STATE" ;;
            *) printf 'worker' > "$STATE" ;;
          esac
        fi
        """,
    )
    _write_executable(
        fake_bin / "ps",
        """\
        #!/bin/sh
        if [ -s "$STATE" ]; then
          echo '101 /venv/ray/raylet/raylet --gcs-address=10.0.0.1:6379 --session-dir=/tmp/ray_tpuswarm_grader/session'
        fi
        """,
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RAY_BIN": str(fake_bin / "ray"),
            "GRADER_RAY_ADDRESS": "10.0.0.1:6379",
            "GRADER_RAY_NODE_IP": "10.0.0.1",
            "STATE": str(state),
            "CALLS": str(calls),
        }
    )
    subprocess.run([str(helper), "head"], env=env, check=True)
    head_call = calls.read_text()
    assert "--port=6379" in head_call
    assert "--temp-dir=/tmp/ray_tpuswarm_grader" in head_call
    assert "--ray-client-server-port=10002" in head_call
    assert "--min-worker-port=20000" in head_call
    assert "--max-worker-port=29999" in head_call
    assert "--runtime-env-agent-port" not in head_call
    assert "--metrics-export-port" not in head_call
    assert "--dashboard-agent-grpc-port" not in head_call

    state.unlink()
    subprocess.run([str(helper), "worker"], env=env, check=True)
    worker_call = calls.read_text()
    assert "--address=10.0.0.1:6379" in worker_call

    source = helper.read_text()
    assert 'pkill -f "ray/core"' not in source
    assert "pgrep -f '[r]ay/core'" not in source
    assert "--gcs-address=$GRADER_RAY_ADDRESS" in source
    assert 'address="--gcs-address=$GRADER_RAY_ADDRESS"' in source


def test_grader_ray_retries_failed_startup(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    helper = repo / "tpu/jobman/grader_ray.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    attempts = tmp_path / "attempts"

    _write_executable(
        fake_bin / "ray",
        """\
        #!/bin/sh
        if [ "$1" = health-check ]; then
          [ "$(cat "$ATTEMPTS" 2>/dev/null)" = 3 ]
          exit
        fi
        count=$(cat "$ATTEMPTS" 2>/dev/null || echo 0)
        count=$((count + 1))
        printf '%s' "$count" > "$ATTEMPTS"
        [ "$count" -eq 3 ]
        """,
    )
    _write_executable(
        fake_bin / "ps",
        """\
        #!/bin/sh
        if [ "$(cat "$ATTEMPTS" 2>/dev/null)" = 3 ]; then
          echo '101 /venv/ray/raylet/raylet --gcs-address=10.0.0.1:6379 --session-dir=/tmp/ray_tpuswarm_grader/session'
        fi
        """,
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RAY_BIN": str(fake_bin / "ray"),
            "GRADER_RAY_ADDRESS": "10.0.0.1:6379",
            "GRADER_RAY_NODE_IP": "10.0.0.1",
            "GRADER_RAY_LOG": str(tmp_path / "ray.log"),
            "ATTEMPTS": str(attempts),
        }
    )

    subprocess.run([str(helper), "head"], env=env, check=True)
    assert attempts.read_text() == "3"


def test_vllm_cleanup_never_stops_skypilot_ray():
    repo = Path(__file__).resolve().parents[2]
    source = (repo / "tpu/start_vllm_tpu.sh").read_text()
    helper = (repo / "tpu/vllm_ray.sh").read_text()

    assert "ray stop" not in source
    assert "ray stop" not in helper
    assert 'bash "\\$HOME/vllm_ray.sh" stop' in source
    assert "/tmp/ray_tpuswarm_vllm" in helper
    assert '"--gcs-address=$VLLM_RAY_ADDRESS"' in helper


def test_cell_worker_forwards_v4_runtime_overrides():
    repo = Path(__file__).resolve().parents[2]
    source = (repo / "tpu/jobman/cell_worker.sh").read_text()

    expected = {
        "SKIP_PRECOMPILE": "VLLM_SKIP_JAX_PRECOMPILE",
        "LORA_RETRIES": "VLLM_LORA_LOAD_RETRIES",
        "LORA_RETRY_SLEEP": "VLLM_LORA_LOAD_RETRY_SLEEP_SEC",
        "REQ_TIMEOUT": "VLLM_REQUEST_TIMEOUT_SEC",
        "HF_OFFLINE": "HF_HUB_OFFLINE",
        "FREE_BASE_STATE": "TUNIX_FREE_BASE_STATE",
    }
    for local_name, environment_name in expected.items():
        assert f'{local_name}="${{{environment_name}:-${local_name}}}"' in source


def test_vllm_launcher_requires_prefix_caching():
    repo = Path(__file__).resolve().parents[2]
    source = (repo / "tpu/start_vllm_tpu.sh").read_text()

    assert '--enable-prefix-caching \\\\' in source
    assert 'if [[ "$VLLM_EXTRA_ARGS" == *"--no-enable-prefix-caching"* ]]' in source
    assert "prefix caching is required" in source


def test_gemma_recipes_use_default_rpa_and_text_only_mm():
    repo = Path(__file__).resolve().parents[2]
    env_recipe = (repo / "tpu/runs/gemma4-31b.env").read_text()
    cell_worker = (repo / "tpu/jobman/cell_worker.sh").read_text()
    gemma_case = cell_worker.split("  g-*)", 1)[1].split("    ;;", 1)[0]

    assert "VLLM_USE_BATCHED_RPA_KERNEL=0" in env_recipe
    assert "VLLM_USE_JAX_RAGGED_CONV1D=0" in env_recipe
    assert cell_worker.count(
        'VLLM_USE_BATCHED_RPA_KERNEL="$BATCHED_RPA_KERNEL"'
    ) == 2
    assert cell_worker.count(
        'VLLM_USE_JAX_RAGGED_CONV1D="$JAX_RAGGED_CONV1D"'
    ) == 2
    for source in (env_recipe, gemma_case):
        assert "--disable-chunked-mm-input" in source
        assert "--no-enable-prefix-caching" not in source
    assert 'LIMIT_MM_PER_PROMPT=\'{"image":0,"audio":0,"video":0}\'' in gemma_case
    assert "--limit-mm-per-prompt" not in gemma_case
