import os
import re
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


def executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


def test_failed_orbax_copy_is_cleaned_before_retry_and_success_is_reused(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    cache = tmp_path / "orbax"
    attempts = tmp_path / "attempts"
    executable(commands / "gsutil", "#!/bin/sh\nprintf '5000\\n'\n")
    executable(commands / "sleep", "#!/bin/sh\nexit 0\n")
    executable(
        commands / "gcloud",
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path
if '--help' in sys.argv:
    sys.exit(0)
assert sys.argv[1:4] == ['storage', 'rsync', '-r'], sys.argv
assert os.environ['CLOUDSDK_STORAGE_THREAD_COUNT'] == '2'
assert os.environ['CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD'] == '0'
attempts = Path(os.environ['TEST_ATTEMPTS'])
count = int(attempts.read_text()) + 1 if attempts.exists() else 1
attempts.write_text(str(count))
destination = Path(sys.argv[-1])
assert not list(destination.iterdir()), 'failed contents survived into retry'
(destination / 'data').write_bytes(b'x' * 5000)
# Leave a normal-sized final file but report a failed checksum/transfer.
sys.exit(1 if count == 1 else 0)
""",
    )
    env = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "JOBMAN_WORKER_ID": "0",
        "TRAIN_WORKERS": "0",
        "TUNIX_MAXTEXT_MODEL_NAME": "target",
        "TUNIX_MAXTEXT_CKPT_CACHE": str(cache),
        "TUNIX_MAXTEXT_CKPT_CACHE_GCS": "gs://test/checkpoints",
        "CKPT_MARGIN_GB": "0",
        "CKPT_PREP_LOG": str(tmp_path / "errors.log"),
        "TEST_ATTEMPTS": str(attempts),
        "CACHE_DOWNLOAD_THREADS": "2",
        "CACHE_DOWNLOAD_SLICED_THRESHOLD": "0",
    }
    command = ["bash", str(REPO / "tpu/jobman/ensure_orbax_ckpt.sh")]
    first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert first.returncode == 0, first.stderr
    assert "copy_rc=1" in first.stdout
    assert attempts.read_text() == "2"
    assert (cache / "target" / ".tpuswarm-complete").is_file()
    data = cache / "target" / "data"
    before = data.stat().st_mtime_ns
    second = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert second.returncode == 0, second.stderr
    assert "already complete" in second.stdout
    assert attempts.read_text() == "2"
    assert data.stat().st_mtime_ns == before


@pytest.mark.parametrize("failed_worker", ["", "1"])
def test_vllm_readiness_polls_all_hosts_concurrently_and_joins(tmp_path, failed_worker):
    source = (REPO / "tpu/start_colocated_vllm_tinker.sh").read_text()
    function = re.search(r"^wait_for_vllm\(\) \{.*?^\}", source, re.M | re.S).group()
    # A serial implementation deadlocks worker 0 waiting for worker 1's probe.
    script = """set -euo pipefail
vllm_workers=(0 1)
VLLM_ENGINES_PER_HOST=1
VLLM_PORT=8001
wait_from_worker() {
  touch "$TEST_ROOT/started-$1"
  for i in $(seq 1 100); do
    if [[ -f "$TEST_ROOT/started-0" && -f "$TEST_ROOT/started-1" ]]; then
      touch "$TEST_ROOT/done-$1"
      [[ "$1" != "$TEST_FAILED_WORKER" ]]
      return
    fi
    sleep 0.01
  done
  return 99
}
""" + function + "\nwait_for_vllm\n"
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "TEST_ROOT": str(tmp_path), "TEST_FAILED_WORKER": failed_worker},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == (1 if failed_worker else 0), result.stderr
    assert (tmp_path / "done-0").exists()
    assert (tmp_path / "done-1").exists()


def test_trainer_provisioning_precedes_final_vllm_readiness_barrier():
    source = (REPO / "tpu/start_colocated_vllm_tinker.sh").read_text()
    assert source.index("SKYRL_TRAIN_SETUP_ONLY=1 bash") < source.index("\n  wait_for_vllm\n")


@pytest.mark.parametrize("overrides", [
    {},
    {
        "SKYRL_EXTERNAL_WATCHDOG_ENABLED": "1",
        "SKYRL_EXTERNAL_WATCHDOG_POLL_SEC": "30",
        "SKYRL_EXTERNAL_WATCHDOG_STALE_SEC": "30",
        "SKYRL_EXTERNAL_WATCHDOG_INFLIGHT_SEC": "0",
        "SKYRL_EXTERNAL_WATCHDOG_MAX_REDISPATCH": "4",
        "SKYRL_EXTERNAL_WATCHDOG_ABANDON_SEC": "28800",
    },
    {"SKYRL_EXTERNAL_WATCHDOG_ENABLED": "0", "SKYRL_EXTERNAL_WATCHDOG_INFLIGHT_SEC": "7200"},
])
def test_generated_trainer_exports_watchdog_settings_to_a_clean_shell(overrides):
    source = (REPO / "tpu/start_colocated_vllm_tinker.sh").read_text()
    function = re.search(r"^render_external_watchdog_env\(\) \{.*?^\}", source, re.M | re.S).group()
    template = source.split('cat > "$api_script" <<EOF\n', 1)[1].split('\nEOF', 1)[0]
    assert "$(render_external_watchdog_env)" in template
    assert template.index("$(render_external_watchdog_env)") < template.index('-m skyrl.tinker.api')
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("SKYRL_EXTERNAL_WATCHDOG_")}
    rendered = subprocess.run(
        ["bash", "-c", function + "\nrender_external_watchdog_env"],
        env={**clean_env, **overrides}, capture_output=True, text=True, check=True,
    )
    # Model the fresh SSH/tmux environment: only the rendered startup script
    # can carry the settings across, not inheritance from the launch process.
    started = subprocess.run(
        ["bash", "-c", rendered.stdout + "\nenv -0"],
        env=clean_env, capture_output=True, text=True, check=True,
    )
    actual = dict(item.split("=", 1) for item in started.stdout.split("\0")
                  if item.startswith("SKYRL_EXTERNAL_WATCHDOG_"))
    assert actual == overrides
