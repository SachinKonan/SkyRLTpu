"""Exercise cache cleanup with multiple stale writers, without signaling hosts."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("cache_owner,keep", [(None, True), ("target", True), ("foreign", False)])
def test_reconcile_stops_every_stale_writer_before_deleting_partials(tmp_path, cache_owner, keep):
    repo = Path(__file__).resolve().parents[2]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    hub = tmp_path / "hub" / "models--target"
    hub.mkdir(parents=True)
    partial = hub / "model.safetensors_.gstmp"
    partial.write_bytes(b"partial download")
    compile_cache = tmp_path / "jax_cache"
    compile_cache.mkdir()
    compiled = compile_cache / "verified-hlo-cache"
    compiled.write_bytes(b"compiled program")
    if cache_owner is not None:
        (compile_cache / ".tpuswarm-model").write_text(cache_owner)
    signals = tmp_path / "signals"
    for name, body in {
        "ps": (
            "#!/bin/sh\n"
            "printf '%s\\n' "
            "'91001 bash bash ensure_orbax_ckpt.sh' "
            "'91002 python3 python3 gcloud.py storage cp gs://model/cache' "
            "'91003 python3 python3 gcloud.py storage rsync gs://model/cache' "
            "'91004 python3 python3 /tmp/ray_skypilot/raylet'\n"
        ),
        "tmux": "#!/bin/sh\nexit 0\n",
        "python3": "#!/bin/sh\nexit 0\n",
    }.items():
        executable = fake_bin / name
        executable.write_text(body)
        executable.chmod(0o755)

    # Override the shell builtin: record signal recipients, verify their file
    # still exists at termination, and report that they have exited on poll.
    shell_init = tmp_path / "shell_init"
    shell_init.write_text(
        'kill() {\n'
        '  if [[ "$1" == "-TERM" ]]; then\n'
        '    test -f "$TEST_PARTIAL" || return 99\n'
        '    shift\n'
        '    printf "%s\\n" "$@" >> "$TEST_SIGNALS"\n'
        '    return 0\n'
        '  fi\n'
        '  return 1\n'
        '}\n'
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "BASH_ENV": str(shell_init),
        "TEST_PARTIAL": str(partial),
        "TEST_SIGNALS": str(signals),
        "V4_64_HOST_HOME": str(tmp_path),
        "V4_64_HOST_ROLE": "trainer",
        "SKYRL_REPO_DIR": str(repo),
        "HF_MODEL_CACHE_DIR": str(hub),
        "HF_MODEL_CACHE_GCS": "gs://test/cache",
        "MAXTEXT_MODEL_CACHE_DIR": str(tmp_path / "orbax" / "target"),
        "TUNIX_JAX_CACHE_LOCAL": str(tmp_path / "jax_cache"),
    }
    result = subprocess.run(
        ["bash", str(repo / "tpu/swarm/reconcile_v4_64_host_role.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert signals.read_text().splitlines() == ["91001", "91002", "91003"]
    assert not partial.exists()
    assert compiled.exists() is keep
