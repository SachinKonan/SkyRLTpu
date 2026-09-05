from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path


def test_prepare_qwen35_uses_controller_key_on_head_only(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            if [ "${JOBMAN_WORKER_ID:-}" = 0 ]; then
              printf '%s\n' "$JOBMAN_TPU_INTERNAL_IPS" > "$TPUSWARM_TEST_ROOT/ordered_ips"
            fi
            exit 0
            """
        )
    )
    fake_bash.chmod(0o755)

    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os
            import pathlib
            import sys
            import time

            target = next(arg for arg in sys.argv if "@127.0.0." in arg)
            rank = int(target.rsplit(".", 1)[1]) - 1
            root = pathlib.Path(os.environ["TPUSWARM_TEST_ROOT"])
            marker = root / f"rank{rank}/.cache/tpuswarm/qwen35-v6e32-node-ready-123"
            for _ in range(200):
                if marker.exists():
                    print(marker.read_text().strip())
                    raise SystemExit(0)
                time.sleep(0.05)
            raise SystemExit(1)
            """
        )
    )
    fake_ssh.chmod(0o755)

    ip_order = [0, 2, 5, 1, 7, 3, 6, 4]
    ips = "\n".join(f"127.0.0.{rank + 1}" for rank in ip_order)
    processes = []
    for rank in range(8):
        home = tmp_path / f"rank{rank}"
        home.mkdir()
        if rank == 0:
            (home / "ray_bootstrap_key.pem").write_text("controller-key")
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(home),
                "PATH": f"{fake_bin}:{env['PATH']}",
                "SKYPILOT_SETUP_NODE_RANK": str(rank),
                "SKYPILOT_SETUP_NODE_IPS": ips,
                "TPUSWARM_BUNDLE_GENERATION": "123",
                "TPUSWARM_TEST_ROOT": str(tmp_path),
                "SKYRL_REPO_DIR": str(repo),
            }
        )
        processes.append(
            subprocess.Popen(
                ["/bin/bash", str(repo / "tpu/swarm/prepare_qwen35_v6e32.sh")],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    results = [process.communicate(timeout=30) for process in processes]
    failures = [
        (rank, process.returncode, stdout, stderr)
        for rank, (process, (stdout, stderr)) in enumerate(zip(processes, results))
        if process.returncode != 0
    ]
    assert not failures

    head_key = tmp_path / "rank0/ray_bootstrap_key.pem"
    head_link = tmp_path / "rank0/.ssh/jobman_tpu_ed25519"
    assert head_link.resolve() == head_key
    assert (head_key.stat().st_mode & 0o777) == 0o600
    assert (tmp_path / "ordered_ips").read_text().strip() == ",".join(
        f"127.0.0.{rank + 1}" for rank in range(8)
    )
    for rank in range(1, 8):
        assert not (tmp_path / f"rank{rank}/ray_bootstrap_key.pem").exists()
