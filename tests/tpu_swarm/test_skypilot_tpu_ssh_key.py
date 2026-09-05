from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_skypilot_syncs_controller_key_only_to_tpu_head(
    monkeypatch, tmp_path: Path
):
    from sky import clouds
    from sky.backends import cloud_vm_ray_backend
    from sky.utils import command_runner

    key = tmp_path / "sky-key"
    key.write_text("private-key")

    class FakeRunner(command_runner.SSHCommandRunner):
        def __init__(self):
            self.ssh_private_key = str(key)
            self.rsync_calls = []
            self.run_calls = []

        def rsync(self, **kwargs):
            self.rsync_calls.append(kwargs)

        def run(self, command, **kwargs):
            self.run_calls.append((command, kwargs))
            return 0

    head = FakeRunner()
    peer = FakeRunner()
    resources = SimpleNamespace(cloud=clouds.GCP())
    handle = SimpleNamespace(launched_resources=resources)
    monkeypatch.setattr(
        cloud_vm_ray_backend.gcp_utils, "is_tpu_vm_pod", lambda _: True
    )

    cloud_vm_ray_backend._sync_tpu_vm_pod_ssh_key_to_head(
        handle, [head, peer]
    )

    assert head.rsync_calls == [
        {
            "source": str(key),
            "target": "~/ray_bootstrap_key.pem",
            "up": True,
            "stream_logs": False,
        }
    ]
    assert head.run_calls[0][0] == "chmod 600 ~/ray_bootstrap_key.pem"
    assert peer.rsync_calls == []
