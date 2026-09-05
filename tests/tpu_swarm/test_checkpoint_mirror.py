import subprocess

import pytest

from skyrl.utils.checkpoint_mirror import mirror_checkpoint_to_gcs


def test_checkpoint_mirror_uploads_and_verifies_size(monkeypatch, tmp_path):
    checkpoint = tmp_path / "ss0_seq1.tar.gz"
    checkpoint.write_bytes(b"durable-checkpoint")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "describe" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=str(checkpoint.stat().st_size),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("skyrl.utils.checkpoint_mirror.subprocess.run", fake_run)
    destination = mirror_checkpoint_to_gcs(
        checkpoint,
        "gs://test-bucket/skyrl-checkpoints",
        "model_test",
        "sampler_weights",
    )

    assert destination == (
        "gs://test-bucket/skyrl-checkpoints/model_test/"
        "sampler_weights/ss0_seq1.tar.gz"
    )
    assert calls[0][0] == ["gcloud", "storage", "cp", str(checkpoint), destination]
    assert calls[1][0][:5] == [
        "gcloud",
        "storage",
        "objects",
        "describe",
        destination,
    ]


def test_checkpoint_mirror_rejects_size_mismatch(monkeypatch, tmp_path):
    checkpoint = tmp_path / "step.tar.gz"
    checkpoint.write_bytes(b"checkpoint")

    def fake_run(command, **kwargs):
        stdout = "1" if "describe" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("skyrl.utils.checkpoint_mirror.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="verification failed"):
        mirror_checkpoint_to_gcs(
            checkpoint,
            "gs://test-bucket/skyrl-checkpoints",
            "model_test",
        )
