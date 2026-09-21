import subprocess
from pathlib import Path

import pytest

from skyrl.utils.checkpoint_mirror import (
    mirror_checkpoint_to_gcs,
    restore_checkpoint_from_gcs,
)


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


@pytest.mark.parametrize('always_fail', [False, True])
def test_checkpoint_mirror_retries_interrupted_uploads(monkeypatch, tmp_path, always_fail):
    checkpoint = tmp_path / 'step.tar.gz'
    checkpoint.write_bytes(b'checkpoint')
    uploads = []
    monkeypatch.setenv('CLOUDSDK_STORAGE_PROCESS_COUNT', '32')
    monkeypatch.setattr('skyrl.utils.checkpoint_mirror.time.sleep', lambda _: None)

    def fake_run(command, **kwargs):
        if 'describe' in command:
            return subprocess.CompletedProcess(command, 0, stdout='10', stderr='')
        uploads.append(command)
        assert kwargs['env']['CLOUDSDK_STORAGE_PROCESS_COUNT'] == '1'
        assert kwargs['env']['CLOUDSDK_STORAGE_THREAD_COUNT'] == '1'
        return subprocess.CompletedProcess(command, -15 if always_fail or len(uploads) == 1 else 0,
                                           stdout='', stderr='interrupted')

    monkeypatch.setattr('skyrl.utils.checkpoint_mirror.subprocess.run', fake_run)
    if always_fail:
        with pytest.raises(RuntimeError, match=r'after 3 attempts \(exit -15\)'):
            mirror_checkpoint_to_gcs(checkpoint, 'gs://bucket/checkpoints', 'model')
        assert len(uploads) == 3
    else:
        assert mirror_checkpoint_to_gcs(checkpoint, 'gs://bucket/checkpoints', 'model').endswith('/step.tar.gz')
        assert len(uploads) == 2
    assert all(command == uploads[0] for command in uploads)


def test_checkpoint_restore_downloads_and_verifies_size(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model_test" / "000003.tar.gz"
    payload = b"distributed-checkpoint"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "describe" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=str(len(payload)),
                stderr="",
            )
        Path(command[-1]).write_bytes(payload)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("skyrl.utils.checkpoint_mirror.subprocess.run", fake_run)
    source = restore_checkpoint_from_gcs(
        checkpoint,
        "gs://test-bucket/skyrl-checkpoints",
        "model_test",
    )

    assert source == "gs://test-bucket/skyrl-checkpoints/model_test/000003.tar.gz"
    assert checkpoint.read_bytes() == payload
    assert calls[0][0][2:4] == ["objects", "describe"]
    assert calls[1][0][:4] == ["gcloud", "storage", "cp", source]


def test_checkpoint_restore_rejects_partial_download(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model_test" / "000003.tar.gz"

    def fake_run(command, **kwargs):
        if "describe" in command:
            return subprocess.CompletedProcess(command, 0, stdout="20", stderr="")
        Path(command[-1]).write_bytes(b"short")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("skyrl.utils.checkpoint_mirror.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="download verification failed"):
        restore_checkpoint_from_gcs(
            checkpoint,
            "gs://test-bucket/skyrl-checkpoints",
            "model_test",
        )
    assert not checkpoint.exists()
