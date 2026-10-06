import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2] / "tpu/swarm/ray_serve_canary"


@pytest.fixture
def restore_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location("canary_restore", ROOT / "prepare_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_manifest():
    import hashlib
    payloads = {"config.json": b"{}", "tokenizer.json": b"{}",
                "preprocessor_config.json": b"{}", "video_preprocessor_config.json": b"{}",
                "part.safetensors": b"weights",
                "model.safetensors.index.json": b'{"weight_map":{"w":"part.safetensors"}}'}
    manifest = {"files": {name: {"size": len(data), "lfs_sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in payloads.items()}}
    blobs = {hashlib.sha256(data).hexdigest(): data for data in payloads.values()}
    return manifest, blobs


def test_single_batch_and_warm_reuse(tmp_path, monkeypatch, restore_module):
    manifest, blobs = fixture_manifest()
    calls = []

    def download(command, root, stage, label, inputs, check):
        calls.append(command)
        for url in inputs.splitlines():
            blob = url.rsplit("/", 1)[-1]
            (stage / blob).write_bytes(blobs[blob])
        return 0

    monkeypatch.setattr(restore_module, "run_transfer", download)
    restore_module.restore(tmp_path, "gs://test", manifest)
    assert len(calls) == 1
    assert "--read-paths-from-stdin" in calls[0]
    assert (tmp_path / "model/part.safetensors").read_bytes() == b"weights"
    assert not list((tmp_path / "model-downloads").iterdir())
    restore_module.restore(tmp_path, "gs://test", manifest)
    assert len(calls) == 1
    (tmp_path / "model/part.safetensors").write_bytes(b"bad")
    restore_module.restore(tmp_path, "gs://test", manifest)
    assert len(calls) == 2


def test_incomplete_files_are_not_promoted(tmp_path, monkeypatch, restore_module):
    manifest, blobs = fixture_manifest()
    calls = []

    def download(command, root, stage, label, inputs, check):
        calls.append(command)
        for blob, data in blobs.items():
            (stage / (blob + "_.gstmp")).write_bytes(data)
        return 1

    monkeypatch.setattr(restore_module, "run_transfer", download)
    with pytest.raises(RuntimeError, match="incomplete cache"):
        restore_module.restore(tmp_path, "gs://test", manifest)
    assert len(calls) == 3
    assert not list((tmp_path / "model").iterdir())


def test_rejects_unsafe_manifest(restore_module):
    with pytest.raises(ValueError, match="unsafe"):
        restore_module.selected_files({"files": {"../config.json": {}}})


def test_native_settings_and_no_outer_thread_pool():
    shell = (ROOT / "run.sh").read_text()
    assert '${CANARY_GCLOUD_PROCESSES:-4}' in shell
    assert '${CANARY_GCLOUD_THREADS:-8}' in shell
    assert '${CANARY_GCLOUD_SLICE_THRESHOLD:-0}' in shell
    assert "gcloud storage rsync" not in shell
    assert "ThreadPoolExecutor" not in (ROOT / "prepare_cache.py").read_text()


def test_xla_excludes_gcloud_temporary_objects(tmp_path, monkeypatch):
    import base64
    import hashlib
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location("canary_xla", ROOT / "prepare_xla.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    digest = base64.b64encode(hashlib.md5(b"abc").digest()).decode()
    entries = [{"type": "cloud_object", "metadata": {
        "bucket": "test", "name": "prefix/" + name, "generation": "123",
        "size": "3", "md5Hash": digest}}
        for name in ["jit-test-cache", "jit-test-cache_.gstmp", "jit-test-atime"]]
    assert module.cache_objects(entries, "gs://test/prefix") == [
        ("gs://test/prefix/jit-test-cache#123", "jit-test-cache", 3, digest)]
    target = tmp_path / "cache"
    target.write_bytes(b"abc")
    assert module.valid(target, 3, digest)
    target.write_bytes(b"bad")
    assert not module.valid(target, 3, digest)


def test_transfer_cancellation_stops_child(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), CANARY_ROOT=str(tmp_path))
    command = [sys.executable, str(ROOT / "transfer.py"), "test", str(tmp_path / "data"),
               sys.executable, "-c", "import time; time.sleep(60)"]
    process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL)
    child = None
    try:
        metrics = tmp_path / "transfer-test.jsonl"
        for _ in range(100):
            if metrics.exists() and metrics.stat().st_size:
                child = json.loads(metrics.read_text().splitlines()[0])["pid"]
                break
            time.sleep(0.05)
        assert child is not None
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=20) == 128 + signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if child is not None:
            try:
                os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
