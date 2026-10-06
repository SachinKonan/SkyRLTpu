import importlib.util
import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest


REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("restore_vllm_adapters", REPO / "tpu/jobman/restore_vllm_adapters.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


@pytest.mark.parametrize("already_loaded", [False, True])
@pytest.mark.parametrize("lost_ack", [False, True])
def test_reload_completed_adapter_on_each_engine(tmp_path, monkeypatch, already_loaded, lost_ack):
    name = "model_test_ss0_seq2"
    (tmp_path / ".skyrl-latest-lora").write_text(name)
    (tmp_path / name).mkdir()
    (tmp_path / name / "adapter_config.json").write_text("{}")
    loaded = {8001: already_loaded, 8002: already_loaded}
    uploads = []

    def request(req, timeout):
        url = urlparse(req.full_url)
        if req.get_method() == "POST":
            assert req.data == b""
            assert url.path == "/skyrl/v1/upload_lora_adapter"
            assert parse_qs(url.query) == {"lora_name": [name]}
            uploads.append(url.port)
            loaded[url.port] = True
            if lost_ack:
                raise TimeoutError("lost upload acknowledgment")
            return io.BytesIO(b'{"status":"ok"}')
        assert url.path == "/v1/models"
        return io.BytesIO(json.dumps({"data": [{"id": name}] if loaded[url.port] else []}).encode())

    monkeypatch.setattr(recovery, "urlopen", request)
    recovery.restore_adapters(tmp_path, 8001, 2)
    assert uploads == ([] if already_loaded else [8001, 8002])


def test_no_marker_does_not_load_an_arbitrary_old_adapter(tmp_path, monkeypatch):
    (tmp_path / "old_adapter").mkdir()
    monkeypatch.setattr(recovery, "urlopen", lambda *a, **k: pytest.fail("unexpected network access"))
    recovery.restore_adapters(tmp_path, 8001, 1)


@pytest.mark.parametrize("name", ["", "../foreign", ".hidden", "missing_adapter"])
def test_incomplete_or_invalid_marker_fails_recovery(tmp_path, monkeypatch, name):
    (tmp_path / ".skyrl-latest-lora").write_text(name)
    monkeypatch.setattr(recovery, "urlopen", lambda *a, **k: pytest.fail("unexpected network access"))
    with pytest.raises((ValueError, FileNotFoundError)):
        recovery.restore_adapters(tmp_path, 8001, 1)


def test_upload_ack_without_loaded_adapter_is_not_ready(tmp_path, monkeypatch):
    name = "model_test_ss0_seq2"
    (tmp_path / ".skyrl-latest-lora").write_text(name)
    (tmp_path / name).mkdir()
    (tmp_path / name / "adapter_config.json").write_text("{}")
    monkeypatch.setattr(recovery, "urlopen", lambda *a, **k: io.BytesIO(b'{"data":[]}'))
    with pytest.raises(RuntimeError, match="still absent"):
        recovery.restore_adapters(tmp_path, 8001, 1)


def test_monitor_restores_adapter_before_reporting_recovery():
    source = (REPO / "tpu/jobman/cell_monitor.sh").read_text()
    ready = source.index('if vllm_worker_ready "$worker" "$ip"; then')
    restore = source.index('"$SCRIPT_DIR/restore_vllm_adapters.py" || return 1', ready)
    success = source.index('echo "in-place vLLM restart recovered', restore)
    assert ready < restore < success
