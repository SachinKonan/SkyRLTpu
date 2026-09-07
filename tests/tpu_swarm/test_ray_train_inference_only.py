from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("ray.serve")
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.controller import Controller
from tpu.swarm.ray_train import host
from tpu.swarm.ray_train.commands import inference_command


PROFILE = "tpu/swarm/ray_train/profiles/qwen_v4_32_inference.json"


def test_inference_only_profile_has_four_engines_and_no_trainers():
    cfg = Config.load(PROFILE)
    assert cfg.inference_only and cfg.trainer.hosts == 0
    assert cfg.inference_hosts == 4
    command = inference_command(cfg, Path("/root"), Path("/src"), Path("/hf"), Path("/run"))
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.8"
    assert command[command.index("--max-num-seqs") + 1] == "16"
    assert command[command.index("--tensor-parallel-size") + 1] == "4"
    assert "mem80" in cfg.cache.inference_compile
    raw = cfg.to_dict()
    raw["inference_only"] = False
    with pytest.raises(ValueError, match="nonempty"):
        Config.from_dict(raw)
    raw["inference_only"] = True
    raw["trainer"]["hosts"] = 1
    with pytest.raises(ValueError, match="zero trainer"):
        Config.from_dict(raw)


def test_inference_only_run_waits_without_starting_client():
    obj = SimpleNamespace(config=Config.load(PROFILE), setup=Mock(), report=Mock(),
                          ips=["10.0.0.1"], stopping=Mock(), failure=None)
    obj.stopping.wait.return_value = True
    assert Controller.run(obj) == 143
    obj.setup.assert_called_once()
    obj.report.assert_called_once_with("inference_only_waiting", endpoint="http://10.0.0.1:19800")


def test_inference_head_does_not_install_client_or_orbax(tmp_path, monkeypatch):
    cfg = Config.load(PROFILE)
    store = Mock()
    store.restore_hf.return_value = tmp_path / "hf"
    monkeypatch.setattr(host, "mount_cache", lambda *args: tmp_path / "ram")
    monkeypatch.setattr(host, "CacheStore", lambda *args: store)
    obj = SimpleNamespace(config=cfg, root=tmp_path, rank=0, gcs=Mock(),
                          source=tmp_path / "source", log=tmp_path / "events.jsonl",
                          install_role=Mock(), install_client=Mock(),
                          compile_prefix=lambda: cfg.cache.inference_compile)
    result = host.Host.prepare(obj, "inference")
    assert result["role"] == "inference"
    obj.install_role.assert_called_once_with("inference")
    obj.install_client.assert_not_called()
    store.restore_tree.assert_not_called()
    store.restore_hf.assert_called_once_with(cfg.cache.hf, cfg.model, weights=True)
