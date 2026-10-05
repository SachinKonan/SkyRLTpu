from dataclasses import replace
from unittest.mock import Mock
from pathlib import Path

import pytest

from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.bootstrap import workload_resources


def test_single_engine_diagnostic_reserves_only_selected_host_tpus():
    config = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_concurrency.json")
    assert config.inference_hosts == 1 and config.hosts == 8
    assert workload_resources(config, 0) == {"TPU": 4}
    assert all(workload_resources(config, rank) == {"TPU": 0} for rank in range(1, 8))


def test_existing_training_and_inference_profiles_keep_all_host_resources():
    for path in ("qwen_v4_64_budget", "qwen_v5p_32_budget", "qwen_v4_32_inference"):
        config = Config.load("tpu/swarm/ray_train/profiles/" + path + ".json")
        assert config.inference_hosts == config.hosts-config.trainer.hosts
        assert all(workload_resources(config, rank) == {"TPU": 4} for rank in range(config.hosts))


@pytest.mark.parametrize("ranks", [[], [0, 0], [-1], [8], [True], "0"])
def test_invalid_inference_ranks_rejected(ranks):
    config = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_concurrency.json")
    with pytest.raises(ValueError, match="inference_only_ranks"):
        replace(config, inference_only_ranks=ranks).validate()


def test_training_cannot_hide_hosts_from_resource_scheduler():
    config = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_budget.json")
    with pytest.raises(ValueError, match="inference_only_ranks"):
        replace(config, inference_only_ranks=[0]).validate()


@pytest.mark.parametrize("count", [1, 4])
def test_deployment_pins_single_engine_without_changing_multi_engine_scheduling(monkeypatch, count):
    pytest.importorskip("ray.serve")
    from tpu.swarm.ray_train import serving

    config = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_concurrency.json")
    engine, ingress, serve = Mock(), Mock(), Mock()
    serve.run_many.return_value = ["handle"]
    monkeypatch.setattr(serving, "Engine", engine)
    monkeypatch.setattr(serving, "Ingress", ingress)
    monkeypatch.setattr(serving, "serve", serve)
    prepared = {f"10.0.0.{i+1}": {"role": "inference"} for i in range(count)}
    assert serving.deploy(config, prepared, Mock(), "10.0.0.1") == "handle"
    expected = {"num_replicas": count}
    if count == 1:
        expected["ray_actor_options"] = {"num_cpus": 8, "resources": {"TPU": 4, "node:10.0.0.1": 0.01}}
    engine.options.assert_called_once_with(**expected)


def test_lora_slot_ablation_changes_only_the_requested_cli_argument():
    from tpu.swarm.ray_train.commands import inference_command
    config = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_concurrency.json")
    paths = [Path("/root"), Path("/source"), Path("/snapshot"), Path("/run")]
    original = inference_command(config, *paths)
    changed = replace(config, inference=replace(config.inference, max_loras=2))
    changed.validate()
    command = inference_command(changed, *paths)
    index = original.index("--max-loras") + 1
    assert original[index] == "1" and command[index] == "2"
    assert original[:index] == command[:index] and original[index+1:] == command[index+1:]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_lora_slot_count_must_be_a_positive_integer(value):
    config = Config.load("tpu/swarm/ray_train/profiles/qwen_v4_64_concurrency.json")
    with pytest.raises(ValueError, match="max_loras"):
        replace(config, inference=replace(config.inference, max_loras=value)).validate()
