from pathlib import Path

import pytest
import yaml

from tpu.swarm.ray_train.bootstrap import workload_resources
from tpu.swarm.ray_train.build import build
from tpu.swarm.ray_train.commands import inference_environment
from tpu.swarm.ray_train.config import Config


def v6e8_config():
    return Config.from_dict(dict(
        run_id="inference-farm-v6e8-test",
        accelerator="tpu-v6e-8",
        hosts=1,
        zone="us-central1-b",
        bucket="gs://test",
        base_bundle="gs://test/base.tar.gz",
        base_bundle_sha256="a" * 64,
        model_preset="qwen3.5-27b",
        trainer={"hosts": 0},
        inference_only=True,
        inference_only_ranks=[0],
        inference=dict(tp=8, max_sequences=128, max_model_length=22528,
                       native_thinking_budget=True, routing="ingress",
                       max_loras=1, require_lease=True,
                       external_pool_attestation=True),
        cache=dict(hf="gs://test/hf", orbax="gs://test/orbax",
                   trainer_compile="gs://test/train", inference_compile="gs://test/infer"),
    ))


def test_v6e8_is_one_eight_chip_host_with_one_tp8_engine():
    config = v6e8_config()
    assert config.chips_per_host == 8
    assert config.engines_per_host == 1
    assert config.inference_hosts == 1
    assert workload_resources(config, 0) == {"TPU": 8}
    slots = config.engine_slots(["10.0.0.1"])
    assert [slot["key"] for slot in slots] == ["10.0.0.1"]


def test_v6e8_tp8_uses_all_chips_in_one_libtpu_process():
    config = v6e8_config()
    environment = inference_environment(config, Path("/cache"), Path("/run"), slot=0)
    assert environment["TPU_VISIBLE_CHIPS"] == "0,1,2,3,4,5,6,7"
    assert environment["TPU_CHIPS_PER_PROCESS_BOUNDS"] == "2,4,1"
    assert environment["TPU_PROCESS_PORT"] == str(config.ports.inference_tpu)
    assert environment["TPU_PROCESS_ADDRESSES"] == f"localhost:{config.ports.inference_tpu}"


def test_v6e8_build_uses_central1b_and_v6e_runtime(tmp_path):
    profile = Path("tpu/swarm/ray_train/profiles/inference-farm-v6e8-central1b-qwen-1-20260923.json")
    _, _, task_path = build(profile, tmp_path)
    task = yaml.safe_load(task_path.read_text())
    resources = task["resources"]
    assert resources["accelerators"] == "tpu-v6e-8"
    assert resources["zone"] == "us-central1-b"
    assert resources["accelerator_args"]["runtime_version"] == "v2-alpha-tpuv6e"


@pytest.mark.parametrize("model", ["qwen", "gemma", "muse"])
@pytest.mark.parametrize("replica", [1, 2])
def test_v6e8_farm_profiles_are_isolated(model, replica):
    run = f"inference-farm-v6e8-east5b-{model}-{replica}-tp8-20260924"
    config = Config.load(Path("tpu/swarm/ray_train/profiles") / f"{run}.json")
    assert config.run_id == run
    assert config.inference_only and config.inference_only_ranks == [0]
    assert config.inference.tp == 8 and config.engines_per_host == 1
    assert config.inference.max_sequences == 128
    assert config.inference.max_loras == 1 and config.inference.require_lease
    assert config.inference.external_pool_attestation
    assert config.cache.inference_compile_seed == ""
    assert run in config.cache.inference_compile
