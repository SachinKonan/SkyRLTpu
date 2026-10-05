import json
from pathlib import Path
import re

import pytest
import yaml

from tpu.swarm.ray_train.bootstrap import workload_resources
from tpu.swarm.ray_train.build import build
from tpu.swarm.ray_train.commands import inference_environment
from tpu.swarm.ray_train.config import Config

PROFILES = Path("tpu/swarm/ray_train/profiles")
GRADING_FARMS = ["farm-v6e8-east5b-qwen-grading-ac2-20260924",
                 "farm-v6e8-east5b-qwen-grading-ac2-2-20260924"]


def v6e8_config(**inference):
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
        inference=dict(dict(tp=4, max_sequences=32, max_model_length=22528,
                            native_thinking_budget=True, routing="ingress",
                            max_loras=1, require_lease=True,
                            external_pool_attestation=True), **inference),
        cache=dict(hf="gs://test/hf", orbax="gs://test/orbax",
                   trainer_compile="gs://test/train", inference_compile="gs://test/infer"),
    ))


def test_v6e8_is_one_eight_chip_host_with_two_tp4_engines():
    config = v6e8_config()
    assert config.chips_per_host == 8
    assert config.engines_per_host == 2
    assert config.inference_hosts == 1
    assert workload_resources(config, 0) == {"TPU": 8}
    slots = config.engine_slots(["10.0.0.1"])
    assert len(slots) == 2 and len({slot["key"] for slot in slots}) == 2


def test_qwen_tp8_is_rejected_on_v6e8():
    # Its single TP8 engine halts in SparseCore under mixed prefill+decode load.
    with pytest.raises(ValueError, match="two TP4 engines"):
        v6e8_config(tp=8, max_sequences=128)


def test_tp8_is_rejected_outside_single_host_v6e8_farms():
    with pytest.raises(ValueError, match="TP8 inference requires"):
        Config.from_dict(dict(v6e8_config().to_dict(), accelerator="tpu-v4-32", hosts=4,
                              zone="us-central2-b", inference_only_ranks=[0, 1, 2, 3],
                              inference=dict(v6e8_config().to_dict()["inference"], tp=8)))


def test_multi_engine_hosts_pin_each_engine_to_its_own_chips_and_bypass_the_libtpu_lock():
    config = v6e8_config()
    environments = [inference_environment(config, Path("/cache"), Path("/run"), slot=slot) for slot in range(2)]
    assert [env["TPU_VISIBLE_CHIPS"] for env in environments] == ["0,1,2,3", "4,5,6,7"]
    assert len({env["TPU_PROCESS_PORT"] for env in environments}) == 2
    # Without the override the second engine aborts: "TPU is already in use".
    assert all(env["ALLOW_MULTIPLE_LIBTPU_LOAD"] == "1" for env in environments)


def test_single_engine_hosts_keep_the_libtpu_lock():
    config = Config.load(PROFILES / "farm-v5p32-qwen-grading-ac2-20260922.json")
    assert config.engines_per_host == 1
    assert "ALLOW_MULTIPLE_LIBTPU_LOAD" not in inference_environment(config, Path("/cache"), Path("/run"), slot=0)


def test_grading_farm_cpu_budget_covers_every_engine_and_slot():
    farm = v6e8_config()
    farm = Config.from_dict(dict(farm.to_dict(), grading={
        "families": {"ac2": {"slots_per_host": 16, "cpus": 2, "memory_gib": 4}}},
        systemd_runtime=True, cache=dict(farm.to_dict()["cache"], reserve_gib=192)))
    # Two TP4 engines (2 x 8) + ingress (1) + overhead (8) + 16 slots x 2 CPUs.
    assert farm.ray_cpus_per_host == 16 + 9 + 32


@pytest.mark.parametrize("run", GRADING_FARMS)
def test_v6e8_grading_farm_profiles(run):
    config = Config.load(PROFILES / f"{run}.json")
    assert config.run_id == run
    assert config.inference_only and config.inference_only_ranks == [0]
    assert config.inference.tp == 4 and config.engines_per_host == 2
    assert config.inference.max_sequences == 32
    assert config.inference.require_lease and config.inference.external_pool_attestation
    assert set(config.grading_families) == {"ac2"}
    assert run in config.cache.inference_compile


@pytest.mark.parametrize("run", GRADING_FARMS)
def test_v6e8_grading_farms_never_read_another_region(run):
    """Cross-region GCS reads are billed as egress; farm and caches share us-east5."""
    raw = json.loads((PROFILES / f"{run}.json").read_text())
    assert raw["zone"] == "us-east5-b"
    paths = [raw["bucket"], raw["base_bundle"]] + [
        value for value in raw["cache"].values() if isinstance(value, str) and value.startswith("gs://")]
    regions = {re.match(r"gs://sk7524-tinker-tpu-(us-[a-z0-9]+)", path).group(1) for path in paths}
    assert regions == {"us-east5"}


def test_v6e8_grading_farm_build_targets_east5b_v6e_runtime(tmp_path):
    _, _, task_path = build(PROFILES / f"{GRADING_FARMS[0]}.json", tmp_path)
    resources = yaml.safe_load(task_path.read_text())["resources"]
    assert resources["accelerators"] == "tpu-v6e-8"
    assert resources["zone"] == "us-east5-b"
    assert resources["accelerator_args"]["runtime_version"] == "v2-alpha-tpuv6e"
