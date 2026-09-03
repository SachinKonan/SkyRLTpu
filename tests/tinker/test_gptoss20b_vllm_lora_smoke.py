import json
import math
from pathlib import Path

import numpy as np
import yaml
from safetensors.numpy import load_file

from tpu import gptoss20b_vllm_lora_smoke as smoke


_CONFIG_DIR = Path(__file__).parents[2] / "tpu" / "jobman" / "configs"
_RACE_CONFIGS = (
    _CONFIG_DIR / "v6e8_gptoss20b_vllm_lora_smoke_asia_ne1.yaml",
    _CONFIG_DIR / "v6e8_gptoss20b_vllm_lora_smoke_east5b.yaml",
    _CONFIG_DIR / "v6e8_gptoss20b_vllm_lora_smoke_europe_w4a.yaml",
)


def test_synthetic_adapter_matches_split_tunix_export_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "LAYERS", 2)
    monkeypatch.setattr(smoke, "EXPERTS", 3)
    monkeypatch.setattr(smoke, "HIDDEN", 5)
    monkeypatch.setattr(smoke, "INTERMEDIATE", 4)

    adapter = smoke._write_adapter(
        tmp_path,
        "fixture",
        rank=2,
        alpha=4.0,
        router_seed=7524,
        expert_seed=9752,
    )
    router = load_file(str(adapter / "adapter_model.safetensors"))
    experts = load_file(str(adapter / "moe_lora.safetensors"))
    meta = json.loads((adapter / "moe_lora.json").read_text())

    assert len(router) == 2 * 2
    assert len(experts) == 2 * 6
    assert router[
        "base_model.model.model.layers.0.mlp.router.lora_A.weight"
    ].shape == (2, 5)
    assert router[
        "base_model.model.model.layers.0.mlp.router.lora_B.weight"
    ].shape == (3, 2)
    assert experts["layers.0.wi_0.lora_a"].shape == (5, 2)
    assert experts["layers.0.wi_0.lora_b"].shape == (2, 3, 4)
    assert experts["layers.0.wo.lora_a"].shape == (3, 4, 2)
    assert experts["layers.0.wo.lora_b"].shape == (2, 5)
    assert meta["format"] == "gptoss-moe-lora/v1"
    assert meta["scale"] == 2.0


def test_zero_fixture_has_no_router_or_expert_delta(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "LAYERS", 1)
    monkeypatch.setattr(smoke, "EXPERTS", 3)
    monkeypatch.setattr(smoke, "HIDDEN", 5)
    monkeypatch.setattr(smoke, "INTERMEDIATE", 4)

    adapter = smoke._write_adapter(
        tmp_path,
        "zero",
        rank=2,
        alpha=4.0,
        router_seed=None,
        expert_seed=0,
    )

    for path in ("adapter_model.safetensors", "moe_lora.safetensors"):
        assert all(np.count_nonzero(value) == 0 for value in load_file(str(adapter / path)).values())


def test_distance_is_finite_and_detects_discrete_and_numeric_changes():
    assert smoke._distance({"x": [1.0]}, {"x": [1.0]}) == 0.0
    assert smoke._distance({"x": [1.0]}, {"x": [1.25]}) == 0.25
    discrete = smoke._distance({"token": "a"}, {"token": "b"})
    assert discrete > 1e100
    assert math.isfinite(discrete)


def test_worker_update_gate_accepts_nested_rpc_results():
    smoke._assert_update(
        {
            "moe_update": [
                {
                    "layers": smoke.LAYERS,
                    "cleared": False,
                    "base_weights_mutated": False,
                }
            ]
        },
        cleared=False,
    )


def test_remote_runner_pins_runtime_and_publishes_success_conditionally():
    runner = (
        Path(__file__).parents[2]
        / "tpu"
        / "run_gptoss20b_vllm_lora_smoke.sh"
    ).read_text()

    assert '"vllm-tpu==0.23.0"' in runner
    assert "24c767036ccfa3d4e010f72f4bfe7a91ca3afc05" in runner
    assert "--tensor-parallel-size 8" in runner
    assert "--enable-lora" in runner
    assert "--if-generation-match=0" in runner
    assert "acceptance passed" in runner


def test_regional_jobman_race_has_one_shared_run_and_success_contract():
    configs = [yaml.safe_load(path.read_text()) for path in _RACE_CONFIGS]
    zones = {config["tpu"]["zone"] for config in configs}
    expected_zones = {"asia-northeast1-b", "us-east5-b", "europe-west4-a"}

    assert zones == expected_zones
    assert all(config["tpu"]["accelerator"] == "v6e-8" for config in configs)
    assert all(config["tpu"]["pricing"] == "spot" for config in configs)
    assert len({config["resumable"]["run_id_prefix"] for config in configs}) == 1
    assert len(
        {json.dumps(config["resumable"]["run_spec"], sort_keys=True) for config in configs}
    ) == 1

    environments = [config["resumable"]["env"] for config in configs]
    assert len({env["SMOKE_RESULT_GCS"] for env in environments}) == 1
    assert len({env["SKYRLTPU_BUNDLE_URL"] for env in environments}) == 1
    assert len({env["SKYRLTPU_BUNDLE_SHA256"] for env in environments}) == 1
    assert len({env["SMOKE_LOG_GCS"] for env in environments}) == 3
    assert all(
        "gcloud storage objects describe" in config["resumable"]["completion_probe"]["command"]
        for config in configs
    )
