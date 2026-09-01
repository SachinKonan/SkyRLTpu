from types import SimpleNamespace

import numpy as np

from skyrl.backends.tunix_backend import TunixBackend


def _path(module, leaf):
    return "['adapter']['base']['decoder']['scanned_blocks']['layers_0']" f"['GptOssMlp']['{module}']['{leaf}']"


def test_gptoss_router_uses_standard_peft_and_experts_use_sidecar():
    # One two-layer scan group, H=5, E=3, I=4, R=2. E != R makes
    # the expert-first grouped-GMM layout unambiguous.
    flat = {
        _path("gate", "kernel_lora_a"): np.arange(5 * 2 * 2, dtype=np.float32).reshape(5, 2, 2),
        _path("gate", "kernel_lora_b"): np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3),
        _path("wi_0_lora_a", "value"): np.ones((5, 2, 2), np.float32),
        _path("wi_0_lora_b", "value"): np.ones((2, 2, 3, 4), np.float32),
        _path("wi_1_lora_a", "value"): np.ones((5, 2, 2), np.float32),
        _path("wi_1_lora_b", "value"): np.ones((2, 2, 3, 4), np.float32),
        _path("wo_lora_a", "value"): np.ones((3, 2, 4, 2), np.float32),
        _path("wo_lora_b", "value"): np.ones((2, 2, 5), np.float32),
    }
    slot = SimpleNamespace(
        lora_state={},
        lora_config=SimpleNamespace(rank=2, alpha=4),
    )
    backend = object.__new__(TunixBackend)

    peft, experts, meta = backend._peft_tensors_gptoss(slot, flat)

    for layer in range(2):
        prefix = f"base_model.model.model.layers.{layer}.mlp.router"
        assert peft[f"{prefix}.lora_A.weight"].shape == (2, 5)
        assert peft[f"{prefix}.lora_B.weight"].shape == (3, 2)
        assert experts[f"layers.{layer}.wi_0.lora_a"].shape == (5, 2)
        assert experts[f"layers.{layer}.wi_0.lora_b"].shape == (2, 3, 4)
        assert experts[f"layers.{layer}.wi_1.lora_a"].shape == (5, 2)
        assert experts[f"layers.{layer}.wi_1.lora_b"].shape == (2, 3, 4)
        assert experts[f"layers.{layer}.wo.lora_a"].shape == (3, 4, 2)
        assert experts[f"layers.{layer}.wo.lora_b"].shape == (2, 5)

    assert not any(".router." in key for key in experts)
    assert "router" not in meta["contraction"]
    assert "B[r,e,f]" in meta["contraction"]["wi_0"]
    assert meta["scale"] == 2.0


def test_gptoss_template_key_includes_alpha_and_mlp_regex_only_targets_router():
    backend = object.__new__(TunixBackend)
    backend.config = SimpleNamespace(model_source="maxtext", lora_attn_regex=None, lora_mlp_regex=None)
    config = SimpleNamespace(rank=4, alpha=8.0, train_attn=False, train_mlp=True)

    assert backend._template_key(config) != backend._template_key(SimpleNamespace(**{**vars(config), "alpha": 16.0}))
    regex = backend._module_path_regex(config)
    assert "GptOssMlp/gate" in regex
    assert "GptOssMlp(?:/.*)?" not in regex
