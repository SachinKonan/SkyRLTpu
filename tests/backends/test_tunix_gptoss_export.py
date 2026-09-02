from types import SimpleNamespace

from flax import nnx
from flax.core.spmd import get_logical_axis_rules
import jax.numpy as jnp
import numpy as np

from skyrl.backends.tunix_backend import (
    _MaxTextAdapterShim,
    _repair_maxtext_scanned_lora_metadata,
    TunixBackend,
)


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


def test_maxtext_adapter_restores_logical_axis_rules_for_every_trace():
    seen = []

    class Adapter:
        def __init__(self):
            self.base = self

        def __call__(self, input_tokens, positions, cache, attention_mask):
            seen.append(("model", get_logical_axis_rules()))
            return input_tokens, cache

        def logits_from_hidden_states_for_vocab_tiling(self, hidden, *_args):
            seen.append(("logits", get_logical_axis_rules()))
            return hidden

    rules = (("activation_embed", "tensor"), ("mlp_moe", ("fsdp", "tensor")))
    shim = _MaxTextAdapterShim(Adapter(), rules)
    values = jnp.ones((2, 4), dtype=jnp.float32)

    output, _ = shim(values, values, None, values)
    logits = shim.logits_from_hidden(values)

    np.testing.assert_array_equal(output, values)
    np.testing.assert_array_equal(logits, values)
    assert seen == [("model", rules), ("logits", rules)]
    assert get_logical_axis_rules() == ()


def test_maxtext_adapter_requests_hidden_state_when_skipping_lm_head():
    requested = []

    class Adapter:
        def __call__(self, input_tokens, positions, cache, attention_mask, *, output_hidden_states=False):
            requested.append(output_hidden_states)
            return input_tokens, cache

    shim = _MaxTextAdapterShim(Adapter())
    values = jnp.ones((2, 4), dtype=jnp.float32)

    output, _ = shim(values, values, None, values, skip_lm_head=True)

    np.testing.assert_array_equal(output, values)
    assert requested == [True]


def test_repair_maxtext_scanned_lora_metadata_uses_factor_axes():
    class Projection(nnx.Module):
        def __init__(self, *, base_shape, base_axes, factor_shapes, axis):
            metadata = {
                "out_sharding": base_axes,
                nnx.PARTITION_NAME: "layers",
                "param_scan_axis": 1,
            }
            self.axis = axis
            self.kernel = nnx.Param(jnp.zeros(base_shape))
            self.kernel_lora_a = nnx.LoRAParam(jnp.zeros(factor_shapes[0]))
            self.kernel_lora_b = nnx.LoRAParam(jnp.zeros(factor_shapes[1]))
            for variable in (self.kernel, self.kernel_lora_a, self.kernel_lora_b):
                for key, value in metadata.items():
                    variable.set_metadata(key, value)

    class Model(nnx.Module):
        def __init__(self):
            # K/V-style projection: one contracting axis and two output axes.
            self.key = Projection(
                base_shape=(8, 2, 4, 3),
                base_axes=("embed_attn", "layers", "kv_heads", "kv_head_dim"),
                factor_shapes=((8, 2, 5), (5, 2, 12)),
                axis=(-1,),
            )
            # Output projection: two contracting axes before the output axis.
            self.out = Projection(
                base_shape=(4, 2, 3, 8),
                base_axes=("heads", "layers", "kv", "embed_attn"),
                factor_shapes=((12, 2, 5), (5, 2, 8)),
                axis=(-2, -1),
            )

    model = Model()
    assert _repair_maxtext_scanned_lora_metadata(model) == 4

    assert model.key.kernel_lora_a.get_metadata()["out_sharding"] == ("embed_attn", "layers", None)
    assert model.key.kernel_lora_b.get_metadata()["out_sharding"] == (None, "layers", "kv_heads")
    assert model.out.kernel_lora_a.get_metadata()["out_sharding"] == ("heads", "layers", None)
    assert model.out.kernel_lora_b.get_metadata()["out_sharding"] == (None, "layers", "embed_attn")
    assert model.key.kernel.get_metadata()["out_sharding"] == (
        "embed_attn",
        "layers",
        "kv_heads",
        "kv_head_dim",
    )
