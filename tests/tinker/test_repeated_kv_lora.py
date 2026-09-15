"""Native-GQA equivalence, including Adam moments and strict PEFT export."""
import ast
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from skyrl.backends.lora_init import RepeatedKVHeads


@pytest.mark.parametrize('native_heads', [2, 4])
def test_tied_gradients_across_eight_device_shards(native_heads):
    if len(jax.devices()) < 8:
        pytest.skip("requires eight CPU or TPU devices")
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    sharding = NamedSharding(Mesh(np.asarray(jax.devices()[:8]), ("model",)), P(None, None, "model"))
    raw = np.arange(2 * 3 * 32, dtype=np.float32).reshape(2, 3, 32)
    tree = {"key": {"kernel_lora_b": jax.device_put(raw, sharding)}}
    spec = RepeatedKVHeads(native_heads, 8, 4)
    tied, norm = spec.tie_gradients_and_norm(tree)
    summed = raw.reshape(2, 3, native_heads, 8 // native_heads, 4).sum(axis=-2)
    expected = np.repeat(summed[..., None, :], 8 // native_heads, axis=-2).reshape(raw.shape)
    actual = tied["key"]["kernel_lora_b"]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(norm, np.linalg.norm(summed), rtol=1e-6)
    assert actual.sharding.is_equivalent_to(sharding, actual.ndim)


@pytest.mark.parametrize('logical_heads', [4, 8])
def test_repeated_heads_match_canonical_training_over_multiple_adam_steps(logical_heads):
    spec = RepeatedKVHeads(2, logical_heads, 3)
    rng = np.random.default_rng(12)
    x = jnp.asarray(rng.normal(size=(5, 4)), dtype=jnp.float32)
    target = jnp.asarray(rng.normal(size=(5, logical_heads * 3)), dtype=jnp.float32)
    params = {"key": {"kernel_lora_a": jnp.asarray(rng.normal(size=(4, 2)), dtype=jnp.float32),
                       "kernel_lora_b": jnp.zeros((2, 6))}}
    expanded = {"key": {**params["key"], "kernel_lora_b": jnp.zeros((2, logical_heads * 3))}}

    def loss(p, native):
        b = p["key"]["kernel_lora_b"]
        if native:
            b = jnp.repeat(b.reshape(2, 2, 3), logical_heads // 2, axis=1).reshape(2, logical_heads * 3)
        return jnp.mean((x @ p["key"]["kernel_lora_a"] @ b - target) ** 2)

    adam = optax.adamw(0.03, eps=0.1, weight_decay=0.02)
    native_state, expanded_state = adam.init(params), adam.init(expanded)
    for _ in range(4):
        native_loss, native_grad = jax.value_and_grad(loss)(params, True)
        expanded_loss, expanded_grad = jax.value_and_grad(loss)(expanded, False)
        np.testing.assert_allclose(native_loss, expanded_loss, rtol=2e-6)
        raw = spec.grouped(expanded_grad["key"]["kernel_lora_b"])
        assert not np.allclose(raw[..., 0, :], raw[..., 1, :])
        tied, norm = spec.tie_gradients_and_norm(expanded_grad)
        np.testing.assert_allclose(norm, optax.global_norm(native_grad), rtol=2e-6)
        for p, grad, state, native in [(params, native_grad, native_state, True),
                                      (expanded, tied, expanded_state, False)]:
            updates, state = adam.update(grad, state, p)
            p = optax.apply_updates(p, updates)
            if native:
                params, native_state = p, state
            else:
                expanded, expanded_state = p, state
        spec.validate(expanded)
        spec.validate(expanded_state)
        np.testing.assert_allclose(params["key"]["kernel_lora_b"],
            spec.collapse_numpy(expanded["key"]["kernel_lora_b"]), rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(params["key"]["kernel_lora_a"],
                                   expanded["key"]["kernel_lora_a"], rtol=2e-6, atol=1e-7)


def test_scanned_export_preserves_heads_and_rejects_divergence():
    spec = RepeatedKVHeads(2, 4, 3)
    native = np.arange(2 * 3 * 6, dtype=np.float32).reshape(2, 3, 6)
    repeated = np.repeat(native.reshape(2, 3, 2, 3), 2, axis=-2).reshape(2, 3, 12)
    np.testing.assert_array_equal(spec.collapse_numpy(repeated), native)
    repeated[..., 3] += 1
    with pytest.raises(ValueError, match="diverged"):
        spec.collapse_numpy(repeated)
    with pytest.raises(ValueError, match="divergent replicas"):
        spec.validate({"key": {"kernel_lora_b": jnp.array(repeated)}})
    with pytest.raises(ValueError, match="unexpected"):
        spec.collapse_numpy(np.zeros((2, 3, 11)))


@pytest.mark.parametrize('model', ['qwen3.5-27b', 'muse-glimmer-30b'])
def test_actual_peft_export_folds_only_kv_b(model):
    tree = ast.parse(Path("skyrl/backends/tunix_backend.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TunixBackend")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ("_peft_tensors_maxtext", "_maxtext_path_to_hf")]
    mapping = next(n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "_MAXTEXT_PROJ_TO_HF" for t in n.targets))
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             mapping, ast.ClassDef(name="Exporter", bases=[], keywords=[], body=methods,
                                                   decorator_list=[])], type_ignores=[])
    scope = dict(np=np, logger=logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(module), "actual_export", "exec"), scope)
    exporter = scope["Exporter"]()
    exporter._maxtext_model_name = lambda: model
    exporter._repeated_kv_heads = RepeatedKVHeads(2, 4, 3)
    prefix = "['adapter']['base']['decoder']['layers']['layer_3']['attention']['attention']"
    if model.startswith('muse'):
        prefix = "['adapter']['base']['decoder']['scanned_blocks']['layers_3']['self_attention']"
    a = np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2)
    canonical_b = np.arange(2 * 2 * 6, dtype=np.float32).reshape(2, 2, 6)
    b = np.repeat(canonical_b.reshape(2, 2, 2, 3), 2, axis=-2).reshape(2, 2, 12)
    flat = {f"{prefix}['{proj}']['kernel_lora_{factor}'].value": value
            for proj in ("query", "key", "value") for factor, value in (("a", a), ("b", b))}
    tensors = exporter._peft_tensors_maxtext(None, flat)
    for proj in ("q_proj", "k_proj", "v_proj"):
        for j, layer in enumerate((3, 7)):
            hf_base = 'model.language_model.layers' if model.startswith('qwen') else 'model.layers'
            name = f"base_model.model.{hf_base}.{layer}.self_attn.{proj}"
            np.testing.assert_array_equal(tensors[name + ".lora_A.weight"], a[:, j, :].T)
            expected = b if proj == "q_proj" else canonical_b
            np.testing.assert_array_equal(tensors[name + ".lora_B.weight"], expected[:, j, :].T)
