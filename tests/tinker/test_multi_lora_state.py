"""CPU execution of the actual Tunix create/step methods on a tiny shared base."""
import ast
import gc
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from skyrl.backends.lora_init import initialize_dense_lora
from skyrl.backends import lora_mix
from skyrl.tinker import types
from skyrl.tinker.loss_fns import cispo_loss, LossFnConfig


def test_initialization_is_seeded_sharded_and_has_zero_delta():
    reference = {"kernel_lora_a": jnp.zeros((16, 2, 4)), "kernel_lora_b": jnp.ones((4, 2, 8))}
    a, b = [initialize_dense_lora(reference, seed) for seed in (3, 4)]
    same = initialize_dense_lora(reference, 3)
    np.testing.assert_array_equal(a["kernel_lora_a"], same["kernel_lora_a"])
    assert not np.array_equal(a["kernel_lora_a"], b["kernel_lora_a"])
    np.testing.assert_array_equal(a["kernel_lora_b"], 0)
    assert a["kernel_lora_a"].sharding == reference["kernel_lora_a"].sharding
    np.testing.assert_array_equal(reference["kernel_lora_b"], 1)


def test_capped_is_extreme_ratios_and_negative_advantages():
    lp = jnp.array([-1., -1000., -2., -3.])
    behavior = jnp.array([-1000., -1., -1000., -1000.])
    advantage = jnp.array([1., 1., -1., 1.])
    mask = jnp.array([1., 1., 1., 0.])
    cfg = LossFnConfig(jnp.array(0.), jnp.array(2.))
    loss = lambda x: cispo_loss(x, mask, behavior, advantage, cfg).sum()
    assert np.isfinite(loss(lp))
    np.testing.assert_allclose(jax.grad(loss)(lp), [-2., 0., 2., 0.])


def actual_backend_methods():
    source = ast.parse(Path("skyrl/backends/tunix_backend.py").read_text())
    original = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "TunixBackend")
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in
               {"create_model", "optim_step", "_init_lora_state", "_template_key", "abort_multi_lora_training", "load_checkpoint"}]
    model_slot = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "ModelSlot")
    cls = ast.ClassDef(name="Backend", bases=[], keywords=[], body=methods, decorator_list=[])
    tree = ast.Module(body=[model_slot, cls], type_ignores=[])
    scope = dict(jax=jax, jnp=jnp, np=np, nnx=nnx, optax=optax, types=types, os=os, gc=gc,
                 lora_mix=lora_mix, dataclass=dataclass, field=field, Any=Any, AnyPath=Path, Path=Path, logger=logging.getLogger(__name__),
                 _jitted_global_norm=jax.jit(optax.global_norm))
    exec(compile(ast.fix_missing_locations(tree), "actual_tunix_methods", "exec"), scope)
    scope["Backend"]._log_hbm = lambda self, stage: None
    return scope["Backend"]


def test_one_adapter_step_preserves_peer_weights_moments_and_accumulator(monkeypatch):
    monkeypatch.delenv("TUNIX_GRADIENT_PROBE_DIR", raising=False)
    monkeypatch.delenv("TUNIX_REPLAY_DIAGNOSTICS", raising=False)

    class Tiny(nnx.Module):
        def __init__(self):
            self.base = nnx.Param(jnp.eye(4))
            # Real GPT-OSS scanned expert layouts, with expert count != rank.
            self.wo_lora_a = nnx.LoRAParam(jnp.zeros((3, 2, 5, 2)))
            self.wo_lora_b = nnx.LoRAParam(jnp.zeros((2, 2, 4)))
            self.wi_0_lora_a = nnx.LoRAParam(jnp.zeros((4, 2, 2)))
            self.wi_0_lora_b = nnx.LoRAParam(jnp.zeros((2, 2, 3, 5)))
            self.kernel_lora_a = nnx.LoRAParam(jnp.zeros((4, 2)))
            self.kernel_lora_b = nnx.LoRAParam(jnp.zeros((2, 4)))

    backend = actual_backend_methods()()
    model = Tiny()
    template = SimpleNamespace(model=model, lora_shape=nnx.state(model, nnx.LoRAParam))
    backend.models = {}
    backend.templates = {(2, 2.0, True, True): template}
    backend._get_template = lambda cfg: template
    backend.base_state = None
    backend.config = SimpleNamespace(max_lora_rank=2, free_base_state_after_template=True,
                                     independent_lora_init=True)
    for name, seed in (("a", 1), ("b", 2)):
        backend.create_model(name, types.LoraConfig(rank=2, alpha=2., seed=seed))
    a, b = backend.models["a"], backend.models["b"]
    for slot in (a, b):
        slot.accum_grads = jax.tree.map(jnp.ones_like, slot.lora_state)
        slot.accum_count = 2

    def snapshot(tree):
        return [np.array(x) for x in jax.tree.leaves(tree)]

    peer_weights = snapshot(b.lora_state)
    peer_optimizer = snapshot(nnx.state(b.optimizer))
    peer_grad = snapshot(b.accum_grads)
    own_weights = snapshot(a.lora_state)
    before_base = np.array(model.base[...])
    request = SimpleNamespace(adam_params=types.AdamParams(
        learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0))
    backend.optim_step("a", request)
    assert a.accum_count == 0 and a.accum_grads is None
    assert b.accum_count == 2
    for before, after in zip(peer_weights, snapshot(b.lora_state), strict=True):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(peer_optimizer, snapshot(nnx.state(b.optimizer)), strict=True):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(peer_grad, snapshot(b.accum_grads), strict=True):
        np.testing.assert_array_equal(before, after)
    assert any(not np.array_equal(x, y) for x, y in zip(own_weights, snapshot(a.lora_state)))
    np.testing.assert_array_equal(model.base[...], before_base)
    a_after = snapshot(a.lora_state)
    backend.optim_step("b", request)
    for before, after in zip(a_after, snapshot(a.lora_state), strict=True):
        np.testing.assert_array_equal(before, after)


def test_failed_population_clears_every_accumulator_and_blocks_optimizer():
    import pytest
    backend = actual_backend_methods()()
    backend.models = {name: SimpleNamespace(accum_grads={"x": jnp.ones(2)}, accum_count=7,
                                            training_failed=False) for name in ("a", "b")}
    backend.abort_multi_lora_training(["a", "b"])
    for name, slot in backend.models.items():
        assert slot.accum_grads is None and slot.accum_count == 0 and slot.training_failed
        with pytest.raises(RuntimeError, match="reload a checkpoint"):
            backend.optim_step(name, None)


def test_backend_ties_kv_before_adam_and_rejects_untied_restore(monkeypatch):
    import pytest
    from skyrl.backends.lora_init import RepeatedKVHeads
    monkeypatch.delenv("TUNIX_GRADIENT_PROBE_DIR", raising=False)
    monkeypatch.delenv("TUNIX_REPLAY_DIAGNOSTICS", raising=False)

    class Key(nnx.Module):
        def __init__(self):
            self.kernel_lora_a = nnx.LoRAParam(jnp.zeros((4, 2)))
            self.kernel_lora_b = nnx.LoRAParam(jnp.zeros((2, 4)))

    class Tiny(nnx.Module):
        def __init__(self):
            self.key = Key()

    backend = actual_backend_methods()()
    model = Tiny()
    template = SimpleNamespace(model=model, lora_shape=nnx.state(model, nnx.LoRAParam))
    backend.models = {}
    backend.templates = {(2, 2.0, True, True): template}
    backend._get_template = lambda cfg: template
    backend.base_state = None
    backend.config = SimpleNamespace(max_lora_rank=2, free_base_state_after_template=True,
                                     independent_lora_init=True, checkpoint_mirror_gcs="")
    backend._repeated_kv_heads = RepeatedKVHeads(2, 4, 1)
    backend.create_model("a", types.LoraConfig(rank=2, alpha=2., seed=1))
    slot = backend.models["a"]
    slot.accum_grads = jax.tree_util.tree_map_with_path(
        lambda path, x: jnp.broadcast_to(jnp.arange(1, 5, dtype=jnp.float32), x.shape)
        if backend._repeated_kv_heads.is_kv_b(path) else jnp.zeros_like(x), slot.lora_state)
    slot.accum_count = 1
    result = backend.optim_step("a", SimpleNamespace(adam_params=types.AdamParams(
        learning_rate=0.01, beta1=0.9, beta2=0.999, eps=0.1, weight_decay=0.0)))
    expected = -0.01 * np.array([3., 3., 7., 7.]) / np.array([3.1, 3.1, 7.1, 7.1])
    np.testing.assert_allclose(slot.lora_state["key"]["kernel_lora_b"][...],
                               np.broadcast_to(expected, (2, 4)), rtol=2e-5)
    np.testing.assert_allclose(result.metrics["skyrl.ai/grad_norm"], np.sqrt(116.), rtol=1e-6)
    old_state = slot.lora_state
    bad = jax.tree_util.tree_map_with_path(
        lambda path, x: x.at[..., 1].add(1) if backend._repeated_kv_heads.is_kv_b(path) else x,
        old_state)
    backend._read_checkpoint_archive = lambda path: {"lora_weights": bad}
    backend._state_from_flat = lambda reference, flat, layouts: flat
    with pytest.raises(ValueError, match="divergent replicas"):
        backend.load_checkpoint(Path("unused-checkpoint"), "a")
    assert slot.lora_state is old_state


def test_gptoss_expert_initialization_uses_input_width_and_preserves_sharding():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from skyrl.backends.lora_init import initialize_lora
    mesh = Mesh(np.array(jax.devices()), ('tp',))
    placement = NamedSharding(mesh, P('tp', None, None, None))
    reference = {
        'wo_lora_a': jax.device_put(jnp.zeros((8, 2, 512, 4)), placement),
        'wi_0_lora_b': jax.device_put(jnp.ones((8, 2, 8, 32)), placement),
    }
    a, b = [initialize_lora(reference, seed) for seed in (1, 2)]
    same = initialize_lora(reference, 1)
    np.testing.assert_array_equal(a['wo_lora_a'], same['wo_lora_a'])
    assert not np.array_equal(a['wo_lora_a'], b['wo_lora_a'])
    np.testing.assert_allclose(np.std(np.asarray(a['wo_lora_a'])), 1 / np.sqrt(512), rtol=.03)
    assert a['wo_lora_a'].sharding == placement
    assert a['wi_0_lora_b'].sharding == placement
    np.testing.assert_array_equal(a['wi_0_lora_b'], 0)
    np.testing.assert_array_equal(reference['wi_0_lora_b'], 1)
