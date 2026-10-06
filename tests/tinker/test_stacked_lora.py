"""Execute the production NNX/FLCE backward under adapter-axis batching."""
import ast
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from skyrl.backends.stacked_lora import stack_states, stacked_model, unstack_state
from skyrl.tinker.loss_fns import LOSS_FUNCTIONS, LossFnConfig
from skyrl.tinker import types
from skyrl.tinker.types import LOSS_TYPES
from skyrl.backends.renderer import render_model_input
from skyrl.backends.utils import pad_batch, pad_to_fsdp
from jax.experimental import multihost_utils


def test_gradient_comparison_reports_direction_and_largest_leaf():
    from skyrl.backends.stacked_lora import gradient_comparison
    actual = {'changed': jnp.array([3., 4.]), 'same': jnp.array([1.])}
    expected = {'changed': jnp.array([3., -4.]), 'same': jnp.array([1.])}
    report = gradient_comparison(actual, expected)
    assert report['error_norm'] == 8.
    assert report['actual_norm'] == pytest.approx(np.sqrt(26))
    assert report['expected_norm'] == pytest.approx(np.sqrt(26))
    assert report['cosine'] == pytest.approx(-6 / 26)
    assert 'changed' in report['largest_error_leaves'][0]['path']


def backend_functions():
    module = ast.parse(Path('skyrl/backends/tunix_backend.py').read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'TunixBackend')
    names = {'_build_model_pass_fns_nnx', '_flce_target_logprobs', '_loss_from_logits',
             '_model_pass', 'forward_backward', 'forward_backward_multi_lora', '_round_seq_len',
             '_build_loss_fn_config', '_positions_and_masks', '_row_shard', '_shard_batch_arrays',
             '_micro_batch_size', '_jit_timing_context', '_pass_lora_state'}
    cls = ast.ClassDef(name='Backend', bases=[], keywords=[], decorator_list=[],
                      body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names])
    scope = dict(nnx=nnx, jax=jax, jnp=jnp, np=np, Callable=Callable, LOSS_FUNCTIONS=LOSS_FUNCTIONS,
                 ModelSlot=SimpleNamespace, types=types, LOSS_TYPES=LOSS_TYPES, LossFnConfig=LossFnConfig, os=os, time=time,
                 contextmanager=contextmanager, logger=logging.getLogger(__name__),
                 render_model_input=render_model_input, pad_batch=pad_batch, pad_to_fsdp=pad_to_fsdp,
                 multihost_utils=multihost_utils, _MAXTEXT_SEQ_BLOCK=512,
                 _DEFAULT_PPO_CLIP_LOW_THRESHOLD=.8, _DEFAULT_PPO_CLIP_HIGH_THRESHOLD=1.2)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), 'production_backward', 'exec'), scope)
    return scope['Backend']()


class Tiny(nnx.Module):
    def __init__(self, seed, sharding):
        def place(x):
            return jax.device_put(x, sharding)
        self.embedding = nnx.Param(place(jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 32))
        self.kernel_lora_a = nnx.LoRAParam(place(jax.random.normal(jax.random.key(seed), (4, 2)) / 8))
        self.kernel_lora_b = nnx.LoRAParam(place(jax.random.normal(jax.random.key(seed + 10), (2, 4)) / 8))

        self.kernel_lora_a.set_metadata(out_sharding=(None, None))
        self.kernel_lora_b.set_metadata(out_sharding=(None, None))

    def __call__(self, ids, positions, cache, mask, skip_lm_head=True):
        x = self.embedding[ids]
        # Multiple layers: adapter-dependent activations, including sown scratch.
        for _ in range(2):
            x = jnp.tanh(x + x @ self.kernel_lora_a[...] @ self.kernel_lora_b[...])
        self.sow(nnx.Intermediate, 'hidden_states', x)
        return x, None

    def logits_from_hidden(self, hidden):
        return hidden @ self.embedding[...].T


@pytest.mark.parametrize('tile', [2, 3])
def test_stacked_matches_sequential_gradients_and_independent_adam(tile):
    mesh = Mesh(np.array(jax.devices()), ('fsdp',))
    replicas = NamedSharding(mesh, P())
    model = Tiny(1, replicas)
    states = [nnx.state(Tiny(seed, replicas), nnx.LoRAParam) for seed in (1, 2)]
    base_before = np.array(model.embedding[...])
    batch = stacked_model(model, states)
    assert batch.embedding[...] is model.embedding[...]
    backend = backend_functions()
    backend.config = SimpleNamespace(enforce_eager=False, flce_tile_size=tile)
    sequential, _ = backend._build_model_pass_fns_nnx()
    stacked, _ = backend._build_model_pass_fns_nnx(stacked=True)
    rows = len(jax.devices())
    ids = jax.device_put(jnp.tile(jnp.array([[0, 1, 2, 3]]), (rows, 1)), NamedSharding(mesh, P('fsdp')))
    # Last row is padding; it must contribute no gradient.
    mask = jnp.ones_like(ids, dtype=jnp.float32).at[-1].set(0)
    args = (ids, ids, jnp.ones((rows, 4, 4), bool), (ids + 1) % 8, mask,
            jnp.full((rows,), len(LOSS_FUNCTIONS)-1, dtype=jnp.int32),
            jnp.full(ids.shape, -2.), jnp.tile(jnp.array([[1., -1., 2., -2.]]), (rows, 1)),
            LossFnConfig(jnp.zeros(rows), jnp.full((rows,), 2.)))
    # Use the actual CISPO loss index, independently of registry order.
    from skyrl.tinker.types import LOSS_TYPES
    args = (*args[:5], jnp.full((rows,), LOSS_TYPES['cispo'], dtype=jnp.int32), *args[6:])
    accum = stack_states([jax.tree.map(jnp.zeros_like, s) for s in states])
    for _ in range(2):
        losses, logps, accum = stacked(batch, accum, args)
    assert not nnx.to_flat_state(nnx.state(batch, nnx.Intermediate))
    for index, state in enumerate(states):
        nnx.update(model, state)
        expected = jax.tree.map(jnp.zeros_like, state)
        for _ in range(2):
            expected_loss, expected_lp, expected = sequential(model, expected, *args)
            nnx.pop(model, nnx.Intermediate)
        actual = unstack_state(accum, index, state)
        np.testing.assert_allclose(losses[index], expected_loss, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(logps[index], expected_lp, rtol=1e-5, atol=1e-6)
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        # Different hyperparameters and clipping, separate moment trees.
        tx = optax.chain(optax.clip_by_global_norm(.1 + index), optax.adam(1e-3 * (index + 1)))
        updates, moments = tx.update(actual, tx.init(state), state)
        ref_updates, ref_moments = tx.update(expected, tx.init(state), state)
        for a, b in zip(jax.tree.leaves((updates, moments)), jax.tree.leaves((ref_updates, ref_moments)), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(batch.embedding[...], base_before)


def test_production_microbatch_packing_and_adapter_output_routing(monkeypatch):
    monkeypatch.setenv('TUNIX_ROW_SHARD', '2')
    monkeypatch.setenv('TUNIX_UNIFORM_SEQ_LEN', '128')
    monkeypatch.setenv('TUNIX_MINIMAL_FB_OUTPUT', '0')
    from skyrl.tinker.engine import prepare_model_pass_batch
    mesh = Mesh(np.array(jax.devices()).reshape(2, -1), ('fsdp', 'tensor'))
    replicas = NamedSharding(mesh, P())
    model = Tiny(1, replicas)
    backend = backend_functions()
    backend.config = SimpleNamespace(enforce_eager=False, flce_tile_size=32, train_token_budget=256,
                                     train_micro_batch_size=2, stacked_lora_verify=True)
    backend._mesh = mesh
    backend._with_oom_recovery = lambda fn, label: fn()
    backend.metrics = SimpleNamespace(train_seq_len_jit_times={})
    fb, fwd = backend._build_model_pass_fns_nnx()
    backend.templates = {'same': SimpleNamespace(model=model, kind='maxtext', forward_backward_fn=fb, forward_fn=fwd)}
    backend.models = {name: SimpleNamespace(lora_state=nnx.state(Tiny(seed, replicas), nnx.LoRAParam),
                      accum_grads=None, accum_count=0, training_failed=False, template_key='same', mix=None)
                      for name, seed in [('a', 1), ('b', 2)]}
    data = types.ForwardBackwardInput(data=[types.Datum(
        model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[i, 1, 2, 3])]),
        loss_fn_inputs=types.LossFnInputs(target_tokens=types.TensorData(data=[1, 2, 3, 4]),
            weights=types.TensorData(data=[1., 1., 1., 1.]), advantages=types.TensorData(data=[1., -1., 2., -2.]),
            logprobs=types.TensorData(data=[-2., -2., -2., -2.]))) for i in range(3)],
        loss_fn='cispo', loss_fn_config={'clip_low_threshold': 0., 'clip_high_threshold': 2.})
    shared = prepare_model_pass_batch({'shared': ('a', data)})
    actual = backend.forward_backward_multi_lora(shared, ['a', 'b'])
    saved = {name: slot.accum_grads for name, slot in backend.models.items()}
    assert set(actual) == {'a', 'b'}
    assert all(output.metrics['stacked_replay_verified'] == 1 for output in actual.values())
    for name, slot in backend.models.items():
        assert slot.accum_count == 3  # final padded row must not count
        slot.accum_grads, slot.accum_count = None, 0
        ref = backend.forward_backward(prepare_model_pass_batch({'shared': (name, data)}))['shared']
        for a, b in zip(actual[name].loss_fn_outputs, ref.loss_fn_outputs, strict=True):
            for field in ('logprobs', 'elementwise_loss'):
                np.testing.assert_allclose(a[field]['data'], b[field]['data'], rtol=1e-5, atol=1e-6)
        for a, b in zip(jax.tree.leaves(saved[name]), jax.tree.leaves(slot.accum_grads), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
