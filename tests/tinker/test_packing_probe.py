import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.backends.packing_probe import verify_once
from skyrl.backends.tunix_backend import TunixBackend
from skyrl.tinker import types
from skyrl.tinker.loss_fns import LossFnConfig


def fixture():
    return types.PreparedModelPassBatch(
        all_model_inputs=[types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2])])] * 4,
        all_targets=[[2, 3]] * 4, all_token_weights=[[1., 1.]] * 4,
        all_sampling_logprobs=[[-1., -2.]] * 4, all_advantages=[[1., -1.]] * 4,
        all_values=[[]] * 4, all_returns=[[]] * 4, all_model_ids=['m'] * 4,
        all_loss_fns=['importance_sampling'] * 4, all_loss_fn_configs=[None] * 4,
        request_batch_slices=[('r', 'm', 0, 4)])


def test_actual_loss_gradient_is_independent_of_padding_and_grouping():
    x = jnp.arange(24, dtype=jnp.float32).reshape(4, 6) / 30
    weights = jnp.array([[1,1,1,0,0,0], [1,1,1,1,1,0], [1,0,0,0,0,0], [1,1,1,1,1,1]], dtype=jnp.float32)
    advantages = jnp.array([1., -.8, .3, -1.])[:, None] * jnp.ones_like(x)
    behavior = -jnp.ones_like(x)
    cfg = LossFnConfig(clip_low_threshold=jnp.full((4,), .8), clip_high_threshold=jnp.full((4,), 1.2))
    def loss(theta, enabled):
        mask = weights * enabled[:, None]
        return TunixBackend._loss_from_logits(
            None, jnp.zeros_like(x, dtype=jnp.int32), mask,
            jnp.full((4,), types.LOSS_TYPES['importance_sampling']), behavior,
            advantages, cfg, target_logprobs=-1 + theta * x)[0]
    packed = jax.grad(loss)(jnp.float32(.2), jnp.ones(4)) / 4
    sequential = sum(jax.grad(loss)(jnp.float32(.2), jnp.eye(4)[i]) for i in range(4)) / 4
    np.testing.assert_allclose(packed, sequential, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('corrupt', [False, True])
def test_probe_restores_accumulation_and_never_updates_weights(monkeypatch, tmp_path, corrupt):
    path = tmp_path / 'probe.json'
    monkeypatch.setenv('TUNIX_PACKING_PROBE_REPORT', str(path))
    monkeypatch.setenv('TUNIX_PACKING_PROBE_BASE_BUDGET', '22528')
    monkeypatch.setenv('TUNIX_SEQ_BUCKETS', '18432,22528')
    monkeypatch.setenv('TUNIX_UNIFORM_SEQ_LEN', '0')
    slot = SimpleNamespace(accum_grads=None, accum_count=0, template_key='t')
    backend = SimpleNamespace(config=SimpleNamespace(train_token_budget=90112), models={'m': slot},
                              templates={'t': SimpleNamespace(kind='maxtext')},
                              _round_seq_len=lambda n, kind: n, _row_shard=lambda: 4)
    calls = []
    def model_pass(batch, with_grads):
        verify_once(backend, batch)  # Recursion guard must work.
        calls.append(backend.config.train_token_budget)
        assert batch.request_batch_slices == [('packing_probe', 'm', 0, 4)]
        slot.accum_count = 4
        slot.accum_grads = {'a': jnp.array([1., 2.]) * (2 if corrupt and calls[-1] == 90112 else 1)}
    backend._model_pass = model_pass
    batch = fixture()
    short = batch.model_copy(update={key: value[:3] for key, value in batch.model_dump().items()
                                     if key.startswith('all_') and key != 'all_model_inputs'} |
                                   {'all_model_inputs': batch.all_model_inputs[:3]})
    verify_once(backend, short)
    assert not calls and backend.config.train_token_budget == 22528
    if corrupt:
        with pytest.raises(RuntimeError, match='comparison failed'):
            verify_once(backend, fixture())
    else:
        verify_once(backend, fixture())
    assert slot.accum_grads is None and slot.accum_count == 0
    assert calls == [22528, 22528, 90112, 90112]
    assert backend.config.train_token_budget == (22528 if corrupt else 90112)
    assert json.loads(path.read_text())['state'] == ('failed' if corrupt else 'passed')
    verify_once(backend, fixture())
    assert len(calls) == 4


def test_probe_is_disabled_without_explicit_environment(monkeypatch):
    monkeypatch.delenv('TUNIX_PACKING_PROBE_REPORT', raising=False)
    verify_once(None, None)
