"""Analytic checks for gradient geometry and the frozen backend drain."""

import ast
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skyrl.backends.gradient_probe import compare, export_probe, fingerprint

_prune_spec = importlib.util.spec_from_file_location(
    'gradient_prune_canary', Path(__file__).parents[2]/'tpu/gradient_conflict/prune_canary_weights.py')
_prune_module = importlib.util.module_from_spec(_prune_spec)
_prune_spec.loader.exec_module(_prune_module)
prune = _prune_module.prune


def test_reclaim_canary_weights_preserves_results_and_runtime(tmp_path):
    root = tmp_path/'ray-serve-canary-v4-64-v3'
    model = root/'model'
    model.mkdir(parents=True)
    weight = model/'model-00001.safetensors'
    weight.write_bytes(b'weights')
    (root/'model-ready').write_text('gs://pinned-manifest')
    protected = ['events.jsonl', 'serve.log', 'venv/python', 'code/client.py',
                 'xla/compiled', 'model/tokenizer.json', 'model/model.safetensors.index.json']
    for name in protected:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('keep')
    assert prune(tmp_path) == (1, 7)
    assert not weight.exists() and not (root/'model-ready').exists()
    assert all((root/name).read_text() == 'keep' for name in protected)
    assert prune(tmp_path) == (0, 0)


def test_reclaim_rejects_canary_cache_symlink(tmp_path):
    shared = tmp_path/'shared-model'
    shared.mkdir()
    weight = shared/'weights.safetensors'
    weight.write_bytes(b'keep')
    root = tmp_path/'ray-serve-canary-v4-64-v3'
    root.mkdir()
    (root/'model').symlink_to(shared, target_is_directory=True)
    with pytest.raises(ValueError, match='linked canary cache'):
        prune(tmp_path)
    assert weight.read_bytes() == b'keep'


def _cache_class(run):
    path = Path(__file__).parents[2] / 'tpu/gradient_conflict/run.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'DurableCache')
    ns = {'Path': Path, 'hashlib': hashlib, 'json': json, 'gzip': gzip,
          'subprocess': SimpleNamespace(run=run)}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), ns)
    return ns['DurableCache']


def test_resume_cache_is_bound_to_exact_input_bytes(tmp_path):
    source = tmp_path/'input.gz'
    source.write_bytes(b'first input')
    cls = _cache_class(None)
    first = cls(source, 'gs://experiment')
    assert cls(source, 'gs://experiment').remote == first.remote
    source.write_bytes(b'different input')
    assert cls(source, 'gs://experiment').remote != first.remote


def test_resume_cache_distinguishes_missing_artifact_from_access_failure(tmp_path):
    source = tmp_path/'input.gz'
    source.write_bytes(b'input')
    response = SimpleNamespace(returncode=1, stdout='', stderr='The following URLs matched no objects or files')
    cache = _cache_class(lambda *a, **kw: response)(source, 'gs://experiment')
    assert cache.fetch('missing.json') is None
    response.stderr = '403 permission denied'
    with pytest.raises(RuntimeError, match='403'):
        cache.fetch('inaccessible.json')


def test_cached_reference_scores_round_trip_without_precision_loss(tmp_path):
    source = tmp_path/'input.gz'
    source.write_bytes(b'input')
    calls = []
    cache = _cache_class(lambda *a, **kw: calls.append(a))(source, 'gs://experiment')
    scores = [[-0.000000012345678, -123.456789], [-3.25]]
    cache.save_scores('scores/g/000.json.gz', scores)
    with gzip.open(cache.local/'scores/g/000.json.gz', 'rt') as stream:
        assert json.load(stream) == scores
    assert calls[0][0][-1] == cache.remote+'/scores/g/000.json.gz'


def test_opposition_and_population_scaling():
    a = {'x': np.array([3., 4.])}
    b = {'x': np.array([-6., -8.])}
    result = compare(a, b, 1, 2)
    assert result['cosine'] == pytest.approx(-1)
    assert result['left_mean_norm'] == result['right_mean_norm'] == 5
    assert result['right_to_left_norm'] == 2
    assert result['cancellation_ratio'] == pytest.approx(1 / 3)
    assert result['combined_dot_left'] < 0


def test_zero_gradients_have_undefined_cosine():
    result = compare({'x': np.zeros(3)}, {'x': np.ones(3)}, 0, 3)
    assert result['cosine'] is None
    assert result['left_mean_norm'] is None
    assert result['cancellation_ratio'] == 1


def test_export_preserves_sum_and_parameter_values(tmp_path):
    gradients = {'layer': np.array([2., -4.], dtype=np.float32)}
    params = {'layer': np.array([1., 3.], dtype=np.float32)}
    before = fingerprint(params)
    record = export_probe(tmp_path, 'm', 0, gradients, params, 2)
    assert record['sum_norm'] == pytest.approx(np.sqrt(20))
    assert record['mean_norm'] == pytest.approx(np.sqrt(5))
    assert record['parameter_sha256'] == before == fingerprint(params)
    with np.load(tmp_path / 'm/000000.npz') as loaded:
        np.testing.assert_array_equal(loaded['layer'], gradients['layer'])


def test_nonfinite_gradient_rejected(tmp_path):
    with pytest.raises(ValueError, match='Non-finite'):
        export_probe(tmp_path, 'm', 0, {'x': np.array([np.nan])}, {}, 1)


def _optim_method(process_index=0):
    # Exercise the actual method without importing TPU-only runtime packages.
    path = Path(__file__).parents[2] / 'skyrl/backends/tunix_backend.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TunixBackend')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'optim_step')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    ns = {'os': __import__('os'), 'jax': SimpleNamespace(process_index=lambda: process_index),
          'types': SimpleNamespace(OptimStepOutput=lambda **kw: SimpleNamespace(**kw))}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
    return ns['optim_step']


def test_probe_drain_never_accesses_optimizer_or_changes_parameters(tmp_path, monkeypatch):
    monkeypatch.setenv('TUNIX_GRADIENT_PROBE_DIR', str(tmp_path))
    params = {'x': np.array([1., 2.])}
    # No optimizer/template attributes: touching either must fail this test.
    slot = SimpleNamespace(accum_count=2, accum_grads={'x': np.array([6., 8.])},
                           lora_state=params, diagnostic_grad_index=0)
    backend = SimpleNamespace(models={'m': slot}, _flat_numpy=lambda x:x,
                              _host_global_norm=lambda x: np.linalg.norm(x['x']))
    request = SimpleNamespace(adam_params=SimpleNamespace(learning_rate=0, weight_decay=0))
    output = _optim_method()(backend, 'm', request)
    assert output.metrics['skyrl.ai/grad_norm'] == 5
    assert slot.accum_count == 0 and slot.accum_grads is None
    assert slot.diagnostic_grad_index == 1
    np.testing.assert_array_equal(params['x'], [1., 2.])
    request.adam_params.learning_rate = 1e-4
    with pytest.raises(ValueError, match='zero learning rate'):
        _optim_method()(backend, 'm', request)


@pytest.mark.parametrize('process_index,writer,expected', [(2, '1', True), (0, '0', False)])
def test_export_writer_follows_api_host_with_permuted_jax_ranks(tmp_path, monkeypatch, process_index, writer, expected):
    monkeypatch.setenv('TUNIX_GRADIENT_PROBE_DIR', str(tmp_path))
    monkeypatch.setenv('TUNIX_GRADIENT_PROBE_WRITER', writer)
    slot = SimpleNamespace(accum_count=1, accum_grads={'x': np.ones(2)},
                           lora_state={'x': np.zeros(2)}, diagnostic_grad_index=0)
    backend = SimpleNamespace(models={'m': slot}, _flat_numpy=lambda x:x,
                              _host_global_norm=lambda x: np.linalg.norm(x['x']))
    request = SimpleNamespace(adam_params=SimpleNamespace(learning_rate=0, weight_decay=0))
    _optim_method(process_index)(backend, 'm', request)
    assert (tmp_path/'m/000000.json').is_file() == expected
    assert slot.accum_grads is None and slot.accum_count == 0


def test_signed_score_gradient_partition_matches_finite_difference():
    # Same sequence-mean signed objective as the probe, with references fixed
    # at theta0. Its derivative is sum(-adv * dlogp / sequence_length).
    features = np.array([[1., 2.], [-3., 1.], [2., -1.], [1., -2.]])
    advantage = np.array([2., -1., 3., -4.])
    lengths = np.array([3., 7., 5., 9.])
    theta0 = np.array([.3, -.2])
    analytic = -advantage[:,None] * features / lengths[:,None]
    def loss(theta):
        return np.sum(-advantage * np.exp(features @ (theta-theta0)) / lengths)
    numeric = np.array([(loss(theta0+np.eye(2)[i]*1e-6)-loss(theta0-np.eye(2)[i]*1e-6))/2e-6 for i in range(2)])
    np.testing.assert_allclose(analytic.sum(0), numeric, rtol=1e-8)
    np.testing.assert_allclose(analytic[advantage>0].sum(0)+analytic[advantage<0].sum(0), numeric, rtol=1e-8)
