import json
from pathlib import Path
import numpy as np
import pytest
from tpu.science.fair_universe.contract import check_independence, checked_scores, bundle_identity
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.bootstrap import workload_resources


def profile():
    p = Path('tpu/swarm/ray_train/profiles/inference-farm-rq4-qwen38-east5a-v5p32-001.json')
    raw = json.loads(p.read_text())
    raw['inference_only_ranks'] = [0, 1]
    raw['grading'] = dict(families={'fair_universe': {}}, tpu_ranks=[2, 3],
                          dataset_root='/dev/shm/fair/data', candidate_python='/runtime/bin/python')
    return raw


def test_host_assignment_does_not_advertise_grading_on_inference_chips():
    config = Config.from_dict(profile())
    assert config.inference_hosts == 2
    assert workload_resources(config, 0) == {'TPU': 4}
    assert workload_resources(config, 1) == {'TPU': 4}
    assert workload_resources(config, 2) == {'TPU': 4, 'grading_fair_universe': 4}
    assert workload_resources(config, 3) == {'TPU': 4, 'grading_fair_universe': 4}


@pytest.mark.parametrize('ranks', [[0, 2], [2, 2], [], [4], [True]])
def test_reject_invalid_grading_hosts(ranks):
    raw = profile(); raw['grading']['tpu_ranks'] = ranks
    with pytest.raises(ValueError): Config.from_dict(raw)


def test_reject_more_slots_than_chips():
    raw = profile(); raw['grading']['families']['fair_universe']['slots_per_host'] = 5
    with pytest.raises(ValueError, match='one trial'): Config.from_dict(raw)


@pytest.mark.parametrize('field,value', [('cpus', 2), ('memory_gib', 8), ('memory_max_gib', 32)])
def test_reject_unaccounted_trial_resources(field, value):
    raw = profile(); raw['grading']['families']['fair_universe'][field] = value
    with pytest.raises(ValueError, match='trial contract'): Config.from_dict(raw)


def test_reject_bootstrap_reusing_grading_chips():
    raw = profile(); raw['bootstrap_all_hosts'] = True
    with pytest.raises(ValueError): Config.from_dict(raw)


def test_prediction_gate_rejects_batch_normalization():
    maps = np.arange(24.).reshape(4, 2, 3)
    assert check_independence(lambda x:x.mean(axis=(1, 2)), maps)['passed']
    with pytest.raises(ValueError, match='independence'):
        check_independence(lambda x:x.mean(axis=(1, 2))-x.mean(), maps)


def test_prediction_gate_rejects_stateful_score():
    calls = []
    def predict(x):
        calls.append(1); return x.mean(axis=(1,2))+len(calls)
    with pytest.raises(ValueError, match='independence'):
        check_independence(predict, np.ones((4, 2, 3)))


def test_fixed_shape_preserves_independence_gate():
    from types import SimpleNamespace
    from tpu.science.fair_universe.child import fixed_predict
    maps = np.arange(24., dtype=np.float32).reshape(4, 2, 3)
    shapes = []
    def independent(model, x, mask, config):
        shapes.append(x.shape)
        return x.mean(axis=(1, 2))
    candidate = SimpleNamespace(predict=independent)
    wrapper = lambda x: fixed_predict(candidate, None, x, None, {})
    assert check_independence(wrapper, maps)['passed']
    assert set(shapes) == {(32, 2, 3)}
    candidate.predict = lambda model,x,mask,config: x.mean(axis=(1,2))-x.mean()
    with pytest.raises(ValueError, match='independence'):
        check_independence(wrapper, maps)


def test_corrected_hybrid_features_do_not_mix_neighbor_maps():
    pytest.importorskip('jax')
    pytest.importorskip('sklearn')
    from tpu.science.fair_universe.candidates.hybrid_physics_axes_fixed import features
    maps = np.random.default_rng(42).normal(0, .03, (4, 32, 16)).astype(np.float32)
    mask = np.ones((32, 16), dtype=bool)
    before = features(maps, mask)[0]
    maps[1:] *= 5
    after = features(maps, mask)[0]
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('scores', [[float('nan')], [float('inf')], [[1.]], []])
def test_invalid_scores(scores):
    with pytest.raises(ValueError): checked_scores(scores, 1)


def test_cache_key_includes_seed_contract_and_candidate(tmp_path):
    candidate = tmp_path/'c.py'; candidate.write_text('x=1')
    original = bundle_identity(candidate, {'seed': 0}, {'split': 1})
    assert bundle_identity(candidate, {'seed': 1}, {'split': 1}) != original
    assert bundle_identity(candidate, {'seed': 0}, {'split': 2}) != original
    candidate.write_text('x=2')
    assert bundle_identity(candidate, {'seed': 0}, {'split': 1}) != original


def test_submission_exact_count_and_archive_contract(tmp_path):
    import zipfile
    from tpu.science.fair_universe_export import write_result
    with pytest.raises(ValueError):
        write_result([0.] * 9999, tmp_path/'short.zip')
    with pytest.raises(ValueError):
        write_result([float('nan')] * 10000, tmp_path/'nan.zip')
    result = write_result(np.arange(10000) / 10000., tmp_path/'submission.zip')
    assert result['count'] == 10000
    with zipfile.ZipFile(tmp_path/'submission.zip') as archive:
        assert archive.namelist() == ['result.json']
        payload = json.loads(archive.read('result.json'))
        assert list(payload) == ['ood_scores']
        assert len(payload['ood_scores']) == 10000
