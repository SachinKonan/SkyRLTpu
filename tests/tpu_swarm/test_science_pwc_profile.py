"""Science estimator interventions retain identical packaged training code."""
import copy
import json
from pathlib import Path
import pytest
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.overlay import ADAPTIVE_PWC_FILE, manifest

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / 'tpu/swarm/ray_train/profiles'

@pytest.mark.parametrize('name,helper', [
    ('science-qubit-q20-v6e-qwen-seeded-001', False),
    ('science-circuit-v4-qwen-cpu-seeded-train-001', True),
])
def test_both_estimators_use_identical_overlay(name, helper):
    control = json.loads((PROFILES / f'{name}.json').read_text())
    if helper:
        control['client_env']['SCIENCE_PLACEMENT_HELPER'] = 'fast_proxy_v1'
    candidate = copy.deepcopy(control)
    candidate['client_env'].update(TTD_ADV_ESTIMATOR='piecewise_valid_entropic_centered_adaptive',
        TTD_ADV_PIECEWISE_RHO='0.5', TTD_ADV_PIECEWISE_INVALID_REWARD='0')
    a, b = Config.from_dict(control), Config.from_dict(candidate)
    assert not a.has_adaptive_pwc_overlay and b.has_adaptive_pwc_overlay
    assert ADAPTIVE_PWC_FILE in manifest(ROOT, a)
    assert manifest(ROOT, a) == manifest(ROOT, b)

@pytest.mark.parametrize('task', ['routing', 'placement'])
def test_other_science_modes_are_not_implicitly_enabled(task):
    name = ('science-qubit-q20-v6e-qwen-seeded-001' if task == 'routing'
            else 'science-circuit-v4-qwen-cpu-seeded-train-001')
    candidate = json.loads((PROFILES / f'{name}.json').read_text())
    candidate['client_env']['TTD_ADV_ESTIMATOR'] = 'piecewise_valid_entropic_centered_adaptive'
    if task == 'routing':
        candidate['client_env']['SCIENCE_ROUTING_SUITE'] = 'full'
    with pytest.raises(ValueError, match='adaptive PWC'):
        Config.from_dict(candidate)


@pytest.mark.parametrize('model', ['qwen', 'gemma', 'muse'])
def test_full_routing_parallel_pwc_matches_regraded_control(model):
    control_name = f'qubit-v4-{model}-parallel2-20260921' + ('-r2' if model == 'qwen' else '')
    candidate_name = f'qubit-v4-{model}-parallel2-pwc-rho05-20260921'
    control = json.loads((PROFILES / f'{control_name}.json').read_text())
    candidate = json.loads((PROFILES / f'{candidate_name}.json').read_text())
    a, b = Config.from_dict(control), Config.from_dict(candidate)
    assert b.science_routing_evaluator == 'parallel-v2'
    assert b.client_env['SCIENCE_ROUTING_SUITE'] == 'full'
    assert b.client_env['NUM_EPOCHS'] == '10'
    assert b.client_env['GROUPS_PER_BATCH'] == '16'
    assert b.client_env['GROUP_SIZE'] == '32'
    assert b.seed_pool_sha256 == a.seed_pool_sha256
    assert manifest(ROOT, a) == manifest(ROOT, b)
    candidate['run_id'] = control['run_id']
    candidate['root'] = control['root']
    candidate['client_env']['TTD_SICK_MARKER'] = control['client_env']['TTD_SICK_MARKER']
    candidate['client_env']['TTD_ADV_ESTIMATOR'] = control['client_env']['TTD_ADV_ESTIMATOR']
    assert candidate['client_env'].pop('TTD_ADV_PIECEWISE_RHO') == '0.5'
    assert candidate['client_env'].pop('TTD_ADV_PIECEWISE_INVALID_REWARD') == '0'
    assert candidate == control


@pytest.mark.parametrize('model', ['qwen','gemma','muse'])
@pytest.mark.parametrize('task', ['q20','circuit'])
def test_approved_pair_changes_only_estimator_and_destinations(model, task):
    a_name=f'science-{task}-v4-{model}-grpo-20260918'
    b_name=f'science-{task}-v4-{model}-pwc-20260918'
    a=json.loads((PROFILES/f'{a_name}.json').read_text())
    b=json.loads((PROFILES/f'{b_name}.json').read_text().replace(b_name,a_name))
    assert b['client_env'].pop('TTD_ADV_PIECEWISE_RHO') == '0.5'
    assert b['client_env'].pop('TTD_ADV_PIECEWISE_INVALID_REWARD') == '0'
    b['client_env']['TTD_ADV_ESTIMATOR']='mean_baseline'
    assert a == b
    ca,cb=Config.load(PROFILES/f'{a_name}.json'),Config.load(PROFILES/f'{b_name}.json')
    assert manifest(ROOT,ca)==manifest(ROOT,cb)
    assert ca.accelerator=='tpu-v4-64' and ca.hosts==8 and ca.trainer.hosts==4
    assert ca.client_learning_rate=='4e-5'
    assert ca.client_env['GROUPS_PER_BATCH']=='16' and ca.client_env['GROUP_SIZE']=='32'
