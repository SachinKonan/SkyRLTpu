"""Full-suite objective, restored-state isolation, and v5p role contract."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tpu.science.challenge_contract import CASES, SUITE, aggregate
from tpu.science.feedback import observation
from tpu.science.placement_suite_guard import validate_restored_pools, validate_state
from tpu.science.rewards import valid
from tpu.science.training_setup import split_roles
from tpu.swarm.ray_train.bootstrap import workload_resources
from tpu.swarm.ray_train.commands import client_environment
from tpu.swarm.ray_train.config import Config


def rows():
    return [valid(.5, dict(case=c, proxy_cost=i + 1., overlap_count=0)) for i, c in enumerate(CASES)]


def test_full17_mean_and_invalid_suite_never_receive_partial_credit():
    assert len(CASES) == 17 and 'ibm09' in CASES and 'ibm05' not in CASES
    result = aggregate(rows())
    assert result['metrics']['mean_proxy_cost'] == 9.
    assert result['reward'] == result['raw_score'] == .1
    assert result['metrics']['leaderboard_verified'] is False
    for bad in (rows()[:4], rows()[:-1], rows()[:-1] + [rows()[0]]):
        assert aggregate(bad)['reward'] == 0
    for value in (float('nan'), float('inf'), -1):
        bad = rows(); bad[8]['metrics']['proxy_cost'] = value
        assert aggregate(bad)['reward'] == 0
    for field, value in [('overlap_count', 1), ('overlap_count', None), ('proxy_cost', None)]:
        bad = rows()
        if value is None:
            del bad[8]['metrics'][field]
        else:
            bad[8]['metrics'][field] = value
        assert aggregate(bad)['reward'] == 0
    bad = rows(); bad[8]['correctness'] = 0
    assert aggregate(bad)['reward'] == 0


def test_full17_feedback_survives_pool_restore_and_subset_pool_is_rejected(tmp_path):
    state = dict(code='pass', observation=observation('placement', aggregate(rows())))
    validate_state(state)
    path = tmp_path / 'puct_sampler_step_000002.json'
    path.write_text(json.dumps(dict(states=[state], initial_states=[dict(code='')])) )
    assert validate_restored_pools(tmp_path) == path
    feedback = json.loads(state['observation'])
    assert len(feedback['metrics']['cases']) == 17
    feedback['metrics']['cases'] = feedback['metrics']['cases'][:4]
    state['observation'] = json.dumps(feedback)
    path.write_text(json.dumps(dict(states=[state])))
    with pytest.raises(ValueError, match='regrade'):
        validate_restored_pools(tmp_path)


@pytest.mark.parametrize('model,mesh,engines', [('qwen',(1,4),3), ('muse',(1,4),6), ('gemma',(4,1),3)])
def test_v5p_profiles_preserve_native_settings_and_all17_cpu_contract(model, mesh, engines):
    profiles = Path('tpu/swarm/ray_train/profiles')
    c = Config.load(profiles / f'science-circuit-v5p-{model}-ibm17-helper-grpo-20260919.json')
    suffix = '-bwd256' if model == 'gemma' else ''
    native = Config.load(profiles / f'native-v5p-{model}-ac2-grpo-lr4e5-s1{suffix}-retryfix-004.json')
    assert c.trainer == native.trainer
    assert c.inference == native.inference
    assert c.hosts == 4 and c.trainer.hosts == 1 and c.inference_hosts == 3
    assert (c.trainer.tp,c.trainer.fsdp) == mesh
    assert len(c.engine_slots(['host1','host2','host3'])) == engines
    assert split_roles('placement', [0], [1,2,3], placement_backend='cpu', accelerator=c.accelerator) == ([0],[1,2,3],None)
    for rank in range(4):
        assert workload_resources(c,rank) == {'TPU':4, 'placement_cpu_host':16}
    env = client_environment(c,Path('/runtime'),'head')
    assert env['SCIENCE_PLACEMENT_SUITE'] == SUITE
    assert (env['GROUPS_PER_BATCH'],env['GROUP_SIZE']) == ('16','32')
    assert env['TTD_ADV_ESTIMATOR'] == 'mean_baseline'
    assert not c.seed_pool_sha256 and c.bootstrap_layers == 0
    assert c.cache.trainer_compile_seed == native.cache.trainer_compile
    assert c.cache.inference_compile_seed == native.cache.inference_compile
    assert c.run_id in c.cache.trainer_compile and c.run_id in c.cache.inference_compile
    assert c.client_context_window == 22528 and c.client_phase1_max_tokens == 16384
    # Do not silently admit TPU grading or a different host split on v5p.
    with pytest.raises(ValueError):
        replace(c, science_placement_backend='tpu').validate()
    with pytest.raises(ValueError):
        split_roles('placement',[0,1],[2,3],placement_backend='cpu',accelerator=c.accelerator)
