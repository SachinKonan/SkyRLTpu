"""Direct circuit training starts from prompt roots, not a frozen seed stage."""
from pathlib import Path
from unittest.mock import patch
import pytest
from tpu.swarm.ray_train.config import Config

@pytest.mark.parametrize('model', ['qwen','gemma','muse'])
def test_direct_circuit_profile(model):
    c=Config.load(f'tpu/swarm/ray_train/profiles/science-circuit-v6e-{model}-helper-grpo-direct-20260918.json')
    assert c.accelerator=='tpu-v6e-32' and c.trainer.hosts==4 and c.inference_hosts==4
    assert c.bootstrap_layers==0 and not c.bootstrap_only and not c.bootstrap_all_hosts
    assert not c.seed_pool_sha256 and not c.resume_min_checkpoint_step
    assert c.systemd_runtime and c.checkpoint_resume
    assert c.client_env['SCIENCE_PLACEMENT_HELPER']=='fast_proxy_v1'
    assert c.client_env['TTD_ADV_ESTIMATOR']=='mean_baseline'
    assert 'TTD_ADV_PIECEWISE_RHO' not in c.client_env
    assert c.client_env['GROUPS_PER_BATCH']=='16' and c.client_env['GROUP_SIZE']=='32'
    assert c.client_env['NUM_EPOCHS']=='15'
    assert c.science_placement_backend=='cpu' and c.science_placement_slots_per_host==16


def test_first_batch_starts_with_helper_prompt_and_no_preselected_program(tmp_path):
    from tpu.science.training_env import PlacementTrainingEnv, candidate_prompt
    from ttt_discover.tinker_utils.sampler import PUCTSampler
    with patch.dict('os.environ',SCIENCE_PLACEMENT_BACKEND='cpu',SCIENCE_PLACEMENT_HELPER='fast_proxy_v1'):
        sampler=PUCTSampler(str(tmp_path/'puct_sampler.json'),env_type=PlacementTrainingEnv,
                            problem_type='placement',batch_size=16)
        assert len(sampler._states)==16
        assert all(not s.code for s in sampler._states)
        prompt=candidate_prompt('placement')
        assert 'Evaluator is already imported' in prompt
        assert 'Starting implementation (replace with your improved algorithm):' in prompt
        assert 'Previous candidate:' not in prompt

@pytest.mark.parametrize('model', ['qwen','gemma','muse'])
def test_q20_continues_with_original_grpo_estimator(model):
    c=Config.load(f'tpu/swarm/ray_train/profiles/science-q20-v4-{model}-grpo-clean-20260918.json')
    assert c.client_env['TTD_ADV_ESTIMATOR']=='mean_baseline'
    assert not any(k.startswith('TTD_ADV_PIECEWISE') for k in c.client_env)
    assert c.client_env['SCIENCE_ROUTING_SUITE']=='q20'
    assert c.resume_min_checkpoint_step==(0 if model=='muse' else 2)
    assert c.seed_pool_sha256 and c.systemd_runtime and c.checkpoint_resume

@pytest.mark.parametrize('model', ['qwen','gemma','muse'])
def test_additional_circuit_pwc_starts_fresh_and_preserves_grpo_control(model):
    import json
    grpo=f'science-circuit-v6e-{model}-helper-grpo-direct-20260918'
    pwc=f'science-circuit-v6e-{model}-helper-pwc-direct-20260918'
    p=Path('tpu/swarm/ray_train/profiles')
    a=json.loads((p/(grpo+'.json')).read_text())
    b=json.loads((p/(pwc+'.json')).read_text().replace(pwc,grpo))
    assert b['client_env']['TTD_ADV_ESTIMATOR']=='piecewise_valid_entropic_centered_adaptive'
    assert b['client_env'].pop('TTD_ADV_PIECEWISE_RHO')=='0.5'
    assert b['client_env'].pop('TTD_ADV_PIECEWISE_INVALID_REWARD')=='0'
    b['client_env']['TTD_ADV_ESTIMATOR']='mean_baseline'
    assert a==b
    for name in (grpo,pwc):
        c=Config.load(p/(name+'.json'))
        assert not c.seed_pool_sha256 and not c.resume_min_checkpoint_step and not c.bootstrap_layers

@pytest.mark.parametrize('model', ['qwen','gemma','muse'])
def test_q20_v6e_estimator_pair_reuses_only_original_bootstrap(model):
    import json
    p=Path('tpu/swarm/ray_train/profiles')
    grpo=f'science-q20-v6e-{model}-grpo-seedonly-20260918'
    pwc=f'science-q20-v6e-{model}-pwc-seedonly-20260918'
    original=Config.load(p/f'science-qubit-q20-v6e-{model}-seeded-001.json')
    a=json.loads((p/(grpo+'.json')).read_text())
    b=json.loads((p/(pwc+'.json')).read_text().replace(pwc,grpo))
    assert b['client_env']['TTD_ADV_ESTIMATOR']=='piecewise_valid_entropic_centered_adaptive'
    assert b['client_env'].pop('TTD_ADV_PIECEWISE_RHO')=='0.5'
    assert b['client_env'].pop('TTD_ADV_PIECEWISE_INVALID_REWARD')=='0'
    b['client_env']['TTD_ADV_ESTIMATOR']='mean_baseline'
    assert a==b
    for name in (grpo,pwc):
        c=Config.load(p/(name+'.json'))
        assert c.seed_pool_sha256==original.seed_pool_sha256
        assert c.seed_pool_sha256 and c.resume_min_checkpoint_step==0
        assert not c.bootstrap_layers and not c.bootstrap_only
        assert c.accelerator=='tpu-v6e-32' and c.systemd_runtime
        assert c.client_env['NUM_EPOCHS']=='15'
        assert c.client_env['SCIENCE_ROUTING_SUITE']=='q20'
        assert c.science_routing_slots_per_host==16
        assert c.run_gcs!=original.run_gcs
