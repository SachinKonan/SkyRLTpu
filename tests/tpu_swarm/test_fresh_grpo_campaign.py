import json
from dataclasses import replace
from pathlib import Path

import pytest

from tpu.science.training_setup import split_roles
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.commands import client_environment

PROFILES = sorted(p for p in Path('tpu/swarm/ray_train/profiles').glob('fresh-*-20260919.json')
                  if '-rglru-' not in p.name)


@pytest.mark.parametrize('profile', PROFILES, ids=lambda p:p.stem)
def test_fresh_campaign_contract(profile):
    c = Config.load(profile)
    v5p = c.accelerator == 'tpu-v5p-32'
    assert (c.hosts, c.trainer.hosts, c.inference_hosts) == ((4, 1, 3) if v5p else (8, 4, 4))
    assert c.bootstrap_max_drafts == 1024 and c.bootstrap_target_valid == 512
    assert c.bootstrap_group_size == 16 and c.bootstrap_max_groups == 32
    assert c.bootstrap_layers == 1 and c.bootstrap_all_hosts and not c.bootstrap_only
    assert not c.seed_pool_sha256 and c.resume_min_checkpoint_step == 0
    assert c.systemd_runtime and c.checkpoint_resume
    assert c.inference.tp == 4 and c.inference.max_sequences == 16
    assert len(c.engine_slots([f'host{i}' for i in range(c.hosts)])) == c.hosts
    assert len(c.engine_slots([f'host{i}' for i in range(c.trainer.hosts,c.hosts)])) == c.inference_hosts
    assert c.client_context_window == 22528 and c.client_phase1_max_tokens == 16384
    env = client_environment(c, Path('/runtime'), 'head')
    assert env['TTD_ADV_ESTIMATOR'] == 'mean_baseline' and env['TTD_LOSS_FN'] == 'importance_sampling'
    assert (env['GROUPS_PER_BATCH'], env['GROUP_SIZE'], env['NUM_EPOCHS']) == ('16', '32', '15')
    family = env['TTD_ANSWER_MODEL_FAMILY']
    assert float(c.client_learning_rate) == (1.5e-4 if family == 'qwen' else 4e-5)
    for key in ('trainer_compile', 'inference_compile'):
        assert c.run_id in getattr(c.cache, key)
        assert c.run_id not in getattr(c.cache, key + '_seed')
    if c.science_task:
        train, infer, grader = split_roles(c.science_task, list(range(c.trainer.hosts)),
            list(range(c.trainer.hosts,c.hosts)), accelerator=c.accelerator,
            placement_backend=c.science_placement_backend)
        assert len(train) == c.trainer.hosts and len(infer) == c.inference_hosts and grader is None
        if c.science_task == 'routing':
            assert env['SCIENCE_ROUTING_SUITE'] == 'full' and c.science_routing_slots_per_host == 16
        else:
            assert env['SCIENCE_PLACEMENT_SUITE'] == 'ibm17-proxy-v1'
            assert env['SCIENCE_PLACEMENT_HELPER'] == 'fast_proxy_v1'
            assert c.science_placement_slots_per_host == 16 and c.science_placement_backend == 'cpu'
    elif env['TTD_ENV'] == 'circle_packing':
        assert env['TTD_PROBLEM_TYPE'] == '26'
    else:
        assert env['TTD_ENV'] == 'ac_inequalities' and env['TTD_PROBLEM_TYPE'] == 'ac2'
    if v5p:
        assert (c.trainer.tp,c.trainer.fsdp) == ((4,1) if family == 'gemma' else (1,4))
        with pytest.raises(ValueError):
            replace(c, trainer=replace(c.trainer, hosts=2)).validate()


def test_unique_twenty_four_configs_and_cache_destinations():
    assert len(PROFILES) == 24
    configs = [Config.load(p) for p in PROFILES]
    assert len({c.run_id for c in configs}) == 24
    assert len({getattr(c.cache,k) for c in configs for k in ('trainer_compile','inference_compile')}) == 48
    rows = json.loads(Path('tpu/science/results/fresh-grpo-campaign-20260919/jobs.json').read_text())['jobs']
    assert len(rows) == 33
    assert all(r['priority'] == (100 if r['task'] in ('ac2','cp26') else 50) for r in rows)
