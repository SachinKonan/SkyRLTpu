"""CPU placement admission, topology and feedback contracts."""
import asyncio
from concurrent.futures import Future
from dataclasses import replace
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.bootstrap import workload_resources
from tpu.science.training_setup import split_roles, check_references

PROFILE = 'tpu/swarm/ray_train/profiles/science-placement-v6e-muse-cpu-seeded-train-001.json'


def test_cpu_profile_keeps_four_inference_hosts_and_limits_admission():
    c = Config.load(PROFILE)
    c.validate()
    assert c.inference_hosts == 4
    assert c.ray_cpus_per_host == 72
    assert workload_resources(c, 0) == {'TPU': 4, 'placement_cpu_host': 16}
    assert split_roles('placement', [6, 0, 7, 2], [1, 3, 4, 5], placement_backend='cpu') == (
        [6, 0, 7, 2], [1, 3, 4, 5], None)
    with pytest.raises(ValueError):
        replace(c, science_placement_slots_per_host=17).validate()
    with pytest.raises(ValueError):
        replace(c, cache=replace(c.cache, trainer_gib=256)).validate()


def test_reference_gate_requires_every_host_case_and_cpu_device():
    from tpu.science.challenge_contract import CASES
    rows = [dict(correctness=1, reward=.5, metrics=dict(
        ray_node_id=str(host),case=case,physical_tpu_chips=0,hard_memory_gib=8,
        hard_cpus=[16,17,18,19],candidate_device={'device_kind':'cpu'}))
        for host in range(8) for _ in range(2) for case in CASES]
    check_references('placement', rows, placement_backend='cpu')
    rows[-1]['metrics']['ray_node_id']='0'
    with pytest.raises(RuntimeError):
        check_references('placement', rows, placement_backend='cpu')
    rows[-1]['metrics']['ray_node_id']='7'
    rows[-1]['metrics']['candidate_device']['device_kind']='TPU v6e'
    with pytest.raises(RuntimeError):
        check_references('placement', rows, placement_backend='cpu')


def test_cpu_dispatch_never_requests_a_tpu_placement_group():
    from tpu.science import training_env as env, placement_ray, challenge_contract
    class Ref:
        def future(self):
            f=Future(); f.set_result({'reward':.5,'raw_score':.5}); return f
    with patch.object(env, 'connect'), patch.dict(os.environ,
            SCIENCE_WORKER_ROOT='/payload', SCIENCE_PLACEMENT_BACKEND='cpu',
            SCIENCE_PLACEMENT_SLOTS_PER_HOST='16'), \
            patch.object(placement_ray.grade_cpu_case, 'options') as opts, \
            patch('ray.util.placement_group.get_placement_group') as pg, \
            patch.object(env.ray,'cancel'), \
            patch.object(challenge_contract,'aggregate',return_value={'reward':.5,'raw_score':.5}):
        opts.return_value.remote.side_effect=lambda *a,**k: Ref()
        assert asyncio.run(env.evaluate('placement','candidate',1200))['reward']==.5
        assert opts.return_value.remote.call_count==4
        for call in opts.return_value.remote.call_args_list:
            assert call.kwargs==dict(admission_timeout_s=1200,slots_per_host=16)
        pg.assert_not_called()
        prompt=env.task_prompt('placement')
        assert 'CPU only' in prompt and '8 GiB' in prompt
        assert 'Allowed imports: numpy, scipy, networkx, jax' in prompt
        assert 'exclusive physical TPU' not in prompt
    options=placement_ray.grade_cpu_case._default_options
    assert options['num_cpus']==4 and options['memory']==8*1024**3
    assert options['resources']=={'placement_cpu_host':1}
