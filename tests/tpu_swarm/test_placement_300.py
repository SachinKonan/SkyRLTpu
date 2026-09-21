import asyncio
from concurrent.futures import Future
from contextlib import ExitStack
from dataclasses import replace
import os
from unittest.mock import patch
import pytest
from tpu.science import placement_resources as r
from tpu.science.cpu_slots import acquire_slot
from tpu.swarm.ray_train.config import Config


def test_32_disjoint_slots_and_legacy_exclusion(tmp_path):
    with patch.object(r,'partition',return_value=(list(range(80,208)),list(range(80)))), ExitStack() as held:
        cpus=[]
        for i in range(32):
            slot,c,lock=r.acquire(32,root=tmp_path,deadline_seconds=.2)
            held.enter_context(lock);cpus+=c
        assert len(set(cpus))==128
        with pytest.raises(TimeoutError):r.acquire(32,root=tmp_path,deadline_seconds=.01)
        with pytest.raises(TimeoutError):acquire_slot(tmp_path,slots=16,deadline_seconds=.01)
    slot,lock=acquire_slot(tmp_path)
    with lock,patch.object(r,'partition',return_value=(list(range(128)),list(range(128,160)))):
        with pytest.raises(TimeoutError):r.acquire(32,root=tmp_path,deadline_seconds=.01)


def test_profiles_and_prompt():
    from tpu.science.training_env import task_prompt
    for model in ('gemma','qwen','muse'):
        c=Config.load(f'tpu/swarm/ray_train/profiles/circuit300-v5p32-{model}-10step-20260921.json')
        c.validate();assert c.ray_cpus_per_host==136 and c.hosts==4
        assert c.borrows_inference and c.client_env['NUM_EPOCHS']=='10'
        with pytest.raises(ValueError):replace(c,science_placement_slots_per_host=33).validate()
        with pytest.raises(ValueError):replace(c,cache=replace(c.cache,reserve_gib=128)).validate()
        env={**c.client_env,'SCIENCE_PLACEMENT_RUNTIME':c.science_placement_runtime,'SCIENCE_PLACEMENT_BACKEND':'cpu'}
        for starter in (True,False):
            prompt=task_prompt('placement',include_starter=starter,environment=env)
            assert '300 seconds of search' in prompt and '4 GiB' in prompt
            assert '8 GiB' not in prompt and '390-second' not in prompt
            assert 'ALL 17 IBM cases' in prompt


def test_dispatch_17_cases_with_four_gib_and_explicit_contract():
    from tpu.science import training_env as e,placement_ray,challenge_contract
    class Ref:
        def future(self):
            f=Future();f.set_result(dict(reward=.5,raw_score=.5));return f
    with patch.object(e,'connect'),patch.dict(os.environ,SCIENCE_WORKER_ROOT='/payload',
        SCIENCE_PLACEMENT_BACKEND='cpu',SCIENCE_PLACEMENT_RUNTIME=r.VERSION,
        SCIENCE_PLACEMENT_SLOTS_PER_HOST='32'),patch.object(placement_ray.grade_cpu_case,'options') as opts, \
        patch.object(e.ray,'cancel'),patch.object(challenge_contract,'aggregate',return_value=dict(reward=.5,raw_score=.5)):
        opts.return_value.remote.side_effect=lambda *a,**k:Ref()
        asyncio.run(e.evaluate('placement','source',72000))
        assert opts.call_count==17
        assert all(c.kwargs['memory']==4*1024**3 for c in opts.call_args_list)
        assert all(c.kwargs['resource_contract']==r.contract() for c in opts.return_value.remote.call_args_list)
