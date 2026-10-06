from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch
import pytest
from tpu.science import placement_resources as r
from tpu.swarm.ray_train.config import Config


def test_drain_before_changing_cpu_width_and_96_exclusive_slots(tmp_path):
    with patch.object(r.os, 'sched_getaffinity', return_value=set(range(240))):
        _, _, old = r.acquire(48, root=tmp_path)
        with old:
            with pytest.raises(RuntimeError, match='drain all cases'):
                r.acquire(96, root=tmp_path, cpus_per_case=2)
        with ExitStack() as held:
            assigned = []
            for _ in range(96):
                _, cpus, lock = r.acquire(96, root=tmp_path, cpus_per_case=2)
                held.enter_context(lock)
                assert len(cpus) == 2
                assigned.extend(cpus)
            assert sorted(assigned) == list(range(48, 240))
            with pytest.raises(TimeoutError):
                r.acquire(96, root=tmp_path, cpus_per_case=2, deadline_seconds=.01)
            with pytest.raises(RuntimeError, match='drain all cases'):
                r.acquire(48, root=tmp_path)
        _, cpus, old = r.acquire(48, root=tmp_path)
        with old:
            assert len(cpus) == 4


def test_96_config_keeps_memory_guard():
    c = Config.load('tpu/swarm/ray_train/profiles/circuit300-v464-qwen-pwc05-three-starts-10step-20260922.json')
    c = replace(c, science_placement_slots_per_host=96, science_placement_cpus_per_case=2,
                cache=replace(c.cache, reserve_gib=256))
    with pytest.raises(ValueError, match='reserve'):
        c.validate()
    c = replace(c, science_placement_memory_gib=2)
    c.validate()
    assert c.ray_cpus_per_host == 200
    r.validate(r.contract(96, cpus=2, memory_gib=2))
    with pytest.raises(ValueError):
        r.contract(96)


def test_two_cpu_dispatch_and_memory_prompt(monkeypatch):
    import asyncio
    from concurrent.futures import Future
    from tpu.science import training_env as e, placement_ray, challenge_contract
    class Ref:
        def future(self):
            f = Future(); f.set_result(dict(reward=.5, raw_score=.5)); return f
    env = dict(SCIENCE_WORKER_ROOT='/payload', SCIENCE_PLACEMENT_BACKEND='cpu',
               SCIENCE_PLACEMENT_RUNTIME=r.VERSION, SCIENCE_PLACEMENT_SLOTS_PER_HOST='96',
               SCIENCE_PLACEMENT_CPUS_PER_CASE='2', SCIENCE_PLACEMENT_MEMORY_GIB='2')
    for k, v in env.items(): monkeypatch.setenv(k, v)
    with patch.object(e, 'connect'), patch.object(placement_ray.grade_cpu_case, 'options') as opts, \
         patch.object(e.ray, 'cancel'), patch.object(challenge_contract, 'aggregate', return_value=dict(reward=.5, raw_score=.5)):
        opts.return_value.remote.side_effect = lambda *a, **k: Ref()
        asyncio.run(e.evaluate('placement', 'source', 72000))
        assert opts.call_count == 17
        assert all(c.kwargs['num_cpus'] == 2 and c.kwargs['memory'] == 2*1024**3 for c in opts.call_args_list)
        assert all(c.kwargs['resource_contract'] == r.contract(96, cpus=2, memory_gib=2) for c in opts.return_value.remote.call_args_list)
    assert '2 GiB' in e.task_prompt('placement', environment=env)
