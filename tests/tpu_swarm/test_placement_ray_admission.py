"""Placement Ray admission takes num_cpus and memory from the case resource contract.

The options checked are the merged ones Ray receives in ``RemoteFunction._remote``,
not only the overrides passed to ``.options``."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tpu.science import placement_ray
from tpu.science import placement_resources as r
from tpu.science.grading_transport import GradingRequest, LocalRayPool
from tpu.swarm.ray_train.config import Config

FARM = 'tpu/swarm/ray_train/profiles/inference-farm-rq4-gemma-east5a-v5p32-001.json'
GIB = 1024 ** 3
CONTRACTS = [(2, 2, 80), (4, 4, 32)]  # (cpus, memory_gib, slots): RQ4 farms and the legacy contract


def capture():
    """Patch grade_cpu_case's Ray submission; return the list of merged options per call."""
    calls = []

    def fake(*, args, kwargs, **options):
        calls.append(dict(options, kwargs=kwargs))
        future = asyncio.get_event_loop().create_future()
        future.set_result(dict(correctness=1))
        return future
    return calls, patch.object(placement_ray.grade_cpu_case, '_remote', side_effect=fake)


def test_decorator_default_is_legacy_four_cpus():
    options = placement_ray.grade_cpu_case._default_options
    assert options['num_cpus'] == 4 and options['memory'] == 8 * GIB


@pytest.mark.parametrize('cpus,memory_gib,slots', CONTRACTS)
def test_ray_options_follow_the_contract(cpus, memory_gib, slots):
    assert r.ray_options(r.contract(slots, cpus=cpus, memory_gib=memory_gib)) == dict(
        num_cpus=cpus, memory=memory_gib * GIB)
    with pytest.raises(ValueError):
        r.ray_options(dict(r.contract(slots, cpus=cpus, memory_gib=memory_gib), cpus=3))


@pytest.mark.parametrize('cpus,memory_gib,slots', CONTRACTS)
def test_farm_dispatch_and_warmup_admit_by_contract(tmp_path, cpus, memory_gib, slots):
    from tpu.swarm.ray_train import grading_service as gs

    async def run():
        cfg = Config.load(FARM).to_dict()
        cfg['root'] = str(tmp_path)
        service = gs.GradingService(Config.from_dict(cfg), ['10.0.0.1'], warmup=False)
        try:
            service.families = {'placement': dict(slots_per_host=slots, cpus=cpus, memory_gib=memory_gib)}
            contract = r.contract(slots, cpus=cpus, memory_gib=memory_gib)
            calls, patched = capture()
            with patched:
                spec = service.validate(dict(request_id='c' * 32, task='placement', spec=dict(
                    source='x', case='ibm01', resource_contract=contract, slots_per_host=slots)))
                await service.dispatch('placement', spec)
                # Warmup's reference grade on each host uses the same admission.
                import ray
                manifest = asyncio.get_event_loop().create_future()
                manifest.set_result({})
                with patch.object(ray, 'is_initialized', return_value=True), \
                        patch.object(ray, 'nodes', return_value=[dict(NodeManagerAddress='10.0.0.1', NodeID='a' * 56,
                                                                      Alive=True)]), \
                        patch.object(ray, 'remote', return_value=lambda f: SimpleNamespace(remote=lambda *a, **k: manifest)), \
                        patch('tpu.science.placement_runtime.check_reference'):
                    await service._warmup()
            assert service.readiness == {'placement@10.0.0.1': 'ok'}, service.readiness
            assert len(calls) == 2
            for call in calls:
                assert call['num_cpus'] == cpus and call['memory'] == memory_gib * GIB
                assert call['resources'] == {'placement_cpu_host': 1, 'grading_placement': 1}
                assert call['kwargs']['resource_contract'] == contract
            with pytest.raises(ValueError, match='memory'):
                service.validate(dict(request_id='d' * 32, task='placement', spec=dict(
                    source='x', case='ibm01', resource_contract=dict(contract, memory_gib=8), slots_per_host=slots)))
        finally:
            await service.close()
    asyncio.run(run())


@pytest.mark.parametrize('cpus,memory_gib,slots', CONTRACTS)
def test_local_ray_pool_admits_by_contract(cpus, memory_gib, slots):
    async def run():
        contract = r.contract(slots, cpus=cpus, memory_gib=memory_gib)
        pool = LocalRayPool({}, 1, root='/payload')
        calls, patched = capture()
        with patched:
            await pool.dispatch(GradingRequest(task='placement', spec=dict(
                source='x', case='ibm01', resource_contract=contract, slots_per_host=slots, helper='none')))
        assert len(calls) == 1
        assert calls[0]['num_cpus'] == cpus and calls[0]['memory'] == memory_gib * GIB
        assert calls[0]['kwargs']['resource_contract'] == contract
    asyncio.run(run())
