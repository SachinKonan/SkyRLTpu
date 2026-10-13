"""Hybrid ingress distributes work to two farms and retains local fallback."""
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from tpu.swarm.ray_train import serving
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.multi_borrowing import MultiRunBorrower
from tests.tpu_swarm.test_farm_leases import Remote
from tests.tpu_swarm.test_remote_only import MultiFarm, until

PROFILE = 'tpu/swarm/ray_train/profiles/qcmp-qwen38-qubit-pwc05-s0-shared-v464-wd-20261007-pinned.json'


@pytest.mark.parametrize('change', [dict(external_pool_scheduler=False),
    dict(external_pool_lease_scope='phase'), dict(external_pool_attestation=False,
                                               external_pool_dynamic_engines=False)])
def test_multi_farm_requires_scheduler_run_scope_and_attestation(change):
    raw = Config.load(PROFILE).to_dict()
    raw['inference'].update(external_pool_target_leases=2, **change)
    with pytest.raises(ValueError, match='multiple farms require'):
        Config.from_dict(raw)


def test_shared_pool_recovery_can_require_saved_checkpoint():
    raw = Config.load(PROFILE).to_dict()
    raw['resume_min_checkpoint_step'] = 3
    config = Config.from_dict(raw)
    assert config.checkpoint_resume and config.bootstrap_reuse_source_run_id
    assert config.resume_min_checkpoint_step == 3


def test_hybrid_ingress_two_farms_and_local_fallback(tmp_path):
    async def run():
        raw = Config.load(PROFILE).to_dict()
        raw['root'] = str(tmp_path)
        raw['inference'].update(external_pool_target_leases=2,
            external_pool_urls={raw['model']: ['http://farm1', 'http://farm2', 'http://farm3']},
            external_pool_heartbeat_seconds=1, external_pool_rpc_timeout=1,
            external_pool_health_grace_seconds=1, external_pool_lease_seconds=30)
        cfg = Config.from_dict(raw)
        gate = asyncio.Event()
        local_calls = []
        async def generate(payload):
            local_calls.append(payload)
            await gate.wait()
            return dict(choices=[dict(token_ids=[1, 2])])
        class AsyncRemote:
            remote = staticmethod(generate)
        engine = SimpleNamespace(generate=AsyncRemote(), identity=Remote(
            lambda: dict(sha256='c' * 64, contract={})))
        gateway = serving.Ingress.func_or_class.__mro__[1](cfg.to_dict(), [engine],
            SimpleNamespace(fail=Remote(lambda *a: None)), [])
        assert gateway.multi_farm and not gateway.remote_only
        assert isinstance(gateway.borrower, MultiRunBorrower)
        class Qwen38Farm(MultiFarm):
            async def __call__(self, request):
                if request.url.path == '/v1/models':
                    return httpx.Response(200, json={'data': [{'id': cfg.model}]})
                return await super().__call__(request)
        farm = Qwen38Farm()
        farm.capabilities.update({f'farm{i}': 'c' * 64 for i in range(1, 4)})
        farm.generation_gate = gate
        async with httpx.AsyncClient(transport=httpx.MockTransport(farm)) as client:
            await gateway.http.aclose()
            gateway.http = client
            gateway.borrower.http = client
            for member in gateway.borrower.members.values():
                member.http = client
            tasks = []
            try:
                await gateway.capabilities()
                await gateway.borrower.reserve()
                archive = tmp_path / 'adapter.tar'
                archive.write_bytes(b'weights')
                await gateway.borrower.begin('phase1', cfg.model, archive, expected_n=1)
                await until(lambda: len(gateway.borrower.eligible_pools()) == 2)
                assert len(gateway.borrower.held()) == 2 and 'farm3' not in farm.owners
                assert not gateway.borrower.snapshot()['remote_only']
                payload = dict(model=cfg.model, prompt=[1], n=1, max_tokens=4)
                # Fill one local slot and four slots in each farm. Each request
                # must use the farm selected by the scheduler, not the first lease.
                gateway.active = 9
                tasks = [asyncio.create_task(gateway.finish_admitted('generate', payload)) for _ in range(9)]
                await until(lambda: len(local_calls) == 1 and
                    sum(p == '/v1/completions' for h, p in farm.requests) == 8)
                assert gateway.scheduler.snapshot()['farms'] == {'http://farm1': 4, 'http://farm2': 4}
                gate.set()
                results = await asyncio.gather(*tasks)
                assert all(r['choices'] for r in results) and gateway.active == 0
                # One farm can disappear without preventing local generation.
                gateway.borrower.update_urls(cfg.model, ['http://farm2'])
                farm.down.add('farm1')
                await until(lambda: len(gateway.borrower.eligible_pools()) < 2)
                before = len(local_calls)
                gateway.active = 1
                result = await gateway.finish_admitted('generate', payload, local_only=True)
                assert result['choices'] and len(local_calls) == before + 1
                await gateway.borrower.close()
                gateway.active = 1
                assert (await gateway.finish_admitted('generate', payload))['choices']
            finally:
                gate.set()
                for task in tasks:
                    if not task.done(): task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await gateway.borrower.close()
    asyncio.run(run())
