import asyncio
from dataclasses import replace
import json
import time

import httpx
import pytest

from tpu.swarm.ray_train.farm_admission import assignments
from tpu.swarm.ray_train.hybrid_scheduler import HybridScheduler
from tpu.swarm.ray_train.run_borrowing import RunBorrower
from test_inference_borrowing import Farm, config


def test_run_reservation_survives_phases_and_replaces_adapter(tmp_path):
    async def run():
        farm = Farm()
        cfg = config(tmp_path)
        cfg = replace(cfg, inference=replace(cfg.inference, external_pool_lease_scope='run'))
        async with httpx.AsyncClient(transport=httpx.MockTransport(farm)) as http:
            borrower = RunBorrower(cfg, http)
            try:
                assert (await borrower.reserve())['reserved']
                lease = borrower.lease.lease_id
                path = tmp_path / 'adapter.tar'
                aliases = []
                for step in (1, 2):
                    path.write_bytes(f'weights-{step}'.encode())
                    await borrower.begin(f'phase-{step}', f'local-{step}', path, expected_n=32)
                    await borrower.preparing
                    assert borrower.snapshot()['ready']
                    assert borrower.lease.lease_id == lease
                    aliases.append(borrower.lease.adapter)
                    payload = {'model': f'local-{step}', 'prompt': [1], 'n': 32}
                    result = await borrower.generate(payload, 4, 4)
                    assert len(result['choices']) == 32
                    assert payload['model'] == f'local-{step}'
                    await borrower.end(f'phase-{step}')
                    assert borrower.snapshot()['reserved'] and not borrower.snapshot()['ready']
                    assert not farm.release_seen.is_set()
                assert aliases[0] != aliases[1]
                assert sum(path == '/acquire_lease' for _, path in farm.requests) == 1
                assert not await borrower.end('stale-phase')
            finally:
                await borrower.close()
            assert farm.release_seen.is_set()
    asyncio.run(run())


def test_failed_publication_fences_remote_but_retains_local_path(tmp_path):
    async def run():
        farm = Farm()
        farm.wrong_hash = True
        async with httpx.AsyncClient(transport=httpx.MockTransport(farm)) as http:
            borrower = RunBorrower(config(tmp_path), http)
            path = tmp_path / 'adapter.tar'
            path.write_bytes(b'weights')
            try:
                await borrower.begin('phase', 'local', path, expected_n=32)
                await borrower.preparing
                assert not borrower.snapshot()['ready']
                assert await borrower.generate({'model': 'local', 'n': 32}, 4, 4) is None
            finally:
                await borrower.close()
    asyncio.run(run())


def test_whole_groups_queue_and_use_newly_available_remote_capacity():
    async def run():
        ready = False
        release = asyncio.Event()
        calls = []
        router = HybridScheduler(2, 1, lambda: ready)
        async def local(index):
            calls.append(('local', index))
            await release.wait()
            return {'choices': [{'token_ids': [1, 2]} for _ in range(32)]}
        async def remote():
            calls.append(('remote', 0))
            return {'choices': [{'token_ids': [3, 4]} for _ in range(32)]}
        payload = {'prompt': [1], 'max_tokens': 10, 'n': 32}
        tasks = [asyncio.create_task(router.run(payload, local, remote)) for _ in range(4)]
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(calls) == 2 and router.snapshot()['queued'] == 2
        ready = True
        async with router.condition:
            router.condition.notify_all()
        for _ in range(10):
            await asyncio.sleep(0)
        assert ('remote', 0) in calls
        release.set()
        results = await asyncio.gather(*tasks)
        assert all(len(result['choices']) == 32 for result in results)
        assert router.snapshot()['queued'] == router.snapshot()['local_active'] == router.snapshot()['remote_active'] == 0
    asyncio.run(run())


def test_remote_failure_returns_one_local_result_and_releases_capacity():
    async def run():
        router = HybridScheduler(1, 1, lambda: True)
        router.rates.update(local=1, remote=10)
        calls = []
        async def local(_):
            calls.append('local')
            return {'choices': [{'token_ids': [9]}]}
        async def remote():
            calls.append('remote')
            return None
        result = await router.run({'prompt': [1], 'n': 1}, local, remote)
        assert calls == ['remote', 'local']
        assert result['choices'][0]['token_ids'] == [9]
        assert not any(router.local) and not any(router.remote)
    asyncio.run(run())


def test_slow_remote_tail_is_not_dispatched():
    router = HybridScheduler(1, 1, lambda: True)
    router.rates.update(local=100, remote=1)
    router.local[0] = (time.monotonic(), 10)
    assert router._choose(100, False) is None  # Wait ~0.1s, then finish locally in 1s.


def test_ac2_admission_precedes_qubit_and_preserves_existing_owners():
    rows = [dict(job_id=i, run_id=name, status='RUNNING', priority=0) for i, name in
            [(10, 'qwen-qubit'), (11, 'qwen-ac2'), (12, 'gemma-ac2')]]
    farms = [dict(job_id=1, models=['qwen'], url='q1', state='unleased'),
             dict(job_id=2, models=['qwen'], url='q2', state='ready', owner_run='unrelated:instance'),
             dict(job_id=3, models=['gemma'], url='g1', state='unleased')]
    targets = {r['job_id']: dict(run_id=r['run_id'], model=r['run_id'].split('-')[0]) for r in rows}
    assert assignments(rows, farms, targets) == {10: [], 11: ['q1'], 12: ['g1']}
    farms[1]['owner_run'] = 'qwen-qubit:instance'
    assert assignments(rows, farms, targets) == {10: ['q2'], 11: ['q1'], 12: ['g1']}
    # Expired requests still draining are not a free reservation.
    farms[0].update(state='expired', active=1, owner_run='old:instance')
    assert assignments(rows, farms, targets)[11] == []


def test_pending_ac2_prevents_new_qubit_admission():
    rows = [dict(job_id=1, run_id='muse-ac2', status='PENDING'),
            dict(job_id=2, run_id='qwen-qubit', status='RUNNING')]
    farms = [dict(job_id=3, models=['qwen'], url='q1', state='unleased')]
    targets = {2: dict(run_id='qwen-qubit', model='qwen')}
    assert assignments(rows, farms, targets) == {2: []}


def test_runtime_mismatch_is_rejected_before_acquiring(tmp_path):
    async def run():
        farm = Farm()
        async def transport(request):
            if request.url.path == '/status' and not farm.owners:
                return httpx.Response(200, json={'capabilities': {'compatibility_sha256': 'different'}})
            return await farm(request)
        cfg = config(tmp_path)
        cfg = replace(cfg, inference=replace(cfg.inference, external_pool_attestation=True))
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
            borrower = RunBorrower(cfg, http)
            borrower.required_contract = 'expected'
            try:
                assert not (await borrower.reserve())['reserved']
                assert not any(path == '/acquire_lease' for _, path in farm.requests)
            finally:
                await borrower.close()
    asyncio.run(run())


def test_orphaned_farm_is_quarantined_instead_of_reassigned(tmp_path, monkeypatch):
    from test_farm_leases import farm as deployed_farm, Remote
    async def run():
        async with deployed_farm(tmp_path, monkeypatch) as (client, gateway, _, __, entered, finish):
            gateway.config = replace(gateway.config,
                inference=replace(gateway.config.inference, farm_drain_timeout=1))
            failed = asyncio.Event()
            gateway.catalog.quarantine = Remote(lambda reason: failed.set())
            lease = (await client.post('/acquire_lease', json={'owner_run': 'abandoned'})).json()
            request = asyncio.create_task(client.post('/v1/completions',
                json={'model': gateway.config.model, 'block': True}, headers={'X-Lease-ID': lease['lease_id']}))
            await entered.wait()
            gateway.lease['deadline'] = 0
            await asyncio.wait_for(failed.wait(), 3)
            assert (await client.get('/health')).status_code == 503
            assert (await client.post('/acquire_lease', json={'owner_run': 'new'})).status_code == 503
            assert gateway.lease['owner_run'] == 'abandoned'
            finish.set()
            await request
    asyncio.run(run())


def test_supervisor_assigns_one_farm_per_run_in_priority_order():
    from test_borrowing_supervisor import fixture
    from tpu.swarm.ray_train.borrowing_supervisor import tick
    rows, farms, targets, calls, call = fixture()
    rows[4]['run_id'] = targets['train-10']['run_id'] = 'qwen-qubit'
    rows[5]['run_id'] = targets['train-11']['run_id'] = 'gemma-ac2'
    for row in rows[4:]: row['pool'] = 'east'
    for target in targets.values(): target['lease_scope'] = 'run'
    tick(rows, 'farm', [], call, trainer_pools=['east'])
    assert targets['train-10']['urls'] == []
    assert targets['train-11']['urls'] == ['http://10.0.0.2:24800']


def test_local_only_benchmark_and_broken_telemetry_do_not_lose_groups():
    async def run():
        def report(*args, **kwargs):
            raise OSError('telemetry unavailable')
        router = HybridScheduler(1, 1, lambda: True, report)
        router.rates.update(local=1, remote=100)
        async def local(_):
            return {'choices': [{'token_ids': [1]}]}
        async def remote():
            pytest.fail('local-only benchmark used remote capacity')
        assert await router.run({'prompt': [1], 'n': 1}, local, remote, local_only=True)
        assert not any(router.local) and router.snapshot()['queued'] == 0
    asyncio.run(run())


def test_controller_waits_for_reservation_and_pins_ingress_identity(tmp_path, monkeypatch):
    from tpu.swarm.ray_train import controller
    cfg = config(tmp_path)
    cfg = replace(cfg, inference=replace(cfg.inference, external_pool_lease_scope='run',
                                       external_pool_require_initial=True))
    owner = controller.Controller(cfg, ['127.0.0.1'])
    owner.report = lambda *args, **kw: None
    actions = []
    def transport(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'instance': 'current-ingress'})
        body = json.loads(request.content)
        assert body['instance'] == 'current-ingress' and body['run_id'] == cfg.run_id
        actions.append(body['action'])
        return httpx.Response(200, json={'reserved': len(actions) > 1})
    original_client = httpx.Client
    monkeypatch.setattr(controller.httpx, 'Client', lambda **kw: original_client(
        transport=httpx.MockTransport(transport), **kw))
    monkeypatch.setattr(owner.stopping, 'wait', lambda seconds: False)
    owner.wait_farm_admission()
    assert actions == ['acquire', 'acquire'] and owner.farm_instance == 'current-ingress'


def test_runtime_identity_detects_installed_patch_but_allows_hardware_flags(tmp_path):
    from tpu.swarm.ray_train.serving_identity import identity
    cfg = config(tmp_path)
    source = tmp_path / 'source'
    for name in ('tpu/vllm_tpu_server.py', 'tpu/thinking_budget/server.py'):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('pass\n')
    snapshot = tmp_path / ('a' * 40)
    snapshot.mkdir()
    for name in ('config.json', 'tokenizer.json'):
        (snapshot / name).write_text('{}')
    site = tmp_path / 'envs/serving/lib/python3.12/site-packages'
    for name in ('vllm', 'tpu_inference', 'transformers', 'jax', 'jaxlib'):
        metadata = site / f'{name}-1.0.dist-info'
        metadata.mkdir(parents=True)
        (metadata / 'METADATA').write_text(f'Name: {name}\nVersion: 1.0\n')
        (metadata / 'RECORD').write_text(f'{name}.py,,\n')
        (site / (name + '.py')).write_text('original = True\n')
    initial = identity(cfg, tmp_path, source, snapshot)
    hardware = replace(cfg, inference=replace(cfg.inference, prefix_caching=not cfg.inference.prefix_caching,
                                              memory_utilization=.9))
    assert identity(hardware, tmp_path, source, snapshot) == initial
    (site / 'tpu_inference.py').write_text('modified = True\n')
    assert identity(cfg, tmp_path, source, snapshot)['sha256'] != initial['sha256']
    (site / 'tpu_inference-1.0.dist-info/direct_url.json').write_text(
        json.dumps({'dir_info': {'editable': True}}))
    with pytest.raises(RuntimeError, match='materialized'):
        identity(cfg, tmp_path, source, snapshot)


def test_benchmark_checks_native_contract_without_mutating_response():
    import copy
    from tpu.swarm.ray_train.hybrid_benchmark import validate, comparison
    from test_native_external_transport import choice
    response = {'choices': [dict(choice(), index=0)]}
    before = copy.deepcopy(response)
    request = dict(n=1, max_tokens=100, thinking_token_budget=1)
    validate(response, request)
    assert response == before
    response['choices'][0]['logprobs']['token_logprobs'][0] = float('nan')
    with pytest.raises(ValueError):
        validate(response, request)
    rounds = [dict(route=route, seconds=seconds, warmup=False)
              for _ in range(3) for route, seconds in [('local', 100), ('hybrid', 85)]]
    assert comparison(rounds)['generation_gate_passed']
    assert comparison(rounds)['full_step_gate'] == 'not_measured'


def test_attested_farm_rejects_unverified_claim_but_allows_renewal(tmp_path, monkeypatch):
    from test_farm_leases import farm as deployed_farm
    async def run():
        async with deployed_farm(tmp_path, monkeypatch) as (client, gateway, *rest):
            gateway.config = replace(gateway.config,
                inference=replace(gateway.config.inference, external_pool_attestation=True))
            gateway.compatibility = dict(sha256='expected', contract={})
            response = await client.post('/acquire_lease', json={'owner_run': 'run'})
            assert response.status_code == 409 and gateway.lease is None
            response = await client.post('/acquire_lease', json={
                'owner_run': 'run', 'compatibility_sha256': 'expected'})
            assert response.status_code == 200
            lease = response.json()['lease_id']
            assert (await client.post('/acquire_lease', json={
                'owner_run': 'run', 'lease_id': lease})).status_code == 200
    asyncio.run(run())


def test_hybrid_ingress_preserves_native_response_and_local_only_header(tmp_path, monkeypatch):
    import inspect
    from types import SimpleNamespace
    from tpu.swarm.ray_train import serving
    from test_farm_leases import Remote
    from test_native_external_transport import choice
    async def run():
        cfg = config(tmp_path)
        cfg = replace(cfg, inference=replace(cfg.inference, external_pool_scheduler=True))
        calls = []
        answer = {'choices': [dict(choice(), index=i) for i in range(32)]}
        async def local(payload):
            calls.append(('local', payload))
            return answer
        async def remote(payload, **kw):
            calls.append(('remote', payload))
            return answer
        gateway = serving.Ingress.func_or_class.__mro__[1](cfg.to_dict(),
            [SimpleNamespace(generate=Remote(local))], SimpleNamespace(), ['127.0.0.1'])
        gateway.borrower = SimpleNamespace(lease=object(), _eligible=lambda lease: True, generate=remote)
        gateway.scheduler.rates.update(local=1, remote=100)
        monkeypatch.setattr(serving.serve, 'get_replica_context', lambda: SimpleNamespace(servable_object=gateway))
        app = inspect.getclosurevars(serving.Ingress.func_or_class.__init__).nonlocals['frozen_app_or_func']
        payload = dict(model=cfg.model, prompt=[1], n=32, thinking_token_budget=1, max_tokens=10)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as http:
                for headers in ({}, {'X-SkyRL-Local-Only': '1'}):
                    response = await http.post('/v1/completions', json=payload, headers=headers)
                    assert response.status_code == 200 and response.json() == answer
            assert calls == [('remote', payload), ('local', payload)]
            assert gateway.active == 0
            assert gateway.scheduler.completed == {'local': 1, 'remote': 1}
        finally:
            await gateway.http.aclose()
    asyncio.run(run())
