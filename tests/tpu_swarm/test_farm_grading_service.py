"""Lease-fenced grading endpoints on the real farm Ingress with a fake Ray dispatch."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import inspect
import json
from types import SimpleNamespace
import uuid

import httpx
import pytest

from tpu.swarm.ray_train import serving
from tpu.swarm.ray_train.config import Config
from tests.tpu_swarm.test_farm_leases import Remote

FARM = 'tpu/swarm/ray_train/profiles/farm10-qwen-1-20260921-10step-deployment.json'


@asynccontextmanager
async def grading_farm(tmp_path, monkeypatch, slots_per_host=16):
    cfg = Config.load(FARM).to_dict()
    cfg['root'] = str(tmp_path)
    cfg['inference']['external_pool_attestation'] = False
    cfg['cache']['reserve_gib'] = 192
    cfg['grading'] = {'families': {'ac2': {'slots_per_host': slots_per_host, 'cpus': 2, 'memory_gib': 4}},
                      'result_retention_seconds': 60, 'long_poll_seconds': 2}
    ips = [f'10.0.0.{i}' for i in range(1, 5)]
    catalog = serving.Catalog.__ray_metadata__.modified_class(ips, 0)
    for ip in ips:
        catalog.register(ip, ip)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def generate(payload):
        if payload.get('block'):
            entered.set()
            await finish.wait()
        return payload

    async def transport(request):
        if request.url.path == '/health':
            return httpx.Response(200, json={'status': 'ok'})
        return httpx.Response(200, json={'adapters': {}})

    cls = serving.Ingress.func_or_class.__mro__[1]
    gateway = cls(cfg, SimpleNamespace(generate=Remote(generate), tokenize=Remote(generate)),
                  SimpleNamespace(commit=Remote(catalog.commit), snapshot=Remote(catalog.snapshot),
                                  quarantine=Remote(lambda reason: None)), ips)
    await gateway.http.aclose()
    gateway.http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    service = gateway.grading
    assert service is not None
    if service.warmup_task:
        service.warmup_task.cancel()
        await asyncio.gather(service.warmup_task, return_exceptions=True)
    service.ready = True
    futures, cancelled = [], []

    def dispatch(task, spec):
        future = asyncio.get_running_loop().create_future()
        future.spec = spec
        futures.append(future)
        return future
    service.dispatch = dispatch
    service.cancel_ref = lambda ref: cancelled.append(ref) or ref.cancel()
    monkeypatch.setattr(serving.serve, 'get_replica_context', lambda: SimpleNamespace(servable_object=gateway))
    app = inspect.getclosurevars(serving.Ingress.func_or_class.__init__).nonlocals['frozen_app_or_func']
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://farm') as client:
            yield client, gateway, service, futures, cancelled, (entered, finish)
    finally:
        await service.close()
        await gateway.http.aclose()


def body(task='ac2', **spec):
    default = dict(program_code='def construct_function():\n    return [1.0]\n', function_name='construct_function',
                   eval_timeout_seconds=1105, admission_timeout_s=1100)
    return dict(request_id=uuid.uuid4().hex, task=task, owner_run='pilot', scope={'step': 1},
                spec=dict(default, **spec))


async def lease_for(client, owner='pilot'):
    response = await client.post('/acquire_lease', json={'owner_run': owner})
    assert response.status_code == 200, response.text
    return {'X-Lease-ID': response.json()['lease_id']}


def test_submit_requires_lease_validates_and_is_idempotent(tmp_path, monkeypatch):
    async def run():
        async with grading_farm(tmp_path, monkeypatch) as (client, gateway, service, futures, cancelled, _):
            request = body()
            assert (await client.post('/skyrl/v1/grading/submit', json=request)).status_code == 409
            headers = await lease_for(client)
            bad = await client.post('/skyrl/v1/grading/submit', json=dict(request, task='routing'), headers=headers)
            assert bad.status_code == 400 and 'not served' in bad.text
            bad = await client.post('/skyrl/v1/grading/submit', json=dict(request, request_id='nope'), headers=headers)
            assert bad.status_code == 400
            first = await client.post('/skyrl/v1/grading/submit', json=request, headers=headers)
            assert first.status_code == 202 and first.json()['state'] in ('queued', 'running')
            again = await client.post('/skyrl/v1/grading/submit', json=request, headers=headers)
            assert again.status_code == 200 and again.json()['request_id'] == request['request_id']
            await asyncio.sleep(0)
            assert len(futures) == 1  # Never two executions for one request id.
            assert futures[0].spec['systemd'] is True and futures[0].spec['stdout_limit_bytes'] == 16384
            futures[0].set_result(dict(result=[1.0], error=None, stdout='', metrics={'host': '10.0.0.2'}))
            done = await client.get(f'/skyrl/v1/grading/result/{request["request_id"]}', params={'wait': 1},
                                    headers=headers)
            assert done.json()['state'] == 'done' and done.json()['result']['result'] == [1.0]
            assert done.json()['host'] == '10.0.0.2' and 'queued_seconds' in done.json()['metrics']
            assert service.counters['completed'] == 1
            # Another owner's lease cannot read or resubmit this request.
            await client.post('/release_lease', json={'lease_id': headers['X-Lease-ID']})
            other = await lease_for(client, 'other')
            assert (await client.get(f'/skyrl/v1/grading/result/{request["request_id"]}', headers=other)).status_code == 404
            assert (await client.post('/skyrl/v1/grading/submit', json=request, headers=other)).status_code == 409
    asyncio.run(run())


def test_result_long_poll_then_retention_expiry_returns_404(tmp_path, monkeypatch):
    async def run():
        async with grading_farm(tmp_path, monkeypatch) as (client, gateway, service, futures, cancelled, _):
            headers = await lease_for(client)
            request = body()
            await client.post('/skyrl/v1/grading/submit', json=request, headers=headers)
            pending = asyncio.create_task(client.get(f'/skyrl/v1/grading/result/{request["request_id"]}',
                                                     params={'wait': 5}, headers=headers))
            await asyncio.sleep(.05)
            assert not pending.done()
            futures[0].set_result(dict(result=[2.0], error=None, stdout='', metrics={}))
            response = await asyncio.wait_for(pending, 5)
            assert response.json()['state'] == 'done'
            entry = service.entries[request['request_id']]
            assert entry.fetched is not None
            assert service.reap(now=entry.fetched + 10) == 0
            assert service.reap(now=entry.fetched + 61) == 1
            assert (await client.get(f'/skyrl/v1/grading/result/{request["request_id"]}', headers=headers)).status_code == 404
            assert (await client.get('/skyrl/v1/grading/result/' + 'f' * 32, headers=headers)).status_code == 404
    asyncio.run(run())


def test_cancel_forces_ray_cancel_and_is_idempotent(tmp_path, monkeypatch):
    async def run():
        async with grading_farm(tmp_path, monkeypatch) as (client, gateway, service, futures, cancelled, _):
            headers = await lease_for(client)
            request = body()
            await client.post('/skyrl/v1/grading/submit', json=request, headers=headers)
            await asyncio.sleep(0)
            first = await client.post(f'/skyrl/v1/grading/cancel/{request["request_id"]}', headers=headers)
            assert first.status_code == 200 and first.json()['state'] == 'cancelled'
            assert cancelled == [futures[0]] and service.counters['cancelled'] == 1
            second = await client.post(f'/skyrl/v1/grading/cancel/{request["request_id"]}', headers=headers)
            assert second.json()['state'] == 'cancelled' and len(cancelled) == 1
            assert service.running() == 0
    asyncio.run(run())


def test_release_lease_cancels_all_grading_and_drains(tmp_path, monkeypatch):
    async def run():
        async with grading_farm(tmp_path, monkeypatch) as (client, gateway, service, futures, cancelled, _):
            headers = await lease_for(client)
            requests = [body() for _ in range(3)]
            for request in requests:
                assert (await client.post('/skyrl/v1/grading/submit', json=request, headers=headers)).status_code == 202
            await asyncio.sleep(0)
            assert service.running() == 3 and (await client.get('/status')).json()['grading']['queue_depth'] == 3
            released = await client.post('/release_lease', json={'lease_id': headers['X-Lease-ID']})
            assert released.status_code == 200 and released.json()['released']
            assert service.running() == 0 and len(cancelled) == 3
            assert all(service.entries[r['request_id']].state == 'cancelled' for r in requests)
            events = [json.loads(line) for line in (gateway.run / 'inference-events.jsonl').read_text().splitlines()]
            assert any(e['event'] == 'grading_cancel_all' and e['requests'] == 3 for e in events)
    asyncio.run(run())


def test_expired_lease_cancels_grading_and_frees_the_farm(tmp_path, monkeypatch):
    async def run():
        async with grading_farm(tmp_path, monkeypatch) as (client, gateway, service, futures, cancelled, _):
            gateway.config = replace(gateway.config, inference=replace(
                gateway.config.inference, farm_cancel_grace_seconds=0, farm_drain_timeout=5))
            quarantined = asyncio.Event()
            gateway.catalog.quarantine = Remote(lambda reason: quarantined.set())
            headers = await lease_for(client)
            request = body()
            await client.post('/skyrl/v1/grading/submit', json=request, headers=headers)
            await asyncio.sleep(0)
            gateway.lease['deadline'] = 0
            for _ in range(60):
                if service.running() == 0:
                    break
                await asyncio.sleep(.05)
            assert service.running() == 0 and cancelled and not quarantined.is_set()
            granted = await client.post('/acquire_lease', json={'owner_run': 'next'})
            assert granted.status_code == 200
    asyncio.run(run())


def test_capacity_status_and_backpressure(tmp_path, monkeypatch):
    async def run():
        async with grading_farm(tmp_path, monkeypatch, slots_per_host=1) as (client, gateway, service, futures, cancelled, _):
            headers = await lease_for(client)
            capacity = (await client.get('/skyrl/v1/grading/capacity', headers=headers)).json()
            assert capacity['ready'] and capacity['hosts'] == 4
            assert capacity['families']['ac2'] == dict(slots_per_host=1, cpus=2, memory_gib=4, total=4, running=0, queued=0)
            statuses = [(await client.post('/skyrl/v1/grading/submit', json=body(), headers=headers)).status_code
                        for _ in range(9)]
            assert statuses == [202] * 8 + [429]
            status = (await client.get('/status')).json()['grading']
            assert status['queue_depth'] == 8 and status['ready'] and status['families'] == ['ac2']
            capacity = (await client.get('/skyrl/v1/grading/capacity', headers=headers)).json()
            assert capacity['families']['ac2']['running'] == 8
            for future in futures:
                future.set_result(dict(result=[1.0], error=None, stdout='', metrics={'host': '10.0.0.1'}))
            await asyncio.sleep(.05)
            assert (await client.post('/skyrl/v1/grading/submit', json=body(), headers=headers)).status_code == 202
            await asyncio.sleep(0)
            assert service.running_by_host() == {'pending': 1}
    asyncio.run(run())


def test_infrastructure_and_candidate_errors_are_reported_distinctly(tmp_path, monkeypatch):
    async def run():
        async with grading_farm(tmp_path, monkeypatch) as (client, gateway, service, futures, cancelled, _):
            headers = await lease_for(client)
            infra, candidate = body(), body()
            await client.post('/skyrl/v1/grading/submit', json=infra, headers=headers)
            await client.post('/skyrl/v1/grading/submit', json=candidate, headers=headers)
            await asyncio.sleep(0)
            futures[0].set_exception(RuntimeError('AC2 unit exited 1 before the candidate started'))
            futures[1].set_result(dict(result=None, error='Process timed out after 1105 seconds', stdout='', metrics={}))
            await asyncio.sleep(.02)
            view = (await client.get(f'/skyrl/v1/grading/result/{infra["request_id"]}', headers=headers)).json()
            assert view['state'] == 'failed' and view['error'] == {'class': 'infrastructure',
                                                                   'detail': 'RuntimeError: AC2 unit exited 1 before the candidate started'}
            view = (await client.get(f'/skyrl/v1/grading/result/{candidate["request_id"]}', headers=headers)).json()
            assert view['state'] == 'done' and view['error'] is None and 'timed out' in view['result']['error']
            assert service.counters == dict(completed=0, infra_failures=1, candidate_failures=1, cancelled=0)
    asyncio.run(run())


def test_science_specs_validate_and_dispatch_signatures(tmp_path, monkeypatch):
    from tpu.swarm.ray_train.grading_service import GradingService

    async def run():
        cfg = Config.load(FARM).to_dict()
        cfg['root'] = str(tmp_path)
        cfg['cache']['reserve_gib'] = 256
        cfg['grading'] = {'families': {'routing': {'slots_per_host': 4}}}
        config = Config.from_dict(cfg)
        service = GradingService(config, ['10.0.0.1'], warmup=False)
        try:
            spec = service.validate(dict(request_id='a' * 32, task='routing',
                                         spec=dict(source='print(1)', routing_suite='full', slots_per_host=4)))
            assert spec == dict(source='print(1)', routing_suite='full', resource_contract=None,
                                slots_per_host=4, admission_timeout_s=2400)
            with pytest.raises(ValueError, match='routing_suite'):
                service.validate(dict(request_id='a' * 32, task='routing', spec=dict(source='x', routing_suite='half')))
            with pytest.raises(ValueError, match='not served'):
                service.validate(dict(request_id='a' * 32, task='placement', spec=dict(source='x', case='ibm01')))
        finally:
            await service.close()
        cfg['grading'] = {'families': {'placement': {'slots_per_host': 8, 'memory_gib': 4}}}
        service = GradingService(Config.from_dict(cfg), ['10.0.0.1'], warmup=False)
        try:
            spec = service.validate(dict(request_id='b' * 32, task='placement',
                                         spec=dict(source='x', case='ibm01', helper='none')))
            assert spec['case'] == 'ibm01' and spec['slots_per_host'] == 8
            with pytest.raises(ValueError, match='case'):
                service.validate(dict(request_id='b' * 32, task='placement', spec=dict(source='x', case='../etc')))
        finally:
            await service.close()
    asyncio.run(run())
