"""Exercise lease ownership and hash readiness through the actual ingress HTTP API."""
import asyncio
import hashlib
import inspect
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from tpu.swarm.ray_train import serving
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.overlay import manifest


class Remote:
    def __init__(self, fn):
        self.fn = fn

    async def remote(self, *args):
        value = self.fn(*args)
        return await value if inspect.isawaitable(value) else value


@asynccontextmanager
async def farm(tmp_path, monkeypatch):
    cfg = Config.load('tpu/swarm/ray_train/profiles/qwen35-v432-native-multi-lora-inference-20260919.json').to_dict()
    cfg['root'] = str(tmp_path)
    ips = [f'10.0.0.{i}' for i in range(1, 5)]
    catalog = serving.Catalog.__ray_metadata__.modified_class(ips, 0)
    for ip in ips:
        catalog.register(ip, ip)
    weights = {ip: {} for ip in ips}
    wrong_hash = set()
    entered, finish = asyncio.Event(), asyncio.Event()

    async def generate(payload):
        if payload.get('block'):
            entered.set()
            await finish.wait()
        return payload

    async def transport(request):
        ip = request.url.host
        if request.url.path == '/health':
            return httpx.Response(200, json={'status': 'ok'})
        if request.url.path == '/skyrl/v1/adapter_status':
            return httpx.Response(200, json={'adapters': weights[ip]})
        identity = hashlib.sha256(await request.aread()).hexdigest()
        weights[ip] = {request.url.params['lora_name']: identity}
        return httpx.Response(200, json={'sha256': 'bad' if ip in wrong_hash else identity})

    cls = serving.Ingress.func_or_class.__mro__[1]
    gateway = cls(cfg, SimpleNamespace(generate=Remote(generate), tokenize=Remote(generate)),
                  SimpleNamespace(commit=Remote(catalog.commit), snapshot=Remote(catalog.snapshot)), ips)
    await gateway.http.aclose()
    gateway.http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    monkeypatch.setattr(serving.serve, 'get_replica_context', lambda: SimpleNamespace(servable_object=gateway))
    app = inspect.getclosurevars(serving.Ingress.func_or_class.__init__).nonlocals['frozen_app_or_func']
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://farm') as client:
            yield client, gateway, weights, wrong_hash, entered, finish
    finally:
        await gateway.http.aclose()


def test_exclusive_lease_hash_status_and_stale_owner_fencing(tmp_path, monkeypatch):
    async def run():
        async with farm(tmp_path, monkeypatch) as (client, gateway, weights, wrong_hash, _, __):
            model = gateway.config.model
            assert (await client.post('/v1/completions', json={'model': model})).status_code == 409
            assert (await client.post('/acquire_lease', json={'owner_run': 'run-a', 'ttl_seconds': True})).status_code == 400
            acquired = await client.post('/acquire_lease', json={'owner_run': 'run-a'})
            assert acquired.status_code == 200
            a = acquired.json()
            assert a['state'] == 'awaiting_adapter' and a['ready_engines'] == 0
            headers = {'X-Lease-ID': a['lease_id']}
            assert (await client.post('/acquire_lease', json={'owner_run': 'run-b'})).status_code == 409
            assert (await client.post('/acquire_lease', json={'owner_run': 'run-a'})).status_code == 409
            # Renewal must remain available while an upload owns this lock.
            async with gateway.upload_lock:
                renewed = await asyncio.wait_for(client.post('/acquire_lease', json={
                    'owner_run': 'run-a', 'lease_id': a['lease_id'], 'ttl_seconds': 600}), 1)
            assert renewed.json()['lease_id'] == a['lease_id']
            assert (await client.post('/release_lease', json={'lease_id': 'wrong'})).status_code == 409
            assert (await client.post('/skyrl/v1/upload_lora_adapter?lora_name=A', content=b'weights')).status_code == 409
            assert (await client.post('/skyrl/v1/upload_lora_adapter?lora_name=A', content=b'weights',
                                     headers={**headers, 'X-Adapter-SHA256': 'bad'})).status_code == 400
            uploaded = await client.post('/skyrl/v1/upload_lora_adapter?lora_name=A', content=b'weights', headers=headers)
            assert uploaded.status_code == 200, uploaded.text
            identity = hashlib.sha256(b'weights').hexdigest()
            status = (await client.get('/status')).json()
            assert (status['owner_run'], status['state'], status['ready_engines'], status['adapter_sha256']) == ('run-a', 'ready', 4, identity)
            assert (await client.post('/v1/completions', json={'model': 'A'}, headers=headers)).status_code == 200
            weights['10.0.0.4']['A'] = 'wrong'
            status = (await client.get('/status')).json()
            assert status['state'] == 'degraded' and status['ready_engines'] == 3
            gateway.lease['deadline'] = 0
            assert (await client.get('/status')).json()['state'] == 'expired'
            assert (await client.post('/v1/completions', json={'model': 'A'}, headers=headers)).status_code == 409
            assert (await client.post('/acquire_lease', json={'owner_run': 'run-a', 'lease_id': a['lease_id']})).status_code == 409
            b = (await client.post('/acquire_lease', json={'owner_run': 'run-b'})).json()
            assert b['lease_id'] != a['lease_id'] and b['adapter_sha256'] is None
            b_headers = {'X-Lease-ID': b['lease_id']}
            assert (await client.post('/v1/completions', json={'model': 'A'}, headers=b_headers)).status_code == 409
            assert (await client.post('/skyrl/v1/upload_lora_adapter?lora_name=B', content=b'new', headers=headers)).status_code == 409
            wrong_hash.add('10.0.0.4')
            response = await client.post('/skyrl/v1/upload_lora_adapter?lora_name=B', content=b'new', headers=b_headers)
            assert response.status_code == 503 and gateway.updating
            assert (await client.post('/v1/completions', json={'model': model}, headers=b_headers)).status_code == 409
            # A lost owner during partial fanout must allow takeover + repair,
            # while keeping generation closed until every hash is acknowledged.
            gateway.lease['deadline'] = 0
            c = (await client.post('/acquire_lease', json={'owner_run': 'run-c'})).json()
            wrong_hash.clear()
            c_headers = {'X-Lease-ID': c['lease_id']}
            assert (await client.post('/skyrl/v1/upload_lora_adapter?lora_name=C', content=b'repaired', headers=c_headers)).status_code == 200
            assert (await client.get('/status')).json()['state'] == 'ready'
            assert (await client.post('/release_lease', json={'lease_id': c['lease_id']})).status_code == 200
            assert (await client.get('/status')).json()['state'] == 'unleased'
            assert (await client.post('/v1/completions', json={'model': 'C'}, headers=c_headers)).status_code == 409
    asyncio.run(run())


@pytest.mark.parametrize('operation', ['release', 'expire_and_acquire', 'disconnect_and_release'])
def test_lease_handoff_drains_admitted_requests(tmp_path, monkeypatch, operation):
    async def run():
        async with farm(tmp_path, monkeypatch) as (client, gateway, _, __, entered, finish):
            a = (await client.post('/acquire_lease', json={'owner_run': 'a'})).json()
            headers = {'X-Lease-ID': a['lease_id']}
            generation = asyncio.create_task(client.post('/v1/completions', json={'model': gateway.config.model, 'block': True}, headers=headers))
            await asyncio.wait_for(entered.wait(), 1)
            if operation == 'disconnect_and_release':
                generation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await generation
                assert gateway.active == 1
            if operation in ('release', 'disconnect_and_release'):
                handoff = asyncio.create_task(client.post('/release_lease', json={'lease_id': a['lease_id']}))
            else:
                gateway.lease['deadline'] = 0
                handoff = asyncio.create_task(client.post('/acquire_lease', json={'owner_run': 'b'}))
            for _ in range(100):
                if gateway.lease['releasing']:
                    break
                await asyncio.sleep(0)
            assert gateway.lease['releasing'] and not handoff.done()
            assert (await client.post('/v1/completions', json={'model': gateway.config.model}, headers=headers)).status_code == 409
            finish.set()
            if operation != 'disconnect_and_release':
                assert (await generation).status_code == 200
            assert (await asyncio.wait_for(handoff, 1)).status_code == 200
    asyncio.run(run())


def test_lease_config_ships_hash_capable_engine_source():
    cfg = Config.load('tpu/swarm/ray_train/profiles/qwen35-v432-native-multi-lora-inference-20260919.json')
    assert cfg.requires_source_overlay
    assert 'tpu/vllm_tpu_server.py' in manifest('.', cfg)
    raw = cfg.to_dict()
    raw['inference']['max_loras'] = 2
    with pytest.raises(ValueError, match='leases require'):
        Config.from_dict(raw)
