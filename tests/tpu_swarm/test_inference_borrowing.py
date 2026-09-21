import asyncio
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tpu.swarm.ray_train.borrowing import Borrower
from tpu.swarm.ray_train.config import Config

PROFILE = 'tpu/swarm/ray_train/profiles/science-q20-v4-qwen-grpo-clean-20260918.json'


def config(tmp_path, enabled=True):
    raw = Config.load(PROFILE).to_dict()
    raw['root'] = str(tmp_path)
    raw['inference']['external_pool_max_n'] = 32
    raw['inference']['external_pool_max_concurrent_requests'] = 4
    raw['inference']['external_pool_urls'] = ({raw['model']: ['http://farm1', 'http://farm2']} if enabled else {})
    return Config.from_dict(raw)


class Farm:
    def __init__(self):
        self.owners = {}
        self.requests = []
        self.digest = None
        self.wrong_hash = False
        self.busy = set()
        self.down = set()
        self.lost_acquire = False
        self.fail_generate = False
        self.fail_heartbeat = False
        self.heartbeat_override = {}
        self.heartbeat_error = None
        self.heartbeat_gate = None
        self.heartbeat_seen = asyncio.Event()
        self.fail_release = False
        self.max_n = 32
        self.release_seen = asyncio.Event()
        self.generation_started = asyncio.Event()
        self.generation_gate = None
        self.incarnation = 'farm-incarnation-1'

    def identity(self, lease):
        return dict(owner_run=lease['owner_run'], lease_id=lease['lease_id'], expires_at=2000000000)

    def status(self, lease):
        digest = 'wrong' if self.wrong_hash else lease.get('adapter_sha256')
        return dict(self.identity(lease), state='ready' if digest else 'awaiting_adapter',
                    adapter_name=lease.get('adapter_name'), adapter_sha256=digest,
                    expected_engines=4, ready_engines=4 if digest else 0)

    async def __call__(self, request):
        host, path = request.url.host, request.url.path
        self.requests.append((host, path))
        if host in self.down:
            raise httpx.ConnectError('offline', request=request)
        if path == '/health':
            return httpx.Response(200, json={})
        if path == '/v1/models':
            return httpx.Response(200, json={'data': [{'id': 'Qwen/Qwen3.5-27B'}]})
        if path == '/status' and 'X-Lease-ID' not in request.headers:
            # Legacy farms have no cancellable-acquire protocol.
            return httpx.Response(200, json={'instance': self.incarnation})
        if path == '/acquire_lease':
            body = json.loads(request.content)
            if body.get('lease_id'):
                self.heartbeat_seen.set()
                if self.heartbeat_gate:
                    await self.heartbeat_gate.wait()
                if self.heartbeat_error:
                    raise self.heartbeat_error
                lease = self.owners[host]
                assert body['lease_id'] == lease['lease_id']
                return httpx.Response(503 if self.fail_heartbeat else 200,
                                      json=self.status(lease) | self.heartbeat_override)
            if host in self.owners or host in self.busy:
                return httpx.Response(409, json={'state': 'busy'})
            lease = dict(body, lease_id='lease-'+host, adapter_name=None, adapter_sha256=None)
            self.owners[host] = lease
            if self.lost_acquire:
                raise httpx.ReadTimeout('ack lost', request=request)
            return httpx.Response(200, json=self.status(lease))
        lease = self.owners[host]
        assert request.headers['X-Lease-ID'] == lease['lease_id']
        if path.endswith('/upload_lora_adapter'):
            self.digest = hashlib.sha256(await request.aread()).hexdigest()
            assert self.digest == request.headers['X-Adapter-SHA256']
            lease['adapter_sha256'] = self.digest
            lease['adapter_name'] = request.url.params['lora_name']
            return httpx.Response(200, json={'sha256': self.digest,
                'lora_name': lease['adapter_name'], 'loaded': [str(i) for i in range(4)]})
        if path == '/release_lease':
            assert json.loads(request.content)['lease_id'] == lease['lease_id']
            self.release_seen.set()
            if self.fail_release:
                return httpx.Response(503, json={})
            del self.owners[host]
            return httpx.Response(200, json={'state': 'unleased', 'released': True})
        if path == '/v1/completions':
            self.generation_started.set()
            if self.generation_gate:
                await self.generation_gate.wait()
            if self.fail_generate:
                return httpx.Response(503, json={})
            payload = json.loads(request.content)
            assert payload['model'] == (lease['adapter_name'] or 'Qwen/Qwen3.5-27B')
            body = {'choices': [{'index': i, 'text': 'test', 'token_ids': [3, 4],
                    'logprobs': {'token_logprobs': [-.2, -.3]}, 'loss_mask': [0, 1],
                    'thinking_budget_audit': {'passed': True}} for i in range(payload.get('n', 1))]}
            return httpx.Response(200, json=body)
        assert path == '/status'
        return httpx.Response(200, json=self.status(lease))


def run_case(tmp_path, operation):
    async def run():
        cfg = config(tmp_path)
        archive = tmp_path/'adapter.tar'
        archive.write_bytes(b'production-shaped immutable archive bytes')
        farm = Farm()
        events = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(farm)) as client:
            borrower = Borrower(cfg, client, lambda event, **fields: events.append((event, fields)))
            try:
                await operation(borrower, farm, archive, events)
            finally:
                await borrower.end()
            assert 'secret-' not in json.dumps(events)
    asyncio.run(run())


def test_ready_generation_preserves_native_fields_and_releases(tmp_path):
    async def test(b, farm, archive, events):
        state = await b.begin('phase1', 'local-adapter', archive)
        assert state['ready'] and state['engines'] == 4
        assert farm.digest == hashlib.sha256(archive.read_bytes()).hexdigest()
        result = await b.generate({'model': 'local-adapter', 'n': 2}, 0, 4)
        assert len(result['choices']) == 2
        assert result['choices'][0]['loss_mask'] == [0, 1]
        assert result['choices'][0]['logprobs']['token_logprobs'] == [-.2, -.3]
        assert result['choices'][0]['thinking_budget_audit'] == {'passed': True}
        await b.end('phase1')
        assert not farm.owners
    run_case(tmp_path, test)


def test_concurrent_begin_claims_at_most_one_service(tmp_path):
    async def test(b, farm, archive, events):
        await asyncio.gather(*(b.begin('phase1', 'adapter', archive) for _ in range(12)))
        assert len(farm.owners) == 1
        assert sum(p == '/acquire_lease' for _, p in farm.requests) == 1
    run_case(tmp_path, test)


def test_two_jobs_cannot_share_a_reserved_service(tmp_path):
    async def test(b, farm, archive, events):
        other = Borrower(b.config, b.http)
        try:
            await asyncio.gather(b.begin('phase1', 'adapter', archive), other.begin('phase2', 'adapter', archive))
            assert b.lease.url != other.lease.url
            assert len(farm.owners) == 2
        finally:
            await other.end()
    run_case(tmp_path, test)


@pytest.mark.parametrize('condition', ['busy', 'down'])
def test_unavailable_first_service_tries_second(tmp_path, condition):
    async def test(b, farm, archive, events):
        getattr(farm, condition).add('farm1')
        assert (await b.begin('phase1', 'adapter', archive))['ready']
        assert b.lease.url == 'http://farm2'
    run_case(tmp_path, test)


def test_both_busy_stays_local(tmp_path):
    async def test(b, farm, archive, events):
        farm.busy.update(['farm1', 'farm2'])
        assert not (await b.begin('phase1', 'adapter', archive))['ready']
        assert await b.generate({'model': 'adapter'}, 50, 4) is None
    run_case(tmp_path, test)


def test_lost_acquire_does_not_claim_second_service_or_next_phase(tmp_path):
    async def test(b, farm, archive, events):
        farm.lost_acquire = True
        assert not (await b.begin('phase1', 'adapter', archive))['ready']
        assert list(farm.owners) == ['farm1']
        assert b.uncertain_until == float('inf')
        await b.end('phase1')
        await b.begin('phase2', 'adapter', archive)
        assert sum(p == '/acquire_lease' for _, p in farm.requests) == 1
    run_case(tmp_path, test)


def test_wrong_hash_never_receives_generation(tmp_path):
    async def test(b, farm, archive, events):
        farm.wrong_hash = True
        assert not (await b.begin('phase1', 'adapter', archive))['ready']
        assert not farm.owners
        assert await b.generate({'model': 'adapter'}, 50, 4) is None
        assert not any(p == '/v1/completions' for _, p in farm.requests)
    run_case(tmp_path, test)


def test_generation_failure_falls_back_and_disables_service(tmp_path):
    async def test(b, farm, archive, events):
        await b.begin('phase1', 'adapter', archive)
        farm.fail_generate = True
        assert await b.generate({'model': 'adapter'}, 50, 4) is None
        assert not b.snapshot()['ready']
    run_case(tmp_path, test)


def test_advertised_n_and_expiry_are_enforced(tmp_path):
    async def test(b, farm, archive, events):
        b.settings = replace(b.settings, external_pool_max_n=1)
        await b.begin('phase1', 'adapter', archive)
        assert await b.generate({'model': 'adapter', 'n': 32}, 50, 4) is None
        b.lease.valid_until = 0
        assert await b.generate({'model': 'adapter'}, 50, 4) is None
    run_case(tmp_path, test)


def test_release_drains_active_generation(tmp_path):
    async def test(b, farm, archive, events):
        await b.begin('phase1', 'adapter', archive)
        farm.generation_gate = asyncio.Event()
        generation = asyncio.create_task(b.generate({'model': 'adapter'}, 50, 4))
        await farm.generation_started.wait()
        release = asyncio.create_task(b.end('phase1'))
        await asyncio.sleep(.01)
        assert not farm.release_seen.is_set()
        farm.generation_gate.set()
        assert await generation
        await release
        assert farm.release_seen.is_set()
    run_case(tmp_path, test)


def test_stale_release_cannot_end_new_phase(tmp_path):
    async def test(b, farm, archive, events):
        await b.begin('phase2', 'adapter', archive)
        assert not await b.end('phase1')
        assert b.snapshot()['ready']
    run_case(tmp_path, test)


def test_base_bootstrap_needs_no_upload(tmp_path):
    async def test(b, farm, archive, events):
        assert (await b.begin('bootstrap', b.config.model))['ready']
        assert not any(p.endswith('/upload_lora_adapter') for _, p in farm.requests)
        assert await b.generate({'model': b.config.model}, 0, 4)
    run_case(tmp_path, test)


def test_cancelled_generation_cleans_up(tmp_path):
    async def test(b, farm, archive, events):
        await b.begin('phase1', 'adapter', archive)
        farm.generation_gate = asyncio.Event()
        request = asyncio.create_task(b.generate({'model': 'adapter'}, 0, 4))
        await farm.generation_started.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert b.lease.active == 0
        await b.end()
        assert not farm.owners
    run_case(tmp_path, test)


@pytest.mark.parametrize('change', [
    {'routing': 'direct'}, {'external_pool_urls': ['http://farm']},
    {'external_pool_urls': {'Qwen/Qwen3.5-27B': ['http://user:pass@farm']}},
    {'external_pool_urls': {'Qwen/Qwen3.5-27B': ['http://farm/']}},
    {'external_pool_lease_seconds': 30}, {'external_pool_engines': 0},
    {'external_pool_initial_wait_seconds': 0},
])
def test_reject_invalid_configuration(tmp_path, change):
    raw = config(tmp_path).to_dict()
    raw['inference'].update(change)
    with pytest.raises(ValueError):
        Config.from_dict(raw)


def test_phase_hook_releases_on_failure(tmp_path, monkeypatch):
    from tpu.swarm.ray_train import borrowing_phase
    calls = []
    async def transport(request):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={})
    original_client = httpx.AsyncClient
    monkeypatch.setattr(borrowing_phase.httpx, 'AsyncClient',
                        lambda: original_client(transport=httpx.MockTransport(transport)))
    monkeypatch.setenv('SKYRL_BORROWING_URL', 'http://local')
    async def original(*args):
        calls.append(('sampling', None))
        raise RuntimeError('sampling failed')
    ensemble = SimpleNamespace(_pipelined_sampling_phase=original)
    cfg = SimpleNamespace(pipeline_dataflow=True, distill_enabled=False, num_substeps=1, pooled_multi_lora=False)
    borrowing_phase.install(ensemble, cfg)
    async def run():
        with pytest.raises(RuntimeError, match='sampling failed'):
            await ensemble._pipelined_sampling_phase()
    asyncio.run(run())
    assert [p.rsplit('/', 1)[-1] for p, _ in calls] == ['begin', 'sampling', 'end']
    assert calls[0][1]['phase_id'] == calls[-1][1]['phase_id']


def test_persistent_heartbeat_failure_interrupts_hung_generation(tmp_path):
    async def test(b, farm, archive, events):
        # Speed up only the watchdog clock; production remains 30s/300s.
        b.settings = replace(b.settings, external_pool_heartbeat_seconds=.01,
                             external_pool_health_grace_seconds=.04)
        await b.begin('phase1', 'adapter', archive)
        farm.generation_gate = asyncio.Event()
        request = asyncio.create_task(b.generate({'model': 'adapter'}, 0, 4))
        await farm.generation_started.wait()
        farm.fail_heartbeat = True
        assert await asyncio.wait_for(request, timeout=.5) is None
        assert not b.snapshot()['ready']
        assert b.lease.active == 0
        assert await b.generate({'model': 'adapter'}, 50, 4) is None
    run_case(tmp_path, test)


def test_expiry_interrupts_hung_generation(tmp_path):
    async def test(b, farm, archive, events):
        import time
        await b.begin('phase1', 'adapter', archive)
        b.lease.valid_until = time.monotonic() + .02
        farm.generation_gate = asyncio.Event()
        assert await asyncio.wait_for(b.generate({'model': 'adapter'}, 0, 4), timeout=.5) is None
    run_case(tmp_path, test)


def test_disabled_hook_and_environment_are_unchanged(tmp_path, monkeypatch):
    from tpu.swarm.ray_train.borrowing_phase import install
    from tpu.swarm.ray_train.commands import client_environment
    cfg = config(tmp_path, enabled=False)
    env = client_environment(cfg, tmp_path, '10.0.0.1')
    assert not any(key.startswith('SKYRL_BORROWING_') for key in env)
    monkeypatch.delenv('SKYRL_BORROWING_URL', raising=False)
    original = object()
    module = SimpleNamespace(_pipelined_sampling_phase=original)
    install(module, None)
    assert module._pipelined_sampling_phase is original


def test_client_disappears_releases_remote_and_interrupts_pending_request(tmp_path):
    async def test(b, farm, archive, events):
        import time
        b.settings = replace(b.settings, external_pool_heartbeat_seconds=.01)
        await b.begin('phase1', 'adapter', archive)
        farm.generation_gate = asyncio.Event()
        request = asyncio.create_task(b.generate({'model': 'adapter'}, 0, 4))
        await farm.generation_started.wait()
        b.phase_deadline = time.monotonic() - 1
        assert await asyncio.wait_for(request, .5) is None
        await asyncio.wait_for(farm.release_seen.wait(), .5)
        assert not farm.owners
        assert not b.touch('phase1-stale')
    run_case(tmp_path, test)


def test_unsupported_group_size_does_not_claim_idle_farm(tmp_path):
    async def test(b, farm, archive, events):
        b.settings = replace(b.settings, external_pool_max_n=1)
        state = await b.begin('phase1', 'adapter', archive, expected_n=32)
        assert not state['ready']
        assert not farm.requests
    run_case(tmp_path, test)


def test_telemetry_failure_does_not_break_fallback(tmp_path):
    async def test(b, farm, archive, events):
        def failed_report(*args, **kwargs):
            raise OSError('disk full')
        b.report = failed_report
        await b.begin('phase1', 'adapter', archive)
        farm.fail_generate = True
        assert await b.generate({'model': 'adapter'}, 0, 4) is None
    run_case(tmp_path, test)


def test_actual_ingress_retries_hung_remote_locally_without_changing_payload(tmp_path, monkeypatch):
    from tpu.swarm.ray_train import serving

    class Remote:
        def __init__(self, fn):
            self.fn = fn

        async def remote(self, *args):
            return self.fn(*args)

    async def run():
        cfg = config(tmp_path)
        local_payloads = []
        def generate(payload):
            local_payloads.append(payload)
            return {'choices': [{'index': i, 'text': 'local'} for i in range(payload['n'])]}
        gateway = serving.Ingress.func_or_class.__mro__[1](cfg.to_dict(),
            [SimpleNamespace(generate=Remote(generate)) for _ in range(4)], None,
            ['10.0.0.' + str(i) for i in range(1, 5)])
        gateway.version = 'local-step-7'
        gateway.versions.add(gateway.version)
        (gateway.archives / (gateway.version + '.tar')).write_bytes(b'committed weights')
        await gateway.http.aclose()
        farm = Farm()
        gateway.http = httpx.AsyncClient(transport=httpx.MockTransport(farm))
        gateway.borrower.http = gateway.http
        gateway.borrower.settings = replace(cfg.inference, external_pool_heartbeat_seconds=.01,
                                            external_pool_health_grace_seconds=.04)
        monkeypatch.setattr(serving.serve, 'get_replica_context', lambda: SimpleNamespace(servable_object=gateway))
        app = inspect.getclosurevars(serving.Ingress.func_or_class.__init__).nonlocals['frozen_app_or_func']
        payload = {'model': gateway.version, 'n': 2, 'prompt': [1, 2, 3], 'temperature': .8,
                   'max_tokens': 22000, 'logprobs': 1, 'native_thinking_budget': 16000}
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
                response = await client.post('/skyrl/v1/borrowing/begin', json={'phase_id': 'phase-1', 'expected_n': 2})
                assert response.status_code == 200 and response.json()['ready']
                farm.generation_gate = asyncio.Event()
                pending = asyncio.create_task(client.post('/v1/completions', json=payload))
                await farm.generation_started.wait()
                farm.fail_heartbeat = True
                response = await asyncio.wait_for(pending, 1)
                assert response.status_code == 200
                assert response.json()['choices'][0]['text'] == 'local'
                assert local_payloads == [payload]
                response = await client.post('/v1/completions', json=payload)
                assert response.status_code == 200 and local_payloads == [payload, payload]
                assert sum(path == '/v1/completions' for _, path in farm.requests) == 1
                assert gateway.active == 0
                response = await client.post('/skyrl/v1/borrowing/end', json={'phase_id': 'phase-1'})
                assert response.json()['ended']
        finally:
            await gateway.borrower.end()
            await gateway.http.aclose()
    asyncio.run(run())


def test_borrowing_bundle_contains_ingress_and_frozen_client_hook(tmp_path, monkeypatch):
    import tarfile
    from tpu.swarm.ray_train import build, overlay
    cfg = config(tmp_path)
    monkeypatch.setattr(build.Config, 'load', lambda profile: cfg)
    archive, _, _ = build.build(PROFILE, tmp_path / 'bundle')
    with tarfile.open(archive) as bundle:
        names = set(bundle.getnames())
        assert 'tpu/swarm/ray_train/borrowing.py' in names
        prefix = 'tpu/swarm/ray_train/source_overlay/'
        assert prefix + 'tpu/swarm/ray_train/borrowing_phase.py' in names
        records = json.load(bundle.extractfile(prefix + 'manifest.json'))
        stage = tmp_path / 'overlay'
        for name in records:
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bundle.extractfile(prefix + name).read())
        (stage / 'manifest.json').write_text(json.dumps(records))
    overlay.install(stage, tmp_path / 'frozen-client')
    assert (tmp_path / 'frozen-client/tpu/swarm/ray_train/borrowing_phase.py').is_file()


def test_client_heartbeats_during_sampling_then_stops_on_release(monkeypatch):
    from tpu.swarm.ray_train import borrowing_phase
    calls = []
    heartbeat_seen = asyncio.Event()
    async def transport(request):
        calls.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith('/heartbeat'):
            heartbeat_seen.set()
        return httpx.Response(200, json={})
    original = httpx.AsyncClient
    monkeypatch.setattr(borrowing_phase.httpx, 'AsyncClient', lambda: original(transport=httpx.MockTransport(transport)))
    monkeypatch.setenv('SKYRL_BORROWING_HEARTBEAT_SECONDS', '.01')
    async def run():
        async with borrowing_phase.sampling_phase('http://local', expected_n=8):
            await asyncio.wait_for(heartbeat_seen.wait(), .5)
        count = len(calls)
        await asyncio.sleep(.03)
        assert len(calls) == count
    asyncio.run(run())
    assert calls[0][1]['expected_n'] == 8
    assert calls[-1][0].endswith('/end')
    assert len({body['phase_id'] for _, body in calls}) == 1


async def wait_event(events, name):
    async def wait():
        while not any(event == name for event, _ in events):
            await asyncio.sleep(.001)
    await asyncio.wait_for(wait(), 1)


@pytest.mark.parametrize('failure', ['degraded', '503', 'timeout'])
@pytest.mark.parametrize('complete_while_paused', [False, True])
def test_transient_health_failure_preserves_inflight_and_blocks_new_work(tmp_path, failure, complete_while_paused):
    async def test(b, farm, archive, events):
        b.settings = replace(b.settings, external_pool_heartbeat_seconds=.01,
                             external_pool_health_grace_seconds=.3)
        await b.begin('phase1', 'adapter', archive)
        farm.generation_gate = asyncio.Event()
        pending = [asyncio.create_task(b.generate({'model': 'adapter', 'n': 32}, 4, 4)) for _ in range(4)]
        await farm.generation_started.wait()
        if failure == 'degraded':
            farm.heartbeat_override = {'state': 'degraded', 'ready_engines': 3}
        elif failure == '503':
            farm.fail_heartbeat = True
        else:
            farm.heartbeat_error = httpx.ReadTimeout('secret-transport-detail')
        await wait_event(events, 'borrow_health_paused')
        assert all(not task.done() for task in pending)
        assert await b.generate({'model': 'adapter'}, 10, 4) is None
        assert sum(path == '/v1/completions' for _, path in farm.requests) == 4
        if complete_while_paused:
            farm.generation_gate.set()
            results = await asyncio.wait_for(asyncio.gather(*pending), 1)
        farm.heartbeat_override = {}
        farm.fail_heartbeat = False
        farm.heartbeat_error = None
        await wait_event(events, 'borrow_health_recovered')
        if not complete_while_paused:
            farm.generation_gate.set()
            results = await asyncio.wait_for(asyncio.gather(*pending), 1)
        assert all(len(result['choices']) == 32 for result in results)
        assert b.snapshot()['ready']
        assert (await b.generate({'model': 'adapter'}, 0, 4))['choices'][0]['text'] == 'test'
        assert not any(event in ('borrow_generation_failed', 'borrow_heartbeat_failed') for event, _ in events)
    run_case(tmp_path, test)


def test_persistent_degradation_has_fixed_grace_despite_successful_renewals(tmp_path):
    async def test(b, farm, archive, events):
        b.settings = replace(b.settings, external_pool_heartbeat_seconds=.01,
                             external_pool_health_grace_seconds=.06)
        await b.begin('phase1', 'adapter', archive)
        farm.generation_gate = asyncio.Event()
        pending = asyncio.create_task(b.generate({'model': 'adapter'}, 0, 4))
        await farm.generation_started.wait()
        farm.heartbeat_override = {'state': 'degraded', 'ready_engines': 3}
        await wait_event(events, 'borrow_health_paused')
        assert not pending.done()
        assert await asyncio.wait_for(pending, .5) is None
        assert sum(event == 'borrow_health_paused' for event, _ in events) == 1
        assert sum(path == '/acquire_lease' for _, path in farm.requests) >= 3
        assert not b.snapshot()['ready']
    run_case(tmp_path, test)


@pytest.mark.parametrize('change', [
    {'adapter_sha256': 'different'}, {'adapter_name': 'different'},
    {'owner_run': 'different'}, {'lease_id': 'different'},
    {'state': 'expired'}, {'state': 'draining'}, {'state': 'updating'},
    {'expected_engines': 8}, {'ready_engines': -1},
])
def test_identity_or_contract_change_is_immediately_fatal_even_during_grace(tmp_path, change):
    async def test(b, farm, archive, events):
        b.settings = replace(b.settings, external_pool_heartbeat_seconds=.01,
                             external_pool_health_grace_seconds=10)
        await b.begin('phase1', 'adapter', archive)
        farm.generation_gate = asyncio.Event()
        pending = asyncio.create_task(b.generate({'model': 'adapter'}, 0, 4))
        await farm.generation_started.wait()
        farm.heartbeat_override = {'state': 'degraded', 'ready_engines': 3}
        await wait_event(events, 'borrow_health_paused')
        farm.heartbeat_override.update(change)
        assert await asyncio.wait_for(pending, .5) is None
        assert not b.snapshot()['ready']
        assert any(event == 'borrow_heartbeat_failed' for event, _ in events)
    run_case(tmp_path, test)


def test_unacknowledged_renewal_never_extends_lease_past_expiry(tmp_path):
    async def test(b, farm, archive, events):
        import time
        b.settings = replace(b.settings, external_pool_heartbeat_seconds=.01,
                             external_pool_health_grace_seconds=10)
        await b.begin('phase1', 'adapter', archive)
        deadline = b.lease.valid_until = time.monotonic() + .08
        farm.generation_gate = asyncio.Event()
        farm.fail_heartbeat = True
        assert await asyncio.wait_for(b.generate({'model': 'adapter'}, 0, 4), .5) is None
        assert b.lease.valid_until == deadline
    run_case(tmp_path, test)


def test_late_renewal_does_not_resurrect_expired_lease(tmp_path):
    async def test(b, farm, archive, events):
        import time
        b.settings = replace(b.settings, external_pool_heartbeat_seconds=.01)
        await b.begin('phase1', 'adapter', archive)
        b.lease.valid_until = time.monotonic() + .07
        farm.heartbeat_gate = asyncio.Event()
        farm.generation_gate = asyncio.Event()
        pending = asyncio.create_task(b.generate({'model': 'adapter'}, 0, 4))
        await farm.heartbeat_seen.wait()
        assert await asyncio.wait_for(pending, .5) is None
        farm.heartbeat_gate.set()
        await asyncio.wait_for(b.lease.heartbeat, .5)
        assert not b.snapshot()['ready']
        assert await b.generate({'model': 'adapter'}, 0, 4) is None
    run_case(tmp_path, test)


def test_recovery_during_release_cannot_reopen_admission(tmp_path):
    async def test(b, farm, archive, events):
        b.settings = replace(b.settings, external_pool_heartbeat_seconds=.01,
                             external_pool_health_grace_seconds=1)
        await b.begin('phase1', 'adapter', archive)
        farm.generation_gate = asyncio.Event()
        pending = asyncio.create_task(b.generate({'model': 'adapter'}, 0, 4))
        await farm.generation_started.wait()
        farm.heartbeat_override = {'state': 'degraded', 'ready_engines': 3}
        await wait_event(events, 'borrow_health_paused')
        release = asyncio.create_task(b.end('phase1'))
        await asyncio.sleep(0)
        farm.heartbeat_override = {}
        await wait_event(events, 'borrow_health_recovered')
        assert not release.done()
        assert await b.generate({'model': 'adapter'}, 50, 4) is None
        farm.generation_gate.set()
        assert await asyncio.wait_for(pending, .5) is not None
        await asyncio.wait_for(release, .5)
        assert not farm.owners
    run_case(tmp_path, test)


def test_service_list_update_preserves_active_lease_and_changes_next_acquisition(tmp_path):
    async def test(b, farm, archive, events):
        b.settings = replace(b.settings, external_pool_updates=True)
        await b.begin('phase1', 'adapter', archive)
        lease = b.lease
        assert lease.url == 'http://farm1'
        b.update_urls(b.config.model, ['http://farm2'])
        assert b.lease is lease and b.snapshot()['ready']
        assert (await b.generate({'model': 'adapter'}, 0, 4))['choices']
        b.update_urls(b.config.model, [])
        assert b.lease is lease and b.snapshot()['ready']
        await b.end()
        b.update_urls(b.config.model, ['http://farm2'])
        await b.begin('phase2', 'adapter', archive)
        assert b.lease.url == 'http://farm2'
    run_case(tmp_path, test)


def test_dynamic_service_api_is_opt_in_and_fences_target_instance(tmp_path, monkeypatch):
    from tpu.swarm.ray_train import serving
    from tpu.swarm.ray_train.commands import client_environment
    async def run():
        raw = config(tmp_path, enabled=False).to_dict()
        raw['inference']['external_pool_updates'] = True
        cfg = Config.from_dict(raw)
        assert cfg.borrows_inference
        assert 'SKYRL_BORROWING_URL' in client_environment(cfg, tmp_path, '10.0.0.1')
        gateway = serving.Ingress.func_or_class.__mro__[1](cfg.to_dict(), [], None, [])
        monkeypatch.setattr(serving.serve, 'get_replica_context', lambda: SimpleNamespace(servable_object=gateway))
        app = inspect.getclosurevars(serving.Ingress.func_or_class.__init__).nonlocals['frozen_app_or_func']
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
                path = '/skyrl/v1/borrowing/services'
                descriptor = (await client.get(path)).json()
                assert descriptor['enabled'] and descriptor['urls'] == []
                payload = {k:descriptor[k] for k in ('model', 'run_id', 'instance')}
                payload['urls'] = ['http://farm1']
                response = await client.post(path, json=payload)
                assert response.status_code == 200 and response.json()['urls'] == ['http://farm1']
                for change, status in [({'model': 'wrong'}, 400), ({'instance': 'stale'}, 409),
                        ({'run_id': 'other'}, 409), ({'urls': ['file:///tmp']}, 400),
                        ({'urls': None}, 400), ({'urls': ['http://user:secret@farm']}, 400)]:
                    assert (await client.post(path, json=payload | change)).status_code == status
                    assert gateway.borrower.urls == ['http://farm1']
                gateway.config = replace(cfg, inference=replace(cfg.inference, external_pool_updates=False))
                assert (await client.post(path, json=payload)).status_code == 409
        finally:
            await gateway.http.aclose()
    asyncio.run(run())
