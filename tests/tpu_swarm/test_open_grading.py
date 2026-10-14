"""Open grading: lease-free admission on farms, per-run fairness, and the client's open farm directory."""
import asyncio
import json
import uuid

import httpx

from tpu.science import grading_transport as gt
from tpu.swarm.ray_train import grading_service
from tests.tpu_swarm.test_farm_grading_service import body, grading_farm, lease_for


def open_headers(run='open-run', token='t1'):
    return {'X-Grading-Run': run, 'X-Grading-Token': token}


def finish(future, host='10.0.0.1'):
    future.set_result(dict(result=[1.0], error=None, stdout='', metrics={'host': host}))


def test_open_submit_admission_ownership_and_metadata(tmp_path, monkeypatch):
    async def run():
        async with grading_farm(tmp_path, monkeypatch, slots_per_host=1) as (client, gateway, service, futures, _, _e):
            # No lease and no open identity: refused as before.
            assert (await client.post('/skyrl/v1/grading/submit', json=body())).status_code == 409
            capacity = (await client.get('/skyrl/v1/grading/capacity', headers=open_headers())).json()
            assert capacity['open'] and capacity['grader_sha256'] == service.grader_sha256
            assert capacity['families']['math']['open_limit'] == 6
            requests = [body() for _ in range(7)]
            statuses = [(await client.post('/skyrl/v1/grading/submit', json=dict(r, grader_sha256=service.grader_sha256),
                                           headers=open_headers())).status_code for r in requests]
            assert statuses == [202] * 6 + [429]  # 4 slots x 1.5
            full = await client.post('/skyrl/v1/grading/submit', json=body(), headers=open_headers())
            assert full.json()['total'] == 4 and full.json()['running'] == 4 and full.json()['queued'] == 2
            await asyncio.sleep(0)
            finish(futures[0])
            await asyncio.sleep(.02)
            rid = requests[0]['request_id']
            # Only the submitting token can read or cancel it.
            assert (await client.get(f'/skyrl/v1/grading/result/{rid}', headers=open_headers(token='t2'))).status_code == 404
            view = (await client.get(f'/skyrl/v1/grading/result/{rid}', headers=open_headers())).json()
            assert view['state'] == 'done'
            metrics = view['result']['metrics']
            assert metrics['grader_sha256'] == service.grader_sha256 and metrics['grading_cpu_model']
            assert 'grading_host_class' in metrics and metrics['host'] == '10.0.0.1'
    asyncio.run(run())


def test_grader_version_mismatch_is_refused(tmp_path, monkeypatch):
    async def run():
        async with grading_farm(tmp_path, monkeypatch) as (client, gateway, service, futures, _, _e):
            response = await client.post('/skyrl/v1/grading/submit', json=dict(body(), grader_sha256='0' * 64),
                                         headers=open_headers())
            assert response.status_code == 409 and 'grader version' in response.json()['detail']
            assert not futures
    asyncio.run(run())


def test_lease_release_cancels_only_leased_work(tmp_path, monkeypatch):
    async def run():
        async with grading_farm(tmp_path, monkeypatch) as (client, gateway, service, futures, cancelled, _e):
            headers = await lease_for(client)
            leased, opened = body(), body()
            await client.post('/skyrl/v1/grading/submit', json=leased, headers=headers)
            await client.post('/skyrl/v1/grading/submit', json=opened, headers=open_headers())
            await asyncio.sleep(0)
            await client.post('/release_lease', json={'lease_id': headers['X-Lease-ID']})
            await asyncio.sleep(.02)
            assert service.entries[leased['request_id']].state == 'cancelled'
            assert service.entries[opened['request_id']].state == 'running'
    asyncio.run(run())


def test_runs_take_turns_for_free_slots(tmp_path, monkeypatch):
    async def run():
        async with grading_farm(tmp_path, monkeypatch, slots_per_host=2) as (client, gateway, service, futures, _, _e):
            def tagged(tag):
                return body(program_code=f'def construct_function():\n    return [{tag!r}]\n')

            async def submit(tag, run):
                response = await client.post('/skyrl/v1/grading/submit', json=tagged(tag), headers=open_headers(run, 't' + run))
                assert response.status_code == 202
            # Run A takes all 8 slots and queues two more; then B queues one.
            for i in range(10):
                await submit(f'a{i}', 'A')
            await submit('b0', 'B')
            await asyncio.sleep(0)
            assert len(futures) == 8
            for future in futures[:2]:
                finish(future)
            await asyncio.sleep(.05)
            started = [f.spec['program_code'].split("'")[1] for f in futures[8:]]
            # B is not stuck behind A's backlog: the freed slots alternate between runs.
            assert sorted(started) == ['a8', 'b0']
    asyncio.run(run())


class OpenFarms:
    """Fake open-grading farms by host name: full, old (no open grading), other grader, flaky, good."""

    def __init__(self, grader):
        self.grader = grader
        self.kind = {}
        self.submits = []
        self.polls = {}

    async def __call__(self, request):
        await asyncio.sleep(0)
        host, path = request.url.host, request.url.path
        kind = self.kind[host]
        if kind == 'old':
            # Built before open grading: no lease means 409 everywhere.
            return httpx.Response(409, json={'detail': 'missing or stale lease ID'})
        assert request.headers['X-Grading-Run'] == 'run-x' and request.headers['X-Grading-Token']
        if path == '/skyrl/v1/grading/capacity':
            grader = self.grader if kind != 'other' else 'f' * 64
            return httpx.Response(200, json={'ready': True, 'open': True, 'grader_sha256': grader, 'families': {
                'routing': {'total': 8, 'running': 0, 'queued': 0}}})
        if path == '/skyrl/v1/grading/submit':
            payload = json.loads(request.content)
            assert payload['grader_sha256'] == self.grader
            if kind == 'full':
                return httpx.Response(429, json={'detail': 'grading queue full', 'retry_after': 30})
            self.submits.append(host)
            return httpx.Response(202, json={'request_id': payload['request_id'], 'state': 'queued'})
        if path == '/skyrl/v1/grading/results':
            # A farm built before batch polling: the client falls back to one poll per id.
            return httpx.Response(404, json={'detail': 'Not Found'})
        if path.startswith('/skyrl/v1/grading/result/'):
            if kind == 'flaky':
                raise httpx.ConnectError('farm went away', request=request)
            return httpx.Response(200, json={'request_id': path.rsplit('/', 1)[1], 'state': 'done', 'error': None,
                                             'result': {'reward': 1.0, 'metrics': {'grading_cpu_model': host}}})
        if path.startswith('/skyrl/v1/grading/cancel/'):
            return httpx.Response(200, json={})
        raise AssertionError(path)


def registry(tmp_path, hosts):
    path = tmp_path / 'farms.json'
    path.write_text(json.dumps(dict(farms=[dict(name=h, url=f'http://{h}', ok=True, tunnel_ok=True, models=['m'])
                                           for h in hosts])))
    return path


def test_client_grades_on_open_farms_skipping_full_old_and_other_versions(tmp_path, monkeypatch):
    monkeypatch.setenv('SKYRL_RUN_ID', 'run-x')
    grader = gt.local_grader_sha256()
    assert grader and grader == grading_service.grader_sha256()
    farms = OpenFarms(grader)
    farms.kind.update(full='full', old='old', other='other', good='good')

    async def run():
        http = httpx.AsyncClient(transport=httpx.MockTransport(farms))
        transport = gt.GradingTransport(families={}, local_capacity=0, http=http, loop=asyncio.get_running_loop(),
                                        open_registry=str(registry(tmp_path, farms.kind)), open_k=4)
        result = await asyncio.wait_for(transport.grade(gt.GradingRequest(task='routing', spec={'source': 'x'})), 30)
        pools = transport.open_directory.pools
        assert not pools['old'].supported and not pools['other'].supported
        assert result['metrics']['grading_cpu_model'] == 'good' and result['metrics']['grading_open'] is True
        assert farms.submits == ['good']
        await transport.close()
    asyncio.run(run())


def test_open_grade_reroutes_when_a_farm_stops_answering(tmp_path, monkeypatch):
    monkeypatch.setenv('SKYRL_RUN_ID', 'run-x')
    farms = OpenFarms(gt.local_grader_sha256())
    farms.kind.update(flaky='flaky', good='good')

    async def run():
        http = httpx.AsyncClient(transport=httpx.MockTransport(farms))
        transport = gt.GradingTransport(families={}, local_capacity=0, http=http, loop=asyncio.get_running_loop(),
                                        open_registry=str(registry(tmp_path, farms.kind)), open_k=4)
        transport.scheduler.choose = lambda pools, now, task=None: next(
            (p for name in ('flaky', 'good') for p in pools if p.name == name and p.healthy(now)), None)
        result = await asyncio.wait_for(transport.grade(gt.GradingRequest(task='routing', spec={'source': 'x'})), 30)
        assert farms.submits == ['flaky', 'good'] and result['metrics']['grading_pool'] == 'good'
        await transport.close()
    asyncio.run(run())


def test_only_k_least_loaded_open_farms_take_new_grades(tmp_path, monkeypatch):
    monkeypatch.setenv('SKYRL_RUN_ID', 'run-x')
    hosts = [f'f{i}' for i in range(6)]

    async def handler(request):
        load = int(request.url.host[1:])
        return httpx.Response(200, json={'ready': True, 'open': True, 'grader_sha256': None, 'families': {
            'routing': {'total': 10, 'running': load, 'queued': 0}}})

    async def run():
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        directory = gt.OpenFarmDirectory(str(registry(tmp_path, hosts)), http, k=4)
        await directory.refresh()
        assert sorted(n for n, p in directory.pools.items() if p.eligible) == ['f0', 'f1', 'f2', 'f3']
        await http.aclose()
    asyncio.run(run())


def test_runner_passes_open_grading_to_its_client(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tpu.swarm.ray_train import local_notrain
    monkeypatch.setenv('SKYRL_OPEN_GRADING', '1')
    monkeypatch.setenv('RAY_AUTH_TOKEN', uuid.uuid4().hex)
    captured = {}
    monkeypatch.setattr(local_notrain, 'client_environment', lambda config, root, ip: captured.setdefault('env', {}))
    config = SimpleNamespace(run_id='rq4-x', ports=SimpleNamespace(inference=1))
    args = SimpleNamespace(hf_home=tmp_path, package=tmp_path, source=tmp_path, registry=tmp_path / 'farms.json',
                           num_epochs=0)
    env = local_notrain.child_environment(config, tmp_path, args, 'gcs')
    assert env['SKYRL_OPEN_GRADING'] == '1' and env['SKYRL_GRADING_REGISTRY'] == str(tmp_path / 'farms.json')
    assert env['SKYRL_OPEN_GRADING_K'] == '4' and env['SKYRL_RUN_ID'] == 'rq4-x'
