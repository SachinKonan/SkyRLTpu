"""Grader gateway: one della grading endpoint for GCP trainers, behind one reverse SSH tunnel per trainer host."""
import asyncio
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

import httpx
import pytest

from tpu.science import grading_transport as gt
from tpu.swarm.ray_train import discovery, grader_gateway as gg
from test_discovery import args as base_args

REPO = Path(__file__).resolve().parents[2]
GRADER = 'a' * 64


# ------------------------------------------------------------------------------------------------ fixtures
class Router(httpx.AsyncBaseTransport):
    """Route by host: the gateway app, real farm apps and fake farms in one client."""

    def __init__(self, routes):
        self.routes = routes

    async def handle_async_request(self, request):
        return await self.routes[request.url.host].handle_async_request(request)


class FakeFarm:
    """An open-grading farm: per-family slots, load from other clients, 1.5x admission, batch results."""

    def __init__(self, families, grader=GRADER, others=None):
        self.families = families              # family -> slots
        self.grader, self.instance = grader, 'i1'
        self.others = dict(others or {})      # family -> load of clients we do not see
        self.entries = {}                     # id -> dict(task, state, headers, result)
        self.submits, self.cancels = [], []
        self.ready, self.draining = True, False

    def load(self, task):
        return self.others.get(task, 0) + sum(1 for e in self.entries.values()
                                              if e['task'] == task and e['state'] in ('queued', 'running'))

    def ours(self, task=None):
        return sum(1 for e in self.entries.values() if e['state'] in ('queued', 'running')
                   and (task is None or e['task'] == task))

    def finish(self, wire_id, reward=1.0, state='done', error=None):
        self.entries[wire_id].update(state=state, result=None if state != 'done' else
                                     dict(reward=reward, error=None, metrics=dict(grading_farm=self.instance)),
                                     error=error)

    def view(self, wire_id):
        e = self.entries[wire_id]
        return dict(request_id=wire_id, task=e['task'], state=e['state'], result=e.get('result'),
                    error=e.get('error'), host=None, metrics={})

    def owned(self, request, wire_id):
        e = self.entries.get(wire_id)
        return e is not None and e['headers'] == (request.headers['x-grading-run'], request.headers['x-grading-token'])

    async def __call__(self, request):
        path = request.url.path
        if path == '/skyrl/v1/grading/capacity':
            return httpx.Response(200, json=dict(ready=self.ready, draining=self.draining, open=True,
                                                 grader_sha256=self.grader, instance=self.instance, families={
                name: dict(total=total, running=self.load(name), queued=0) for name, total in self.families.items()}))
        if path == '/skyrl/v1/grading/submit':
            body = json.loads(request.content)
            if body.get('grader_sha256') not in (None, self.grader):
                return httpx.Response(409, json=dict(detail='grader version mismatch', grader_sha256=self.grader))
            task = body['task']
            if self.draining:
                return httpx.Response(429, json=dict(detail='farm draining', retry_after=600, draining=True))
            if self.load(task) >= self.families[task] * 1.5:
                return httpx.Response(429, json=dict(detail='grading queue full', retry_after=5))
            self.submits.append(body['request_id'])
            self.entries[body['request_id']] = dict(task=task, state='running', body=body, headers=(
                request.headers['x-grading-run'], request.headers['x-grading-token']))
            return httpx.Response(202, json=self.view(body['request_id']))
        if path == '/skyrl/v1/grading/results':
            ids = json.loads(request.content)['ids']
            return httpx.Response(200, json=dict(views={i: self.view(i) if self.owned(request, i) else None
                                                        for i in ids}))
        if path.startswith('/skyrl/v1/grading/cancel/'):
            wire_id = path.rsplit('/', 1)[1]
            self.cancels.append(wire_id)
            if self.owned(request, wire_id) and self.entries[wire_id]['state'] in ('queued', 'running'):
                self.entries[wire_id]['state'] = 'cancelled'
            return httpx.Response(200, json={})
        raise AssertionError(path)


def broker_for(farms, **kw):
    """A broker whose farm client reaches the fake farms by host name; the gateway app is host 'gateway'."""
    routes = {name: httpx.MockTransport(farm) for name, farm in farms.items()}
    broker = gg.Broker(httpx.AsyncClient(transport=Router(routes)), retry_seconds=0.01, long_poll_seconds=1, **kw)
    routes['gateway'] = httpx.ASGITransport(app=gg.create_app(broker))
    client = httpx.AsyncClient(transport=Router(routes), base_url='http://gateway')
    return broker, client, routes


async def settle(times=20):
    for _ in range(times):
        await asyncio.sleep(0.005)


async def add_farms(broker, farms):
    broker.set_farms([dict(url='http://%s' % name, name=name) for name in farms])
    await settle()


def heads(run='run-a', token='t-a'):
    return {'X-Grading-Run': run, 'X-Grading-Token': token}


def body(task='math', grader=GRADER, wire_id=None, **extra):
    return dict(request_id=wire_id or uuid.uuid4().hex, task=task, owner_run='x', scope={}, grader_sha256=grader,
                spec=dict(program_code='def f():\n    return 1\n', function_name='f', eval_timeout_seconds=60), **extra)


# --------------------------------------------------------------------------------- broker: routing and slots
def test_slot_accounting_counts_other_clients_and_never_oversubscribes():
    farms = dict(f1=FakeFarm(dict(math=4, placement=2), others=dict(math=3)), f2=FakeFarm(dict(math=2)))

    async def run():
        broker, client, _ = broker_for(farms)
        await add_farms(broker, farms)
        capacity = (await client.get('/skyrl/v1/grading/capacity', headers=heads())).json()
        assert capacity['open'] and capacity['ready'] and capacity['grader_sha256'] is None
        assert capacity['families']['math'] == dict(total=6, running=3, queued=0, open_limit=9)
        assert capacity['families']['placement']['total'] == 2
        statuses = [(await client.post('/skyrl/v1/grading/submit', json=body(), headers=heads())).status_code
                    for _ in range(10)]
        await settle()
        # Admission: 1.5 x 6 math slots outstanding; free slots: f1 4-3=1, f2 2: three dispatched, six queued here.
        assert statuses == [202] * 9 + [429]
        assert farms['f1'].ours('math') == 1 and farms['f2'].ours('math') == 2
        capacity = (await client.get('/skyrl/v1/grading/capacity', headers=heads())).json()
        assert capacity['grader_sha256'] == GRADER and capacity['families']['math']['queued'] == 6
        # A finished grade frees its slot, which the next queued grade takes at once.
        farms['f2'].finish(farms['f2'].submits[0])
        await settle()
        assert farms['f2'].ours('math') == 2 and farms['f1'].ours('math') == 1
        # The other client's load drops: the next probe lets the gateway use those slots.
        farms['f1'].others['math'] = 0
        await broker._probe(broker.farms['http://f1'])
        await settle()
        assert farms['f1'].ours('math') == 4
        for name, farm in farms.items():
            assert farm.load('math') <= farm.families['math'], name
        await broker.close()
    asyncio.run(run())


def test_result_reaches_the_trainer_and_terminal_views_pass_through():
    farms = dict(f1=FakeFarm(dict(math=2)))

    async def run():
        broker, client, _ = broker_for(farms)
        await add_farms(broker, farms)
        request = body()
        assert (await client.post('/skyrl/v1/grading/submit', json=request, headers=heads())).status_code == 202
        # Resubmitting the same id is idempotent; another client may not take it.
        assert (await client.post('/skyrl/v1/grading/submit', json=request, headers=heads())).status_code == 200
        assert (await client.post('/skyrl/v1/grading/submit', json=request, headers=heads('run-b', 't-b'))).status_code == 409
        await settle()
        wire_id = request['request_id']
        assert farms['f1'].entries[wire_id]['headers'] == ('run-a', 't-a')  # the trainer's identity, forwarded
        poll = asyncio.ensure_future(client.post('/skyrl/v1/grading/results', json=dict(ids=[wire_id], wait=5),
                                                 headers=heads()))
        await settle()
        assert not poll.done()  # a long poll waits for the result
        farms['f1'].finish(wire_id, reward=0.5)
        views = (await poll).json()['views']
        assert views[wire_id]['state'] == 'done' and views[wire_id]['result']['reward'] == 0.5
        assert (await client.get(f'/skyrl/v1/grading/result/{wire_id}', headers=heads('run-a', 'other'))).status_code == 404
        await broker.close()
    asyncio.run(run())


def test_runs_take_turns_and_a_new_run_is_admitted_its_share():
    farms = dict(f1=FakeFarm(dict(math=2)))

    async def run():
        broker, client, _ = broker_for(farms)
        await add_farms(broker, farms)
        a = [(await client.post('/skyrl/v1/grading/submit', json=body(), headers=heads('A', 'ta'))).status_code
             for _ in range(4)]
        assert a == [202, 202, 202, 429]  # 1.5 x 2 slots
        b = [(await client.post('/skyrl/v1/grading/submit', json=body(), headers=heads('B', 'tb'))).status_code
             for _ in range(3)]
        # Past the limit, B still gets its share (limit 3 / 2 runs -> 2); A gets nothing more.
        assert b == [202, 202, 429]
        assert (await client.post('/skyrl/v1/grading/submit', json=body(), headers=heads('A', 'ta'))).status_code == 429
        await settle()
        order = []
        for _ in range(3):
            first = next(i for i in farms['f1'].submits if farms['f1'].entries[i]['state'] == 'running')
            farms['f1'].finish(first)
            await settle()
            order.append(farms['f1'].entries[farms['f1'].submits[-1]]['headers'][0])
        # Freed slots alternate between runs: B is not stuck behind A's backlog.
        assert order == ['B', 'A', 'B']
        await broker.close()
    asyncio.run(run())


def test_priority_runs_dispatch_first_and_low_priority_keeps_the_reserve_free():
    farms = dict(f1=FakeFarm(dict(math=4)))

    async def run():
        broker, client, _ = broker_for(farms, priorities=gg.parse_priorities(['rl28k-=0']), reserve=0.25)
        await add_farms(broker, farms)
        # The benchmark run (default priority 1) is admitted up to the 1.5x limit but dispatched only while
        # more than the reserve (ceil(4 x 0.25) = 1 slot) stays free: 3 of 4 slots.
        low = [(await client.post('/skyrl/v1/grading/submit', json=body(), headers=heads('codex-a', 'tc'))).status_code
               for _ in range(6)]
        assert low == [202] * 6
        await settle()
        assert farms['f1'].ours() == 3
        # rl28k takes the reserved slot at once; its backlog (and only its) is admitted past the bench's 6.
        high = [(await client.post('/skyrl/v1/grading/submit', json=body(), headers=heads('rl28k-s1', 'tr'))).status_code
                for _ in range(3)]
        assert high == [202, 202, 202]
        await settle()
        running = [farms['f1'].entries[i]['headers'][0] for i in farms['f1'].submits]
        assert running.count('rl28k-s1') == 1 and farms['f1'].ours() == 4
        # While rl28k has grades queued here, the bench is refused.
        refused = await client.post('/skyrl/v1/grading/submit', json=body(), headers=heads('codex-a', 'tc'))
        assert refused.status_code == 429 and 'higher-priority' in refused.json()['detail']
        def running_by_run():
            return Counter(e['headers'][0] for e in farms['f1'].entries.values() if e['state'] == 'running')

        # Freed bench slots go to rl28k's queue first, whatever the round-robin turn.
        for _ in range(2):
            farms['f1'].finish(next(i for i in farms['f1'].submits if farms['f1'].entries[i]['state'] == 'running'
                                    and farms['f1'].entries[i]['headers'][0] == 'codex-a'))
            await settle()
        assert running_by_run() == {'rl28k-s1': 3, 'codex-a': 1}
        # rl28k's queue is empty now, but 1 free slot is only the reserve: the bench does not take it.
        farms['f1'].finish(next(i for i in farms['f1'].submits if farms['f1'].entries[i]['state'] == 'running'
                                and farms['f1'].entries[i]['headers'][0] == 'codex-a'))
        await settle()
        assert running_by_run() == {'rl28k-s1': 3}
        # When rl28k frees a slot, 2 are free and the bench may take one, keeping the reserve.
        farms['f1'].finish(next(i for i in farms['f1'].submits if farms['f1'].entries[i]['state'] == 'running'))
        await settle()
        assert running_by_run() == {'rl28k-s1': 2, 'codex-a': 1}
        status = (await client.get('/status')).json()
        assert status['priorities'] == {'rl28k-': 0} and status['reserve'] == 0.25
        await broker.close()
    asyncio.run(run())


def test_without_priorities_the_gateway_is_unchanged():
    broker = gg.Broker(httpx.AsyncClient())
    assert broker.priority('rl28k-x') == broker.priority('codex-a') == gg.DEFAULT_PRIORITY
    assert not broker.reserved('codex-a') and broker.status()['reserve'] is None
    asyncio.run(broker.http.aclose())


def test_parse_priorities_longest_prefix_wins_and_bad_specs_fail():
    assert gg.parse_priorities(['rl28k-=0', 'codex-=2']) == [('rl28k-', 0), ('codex-', 2)]
    broker = gg.Broker(httpx.AsyncClient(), priorities=gg.parse_priorities(['rl=0', 'rl28k-bench-=3']))
    assert broker.priority('rl28k-s1') == 0 and broker.priority('rl28k-bench-a') == 3 and broker.priority('x') == 1
    asyncio.run(broker.http.aclose())
    for bad in (['rl28k-'], ['=0'], ['rl28k-=x']):
        with pytest.raises(ValueError):
            gg.parse_priorities(bad)


def test_grader_hash_routes_only_to_matching_farms():
    farms = dict(old=FakeFarm(dict(math=4), grader='b' * 64), new=FakeFarm(dict(math=1)))

    async def run():
        broker, client, _ = broker_for(farms)
        await add_farms(broker, farms)
        for _ in range(2):
            await client.post('/skyrl/v1/grading/submit', json=body(), headers=heads())
        await settle()
        assert len(farms['new'].submits) == 1 and farms['old'].submits == []
        assert (await client.get('/skyrl/v1/grading/capacity', headers=heads())).json()['families']['math']['total'] == 1
        # A run whose grader no farm runs: refused (not banned), and its capacity hides the gateway.
        refused = await client.post('/skyrl/v1/grading/submit', json=body(grader='c' * 64), headers=heads('C', 'tc'))
        assert refused.status_code == 429 and refused.json()['retry_after'] == gg.NO_FARM_RETRY_S
        capacity = (await client.get('/skyrl/v1/grading/capacity', headers=heads('C', 'tc'))).json()
        assert capacity['ready'] is False and capacity['families'] == {}
        # No identity: like a farm without a lease.
        assert (await client.get('/skyrl/v1/grading/capacity')).status_code == 409
        await broker.close()
    asyncio.run(run())


def test_lost_grades_move_back_to_the_queue_and_to_another_farm():
    farms = dict(f1=FakeFarm(dict(math=1)), f2=FakeFarm(dict(math=1)))

    def where(grade):
        """(farm name, farm-side state) of the grade's current farm id."""
        return next(((n, f.entries[grade.farm_id]['state']) for n, f in farms.items() if grade.farm_id in f.entries
                     and f.entries[grade.farm_id]['state'] == 'running'), None)

    async def probe(name):
        await broker._probe(broker.farms['http://' + name])
        await settle()

    async def run():
        nonlocal broker
        broker, client, _ = broker_for(farms)
        await add_farms(broker, farms)
        requests = [body() for _ in range(2)]
        for request in requests:
            await client.post('/skyrl/v1/grading/submit', json=request, headers=heads())
        await settle()
        grade, other = (broker.grades[r['request_id']] for r in requests)
        assert where(grade) == ('f1', 'running') and where(other) == ('f2', 'running')
        # f1 forgets the grade (its ingress restarted): after two unknown polls it is regraded under a new farm id.
        del farms['f1'].entries[grade.wire_id]
        await settle()
        assert grade.losses == 1 and grade.state == 'dispatched' and where(grade) == ('f1', 'running')
        assert grade.farm_id != grade.wire_id
        # A draining farm hands its grade back: it moves without reaching the trainer, and waits here for a slot.
        farms['f1'].draining = True
        farms['f1'].entries[grade.farm_id].update(state='cancelled', error={'class': 'draining', 'detail': 'x'})
        await probe('f1')
        assert grade.losses == 2 and grade.state == 'queued'
        farms['f2'].finish(other.farm_id)
        await settle()
        assert other.state == 'done' and where(grade) == ('f2', 'running')
        # A farm that comes back with a new instance loses what it held (no cancel: it no longer knows the id).
        farms['f2'].instance = 'i2'
        cancels = list(farms['f2'].cancels)
        await probe('f2')
        assert grade.losses == 3 and farms['f2'].cancels == cancels and where(grade) == ('f2', 'running')
        # Past MAX_LOSSES the trainer gets an infrastructure failure (it retries elsewhere, charged).
        farms['f2'].entries[grade.farm_id]['state'] = 'cancelled'
        await settle()
        view = (await client.post('/skyrl/v1/grading/results', json=dict(ids=[grade.wire_id]),
                                  headers=heads())).json()['views'][grade.wire_id]
        assert view['state'] == 'failed' and view['error']['class'] == 'infrastructure' and grade.losses == 4
        await broker.close()
    broker = None
    asyncio.run(run())


def test_unreachable_farm_keeps_its_grades_for_a_while_then_loses_them():
    farms = dict(f1=FakeFarm(dict(math=1)), f2=FakeFarm(dict(math=1)))
    clock = [1000.0]

    async def run():
        broker, client, routes = broker_for(farms, now=lambda: clock[0])
        await add_farms(broker, farms)
        await client.post('/skyrl/v1/grading/submit', json=body(), headers=heads())
        await settle()
        holder = 'f1' if farms['f1'].submits else 'f2'

        async def down(request):
            raise httpx.ConnectError('down', request=request)
        routes[holder] = httpx.MockTransport(down)
        await settle()
        grade, = broker.grades.values()
        assert grade.state == 'dispatched' and grade.losses == 0  # a blip does not move a running grade
        clock[0] += gg.FARM_LOST_S + 1
        await settle()
        assert grade.losses == 1 and grade.farm is broker.farms['http://' + ('f2' if holder == 'f1' else 'f1')]
        await broker.close()
    asyncio.run(run())


def test_trainer_cancel_and_ttl_expiry_cancel_on_the_farm():
    farms = dict(f1=FakeFarm(dict(math=2)))
    clock = [1000.0]

    async def run():
        broker, client, _ = broker_for(farms, now=lambda: clock[0])
        await add_farms(broker, farms)
        first, second, third, fourth = body(), body(), body(), body()
        for request in (first, second, third):
            assert (await client.post('/skyrl/v1/grading/submit', json=request, headers=heads())).status_code == 202
        await settle()
        assert broker.grades[third['request_id']].state == 'queued'
        assert (await client.post(f'/skyrl/v1/grading/cancel/{first["request_id"]}', headers=heads('run-a', 'x'))
                ).status_code == 404  # only its owner cancels it
        view = (await client.post(f'/skyrl/v1/grading/cancel/{first["request_id"]}', headers=heads())).json()
        assert view['state'] == 'cancelled'
        await settle()
        assert farms['f1'].cancels == [first['request_id']]
        assert farms['f1'].entries[third['request_id']]['state'] == 'running'  # the freed slot was reused
        # A grade still queued here is cancelled without touching a farm.
        assert (await client.post('/skyrl/v1/grading/submit', json=fourth, headers=heads())).status_code == 202
        await client.post(f'/skyrl/v1/grading/cancel/{fourth["request_id"]}', headers=heads())
        assert broker.grades[fourth['request_id']].state == 'cancelled' and fourth['request_id'] not in farms['f1'].entries
        # The trainer stops polling: its grades expire and are cancelled on the farm.
        clock[0] += gg.TTL_S + 1
        broker.reap()
        await settle()
        assert {second['request_id'], third['request_id']} <= set(farms['f1'].cancels)
        clock[0] += gg.RETENTION_S + 1
        broker.reap()
        assert broker.grades == {}
        await broker.close()
    asyncio.run(run())


# -------------------------------------------------------------------- the real trainer client through the gateway
def registry(tmp_path):
    path = tmp_path / 'della-graders.json'
    path.write_text(json.dumps(dict(farms=[dict(name=gg.ENTRY_NAME, url='http://gateway', ok=True, tunnel_ok=True,
                                                health='healthy', instance='x')])))
    return path


def test_trainer_grades_through_the_gateway_on_a_real_farm(tmp_path, monkeypatch):
    """The unchanged grading_transport (open grading) -> gateway -> the real farm grading service."""
    from tests.tpu_swarm.test_farm_grading_service import grading_farm
    monkeypatch.setenv('SKYRL_RUN_ID', 'rl28k-run')

    async def run():
        async with grading_farm(tmp_path, monkeypatch, slots_per_host=1) as (farm, _g, service, futures, _c, _e):
            routes = {'farm': farm._transport}
            broker = gg.Broker(httpx.AsyncClient(transport=Router(routes)), retry_seconds=0.01, long_poll_seconds=1)
            routes['gateway'] = httpx.ASGITransport(app=gg.create_app(broker))
            broker.set_farms([dict(url='http://farm', name='della-003@4123')])
            await settle()
            http = httpx.AsyncClient(transport=Router(routes))
            transport = gt.GradingTransport(families={}, local_capacity=0, http=http, loop=asyncio.get_running_loop(),
                                            open_registry=str(registry(tmp_path)), open_k=4, poll_seconds=0.01,
                                            long_poll_seconds=1)
            spec = dict(program_code='def construct_function():\n    return [1.0]\n',
                        function_name='construct_function', eval_timeout_seconds=600)
            grades = [asyncio.ensure_future(transport.grade(gt.GradingRequest(task='math', spec=dict(spec))))
                      for _ in range(6)]
            for _ in range(200):
                await asyncio.sleep(0.01)
                if len(futures) == 4:
                    break
            await settle()
            assert len(futures) == 4  # 4 slots on the farm: never more dispatched
            assert {e.owner_run for e in service.entries.values()} == {'rl28k-run'}
            for i in range(6):
                for _ in range(200):
                    if len(futures) > i:
                        break
                    await asyncio.sleep(0.01)
                futures[i].set_result(dict(result=[1.0], error=None, stdout='', metrics={'host': 'della-r1c2'}))
            results = await asyncio.wait_for(asyncio.gather(*grades), 30)
            assert len(futures) == 6
            for result in results:
                assert result['metrics']['grading_pool'] == gg.ENTRY_NAME and result['metrics']['grading_open'] is True
                assert result['metrics']['grader_sha256'] == service.grader_sha256
            await transport.close()
            await broker.close()
    asyncio.run(run())


def test_gateway_restart_loses_nothing_silently(tmp_path, monkeypatch):
    """A restarted gateway does not know in-flight grades: the trainer declares them lost and regrades them."""
    monkeypatch.setenv('SKYRL_RUN_ID', 'run-x')
    grader = gt.local_grader_sha256()
    farms = dict(f1=FakeFarm(dict(math=2), grader=grader), f2=FakeFarm(dict(math=2), grader=grader))

    async def run():
        broker, _client, routes = broker_for(farms)
        await add_farms(broker, dict(f1=farms['f1']))  # f1 behind the gateway; f2 is another open farm
        path = registry(tmp_path)
        document = json.loads(path.read_text())
        document['farms'].append(dict(name='f2', url='http://f2', ok=True, tunnel_ok=True))
        path.write_text(json.dumps(document))
        events = tmp_path / 'events.jsonl'
        transport = gt.GradingTransport(families={}, local_capacity=0, http=httpx.AsyncClient(transport=Router(routes)),
                                        loop=asyncio.get_running_loop(), open_registry=str(path), open_k=4,
                                        poll_seconds=0.01, long_poll_seconds=1, events=str(events), retry_jitter_s=0)
        grade = asyncio.ensure_future(transport.grade(gt.GradingRequest(task='math', spec={'x': 1})))
        for _ in range(300):
            await asyncio.sleep(0.01)
            if farms['f1'].submits:
                break
        assert farms['f1'].submits and not farms['f2'].submits  # went through the gateway
        # Restart: a new broker (empty state) behind the same URL.
        await broker.close()
        broker, _c, fresh = broker_for(farms)
        routes['gateway'] = fresh['gateway']
        await add_farms(broker, dict(f1=farms['f1']))
        for _ in range(500):
            await asyncio.sleep(0.01)
            if farms['f2'].submits:
                break
        farms['f2'].finish(farms['f2'].submits[0], reward=0.25)
        result = await asyncio.wait_for(grade, 30)
        assert result['reward'] == 0.25 and result['metrics']['grading_pool'] == 'f2'
        records = [json.loads(line) for line in events.read_text().splitlines()]
        assert any(r.get('event') == 'grading_reroute' and r.get('kind') == 'poll_miss' and r.get('charged')
                   for r in records)
        await transport.close()
        await broker.close()
    asyncio.run(run())


def test_gateway_server_serves_on_its_socket_and_a_taken_port_fails_fast():
    server = gg.GatewayServer(0)
    try:
        status = httpx.get('http://127.0.0.1:%d/status' % server.port, timeout=10).json()
        assert status['instance'] == server.instance and status['gateway'] is True
        server.set_farms([])
        assert server.stats()['usable_farms'] == 0 and server.alive()
        with pytest.raises(OSError):
            gg.GatewayServer(server.port)
    finally:
        server.close()
    assert not server.alive()


# ------------------------------------------------------------------------- discovery: tunnels and registries
TRAINER = dict(job_id=3301, status='RUNNING', cluster='tpuswarm-v4-32-central2-rl28k-41',
               run_id='rl28k-qwen35-circuit-s1', pool='tpuswarm-v4-32-central2-rl28k', priority=0)
DELLA = dict(job_id=4123, status='RUNNING', cluster='tpuswarm-della-grader-615',
             run_id='inference-farm-rq4-grading-della-003', pool='tpuswarm-della-grader', priority=0)


def della_probe(instance='d1', ip='10.33.1.7', port=24123):
    return dict(ok=True, result=dict(models=[], url='http://%s:%d' % (ip, port), owner_run=None, state='unleased',
                                     active=0, capabilities=None, instance=instance, engines=0, direct=True,
                                     grading=dict(ready=True, families=['math', 'placement', 'routing'])))


def ssh_config(tmp_path, cluster, address):
    (tmp_path / cluster).write_text('Host %s\n  HostName %s\n  User gcpuser\n\nHost %s-worker1\n  HostName 9.9.9.9\n'
                                    % (cluster, address, cluster))


class FakeProcess:
    def __init__(self, argv):
        self.argv, self.code = argv, None

    def poll(self):
        return self.code

    def terminate(self):
        self.code = -15

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


class FakeServer:
    port, instance = 28650, 'gw1'

    def __init__(self):
        self.farms = None

    def alive(self):
        return True

    def set_farms(self, farms):
        self.farms = farms

    def stats(self):
        return dict(grades={}, counters={}, runs={}, usable_farms=len(self.farms or []))

    def close(self):
        pass


def service(tmp_path, monkeypatch, verified=lambda names: names, push=True, private=None, probe=None):
    started, pushed = [], []
    monkeypatch.setattr(discovery, 'slurm_nodes', lambda: {'tpuswarm-della-grader-615-7bfcb694': 'della-r1c2'})
    monkeypatch.setattr(discovery, 'local_rpc', lambda url: probe() if probe else della_probe())
    monkeypatch.setattr(discovery, 'rpc', lambda *a, **k: (_ for _ in ()).throw(AssertionError('no cloud rpc')))
    server = FakeServer()
    monkeypatch.setattr(gg, 'GatewayServer', lambda *a, **k: server)
    args = base_args(tmp_path, name_contains=['rq4-grading-della'], slurm_pool=['tpuswarm-della-grader'],
                     trainer_pool=['tpuswarm-v4-32-central2-rl28k'], grader_gateway=True, gateway_only=True,
                     gateway_registry_path=gg.DEFAULT_REGISTRY, gateway_port=28650, gateway_remote_port=26000,
                     push=push)
    svc = discovery.Service(args, private or dict(farms=[], trainers=[]), now=lambda: 0)

    def popen(argv, **kw):
        started.append(FakeProcess(argv))
        return started[-1]

    def run(argv, input, **kw):
        params = eval(input.split('\n', 1)[0].split('=', 1)[1])
        pushed.append(dict(argv=argv, params=params))
        return SimpleNamespace(returncode=0, stdout=json.dumps(dict(ok=True, verified=verified(
            [f['name'] for f in params['farms']]))))
    svc.gateway.popen, svc.gateway.run = popen, run
    return svc, started, pushed, server


def test_one_tunnel_and_registry_per_trainer_host(tmp_path, monkeypatch):
    ssh_config(tmp_path, TRAINER['cluster'], '35.1.2.3')
    svc, started, pushed, server = service(tmp_path, monkeypatch)
    doc = svc.tick([TRAINER, DELLA])
    # The della farm is ready and handed to the gateway by its direct URL; trainers' lists are never touched.
    assert server.farms == [dict(url='http://10.33.1.7:24123', name='inference-farm-rq4-grading-della-003@4123')]
    assert doc['trainers'] == dict(pool=[], direct=[])
    tunnel, = started
    assert tunnel.argv[:3] == ['ssh', '-F', str(tmp_path / TRAINER['cluster'])]
    assert tunnel.argv[-3:] == ['-R', '127.0.0.1:26000:127.0.0.1:28650', TRAINER['cluster']]
    assert 'ExitOnForwardFailure=yes' in tunnel.argv and '-N' in tunnel.argv
    push, = pushed
    assert push['argv'][-2:] == [TRAINER['cluster'], 'python3 -']
    assert push['params']['path'] == gg.DEFAULT_REGISTRY
    assert push['params']['farms'] == [dict(name='della-gateway', url='http://127.0.0.1:26000', instance='gw1',
                                            engines=0, grading=True, source='della-gateway')]
    host, = doc['gateway']['hosts']
    assert host['host'] == '35.1.2.3' and host['verified'] is True and host['error'] is None
    # Steady state: no new ssh while the forward lives; more della farms never add tunnels.
    other = dict(DELLA, job_id=4124, cluster='tpuswarm-della-grader-616', run_id='inference-farm-rq4-grading-della-004')
    monkeypatch.setattr(discovery, 'slurm_nodes', lambda: {'tpuswarm-della-grader-615': 'n1',
                                                          'tpuswarm-della-grader-616': 'n2'})
    svc.tick([TRAINER, DELLA, other])
    assert len(started) == 1 and len(pushed) == 2 and len(server.farms) == 2
    # The trainer job ends: its tunnel closes.
    svc.tick([DELLA])
    assert tunnel.code == -15


def test_shadow_gateway_opens_nothing(tmp_path, monkeypatch):
    ssh_config(tmp_path, TRAINER['cluster'], '35.1.2.3')
    svc, started, pushed, server = service(tmp_path, monkeypatch, push=False)
    doc = svc.tick([TRAINER, DELLA])
    assert doc['gateway'] == dict(shadow=True, hosts=['35.1.2.3'], farms=['inference-farm-rq4-grading-della-003@4123'])
    assert started == [] and pushed == [] and server.farms is None


def test_unready_or_serving_farms_are_not_gateway_targets(tmp_path, monkeypatch):
    ssh_config(tmp_path, TRAINER['cluster'], '35.1.2.3')
    draining = dict(ok=False, error='ValueError')  # /health fails while the farm drains before its time limit
    svc, started, pushed, server = service(tmp_path, monkeypatch, probe=lambda: draining)
    svc.tick([TRAINER, DELLA])
    assert server.farms == [] and len(started) == 1  # the tunnel still serves (an empty gateway is not ready)
    tpu = dict(job_id='9', name='f', direct=False, engines=4, grading=dict(ready=True),
               internal_url='http://10.0.0.5:24800', incarnation='x')
    assert gg.della_farms([tpu]) == []


def test_refused_forward_moves_to_the_fallback_port(tmp_path, monkeypatch):
    ssh_config(tmp_path, TRAINER['cluster'], '35.1.2.3')
    svc, started, pushed, _ = service(tmp_path, monkeypatch, verified=lambda names: [])
    for _ in range(3):
        svc.tick([TRAINER, DELLA])
        started[-1].code = 255  # ExitOnForwardFailure: the port is taken on the trainer host
    ports = [p.argv[-2].split(':')[1] for p in started]
    assert ports[:2] == ['26000', '26000']
    svc.tick([TRAINER, DELLA])
    assert started[-1].argv[-2] == '127.0.0.1:27000:127.0.0.1:28650'
    assert pushed[-1]['params']['farms'][0]['url'] == 'http://127.0.0.1:27000'


def test_live_forward_that_stops_answering_is_restarted(tmp_path, monkeypatch):
    ssh_config(tmp_path, TRAINER['cluster'], '35.1.2.3')
    answers = [['della-gateway'], [], [], [], []]
    svc, started, pushed, _ = service(tmp_path, monkeypatch, verified=lambda names: answers.pop(0))
    for _ in range(4):
        svc.tick([TRAINER, DELLA])
    assert started[0].code == -15  # closed after 3 missed checks
    svc.tick([TRAINER, DELLA])
    assert len(started) == 2 and started[1].argv[-2].startswith('127.0.0.1:26000:')  # was verified: same port


def test_failed_tick_keeps_the_tunnels_and_registries_of_the_last_good_tick(tmp_path, monkeypatch):
    ssh_config(tmp_path, TRAINER['cluster'], '35.1.2.3')
    svc, started, pushed, server = service(tmp_path, monkeypatch)
    assert svc.keep_gateway() is None  # nothing known yet: no tunnel from an empty view
    svc.tick([TRAINER, DELLA])
    # The SkyPilot API fails (inventory raises before tick): the dead forward is restarted and the registry pushed.
    started[0].code = 255
    kept = svc.keep_gateway()
    assert len(started) == 2 and len(pushed) == 2
    assert kept['hosts'][0]['host'] == '35.1.2.3' and kept['farms'] == ['inference-farm-rq4-grading-della-003@4123']
    assert server.farms == [dict(url='http://10.33.1.7:24123', name='inference-farm-rq4-grading-della-003@4123')]


def test_pool_and_private_trainers_sharing_a_host_share_one_tunnel(tmp_path, monkeypatch):
    od = dict(TRAINER, job_id=3402, cluster='tpuswarm-v4-64-central2-od-1', pool='tpuswarm-v4-64-central2-od',
              run_id='rl28k-qwen35-erdos-s2-od')
    ssh_config(tmp_path, od['cluster'], '35.186.57.88')
    private = dict(farms=[], trainers=[
        dict(run_id='rl28k-qwen35-erdos-s2-od16', model='Qwen/Qwen3.5-27B', group='rl28k-qwen35',
             external_ip='35.186.57.88', ssh_key='~/.ssh/k', stable_urls=[]),
        dict(run_id='other', model='Qwen/Qwen3.5-27B', group='g', external_ip='34.0.0.9', ssh_key='~/.ssh/k',
             stable_urls=[]),
        dict(run_id='opted-out', model='m', group='g', external_ip='34.0.0.10', ssh_key='~/.ssh/k', stable_urls=[],
             grader_gateway=False)])
    svc, started, pushed, _ = service(tmp_path, monkeypatch, private=private)
    svc.args.trainer_pool = ['tpuswarm-v4-64-central2-od']
    doc = svc.tick([od, DELLA])
    assert sorted(h['host'] for h in doc['gateway']['hosts']) == ['34.0.0.9', '35.186.57.88']
    assert len(started) == 2
    private_tunnel = next(p for p in started if p.argv[-1] == 'gcpuser@34.0.0.9')
    assert '-i' in private_tunnel.argv and private_tunnel.argv[-3:-1] == ['-R', '127.0.0.1:26000:127.0.0.1:28650']


def test_ssh_host_name_reads_the_head_entry(tmp_path):
    ssh_config(tmp_path, 'c', '1.2.3.4')
    assert gg.ssh_host_name(tmp_path / 'c', 'c') == '1.2.3.4'
    assert gg.ssh_host_name(tmp_path / 'missing', 'c') is None


def test_remote_script_lists_the_gateway_only_if_its_tunnel_answers_with_its_instance(tmp_path):
    """Run the trainer-host script against a local stand-in for the tunnel end."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps(dict(instance='gw1')).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = 'http://127.0.0.1:%d' % server.server_address[1]
    path = tmp_path / 'home/.cache/skyrl/della-graders.json'
    try:
        outputs = []
        for instance in ('gw1', 'stale'):
            proc = subprocess.run([sys.executable, '-'], input='PARAMS = %r\n' % dict(path=str(path), farms=[
                dict(name='della-gateway', url=url, instance=instance)]) + gg.REMOTE,
                                  text=True, capture_output=True, timeout=60)
            assert proc.returncode == 0, proc.stderr
            outputs.append((json.loads(proc.stdout), json.loads(path.read_text())))
    finally:
        server.shutdown()
    (good, listed), (bad, unlisted) = outputs
    assert good == dict(ok=True, verified=['della-gateway']) and [f['url'] for f in listed['farms']] == [url]
    assert bad == dict(ok=True, verified=[]) and unlisted['farms'] == [] and unlisted['unready'][0]['name'] == 'della-gateway'
    assert not path.with_name(path.name + '.partial').exists()


def test_trainer_open_grading_reads_the_pushed_registry(tmp_path):
    """The pushed document is what grading_transport.OpenFarmDirectory lists (unchanged client code)."""
    import time
    path = tmp_path / 'della-graders.json'
    entry = dict(name='della-gateway', url='http://127.0.0.1:26000', instance='gw1', engines=0, grading=True,
                 source='della-gateway', ok=True, tunnel_ok=True, health='healthy')
    path.write_text(json.dumps(dict(updated=time.time(), contract='ready-only-v1', farms=[entry], unready=[])))
    seen = []

    def handler(request):
        seen.append((request.url.host, request.url.port, request.url.path, request.headers.get('x-grading-run')))
        return httpx.Response(200, json=dict(open=True, ready=True, grader_sha256=None,
                                             families=dict(placement=dict(total=120, running=0, queued=0))))

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            directory = gt.OpenFarmDirectory(str(path), http, k=4)
            await directory.refresh()
            return directory
    directory = asyncio.run(go())
    pool, = directory.pools.values()
    assert pool.eligible and pool.url == 'http://127.0.0.1:26000' and pool.capacities == dict(placement=120)
    assert seen[0][:3] == ('127.0.0.1', 26000, '/skyrl/v1/grading/capacity')


def test_trainer_client_env_from_profile_enables_open_grading_on_the_gateway():
    from tpu.swarm.ray_train.commands import client_environment
    from tpu.swarm.ray_train.config import Config
    raw = json.loads((REPO / 'tpu/swarm/ray_train/profiles/rl28k-qwen35-circuit-s1.json').read_text())
    raw['client_env'].update(SKYRL_OPEN_GRADING='1', SKYRL_RUN_ID=raw['run_id'],
                             SKYRL_GRADING_REGISTRY=gg.DEFAULT_REGISTRY)
    env = client_environment(Config.from_dict(raw), Path('/home/gcpuser/.cache/rl28k-qwen35-circuit-s1'), '10.0.0.2')
    assert env['SKYRL_OPEN_GRADING'] == '1' and env['SKYRL_GRADING_REGISTRY'] == gg.DEFAULT_REGISTRY
    assert env['SKYRL_RUN_ID'] == 'rl28k-qwen35-circuit-s1'
    assert env['SKYRL_GRADING_URL'] == 'http://10.0.0.2:24800'  # leased TPU farms keep grading too


def test_trainer_host_gateway_ports_avoid_the_trainer_and_partition_ports():
    from tpu.swarm.ray_train.config import Config
    used = set()
    for name in ('rl28k-qwen35-circuit-s1.json', 'rl28k-qwen35-erdos-s2-od16.json'):
        raw = json.loads((REPO / 'tpu/swarm/ray_train/profiles' / name).read_text())
        ports = Config.from_dict(raw).ports
        used |= set(range(ports.worker_min, ports.worker_max + 1))
        used |= {v for k, v in vars(ports).items() if k not in ('worker_min', 'worker_max')}
        used.add(raw.get('systemd_port', 24900))
    gateway = {gg.DEFAULT_REMOTE_PORT, gg.DEFAULT_REMOTE_PORT + 1000}
    assert not gateway & used and not gateway & set(range(10002, 20000)) and max(gateway) < 32768
    # On della-vis1: clear of discovery's local tunnels (29000 + id % 1000) and local_farms' (27000 + id % 1000).
    assert not 27000 <= gg.DEFAULT_PORT < 28000 and not 29000 <= gg.DEFAULT_PORT < 30000
