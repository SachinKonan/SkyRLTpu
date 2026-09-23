"""Client-side grading transport: local + farm pools, retries, first-result-wins."""
import asyncio
import json
import threading
import time

import httpx
import pytest

from tpu.science import grading_transport as gt

FAMILIES = {'ac2': {'slots_per_host': 2, 'cpus': 2, 'memory_gib': 4}}


class GradingFarm:
    """Fake farm grading endpoints keyed by host; each host has its own registry."""

    def __init__(self, capacity=4):
        self.capacity = capacity
        self.entries = {}
        self.down = set()
        self.submits = []
        self.busy = set()
        self.cancels = []
        self.outcomes = {}  # host -> callable(request_id) -> view dict

    async def __call__(self, request):
        await asyncio.sleep(0)  # A real HTTP round trip yields to the loop.
        host, path = request.url.host, request.url.path
        if host in self.down:
            raise httpx.ConnectError('offline', request=request)
        assert request.headers['X-Lease-ID'] == f'token-{host}'
        if path == '/skyrl/v1/grading/capacity':
            return httpx.Response(200, json={'ready': True, 'hosts': 1, 'families': {
                'ac2': dict(FAMILIES['ac2'], total=self.capacity, running=0, queued=0)}})
        if path == '/skyrl/v1/grading/submit':
            body = json.loads(request.content)
            if host in self.busy:
                return httpx.Response(429, json={'detail': 'busy', 'retry_after': 0.01})
            self.submits.append((host, body['request_id']))
            self.entries[(host, body['request_id'])] = body
            return httpx.Response(202, json={'request_id': body['request_id'], 'state': 'queued'})
        if path.startswith('/skyrl/v1/grading/result/'):
            request_id = path.rsplit('/', 1)[1]
            if (host, request_id) not in self.entries:
                return httpx.Response(404, json={'detail': 'unknown'})
            outcome = self.outcomes.get(host)
            view = outcome(request_id) if outcome else {'request_id': request_id, 'state': 'running'}
            return httpx.Response(200, json=view)
        if path.startswith('/skyrl/v1/grading/cancel/'):
            self.cancels.append((host, path.rsplit('/', 1)[1]))
            return httpx.Response(200, json={'state': 'cancelled'})
        raise AssertionError(path)


def done(result):
    return lambda request_id: {'request_id': request_id, 'state': 'done', 'result': result, 'error': None}


def failed(detail):
    return lambda request_id: {'request_id': request_id, 'state': 'failed', 'result': None,
                               'error': {'class': 'infrastructure', 'detail': detail}}


class FakeLocal(gt.LocalRayPool):
    def __init__(self, capacity, outcome):
        super().__init__(FAMILIES, capacity)
        self.outcome = outcome
        self.calls = []

    def dispatch(self, request):
        self.calls.append(request.request_id)
        future = asyncio.get_running_loop().create_future()
        outcome = self.outcome
        asyncio.get_running_loop().call_soon(lambda: future.done() or (
            future.set_exception(outcome) if isinstance(outcome, Exception) else future.set_result(outcome)))
        return future


def make_transport(farm, farms, *, local=None, loop=None, refresh_seconds=.05, **kwargs):
    """``farms`` is a live list: mutate it to change what the fake ingress lists."""
    async def route(request):
        if request.url.host == 'head':
            assert request.url.path == '/skyrl/v1/grading/farms'
            return httpx.Response(200, json={'instance': 'x', 'farms': [
                dict(farm_id=host, url=f'http://{host}', token=f'token-{host}', eligible=True, capacity=4)
                for host in farms]})
        return await farm(request)
    http = httpx.AsyncClient(transport=httpx.MockTransport(route))
    t = gt.GradingTransport(families=FAMILIES, local_capacity=0, farm_url='http://head', http=http,
                            loop=loop, long_poll_seconds=0, poll_seconds=.01, refresh_seconds=refresh_seconds, **kwargs)
    t.local = local
    t.reports = []
    t.report = lambda event, **fields: t.reports.append((event, fields))
    return t


def transport(farm, farms, **kwargs):
    return make_transport(farm, farms, loop=asyncio.get_running_loop(), **kwargs)


def request(**spec):
    return gt.GradingRequest(task='ac2', spec=dict(program_code='x', function_name='construct_function',
                                                   eval_timeout_seconds=1105, **spec), admission_timeout_s=1100)


def test_local_and_farm_pools_share_candidates_by_predicted_admission():
    async def run():
        farm = GradingFarm(capacity=4)
        farm.outcomes['a'] = done({'result': [1.0], 'error': None, 'metrics': {'admission_wait_seconds': 0}})
        farm.outcomes['b'] = done({'result': [2.0], 'error': None, 'metrics': {}})
        local = FakeLocal(2, {'result': [0.0], 'error': None, 'metrics': {}})
        t = transport(farm, ['a', 'b'], local=local)
        try:
            await t._ensure_started()
            assert t.directory.pools['a'].capacity_for('ac2') == 4
            results = await asyncio.gather(*(t.grade(request()) for _ in range(12)))
            assert len(results) == 12 and all('result' in r for r in results)
            pools = {f['pool'] for e, f in t.reports if e == 'grading_dispatched'}
            assert pools == {'local', 'a', 'b'}
            assert len(t.results) == 12 and local.running == 0
            assert all(p.running == 0 and p.queued == 0 for p in t.pools())
            assert any(e == 'grading_farms_changed' and sorted(f['added']) == ['a', 'b'] for e, f in t.reports)
        finally:
            await t.close()
    asyncio.run(run())


def test_farm_loss_mid_grade_retries_elsewhere_with_same_request_id():
    async def run():
        farm = GradingFarm()
        farm.outcomes['b'] = done({'result': [2.0], 'error': None, 'metrics': {}})
        t = transport(farm, ['a', 'b'])
        try:
            await t._ensure_started()
            t.directory.pools['b'].unhealthy_until = time.monotonic() + .3  # 'a' first, then 'b'.
            req = request()
            task = asyncio.create_task(t.grade(req))
            for _ in range(50):
                if ('a', req.request_id) in farm.entries:
                    break
                await asyncio.sleep(.01)
            farm.down.add('a')
            result = await asyncio.wait_for(task, 10)
            assert result['result'] == [2.0]
            assert [h for h, _ in farm.submits] == ['a', 'b']
            assert farm.submits[0][1] == farm.submits[1][1] == req.request_id
            retries = [f for e, f in t.reports if e == 'grading_retry']
            assert retries and retries[0]['from_pool'] == 'a' and retries[0]['attempt'] == 1
            assert not any(e == 'grading_infra_failure' for e, _ in t.reports)
            assert not t.directory.pools['a'].healthy(time.monotonic())
        finally:
            await t.close()
    asyncio.run(run())


def test_first_result_wins_for_duplicate_ids_and_cancel_reaches_the_farm():
    async def run():
        farm = GradingFarm()
        farm.outcomes['a'] = done({'result': [1.0], 'error': None, 'metrics': {}})
        t = transport(farm, ['a'])
        try:
            req = request()
            first = await t.grade(req)
            again = await t.grade(gt.GradingRequest(task='ac2', spec=req.spec, request_id=req.request_id))
            assert first is again and len(farm.submits) == 1
            assert any(e == 'grading_duplicate_request' for e, _ in t.reports)
            # A cancelled in-flight grade is cancelled on the farm too.
            farm.outcomes['a'] = None
            slow = asyncio.create_task(t.grade(request()))
            for _ in range(50):
                if len(farm.submits) == 2:
                    break
                await asyncio.sleep(.01)
            slow.cancel()
            await asyncio.gather(slow, return_exceptions=True)
            assert farm.cancels == [('a', farm.submits[1][1])]
            assert t.directory.pools['a'].running == 0 and t.directory.pools['a'].queued == 0
        finally:
            await t.close()
    asyncio.run(run())


def test_candidate_failure_is_not_retried_and_infra_failure_escalates():
    async def run():
        farm = GradingFarm()
        farm.outcomes['a'] = done({'result': None, 'error': 'Process timed out after 1105 seconds', 'metrics': {}})
        t = transport(farm, ['a'], max_infra_retries=1)
        try:
            result = await t.grade(request())
            assert 'timed out' in result['error'] and not any(e == 'grading_retry' for e, _ in t.reports)
            farm.outcomes['a'] = failed('WorkerCrashedError: node lost')
            with pytest.raises(gt.GradingInfrastructureError) as info:
                await t.grade(gt.GradingRequest(task='ac2', spec=request().spec, deadline_s=2))
            assert getattr(info.value, 'abort_training_step') is True
            failures = [f for e, f in t.reports if e == 'grading_infra_failure']
            assert failures and failures[0]['attempts'] >= 1
            assert [f['attempt'] for e, f in t.reports if e == 'grading_retry'] == [1]
        finally:
            await t.close()
    asyncio.run(run())


def test_429_reschedules_without_counting_a_retry():
    async def run():
        farm = GradingFarm()
        farm.busy.add('a')
        farm.outcomes['a'] = done({'result': [1.0], 'error': None, 'metrics': {}})
        t = transport(farm, ['a'])
        try:
            task = asyncio.create_task(t.grade(request()))
            await asyncio.sleep(.05)
            assert not task.done() and t.directory.pools['a'].queued == 1
            farm.busy.discard('a')
            result = await asyncio.wait_for(task, 5)
            assert result['result'] == [1.0] and not any(e == 'grading_retry' for e, _ in t.reports)
        finally:
            await t.close()
    asyncio.run(run())


def test_no_pool_waits_without_deadline_then_uses_a_late_farm():
    async def run():
        farm = GradingFarm()
        farm.outcomes['late'] = done({'result': [3.0], 'error': None, 'metrics': {}})
        farms = []
        t = transport(farm, farms)
        try:
            task = asyncio.create_task(t.grade(request()))
            await asyncio.sleep(.3)
            assert not task.done() and any(e == 'grading_no_pool' for e, _ in t.reports)
            farms.append('late')
            result = await asyncio.wait_for(task, 10)
            assert result['result'] == [3.0]
        finally:
            await t.close()
    asyncio.run(run())


def test_lease_loss_retires_pool_and_refresh_replaces_it():
    async def run():
        farm = GradingFarm()
        farms = ['a']
        t = transport(farm, farms, refresh_seconds=.02)
        try:
            await t._ensure_started()
            assert set(t.directory.pools) == {'a'}
            farms.clear()
            for _ in range(50):
                await asyncio.sleep(.02)
                if 'a' not in t.directory.pools:
                    break
            assert 'a' not in t.directory.pools
            farms.append('a')
            for _ in range(50):
                await asyncio.sleep(.02)
                if 'a' in t.directory.pools:
                    break
            assert t.directory.pools['a'].healthy(time.monotonic())
            assert any(e == 'grading_farms_changed' and f['removed'] == ['a'] for e, f in t.reports)
        finally:
            await t.close()
    asyncio.run(run())


def test_grade_sync_from_worker_threads_uses_the_background_loop():
    farm = GradingFarm()
    farm.outcomes['a'] = done({'result': [4.0], 'error': None, 'metrics': {}})
    t = make_transport(farm, ['a'])
    results = []

    def worker():
        results.append(t.grade_sync(request(), timeout=10))
    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
    assert [r['result'] for r in results] == [[4.0]] * 3
    asyncio.run_coroutine_threadsafe(t.close(), t.loop).result(5)
    shutdown(t)


def shutdown(t):
    t.loop.call_soon_threadsafe(t.loop.stop)
    t.thread.join(5)
    t.loop.close()


def test_from_environment_reads_launcher_settings(monkeypatch):
    monkeypatch.setenv('SKYRL_GRADING_FAMILIES', json.dumps(FAMILIES))
    monkeypatch.setenv('SKYRL_GRADING_LOCAL_SLOTS', '8')
    monkeypatch.setenv('SKYRL_GRADING_URL', 'http://head:24800')
    monkeypatch.setenv('SKYRL_GRADING_MAX_INFRA_RETRIES', '2')
    monkeypatch.setenv('SKYRL_GRADING_LOCAL_SYSTEMD', '0')
    monkeypatch.setenv('SCIENCE_WORKER_ROOT', '/code')
    t = gt.GradingTransport.from_environment()
    try:
        assert t.local.capacity == 8 and t.local.root == '/code' and t.local.systemd is False
        assert t.farm_url == 'http://head:24800' and t.max_infra_retries == 2
    finally:
        shutdown(t)


def test_is_infrastructure_classification():
    assert gt.is_infrastructure(httpx.ConnectError('x'))
    assert gt.is_infrastructure(RuntimeError('Routing grader infrastructure failed: y'))
    assert gt.is_infrastructure(TimeoutError('admission timed out'))
    assert not gt.is_infrastructure(ValueError('bad candidate'))
