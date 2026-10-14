"""Generation-scoped farm leases and acknowledged, polled generation (redesign section (a))."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from tpu.swarm.ray_train.generation_jobs import GenerationJobs
from tests.tpu_swarm.test_remote_only import MultiFarm, PAYLOAD, eligible, held, multi_case, until


# --- Farm side: acknowledged generation jobs --------------------------------

def test_submit_acknowledges_then_poll_hands_the_result_over_once():
    async def run():
        jobs = GenerationJobs()
        gate = asyncio.Event()

        async def work(mark_running):
            mark_running()
            await gate.wait()
            return {'choices': [1]}
        status, view = jobs.submit('r1', 'lease-a', work)
        assert status == 202 and view['state'] == 'queued'
        await asyncio.sleep(0)
        assert (await jobs.result('r1', 'lease-a'))['state'] == 'running'
        gate.set()
        done = await jobs.result('r1', 'lease-a', wait=2)
        assert done['state'] == 'done' and done['result'] == {'choices': [1]}
        # Handed over once: a late poll (after a reroute) finds nothing.
        assert await jobs.result('r1', 'lease-a') is None
    asyncio.run(run())


def test_repeated_submit_of_an_id_never_runs_the_group_twice():
    async def run():
        jobs, calls = GenerationJobs(), []

        async def work(mark_running):
            calls.append(1)
            await asyncio.sleep(.05)
            return {}
        assert jobs.submit('r1', 'lease-a', work)[0] == 202
        assert jobs.submit('r1', 'lease-a', work)[0] == 202  # lost acknowledgement, resubmitted
        assert jobs.submit('r1', 'lease-b', work)[0] == 409  # another lease cannot claim the id
        await jobs.result('r1', 'lease-a', wait=2)
        assert calls == [1]
    asyncio.run(run())


def test_cancel_and_lease_release_cancel_unfinished_groups():
    async def run():
        jobs = GenerationJobs()

        async def forever(mark_running):
            mark_running()
            await asyncio.Event().wait()
        jobs.submit('r1', 'lease-a', forever)
        jobs.submit('r2', 'lease-a', forever)
        jobs.submit('r3', 'lease-b', forever)
        await asyncio.sleep(0)
        assert jobs.cancel('r1', 'lease-b') is None  # not the owner
        assert jobs.cancel('r1', 'lease-a')['state'] == 'cancelling'
        assert jobs.cancel_lease('lease-a', 'lease released') == 1
        await asyncio.sleep(0)
        assert (await jobs.result('r2', 'lease-a'))['state'] == 'cancelled'
        assert (await jobs.result('r3', 'lease-b'))['state'] == 'running'
        jobs.cancel_lease('lease-b', 'test over')
    asyncio.run(run())


def test_failed_group_reports_the_farm_status():
    async def run():
        jobs = GenerationJobs()

        class Rejected(Exception):
            status_code, detail = 422, 'bad prompt'

        async def bad(mark_running):
            raise Rejected()
        jobs.submit('r1', 'lease-a', bad)
        view = await jobs.result('r1', 'lease-a', wait=2)
        assert view['state'] == 'failed' and view['status'] == 422 and 'bad prompt' in view['error']
    asyncio.run(run())


# --- Farm side: fair lease queue ----------------------------------------------

def test_lease_queue_serves_the_longest_waiting_run_and_drops_silent_waiters(monkeypatch):
    from tpu.swarm.ray_train import serving
    position = serving.Ingress.func_or_class.__mro__[1].lease_queue_position
    now = [1000.0]
    monkeypatch.setattr(serving.time, 'monotonic', lambda: now[0])
    farm = SimpleNamespace(lease_waiters={}, config=SimpleNamespace(
        inference=SimpleNamespace(external_pool_heartbeat_seconds=30)))
    assert position(farm, 'run-a') == 0
    now[0] += 5
    assert position(farm, 'run-b') == 1  # run-a has waited longer
    assert position(farm, 'run-a') == 0
    now[0] += 200  # run-a stops retrying (it leased elsewhere or exited)
    assert position(farm, 'run-b') == 0


# --- Client side: acknowledged, polled generation -----------------------------

class PollFarm(MultiFarm):
    """MultiFarm that acknowledges groups and is polled for them, per-host behaviour programmable."""

    def __init__(self):
        super().__init__()
        self.mode = {}  # host -> 'ok' | 'dead' | 'queued' | 'busy' | 'fail'
        self.jobs = {}
        self.cancels = []

    def status(self, lease):
        return dict(super().status(lease), generation_protocol=1)

    async def __call__(self, request):
        host, path = request.url.host, request.url.path
        mode = self.mode.get(host, 'ok')
        if path == '/skyrl/v1/generation/submit' and host not in self.down:
            self.requests.append((host, path))
            if mode == 'busy':
                return httpx.Response(429, json={'detail': 'no capacity'})
            body = json.loads(request.content)
            self.jobs[body['request_id']] = (host, body['payload'])
            return httpx.Response(202, json={'request_id': body['request_id'], 'state': 'queued'})
        if path.startswith('/skyrl/v1/generation/result/'):
            self.requests.append((host, path))
            if mode == 'dead' or host in self.down:
                raise httpx.ConnectError('farm gone', request=request)
            request_id = path.rsplit('/', 1)[1]
            if request_id not in self.jobs:
                return httpx.Response(404, json={})
            if mode == 'queued':
                return httpx.Response(200, json={'request_id': request_id, 'state': 'queued'})
            if mode == 'fail':
                return httpx.Response(200, json={'request_id': request_id, 'state': 'failed',
                                                 'error': 'engine died', 'status': 500})
            _, payload = self.jobs.pop(request_id)
            choices = [{'index': i, 'text': 't', 'token_ids': [3, 4], 'logprobs': {'token_logprobs': [-.2, -.3]},
                        'finish_reason': 'stop'} for i in range(payload.get('n', 1))]
            return httpx.Response(200, json={'request_id': request_id, 'state': 'done', 'result': {'choices': choices}})
        if path.startswith('/skyrl/v1/generation/cancel/'):
            self.cancels.append((host, path.rsplit('/', 1)[1]))
            return httpx.Response(200, json={'state': 'cancelling'})
        return await super().__call__(request)


def poll_case(tmp_path, operation, urls=('http://farm1', 'http://farm2'), **inference):
    from tests.tpu_swarm import test_remote_only as base
    original = base.MultiFarm
    base.MultiFarm = PollFarm
    try:
        multi_case(tmp_path, operation, urls=urls, **inference)
    finally:
        base.MultiFarm = original


def test_polled_generation_returns_the_acknowledged_result(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        assert all(m.lease.generation_poll for m in borrower.held())
        result = await borrower.generate(PAYLOAD, farm='http://farm1')
        assert len(result['choices']) == 2
        assert ('farm1', '/skyrl/v1/generation/submit') in farm.requests
        assert not any(p == '/v1/completions' for _, p in farm.requests)
    poll_case(tmp_path, operation)


def test_farm_that_stops_answering_polls_is_dropped_and_the_group_rerouted(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        farm.mode['farm1'] = 'dead'
        assert await borrower.generate(PAYLOAD, farm='http://farm1') is None  # caller requeues the group
        polls = [p for h, p in farm.requests if h == 'farm1' and p.startswith('/skyrl/v1/generation/result/')]
        assert len(polls) == 3  # external_pool_generation_poll_failures consecutive failures
        await until(lambda: 'http://farm1' not in eligible(borrower))
        result = await borrower.generate(PAYLOAD)  # rerouted to the live farm
        assert len(result['choices']) == 2
    poll_case(tmp_path, operation)


def test_group_queued_too_long_is_cancelled_once_and_rerouted_keeping_the_lease(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        farm.mode['farm1'] = 'queued'
        assert await borrower.generate(PAYLOAD, farm='http://farm1') is None
        await until(lambda: len(farm.cancels) == 1)
        assert farm.cancels[0][0] == 'farm1'
        assert 'http://farm1' in held(borrower)  # a slow farm is not a dead farm
        assert any(e == 'borrow_generation_queue_timeout' for e, _ in events)
    poll_case(tmp_path, operation, external_pool_generation_queue_timeout=1, external_pool_generation_poll_wait=1)


def test_farm_without_capacity_refuses_and_keeps_the_lease(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        farm.mode['farm1'] = 'busy'
        assert await borrower.generate(PAYLOAD, farm='http://farm1') is None
        assert 'http://farm1' in eligible(borrower)
        assert any(e == 'borrow_generation_refused' for e, _ in events)
    poll_case(tmp_path, operation)


def test_late_result_from_an_abandoned_farm_is_never_taken(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        farm.mode['farm1'] = 'dead'
        assert await borrower.generate(PAYLOAD, farm='http://farm1') is None
        abandoned = [rid for rid, (host, _) in farm.jobs.items() if host == 'farm1']
        farm.mode['farm1'] = 'ok'
        result = await borrower.generate(PAYLOAD, farm='http://farm2')
        # The retry used a fresh id; the abandoned id is still unclaimed on farm1.
        assert result and abandoned and all(rid in farm.jobs for rid in abandoned)
    poll_case(tmp_path, operation)


# --- Client side: leases held only while generating ----------------------------

def test_leases_follow_generation_demand_and_are_released_when_generation_ends(tmp_path):
    async def operation(borrower, farm, archive, events):
        work = {'outstanding': 0, 'completed': 0}
        borrower.demand = lambda: work['outstanding']
        borrower.completed = lambda: work['completed']
        snapshot = await borrower.reserve()
        assert snapshot['reserved'] and len(held(borrower)) == 1  # one admission lease only
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        assert len(held(borrower)) == 1  # nothing generated yet, no demand yet
        work['outstanding'] = 8  # 8 groups, 4 per farm -> two farms wanted
        borrower.wake.set()
        await until(lambda: len(held(borrower)) == 2)
        work['outstanding'], work['completed'] = 0, 8  # last group generated; grading needs no lease
        borrower.wake.set()
        await until(lambda: held(borrower) == [])
        assert {f['service'] for e, f in events if e == 'borrow_member_released'} == {'http://farm1', 'http://farm2'}
        await borrower.end('phase-1')
        assert held(borrower) == [] and borrower.wanted() == 0
    multi_case(tmp_path, operation, external_pool_release_after_generation=True,
               external_pool_max_concurrent_requests=4)


def test_phase_end_releases_every_lease(tmp_path):
    async def operation(borrower, farm, archive, events):
        work = {'outstanding': 8, 'completed': 0}
        borrower.demand = lambda: work['outstanding']
        borrower.completed = lambda: work['completed']
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        borrower.wake.set()
        await until(lambda: len(held(borrower)) == 2)
        await borrower.end('phase-1')
        assert held(borrower) == [] and not farm.owners
    multi_case(tmp_path, operation, external_pool_release_after_generation=True,
               external_pool_max_concurrent_requests=4)


def test_returned_farm_is_acquired_mid_generation_when_demand_needs_it(tmp_path):
    async def operation(borrower, farm, archive, events):
        work = {'outstanding': 8, 'completed': 0}
        borrower.demand = lambda: work['outstanding']
        borrower.completed = lambda: work['completed']
        farm.down.add('farm2')
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        borrower.wake.set()
        await until(lambda: held(borrower) == ['http://farm1'])
        farm.down.discard('farm2')  # preempted farm comes back
        borrower.wake.set()
        await until(lambda: held(borrower) == ['http://farm1', 'http://farm2'])
    multi_case(tmp_path, operation, external_pool_release_after_generation=True,
               external_pool_max_concurrent_requests=4)


def test_run_scoped_borrower_keeps_retention_by_default(tmp_path):
    async def operation(borrower, farm, archive, events):
        assert not borrower.release_after_generation and borrower.wanted() == borrower.target
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        await borrower.end('phase-1')
        assert held(borrower) == ['http://farm1', 'http://farm2']
    multi_case(tmp_path, operation)


def test_release_after_generation_requires_the_multi_farm_borrower(tmp_path):
    from tests.tpu_swarm.test_inference_borrowing import config as borrowing_config
    from tpu.swarm.ray_train.config import Config
    raw = borrowing_config(tmp_path).to_dict()
    raw['inference']['external_pool_release_after_generation'] = True
    with pytest.raises(ValueError, match='multi-farm borrower'):
        Config.from_dict(raw)
