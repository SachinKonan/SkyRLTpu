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
        status, view = jobs.submit('r1', dict(lease_id='lease-a'), work)
        assert status == 202 and view['state'] == 'queued'
        await asyncio.sleep(0)
        assert (await jobs.result('r1', dict(lease_id='lease-a')))['state'] == 'running'
        gate.set()
        done = await jobs.result('r1', dict(lease_id='lease-a'), wait=2)
        assert done['state'] == 'done' and done['result'] == {'choices': [1]}
        # Handed over once: a late poll (after a reroute) finds nothing.
        assert await jobs.result('r1', dict(lease_id='lease-a')) is None
    asyncio.run(run())


def test_repeated_submit_of_an_id_never_runs_the_group_twice():
    async def run():
        jobs, calls = GenerationJobs(), []

        async def work(mark_running):
            calls.append(1)
            await asyncio.sleep(.05)
            return {}
        assert jobs.submit('r1', dict(lease_id='lease-a'), work)[0] == 202
        assert jobs.submit('r1', dict(lease_id='lease-a'), work)[0] == 202  # lost acknowledgement, resubmitted
        assert jobs.submit('r1', dict(lease_id='lease-b'), work)[0] == 409  # another lease cannot claim the id
        await jobs.result('r1', dict(lease_id='lease-a'), wait=2)
        assert calls == [1]
    asyncio.run(run())


def test_cancel_and_lease_release_cancel_unfinished_groups():
    async def run():
        jobs = GenerationJobs()

        async def forever(mark_running):
            mark_running()
            await asyncio.Event().wait()
        jobs.submit('r1', dict(lease_id='lease-a'), forever)
        jobs.submit('r2', dict(lease_id='lease-a'), forever)
        jobs.submit('r3', dict(lease_id='lease-b'), forever)
        await asyncio.sleep(0)
        assert jobs.cancel('r1', dict(lease_id='lease-b')) is None  # not the owner
        assert jobs.cancel('r1', dict(lease_id='lease-a'))['state'] == 'cancelling'
        assert jobs.cancel_lease('lease-a', 'lease released') == 1
        await asyncio.sleep(0)
        assert (await jobs.result('r2', dict(lease_id='lease-a')))['state'] == 'cancelled'
        assert (await jobs.result('r3', dict(lease_id='lease-b')))['state'] == 'running'
        jobs.cancel_lease('lease-b', 'test over')
    asyncio.run(run())


def test_failed_group_reports_the_farm_status():
    async def run():
        jobs = GenerationJobs()

        class Rejected(Exception):
            status_code, detail = 422, 'bad prompt'

        async def bad(mark_running):
            raise Rejected()
        jobs.submit('r1', dict(lease_id='lease-a'), bad)
        view = await jobs.result('r1', dict(lease_id='lease-a'), wait=2)
        assert view['state'] == 'failed' and view['error']['status'] == 422 and 'bad prompt' in view['error']['detail']
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
        if path == '/health' and mode == 'dead':
            raise httpx.ConnectError('farm gone', request=request)
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


def test_farm_without_capacity_refuses_and_keeps_the_lease(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        farm.mode['farm1'] = 'busy'
        from tpu.swarm.ray_train.hybrid_scheduler import REFUSED
        assert await borrower.generate(PAYLOAD, farm='http://farm1') is REFUSED  # uncharged reroute
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
    multi_case(tmp_path, operation, external_pool_max_concurrent_requests=4)


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
    multi_case(tmp_path, operation, external_pool_max_concurrent_requests=4)


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
    multi_case(tmp_path, operation, external_pool_max_concurrent_requests=4)


# --- Robustness: TTL, restart reclaim, path check, retry cap, contract, transitions --------

def test_unpolled_group_is_cancelled_by_its_ttl_and_a_poll_renews_it():
    async def run():
        now = [0.0]
        jobs = GenerationJobs(ttl=10, clock=lambda: now[0])

        async def forever(mark_running):
            mark_running()
            await asyncio.Event().wait()
        jobs.submit('kept', dict(lease_id='l'), forever)
        jobs.submit('orphan', dict(lease_id='l'), forever)
        await asyncio.sleep(0)
        now[0] = 8
        assert (await jobs.result('kept', dict(lease_id='l')))['state'] == 'running'  # renews to 18
        now[0] = 12
        jobs.expire()
        await asyncio.sleep(0)
        assert jobs.entries['orphan']['state'] == 'cancelled' and 'TTL' in jobs.entries['orphan']['error']
        assert jobs.entries['kept']['state'] == 'running'
        jobs.cancel_lease('l', 'done')
    asyncio.run(run())


def test_run_identity_reclaims_a_group_after_the_lease_is_gone():
    async def run():
        jobs = GenerationJobs()
        owner = dict(lease_id='old-lease', run='run-a', token='t')

        async def work(mark_running):
            return {'choices': ['x']}
        jobs.submit('g1', owner, work)
        await asyncio.sleep(0)
        jobs.cancel_lease('old-lease', 'lease released')  # finished groups are not cancelled
        assert await jobs.result('g1', dict(run='run-a', token='wrong')) is None
        view = await jobs.result('g1', dict(lease_id='new-lease', run='run-a', token='t'))
        assert view['state'] == 'done' and view['result'] == {'choices': ['x']}
        assert jobs.submit('g2', owner, work)[0] == 202 and jobs.submit('g2', owner, work)[1]['duplicate']
        assert jobs.duplicates == 1
    asyncio.run(run())


def test_restarted_client_reclaims_finished_groups_and_regenerates_only_missing_ones(tmp_path):
    from tpu.swarm.ray_train.generation_journal import GenerationJournal, group_key

    async def operation(borrower, farm, archive, events):
        journal = GenerationJournal(tmp_path / 'run', 'run-a')
        borrower.journal = journal
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        # Previous incarnation: the farm finished one group, never answered for the other.
        key_done = group_key(PAYLOAD)
        other = dict(PAYLOAD, prompt=[9, 9])
        journal.record('acknowledged', key_done, url='http://farm1', request_id='old-1')
        journal.record('acknowledged', group_key(other), url='http://farm1', request_id='old-2')
        farm.jobs['old-1'] = ('farm1', PAYLOAD)
        restarted = GenerationJournal(tmp_path / 'run', 'run-a')
        assert set(restarted.open) == {key_done, group_key(other)} and restarted.token == journal.token
        borrower.journal = restarted
        await borrower.reclaim()
        before = sum(p == '/skyrl/v1/generation/submit' for _, p in farm.requests)
        assert len((await borrower.generate(PAYLOAD))['choices']) == 2  # served from the reclaim
        assert sum(p == '/skyrl/v1/generation/submit' for _, p in farm.requests) == before
        assert len((await borrower.generate(other))['choices']) == 2  # missing: regenerated
        assert sum(p == '/skyrl/v1/generation/submit' for _, p in farm.requests) == before + 1
        rows = [json.loads(line) for line in restarted.path.read_text().splitlines()]
        trail = [r['event'] for r in rows if r['key'] == group_key(other)]
        missing = trail.index('reclaim_missing')
        assert 'acknowledged' in trail[missing:] and trail[-1] == 'done'  # regenerated after the reclaim found nothing
    poll_case(tmp_path, operation)


def test_unused_reclaimed_results_are_dropped_and_logged_at_phase_end(tmp_path):
    from tpu.swarm.ray_train.generation_journal import GenerationJournal
    journal = GenerationJournal(tmp_path, 'run-a')
    journal.reclaimed('k1', {'choices': []})
    journal.discard_unused()
    rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert rows[-1]['event'] == 'duplicate_dropped' and not journal.cache


def test_path_failure_gets_one_more_round_before_the_farm_is_given_up(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        verdicts = []
        borrower.path_check = lambda url: verdicts.append(url) or 'farm_ok'
        farm.mode['farm1'] = 'dead'
        assert await borrower.generate(PAYLOAD, farm='http://farm1') is None
        polls = [p for h, p in farm.requests if h == 'farm1' and p.startswith('/skyrl/v1/generation/result/')]
        assert verdicts == ['http://farm1'] and len(polls) == 6  # two rounds of three misses
        assert any(e == 'borrow_generation_path_recovering' for e, _ in events)
    poll_case(tmp_path, operation)


def test_farm_refusing_the_group_contract_is_dropped(tmp_path):
    class ContractFarm(PollFarm):
        async def __call__(self, request):
            if request.url.path == '/skyrl/v1/generation/submit' and request.url.host == 'farm1':
                self.requests.append(('farm1', request.url.path))
                return httpx.Response(409, text='contract mismatch: serving compatibility hash')
            return await super().__call__(request)

    from tests.tpu_swarm import test_remote_only as base
    original = base.MultiFarm
    base.MultiFarm = ContractFarm

    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        assert await borrower.generate(PAYLOAD, farm='http://farm1') is None
        await until(lambda: 'http://farm1' not in eligible(borrower))
    try:
        multi_case(tmp_path, operation)
    finally:
        base.MultiFarm = original


def test_submit_carries_the_group_contract(tmp_path):
    async def operation(borrower, farm, archive, events):
        await borrower.reserve()
        await borrower.begin('phase-1', 'Qwen/Qwen3.5-27B', archive, expected_n=2)
        await until(lambda: len(eligible(borrower)) == 2)
        bodies = []
        original = farm.__call__

        async def spy(request):
            if request.url.path == '/skyrl/v1/generation/submit':
                bodies.append(json.loads(request.content))
            return await original(request)
        farm.__call__ = spy
        member = borrower.members['http://farm1']
        contract = member._generation_contract(member.lease, dict(PAYLOAD, model=member.lease.adapter))
        assert contract['compatibility_sha256'] == 'c' * 64 and contract['adapter'] == member.lease.adapter
    poll_case(tmp_path, operation)


def test_farm_side_contract_check_refuses_a_different_hash_or_adapter():
    from fastapi import HTTPException
    from tpu.swarm.ray_train import serving
    check = serving.Ingress.func_or_class.__mro__[1].check_generation_contract

    async def run():
        async def capabilities():
            return {'compatibility_sha256': 'a' * 64}
        farm = SimpleNamespace(capabilities=capabilities,
                               lease={'adapter_name': 'borrow-x', 'adapter_sha256': 'f' * 64})
        payload = {'model': 'borrow-x'}
        await check(farm, None, payload)  # older clients send no contract
        await check(farm, {'model': 'borrow-x', 'compatibility_sha256': 'a' * 64, 'adapter': 'borrow-x'}, payload)
        for bad in ({'compatibility_sha256': 'b' * 64}, {'adapter': 'borrow-y'}, {'model': 'other'}):
            with pytest.raises(HTTPException) as caught:
                await check(farm, bad, payload)
            assert caught.value.status_code == 409 and 'contract mismatch' in caught.value.detail
    asyncio.run(run())


def test_group_failing_on_farms_five_times_is_dropped_and_every_regeneration_logged():
    from tpu.swarm.ray_train.hybrid_scheduler import GroupDropped, HybridScheduler, REFUSED

    async def run():
        events = []
        router = HybridScheduler(0, 0, lambda: True, lambda e, **f: events.append((e, f)),
                                 farms=lambda: [('a', 1), ('b', 1)], max_remote_failures=5)
        calls = []

        async def remote(key):
            calls.append(key)
            return REFUSED if len(calls) == 1 else None  # one refusal (uncharged), then failures
        with pytest.raises(GroupDropped) as dropped:
            await router.run({'prompt': [1], 'n': 2}, None, remote)
        assert dropped.value.attempts == 5 and len(calls) == 6
        requeued = [f for e, f in events if e == 'remote_group_requeued']
        assert [f['attempt'] for f in requeued] == [2, 3, 4, 5] and all(f['group'] for f in requeued)
        assert router.regenerated == 4 and router.dropped == 1
        assert any(e == 'remote_group_refused' for e, _ in events)
        assert any(e == 'generation_group_dropped' and f['attempts'] == 5 for e, f in events)
    asyncio.run(run())


def test_phase_end_puts_regeneration_counts_into_the_step_metrics(monkeypatch):
    from tpu.swarm.ray_train import borrowing_phase

    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, timeout=None):
            if url.endswith('/end'):
                return Response({'ended': True, 'groups_regenerated': 3, 'groups_dropped': 1})
            return Response({})
    monkeypatch.setattr(borrowing_phase.httpx, 'AsyncClient', Client)
    metrics = {}

    async def run():
        async with borrowing_phase.sampling_phase('http://ingress', expected_n=2, metrics=metrics):
            pass
    asyncio.run(run())
    assert metrics == {'infra/groups_regenerated': 3, 'infra/groups_dropped': 1}


def test_runs_start_their_farm_walk_at_different_farms():
    import random
    starts = {random.Random(f'run-{i}:x').randrange(8) for i in range(20)}
    assert len(starts) > 1  # rotation by owner spreads first acquisitions


def test_wanted_counts_the_real_capacity_of_each_held_farm():
    """A v6e-8 lease takes 4 groups and a v4-32 8 (requests_per_engine x engines), not the configured cap."""
    from tpu.swarm.ray_train.multi_borrowing import MultiRunBorrower
    lease = lambda n: SimpleNamespace(lease=SimpleNamespace(max_requests=n))
    b = SimpleNamespace(phase='p', target=8, settings=SimpleNamespace(external_pool_max_concurrent_requests=8),
                        generated_this_phase=lambda: 1)
    want = lambda held, outstanding: MultiRunBorrower.wanted(
        SimpleNamespace(**vars(b), held=lambda: held, outstanding=lambda: outstanding))
    assert want([], 14) == 2                               # nothing held: size by the configured cap
    assert want([lease(4), lease(4)], 14) == 4             # two v6e-8 cover 8 of 14: two more like them
    assert want([lease(8), lease(4)], 6) == 1              # the v4-32 alone covers 6: the v6e-8 is surplus
    assert want([lease(8), lease(4)], 12) == 2
    assert want([lease(4)] * 8, 40) == 8                   # never above target
