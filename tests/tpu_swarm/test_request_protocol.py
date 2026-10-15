"""Request protocol (ACK / refuse / poll / reroute) and the grading transport's use of it."""
import asyncio
import time

import pytest

from tpu.science import grading_transport as gt
from tpu.swarm.ray_train.request_protocol import (ALLOWED, Config, Contract, Lost, Refused, State, Ticket,
                                                  drive, transition)
from tests.tpu_swarm.test_grading_transport import FakeLocal, GradingFarm, done, request, transport


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    async def sleep(self, seconds):
        self.t += seconds
        await asyncio.sleep(0)


def run_drive(views, *, queue_limit_s=None, alternative=lambda: True, retired=lambda: False, submit_error=None,
              confirm_lost=None, resume=False, config=Config()):
    """Drive one ticket against scripted polls (a view dict, None = unknown id, an Exception = no answer)."""
    clock, calls, log = Clock(), {'cancel': 0, 'accept': 0}, []
    views = list(views)
    ticket = Ticket('r1', 'farm-a')

    async def submit():
        if submit_error:
            raise submit_error

    async def poll():
        clock.t += 1
        view = views.pop(0) if views else {'state': 'done', 'result': {}}
        if isinstance(view, Exception):
            raise view
        return view

    async def cancel():
        calls['cancel'] += 1

    async def go():
        return await drive(ticket, submit, poll, cancel, config=config, queue_limit_s=queue_limit_s,
                           alternative=alternative, retired=retired, confirm_lost=confirm_lost, resume=resume,
                           on_accept=lambda: calls.__setitem__('accept', calls['accept'] + 1),
                           report=lambda event, **f: log.append((event, f)), now=clock, sleep=clock.sleep)
    try:
        view = asyncio.run(go())
    except (Refused, Lost) as exc:
        view = exc
    return view, ticket, calls, log


def path(log):
    return [f['to'] for e, f in log if e == 'request_state']


def test_every_transition_is_checked_and_logged():
    ticket = Ticket('r', 's')
    with pytest.raises(RuntimeError):
        transition(ticket, State.DONE)  # SUBMITTED cannot jump to DONE
    for final in (State.DONE, State.LOST, State.REFUSED, State.CANCELLED, State.FAILED_INFRA):
        assert not ALLOWED.get(final)  # final states have no exits


def test_ack_running_done():
    view, ticket, calls, log = run_drive([{'state': 'queued'}, {'state': 'running'}, {'state': 'done', 'result': {'r': 1}}])
    assert ticket.state == State.DONE and view['result'] == {'r': 1} and calls['accept'] == 1
    assert path(log) == ['accepted', 'running', 'done']


def test_candidate_failure_is_final_not_infrastructure():
    view, ticket, _, _ = run_drive([{'state': 'done', 'result': {'error': 'SyntaxError'}}])
    assert ticket.state == State.FAILED_CANDIDATE


def test_server_infrastructure_failure_and_server_cancel_are_final_views():
    _, ticket, _, _ = run_drive([{'state': 'failed', 'error': {'detail': 'worker died'}}])
    assert ticket.state == State.FAILED_INFRA
    _, ticket, _, _ = run_drive([{'state': 'cancelled'}])
    assert ticket.state == State.CANCELLED


def test_refusal_before_any_ack():
    exc, ticket, calls, log = run_drive([], submit_error=Refused('full', 3))
    assert isinstance(exc, Refused) and ticket.state == State.REFUSED and calls['accept'] == 0


def test_poll_misses_lose_only_a_confirmed_dead_server():
    _, ticket, _, _ = run_drive([ConnectionError('x'), None, {'state': 'done', 'result': {}}])
    assert ticket.state == State.DONE  # two misses, then an answer
    exc, ticket, _, _ = run_drive([ConnectionError('x')] * 3)
    assert isinstance(exc, Lost) and exc.kind == 'poll_miss' and ticket.state == State.LOST

    async def path_failure():
        return False  # the farm answers a direct probe / discovery still sees it healthy
    _, ticket, _, log = run_drive([ConnectionError('x')] * 3 + [{'state': 'done', 'result': {}}],
                                  confirm_lost=path_failure)
    assert ticket.state == State.DONE and any(e == 'request_path_recovering' for e, _ in log)
    exc, _, _, _ = run_drive([ConnectionError('x')] * 6, confirm_lost=path_failure)
    assert isinstance(exc, Lost)  # only one extra round


def test_queued_too_long_moves_only_with_a_free_slot_elsewhere_and_never_once_running():
    queued = [{'state': 'queued'}] * 10
    exc, ticket, calls, _ = run_drive(queued, queue_limit_s=3)
    assert isinstance(exc, Lost) and exc.kind == 'queued_too_long' and ticket.state == State.LOST
    _, ticket, calls, _ = run_drive(queued, queue_limit_s=3, alternative=lambda: False)
    assert ticket.state == State.DONE and calls['cancel'] == 0
    _, ticket, calls, _ = run_drive([{'state': 'running'}] * 10, queue_limit_s=3)
    assert ticket.state == State.DONE and calls['cancel'] == 0


def test_retired_server_and_resume_of_an_unknown_id():
    exc, _, _, _ = run_drive([{'state': 'running'}], retired=lambda: True)
    assert exc.kind == 'retired'
    exc, ticket, _, log = run_drive([None], resume=True)
    assert exc.kind == 'resume_unknown' and path(log) == ['accepted', 'lost']
    _, ticket, _, _ = run_drive([{'state': 'done', 'result': {}}], resume=True)
    assert ticket.state == State.DONE


def test_contract_digest_changes_with_any_field():
    a, b = Contract('grading', 'x'), Contract('grading', 'y')
    assert a.digest() != b.digest() and a.wire()['sha256'] == a.digest()
    assert Contract('generation', extra=(('swa_kv', '1'),)).digest() != Contract('generation').digest()


# --- The grading transport on top of it -------------------------------------------------------------------

def test_a_refusing_farm_sends_the_grade_to_another_uncharged():
    async def run():
        farm = GradingFarm()
        farm.busy.add('a')
        farm.outcomes['b'] = done({'result': [2.0], 'error': None, 'metrics': {}})
        t = transport(farm, ['a', 'b'])
        try:
            result = await asyncio.wait_for(t.grade(request()), 5)
            assert result['result'] == [2.0] and [h for h, _ in farm.submits] == ['b']
            assert not any(e == 'grading_retry' for e, _ in t.reports)
        finally:
            await t.close()
    asyncio.run(run())


def test_when_every_farm_refuses_the_grade_waits_then_goes_through():
    async def run():
        farm = GradingFarm()
        farm.busy.update({'a', 'b'})
        farm.outcomes['b'] = done({'result': [2.0], 'error': None, 'metrics': {}})
        t = transport(farm, ['a', 'b'])
        try:
            task = asyncio.create_task(t.grade(request()))
            await asyncio.sleep(.1)
            assert not task.done() and not farm.submits
            farm.busy.discard('b')
            assert (await asyncio.wait_for(task, 5))['result'] == [2.0]
        finally:
            await t.close()
    asyncio.run(run())


def test_queued_too_long_cancels_on_the_first_farm_and_regrades_elsewhere():
    async def run():
        farm = GradingFarm()
        farm.outcomes['a'] = lambda rid: {'request_id': rid, 'state': 'queued'}
        farm.outcomes['b'] = done({'result': [2.0], 'error': None, 'metrics': {}})
        t = transport(farm, ['a', 'b'])
        try:
            await t._ensure_started()
            t.directory.pools['b'].unhealthy_until = time.monotonic() + .2  # 'a' first.
            req = request()
            req.time_cap_s = .05  # queue limit = 2 x 0.05 s
            result = await asyncio.wait_for(t.grade(req), 5)
            assert result['result'] == [2.0]
            assert [h for h, _ in farm.submits] == ['a', 'b']
            assert [h for h, _ in farm.cancels] == ['a']  # once, on the farm it left
            kinds = [f['kind'] for e, f in t.reports if e == 'grading_reroute']
            assert kinds == ['queued_too_long'] and not any(e == 'grading_retry' for e, _ in t.reports)
            assert len(t.results) == 1
        finally:
            await t.close()
    asyncio.run(run())


def test_a_farm_leaving_the_directory_mid_grade_reroutes_its_grade():
    async def run():
        farm = GradingFarm()
        farm.outcomes['b'] = done({'result': [2.0], 'error': None, 'metrics': {}})
        farms = ['a', 'b']
        t = transport(farm, farms)
        try:
            await t._ensure_started()
            t.directory.pools['b'].unhealthy_until = time.monotonic() + .2
            req = request()
            task = asyncio.create_task(t.grade(req))
            for _ in range(100):
                if ('a', req.request_id) in farm.entries:
                    break
                await asyncio.sleep(.01)
            farms.remove('a')  # the live directory drops 'a'
            assert (await asyncio.wait_for(task, 5))['result'] == [2.0]
            assert [f['kind'] for e, f in t.reports if e == 'grading_reroute'] == ['retired']
        finally:
            await t.close()
    asyncio.run(run())


def test_a_farm_joining_the_directory_is_used_at_once():
    async def run():
        farm = GradingFarm()
        farm.outcomes['late'] = done({'result': [3.0], 'error': None, 'metrics': {}})
        farms = []
        t = transport(farm, farms)
        try:
            task = asyncio.create_task(t.grade(request()))
            await asyncio.sleep(.1)
            farms.append('late')
            assert (await asyncio.wait_for(task, 5))['result'] == [3.0]
        finally:
            await t.close()
    asyncio.run(run())


def test_local_slots_fill_first_then_farms_and_a_busy_local_pool_is_no_candidate():
    now = time.monotonic()
    sched = gt.GradingScheduler()
    local = FakeLocal(2, {'result': [0.0], 'error': None, 'metrics': {}})
    farm = gt.FarmPool(dict(farm_id='a', url='http://a', token='t'), http=None)
    farm.capacities = {'math': 4}
    assert sched.choose([local, farm], now, 'math') is local  # free local slot wins
    local.running = 2
    assert sched.choose([local, farm], now, 'math') is farm  # local full: spread to farms
    farm.unhealthy_until = now + 60
    assert sched.choose([local, farm], now, 'math') is None  # no farm, local busy: wait, never queue locally
    local.running = 1
    assert sched.choose([local, farm], now, 'math') is local  # a local slot frees: it takes the wait


# --- Journal, exactly-once, retry cap, contract, registry, logging ----------------------------------------

def test_a_restarted_client_resumes_journaled_requests_instead_of_regrading(tmp_path):
    async def run():
        farm = GradingFarm()
        farm.outcomes['a'] = lambda rid: {'request_id': rid, 'state': 'running'}
        journal = tmp_path / 'journal.jsonl'
        first = transport(farm, ['a'], journal=str(journal))
        req = request()
        task = asyncio.create_task(first.grade(req))
        for _ in range(100):
            if journal.exists() and 'submitted' in journal.read_text():
                break
            await asyncio.sleep(.01)
        task.cancel()  # the client dies with the grade still running on the farm
        await asyncio.gather(task, return_exceptions=True)
        await first.close()
        farm.cancels.clear()
        farm.outcomes['a'] = done({'result': [5.0], 'error': None, 'metrics': {}})
        again = transport(farm, ['a'], journal=str(journal))
        result = await asyncio.wait_for(again.grade(gt.GradingRequest(task='math', spec=req.spec,
                                                                      request_id=req.request_id)), 5)
        assert result['result'] == [5.0] and len(farm.submits) == 1  # re-polled, not resubmitted
        await again.close()
        # A third client finds the finished result in the journal and never contacts a farm.
        third = transport(farm, ['a'], journal=str(journal))
        assert (await third.grade(gt.GradingRequest(task='math', spec=req.spec, request_id=req.request_id)))['result'] == [5.0]
        assert len(farm.submits) == 1
        await third.close()
    asyncio.run(run())


def test_concurrent_callers_of_one_id_share_one_grade():
    async def run():
        farm = GradingFarm()
        farm.outcomes['a'] = done({'result': [1.0], 'error': None, 'metrics': {}})
        t = transport(farm, ['a'])
        try:
            req = request()
            results = await asyncio.gather(*(t.grade(req) for _ in range(5)))
            assert all(r is results[0] for r in results) and len(farm.submits) == 1
        finally:
            await t.close()
    asyncio.run(run())


def test_infrastructure_failures_past_the_cap_drop_the_candidate_without_aborting_the_step():
    async def run():
        farm = GradingFarm()
        farm.outcomes['a'] = lambda rid: {'request_id': rid, 'state': 'failed',
                                          'error': {'class': 'infrastructure', 'detail': 'worker died'}}
        t = transport(farm, ['a'], max_infra_retries=2)
        try:
            await t._ensure_started()
            t.directory.pools['a'].unhealthy_until = 0
            with pytest.raises(gt.GradingInfrastructureFailed) as failure:
                for pool in t.pools():
                    pool.unhealthy_until = 0
                orig = t.directory.pools['a']
                # Keep the only farm usable between attempts so the cap, not the wait, ends it.
                orig.healthy = lambda now: True
                await asyncio.wait_for(t.grade(request()), 5)
            assert failure.value.drop_candidate and not failure.value.abort_training_step
            assert len(farm.submits) == 3
            assert any(e == 'grading_infra_failure' and f['dropped'] for e, f in t.reports)
        finally:
            await t.close()
    asyncio.run(run())


def test_a_contract_mismatch_bans_that_farm_for_this_client():
    async def run():
        farm = GradingFarm()
        farm.mismatch.add('a')
        farm.outcomes['b'] = done({'result': [2.0], 'error': None, 'metrics': {}})
        t = transport(farm, ['a', 'b'])
        try:
            await t._ensure_started()
            t.directory.pools['b'].unhealthy_until = time.monotonic() + .1  # 'a' first
            assert (await asyncio.wait_for(t.grade(request()), 5))['result'] == [2.0]
            assert t.directory.pools['a'].contract_mismatch
            assert not t.directory.pools['a'].healthy(time.monotonic() + 10 ** 6)
        finally:
            await t.close()
    asyncio.run(run())


def test_every_ticket_transition_is_logged_with_id_server_and_attempt():
    async def run():
        farm = GradingFarm()
        farm.outcomes['a'] = done({'result': [1.0], 'error': None, 'metrics': {}})
        t = transport(farm, ['a'])
        try:
            req = request()
            await t.grade(req)
            states = [(f['from'], f['to']) for e, f in t.reports if e == 'request_state']
            assert states == [('submitted', 'accepted'), ('accepted', 'done')]
            assert all(f['request_id'] == req.request_id and f['server'] == 'a' and 'attempt' in f
                       for e, f in t.reports if e == 'request_state')
        finally:
            await t.close()
    asyncio.run(run())


def test_a_stale_registry_is_reported_and_its_farms_probed_directly(tmp_path, monkeypatch):
    import json
    import os
    path = tmp_path / 'farms.json'
    path.write_text(json.dumps({'farms': [{'name': 'x', 'url': 'http://x', 'ok': False, 'tunnel_ok': False}]}))
    old = time.time() - 10_000
    os.utime(path, (old, old))
    reports = []
    directory = gt.OpenFarmDirectory(str(path), http=None, report=lambda e, **f: reports.append(e))

    async def no_probe(self):
        return False
    monkeypatch.setattr(gt.OpenFarmPool, 'probe_capacity', no_probe)
    asyncio.run(directory.refresh())
    assert 'grading_registry_stale' in reports and 'x' in directory.pools  # listed despite ok=False


# --- Server side: TTL, batch long-poll, original age -------------------------------------------------------

def test_server_ttl_cancels_unpolled_requests_and_batch_polls_renew_it(tmp_path, monkeypatch):
    from tests.tpu_swarm.test_farm_grading_service import body, grading_farm
    from tests.tpu_swarm.test_open_grading import open_headers

    async def run():
        async with grading_farm(tmp_path, monkeypatch) as (client, gateway, service, futures, cancelled, _e):
            a, b = body(), body()
            for r in (a, b):
                assert (await client.post('/skyrl/v1/grading/submit', json=r, headers=open_headers())).status_code == 202
            views = (await client.post('/skyrl/v1/grading/results', headers=open_headers(),
                                       json={'ids': [a['request_id'], b['request_id'], 'f' * 32]})).json()['views']
            assert views[a['request_id']]['state'] in ('queued', 'running') and views['f' * 32] is None
            service.entries[b['request_id']].expires = 0  # b's client stopped polling
            assert await service.expire() == 1
            assert service.entries[b['request_id']].state == 'cancelled'
            assert service.entries[a['request_id']].state != 'cancelled'
    asyncio.run(run())


def test_server_queues_a_moved_request_by_its_original_age(tmp_path, monkeypatch):
    from tests.tpu_swarm.test_farm_grading_service import body, grading_farm
    from tests.tpu_swarm.test_open_grading import open_headers

    async def run():
        async with grading_farm(tmp_path, monkeypatch, slots_per_host=1) as (client, gateway, service, futures, _c, _e):
            blockers = [body() for _ in range(4)]  # fill the 4 slots
            for r in blockers:
                await client.post('/skyrl/v1/grading/submit', json=r, headers=open_headers())
            young, old = body(), body()
            await client.post('/skyrl/v1/grading/submit', json=young, headers=open_headers())
            await client.post('/skyrl/v1/grading/submit', json=dict(old, submitted_at=time.time() - 3600),
                              headers=open_headers())
            queue = service.queues['math']['open-run']
            assert [e.request_id for e in queue] == [old['request_id'], young['request_id']]
    asyncio.run(run())


def test_journaled_runs_give_each_grade_a_stable_id(monkeypatch):
    from tpu.science.training_env import transport_requests
    monkeypatch.setenv('SKYRL_RUN_ID', 'run-x')
    monkeypatch.delenv('SKYRL_GRADING_JOURNAL', raising=False)
    assert transport_requests('placement', 'src', 600)[0].request_id != transport_requests('placement', 'src', 600)[0].request_id
    monkeypatch.setenv('SKYRL_GRADING_JOURNAL', '/tmp/j.jsonl')
    first, again = transport_requests('placement', 'src', 600), transport_requests('placement', 'src', 600)
    assert [r.request_id for r in first] == [r.request_id for r in again]
    assert len({r.request_id for r in first}) == len(first)  # one id per case
    assert transport_requests('placement', 'other', 600)[0].request_id != first[0].request_id
