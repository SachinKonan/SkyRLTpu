"""Request protocol (ACK / refuse / poll / reroute) and the grading transport's use of it."""
import asyncio
import time

import pytest

from tpu.science import grading_transport as gt
from tpu.swarm.ray_train.request_protocol import Limits, Lost, Refused, drive
from tests.tpu_swarm.test_grading_transport import FakeLocal, GradingFarm, done, request, transport


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    async def sleep(self, seconds):
        self.t += seconds
        await asyncio.sleep(0)


def run_drive(views, *, limits, alternative=lambda: True, retired=lambda: False, submit_error=None):
    """Drive one request against scripted poll outcomes (a view dict, None = unknown, an Exception = no answer)."""
    clock, calls = Clock(), {'cancel': 0, 'ack': 0, 'polls': 0}
    views = list(views)

    async def submit():
        if submit_error:
            raise submit_error

    async def poll():
        calls['polls'] += 1
        clock.t += 1
        view = views.pop(0) if views else {'state': 'done', 'result': {}}
        if isinstance(view, Exception):
            raise view
        return view

    async def cancel():
        calls['cancel'] += 1

    async def go():
        return await drive(submit, poll, cancel, limits=limits, alternative=alternative, retired=retired,
                           on_ack=lambda: calls.__setitem__('ack', calls['ack'] + 1), now=clock, sleep=clock.sleep)
    return asyncio.run(go()), calls


def test_ack_then_done_returns_the_servers_view():
    view, calls = run_drive([{'state': 'running'}, {'state': 'done', 'result': {'r': 1}}], limits=Limits())
    assert view['result'] == {'r': 1} and calls['ack'] == 1 and calls['cancel'] == 0


def test_refusal_raises_before_any_ack():
    with pytest.raises(Refused):
        run_drive([], limits=Limits(), submit_error=Refused('full', 3))


def test_consecutive_poll_misses_lose_the_request_but_one_miss_does_not():
    view, _ = run_drive([ConnectionError('x'), None, {'state': 'done', 'result': {}}], limits=Limits(poll_miss_limit=3))
    assert view['state'] == 'done'
    with pytest.raises(Lost) as lost:
        run_drive([ConnectionError('x'), None, ConnectionError('y')], limits=Limits(poll_miss_limit=3))
    assert lost.value.kind == 'poll_miss'


def test_queued_too_long_cancels_once_and_moves_only_if_something_else_can_take_it():
    queued = [{'state': 'queued'}] * 10
    with pytest.raises(Lost) as lost:
        run_drive(queued, limits=Limits(queue_limit_s=3))
    assert lost.value.kind == 'queued_too_long'
    # Nowhere else to go: it stays queued where it is until it runs.
    view, calls = run_drive(queued + [{'state': 'running'}, {'state': 'done', 'result': {}}],
                            limits=Limits(queue_limit_s=3), alternative=lambda: False)
    assert view['state'] == 'done' and calls['cancel'] == 0
    # Running (not queued) past the limit is never moved.
    view, _ = run_drive([{'state': 'running'}] * 10, limits=Limits(queue_limit_s=3))
    assert view['state'] == 'done'


def test_a_retired_server_loses_the_request():
    with pytest.raises(Lost) as lost:
        run_drive([{'state': 'running'}], limits=Limits(), retired=lambda: True)
    assert lost.value.kind == 'retired'


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


def test_local_slots_fill_first_then_farms_then_local_queue_when_no_farm():
    now = time.monotonic()
    sched = gt.GradingScheduler()
    local = FakeLocal(2, {'result': [0.0], 'error': None, 'metrics': {}})
    farm = gt.FarmPool(dict(farm_id='a', url='http://a', token='t'), http=None)
    farm.capacities = {'math': 4}
    assert sched.choose([local, farm], now, 'math') is local  # free local slot wins
    local.running = 2
    assert sched.choose([local, farm], now, 'math') is farm  # local full: spread to farms
    farm.unhealthy_until = now + 60
    assert sched.choose([local, farm], now, 'math') is local  # no farm: queue locally
