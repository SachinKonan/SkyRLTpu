import asyncio

import pytest

from tpu.science.grading_dedup import BatchGrader


def test_concurrent_and_late_duplicates_keep_separate_results():
    async def run():
        grader = BatchGrader()
        calls = 0

        async def evaluate():
            nonlocal calls
            calls += 1
            await asyncio.sleep(.01)
            return {'reward': .5, 'metrics': {'cases': [1, 2]}}

        a, b = await asyncio.gather(*(grader.grade(('run', 0), 'code', evaluate) for _ in range(2)))
        assert calls == 1 and {a[1], b[1]} == {True, False}
        a[0]['metrics']['cases'].append(3)
        c, reused = await grader.grade(('run', 0), 'code', evaluate)
        assert reused and c['metrics']['cases'] == b[0]['metrics']['cases'] == [1, 2]
        await grader.grade(('run', 0), 'code\n', evaluate)
        await grader.grade(('run', 1), 'code', evaluate)
        await grader.grade(('other-run', 0), 'code', evaluate)
        assert calls == 4
    asyncio.run(run())


def test_infrastructure_failure_is_retried():
    async def run():
        grader = BatchGrader()
        calls = 0

        async def evaluate():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError('worker lost')
            return {'reward': 0}

        with pytest.raises(RuntimeError, match='worker lost'):
            await grader.grade(0, 'code', evaluate)
        assert await grader.grade(0, 'code', evaluate) == ({'reward': 0}, False)
        assert calls == 2
    asyncio.run(run())


def test_cancellation_preserves_other_consumer_and_cancels_unneeded_work():
    async def run():
        grader = BatchGrader()
        started, finish, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def evaluate():
            started.set()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return {'reward': .5}

        first = asyncio.create_task(grader.grade(0, 'code', evaluate))
        await started.wait()
        second = asyncio.create_task(grader.grade(0, 'code', evaluate))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not cancelled.is_set()
        finish.set()
        assert await second == ({'reward': .5}, True)
        finish.clear()
        started.clear()
        last = asyncio.create_task(grader.grade(1, 'code', evaluate))
        await started.wait()
        last.cancel()
        with pytest.raises(asyncio.CancelledError):
            await last
        await asyncio.wait_for(cancelled.wait(), 1)
        assert 'code' not in grader.batches[1]
    asyncio.run(run())
