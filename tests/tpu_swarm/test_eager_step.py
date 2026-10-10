"""Frozen-policy eager steps: a step's groups return once most candidates are graded; the rest land later."""
import asyncio
import time
from types import SimpleNamespace

from ttt_discover.rl import train
from ttt_discover.rl.types import StepResult


class Ob:
    def to_ints(self):
        return [1, 2, 3]


class Env:
    def __init__(self, delay, log):
        self.delay, self.log = delay, log

    async def initial_observation(self):
        return Ob(), 'stop'

    async def step(self, action, step_idx):
        await asyncio.sleep(self.delay)
        self.log.append(self.delay)  # stands in for the env's own PUCT tree update
        return StepResult(reward=self.delay, episode_done=True, next_observation=Ob(), next_stop_condition='stop',
                          metrics={'reward': self.delay})


class Builder:
    def __init__(self, delays, log):
        self.envs = [Env(d, log) for d in delays]

    async def make_envs(self):
        return self.envs

    async def compute_group_rewards(self, trajectories, envs):
        assert len(trajectories) == len(envs)
        return [(0.0, {}) for _ in trajectories]


class Policy:
    generated = 0

    async def sample_group(self, ob, stop, n):
        await asyncio.sleep(0.01)
        Policy.generated += 1
        return [SimpleNamespace(tokens=[7]) for _ in range(n)]


def test_groups_return_at_the_release_fraction_and_late_candidates_land_afterwards():
    async def run():
        log, late = [], []
        eager = train.EagerStep(groups=2, group_size=4, fraction=0.75,
                                on_late=lambda builder, task: late.append(task.result()))
        builders = [Builder([0.01, 0.01, 0.01, 1.0], log), Builder([0.01, 0.02, 0.01, 1.0], log)]
        start = time.monotonic()
        groups = await asyncio.gather(*(train._eager_group_rollout(b, Policy(), 3, eager) for b in builders))
        assert time.monotonic() - start < 0.5
        assert [len(g.trajectories_G) for g in groups] == [3, 3]
        assert eager.late_candidates == 2 and len(train.LATE_CANDIDATES) == 2
        await asyncio.gather(*list(train.LATE_CANDIDATES))
        await asyncio.sleep(0)
        assert len(late) == 2 and sorted(log) == [0.01] * 5 + [0.02] + [1.0] * 2
        assert not train.LATE_CANDIDATES
    asyncio.run(run())


def test_release_waits_for_every_group_to_generate():
    async def run():
        class SlowPolicy(Policy):
            async def sample_group(self, ob, stop, n):
                await asyncio.sleep(0.3)
                return await super().sample_group(ob, stop, n)
        eager = train.EagerStep(groups=2, group_size=4, fraction=0.875)
        fast = train._eager_group_rollout(Builder([0.01] * 3 + [5.0], []), Policy(), 0, eager)
        slow = train._eager_group_rollout(Builder([0.01] * 4, []), SlowPolicy(), 0, eager)
        groups = await asyncio.wait_for(asyncio.gather(fast, slow), 2)
        # Release needs 7 of 8 grades and both groups generated: the fast group's 3 quick grades are
        # in long before the slow group generates, so it returns only once the slow group is done.
        assert [len(g.trajectories_G) for g in groups] == [3, 4]
        for task in list(train.LATE_CANDIDATES):
            task.cancel()
        await asyncio.gather(*list(train.LATE_CANDIDATES), return_exceptions=True)
    asyncio.run(run())


def test_without_an_eager_step_the_group_waits_for_every_candidate():
    async def run():
        eager = train.EagerStep(groups=1, group_size=4, fraction=1.0)
        group = await train._eager_group_rollout(Builder([0.01, 0.01, 0.01, 0.2], []), Policy(), 0, eager)
        assert len(group.trajectories_G) == 4 and eager.late_candidates == 0
    asyncio.run(run())
