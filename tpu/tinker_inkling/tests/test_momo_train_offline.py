"""Offline tests for the STREAMING inkling_momo_train.Trainer with fake Tinker/discover pieces (no network, no grading).
Run: PYTHONPATH=<discover pin> <tinker venv>/bin/python tests/test_momo_train_offline.py   (exit 1 on any failure)"""
import asyncio, json, os, sys, tempfile, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['INKLING_MOMO_TRAIN_STORE'] = tempfile.mkdtemp()
import inkling_momo_train as M
import ttt_discover.rl.train as T

ENV = '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs/mc-s5/env.txt'
BASE_S, BASE_SP = 'tinker://base:train:0/weights/000012', 'tinker://base:train:0/sampler_weights/000012'
FAILED = []


def ok(c, msg):
    print(('PASS ' if c else 'FAIL ') + msg)
    if not c:
        FAILED.append(msg)


class G:
    def __init__(self, r): self.r = r
    def get_total_rewards(self): return self.r


class Fut:
    def __init__(self, v=None, fail=False): self.v, self.fail = v, fail
    async def result_async(self):
        if self.fail: raise RuntimeError('injected failure')
        return types.SimpleNamespace(path=self.v)


class TC:
    """Fake training client: records forward_backward datum counts and optim steps."""
    made = 0

    def __init__(self):
        TC.made += 1
        self.id, self.fb, self.optims = TC.made, [], 0
        self.fail_fb = self.fail_save = False

    async def save_state_async(self, name): return Fut(f'tinker://new:train:0/weights/{name}', self.fail_save)
    async def save_weights_for_sampler_async(self, name): return Fut(f'tinker://new:train:0/sampler_weights/{name}')


STATE = {'slow_prep': 0.0}


async def fake_client(self):
    if self.tc is None:
        self.tc = TC()
    return self.tc


async def fake_prep(builders, groups, *a, **k):
    if STATE['slow_prep']:
        time.sleep(STATE['slow_prep'])
    return [object() for g in groups for _ in g.r], {'advantage/mean': 0.0}


async def fake_enqueue_fb(tc, data, loss):
    fail = tc.fail_fb
    tc.fail_fb = False
    if not fail:
        tc.fb.append(len(data))
    return Fut(fail=fail)


async def fake_consume_fb(fut):
    await fut.result_async()


async def fake_enqueue_optim(tc, lr):
    tc.optims += 1
    return Fut()


async def fake_consume_optim(fut):
    return await fut.result_async()


M.Trainer._client = fake_client
T.prepare_minibatch, T.enqueue_forward_backward, T.consume_forward_backward = fake_prep, fake_enqueue_fb, fake_consume_fb
T.enqueue_optim_step, T.consume_optim_step = fake_enqueue_optim, fake_consume_optim


def new_trainer():
    return M.Trainer('circuit', ENV, BASE_S, BASE_SP, 'HARISH', False, 'unit', 'circuit-inkling-s5-a12')


async def stream(t, job, sampler, groups):
    t._save_job(job, sampler, len(groups))
    for i, g in enumerate(groups):
        t._spawn_stream(g, job, i, sampler)
    await asyncio.gather(*list(t.stream_tasks))


async def main():
    t = new_trainer()
    # 1. stream -> train -> v+1
    await stream(t, 'b:1:j1', BASE_SP, [G([0.1, 0.5, 0.0]), G([0.2, 0.9])])
    tc1 = t.tc
    r1 = await t.train(['b:1:j1'], BASE_SP, 1)
    ok(r1['update'] == 1 and r1['n_streamed_groups'] == 2 and r1['n_datums'] == 5 and sorted(tc1.fb) == [2, 3]
       and tc1.optims == 1 and t.newest() == r1['sampler_path'], 'stream 2 groups -> /train -> update 1 (one optim)')
    # 2. lost reply -> idempotent
    r1b = await t.train(['b:1:j1'], BASE_SP, 1)
    ok(r1b['update'] == 1 and tc1.optims == 1, 'lost reply: same job set returns the recorded update, no second step')
    v1 = t.newest()
    # 3. stale group (sampled from the base after update 1) is not streamed
    await stream(t, 'b:1:j2', BASE_SP, [G([0.1, 0.7])])
    ok(t.counts['stale'] == 1 and not list(t._vdir().glob('*.pkl')), 'stale-version group not streamed (counted stale)')
    # 4. constant group not streamed
    await stream(t, 'b:1:j3', v1, [G([0.4, 0.4])])
    ok(t.counts['constant'] == 1 and not list(t._vdir().glob('*.pkl')), 'constant-reward group not streamed')
    # 5. skip: nothing streamed in v1 -> skipped, no new version
    r_skip = await t.train(['b:1:j3'], v1, 2)
    ok(r_skip['skipped'] and t.update == 1 and t.newest() == v1, 'no streamed group -> skipped, update unchanged')
    # 6. late group: graded during the cut -> not streamed; /train counts only what was streamed before it
    await stream(t, 'b:1:j4', v1, [G([0.1, 0.6])])                      # one fast group, streamed
    STATE['slow_prep'] = 1.0
    t._save_job('b:1:j5', v1, 1)
    t._spawn_stream(G([0.3, 0.8]), 'b:1:j5', 0, v1)                     # prep in flight (1 s) when /train cuts
    await asyncio.sleep(0.2)
    r2 = await t.train(['b:1:j4'], v1, 2)
    await asyncio.gather(*list(t.stream_tasks))
    STATE['slow_prep'] = 0.0
    ok(r2['update'] == 2 and r2['n_streamed_groups'] == 1, f'group graded after the cut is late (streamed 1 in v1)')
    ok(not list(t._vdir().glob('*.pkl')) and t.counts['stale'] + t.counts['late'] >= 1,
       'late group never persisted into any version')
    v2 = t.newest()
    # 7. restart mid-version: persisted groups of v2 are re-streamed on the new process's client before the step
    await stream(t, 'b:1:j6', v2, [G([0.1, 0.9]), G([0.0, 0.5, 0.2])])
    t2 = new_trainer()
    ok(t2.dirty and t2.counts['streamed'] == 2 and t2.update == 2, 'restart mid-version: 2 persisted groups pending rebuild')
    r3 = await t2.train(['b:1:j6'], v2, 3)
    ok(r3['update'] == 3 and t2.tc.fb == [2, 3] and t2.tc.optims == 1 and r3['n_datums'] == 5,
       'restart: rebuild re-streams exactly the persisted set, then one step')
    t = t2
    v3 = t.newest()
    # 8. forward_backward failure -> dirty -> rebuild on a fresh client, no double count
    await stream(t, 'b:1:j7', v3, [G([0.1, 0.4])])
    t.tc.fail_fb = True
    await stream(t, 'b:1:j8', v3, [G([0.2, 0.6, 0.0])])
    await asyncio.gather(*t.fb_tasks)
    old, old_optims = t.tc, t.tc.optims       # this client already stepped update 3; it must not step again
    ok(t.dirty, 'failed forward_backward marks the version dirty')
    r4 = await t.train(['b:1:j7', 'b:1:j8'], v3, 4)
    ok(r4["update"] == 4 and t.tc is not old and sorted(t.tc.fb) == [2, 3] and t.tc.optims == 1
       and old.optims == old_optims, 'fb failure: rebuild on a fresh client with exactly the 2 persisted groups, one step')
    v4 = t.newest()
    # 9. save failure -> client discarded; the retry rebuilds and steps exactly once
    await stream(t, 'b:1:j9', v4, [G([0.1, 0.3])])
    bad = t.tc
    bad.fail_save = True
    try:
        await t.train(['b:1:j9'], v4, 5); ok(False, 'save failure raises')
    except RuntimeError:
        ok(t.tc is None and t.dirty and t.update == 4 and t.stream_open, 'save failure: client discarded, update 4, streaming reopened')
    r5 = await t.train(['b:1:j9'], v4, 5)
    ok(r5['update'] == 5 and t.tc is not bad and t.tc.fb == [2] and t.tc.optims == 1,
       'retry after a save failure: fresh client, re-streamed once, one step')
    v5 = t.newest()
    # 10. guards
    try:
        await t.train(['b:1:nope'], v5, 6); ok(False, '404')
    except M.NotFound: ok(True, '404 on an unknown job id')
    t._save_job('b:1:old', v4, 1)
    try:
        await t.train(['b:1:old'], v5, 6); ok(False, '409 stale job')
    except M.Conflict: ok(True, '409 on a job sampled by an older version')
    t._save_job('b:1:j10', v5, 1)
    try:
        await t.train(['b:1:j10'], v5, 9); ok(False, '409 expected_update')
    except M.Conflict: ok(True, '409 on an expected_update mismatch')
    try:
        await t.train(['b:1:j10'], v5, 'x'); ok(False, '400')
    except M.BadRequest: ok(True, '400 on a non-integer expected_update')
    try:
        await t.train(['b:1:j10'], 'tinker://other', 6); ok(False, '409 base')
    except M.Conflict: ok(True, '409 on a base_sampler_path that is not the newest')
    # 11. /sample keeps streaming during a /train (streaming reopens after it; a v5 group arriving mid-train is late)
    ok(t.stream_open, 'streaming open again after /train returns')
    # 12. torn ledger line + restart
    open(t.ledger, 'a').write('{"torn": ')
    t3 = new_trainer()
    ok(t3.update == 5 and t3.newest() == v5, 'restart after a torn ledger line resumes update 5')
    await stream(t3, 'b:1:j11', v5, [G([0.1, 0.2])])
    r6 = await t3.train(['b:1:j11'], v5, 6)
    t4 = new_trainer()
    ok(r6['update'] == 6 and t4.update == 6, 'row appended after a torn line is readable after restart')
    # 13. a store refuses a different starting point
    try:
        M.Trainer('circuit', ENV, 'tinker://x/weights/1', BASE_SP, 'HARISH', False, 'unit', 'circuit-inkling-s5-a12')
        ok(False, 'meta mismatch')
    except SystemExit:
        ok(True, 'store refuses a different starting state')

asyncio.run(main())
print(f'{len(FAILED)} failed' if FAILED else 'all passed')
sys.exit(1 if FAILED else 0)
