"""Offline tests for the STREAMING inkling_momo_train.Trainer with fake Tinker/discover pieces (no network, no grading).
Run: PYTHONPATH=<discover pin> <tinker venv>/bin/python tests/test_momo_train_offline.py   (exit 1 on any failure)"""
import asyncio, json, os, sys, tempfile, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['INKLING_MOMO_TRAIN_STORE'] = tempfile.mkdtemp()
import inkling_momo_train as M
import ttt_discover.rl.train as T

ENV = os.environ.get('INKLING_TEST_ENV_TXT', '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs/mc-s5/env.txt')
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


STATE = {'slow_prep': 0.0, 'slow_optim': 0.0, 'fb_calls': 0, 'raise_on_fb_call': None}


async def fake_client(self):
    if self.tc is None:
        self.tc = TC()
    return self.tc


async def fake_prep(builders, groups, *a, **k):
    if STATE['slow_prep']:
        time.sleep(STATE['slow_prep'])
    return [object() for g in groups for _ in g.r], {'advantage/mean': 0.0}


async def fake_enqueue_fb(tc, data, loss):
    STATE['fb_calls'] += 1
    if STATE['raise_on_fb_call'] == STATE['fb_calls']:
        raise RuntimeError('enqueue failed (injected)')
    fail = tc.fail_fb
    tc.fail_fb = False
    if not fail:
        tc.fb.append(len(data))
    return Fut(fail=fail)


async def fake_consume_fb(fut):
    await fut.result_async()


async def fake_enqueue_optim(tc, lr):
    if STATE['slow_optim']:
        await asyncio.sleep(STATE['slow_optim'])
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
    need = {'update', 'sampler_path', 'state_path', 'base_sampler_path', 'n_streamed_groups', 'n_groups', 'n_datums',
            'n_dropped', 'n_stale_groups', 'n_late_groups', 'n_constant_groups', 'n_refused_groups', 'seconds',
            'published', 'skipped', 'jobs_key', 'job_ids'}
    ok(need <= set(r1) and r1['n_groups'] == r1['n_streamed_groups'] == 2 and r1['skipped'] is False,
       f'reply carries every field the client reads (missing: {sorted(need - set(r1))})')
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
    # 11. a group of the CURRENT version arriving during a SUCCESSFUL /train (stream closed): persisted late, not in the
    #     gradient, and discarded with the old version
    await stream(t, 'b:1:j12', v5, [G([0.1, 0.5])])
    tc_now = t.tc
    STATE['slow_optim'] = 0.5
    t._save_job('b:1:j13', v5, 1)
    tr = asyncio.create_task(t.train(['b:1:j12'], v5, 6))
    await asyncio.sleep(0.2)
    ok(not t.stream_open, 'stream closed during /train')
    await stream(t, 'b:1:j13', v5, [G([0.3, 0.9, 0.0])])       # arrives mid-train
    r_mid = await tr
    STATE['slow_optim'] = 0.0
    v5_dir = t.store / 'stream' / 'v0005'
    ok(r_mid['update'] == 6 and tc_now.fb[-1] == 2 and 3 not in tc_now.fb[-1:]
       and any('b_1_j13' in p.name for p in v5_dir.glob('*.pkl')) and not list(t._vdir().glob('*.pkl'))
       and t.stream_open and not t.dirty, 'mid-train group: persisted late in the old version, not trained, v+1 empty')
    v6 = t.newest()
    # 12. late groups of a version that CONTINUES (the /train fails): the retry's rebuild includes them
    await stream(t, 'b:1:j14', v6, [G([0.1, 0.4])])
    t.tc.fail_save = True
    STATE['slow_optim'] = 0.5
    t._save_job('b:1:j15', v6, 1)
    tr = asyncio.create_task(t.train(['b:1:j14'], v6, 7))
    await asyncio.sleep(0.2)
    await stream(t, 'b:1:j15', v6, [G([0.2, 0.8, 0.1])])       # late during the failing /train
    try:
        await tr; ok(False, 'failing /train raises')
    except RuntimeError:
        pass
    STATE['slow_optim'] = 0.0
    ok(t.dirty and t.update == 6, 'failed /train: version continues, dirty')
    r_retry = await t.train(['b:1:j14'], v6, 7)
    ok(r_retry['update'] == 7 and sorted(t.tc.fb) == [2, 3] and t.tc.optims == 1,
       'late group of a continuing version is included in the retry (2 + 3 datums, one step)')
    v7 = t.newest()
    # 13. a group streamed while dirty is persisted only (no enqueue) and included by the rebuild
    t5 = new_trainer()                                          # restart: v7 has no persisted groups -> clean
    await stream(t5, 'b:1:j16', v7, [G([0.1, 0.6])])
    t5.dirty = True                                             # e.g. an fb failed earlier
    await stream(t5, 'b:1:j17', v7, [G([0.2, 0.7, 0.3])])
    first = t5.tc
    ok(first.fb == [2], 'group streamed while dirty is persisted but not enqueued')
    r13 = await t5.train(['b:1:j16', 'b:1:j17'], v7, 8)
    ok(r13['update'] == 8 and t5.tc is not first and sorted(t5.tc.fb) == [2, 3], 'rebuild includes the persist-only group')
    t = t5
    v8 = t.newest()
    # 14. rebuild failing midway: /train raises, nothing stepped; the next /train rebuilds the full set once
    await stream(t, 'b:1:j18', v8, [G([0.1, 0.5]), G([0.2, 0.6, 0.0])])
    t.dirty = True
    STATE['raise_on_fb_call'] = STATE['fb_calls'] + 2             # the 2nd re-stream of the rebuild fails
    try:
        await t.train(['b:1:j18'], v8, 9); ok(False, 'rebuild failure raises')
    except RuntimeError:
        ok(t.dirty and t.update == 8 and t.stream_open, 'rebuild failing midway: no step, still dirty, streaming open')
    STATE['raise_on_fb_call'] = None
    r14 = await t.train(['b:1:j18'], v8, 9)
    ok(r14['update'] == 9 and sorted(t.tc.fb) == [2, 3] and t.tc.optims == 1, 'next /train rebuilds the full set once')
    v9 = t.newest()
    # 15. enqueue itself raising while streaming: the group is persisted, the version goes dirty, /train includes it
    STATE['raise_on_fb_call'] = STATE['fb_calls'] + 1
    await stream(t, 'b:1:j19', v9, [G([0.1, 0.9])])
    STATE['raise_on_fb_call'] = None
    ok(t.dirty and len(list(t._vdir().glob('*.pkl'))) == 1, 'enqueue failure: persisted + dirty')
    r15 = await t.train(['b:1:j19'], v9, 10)
    ok(r15['update'] == 10 and t.tc.fb == [2], 'enqueue failure: the rebuild trains it exactly once')
    v10 = t.newest()
    # 16. a cancelled /sample (caller gone) does not stop a stream already spawned for a graded group
    t._save_job('b:1:j20', v10, 1)

    async def fake_sample():
        t._spawn_stream(G([0.1, 0.4]), 'b:1:j20', 0, v10)
        await asyncio.sleep(10)
    STATE['slow_prep'] = 0.3
    fs = asyncio.create_task(fake_sample())
    await asyncio.sleep(0.05)
    fs.cancel()
    await asyncio.gather(*list(t.stream_tasks))
    STATE['slow_prep'] = 0.0
    ok(len(list(t._vdir().glob('*.pkl'))) == 1 and t.counts['streamed'] == 1, 'cancelled /sample: the spawned stream still completes')
    # 16b. the ledger append failing after a successful step: no bump, client discarded; the retry steps exactly once
    v_now, u_now = t.newest(), t.update
    t._save_job('b:1:j21', v_now, 1)
    await stream(t, 'b:1:j21', v_now, [G([0.1, 0.5])])
    stepped = t.tc
    real_append = t._append
    calls = {'n': 0}

    def failing_append(row):
        calls['n'] += 1
        if calls['n'] == 1:
            raise OSError('disk full (injected)')
        return real_append(row)
    t._append = failing_append
    try:
        await t.train(['b:1:j20', 'b:1:j21'], v_now, u_now + 1); ok(False, 'ledger failure raises')
    except OSError:
        ok(t.update == u_now and t.newest() == v_now and t.tc is None and t.dirty,
           'ledger append failure: no in-memory bump, client discarded, dirty')
    r16 = await t.train(['b:1:j20', 'b:1:j21'], v_now, u_now + 1)
    t._append = real_append
    ok(r16['update'] == u_now + 1 and t.tc is not stepped and t.tc.optims == 1 and sorted(t.tc.fb) == [2, 2],
       'retry after a ledger failure: fresh client, both groups once, one step')
    # 17. file names never collide
    ok(M.Trainer._safe('a:1') != M.Trainer._safe('a_1'), "job ids 'a:1' and 'a_1' get different files")
    # 12. torn ledger line + restart
    open(t.ledger, 'a').write('{"torn": ')
    u, v = t.update, t.newest()
    t3 = new_trainer()
    ok(t3.update == u and t3.newest() == v, f'restart after a torn ledger line resumes update {u}')
    await stream(t3, 'b:1:j11', v, [G([0.1, 0.2])])
    r6 = await t3.train(['b:1:j11'], v, u + 1)
    t4 = new_trainer()
    ok(r6['update'] == u + 1 and t4.update == u + 1, 'row appended after a torn line is readable after restart')
    # 13. a store refuses a different starting point
    try:
        M.Trainer('circuit', ENV, 'tinker://x/weights/1', BASE_SP, 'HARISH', False, 'unit', 'circuit-inkling-s5-a12')
        ok(False, 'meta mismatch')
    except SystemExit:
        ok(True, 'store refuses a different starting state')

asyncio.run(main())
print(f'{len(FAILED)} failed' if FAILED else 'all passed')
sys.exit(1 if FAILED else 0)
