"""Run: PYTHONPATH=<discover pin> <tinker venv>/bin/python tests/test_momo_train_offline.py (exit 1 on failure).
Offline tests for inkling_momo_train.Trainer.train with fake Tinker/discover pieces (no network, no grading)."""
import asyncio, json, os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tmp = tempfile.mkdtemp()
os.environ['INKLING_MOMO_TRAIN_STORE'] = tmp
import inkling_momo_train as M
import ttt_discover.rl.train as T

ENV = '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs/mc-s5/env.txt'
BASE_S, BASE_SP = 'tinker://base:train:0/weights/000012', 'tinker://base:train:0/sampler_weights/000012'


class G:
    def __init__(self, r): self.r = r
    def get_total_rewards(self): return self.r


class Fut:
    def __init__(self, v, fail=False): self.v, self.fail = v, fail
    async def result_async(self):
        if self.fail: raise RuntimeError('save failed (injected)')
        return types.SimpleNamespace(path=self.v)


class TC:
    def __init__(self, fail_save=False): self.steps, self.fail_save = 0, fail_save
    async def save_state_async(self, name): return Fut(f'tinker://new:train:0/weights/{name}', self.fail_save)
    async def save_weights_for_sampler_async(self, name): return Fut(f'tinker://new:train:0/sampler_weights/{name}')


FAKE = {'tc': None, 'prep_raise': None, 'clients_made': 0}


async def fake_client(self):
    if self.tc is None:
        FAKE['clients_made'] += 1
        self.tc = FAKE['tc']
    return self.tc


async def fake_prep(builders, groups, *a, **k):
    if FAKE['prep_raise']: raise FAKE['prep_raise']
    return [object() for g in groups for _ in g.r], {'advantage/mean': 0.0}


async def fake_step(data, tc, lr, n, loss):
    tc.steps += 1


M.Trainer._client = fake_client
T.prepare_minibatch, T.train_step = fake_prep, fake_step


FAILED = []


async def main():
    def ok(c, msg):
        print(('PASS ' if c else 'FAIL ') + msg)
        if not c:
            FAILED.append(msg)
    t = M.Trainer('circuit', ENV, BASE_S, BASE_SP, 'HARISH', False, 'unit', 'circuit-inkling-s5-a12')
    t._save_job('b:1:j1', [G([0.1, 0.5, 0.0])], BASE_SP)
    t._save_job('b:1:j2', [G([0.2, 0.2])], BASE_SP)            # constant group
    # 404
    try:
        await t.train(['b:1:nope']); ok(False, '404 missing job')
    except M.NotFound: ok(True, '404 on unknown job id')
    # save failure -> 500-class error, client discarded, update unchanged, no ledger row
    FAKE['tc'] = TC(fail_save=True)
    try:
        await t.train(['b:1:j1'], BASE_SP, 1); ok(False, 'save failure raises')
    except RuntimeError: ok(t.tc is None and t.update == 0 and not t._rows(), 'save failure: client discarded, update 0, no row')
    # retry succeeds with a NEW client (reloaded), exactly one step on it
    FAKE['tc'] = TC()
    r = await t.train(['b:1:j1'], BASE_SP, 1)
    ok(r['update'] == 1 and not r['skipped'] and FAKE['tc'].steps == 1 and FAKE['clients_made'] == 2,
       f'retry after failure: update 1, one step on a reloaded client ({FAKE["clients_made"]} clients)')
    # idempotent repeat
    r2 = await t.train(['b:1:j1'], BASE_SP, 1)
    ok(r2['update'] == 1 and FAKE['tc'].steps == 1, 'idempotent repeat: no second step')
    # idempotent with a different base -> 409
    try:
        await t.train(['b:1:j1'], 'tinker://other'); ok(False, 'idem base mismatch')
    except M.Conflict: ok(True, '409 on applied set with a different base_sampler_path')
    # stale job (sampled from the old version) -> 409
    t._save_job('b:1:j3', [G([0.1, 0.9])], BASE_SP)
    try:
        await t.train(['b:1:j3'], r['sampler_path'], 2); ok(False, 'stale job')
    except M.Conflict: ok(True, '409 on a job sampled from an older version')
    # expected_update mismatch -> 409
    t._save_job('b:1:j4', [G([0.1, 0.9])], r['sampler_path'])
    try:
        await t.train(['b:1:j4'], r['sampler_path'], 5); ok(False, 'expected_update')
    except M.Conflict: ok(True, '409 on expected_update mismatch')
    # all-constant -> skipped, update unchanged, recorded
    t._save_job('b:1:j5', [G([0.3, 0.3])], r['sampler_path'])
    r5 = await t.train(['b:1:j5'], r['sampler_path'], 2)
    ok(r5['skipped'] and t.update == 1, f'skipped on constant rewards ({r5.get("reason")})')
    # estimator ValueError -> skipped
    FAKE['prep_raise'] = ValueError('adaptive PWC requires ...')
    r6 = await t.train(['b:1:j4'], r['sampler_path'], 2)
    ok(r6['skipped'] and t.update == 1, f'skipped on estimator ValueError ({r6.get("reason")})')
    FAKE['prep_raise'] = None
    # torn ledger line tolerated + restart resumes the last row
    open(t.ledger, 'a').write('{"torn": ')
    t2 = M.Trainer('circuit', ENV, BASE_S, BASE_SP, 'HARISH', False, 'unit', 'circuit-inkling-s5-a12')
    ok(t2.update == 1 and t2.sampler_path == r['sampler_path'], 'restart after torn line resumes update 1')
    # a row appended after the torn fragment must survive (starts on a new line)
    FAKE['tc'] = TC()
    t2._save_job('b:1:j7', [G([0.1, 0.8])], r['sampler_path'])
    r7 = await t2.train(['b:1:j7'], r['sampler_path'], 2)
    t3 = M.Trainer('circuit', ENV, BASE_S, BASE_SP, 'HARISH', False, 'unit', 'circuit-inkling-s5-a12')
    ok(r7['update'] == 2 and t3.update == 2, 'row appended after a torn line is readable after restart (update 2)')
    # checkpoint names are unique per attempt
    ok(r7['sampler_path'] != r['sampler_path'] and r7['sampler_path'].rsplit('-', 1)[-1].isdigit(), 'unique checkpoint name')
    # a ValueError SUBCLASS (e.g. JSONDecodeError) is not a permanent skip
    t3._save_job('b:1:j8', [G([0.1, 0.7])], r7['sampler_path'])
    FAKE['prep_raise'] = json.JSONDecodeError('transient', 'x', 0)
    try:
        await t3.train(['b:1:j8'], r7['sampler_path'], 3); ok(False, 'subclass ValueError raised')
    except json.JSONDecodeError: ok(not any(x.get('jobs_key') and 'b:1:j8' in x.get('job_ids', []) for x in t3._rows()),
                                   'ValueError subclass is retried, not recorded as skipped')
    FAKE['prep_raise'] = None
    # non-int expected_update -> 400
    try:
        await t3.train(['b:1:j8'], r7['sampler_path'], 'abc'); ok(False, 'non-int expected_update')
    except M.BadRequest: ok(True, '400 on a non-integer expected_update')
    # a skip row does not change the version reported after a restart
    t3._save_job('b:1:j9', [G([0.4, 0.4])], r7['sampler_path'])
    await t3.train(['b:1:j9'], r7['sampler_path'], 3)
    t4 = M.Trainer('circuit', ENV, BASE_S, BASE_SP, 'HARISH', False, 'unit', 'circuit-inkling-s5-a12')
    ok(t4.update == 2 and t4.sampler_path == r7['sampler_path'], 'restart after a skip keeps update 2 and its sampler')
    # /train's CPU work (unpickling + prepare_minibatch) runs off the event loop: a concurrent coroutine (standing in for
    # /sample) keeps ticking while a /train with a 1.5 s BLOCKING prepare step runs
    import threading, time as _time
    main_thread = threading.get_ident()
    seen = {}

    async def slow_prep(builders, groups, *a, **k):
        seen['thread'] = threading.get_ident()
        _time.sleep(1.5)                       # blocking CPU stand-in
        return [object() for g in groups for _ in g.r], {'advantage/mean': 0.0}
    T.prepare_minibatch = slow_prep
    FAKE['tc'] = TC()
    t4.tc = None
    t4._save_job('b:1:j10', [G([0.1, 0.6])], r7['sampler_path'])
    ticks = []

    async def ticker():
        for _ in range(15):
            ticks.append(_time.time()); await asyncio.sleep(0.1)
    tk = asyncio.create_task(ticker())
    r10 = await t4.train(['b:1:j10'], r7['sampler_path'], 3)
    await tk
    T.prepare_minibatch = fake_prep
    gaps = [b_ - a_ for a_, b_ in zip(ticks, ticks[1:])]
    ok(r10['update'] == 3 and seen.get('thread') != main_thread and max(gaps) < 0.6,
       f'prepare_minibatch off the event loop (max tick gap {max(gaps):.2f}s while a 1.5 s prep ran)')
    # store with a different start refuses
    try:
        M.Trainer('circuit', ENV, 'tinker://x/weights/1', BASE_SP, 'HARISH', False, 'unit', 'circuit-inkling-s5-a12')
        ok(False, 'meta mismatch')
    except SystemExit: ok(True, 'store refuses a different starting state')

asyncio.run(main())
print(f'{len(FAILED)} failed' if FAILED else 'all passed')
sys.exit(1 if FAILED else 0)
