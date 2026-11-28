"""MOMO-train Inkling shim, STREAMING (user design 2026-11-27): one trainable Inkling adapter on Tinker, gradients streamed
per graded group like the TPU models; an update is just optim + save + publish.

  python3 inkling_momo_train.py serve circuit 18793 --env-txt RUN/env.txt --state STATE --base-sampler SAMPLER \
      --key HARISH --tag <one tag per MOMO run> --adapter-name NAME [--publish]

A STORE (momo_train/<tag>) belongs to ONE MOMO run (meta.json pins its starting point; another start is refused).

Version v = the newest sampler (base before update 1). One Tinker training client holds v's state WITH optimizer.
/sample {job_id, parent_states, sampler_path, api_key, n, ...}
    Children per parent as inkling_sample.py. As soon as a parent's group is graded (discover do_group_rollout; infra-
    failed children dropped), the group is STREAMED if sampler_path == v, streaming is open and the group is not
    constant-reward: persisted to stream/v<N>/<job>__p<i>.pkl, THEN forward_backward is enqueued on the training client
    (accumulates). Persist + enqueue run under one lock in a task that a cancelled /sample cannot interrupt halfway.
    Stale-version groups, groups graded after a cut and constant groups are never streamed (counted).
POST /train {adapter, job_ids, base_sampler_path, expected_update, ...}
    1. Already-applied job set -> its recorded result.
    2. Guards: expected_update / base_sampler_path / job records (unknown -> 404, sampled by another version -> 409).
    3. CUT (streaming closed), wait for every enqueued forward_backward. If anything failed (or the shim restarted
       mid-version) the client is reloaded from v's state and EVERY persisted group of v is re-streamed, so the gradient
       is always exactly the persisted set.
    4. No streamed group -> {skipped: true}, no new version.
    5. optim_step (constant LR from the recipe) -> save state + sampler (unique names) -> ledger row (fsync) ->
       publish (non-fatal) -> v+1 (empty accumulator on the stepped client). A failed step or save discards the client;
       the retry rebuilds, so the same groups are never applied twice.
    Streaming reopens when /train returns (success, skip or error). The job_ids are MOMO's bookkeeping; groups streamed
    from jobs MOMO cancels later are in the gradient on purpose. Reply: {update, sampler_path, state_path,
    base_sampler_path, n_streamed_groups, n_datums, n_stale_groups, n_late_groups, n_constant_groups, seconds,
    published, skipped}.
GET /status -> {update, newest_sampler, streaming, dirty, counts, persisted_groups, ...}
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, os, pickle, re, sys, threading, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_sample as S   # recipe loading, key resolution + published-adapter rotation, env class, capture

STORE_ROOT = Path(os.environ.get('INKLING_MOMO_TRAIN_STORE', '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/momo_train'))


class HTTPError(Exception):
    code = 500


class Conflict(HTTPError):
    """Rollouts from a version that is not the newest, or an unexpected update number (lost reply + resume)."""
    code = 409


class NotFound(HTTPError):
    code = 404


class BadRequest(HTTPError):
    code = 400


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _log(msg):
    print(f'{_now()} {msg}', file=sys.stderr, flush=True)


class Trainer:
    def __init__(self, problem, env_txt, base_state, base_sampler, owner_key, publish, tag, adapter):
        self.problem, self.owner_key, self.publish, self.tag = problem, owner_key, publish, tag
        self.env = S.load_recipe(problem, env_txt)
        self.base_state, self.base_sampler = base_state, base_sampler
        self.store = STORE_ROOT / tag
        (self.store / 'jobs').mkdir(parents=True, exist_ok=True)
        meta = dict(base_state=base_state, base_sampler=base_sampler, adapter=adapter, key=owner_key, env_txt=env_txt)
        mf = self.store / 'meta.json'
        if mf.exists():
            old = json.load(open(mf))
            diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
            if diff:
                raise SystemExit(f'store {self.store} was started with a different setup {diff}; use a new --tag')
        else:
            json.dump(dict(meta, created=_now()), open(mf, 'w'), indent=1)
        self.ledger = self.store / 'updates.jsonl'
        self.tc = None                     # tinker TrainingClient (lazy; resumed from self.state_path)
        self.service_client = None
        self.lock = asyncio.Lock()
        last = self._last_row()
        self.update = last['update'] if last else 0
        self.sampler_path = last['sampler_path'] if last else None
        self.state_path = last['state_path'] if last else base_state
        # streaming state of the current version
        self.stream_open = True
        self.fb_lock = asyncio.Lock()        # serialises the open-check + persist + enqueue of each streamed group
        self.fb_tasks = []                   # consumers of enqueued forward_backwards (they never raise)
        self.stream_tasks = set()            # per-group stream tasks spawned by /sample
        self.late_pending = False            # late groups persisted during a cut (count only if the version continues)
        self.counts = self._fresh_counts()
        persisted = list(self._vdir().glob('*.pkl'))
        self.counts['streamed'] = len(persisted)
        # A restart mid-version: the new process's client holds none of the persisted groups -> rebuild at /train.
        self.dirty = bool(persisted)

    # ---------- persistence ----------
    def _rows(self):
        out = []
        if self.ledger.exists():
            for l in open(self.ledger):
                try:
                    out.append(json.loads(l))
                except ValueError:          # torn last line after a crash
                    continue
        return out

    def _last_row(self):
        rows = [r for r in self._rows() if 'jobs_key' in r and not r.get('skipped')]
        return rows[-1] if rows else None

    def _published_ok(self, path):
        return any(r.get('event') == 'published' and r.get('sampler_path') == path and r.get('ok') for r in self._rows())

    def _append(self, row):
        if self.ledger.exists() and self.ledger.stat().st_size:
            with open(self.ledger, 'rb') as fh:
                fh.seek(-1, os.SEEK_END)
                torn = fh.read(1) != b'\n'
        else:
            torn = False
        with open(self.ledger, 'a') as fh:
            if torn:                        # crash mid-append left a fragment: never glue the next row onto it
                fh.write('\n')
            fh.write(json.dumps(row) + '\n')
            fh.flush(); os.fsync(fh.fileno())

    def newest(self):
        return self.sampler_path or self.base_sampler

    @staticmethod
    def _fresh_counts():
        return dict(streamed=0, datums=0, stale=0, late=0, constant=0, refused=0)

    @staticmethod
    def _safe(job_id):
        # readable + a hash of the exact id, so 'a:1' and 'a_1' never share a file
        return re.sub(r'[^A-Za-z0-9._-]', '_', job_id) + '-' + hashlib.sha1(job_id.encode()).hexdigest()[:8]

    def _job_file(self, job_id):
        return self.store / 'jobs' / (self._safe(job_id) + '.json')

    def _save_job(self, job_id, sampler_path, n_parents):
        f = self._job_file(job_id)
        tmp = f.with_suffix(f'.tmp{os.getpid()}')
        json.dump(dict(sampler_path=sampler_path, n_parents=n_parents, at=_now()), open(tmp, 'w'))
        os.replace(tmp, f)

    def _load_job(self, job_id):
        return json.load(open(self._job_file(job_id)))

    def _vdir(self):
        d = self.store / 'stream' / f'v{self.update:04d}'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _pickle(path, obj):
        tmp = path.with_suffix(f'.tmp{os.getpid()}')
        with open(tmp, 'wb') as fh:
            pickle.dump(obj, fh)
        os.replace(tmp, path)

    @staticmethod
    def _unpickle(path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    # ---------- sampling ----------
    async def sample(self, parent_states, sampler_path, api_key, n, job_id=None):
        from ttt_discover.tinker_utils.misc_utils import get_tokenizer
        from ttt_discover.tinker_utils.renderers import get_renderer
        from ttt_discover.tinker_utils.completers import InklingTwoPhaseTokenCompleter
        from ttt_discover.tinker_utils.state import State
        from ttt_discover.rl.rollouts import do_group_rollout
        from ttt_discover.rl.types import EnvGroupBuilder
        from types import SimpleNamespace
        E = os.environ
        model = E['TTD_ENSEMBLE_MODELS'].split('|' if '|' in E['TTD_ENSEMBLE_MODELS'] else ':')[0]
        tok = get_tokenizer(model)
        renderer = get_renderer('tml_v0', tok)
        Env = S._env_class()
        cfg = SimpleNamespace(convo_prefix=[], timeout=float(E.get('TTD_GRADE_WAIT_TIMEOUT', 172800)),
                              num_cpus_per_task=int(E.get('NUM_CPUS_PER_TASK', 2)),
                              eval_backend=E.get('TTD_EVAL_BACKEND', 'farm'), eval_timeout=int(E.get('EVAL_TIMEOUT', 1100)),
                              log_path=E['SCIENCE_WORKER_ROOT'], problem_type=E.get('TTD_PROBLEM_TYPE', ''),
                              member_tag='inkling')

        class Builder(EnvGroupBuilder):
            def __init__(self, parent):
                self.parent, self.caps = parent, []

            async def make_envs(self):
                envs = []
                for _ in range(n):
                    cap = S._Capture()
                    cap.dropped = False
                    env = Env(renderer, State.from_dict(self.parent), cap, cfg)
                    orig_step = env.step

                    async def step(*a, _orig=orig_step, _cap=cap, **k):
                        try:
                            return await _orig(*a, **k)
                        except Exception as e:  # noqa: BLE001
                            if getattr(e, 'drop_candidate', False):
                                _cap.dropped = True     # do_group_rollout drops it from the group
                            raise
                    env.step = step
                    self.caps.append(cap)
                    envs.append(env)
                return envs

            def logging_tags(self):
                return ['inkling-momo']

        async def one_parent(idx, parent):
            k = S._pick_key(sampler_path, api_key)
            S._inflight[k] = S._inflight.get(k, 0) + n
            try:
                client = await S._sampling_client(sampler_path, k)
                policy = InklingTwoPhaseTokenCompleter(
                    sampling_client=client, tokenizer=tok,
                    phase1_max_tokens=int(E.get('TTD_M0_PHASE1_MAX_TOKENS', 65536)), temperature=1.0,
                    context_window=int(E.get('TTD_M0_CONTEXT_WINDOW', E.get('CONTEXT_WINDOW', 65536))),
                    min_think_tokens=int(E.get('TTD_MIN_THINK_TOKENS', 0)))
                b = Builder(parent)
                group = await do_group_rollout(b, policy, 0)
            finally:
                S._inflight[k] -= n
            if job_id and group is not None:
                self._spawn_stream(group, job_id, idx, sampler_path)   # graded: stream it now
            children = []
            for cap in b.caps:                     # one per child, in sampling order; always n
                st = cap.state
                if cap.dropped:
                    children.append((None, None, None, '[dropped: grading infrastructure]'))
                elif st is not None:
                    children.append((st.code, st.value, st.construction, st.observation))
                else:                              # invalid / unparsable program (graded, no state)
                    children.append((None, None, None, None))
            return group, children

        res = await asyncio.gather(*(one_parent(i, p) for i, p in enumerate(parent_states)))
        if job_id:
            self._save_job(job_id, sampler_path, len(parent_states))
        return [c for _, c in res]

    # ---------- training ----------
    async def _client(self):
        if self.tc is None:
            import tinker
            self.service_client = self.service_client or tinker.ServiceClient(api_key=S._resolve_key(self.owner_key))
            t0 = time.time()
            self.tc = await self.service_client.create_training_client_from_state_with_optimizer_async(self.state_path)
            _log(f'training client resumed from {self.state_path} in {time.time() - t0:.0f}s (update {self.update})')
        return self.tc

    def _loss_fn(self):
        return os.environ.get('TTD_LOSS_FN', 'importance_sampling')

    async def _prep(self, groups):
        """prepare_minibatch (advantages are within-group, so per-group == batched). Pure CPU when KL is 0: run it in a
        worker thread with its own loop so /sample keeps going; with KL > 0 it calls Tinker, so it stays on this loop."""
        from ttt_discover.rl.train import prepare_minibatch
        from ttt_discover.tinker_utils.misc_utils import get_tokenizer
        E = os.environ
        model = E['TTD_ENSEMBLE_MODELS'].split('|' if '|' in E['TTD_ENSEMBLE_MODELS'] else ':')[0]

        class _B:
            def logging_tags(self):
                return ['inkling-momo']
        kl = float(self.env.get('KL_PENALTY_COEF', 0) or 0)
        prep = lambda: prepare_minibatch(  # noqa: E731
            [_B() for _ in groups], groups, get_tokenizer(model), self.service_client, model_name=model,
            kl_penalty_coef=kl, train_step=self.update,
            adv_estimator=E.get('TTD_ADV_ESTIMATOR', 'piecewise_valid_entropic_centered'),
            train_max_seq=int(E.get('TTD_M0_TRAIN_MAX_SEQ', 65536)),
            context_window=int(E.get('TTD_M0_CONTEXT_WINDOW', 65536)))
        if kl == 0:
            return await asyncio.to_thread(lambda: asyncio.run(prep()))
        await self._client()
        return await prep()

    def _spawn_stream(self, group, job_id, idx, sampler_path):
        # A separate task: cancelling /sample (caller gone) can never stop it between persist and enqueue.
        t = asyncio.create_task(self._stream_group(group, job_id, idx, sampler_path))
        self.stream_tasks.add(t)
        t.add_done_callback(self.stream_tasks.discard)

    async def _stream_group(self, group, job_id, idx, sampler_path):
        from ttt_discover.rl.data_processing import all_same
        from ttt_discover.rl.train import enqueue_forward_backward
        persisted = False
        try:
            if sampler_path != self.newest():
                self.counts['stale'] += 1; return
            if all_same(group.get_total_rewards()):
                self.counts['constant'] += 1; return
            if not self.stream_open:
                await self._persist_late(group, job_id, idx, sampler_path); return
            try:
                data_D, _ = await self._prep([group])
            except ValueError as e:
                if type(e) is not ValueError:
                    raise
                self.counts['refused'] += 1; return            # the estimator refused this group: deterministic
            if not data_D:
                self.counts['refused'] += 1; return
            async with self.fb_lock:
                if sampler_path != self.newest():
                    self.counts['stale'] += 1; return           # a new version came first
                if not self.stream_open:                         # the cut came first: keep it for a skip / failure
                    await self._persist_late(group, job_id, idx, sampler_path, locked=True); return
                path = self._vdir() / f'{self._safe(job_id)}__p{idx}.pkl'
                await asyncio.to_thread(self._pickle, path, group)
                persisted = True
                self.counts['streamed'] += 1
                self.counts['datums'] += len(data_D)
                if self.dirty:
                    return                                       # the rebuild at /train re-streams every persisted group
                tc = await self._client()
                fut = await enqueue_forward_backward(tc, data_D, self._loss_fn())
                self.fb_tasks.append(asyncio.create_task(self._consume(fut, path.name)))
        except Exception as e:  # noqa: BLE001
            if persisted:
                self.dirty = True                                # persisted but not (surely) enqueued: rebuild at /train
            _log(f'stream {job_id} p{idx} failed ({"persisted -> rebuild" if persisted else "not persisted"}): '
                 f'{type(e).__name__}: {str(e)[:300]}')

    async def _persist_late(self, group, job_id, idx, sampler_path, locked=False):
        """A group of the current version graded during a cut: persisted but NOT enqueued. If the /train succeeds it is
        discarded with that version's directory; if it skips or fails, the version continues and the next /train's
        rebuild includes it (dirty)."""
        async def _do():
            if sampler_path != self.newest():
                self.counts['stale'] += 1; return
            path = self._vdir() / f'{self._safe(job_id)}__p{idx}.pkl'
            await asyncio.to_thread(self._pickle, path, group)
            self.counts['late'] += 1
            self.late_pending = True
        if locked:
            await _do()
        else:
            async with self.fb_lock:
                await _do()

    async def _consume(self, fut, what):
        from ttt_discover.rl.train import consume_forward_backward
        try:
            await consume_forward_backward(fut)
        except Exception as e:  # noqa: BLE001
            self.dirty = True
            _log(f'forward_backward for {what} failed -> rebuild at /train: {type(e).__name__}: {str(e)[:300]}')

    async def _rebuild(self, files=None):
        """Fresh client from v's saved state (with optimizer) + re-stream the given persisted groups of v (default: all),
        sequentially. /train passes the set persisted BEFORE its cut, so late groups never enter that update."""
        from ttt_discover.rl.train import enqueue_forward_backward, consume_forward_backward
        self.dirty = True
        self.tc = None
        tc = await self._client()
        files = sorted(self._vdir().glob('*.pkl')) if files is None else list(files)
        n_datums = 0
        for f in files:
            group = await asyncio.to_thread(self._unpickle, f)
            data_D, _ = await self._prep([group])
            fut = await enqueue_forward_backward(tc, data_D, self._loss_fn())
            await consume_forward_backward(fut)
            n_datums += len(data_D)
        self.counts['streamed'], self.counts['datums'] = len(files), n_datums
        self.dirty = False
        _log(f'rebuilt v{self.update}: re-streamed {len(files)} persisted groups ({n_datums} datums)')

    async def train(self, job_ids, base_sampler_path=None, expected_update=None):
        from ttt_discover.rl.train import enqueue_optim_step, consume_optim_step
        if not job_ids:
            raise BadRequest('job_ids is empty')
        key = hashlib.sha256(json.dumps(sorted(job_ids)).encode()).hexdigest()[:16]
        async with self.lock:
            # 1. already applied -> recorded result
            for r in self._rows():
                if r.get('jobs_key') == key:
                    if base_sampler_path and r.get('base_sampler_path') not in (None, base_sampler_path):
                        raise Conflict(f'job set already applied from {r.get("base_sampler_path")}, '
                                       f'not {base_sampler_path}')
                    if not r.get('skipped') and self.publish:
                        r['published'] = self._published_ok(r['sampler_path'])
                    return r
            # 2. guards
            newest = self.newest()
            if expected_update is not None:
                try:
                    expected_update = int(expected_update)
                except (TypeError, ValueError):
                    raise BadRequest(f'expected_update must be an integer, got {expected_update!r}')
                if expected_update != self.update + 1:
                    raise Conflict(f'MOMO expects update {expected_update}, but the next update here is '
                                   f'{self.update + 1}')
            if base_sampler_path and base_sampler_path != newest:
                raise Conflict(f'rollouts were sampled from {base_sampler_path}, but the newest version is {newest} '
                               f'(update {self.update}); refusing a double / off-policy update')
            missing = [j for j in job_ids if not self._job_file(j).exists()]
            if missing:
                raise NotFound(f'no job record for job_ids {missing[:5]} (n={len(missing)})')
            stale = {j: r['sampler_path'] for j in job_ids if (r := self._load_job(j))['sampler_path'] != newest}
            if stale:
                raise Conflict(f'{len(stale)} job(s) were sampled from another version than the newest {newest}: '
                               f'{dict(list(stale.items())[:3])}')
            t0 = time.time()
            # 3. cut: no more groups for v; drain what was enqueued; rebuild if anything is off
            async with self.fb_lock:
                self.stream_open = False
                cut_files = sorted(self._vdir().glob('*.pkl'))   # exactly what was streamed (or persisted) before the cut
            try:
                if self.fb_tasks:
                    await asyncio.gather(*self.fb_tasks)          # _consume never raises; failures set dirty
                self.fb_tasks = []
                if self.dirty:
                    await self._rebuild(cut_files)
                c = dict(self.counts)
                row = dict(update=self.update, jobs_key=key, job_ids=list(job_ids), base_sampler_path=newest,
                           n_streamed_groups=c['streamed'], n_datums=c['datums'], n_stale_groups=c['stale'],
                           n_late_groups=c['late'], n_constant_groups=c['constant'], n_refused_groups=c['refused'],
                           n_groups=c['streamed'], n_dropped=c['constant'] + c['refused'],
                           sampler_path=newest, state_path=self.state_path, skipped=False, published=None, at=_now())
                # 4. nothing streamed -> skip (no new version)
                if c['streamed'] == 0:
                    row.update(skipped=True, reason='no streamed group in this version', seconds=round(time.time() - t0, 1))
                    self._append(row)
                    _log(f'/train skipped (nothing streamed in v{self.update}); update stays {self.update}')
                    return row
                # 5. optim + saves; any failure discards the client (the retry rebuilds: never a double step)
                lr = float(self.env['LEARNING_RATE'])   # per-run keys stay out of os.environ (inkling_sample.RUN_ONLY)
                nxt = self.update + 1
                name = f'momo-{self.tag}-{nxt:04d}-{int(time.time())}'   # unique per attempt
                try:
                    tc = await self._client()
                    await consume_optim_step(await enqueue_optim_step(tc, lr))
                    st = await (await tc.save_state_async(name)).result_async()
                    sp = await (await tc.save_weights_for_sampler_async(name)).result_async()
                except BaseException:
                    self.tc = None
                    self.dirty = True
                    _log(f'/train optim or save failed: client discarded; the retry rebuilds v{self.update}')
                    raise
                row.update(update=nxt, sampler_path=sp.path, state_path=st.path, lr=lr, published=False,
                           seconds=round(time.time() - t0, 1))
                try:
                    self._append(row)                             # durable record FIRST ...
                except BaseException:
                    self.tc = None
                    self.dirty = True
                    _log('/train ledger append failed: client discarded; the retry rebuilds and steps again')
                    raise
                self.update, self.state_path, self.sampler_path = nxt, st.path, sp.path   # ... then the bump
                self.counts = self._fresh_counts()                # v+1: empty accumulator on the stepped client
                self.dirty = False
                self.late_pending = False                         # late groups of v are discarded with v's directory
                if self.publish:
                    ok = await self._publish(sp.path)
                    row['published'] = ok
                    self._append(dict(event='published', sampler_path=sp.path, ok=ok, at=_now()))
                _log(f'/train update {nxt}: {c["streamed"]} streamed groups, {c["datums"]} datums '
                     f'(stale {c["stale"]}, late {c["late"]}, constant {c["constant"]}), {row["seconds"]}s -> '
                     f'{sp.path} (published={row["published"]})')
                return row
            finally:
                if self.late_pending:                             # the version continues (skip / failure): include them
                    self.dirty = True
                    self.late_pending = False
                self.stream_open = True

    async def _publish(self, path):
        import tinker
        for attempt in range(3):
            try:
                rc = tinker.ServiceClient(api_key=S._resolve_key(self.owner_key)).create_rest_client()
                await rc.publish_checkpoint_from_tinker_path_async(path)
                _add_published(path)
                return True
            except Exception as e:  # noqa: BLE001
                if '409' in str(e) or 'already' in str(e).lower():   # already public
                    _add_published(path)
                    return True
                _log(f'publish attempt {attempt + 1} failed for {path}: {type(e).__name__}: {str(e)[:200]}')
                await asyncio.sleep(10 * (attempt + 1))
        return False


def _add_published(path):
    f = S.PUBLISHED_FILE
    d = json.load(open(f))
    d['paths'][f'momo-{path.rsplit("/", 1)[-1]}-{path.split("//")[1][:8]}'] = path
    tmp = f + f'.tmp{os.getpid()}'
    json.dump(d, open(tmp, 'w'), indent=1); os.replace(tmp, f)


def serve(args):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import concurrent.futures, select, socket
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    trainer = asyncio.run_coroutine_threadsafe(_make(args), loop).result()

    class H(BaseHTTPRequestHandler):
        def _gone(self):
            try:
                r, _, _ = select.select([self.connection], [], [], 0)
                return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b''
            except OSError:
                return True

        def _reply(self, code, obj):
            out = json.dumps(obj).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            if self.path.startswith('/status'):
                self._reply(200, dict(update=trainer.update, sampler_path=trainer.sampler_path,
                                      newest_sampler=trainer.newest(), publish=trainer.publish,
                                      adapter=args.adapter_name, tag=trainer.tag, state_path=trainer.state_path,
                                      streaming=trainer.stream_open, dirty=trainer.dirty, counts=trainer.counts,
                                      persisted_groups=len(list(trainer._vdir().glob('*.pkl'))),
                                      jobs_stored=len(list((trainer.store / 'jobs').glob('*.json')))))
            else:
                self._reply(404, {'error': 'unknown path'})

        def do_POST(self):
            try:
                try:
                    body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    if self.path.startswith('/train'):
                        if body.get('adapter') not in (None, args.adapter_name):
                            raise BadRequest(f"this shim trains {args.adapter_name!r}, not {body.get('adapter')!r}")
                        coro = trainer.train(list(body['job_ids']), body.get('base_sampler_path'),
                                             body.get('expected_update'))
                    elif self.path.startswith('/sample'):
                        coro = trainer.sample(body['parent_states'], body['sampler_path'], body['api_key'],
                                              int(body.get('n', 32)), body.get('job_id'))
                    else:
                        raise NotFound(f'unknown path {self.path}')
                except (KeyError, TypeError, ValueError) as e:
                    raise BadRequest(f'bad request: {type(e).__name__}: {e}') from e
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                while True:
                    try:
                        res = fut.result(timeout=10)
                        break
                    except concurrent.futures.TimeoutError:
                        if not self.path.startswith('/train') and self._gone():
                            fut.cancel()               # caller gone: free Tinker + grading (a /train always finishes)
                            _log(f'caller gone: cancelled {self.path} {body.get("job_id")}')
                            return
                self._reply(200, res if self.path.startswith('/train') else {'children': res})
            except Exception as e:  # noqa: BLE001
                code = e.code if isinstance(e, HTTPError) else 500
                _log(f'{code} on {self.path}: {type(e).__name__}: {str(e)[:500]}')
                try:
                    self._reply(code, {'error': f'{type(e).__name__}: {e}'})
                except OSError:
                    pass

        def log_message(self, *a):
            pass
    _log(f'serving {args.problem} on 127.0.0.1:{args.port}; tag {args.tag}; adapter {args.adapter_name}; '
         f'resume state {trainer.state_path}; update {trainer.update}; publish={args.publish}')
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


async def _make(args):
    return Trainer(args.problem, args.env_txt, args.state, args.base_sampler, args.key, args.publish, args.tag,
                   args.adapter_name)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['serve'])
    p.add_argument('problem', choices=['circuit', 'erdos'])
    p.add_argument('port', type=int)
    p.add_argument('--env-txt', required=True, help="the trainable adapter's run env.txt (recipe + grading)")
    p.add_argument('--state', required=True, help='training state (weights + optimizer) the run starts from')
    p.add_argument('--base-sampler', required=True, help='sampler path of --state (newest version before update 1)')
    p.add_argument('--key', required=True, help="owner key name of the adapter's org (training + unpublished sampling)")
    p.add_argument('--tag', required=True, help='store name: ONE per MOMO run (smoke and real block differ)')
    p.add_argument('--adapter-name', required=True, help="MOMO's adapter name; /train with another name -> 400")
    p.add_argument('--publish', action='store_true', help='publish each new sampler checkpoint (5-key rotation)')
    serve(p.parse_args())
