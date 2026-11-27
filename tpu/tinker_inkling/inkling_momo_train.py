"""MOMO-train Inkling shim (user-approved 2026-11-27): sampling + RL updates of ONE trainable Inkling adapter on Tinker.

A separate process from the frozen MOMO shims (inkling_sample.py serve on 18790/18791, which stay untouched):

  python3 inkling_momo_train.py serve circuit 18793 --env-txt RUN/env.txt --state STATE --base-sampler SAMPLER \
      --key HARISH --tag <one tag per MOMO run> --adapter-name NAME [--publish]

A STORE (momo_train/<tag>) belongs to ONE MOMO run: MOMO's job ids restart in every run, so a smoke and the real block
must use different tags. The store records its starting point (meta.json) and refuses to start with another one.

POST /sample {job_id, parent_states, sampler_path, api_key, n, ...}
    Children per parent (code, value, construction, observation), as inkling_sample.py. Each parent group is a discover
    TrajectoryGroup (do_group_rollout: tokens, sampling logprobs, rewards, correctness); with a job_id the job's groups
    AND the sampler_path that produced them are stored (STORE/jobs/<job_id>.pkl). A child whose grading infrastructure
    gave up is dropped from its group (as in the trained runs): (None, None, None, '[dropped: grading infrastructure]').
POST /train {adapter, job_ids, base_sampler_path, expected_update, ...}
    1. Already-applied job set -> the recorded result (if base_sampler_path is given it must match that row's).
    2. Guards -> 409: expected_update != next update; base_sampler_path != newest version; any job sampled from a version
       other than the newest. Unknown job ids -> 404; wrong adapter / bad body -> 400.
    3. One RL step with the run's own recipe (TTD_ADV_ESTIMATOR, TTD_LOSS_FN, LEARNING_RATE, KL; constant-reward groups
       dropped). Nothing to train (no non-constant group, 0 datums, estimator ValueError) -> {skipped: true}: no step,
       no new version, update unchanged (recorded, so a retry returns the same).
    4. train_step, then save full state + sampler weights. If the step or a save fails, the stepped training client is
       discarded and the next /train reloads the last saved state: a retry can never apply the same groups twice.
    5. Ledger row written (fsync) BEFORE publishing; publish (if --publish) is retried and never fails the request.
    Reply: {update, sampler_path, state_path, base_sampler_path, n_groups, n_datums, n_dropped, seconds, published,
    skipped}.
GET /status -> {update, newest_sampler, publish, adapter, tag, jobs_stored, ...}
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
        rows = [r for r in self._rows() if 'jobs_key' in r]
        return rows[-1] if rows else None

    def _append(self, row):
        with open(self.ledger, 'a') as fh:
            fh.write(json.dumps(row) + '\n')
            fh.flush(); os.fsync(fh.fileno())

    def newest(self):
        return self.sampler_path or self.base_sampler

    def _job_file(self, job_id):
        return self.store / 'jobs' / (re.sub(r'[^A-Za-z0-9._-]', '_', job_id) + '.pkl')

    def _save_job(self, job_id, groups, sampler_path):
        f = self._job_file(job_id)
        tmp = f.with_suffix(f'.tmp{os.getpid()}')
        with open(tmp, 'wb') as fh:
            pickle.dump(dict(sampler_path=sampler_path, groups=groups, at=_now()), fh)
        os.replace(tmp, f)

    def _load_job(self, job_id):
        with open(self._job_file(job_id), 'rb') as fh:
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

        async def one_parent(parent):
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

        res = await asyncio.gather(*(one_parent(p) for p in parent_states))
        if job_id:
            self._save_job(job_id, [g for g, _ in res if g is not None], sampler_path)
        return [c for _, c in res]

    # ---------- training ----------
    async def _client(self):
        if self.tc is None:
            import tinker
            self.service_client = tinker.ServiceClient(api_key=S._resolve_key(self.owner_key))
            t0 = time.time()
            self.tc = await self.service_client.create_training_client_from_state_with_optimizer_async(self.state_path)
            _log(f'training client resumed from {self.state_path} in {time.time() - t0:.0f}s (update {self.update})')
        return self.tc

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

    async def train(self, job_ids, base_sampler_path=None, expected_update=None):
        from ttt_discover.rl.train import prepare_minibatch, train_step
        from ttt_discover.rl.data_processing import all_same
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
                    return r
            # 2. guards
            newest = self.newest()
            if expected_update is not None and int(expected_update) != self.update + 1:
                raise Conflict(f'MOMO expects update {expected_update}, but the next update here is {self.update + 1}')
            if base_sampler_path and base_sampler_path != newest:
                raise Conflict(f'rollouts were sampled from {base_sampler_path}, but the newest version is {newest} '
                               f'(update {self.update}); refusing a double / off-policy update')
            missing = [j for j in job_ids if not self._job_file(j).exists()]
            if missing:
                raise NotFound(f'no stored groups for job_ids {missing[:5]} (n={len(missing)})')
            jobs = {j: self._load_job(j) for j in job_ids}
            stale = {j: d['sampler_path'] for j, d in jobs.items() if d['sampler_path'] != newest}
            if stale:
                raise Conflict(f'{len(stale)} job(s) were sampled from another version than the newest {newest}: '
                               f'{dict(list(stale.items())[:3])}')
            groups = [g for d in jobs.values() for g in d['groups']]
            t0 = time.time()
            kept = [g for g in groups if not all_same(g.get_total_rewards())]
            row = dict(update=self.update, jobs_key=key, job_ids=list(job_ids), base_sampler_path=newest,
                       n_groups=len(kept), n_dropped=len(groups) - len(kept), n_datums=0, sampler_path=newest,
                       state_path=self.state_path, skipped=False, published=None, at=_now())
            # 3. data
            data_D, m, why = [], {}, None
            if not kept:
                why = 'no non-constant-reward group'
            else:
                E = os.environ
                from ttt_discover.tinker_utils.misc_utils import get_tokenizer
                model = E['TTD_ENSEMBLE_MODELS'].split('|' if '|' in E['TTD_ENSEMBLE_MODELS'] else ':')[0]

                class _B:
                    def logging_tags(self):
                        return ['inkling-momo']
                await self._client()
                try:
                    data_D, m = await prepare_minibatch(
                        [_B() for _ in kept], kept, get_tokenizer(model), self.service_client, model_name=model,
                        kl_penalty_coef=float(self.env.get('KL_PENALTY_COEF', 0) or 0), train_step=self.update,
                        adv_estimator=E.get('TTD_ADV_ESTIMATOR', 'piecewise_valid_entropic_centered'),
                        train_max_seq=int(E.get('TTD_M0_TRAIN_MAX_SEQ', 65536)),
                        context_window=int(E.get('TTD_M0_CONTEXT_WINDOW', 65536)))
                except ValueError as e:                      # estimator refused this data: deterministic
                    why = f'estimator: {e}'
                if not why and not data_D:
                    why = '0 datums after filtering'
            if why:
                row.update(skipped=True, reason=why, seconds=round(time.time() - t0, 1))
                self._append(row)
                _log(f'/train skipped ({why}) for {len(job_ids)} job(s); update stays {self.update}')
                return row
            # 4. step + saves; on ANY failure discard the stepped client (next /train reloads the last saved state)
            lr = float(self.env['LEARNING_RATE'])   # per-run keys stay out of os.environ (inkling_sample.RUN_ONLY)
            nxt = self.update + 1
            name = f'momo-{self.tag}-{nxt:04d}'
            try:
                await train_step(data_D, self.tc, lr, 1, os.environ.get('TTD_LOSS_FN', 'importance_sampling'))
                st = await (await self.tc.save_state_async(name)).result_async()
                sp = await (await self.tc.save_weights_for_sampler_async(name)).result_async()
            except BaseException:
                self.tc = None
                _log(f'/train step or save failed: training client discarded; next /train reloads {self.state_path}')
                raise
            self.update, self.state_path, self.sampler_path = nxt, st.path, sp.path
            row.update(update=nxt, n_datums=len(data_D), sampler_path=sp.path, state_path=st.path, lr=lr,
                       advantage_mean=m.get('advantage/mean'), published=False,
                       seconds=round(time.time() - t0, 1))
            # 5. ledger first, then publish (non-fatal)
            self._append(row)
            if self.publish:
                ok = await self._publish(sp.path)
                row['published'] = ok
                self._append(dict(event='published', sampler_path=sp.path, ok=ok, at=_now()))
            _log(f'/train update {nxt}: {row["n_groups"]} groups, {row["n_datums"]} datums, {row["seconds"]}s -> '
                 f'{sp.path} (published={row["published"]})')
            return row


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
                                      jobs_stored=len(list((trainer.store / 'jobs').glob('*.pkl')))))
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
                    if isinstance(e, HTTPError):
                        raise
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
