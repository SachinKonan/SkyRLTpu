"""MOMO-train Inkling shim (user-approved 2026-11-27): sampling + RL updates of ONE trainable Inkling adapter on Tinker.

A separate process from the frozen MOMO shims (inkling_sample.py serve on 18790/18791, which stay untouched):

  python3 inkling_momo_train.py serve circuit 18793 --env-txt RUN/env.txt --state STATE_PATH --key HARISH [--publish]

POST /sample {job_id, parent_states, sampler_path, api_key, n}
    Same request/response as inkling_sample.py (children per parent: (code, value, construction, observation)), but each
    parent group is a real discover TrajectoryGroup (do_group_rollout: tokens, sampling logprobs, rewards, correctness),
    and with a job_id the job's groups are persisted to STORE/jobs/<job_id>.pkl for a later /train. A child whose grading
    infrastructure gave up is DROPPED from its group (discover's drop_failed_candidates, as in the trained runs) and
    comes back as (None, None, None, '[dropped: grading infrastructure]') instead of failing the whole call.
POST /train {adapter, job_ids}
    One RL step on the trainable adapter with the run's own recipe (env.txt: TTD_ADV_ESTIMATOR, TTD_LOSS_FN,
    LEARNING_RATE, KL 0, TTD_REJECT_TRUNCATED): constant-reward groups dropped -> prepare_minibatch (advantages,
    datums, OOV/length filters) -> train_step (forward_backward + optim_step) -> save full state AND sampler weights
    (every update, so a restart resumes exactly) -> optional publish -> {sampler_path, state_path, n_groups, n_datums,
    n_dropped, seconds, update}. Idempotent: the same job_ids return the recorded result. The training client is resumed
    from the newest state in STORE/updates.jsonl, else from --state (e.g. mc-s5 weights/000012).
GET /status -> {update, sampler_path, state_path, jobs_stored}
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, os, pickle, re, sys, threading, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_sample as S   # recipe loading, key resolution + published-adapter rotation, env class, capture

STORE_ROOT = Path(os.environ.get('INKLING_MOMO_TRAIN_STORE', '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/momo_train'))


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _log(msg):
    print(f'{_now()} {msg}', file=sys.stderr, flush=True)


class Trainer:
    def __init__(self, problem, env_txt, base_state, owner_key, publish, tag):
        self.problem, self.owner_key, self.publish = problem, owner_key, publish
        self.env = S.load_recipe(problem, env_txt)
        self.base_state = base_state
        self.store = STORE_ROOT / tag
        (self.store / 'jobs').mkdir(parents=True, exist_ok=True)
        self.ledger = self.store / 'updates.jsonl'
        self.tc = None                     # tinker TrainingClient (lazy, resumed on first /train)
        self.lock = asyncio.Lock()
        last = self._last_update()
        self.update = last['update'] if last else 0
        self.sampler_path = last['sampler_path'] if last else None
        self.state_path = last['state_path'] if last else base_state

    # ---------- persistence ----------
    def _last_update(self):
        if not self.ledger.exists():
            return None
        rows = [json.loads(l) for l in open(self.ledger) if l.strip()]
        return rows[-1] if rows else None

    def _job_file(self, job_id):
        return self.store / 'jobs' / (re.sub(r'[^A-Za-z0-9._-]', '_', job_id) + '.pkl')

    def _save_job(self, job_id, groups):
        f = self._job_file(job_id)
        tmp = f.with_suffix(f'.tmp{os.getpid()}')
        with open(tmp, 'wb') as fh:
            pickle.dump(groups, fh)
        os.replace(tmp, f)

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
            self._save_job(job_id, [g for g, _ in res if g is not None])
        return [c for _, c in res]

    # ---------- training ----------
    async def _client(self):
        if self.tc is None:
            import tinker
            sc = tinker.ServiceClient(api_key=S._resolve_key(self.owner_key))
            self.service_client = sc
            t0 = time.time()
            self.tc = await sc.create_training_client_from_state_with_optimizer_async(self.state_path)
            _log(f'training client resumed from {self.state_path} in {time.time() - t0:.0f}s (update {self.update})')
        return self.tc

    async def train(self, job_ids):
        from ttt_discover.rl.train import prepare_minibatch, train_step
        from ttt_discover.rl.data_processing import all_same
        key = hashlib.sha256(json.dumps(sorted(job_ids)).encode()).hexdigest()[:16]
        async with self.lock:
            if self.ledger.exists():                     # idempotent: same job set -> recorded result
                for l in open(self.ledger):
                    r = json.loads(l)
                    if r.get('jobs_key') == key:
                        return r
            missing = [j for j in job_ids if not self._job_file(j).exists()]
            if missing:
                raise KeyError(f'no stored groups for job_ids {missing[:5]} (n={len(missing)})')
            groups = []
            for j in job_ids:
                with open(self._job_file(j), 'rb') as fh:
                    groups += pickle.load(fh)
            t0 = time.time()
            kept = [g for g in groups if not all_same(g.get_total_rewards())]
            row = dict(update=self.update, jobs_key=key, job_ids=list(job_ids), n_groups=len(kept),
                       n_dropped=len(groups) - len(kept), n_datums=0, sampler_path=self.sampler_path,
                       state_path=self.state_path, at=_now())
            if kept:
                E = os.environ
                tc = await self._client()
                from ttt_discover.tinker_utils.misc_utils import get_tokenizer
                model = E['TTD_ENSEMBLE_MODELS'].split('|' if '|' in E['TTD_ENSEMBLE_MODELS'] else ':')[0]

                class _B:
                    def logging_tags(self):
                        return ['inkling-momo']
                data_D, m = await prepare_minibatch(
                    [_B() for _ in kept], kept, get_tokenizer(model), self.service_client, model_name=model,
                    kl_penalty_coef=float(self.env.get('KL_PENALTY_COEF', 0) or 0), train_step=self.update,
                    adv_estimator=E.get('TTD_ADV_ESTIMATOR', 'piecewise_valid_entropic_centered'),
                    train_max_seq=int(E.get('TTD_M0_TRAIN_MAX_SEQ', 65536)),
                    context_window=int(E.get('TTD_M0_CONTEXT_WINDOW', 65536)))
                lr = float(self.env['LEARNING_RATE'])   # per-run keys stay out of os.environ (inkling_sample.RUN_ONLY)
                await train_step(data_D, tc, lr, 1, E.get('TTD_LOSS_FN', 'importance_sampling'))
                self.update += 1
                name = f'momo-{self.update:04d}'
                st = await (await tc.save_state_async(name)).result_async()
                sp = await (await tc.save_weights_for_sampler_async(name)).result_async()
                self.state_path, self.sampler_path = st.path, sp.path
                if self.publish:
                    import tinker
                    rc = tinker.ServiceClient(api_key=S._resolve_key(self.owner_key)).create_rest_client()
                    await rc.publish_checkpoint_from_tinker_path_async(sp.path)
                    _add_published(sp.path)
                row.update(update=self.update, n_datums=len(data_D), sampler_path=sp.path, state_path=st.path,
                           lr=lr, advantage_mean=m.get('advantage/mean'), published=self.publish)
            row['seconds'] = round(time.time() - t0, 1)
            with open(self.ledger, 'a') as fh:
                fh.write(json.dumps(row) + '\n')
                fh.flush(); os.fsync(fh.fileno())
            _log(f'/train update {row["update"]}: {row["n_groups"]} groups, {row["n_datums"]} datums, '
                 f'{row["seconds"]}s -> {row["sampler_path"]}')
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
                                      state_path=trainer.state_path,
                                      jobs_stored=len(list((trainer.store / 'jobs').glob('*.pkl')))))
            else:
                self._reply(404, {'error': 'unknown path'})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path.startswith('/train'):
                    # One trainable adapter per process: refuse a /train meant for another adapter.
                    if args.adapter_name and body.get('adapter') not in (None, args.adapter_name):
                        raise ValueError(f"this shim trains {args.adapter_name!r}, not {body.get('adapter')!r}")
                    coro = trainer.train(list(body['job_ids']))
                else:
                    coro = trainer.sample(body['parent_states'], body['sampler_path'], body['api_key'],
                                          int(body.get('n', 32)), body.get('job_id'))
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
                _log(f'500 on {self.path}: {type(e).__name__}: {str(e)[:500]}')
                try:
                    self._reply(500, {'error': f'{type(e).__name__}: {e}'})
                except OSError:
                    pass

        def log_message(self, *a):
            pass
    _log(f'serving {args.problem} on 127.0.0.1:{args.port}; trainable state {trainer.state_path}; '
         f'update {trainer.update}; publish={args.publish}; store {trainer.store}')
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


async def _make(args):
    return Trainer(args.problem, args.env_txt, args.state, args.key, args.publish, args.tag)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['serve'])
    p.add_argument('problem', choices=['circuit', 'erdos'])
    p.add_argument('port', type=int)
    p.add_argument('--env-txt', required=True, help="the trainable adapter's run env.txt (recipe + grading)")
    p.add_argument('--state', required=True, help='training state (weights + optimizer) to start from')
    p.add_argument('--key', required=True, help="owner key name of the adapter's org (training + unpublished sampling)")
    p.add_argument('--tag', default=None, help='store sub-directory (default: problem + state run id)')
    p.add_argument('--publish', action='store_true', help='publish each new sampler checkpoint (5-key rotation)')
    p.add_argument('--adapter-name', default=None, help="MOMO's name for the trainable adapter; /train with another name is refused")
    a = p.parse_args()
    a.tag = a.tag or f'{a.problem}-{a.state.split("//")[1][:8]}'
    serve(a)
