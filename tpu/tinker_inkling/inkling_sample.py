"""Per-job Inkling-2-Mini expansion for the MOMO runner (user-approved MOMO plan, 2026-11-26).

    await inkling_sample(parent_states, sampler_path, api_key, n) -> list[list[(code, value_or_None, construction, observation)]]

For every parent state (a PUCT state dict as in puct_sampler_step_*.json, from any model's tree) sample n children from the
frozen Inkling adapter at `sampler_path` with EXACTLY the live runs' recipe, extract the answer-only code, and GRADE each
child with the same grader/transport the runs use. Children come back per parent, in order; `value` follows the PUCT tree
convention (higher = better: Erdős -C5, circuit reward), None for an invalid child, and `construction` / `observation` are
what the env would store in a new tree state. Nothing is written to any tree.

Recipe and grading come from a live run's env.txt (load_recipe once per process, before the first call):
    load_recipe('erdos')    -> mn9-s1/env.txt   (tml_v0 effort 0.8, fixed 22k thinking + 10k answer, 600-line rule, ...)
    load_recipe('circuit')  -> mc-s1/env.txt    (same Inkling recipe on the ibm8 placement task, 70-farm circuit registry)
One process serves ONE problem (the recipe lives in os.environ). Grading goes through the SSH forwards on della-vis1, so run
this on della-vis1 -- serve() exposes it over HTTP for a runner elsewhere:  python3 inkling_sample.py serve erdos 8790
[ENV_TXT] [KEY=VALUE ...] (another run's env.txt and/or setting overrides, e.g. a different grading registry)
api_key may be the raw key or a key NAME ('1', '2', '3', 'TAI', 'HARISH') resolved from ~/.bashrc (preferred over HTTP).
"""
from __future__ import annotations

import asyncio, json, os, subprocess, sys, tempfile
from types import SimpleNamespace

RUNS = '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs'
RECIPE_RUN = {'erdos': 'mn9-s1', 'circuit': 'mc-s1'}
# per-run keys in env.txt that must not leak into a sampler process
RUN_ONLY = ('SKYRL_RUN_ID', 'TTD_RUN_DIR', 'EXPERIMENT_NAME', 'SKYRL_GRADING_JOURNAL', 'SKYRL_GRADING_EVENTS', 'TTD_M0_LORA_SEED',
            'TTD_INIT_STATE_SEED', 'NUM_EPOCHS', 'LEARNING_RATE', 'KL_PENALTY_COEF', 'CODE_SKYRL', 'CODE_DISCOVER',
            'TTD_SICK_MARKER', 'SCIENCE_WORKER_ROOT', 'TTD_FROZEN_POLICY', 'TTD_FROZEN_SAMPLER_PATH')
_problem = None
_clients: dict = {}


def _paths(env):
    for p in (env.get('CODE_DISCOVER'), f"{env.get('CODE_DISCOVER')}/examples/erdos_min_overlap",
              env.get('CODE_SKYRL'), f"{env.get('CODE_SKYRL')}/tpu"):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def load_recipe(problem: str, env_txt: str | None = None, overrides: dict | None = None) -> dict:
    """Set os.environ to a live run's recipe + grading settings (and its pinned code on sys.path).
    overrides: KEY=VALUE settings applied on top (e.g. a different grading registry for MOMO)."""
    global _problem
    env_txt = env_txt or f"{RUNS}/{RECIPE_RUN[problem]}/env.txt"
    env = dict(l.rstrip('\n').split('=', 1) for l in open(env_txt) if '=' in l)
    env.update(overrides or {})
    _paths(env)
    for k, v in env.items():
        if k not in RUN_ONLY and not k.startswith('TINKER_API_KEY'):
            os.environ[k] = v
    os.environ['SCIENCE_WORKER_ROOT'] = tempfile.mkdtemp(prefix='inkling-sample-')
    if problem == 'circuit':
        os.environ.setdefault('SKYRL_GRADING_URL', 'http://127.0.0.1:9')
        # env.txt only records TTD_/SKYRL_/... keys; the circuit task's SCIENCE_PLACEMENT_* settings are launch_run.sh's
        # TTD_RECIPE=circuit-fix block, copied here.
        os.environ.update(SCIENCE_PLACEMENT_SUITE='ibm8-proxy-v1', SCIENCE_PLACEMENT_HELPER='fast_proxy_v1',
                          SCIENCE_PLACEMENT_STARTS='abuplace-xplace-three-starts-v1', SCIENCE_PLACEMENT_BACKEND='cpu',
                          SCIENCE_PLACEMENT_RUNTIME='cpu300-4g-v1', SCIENCE_PLACEMENT_SLOTS_PER_HOST='1',
                          SCIENCE_PLACEMENT_CPUS_PER_CASE='1', SCIENCE_PLACEMENT_MEMORY_GIB='4')
    _problem = problem
    return env


def _resolve_key(api_key: str) -> str:
    if api_key in ('1', '2', '3', 'TAI', 'HARISH'):
        var = 'TINKER_API_KEY' if api_key == '1' else f'TINKER_API_KEY_{api_key}'
        return subprocess.run(['bash', '-ic', f'printf %s "${var}"'], capture_output=True, text=True).stdout
    return api_key


class _Capture:
    """Stands in for the PUCT sampler: keeps the child state the env would add to the tree."""
    def __init__(self):
        self.state = None

    def update_states(self, states, parents, save=False):
        self.state = states[0]

    def record_failed_rollout(self, *_a, **_k):
        pass


def _env_class():
    if _problem == 'erdos':
        from env import ErdosMinOverlapEnv
        return ErdosMinOverlapEnv
    from tpu.science.training_env import PlacementTrainingEnv
    return PlacementTrainingEnv


PUBLISHED_FILE = os.environ.get('INKLING_PUBLISHED_ADAPTERS',
                                '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops/published_adapters.json')
_inflight: dict = {}   # key name -> samples currently being generated under it


def _rotation():
    """(published sampler paths, key names). Published checkpoints are readable by any key (user OK 2026-11-27)."""
    try:
        d = json.load(open(PUBLISHED_FILE))
        return set(d['paths'].values()), list(d['keys'])
    except (OSError, ValueError, KeyError):
        return set(), []


def _pick_key(sampler_path: str, requested: str) -> str:
    """Spread published adapters over all keys: the key with the fewest samples in flight (ties -> the requested key).
    Unpublished adapters stay on the requested (owner) key -- other keys get 404 for them."""
    published, keys = _rotation()
    if sampler_path not in published or requested not in keys:
        return requested
    return min(keys, key=lambda k: (_inflight.get(k, 0), k != requested))


async def _sampling_client(sampler_path: str, api_key: str):
    import tinker
    key = (sampler_path, api_key)
    if key not in _clients:
        sc = tinker.ServiceClient(api_key=_resolve_key(api_key))
        _clients[key] = await sc.create_sampling_client_async(model_path=sampler_path)
    return _clients[key]


async def inkling_sample(parent_states: list[dict], sampler_path: str, api_key: str, n: int) -> list[list[tuple]]:
    if _problem is None:
        raise RuntimeError('call load_recipe("erdos" | "circuit") first')
    from ttt_discover.tinker_utils.misc_utils import get_tokenizer
    from ttt_discover.tinker_utils.renderers import get_renderer
    from ttt_discover.tinker_utils.completers import InklingTwoPhaseTokenCompleter
    from ttt_discover.tinker_utils.state import State
    E = os.environ
    model = E['TTD_ENSEMBLE_MODELS'].split('|' if '|' in E['TTD_ENSEMBLE_MODELS'] else ':')[0]
    tok = get_tokenizer(model)
    renderer = get_renderer('tml_v0', tok)
    async def completer_for(key_name):
        client = await _sampling_client(sampler_path, key_name)
        return InklingTwoPhaseTokenCompleter(
            sampling_client=client, tokenizer=tok, phase1_max_tokens=int(E.get('TTD_M0_PHASE1_MAX_TOKENS', 65536)),
            temperature=1.0, context_window=int(E.get('TTD_M0_CONTEXT_WINDOW', E.get('CONTEXT_WINDOW', 65536))),
            min_think_tokens=int(E.get('TTD_MIN_THINK_TOKENS', 0)))
    Env = _env_class()
    cfg = SimpleNamespace(convo_prefix=[], timeout=float(E.get('TTD_GRADE_WAIT_TIMEOUT', 172800)),
                          num_cpus_per_task=int(E.get('NUM_CPUS_PER_TASK', 2)), eval_backend=E.get('TTD_EVAL_BACKEND', 'farm'),
                          eval_timeout=int(E.get('EVAL_TIMEOUT', 1100)), log_path=E['SCIENCE_WORKER_ROOT'],
                          problem_type=E.get('TTD_PROBLEM_TYPE', ''), member_tag='inkling')

    async def one_parent(parent: dict):
        state = State.from_dict(parent)
        probe = Env(renderer, state, _Capture(), cfg)
        ob, stop = await probe.initial_observation()
        k = _pick_key(sampler_path, api_key)      # per parent: spreads one call's 8 parents over the keys
        _inflight[k] = _inflight.get(k, 0) + n
        try:
            seqs = await (await completer_for(k)).sample_group(ob, stop, n)
        finally:
            _inflight[k] -= n

        async def grade(seq):
            cap = _Capture()
            env = Env(renderer, State.from_dict(parent), cap, cfg)
            await env.step(seq.tokens, 0)
            s = cap.state
            if s is None:                        # invalid / unparsable / failed grade
                return (None, None, None, None)
            return (s.code, s.value, s.construction, s.observation)
        return list(await asyncio.gather(*(grade(s) for s in seqs)))

    return list(await asyncio.gather(*(one_parent(p) for p in parent_states)))


def serve(problem: str, port: int, env_txt: str | None = None, overrides: dict | None = None):
    """Tiny HTTP shim on 127.0.0.1:port: POST /sample {parent_states, sampler_path, api_key, n} -> {children: [...]}."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    load_recipe(problem, env_txt, overrides)
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                fut = asyncio.run_coroutine_threadsafe(
                    inkling_sample(body['parent_states'], body['sampler_path'], body['api_key'], int(body.get('n', 32))),
                    loop)
                out = json.dumps({'children': fut.result()}).encode()
                self.send_response(200)
            except Exception as e:  # noqa: BLE001
                out = json.dumps({'error': f'{type(e).__name__}: {e}'}).encode()
                self.send_response(500)
            self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(out)

        def log_message(self, *a):
            pass
    ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()


if __name__ == '__main__':
    if sys.argv[1:2] == ['serve']:
        # serve PROBLEM PORT [ENV_TXT|-] [KEY=VALUE ...]
        rest = sys.argv[4:]
        env_txt = rest[0] if rest and '=' not in rest[0] and rest[0] != '-' else None
        ov = dict(a.split('=', 1) for a in rest if '=' in a)
        serve(sys.argv[2], int(sys.argv[3]), env_txt, ov)
