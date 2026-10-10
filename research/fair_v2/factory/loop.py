"""Proposer loop for the feature factory: an LLM writes candidate feature families, grade.py scores them.

  python factory/loop.py --run <name> --generator replay --rounds 2 --n 6            # offline test
  python factory/loop.py --run <name> --generator completions --url http://HOST:PORT --rounds 20 --n 32
  python factory/loop.py --run <name> --check-prompts                                # render, count tokens

One round: N prompts (improve a parent picked from the graded pool, or explore a new idea) -> generate ->
extract the code -> static checks -> screen grade -> full grade for the best screens -> history. Generation of
round r+1 starts once round r's screens are in, so it overlaps round r's full grades.

All state is in <run>/history.jsonl (append-only events). Restarting with the same --run resumes: finished
grades are read from their result.json, unfinished ones are resubmitted, an interrupted round is regenerated.
The proposer sees only proposer_view.json content (no hidden-shift numbers, no acceptance flag).
"""
import argparse
import ast
import glob
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAIR_V2 = HERE.parent
DATA = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/factory')
LOOPS = DATA / 'loops'
SBATCH = HERE / 'run_grade.sbatch'
GH_PYTHON = '/scratch/gpfs/ZHUANGL/sk7524/fair-env-gh/bin/python'
QWEN_TOKENIZER = sorted(glob.glob('/scratch/gpfs/ZHUANGL/sk7524/rq4-local/hf/hub/models--Qwen--Qwen3.5-27B/'
                                  'snapshots/*'))[-1:]
# graded examples that seed the pool: (id, source, full result dir, screen result dir)
SEEDS = [('seed_dirpower', 'examples/dirpower.py', 'gh_dirpower_full', 'gh_dirpower_screen'),
         ('seed_psband', 'examples/psband.py', 'psband_full', 'psband_screen'),
         ('seed_dog', 'examples/dog.py', 'dog_full', 'dog_screen')]
EXAMPLE = 'seed_dirpower'
# acceptance (recorded only; the stack is never edited here): MOMO_HILLCLIMB_ENV.md section 3 + README
ACCEPT_REWARD = 0.006


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    if RUN is not None:
        with open(RUN / 'loop.log', 'a') as fh:
            fh.write(line + '\n')


RUN = None


# =========================================================================================== generators
class Generator:
    """generate(prompts, n) -> len(prompts) * n texts; texts[i*n:(i+1)*n] answer prompts[i].

    Each text is the model's full assistant turn, thinking included (Qwen: '<think>...</think> answer').
    self.meta[j] holds per-output metadata (tokens, forced thinking stop, errors)."""
    meta = []

    def generate(self, prompts, n=1):
        raise NotImplementedError


def _wrap(code, note='replay'):
    return f'<think>\n{note}\n</think>\n\nHere is the candidate.\n\n```python\n{code.rstrip()}\n```\n'


class ReplayGenerator(Generator):
    """Offline test generator: the example candidates, small mutations of them, and malformed answers."""

    def __init__(self):
        ex = {p.stem: p.read_text() for p in (HERE / 'examples').glob('*.py')}
        dp, ps = ex['dirpower'], ex['psband']
        dp1 = dp.replace('np.geomspace(0.006, 0.7, 13)', 'np.geomspace(0.006, 0.7, 9)').replace(
            'For 12 log |k| bands', 'For 8 log |k| bands')
        dp2 = dp.replace('ang < np.pi / 12, ang > 5 * np.pi / 12', 'ang < np.pi / 8, ang > 3 * np.pi / 8')
        ps1 = ps.replace('np.geomspace(0.004, 0.72, 29)', 'np.geomspace(0.004, 0.72, 15)')
        bad_os = 'import os\n' + ps1.replace('def features(maps, mask):',
                                             'def features(maps, mask):\n    os.getcwd()')
        assert dp1 != dp and dp2 != dp and ps1 != ps
        self.pool = [
            ('dirpower_bands8', _wrap(dp1)), ('psband_bands14', _wrap(ps1)), ('random_feats', _wrap(ex['random_feats'])),
            ('bad_os', _wrap(bad_os)), ('no_code', '<think>\nhmm\n</think>\n\nI could not decide.\n'),
            ('dup_dirpower', _wrap(dp)),
            ('dirpower_wedge8', _wrap(dp2)), ('batch_violator', _wrap(ex['batch_violator'])),
            ('unclosed_think', '<think>\nthinking forever ' + 'x ' * 50),
            ('dup_psband_bands14', _wrap(ps1)),
        ]
        self.i = 0

    def generate(self, prompts, n=1):
        out, self.meta = [], []
        for _ in range(len(prompts) * n):
            name, text = self.pool[self.i % len(self.pool)]
            self.i += 1
            out.append(text)
            self.meta.append(dict(replay=name))
        return out


def validate_native_choice(choice, budget, max_tokens):
    """Copy of ttt_discover.rl.ensemble._validate_native_choice (SkyRL-Main-infer): the farm's native
    thinking-budget audit must be present and consistent."""
    tokens = choice.get('token_ids')
    mask = choice.get('loss_mask')
    audit = choice.get('thinking_budget') or {}
    positions = choice.get('forced_token_positions')
    lps = (choice.get('logprobs') or {}).get('token_logprobs')
    if not isinstance(tokens, list) or len(tokens) > max_tokens:
        raise ValueError('native completion has invalid token IDs/length')
    if not isinstance(mask, list) or len(mask) != len(tokens) or any(type(x) not in (int, float) or x not in (0, 1)
                                                                     for x in mask):
        raise ValueError('native completion has missing/invalid loss mask')
    if positions != [i for i, x in enumerate(mask) if x == 0]:
        raise ValueError('native completion forced positions disagree with mask')
    if audit.get('enforced') is not True or audit.get('budget_basis') != 'phase1_generated_tokens':
        raise ValueError('native completion missing enforcement audit')
    counted = audit.get('counted_phase1_tokens')
    if type(counted) is not int or not 0 <= counted <= budget or bool(positions) != audit.get('forced'):
        raise ValueError('native completion exceeded budget or has inconsistent audit')
    if not isinstance(lps, list) or len(lps) != len(tokens) or any(x is None or not math.isfinite(x) for x in lps):
        raise ValueError('native completion missing finite behavior logprobs')


class CompletionsGenerator(Generator):
    """Qwen3.5 through a farm / ingress /v1/completions endpoint, as the RQ4 no-train client samples it.

    Request = DirectBaseSamplingClient.request (SkyRL-Main-infer third_party/discover/ttt_discover/rl/
    ensemble.py): token-id prompt, logprobs, return_token_ids, native thinking_token_budget, stop <|im_end|>.
    Budgets = QwenTwoPhaseTokenCompleter._native_group with the Qwen3.5 RQ4 profile values: context window
    22528, phase-1 (prompt + thinking) 16384, context buffer 50 -> thinking_token_budget = 16384 - prompt,
    max_tokens = 22528 - prompt - 50. Sampling = tinker.SamplingParams defaults used by the campaigns:
    temperature 1.0, top_p 1.0, top_k -1. Prompt = discover's Qwen3Renderer with no system message:
    '<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n'."""

    RETRY = (409, 429)

    def __init__(self, url, model='Qwen/Qwen3.5-27B', tokenizer=None, context_window=22528,
                 phase1_max_tokens=16384, context_buffer=50, temperature=1.0, top_p=1.0, top_k=-1,
                 concurrency=64, timeout=6 * 3600.0):
        from transformers import AutoTokenizer
        self.url, self.model = url.rstrip('/'), model
        self.tok = AutoTokenizer.from_pretrained(tokenizer or QWEN_TOKENIZER[0])
        self.ctx, self.phase1, self.buffer = context_window, phase1_max_tokens, context_buffer
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.concurrency, self.timeout = concurrency, timeout
        self.stop_ids = [self.tok.convert_tokens_to_ids('<|im_end|>')]

    @staticmethod
    def render(question):
        return f'<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n'

    def encode(self, question):
        return list(self.tok.encode(self.render(question), add_special_tokens=False))

    def payload(self, ids, n):
        cap = self.phase1 - len(ids)
        total = self.ctx - len(ids) - self.buffer
        if cap <= 0 or total <= cap:
            raise ValueError(f'prompt of {len(ids)} tokens leaves no thinking/answer room')
        return {'model': self.model, 'prompt': ids, 'n': n, 'max_tokens': total, 'temperature': self.temperature,
                'top_p': self.top_p, 'top_k': self.top_k, 'logprobs': True, 'stream': False,
                'return_token_ids': True, 'thinking_token_budget': cap, 'stop_token_ids': self.stop_ids}

    def generate(self, prompts, n=1):
        import asyncio
        import httpx

        async def one(http, sem, q):
            ids = self.encode(q)
            pl = self.payload(ids, n)
            delay, t0 = 1.0, time.time()
            async with sem:
                while True:
                    try:
                        r = await http.post(self.url + '/v1/completions', json=pl)
                        if r.status_code in self.RETRY or r.status_code >= 500:
                            raise httpx.HTTPStatusError(f'retryable status {r.status_code}: {r.text[:200]}',
                                                        request=r.request, response=r)
                        r.raise_for_status()
                        res = r.json()
                        break
                    except (httpx.TransportError, httpx.HTTPStatusError) as e:
                        st = getattr(getattr(e, 'response', None), 'status_code', None)
                        if st is not None and st not in self.RETRY and st < 500:
                            return [(None, dict(error=f'HTTP {st}: {e}', prompt_tokens=len(ids)))] * n
                        if time.time() - t0 > self.timeout:
                            return [(None, dict(error=f'timeout: {e}', prompt_tokens=len(ids)))] * n
                        await asyncio.sleep(delay)
                        delay = min(30.0, delay * 2)
            outs = []
            for ch in res['choices']:
                meta = dict(prompt_tokens=len(ids), finish_reason=ch.get('finish_reason'),
                            thinking_budget=pl['thinking_token_budget'], served_by=ch.get('served_by'))
                try:
                    validate_native_choice(ch, pl['thinking_token_budget'], pl['max_tokens'])
                    toks = ch['token_ids']
                    meta.update(tokens=len(toks), forced=bool((ch.get('thinking_budget') or {}).get('forced')))
                    outs.append(('<think>\n' + self.tok.decode(toks, skip_special_tokens=False), meta))
                except (ValueError, KeyError, TypeError) as e:
                    outs.append((None, dict(meta, error=f'invalid choice: {e}')))
            return outs

        async def main():
            sem = asyncio.Semaphore(self.concurrency)
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=60.0)) as http:
                return await asyncio.gather(*[one(http, sem, q) for q in prompts])

        res = asyncio.run(main())
        flat = [x for group in res for x in group]
        self.meta = [m for _, m in flat]
        return [t or '' for t, _ in flat]


# ===================================================================================== code handling
FENCE = re.compile(r'```(?:python|py)[ \t]*\n(.*?)```', re.S)
OPEN_FENCE = re.compile(r'```(?:python|py)[ \t]*\n(.*)$', re.S)


def extract_code(text):
    """Last complete ```python block after the model's last </think> (as discover's answer_only_code for
    Qwen). An unterminated final block (answer cut by max_tokens) is accepted only if it parses."""
    if '</think>' not in text:
        return None, 'no answer: thinking never closed'
    ans = text.rsplit('</think>', 1)[-1]
    blocks = FENCE.findall(ans)
    if blocks:
        return blocks[-1].strip() + '\n', None
    m = OPEN_FENCE.search(ans)
    if m:
        code = m.group(1).strip() + '\n'
        try:
            ast.parse(code)
            return code, None
        except SyntaxError:
            return None, 'unterminated code block that does not parse'
    return None, 'no python code block in the answer'


ALLOWED_IMPORTS = {'numpy', 'torch', 'math', 'functools', 'itertools', 'collections', 'typing'}
BAD_NAMES = {'open', 'eval', 'exec', 'compile', '__import__', 'globals', 'locals', 'vars', 'getattr', 'setattr',
             'delattr', 'input', 'breakpoint', 'os', 'sys', 'subprocess', 'socket', 'pathlib', 'importlib',
             'shutil', 'pickle', 'builtins', '__builtins__', '__file__', '__loader__', '__spec__'}
BAD_ATTRS = {'load', 'save', 'savez', 'savez_compressed', 'loadtxt', 'savetxt', 'genfromtxt', 'fromfile', 'tofile',
             'memmap', 'from_file', 'system', 'popen', 'hub', 'cuda_memory_snapshot'}
SOURCE_DENY = [(re.compile(p), why) for p, why in (
    (r'\bopen\s*\(', 'open('), (r'\bnp\s*\.\s*load\b|\bnumpy\s*\.\s*load\b', 'np.load'),
    (r'\btorch\s*\.\s*load\b', 'torch.load'), (r'\bos\s*\.', 'os.'), (r'\bsubprocess\b', 'subprocess'),
    (r'\bsocket\b', 'socket'), (r'\bpathlib\b', 'pathlib'), (r'\bsys\b', 'sys'), (r'__import__', '__import__'),
    (r'\bimportlib\b', 'importlib'), (r'\b(?:eval|exec)\s*\(', 'eval/exec'), (r'data\.py', 'data.py'))]


def _docstrings(tree):
    ids = set()
    for node in [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef,
                                                                      ast.AsyncFunctionDef))]:
        body = getattr(node, 'body', [])
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            ids.add(id(body[0].value))
    return ids


def static_check(code):
    """(ok, reason). The grader runs candidates in-process, so this is the sandbox: whitelisted imports, no
    file / OS / network / dynamic-code access, no reference to the test file or the shift generators."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f'syntax error: {e}'
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'features']
    if not fns:
        return False, 'no top-level def features(maps, mask)'
    if len(fns[0].args.args) < 2:
        return False, 'features must take (maps, mask)'
    docs = _docstrings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split('.')[0] not in ALLOWED_IMPORTS:
                    return False, f'import {a.name} not allowed'
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or '').split('.')[0] not in ALLOWED_IMPORTS:
                return False, f'from {node.module} import not allowed'
        elif isinstance(node, ast.Name) and (node.id in BAD_NAMES or re.match(r'(?i)(.*_)?test', node.id)):
            return False, f'name {node.id} not allowed'
        elif isinstance(node, ast.Attribute) and (node.attr in BAD_ATTRS or node.attr.startswith('__')
                                                  or re.match(r'(?i)(.*_)?test', node.attr)):
            return False, f'attribute .{node.attr} not allowed'
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            s = node.value.lower()
            if any(t in s for t in ('test', 'data', '.np', '/', '\\', '.pt', '.pkl')):
                return False, f'string literal {node.value[:40]!r} not allowed'
    stripped = '\n'.join(line.split('#', 1)[0] for line in code.splitlines())
    for rx, why in SOURCE_DENY:
        if rx.search(stripped):
            return False, f'{why} not allowed'
    return True, None


def code_key(code):
    """Dedup key: the AST without docstrings (comments and formatting do not matter)."""
    tree = ast.parse(code)
    docs = _docstrings(tree)

    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            return None if id(getattr(node, 'value', None)) in docs else node
    tree = Strip().visit(tree)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:20]


def idea_of(code, limit=240):
    try:
        doc = ast.get_docstring(ast.parse(code)) or ''
    except SyntaxError:
        doc = ''
    doc = ' '.join(doc.split('\n\n')[0].split())
    return (doc[:limit] + ('...' if len(doc) > limit else '')) or '(no docstring)'


# ========================================================================================== diagnostics
PARAMS = ('Omega_m', 'S8', 'T_AGN', 'f0', 'Delta_z')


def diag(view):
    """Proposer-facing text for one proposer_view.json (never reads result.json's hidden field)."""
    mode = view.get('mode', '?')
    if not view.get('ok'):
        return f'- {mode}: FAILED. {(view.get("contract") or {}).get("fail", "grader error")}'
    v, c, t, ct = view['visible'], view['config'], view['tail'], view['contract']
    psg = sorted(v['per_shift_gain'].items(), key=lambda kv: -kv[1])
    helps = ', '.join(f'{s} {g:+.4f}' for s, g in psg[:4] if g > 0) or 'none'
    hurts = ', '.join(f'{s} {g:+.4f}' for s, g in psg[::-1][:3] if g < 0) or 'none'
    sa = v.get('standalone_visible_mean', {})
    probe = [sum(f['calib_mse_per_param'][i] for f in v['theta_probe'].values()) / len(v['theta_probe'])
             for i in range(5)]
    lines = [
        f'- {mode} reward {view["reward"]:+.5f}; per-fold gain '
        + ' '.join(f'{x:+.4f}' for x in v['per_fold_gain'])
        + f'; best config: misfit weight {c["misfit_weight"]}, vote {c["vote"]}',
        f'- visible-shift metric of the detector: {v["base_mean"]:.4f} -> {v["cand_mean"]:.4f}',
        f'- per-shift gains, best: {helps}; worst: {hurts}',
    ]
    parts = []
    if 'misfit' in sa:
        parts.append(f'misfit alone {sa["misfit"]:.4f}')
    if 'consRplus' in sa:
        parts.append(f'disagreement term {sa["consRplus"]:.4f} without this family'
                     + ''.join(f', {sa[k]:.4f} with its {k.split("+")[1]} vote' for k in sa if k.startswith('consRplus+')))
    if parts:
        lines.append('- standalone metrics: ' + '; '.join(parts))
    lines += [
        '- theta-probe MSE (1 = no information): ' + ', '.join(f'{p} {x:.2f}' for p, x in zip(PARAMS, probe)),
        '- tail exceedance at the 1% / 0.5% / 0.1% normal thresholds: '
        + '/'.join(f'{x:.2f}' for x in t['exc_base_pct']) + ' (detector) -> '
        + '/'.join(f'{x:.2f}' for x in t['exc_cand_pct']) + f' (with this family); penalty {t["penalty"]:.4f}',
        f'- {ct["dim"]} features, {ct["sec_per_10k"]} s per 10k maps',
    ]
    return '\n'.join(lines)


def short(view):
    if not view.get('ok'):
        return f'{view.get("mode")} FAILED: {(view.get("contract") or {}).get("fail", "grader error")}'
    v, c = view['visible'], view['config']
    psg = sorted(v['per_shift_gain'].items(), key=lambda kv: -kv[1])
    helps = ', '.join(f'{s} {g:+.3f}' for s, g in psg[:3] if g > 0) or 'none'
    return (f'{view["mode"]} reward {view["reward"]:+.4f} (folds ' + ' '.join(f'{x:+.4f}' for x in v['per_fold_gain'])
            + f'), misfit w {c["misfit_weight"]}, vote {c["vote"]}, helps {helps}')


# =============================================================================================== state
class State:
    """Event-sourced loop state. Events are appended to history.jsonl and applied in memory."""

    def __init__(self, run):
        self.path = run / 'history.jsonl'
        self.cands, self.rounds, self.pending, self.attempts = {}, {}, {}, {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.apply(json.loads(line))

    def emit(self, ev, **kw):
        rec = dict(t=round(time.time(), 1), ev=ev, **kw)
        with open(self.path, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        self.apply(rec)

    def apply(self, r):
        ev = r['ev']
        if ev == 'cand':
            self.cands[r['id']] = dict(r, results={}, accepted=False)
            if r.get('round') is not None and r['round'] >= 0:
                ids = self.rounds.setdefault(r['round'], dict(ids=[], generated=False, promoted=None))['ids']
                if r['id'] not in ids:            # a round regenerated after a crash re-emits its ids
                    ids.append(r['id'])
        elif ev == 'round_generated':
            self.rounds.setdefault(r['round'], dict(ids=[], generated=False, promoted=None))['generated'] = True
        elif ev == 'submit':
            self.pending[(r['id'], r['mode'])] = r
        elif ev == 'result':
            self.cands[r['id']]['results'][r['mode']] = r
            self.pending.pop((r['id'], r['mode']), None)
        elif ev == 'error':
            self.pending.pop((r['id'], r['mode']), None)
            k = (r['id'], r['mode'])
            self.attempts[k] = self.attempts.get(k, 0) + 1
        elif ev == 'promote':
            self.rounds[r['round']]['promoted'] = r['ids']
        elif ev == 'accept':
            self.cands[r['id']]['accepted'] = True


# ============================================================================================ grading
def grade_dir(cid, mode):
    return RUN / 'grades' / f'{cid}_{mode}'


def read_view(cid, mode, st):
    res = st.cands[cid]['results'].get(mode)
    if res is None:
        return None
    return json.loads(Path(res['view']).read_text())


def summarize(result_path, view_path):
    r = json.loads(Path(result_path).read_text())
    s = dict(ok=bool(r.get('ok')), reward=r.get('reward'), view=str(view_path), result=str(result_path))
    if r.get('ok'):
        s.update(folds=r['visible']['per_fold_gain'], config=r['config'], dim=r['contract']['dim'],
                 sec10k=r['contract']['sec_per_10k'], hidden_ok=r['hidden']['hidden_ok'],
                 hidden_gain=r['hidden']['mean_gain'], all19=[r['hidden']['all19_base'], r['hidden']['all19_cand']],
                 total_s=r['runtime']['total_s'])
    else:
        s['fail'] = (r.get('contract') or {}).get('fail')
    return s


class Slurm:
    def __init__(self, args):
        self.max_jobs, self.dry = args.slurm_jobs, args.dry_grade
        self.batch = dict(screen=args.batch_screen, full=args.batch_full)
        self.name = f'floop-{args.run}'[:40]
        self._live, self._t = set(), 0.0

    def live(self):
        if time.time() - self._t > 15:
            try:
                out = subprocess.run(['squeue', '-h', '--me', '-o', '%i'], capture_output=True, text=True,
                                     timeout=60).stdout
                self._live = set(out.split())
                self._t = time.time()
            except subprocess.SubprocessError as e:
                log(f'squeue failed: {e}')
        return self._live

    def submit(self, items):
        specs = [f'{RUN / "cands" / (cid + ".py")}:{mode}:{grade_dir(cid, mode)}' for cid, mode in items]
        if self.dry:
            return f'dry{int(time.time() * 1000) % 10**9}'
        out = subprocess.run(['sbatch', '--parsable', '-J', self.name, str(SBATCH)] + specs, capture_output=True,
                             text=True, timeout=120)
        if out.returncode != 0:
            raise RuntimeError(f'sbatch failed: {out.stderr.strip()[:300]}')
        jid = out.stdout.strip().split(';')[0]
        self._live.add(jid)
        return jid


class GH:
    """della-gh (GH200): one ssh process per grade. The node kills processes after ~10 min; a screen takes
    3-60 s and a full grade 109-207 s there. Its Grace CPU is slow for Python-loop candidates, so the runtime
    budget is relaxed to 150 s per 10k maps here (an A100 grade enforces the real 60 s)."""

    def __init__(self, args):
        self.workers = args.gh_workers
        self.procs = {}

    def free(self):
        self.procs = {k: p for k, p in self.procs.items() if p.poll() is None}
        return self.workers - len(self.procs)

    def submit(self, cid, mode):
        out = grade_dir(cid, mode)
        out.mkdir(parents=True, exist_ok=True)
        cmd = (f'cd {FAIR_V2} && timeout 540 {GH_PYTHON} factory/grade.py {RUN / "cands" / (cid + ".py")} '
               f'--mode {mode} --out {out} --budget 150')
        p = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'della-gh', cmd], stdout=open(out / 'gh.log', 'w'),
                             stderr=subprocess.STDOUT)
        self.procs[(cid, mode)] = p
        return f'gh{p.pid}'

    def running(self, key):
        p = self.procs.get(key)
        return p is not None and p.poll() is None


# =============================================================================================== loop
class Loop:
    def __init__(self, args):
        self.a = args
        self.st = State(RUN)
        self.slurm, self.gh = Slurm(args), GH(args)
        self.rng = random.Random(args.seed)
        self.problem = (HERE / 'prompt' / 'problem.md').read_text()
        self.t_improve = (HERE / 'prompt' / 'task_improve.md').read_text()
        self.t_explore = (HERE / 'prompt' / 'task_explore.md').read_text()
        for d in ('cands', 'prompts', 'raw', 'grades'):
            (RUN / d).mkdir(parents=True, exist_ok=True)
        if not any(c.get('kind') == 'seed' for c in self.st.cands.values()):
            self.import_seeds()
        ex = self.st.cands[EXAMPLE]
        self.problem = self.problem.replace('{EXAMPLE_CODE}', Path(ex['code']).read_text().rstrip()).replace(
            '{EXAMPLE_DIAG}', diag(read_view(EXAMPLE, 'full', self.st)))
        self.gen = None
        self.future = None
        self.pool = ThreadPoolExecutor(1)

    # ------------------------------------------------------------------------------------------ seeds
    def import_seeds(self):
        runs = DATA / 'runs'
        for cid, src, full, screen in SEEDS:
            code = (HERE / src).read_text()
            path = RUN / 'cands' / f'{cid}.py'
            path.write_text(code)
            self.st.emit('cand', id=cid, round=-1, kind='seed', parent=None, code=str(path), status='ok',
                         key=code_key(code), idea=idea_of(code))
            for mode, d in (('screen', screen), ('full', full)):
                self.ingest(cid, mode, runs / d / 'result.json', runs / d / 'proposer_view.json')

    # ------------------------------------------------------------------------------------- generation
    def values(self):
        """Parent value: full reward; screen-only candidates count with a discount (the screen ranks poorly:
        dirpower screen -0.0004 vs full +0.0086, random features screen +0.0008)."""
        val = {}
        for cid, c in self.st.cands.items():
            f, s = c['results'].get('full'), c['results'].get('screen')
            if f and f['ok']:
                val[cid] = f['reward']
            elif s and s['ok'] and not c['results'].get('full'):
                val[cid] = s['reward'] - 0.005
        return val

    def plan(self, n):
        val = self.values()
        full_pool = [c for c in val if self.st.cands[c]['results'].get('full', {}).get('ok')]
        pool = full_pool if len(full_pool) >= 3 else list(val)
        ranked = sorted(pool, key=lambda c: -val[c])
        rank = {c: 1 - i / max(len(ranked) - 1, 1) for i, c in enumerate(ranked)}
        kids = {c: sum(1 for x in self.st.cands.values() if x.get('parent') == c) for c in pool}
        total = sum(kids.values())
        plans = []
        for _ in range(n):
            if not pool or self.rng.random() < self.a.explore:
                plans.append(('explore', None))
                continue
            score = {c: rank[c] + self.a.puct_c * math.sqrt(1 + total) / (1 + kids[c]) + 1e-6 * self.rng.random()
                     for c in pool}
            p = max(pool, key=score.get)
            kids[p] += 1
            total += 1
            plans.append(('improve', p))
        return plans

    def leaderboard_text(self, exclude=None, k=8):
        val = self.values()
        ids = [c for c in sorted(val, key=lambda c: -val[c]) if c != exclude][:k]
        lines = []
        for i, cid in enumerate(ids, 1):
            c = self.st.cands[cid]
            mode = 'full' if c['results'].get('full', {}).get('ok') else 'screen'
            lines.append(f'{i}. {c["idea"]}\n   {short(read_view(cid, mode, self.st))}')
        fails = [c for c in self.st.cands.values() if c.get('status') == 'ok'
                 and any(not r['ok'] for r in c['results'].values())][-3:]
        if fails:
            lines.append('Recent contract failures (avoid these mistakes):')
            for c in fails:
                r = next(r for r in c['results'].values() if not r['ok'])
                lines.append(f'- {c["idea"][:120]} -> {r.get("fail")}')
        return '\n'.join(lines) or '(none yet)'

    def render(self, kind, parent):
        if kind == 'explore':
            task = self.t_explore.replace('{LEADERBOARD}', self.leaderboard_text(k=10))
        else:
            c = self.st.cands[parent]
            mode = 'full' if c['results'].get('full', {}).get('ok') else 'screen'
            has_screen = mode == 'full' and c['results'].get('screen', {}).get('ok')
            if parent == EXAMPLE:
                sec = 'The worked example above (its code and full diagnostics are shown there).'
                if has_screen:
                    sec += ' Its screen result:\n' + diag(read_view(parent, 'screen', self.st))
            else:
                dg = diag(read_view(parent, mode, self.st))
                if has_screen:
                    dg += '\n' + diag(read_view(parent, 'screen', self.st))
                sec = f'```python\n{Path(c["code"]).read_text().rstrip()}\n```\n\nIts grader diagnostics:\n{dg}'
            task = (self.t_improve.replace('{PARENT}', sec)
                    .replace('{LEADERBOARD}', self.leaderboard_text(exclude=parent)))
        return self.problem.rstrip() + '\n\n' + task

    def generator(self):
        if self.gen is None:
            if self.a.generator == 'replay':
                self.gen = ReplayGenerator()
                self.gen.i = sum(1 for c in self.st.cands.values() if c.get('kind') != 'seed')
            else:
                self.gen = CompletionsGenerator(self.a.url, self.a.model, self.a.tokenizer,
                                                temperature=self.a.temperature, top_p=self.a.top_p,
                                                top_k=self.a.top_k, concurrency=self.a.concurrency)
        return self.gen

    def start_round(self, r):
        n_prompts = math.ceil(self.a.n / self.a.samples)
        plans = self.plan(n_prompts)
        prompts = [self.render(k, p) for k, p in plans]
        log(f'round {r}: generating {self.a.n} candidates ({sum(k == "explore" for k, _ in plans)} explore prompts, '
            f'parents {sorted({p for _, p in plans if p})}; prompt chars {min(map(len, prompts))}-'
            f'{max(map(len, prompts))})')
        gen = self.generator()

        def work():
            texts = gen.generate(prompts, self.a.samples)
            return texts, list(gen.meta)
        self.future = (r, plans, prompts, self.pool.submit(work), time.time())

    def finish_round(self):
        r, plans, prompts, fut, t0 = self.future
        self.future = None
        try:
            texts, meta = fut.result()
        except Exception as e:  # noqa: BLE001 - a failed generation call is retried next tick
            log(f'round {r}: generation failed ({type(e).__name__}: {e}); retrying')
            return
        keys = {c['key']: cid for cid, c in self.st.cands.items() if c.get('key')}
        s = self.a.samples
        for j, text in enumerate(texts[:self.a.n]):
            kind, parent = plans[j // s]
            cid = f'r{r:03d}c{j:02d}'
            (RUN / 'prompts' / f'{cid}.txt').write_text(prompts[j // s])
            (RUN / 'raw' / f'{cid}.txt').write_text(text)
            code, why = extract_code(text)
            ev = dict(id=cid, round=r, kind=kind, parent=parent, prompt=str(RUN / 'prompts' / f'{cid}.txt'),
                      raw=str(RUN / 'raw' / f'{cid}.txt'), gen=meta[j] if j < len(meta) else {}, code=None)
            if code is None:
                ev.update(status='no_code', reason=why)
            else:
                path = RUN / 'cands' / f'{cid}.py'
                path.write_text(code)
                ok, why = static_check(code)
                ev.update(code=str(path), idea=idea_of(code))
                if not ok:
                    ev.update(status='static_fail', reason=why)
                else:
                    key = code_key(code)
                    ev['key'] = key
                    if key in keys:
                        ev.update(status='duplicate', reason=f'same code as {keys[key]}')
                    else:
                        ev.update(status='ok')
                        keys[key] = cid
            self.st.emit('cand', **ev)
        self.st.emit('round_generated', round=r, seconds=round(time.time() - t0, 1))
        stats = {}
        for cid in self.st.rounds[r]['ids']:
            stats[self.st.cands[cid]['status']] = stats.get(self.st.cands[cid]['status'], 0) + 1
        log(f'round {r}: generated in {time.time() - t0:.0f}s: {stats}')

    # ------------------------------------------------------------------------------------------ grading
    def ingest(self, cid, mode, result, view):
        s = summarize(result, view)
        self.st.emit('result', id=cid, mode=mode, **s)
        c = self.st.cands[cid]
        if mode == 'full' and s['ok'] and s['reward'] > ACCEPT_REWARD and min(s['folds']) > 0 and s['hidden_ok']:
            self.st.emit('accept', id=cid, reward=s['reward'])
            log(f'ACCEPTED {cid} ({c.get("idea", "")[:80]}): reward {s["reward"]:+.5f} folds {s["folds"]}')
        if c.get('kind') != 'seed':
            log(f'{cid} {mode}: ' + (f'reward {s["reward"]:+.5f} folds {s.get("folds")}' if s['ok']
                                     else f'FAILED {s.get("fail")}'))

    def needed(self):
        items = []
        for r, rd in sorted(self.st.rounds.items()):
            for cid in rd['ids']:
                c = self.st.cands[cid]
                if c['status'] != 'ok':
                    continue
                modes = ['screen'] + (['full'] if rd['promoted'] and cid in rd['promoted'] else [])
                for mode in modes:
                    k = (cid, mode)
                    if mode in c['results'] or k in self.st.pending or self.st.attempts.get(k, 0) >= 2:
                        continue
                    if mode == 'full' and not c['results'].get('screen', {}).get('ok'):
                        continue
                    items.append((r, cid, mode))
        return items

    def poll(self):
        live = self.slurm.live()
        for (cid, mode), sub in list(self.st.pending.items()):
            res, view = grade_dir(cid, mode) / 'result.json', grade_dir(cid, mode) / 'proposer_view.json'
            if res.exists() and view.exists() and res.stat().st_mtime >= sub['t'] - 5:
                self.ingest(cid, mode, res, view)
                continue
            job = sub['job']
            alive = self.gh.running((cid, mode)) if job.startswith('gh') else (job in live or job.startswith('dry'))
            if not alive and time.time() - sub['t'] > 30:
                self.st.emit('error', id=cid, mode=mode, job=job, reason='grade ended without result.json')
                log(f'{cid} {mode}: grade ended without a result (job {job}); attempt '
                    f'{self.st.attempts[(cid, mode)]} of 2')

    def dispatch(self):
        todo = self.needed()
        if not todo:
            return
        todo.sort(key=lambda x: (x[0], x[2] != 'screen'))
        while self.gh.workers and self.gh.free() > 0:
            mine = [x for x in todo if x[2] in self.a.gh_modes]
            if not mine:
                break
            _, cid, mode = mine[0]
            self.st.emit('submit', id=cid, mode=mode, backend='gh', job=self.gh.submit(cid, mode))
            log(f'{cid} {mode}: started on della-gh')
            todo.remove(mine[0])
        live = self.slurm.live()
        jobs = {s['job'] for s in self.st.pending.values() if s['backend'] == 'slurm' and s['job'] in live}
        free = self.slurm.max_jobs - len(jobs)
        while free > 0 and todo:
            mode = todo[0][2]
            batch = [x for x in todo if x[2] == mode][:self.slurm.batch[mode]]
            items = [(cid, m) for _, cid, m in batch]
            try:
                jid = self.slurm.submit(items)
            except (RuntimeError, subprocess.SubprocessError) as e:
                log(f'submit failed: {e}')
                return
            for cid, m in items:
                self.st.emit('submit', id=cid, mode=m, backend='slurm', job=jid)
            log(f'submitted slurm job {jid}: {len(items)} {mode} grades')
            todo = [x for x in todo if x not in batch]
            free -= 1

    def promote(self):
        for r, rd in sorted(self.st.rounds.items()):
            if rd['promoted'] is not None or not rd['generated']:
                continue
            ok = [cid for cid in rd['ids'] if self.st.cands[cid]['status'] == 'ok']
            waiting = [cid for cid in ok if 'screen' not in self.st.cands[cid]['results']
                       and self.st.attempts.get((cid, 'screen'), 0) < 2]
            if waiting:
                continue
            scr = {cid: self.st.cands[cid]['results']['screen'] for cid in ok if 'screen' in self.st.cands[cid]['results']}
            elig = [cid for cid, s in scr.items() if s['ok'] and s['reward'] >= self.a.screen_floor]
            top = sorted(elig, key=lambda c: -scr[c]['reward'])[:self.a.full_top]
            self.st.emit('promote', round=r, ids=top)
            log(f'round {r}: screens done ({len(scr)} graded, {len(elig)} >= floor {self.a.screen_floor}); '
                f'full grades for {top}')

    def round_done(self, r):
        rd = self.st.rounds.get(r)
        if not rd or rd['promoted'] is None:
            return False
        return all('full' in self.st.cands[c]['results'] or self.st.attempts.get((c, 'full'), 0) >= 2
                   for c in rd['promoted'])

    def write_leaderboard(self):
        rows = []
        for cid, c in self.st.cands.items():
            if c['status'] != 'ok' or not c['results']:
                continue
            f, s = c['results'].get('full'), c['results'].get('screen')
            rows.append(dict(id=cid, round=c['round'], kind=c['kind'], parent=c.get('parent'), idea=c.get('idea'),
                             screen=s and s.get('reward'), full=f and f.get('reward'),
                             folds=f and f.get('folds'), config=f and f.get('config'),
                             hidden_ok=f and f.get('hidden_ok'), all19=f and f.get('all19'),
                             accepted=c['accepted'], fail=(f or s or {}).get('fail'),
                             dim=(f or s or {}).get('dim'), sec10k=(f or s or {}).get('sec10k'), code=c['code']))
        rows.sort(key=lambda x: (x['full'] is None, -(x['full'] if x['full'] is not None else -1),
                                 -(x['screen'] if x['screen'] is not None else -1)))
        (RUN / 'leaderboard.json').write_text(json.dumps(rows, indent=1))
        md = ['| id | kind | parent | full | folds | hidden_ok | accepted | screen | idea |', '|' + '---|' * 9]
        for x in rows:
            fmt = lambda v: '' if v is None else f'{v:+.4f}'  # noqa: E731
            md.append(f'| {x["id"]} | {x["kind"]} | {x["parent"] or ""} | {fmt(x["full"])} | '
                      f'{" ".join(f"{g:+.4f}" for g in x["folds"] or [])} | {x["hidden_ok"] if x["full"] is not None else ""} | '
                      f'{"YES" if x["accepted"] else ""} | {fmt(x["screen"])} | '
                      f'{(x["fail"] or x["idea"] or "")[:100].replace("|", "/")} |')
        (RUN / 'leaderboard.md').write_text('\n'.join(md) + '\n')

    # --------------------------------------------------------------------------------------------- run
    def run(self):
        a = self.a
        t_end = time.time() + a.max_hours * 3600
        last_board = 0
        while time.time() < t_end:
            self.poll()
            self.promote()
            self.dispatch()
            if self.future is not None and self.future[3].done():
                self.finish_round()
            gen_rounds = sorted(r for r, rd in self.st.rounds.items() if rd['generated'])
            nxt = (gen_rounds[-1] + 1) if gen_rounds else 0
            prev_ready = nxt == 0 or self.st.rounds[nxt - 1]['promoted'] is not None
            if self.future is None and nxt < a.rounds and prev_ready:
                self.start_round(nxt)
            if time.time() - last_board > 60:
                self.write_leaderboard()
                last_board = time.time()
            if nxt >= a.rounds and self.future is None and all(self.round_done(r) for r in range(a.rounds)) \
                    and not self.st.pending:
                break
            time.sleep(a.tick)
        self.write_leaderboard()
        acc = [cid for cid, c in self.st.cands.items() if c['accepted'] and c.get('kind') != 'seed']
        log(f'loop finished: {len(self.st.cands)} candidates, accepted {acc}; leaderboard {RUN / "leaderboard.md"}')
        self.pool.shutdown(wait=False)


def check_prompts(args):
    lp = Loop(args)
    plans = lp.plan(math.ceil(args.n / args.samples))
    gen = None
    if QWEN_TOKENIZER:
        try:
            from transformers import AutoTokenizer
            gen = AutoTokenizer.from_pretrained(args.tokenizer or QWEN_TOKENIZER[0])
        except ImportError:
            pass
    for i, (k, p) in enumerate(plans):
        q = lp.render(k, p)
        (RUN / 'prompts' / f'check_{i:02d}_{k}.txt').write_text(q)
        n = len(gen.encode(CompletionsGenerator.render(q), add_special_tokens=False)) if gen else None
        print(f'{i:2d} {k:8s} parent={p} chars={len(q)} tokens={n} '
              f'thinking_budget={16384 - n if n else "?"}')
    print(f'prompts written to {RUN / "prompts"}/check_*.txt')


def main():
    global RUN
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run name (state in fair-data/factory/loops/<run>/)')
    ap.add_argument('--generator', default='replay', choices=['replay', 'completions'])
    ap.add_argument('--url', default='', help='farm / ingress base URL serving /v1/completions')
    ap.add_argument('--model', default='Qwen/Qwen3.5-27B')
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=1.0)
    ap.add_argument('--top-k', type=int, default=-1)
    ap.add_argument('--concurrency', type=int, default=64, help='in-flight generation requests')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--n', type=int, default=8, help='candidates per round')
    ap.add_argument('--samples', type=int, default=1, help='samples per prompt (shares the prompt prefix)')
    ap.add_argument('--explore', type=float, default=0.25, help='fraction of explore prompts')
    ap.add_argument('--puct-c', type=float, default=0.5)
    ap.add_argument('--screen-floor', type=float, default=-0.003)
    ap.add_argument('--full-top', type=int, default=8, help='full grades per round (best screens >= floor)')
    ap.add_argument('--slurm-jobs', type=int, default=3, help='concurrent gpu-test jobs (4 max for the user)')
    ap.add_argument('--batch-screen', type=int, default=6)
    ap.add_argument('--batch-full', type=int, default=3)
    ap.add_argument('--gh-workers', type=int, default=0, help='concurrent della-gh grade processes')
    ap.add_argument('--gh-modes', default='screen', help='grade modes della-gh takes: screen or screen,full')
    ap.add_argument('--dry-grade', action='store_true', help='never submit (testing)')
    ap.add_argument('--tick', type=float, default=15.0)
    ap.add_argument('--max-hours', type=float, default=24.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--check-prompts', action='store_true')
    args = ap.parse_args()
    args.gh_modes = [m for m in args.gh_modes.split(',') if m]
    RUN = LOOPS / args.run
    RUN.mkdir(parents=True, exist_ok=True)
    if args.generator == 'completions' and not args.url and not args.check_prompts:
        sys.exit('--generator completions needs --url')
    cfg = RUN / 'config.json'
    if not cfg.exists():
        cfg.write_text(json.dumps(vars(args), indent=1))
    if args.check_prompts:
        return check_prompts(args)
    log(f'loop start: {vars(args)}')
    Loop(args).run()


if __name__ == '__main__':
    main()
