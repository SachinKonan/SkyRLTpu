"""Grade saved Erdős/AC1 candidates with the legacy and the pooled math grader; compare.

The legacy path is discover's ``run_with_timeout`` child (what the ``ray`` and
``local`` eval backends both execute) with no memory limit. The pooled path is
``tpu.science.math_runner`` reached through the evaluator's transport hooks
(verifier injection, flat result, Erdős wrapper) exactly as trainers and farms
use it. Run inside a Slurm allocation: every grade is its own job step, so the
step's cgroup enforces memory the way systemd's MemoryMax does on TPU hosts
(legacy: --legacy-mem GiB, effectively unbounded; pooled: the 8 GiB kill limit).

    python -m tpu.science.sandbox_parity collect --runs DIR... --out cands.jsonl
    python -m tpu.science.sandbox_parity drive --candidates cands.jsonl --out report.json
"""
import argparse
import concurrent.futures as cf
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

EVAL_TIMEOUT = 1100
ENVS = {
    'erdos': ('examples.erdos_min_overlap.env', 'ErdosMinOverlapEnv', 'ErdosMinOverlapRewardEvaluator', ''),
    'ac1': ('examples.ac_inequalities.env', 'AutoCorrInequalityEnv', 'ACInequalitiesRewardEvaluator', 'ac1'),
}

# Edge cases the real rollouts may not cover. {sleep} is replaced per kind.
SYNTHETIC = {
    'erdos': {
        'valid': 'import numpy as np\ndef run():\n    h = np.array(initial_h_values, dtype=float)\n    n = len(h)\n    h = h * ((n / 2.0) / float(np.sum(h)))\n    return h, float(np.max(np.correlate(h, 1 - h, mode="full") * (2.0 / n))), n\n',
        'wrong_claim': 'import numpy as np\ndef run():\n    h = np.array(initial_h_values, dtype=float)\n    return h, 0.1, len(h)\n',
        'nan': 'import numpy as np\ndef run():\n    h = np.full(len(initial_h_values), np.nan)\n    return h, 0.5, len(h)\n',
        'crash': 'def run():\n    raise ValueError("boom")\n',
        'not_a_tuple': 'def run():\n    return 1.0\n',
        'timeout': 'import time\ndef run():\n    time.sleep(10_000)\n',
        'pool': 'import numpy as np\nfrom concurrent.futures import ProcessPoolExecutor\ndef sq(x):\n    return x * x\ndef run():\n    with ProcessPoolExecutor(64) as ex:\n        list(ex.map(sq, range(100)))\n    h = np.array(initial_h_values, dtype=float)\n    n = len(h)\n    h = h * ((n / 2.0) / float(np.sum(h)))\n    return h, float(np.max(np.correlate(h, 1 - h, mode="full") * (2.0 / n))), n\n',
        'mem_6g': 'import numpy as np\ndef run():\n    x = np.ones((6 * 1024 ** 3) // 8)\n    x[::4096] += 1\n    h = np.array(initial_h_values, dtype=float)\n    n = len(h)\n    h = h * ((n / 2.0) / float(np.sum(h)))\n    return h, float(np.max(np.correlate(h, 1 - h, mode="full") * (2.0 / n))), n\n',
        'mem_12g': 'import numpy as np\ndef run():\n    x = np.ones((12 * 1024 ** 3) // 8)\n    x[::4096] += 1\n    h = np.array(initial_h_values, dtype=float)\n    return h, 0.5, len(h)\n',
    },
    'ac1': {
        'valid': 'def propose_candidate():\n    return list(height_sequence_1)\n',
        'negative': 'def propose_candidate():\n    return [-1.0] * 100\n',
        'crash': 'def propose_candidate():\n    raise RuntimeError("boom")\n',
        'stdout': 'def propose_candidate():\n    print("x" * 200000)\n    return list(height_sequence_1)\n',
        'mem_6g': 'import numpy as np\ndef propose_candidate():\n    x = np.ones((6 * 1024 ** 3) // 8)\n    x[::4096] += 1\n    return list(height_sequence_1)\n',
    },
}


def load_kind(kind):
    import importlib
    module, env, evaluator, problem = ENVS[kind]
    m = importlib.import_module(module)
    return getattr(m, env), getattr(m, evaluator), problem


def collect(args):
    rows = []
    for run in args.runs:
        run = Path(run)
        kind = 'erdos' if 'erdos' in run.name else 'ac1'
        snap = json.loads((run / 'puct_sampler_step_000000.json').read_text())
        states = {s['id']: s for s in snap['states'] + snap.get('initial_states', [])}
        for line in gzip.open(run / 'qwen_step_000000.jsonl.gz', 'rt'):
            t = json.loads(line)
            parent = states.get(t['parent_id'])
            if parent is None or not t.get('parsed_code'):
                continue
            rows.append(dict(id=f"{run.name}/{len(rows)}", kind=kind, state=parent, code=t['parsed_code'],
                             recorded=dict(correctness=t['correctness'], reward=t['reward'], message=t['message'][:200])))
    for kind, cases in SYNTHETIC.items():
        env, _, problem = load_kind(kind)
        state = env.create_initial_state(problem).to_dict()
        for name, code in cases.items():
            rows.append(dict(id=f'synthetic/{kind}/{name}', kind=kind, state=state,
                             code='```python\n' + code + '```', recorded=None))
    with open(args.out, 'w') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    print(f'{len(rows)} candidates -> {args.out}')


class RunnerTransport:
    """The pooled executor's runner without systemd: the Slurm step is the cgroup."""
    def __init__(self, work):
        self.work = Path(work)

    def grade_sync(self, request, timeout=None):
        from tpu.science.worker import process_identity
        folder = Path(tempfile.mkdtemp(dir=self.work))
        spec = dict(request.spec, cpus=sorted(os.sched_getaffinity(0)), stdout_limit_bytes=16384,
                    memory_gib=8, systemd=False)
        (folder / 'request.json').write_text(json.dumps(spec))
        code = subprocess.run([sys.executable, '-m', 'tpu.science.math_runner', '--request', str(folder / 'request.json'),
                               '--result', str(folder / 'result.json'), '--owner-pid', str(os.getpid()),
                               '--owner-start', process_identity(os.getpid())],
                              timeout=spec['eval_timeout_seconds'] + 60).returncode
        if not (folder / 'result.json').is_file():
            # The pooled executor maps a dead unit to this candidate failure.
            return dict(result=None, error=f'unit exited {code} (timeout, resource limit, or worker failure)', stdout='')
        return json.loads((folder / 'result.json').read_text())


class SystemdTransport:
    """The production pooled path on a TPU host: core pool + systemd MemoryMax."""
    FAMILIES = {'math': {'slots_per_host': 64, 'cpus': 2, 'memory_gib': 4, 'memory_max_gib': 8}}

    def __init__(self, root):
        self.root = Path(root)

    def grade_sync(self, request, timeout=None):
        from tpu.science.math_grade import grade_math_admitted, GradingInfrastructureFailure
        spec = dict(request.spec, admission_timeout_s=request.admission_timeout_s, systemd=True, stdout_limit_bytes=16384)
        try:
            return grade_math_admitted(spec, self.FAMILIES, root=self.root)
        except GradingInfrastructureFailure as exc:
            return dict(result=None, error=f'infrastructure: {exc}', stdout='')


def grade_one(args):
    row = json.loads(Path(args.candidate).read_text())
    env, evaluator_type, problem = load_kind(row['kind'])
    state = env.state_type.from_dict(row['state'])
    with tempfile.TemporaryDirectory() as tmp:
        backend = 'local' if args.mode == 'legacy' else 'hybrid'
        evaluator = evaluator_type(problem_type=problem, log_dir=tmp, num_cpus_per_task=2,
                                   eval_timeout=EVAL_TIMEOUT, eval_backend=backend)
        if args.mode == 'pooled':
            evaluator.transport = RunnerTransport(tmp)
        elif args.mode == 'systemd':
            evaluator.transport = SystemdTransport(args.root)
        started = time.monotonic()
        out = evaluator.get_reward(row['code'], state)
    print(json.dumps(dict(correctness=out.get('correctness'), reward=out.get('reward'), raw_score=out.get('raw_score'),
                          msg=str(out.get('msg', ''))[:200], seconds=round(time.monotonic() - started, 1)),
                     default=float))


def step(mode, path, mem):
    command = ['srun', '--exact', '--ntasks=1', '--cpus-per-task=2', f'--mem={mem}G', '--quiet',
               sys.executable, '-m', 'tpu.science.sandbox_parity', 'one', '--mode', mode, '--candidate', str(path)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=EVAL_TIMEOUT + 600)
    lines = [l for l in proc.stdout.splitlines() if l.startswith('{')]
    if not lines:
        return dict(correctness=0.0, reward=0.0, raw_score=None, msg=f'step exited {proc.returncode}: {proc.stderr[-300:]}')
    return json.loads(lines[-1])


def same(a, b):
    if a['correctness'] != b['correctness']:
        return False
    if a['correctness'] != 1:
        return True
    close = lambda x, y: x == y or (x is not None and y is not None and math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-15))
    return close(a['raw_score'], b['raw_score']) and close(a['reward'], b['reward'])


def drive_host(args):
    """On a TPU host: legacy grades pinned to idle service cores, pooled grades via systemd."""
    from tpu.science import core_pool
    pool = core_pool.read()
    pairs = [pool['service'][i:i + 2] for i in range(0, len(pool['service']) - 1, 2)][:args.parallel]
    rows = [json.loads(l) for l in open(args.candidates)]
    work = Path(args.out).with_suffix('.d')
    work.mkdir(exist_ok=True)
    jobs = []
    for i, row in enumerate(rows):
        path = work / f'{i:04d}.json'
        path.write_text(json.dumps(row))
        jobs.append((row, path))
    free = list(pairs)
    import threading
    lock = threading.Lock()

    def run(mode, path):
        cores = None
        if mode == 'legacy':
            with lock:
                cores = free.pop()
        try:
            command = [sys.executable, '-m', 'tpu.science.sandbox_parity', 'one', '--mode', mode,
                       '--candidate', str(path), '--root', args.root]
            if cores:
                command = ['taskset', '-c', ','.join(map(str, cores))] + command
            proc = subprocess.run(command, capture_output=True, text=True, timeout=EVAL_TIMEOUT + 600, cwd=args.root)
            lines = [l for l in proc.stdout.splitlines() if l.startswith('{')]
            return json.loads(lines[-1]) if lines else dict(correctness=0.0, reward=0.0, raw_score=None,
                                                            msg=f'exited {proc.returncode}: {proc.stderr[-300:]}')
        finally:
            if cores:
                with lock:
                    free.append(cores)

    results = {}
    with cf.ThreadPoolExecutor(len(pairs)) as legacy_pool, cf.ThreadPoolExecutor(args.parallel) as pooled_pool:
        futures = {}
        for row, path in jobs:
            futures[legacy_pool.submit(run, 'legacy', path)] = (row['id'], 'legacy')
            futures[pooled_pool.submit(run, 'systemd', path)] = (row['id'], 'pooled')
        for future in cf.as_completed(futures):
            key, mode = futures[future]
            results.setdefault(key, {})[mode] = future.result()
            print(key, mode, results[key][mode]['correctness'], flush=True)
    finish(jobs, results, args.out)


def finish(jobs, results, out):
    report = []
    for row, _ in jobs:
        legacy, pooled = results[row['id']]['legacy'], results[row['id']]['pooled']
        report.append(dict(id=row['id'], kind=row['kind'], same=same(legacy, pooled), legacy=legacy, pooled=pooled,
                           recorded=row['recorded']))
    summary = dict(candidates=len(report), identical=sum(r['same'] for r in report),
                   differ=[r['id'] for r in report if not r['same']],
                   valid_legacy=sum(r['legacy']['correctness'] == 1 for r in report),
                   valid_pooled=sum(r['pooled']['correctness'] == 1 for r in report))
    Path(out).write_text(json.dumps(dict(summary=summary, rows=report), indent=1, default=float))
    print(json.dumps(summary, indent=1))


def drive(args):
    rows = [json.loads(l) for l in open(args.candidates)]
    index, count = map(int, args.shard.split('/'))
    rows = rows[index::count]
    work = Path(args.out).with_suffix('.d')
    work.mkdir(exist_ok=True)
    jobs = []
    for i, row in enumerate(rows):
        path = work / f'{i:04d}.json'
        path.write_text(json.dumps(row))
        jobs.append((row, path))
    results = {}
    with cf.ThreadPoolExecutor(args.parallel) as pool:
        futures = {}
        for row, path in jobs:
            futures[pool.submit(step, 'legacy', path, args.legacy_mem)] = (row['id'], 'legacy')
            futures[pool.submit(step, 'pooled', path, 8)] = (row['id'], 'pooled')
        for future in cf.as_completed(futures):
            key, mode = futures[future]
            results.setdefault(key, {})[mode] = future.result()
            print(key, mode, results[key][mode]['correctness'], flush=True)
    report = []
    for row, _ in jobs:
        legacy, pooled = results[row['id']]['legacy'], results[row['id']]['pooled']
        report.append(dict(id=row['id'], kind=row['kind'], same=same(legacy, pooled), legacy=legacy, pooled=pooled,
                           recorded=row['recorded']))
    summary = dict(candidates=len(report), identical=sum(r['same'] for r in report),
                   differ=[r['id'] for r in report if not r['same']],
                   valid_legacy=sum(r['legacy']['correctness'] == 1 for r in report),
                   valid_pooled=sum(r['pooled']['correctness'] == 1 for r in report))
    Path(args.out).write_text(json.dumps(dict(summary=summary, rows=report), indent=1, default=float))
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect'); c.add_argument('--runs', nargs='+', required=True); c.add_argument('--out', required=True)
    o = sub.add_parser('one'); o.add_argument('--mode', choices=['legacy', 'pooled', 'systemd'], required=True); o.add_argument('--candidate', required=True)
    o.add_argument('--root', default=os.getcwd())
    h = sub.add_parser('drive-host'); h.add_argument('--candidates', required=True); h.add_argument('--out', required=True)
    h.add_argument('--parallel', type=int, default=20); h.add_argument('--root', default=os.getcwd())
    d = sub.add_parser('drive'); d.add_argument('--candidates', required=True); d.add_argument('--out', required=True)
    d.add_argument('--parallel', type=int, default=30); d.add_argument('--legacy-mem', type=int, default=64)
    d.add_argument('--shard', default='0/1', help='i/n: grade every n-th candidate starting at i')
    args = parser.parse_args()
    {'collect': collect, 'one': grade_one, 'drive': drive, 'drive-host': drive_host}[args.command](args)


if __name__ == '__main__':
    main()
