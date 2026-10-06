"""Grade Erdős, AC1, qubit and circuit together on one leased farm, then audit isolation.

Driven from the login node over the pool's SSH config (like ``farm_probe``).
Candidates are built with the same code trainers use: the discover evaluators
prepare Erdős/AC1 programs (verifier injection, flat sandbox wrapper) and
``training_env.transport_requests`` builds the qubit and circuit requests with
their resource contracts. Every family is submitted at once under one lease;
while they run, every host is checked for grading units whose pinned CPUs
overlap or leave the host's core pool. Finally each verdict is re-verified on
the driver exactly as the trainer would, and one generation request checks
that serving still works while the pool is busy.

    python -m tpu.swarm.ray_train.grading_matrix --cluster <worker> --hosts 4 \
        --ssh-dir <dir> --model Qwen/Qwen3.5-27B --out report.json
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from .farm_probe import REMOTE

REPO = Path(__file__).resolve().parents[3]
TERMINAL = ('done', 'failed', 'cancelled')

UNITS = r'''
import json, os, socket, subprocess
from pathlib import Path
def cpus(text):
    out = set()
    for part in text.replace(' ', ',').split(','):
        if part:
            a, _, b = part.partition('-')
            out.update(range(int(a), int(b or a) + 1))
    return sorted(out)
pool = Path(f'/tmp/science-cpu-locks-{os.getuid()}/core-pool-v1.json')
names = subprocess.run(['systemctl', 'list-units', '--no-legend', '--plain', '--type=service', '--state=running',
                        'math-grade-*', 'science-grade-*', 'placement-grade-*'], capture_output=True, text=True).stdout.split('\n')
units = {}
for line in names:
    if not line.strip():
        continue
    name = line.split()[0]
    show = subprocess.run(['systemctl', 'show', name, '-p', 'AllowedCPUs', '-p', 'MemoryMax'], capture_output=True, text=True).stdout
    props = dict(l.split('=', 1) for l in show.splitlines() if '=' in l)
    units[name] = dict(cpus=cpus(props.get('AllowedCPUs', '')), memory=props.get('MemoryMax'))
print(json.dumps(dict(host=socket.gethostname(), pool=json.loads(pool.read_text()) if pool.exists() else None, units=units)))
'''


def ssh(args, host, script, params=None, timeout=None):
    config = Path(args.ssh_dir) / args.cluster
    text = ('PARAMS = ' + repr(params) + '\n' if params is not None else '') + script
    for attempt in range(6):
        # Connection-level failures (255: banner timeouts, resets) are retried;
        # the remote calls are idempotent (submit/result/renew by id or lease).
        proc = subprocess.run(['ssh', '-F', str(config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', host, 'python3 -'],
                              input=text, text=True, capture_output=True, timeout=timeout or args.timeout)
        if proc.returncode != 255:
            break
        time.sleep(10 * (attempt + 1))
    if proc.returncode:
        raise RuntimeError(f'ssh {host} failed ({proc.returncode}): {proc.stderr[-800:]}')
    return json.loads(proc.stdout)


def head(args, params):
    return ssh(args, args.cluster, REMOTE, params)


def hosts(args):
    return [args.cluster] + [f'{args.cluster}-worker{i}' for i in range(1, args.hosts)]


# -- candidates --------------------------------------------------------------
ERDOS = '''```python
import time
import numpy as np

def run():
    time.sleep({sleep})
    h = np.array(initial_h_values, dtype=float)
    n = len(h)
    h = h * ((n / 2.0) / float(np.sum(h)))
    c5 = float(np.max(np.correlate(h, 1.0 - h, mode="full") * (2.0 / n)))
    return h, c5, n
```'''

AC1 = '''```python
import time

def propose_candidate():
    time.sleep({sleep})
    return list(height_sequence_1)
```'''


def sandbox_items(kind, count, sleep, log_dir):
    """[(request, check)] for Erdős or AC1 built by the trainer's evaluator."""
    if kind == 'erdos':
        from examples.erdos_min_overlap.env import ErdosMinOverlapEnv as Env, ErdosMinOverlapRewardEvaluator as Eval
        state, problem, code = Env.create_initial_state(''), '', ERDOS
    else:
        from examples.ac_inequalities.env import AutoCorrInequalityEnv as Env, ACInequalitiesRewardEvaluator as Eval
        state, problem, code = Env.create_initial_state('ac1'), 'ac1', AC1
    evaluator = Eval(problem_type=problem, log_dir=str(log_dir), num_cpus_per_task=2, eval_timeout=1100,
                     eval_backend='hybrid')
    items = []
    for _ in range(count):
        program = evaluator.preprocess_generation(evaluator._extract_code(code.format(sleep=sleep)), state)
        request = evaluator.grading_request(program)

        def check(payload, evaluator=evaluator):
            if payload.get('error'):
                return False, f"sandbox error: {payload['error'][:200]}"
            output = evaluator.transport_result(payload['result'])
            if kind == 'erdos':
                from examples.erdos_min_overlap.env import verify_erdos_solution, evaluate_erdos_solution
                ok = verify_erdos_solution(output)
                return ok, f'c5={evaluate_erdos_solution(*output) if ok else None}'
            from examples.ac_inequalities.env import verify_ac1_solution, evaluate_sequence_ac1
            ok = verify_ac1_solution(output)
            return ok, f'ac1={evaluate_sequence_ac1(output) if ok else None}'
        items.append((request, check))
    return items


def science_items(task, count, timeout=7200):
    os.environ.setdefault('SCIENCE_ROUTING_EVALUATOR', 'parallel-v2')
    os.environ.setdefault('SCIENCE_ROUTING_SUITE', 'full')
    os.environ.setdefault('SCIENCE_ROUTING_SLOTS_PER_HOST', '8')
    os.environ.setdefault('SCIENCE_PLACEMENT_RUNTIME', 'cpu300-4g-v1')
    os.environ.setdefault('SCIENCE_PLACEMENT_SLOTS_PER_HOST', '48')
    os.environ.setdefault('SCIENCE_PLACEMENT_HELPER', 'fast_proxy_v1')
    from tpu.science.training_env import transport_requests
    name = 'seed_routing.py' if task == 'routing' else 'challenge_seed_fast_proxy.py'
    source = (REPO / 'tpu/science' / name).read_text()
    groups = []
    for _ in range(count):
        groups.append(transport_requests(task, source, timeout))
    return groups


def body(request):
    return dict(request_id=uuid.uuid4().hex, task=request.task, scope=dict(request.scope or {}, probe='matrix'),
                spec=dict(request.spec, admission_timeout_s=request.admission_timeout_s))


# -- run ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cluster', required=True, help='pool worker cluster name of the farm (head alias)')
    parser.add_argument('--hosts', type=int, required=True)
    parser.add_argument('--ssh-dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--erdos', type=int, default=24)
    parser.add_argument('--ac1', type=int, default=24)
    parser.add_argument('--qubit', type=int, default=2)
    parser.add_argument('--circuit', type=int, default=2)
    parser.add_argument('--sleep', type=int, default=120, help='seconds each sandbox candidate holds its cores')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--deadline', type=int, default=7200)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = dict(cluster=args.cluster, started=time.time(), checks={})

    owner = f'grading-matrix-{int(time.time())}'
    # Farms admit only a lessee that names their serving runtime's attestation.
    sha = (head(args, dict(action='status'))['status'][1].get('capabilities') or {}).get('compatibility_sha256')
    report['compatibility_sha256'] = sha
    ack = head(args, dict(action='lease', owner=owner, ttl=600, sha=sha))['acquire']
    if ack[0] != 200:
        raise SystemExit(f'lease refused: {ack}')
    lease = ack[1]['lease_id']
    try:
        with tempfile.TemporaryDirectory() as tmp:
            entries = []  # (family, label, body, check)
            for kind, count in (('erdos', args.erdos), ('ac1', args.ac1)):
                for request, check in sandbox_items(kind, count, args.sleep, Path(tmp) / kind):
                    entries.append((kind, kind, body(request), check))
            for task, family, count in (('routing', 'qubit', args.qubit), ('placement', 'circuit', args.circuit)):
                for index, group in enumerate(science_items(task, count)):
                    for request in group:
                        entries.append((family, f'{family}-{index}', body(request), None))
            report['submitted'] = {f: sum(1 for e in entries if e[0] == f) for f in ('erdos', 'ac1', 'qubit', 'circuit')}
            print('submitting', report['submitted'], flush=True)
            for start in range(0, len(entries), 16):
                chunk = [e[2] for e in entries[start:start + 16]]
                for code, view in head(args, dict(action='submit', lease_id=lease, bodies=chunk))['submit']:
                    if code not in (200, 202):
                        raise SystemExit(f'submit refused {code}: {view}')

            # Serving still answers while every family grades.
            # Gemma degenerates without <bos> and its chat template; trainers always send both.
            prompt = ('<bos><start_of_turn>user\nName three prime numbers.<end_of_turn>\n<start_of_turn>model\n'
                      if 'gemma' in args.model.lower() else 'Name three prime numbers.')
            generation = head(args, dict(action='generate', lease_id=lease, model=args.model, prompt=prompt))
            report['generation'] = generation
            audits, deadline, views = [], time.monotonic() + args.deadline, {}
            ids = [e[2]['request_id'] for e in entries]
            next_audit = time.monotonic() + 45
            while time.monotonic() < deadline:
                head(args, dict(action='renew', owner=owner, lease_id=lease, ttl=600, sha=sha))
                for start in range(0, len(ids), 32):
                    for code, view in head(args, dict(action='result', lease_id=lease, ids=ids[start:start + 32]))['result']:
                        if code == 200:
                            views[view['request_id']] = view
                if time.monotonic() >= next_audit:
                    audits.append([ssh(args, h, UNITS) for h in hosts(args)])
                    next_audit = time.monotonic() + 90
                states = [views.get(i, {}).get('state') for i in ids]
                running = {f: sum(1 for e, s in zip(entries, states) if e[0] == f and s not in TERMINAL)
                           for f in ('erdos', 'ac1', 'qubit', 'circuit')}
                print(time.strftime('%H:%M:%S'), 'running', running,
                      'units', [sum(len(h['units']) for h in a) for a in audits[-1:]], flush=True)
                if all(s in TERMINAL for s in states):
                    break
                time.sleep(20)
            report['audits'] = audits
            report['checks'] = verdicts(entries, views, audits)
    finally:
        head(args, dict(action='release', lease_id=lease))
    report['finished'] = time.time()
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report['checks'], indent=1, default=str))
    if not all(c.get('ok') for c in report['checks'].values()):
        raise SystemExit('grading matrix FAILED')


def verdicts(entries, views, audits):
    from tpu.science.challenge_contract import aggregate
    checks, by_label = {}, {}
    for family, label, request, check in entries:
        view = views.get(request['request_id'], {})
        by_label.setdefault((family, label), []).append((view, check))
    for family in ('erdos', 'ac1', 'qubit', 'circuit'):
        rows = [(label, items) for (f, label), items in by_label.items() if f == family]
        ok, notes, admission = True, [], set()
        for label, items in rows:
            results = [v.get('result') or {} for v, _ in items]
            for view, _ in items:
                admission.add((view.get('result') or {}).get('metrics', {}).get('admission'))
                if view.get('state') != 'done':
                    ok = False
                    notes.append(f"{label}: state={view.get('state')} error={view.get('error')}")
            if not ok:
                continue
            if family in ('erdos', 'ac1'):
                for view, check in items:
                    good, note = check(view['result'])
                    ok &= bool(good)
                    notes.append(note)
            elif family == 'qubit':
                result = results[0]
                good = result.get('correctness') == 1 and result.get('metrics', {}).get('case_count') == 72
                ok &= good
                notes.append(f"reward={result.get('reward')} cases={result.get('metrics', {}).get('case_count')}")
            else:
                result = aggregate(results)
                helper = all(r.get('metrics', {}).get('helper') == 'fast_proxy_v1' for r in results)
                ok &= result.get('correctness') == 1 and helper and len(results) == 17
                notes.append(f"reward={result.get('reward')} cases={len(results)} helper={helper}")
        checks[family] = dict(ok=ok and bool(rows), graded=sum(len(i) for _, i in rows),
                              admission=sorted(map(str, admission)), notes=notes[:6])
    # Isolation: every running unit pinned inside the pool, never sharing a core.
    overlap, outside, peak, families = [], [], 0, set()
    for audit in audits:
        for host in audit:
            pool = set((host.get('pool') or {}).get('grading', []))
            seen = {}
            for name, unit in host['units'].items():
                families.add(name.split('-grade-')[0])
                if not set(unit['cpus']) <= pool:
                    outside.append((host['host'], name))
                for cpu in unit['cpus']:
                    if cpu in seen:
                        overlap.append((host['host'], name, seen[cpu], cpu))
                    seen[cpu] = name
            peak = max(peak, len(host['units']))
    checks['isolation'] = dict(ok=bool(audits) and not overlap and not outside, audits=len(audits),
                               peak_units_per_host=peak, unit_kinds=sorted(families),
                               overlap=overlap[:5], outside_pool=outside[:5])
    return checks


if __name__ == '__main__':
    main()
