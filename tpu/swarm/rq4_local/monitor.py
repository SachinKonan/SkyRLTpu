"""RQ4 no-train monitor: runs, farms and pools in one report, plus flags that need action.

Writes logs/monitor-latest.txt (and appends a JSON line to logs/monitor.jsonl). Read-only: it never
cancels, launches or restarts anything; the 20-minute check acts on its flags.
Usage: python3 monitor.py [--window-min 20]
"""
import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time

L = Path('/scratch/gpfs/ZHUANGL/sk7524/rq4-local')
SKY = '/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/sky'
ENV = dict(os.environ, SKYPILOT_DISABLE_LOCAL_API_SERVER='1')
MODELS = {'qwen35': 'Qwen3.5', 'gemma': 'Gemma', 'muse': 'Muse'}
PROBLEMS = ('qubit', 'circuit', 'ac1', 'erdos')
TARGET = 25
ANSI = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]')


def run_ids():
    for p in PROBLEMS:
        for m in MODELS:
            for s in (1, 2):
                rid = f'rq4-notrain-local-{m}-{p}-s{s}-20261007'
                if (m, p, s) == ('muse', 'qubit', 2):
                    rid = 'rq4-notrain-local-muse-qubit-s2-20261013'
                yield m, p, s, rid


def alive(rid):
    try:
        os.kill(int((L / 'runs' / rid / 'runner.pid').read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def client_env(rid):
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            env = (Path('/proc') / pid / 'environ').read_bytes().split(b'\0')
            cmd = (Path('/proc') / pid / 'cmdline').read_bytes()
        except OSError:
            continue
        if f'EXPERIMENT_NAME={rid}'.encode() in env and (b'run_ttd_ensemble' in cmd or b'seed_bootstrap' in cmd):
            return {e.split(b'=', 1)[0].decode(): e.split(b'=', 1)[1].decode() for e in env if b'=' in e}, \
                ('bootstrap' if b'seed_bootstrap' in cmd else 'client')
    return None, None


def tail_events(path, since, max_bytes=40_000_000):
    """Count events newer than `since` from the end of a large jsonl file."""
    counts = collections.Counter()
    reasons = collections.Counter()
    try:
        size = path.stat().st_size
        with open(path, 'rb') as f:
            f.seek(max(0, size - max_bytes))
            for line in f:
                if b'"time"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get('time', 0) < since:
                    continue
                counts[e.get('event')] += 1
                if e.get('event') == 'grading_retry':
                    reasons[str(e.get('reason', ''))[:40]] += 1
    except OSError:
        pass
    return counts, reasons


def runs_report(window):
    now = time.time()
    out = []
    for m, p, s, rid in run_ids():
        d = L / 'runs' / rid / 'runs' / rid
        tl = d / 'client/tinker_log' / rid
        try:
            step = json.loads((tl / 'global_step.json').read_text())['global_step']
        except (OSError, ValueError, KeyError):
            step = 0
        rows = []
        try:
            rows = [json.loads(x) for x in (tl / 'metrics.jsonl').read_text().splitlines() if x.strip()]
        except OSError:
            pass
        last_step_h = round((now - (tl / 'metrics.jsonl').stat().st_mtime) / 3600, 1) if rows else None
        env, kind = client_env(rid)
        ev, reasons = tail_events(d / 'inference-events.jsonl', now - window)
        lev, _ = tail_events(d / 'local.jsonl', now - window)
        out.append(dict(
            model=m, problem=p, seed=s, run=rid, step=step, target=TARGET, alive=alive(rid), process=kind,
            bootstrap_done=(d / 'client/bootstrap/complete.json').exists(),
            step_min=[round(r.get('time/total', 0) / 60) for r in rows[-3:]],
            late=[r.get('eager/late_candidates') for r in rows[-2:]],
            eager=(env or {}).get('TTD_EAGER_STEP_FRACTION'), hours_since_step=last_step_h,
            graded=ev['grading_completed'], generated=ev['hybrid_generated'] + ev['borrow_generated'],
            retries=ev['grading_retry'], no_pool=ev['grading_no_pool'], failed=lev['run_failed'],
            stalled=lev['client_stalled'], retry_reasons=dict(reasons.most_common(2))))
    return out


def farms_report():
    proc = subprocess.run(['bash', str(L / 'farm_status.sh'), '--json'], capture_output=True, text=True, timeout=600)
    try:
        return json.loads(proc.stdout)
    except ValueError:
        return []


def sky(*args, timeout=300):
    try:
        proc = subprocess.run([SKY, *args], capture_output=True, text=True, timeout=timeout, env=ENV)
        return ANSI.sub('', proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired:
        return ''


def pools_report():
    pools = {}
    for pool in ('tpuswarm-v5p32-east5a-erdos', 'tpuswarm-v6e8-east5b'):
        text = sky('jobs', 'pool', 'status', pool, '--all')
        ready = [ln.split() for ln in text.splitlines() if ' READY ' in ln and ln.startswith(pool)]
        workers = [(r[1], r[-1]) for r in ready if r[1] != '3' and r[1] != '7']
        pools[pool] = dict(ready=len(workers), idle=[w for w, j in workers if j == '-'])
    return pools


def farm_jobs():
    """Latest job id per farm profile we launched, with its SkyPilot status."""
    latest = {}
    for line in (L / 'scripts/farm_jobs.txt').read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[3].isdigit() and parts[0].startswith('inference-farm-rq4-'):
            latest[parts[0]] = (parts[3], parts[2])
    text = sky('jobs', 'queue', '--all')
    status = {}
    for ln in text.splitlines():
        m = re.match(r'^(\d+)\s', ln)
        if m:
            st = re.search(r'\b(PENDING|STARTING|RUNNING|RECOVERING|CANCELLED|CANCELLING|FAILED[A-Z_]*|SUCCEEDED)\b', ln)
            status[m.group(1)] = st.group(1) if st else '?'
    return {name: dict(job=job, pool=pool, status=status.get(job, '?')) for name, (job, pool) in latest.items()}


def flags(runs, farms, pools, jobs):
    out = []
    by_model = collections.defaultdict(lambda: dict(free=[], leased=0, runs_without=[]))
    leased_to = collections.defaultdict(list)
    for f in farms:
        model = {'Qwen3.5': 'qwen35', 'Gemma': 'gemma', 'Muse': 'muse'}.get(f.get('model'))
        if not model or f.get('health') != 'healthy':
            continue
        owner = f.get('leased_to') or ''
        if owner.startswith('rq4-notrain-local-'):
            by_model[model]['leased'] += 1
            leased_to[owner].append(f['farm'])
        elif owner in ('free', '', '-'):
            by_model[model]['free'].append(f['farm'])
    for r in runs:
        if r['step'] >= r['target'] or r['problem'] not in ('qubit', 'circuit'):
            continue
        if not r['alive']:
            out.append(f"STOPPED {r['run']} at {r['step']}/{r['target']} (failed in window: {r['failed']})")
        elif not leased_to.get(r['run']):
            by_model[r['model']]['runs_without'].append(r['run'])
        if r['retries'] > 200:
            out.append(f"RETRIES {r['run']}: {r['retries']} grading retries in window {r['retry_reasons']}")
        if r['alive'] and leased_to.get(r['run']) and r['graded'] == 0 and r['generated'] == 0 \
                and (r['hours_since_step'] or 0) > 3:
            out.append(f"NO-WORK {r['run']}: holds {leased_to[r['run']]} but nothing graded/generated in window, "
                       f"{r['hours_since_step']} h since last step")
    for model, v in by_model.items():
        if v['runs_without'] and not v['free']:
            out.append(f"FARM-SHORT {MODELS[model]}: {len(v['runs_without'])} qubit/circuit runs without a farm, "
                       f"none free ({', '.join(x.split('local-')[1] for x in v['runs_without'])})")
        if v['free'] and not v['runs_without']:
            out.append(f"FARM-SPARE {MODELS[model]}: free {v['free']}")
    for name, j in sorted(jobs.items()):
        # Only our current RQ4 model farms in us-east5; deliberately cancelled jobs are not relaunch candidates.
        if not re.match(r'inference-farm-rq4-(qwen35|gemma|muse)-east5', name):
            continue
        if j['status'].startswith('FAILED'):
            out.append(f"FARM-JOB-ENDED {name} job {j['job']} {j['status']} (relaunch candidate)")
    for pool, p in pools.items():
        if p['idle']:
            out.append(f"IDLE-WORKERS {pool}: {len(p['idle'])} idle {p['idle'][:6]}")
    try:
        os.kill(int((L / 'scheduler.pid').read_text()), 0)
    except (OSError, ValueError):
        out.append('SCHEDULER DOWN')
    # Discovery publishes `updated` (older registries: `time`) every interval; a registry that stopped moving
    # is not the current fleet (same rule as discovery.registry_is_fresh, inlined: this runs outside the package).
    try:
        doc = json.loads((L / 'farms.json').read_text())
        age = time.time() - float(doc.get('updated', doc.get('time')))
    except (OSError, ValueError, TypeError):
        age = None
    if age is None or age > 300:
        out.append(f'DISCOVERY STALE: farms.json not updated for {age if age is None else round(age)}s '
                   '(check the discovery unit)')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-min', type=float, default=20)
    args = parser.parse_args()
    window = args.window_min * 60
    runs = runs_report(window)
    farms = farms_report()
    pools = pools_report()
    jobs = farm_jobs()
    fl = flags(runs, farms, pools, jobs)
    now = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')
    lines = [f'RQ4 monitor {now}', '', 'Runs (qubit/circuit first):']
    leased = collections.defaultdict(list)
    for f in farms:
        if (f.get('leased_to') or '').startswith('rq4-notrain-local-'):
            leased[f['leased_to']].append(f"{f['farm'].replace('rq4-', '')}:{f.get('gen_tok_s') or 0}")
    for r in runs:
        if r['problem'] in ('erdos',) or (r['problem'] == 'ac1' and r['step'] >= r['target']):
            continue
        lines.append(f"  {r['problem']:7s} {MODELS[r['model']]:7s} s{r['seed']} {r['step']:2d}/{r['target']} "
                     f"{'run ' if r['alive'] else 'STOP'} {r['process'] or '-':9s} eager={r['eager'] or '-':4s} "
                     f"step_min={r['step_min']} late={r['late']} since_step={r['hours_since_step']}h "
                     f"graded={r['graded']} gen={r['generated']} retries={r['retries']} no_pool={r['no_pool']} "
                     f"farms={leased.get(r['run']) or '-'}")
    lines += ['', 'Farms:']
    by = collections.defaultdict(list)
    for f in farms:
        if f.get('model') in ('Qwen3.5', 'Gemma', 'Muse'):
            by[f['model']].append(f"{f['farm'].replace('rq4-', '').replace('-east5a-v5p32-', ' v5p ').replace('-east5-v6e8-', ' v6e ')}"
                                  f"[{f.get('health')}{'→' + f['leased_to'].replace('rq4-notrain-local-', '').replace('-20261007', '') if (f.get('leased_to') or '').startswith('rq4-notrain') else ''}]")
    for model, items in by.items():
        lines.append(f'  {model}: ' + ', '.join(items))
    lines += ['', 'Farm jobs not running: ' + ', '.join(f"{n.replace('inference-farm-rq4-', '')}={j['job']}:{j['status']}" for n, j in sorted(jobs.items())
                                                      if j['status'] not in ('RUNNING', 'CANCELLED') and re.match(r'inference-farm-rq4-(qwen35|gemma|muse)-east5', n))]
    lines += ['', 'Pools: ' + '; '.join(f"{p}: {v['ready']} ready, {len(v['idle'])} idle" for p, v in pools.items())]
    lines += ['', 'FLAGS:'] + [f'  - {x}' for x in fl] + ([] if fl else ['  (none)'])
    text = '\n'.join(lines)
    (L / 'logs/monitor-latest.txt').write_text(text + '\n')
    with open(L / 'logs/monitor.jsonl', 'a') as out:
        out.write(json.dumps(dict(time=time.time(), runs=runs, pools=pools, jobs=jobs, flags=fl)) + '\n')
    print(text)


if __name__ == '__main__':
    main()
