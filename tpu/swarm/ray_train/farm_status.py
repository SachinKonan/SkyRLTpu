"""On-demand inference farm status, grouped by pool.

For every inference-farm job: model, job state, health, lease (owner job), and
live engine load (generation tokens/s over a short window, running/waiting
requests, KV-cache use). One SSH per running farm, all in parallel; each farm
head reads its own ingress and every engine's /metrics twice, WINDOW seconds
apart. Read-only: it never leases, cancels or changes anything.

Run on the SkyPilot API host (della-milan):
  SKYPILOT_DISABLE_LOCAL_API_SERVER=1 HOME=<sky home> python -m tpu.swarm.ray_train.farm_status [--json]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

SSH_DIR = Path('/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/generated/ssh')
PORT = 24800
ENGINE_PORT = 24801

REMOTE = r'''
import json, re, time, urllib.request
P = PARAMS
def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()
out = dict(health=None, status=None, engines=[])
try:
    get(f'http://127.0.0.1:{P["port"]}/health'); out['health'] = 'ok'
except Exception as e:
    out['health'] = type(e).__name__
try:
    s = json.loads(get(f'http://127.0.0.1:{P["port"]}/status'))
    out['status'] = {k: s.get(k) for k in ('state', 'owner_run', 'expected', 'replicas', 'ready_engines',
                                           'expected_engines', 'fatal_error', 'exhausted')}
    out['status']['replicas'] = len(s.get('replicas') or [])
except Exception as e:
    out['status_error'] = type(e).__name__
endpoints = []
for item in (out.get('status') or {}).get('expected') or []:
    endpoints.append(item if ':' in item else f'{item}:{P["engine_port"]}')
def sample():
    vals = {}
    for ep in endpoints:
        try:
            text = get(f'http://{ep}/metrics', timeout=4)
        except Exception:
            vals[ep] = None; continue
        m = {}
        for name in ('generation_tokens_total', 'prompt_tokens_total', 'num_requests_running',
                     'num_requests_waiting', 'kv_cache_usage_perc'):
            hit = re.search(r'^vllm:' + name + r'\{[^}]*\} ([0-9.e+]+)$', text, re.M)
            m[name] = float(hit.group(1)) if hit else None
        vals[ep] = m
    return time.time(), vals
t0, a = sample()
time.sleep(P['window'])
t1, b = sample()
for ep in endpoints:
    x, y = a.get(ep), b.get(ep)
    e = dict(endpoint=ep, ok=bool(x and y))
    if x and y:
        dt = t1 - t0
        for k in ('generation_tokens_total', 'prompt_tokens_total'):
            if x[k] is not None and y[k] is not None:
                e[k.replace('_total', '_per_s')] = (y[k] - x[k]) / dt
        e.update(running=y['num_requests_running'], waiting=y['num_requests_waiting'],
                 kv=y['kv_cache_usage_perc'])
    out['engines'].append(e)
print(json.dumps(out))
'''


def inventory():
    import sky
    rows, *_ = sky.get(sky.jobs.queue_v2(refresh=False, skip_finished=True,
                                         fields=['job_id', 'job_name', 'status', 'current_cluster_name', 'pool']))
    return [dict(job_id=r.job_id, name=r.job_name, status=getattr(r.status, 'value', str(r.status)),
                 cluster=r.current_cluster_name, pool=r.pool) for r in rows
            if 'inference-farm' in (r.job_name or '')]


def model_of(name):
    for key, label in (('qwen35', 'Qwen3.5'), ('qwen38', 'Qwen3.8'), ('gemma', 'Gemma'), ('muse', 'Muse')):
        if key in name:
            return label
    return '?'


def probe(row, window, timeout):
    cluster = row.get('cluster')
    if row['status'] != 'RUNNING' or not cluster or not (SSH_DIR / cluster).is_file():
        return row
    params = dict(port=PORT, engine_port=ENGINE_PORT, window=window)
    try:
        proc = subprocess.run(['ssh', '-F', str(SSH_DIR / cluster), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                               cluster, 'python3 -'], input=f'PARAMS = {params!r}\n{REMOTE}', text=True,
                              capture_output=True, timeout=timeout)
        row['probe'] = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else \
            {'error': f'ssh rc={proc.returncode}'}
    except (subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        row['probe'] = {'error': type(exc).__name__}
    return row


def summarize(row):
    p = row.get('probe') or {}
    status = p.get('status') or {}
    engines = p.get('engines') or []
    if row['status'] != 'RUNNING':
        health = f'no worker ({row["status"].lower()})'
    elif 'error' in p:
        health = f'not responding ({p["error"]})'
    elif p.get('health') != 'ok':
        busy = status.get('replicas', 0) < (status.get('expected_engines') or 1)
        health = 'starting (engines loading)' if busy and not status.get('fatal_error') else \
            f'not responding ({p.get("health")})'
    else:
        health = 'healthy'
    owner = (status.get('owner_run') or '').split(':')[0]
    lease = status.get('state') or '-'
    leased_to = owner if owner and lease not in ('unleased', 'expired') else ('free' if health == 'healthy' else '-')
    live = [e for e in engines if e.get('ok')]
    tps = sum(e.get('generation_tokens_per_s') or 0 for e in live)
    running = sum(e.get('running') or 0 for e in live)
    waiting = sum(e.get('waiting') or 0 for e in live)
    kv = max((e.get('kv') or 0 for e in live), default=None)
    return dict(pool=(row.get('pool') or '?').replace('tpuswarm-', ''), farm=row['name'].replace('inference-farm-', ''),
                model=model_of(row['name']), job=row['job_id'], health=health, lease=lease, leased_to=leased_to,
                engines=f'{len(live)}/{len(engines)}' if engines else '-',
                gen_tok_s=round(tps) if live else None, running=int(running) if live else None,
                waiting=int(waiting) if live else None, kv_max=round(100 * kv, 1) if kv is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--window', type=float, default=5.0, help='seconds between the two metric samples')
    parser.add_argument('--timeout', type=float, default=60.0)
    args = parser.parse_args()
    started = time.time()
    rows = inventory()
    with ThreadPoolExecutor(max_workers=48) as pool:
        rows = list(pool.map(lambda r: probe(r, args.window, args.timeout), rows))
    out = sorted((summarize(r) for r in rows), key=lambda r: (r['pool'], r['model'], r['farm']))
    if args.json:
        print(json.dumps(out, indent=1))
        return
    by_pool = {}
    for r in out:
        by_pool.setdefault(r['pool'], []).append(r)
    for pool_name, items in by_pool.items():
        healthy = sum(r['health'] == 'healthy' for r in items)
        tok = sum(r['gen_tok_s'] or 0 for r in items)
        print(f'\n## {pool_name}: {len(items)} farms, {healthy} healthy, {tok} gen tok/s total')
        print('| model | farm | job | health | leased to | engines | gen tok/s | running | waiting | KV max % |')
        print('|---|---|---|---|---|---|---|---|---|---|')
        for r in items:
            cells = [r['model'], r['farm'], r['job'], r['health'], r['leased_to'], r['engines'],
                     r['gen_tok_s'], r['running'], r['waiting'], r['kv_max']]
            print('| ' + ' | '.join('-' if c is None else str(c) for c in cells) + ' |')
    print(f'\n({len(out)} farm jobs, {time.time() - started:.0f}s)', file=sys.stderr)


if __name__ == '__main__':
    main()
