"""Read-only status page served by the discovery service.

A background collector refreshes every REFRESH seconds:
  * farms: every inference-farm job, grouped by pool, with health, lease owner
    and live engine load (farm_status.probe: one SSH per farm, in parallel);
  * trainers: every other managed job, with its step read from the run
    directory on its head host (one SSH per job, in parallel);
  * local runs: RQ4 no-train runs on this host (rq4-local), read from disk.
The HTTP server only renders the latest snapshot, so a slow cluster never
blocks a page load. It never leases, cancels or changes anything.

Endpoints: `/` (HTML, auto-refresh) and `/api/status.json`.
"""
from concurrent.futures import ThreadPoolExecutor
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

from . import farm_status

REFRESH = 30
NOISY = {'writeback_complete', 'inference_status', 'hosts', 'remote_leases', 'transfer_progress',
         'transfer_complete', 'host_heartbeat', 'heartbeat'}

TRAINER_REMOTE = r'''
import glob, json, os
n = PARAMS['name']
R = os.path.expanduser(f'~/.cache/{n}/runs/{n}')
out = dict(found=os.path.isdir(R))
def last_line(path):
    with open(path, 'rb') as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 65536))
        lines = f.read().splitlines()
    return lines[-1].decode(errors='replace') if lines else None
if out['found']:
    log = f'{R}/client/tinker_log/{n}'
    try: out['global_step'] = json.load(open(f'{log}/global_step.json'))['global_step']
    except Exception: pass
    try:
        with open(f'{log}/metrics.jsonl', 'rb') as f: out['metric_rows'] = sum(1 for _ in f)
        row = json.loads(last_line(f'{log}/metrics.jsonl'))
        out['best'] = row.get('pool/best_value')
        out['correct'] = next((v for k, v in row.items() if k.endswith('env/all/correctness')), None)
        out['step_seconds'] = row.get('time/total')
    except Exception: pass
    try: out['num_epochs'] = int(json.load(open(f'{R}/launch-client-0.json'))['environment'].get('NUM_EPOCHS'))
    except Exception: pass
    b = f'{R}/client/bootstrap'
    if os.path.isdir(b):
        out['bootstrap_done'] = os.path.exists(f'{b}/complete.json')
        out['bootstrap_groups'] = len(glob.glob(f'{b}/drafts/group-*/generation.json'))
    try:
        with open(f'{R}/controller.jsonl', 'rb') as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 262144))
            for line in reversed(f.read().splitlines()):
                try: e = json.loads(line)
                except Exception: continue
                if e.get('event') not in PARAMS['noisy']:
                    out['last_event'] = e.get('event'); out['last_event_time'] = e.get('time'); break
    except Exception: pass
print(json.dumps(out))
'''


def ssh_python(cluster, params, script, timeout=40):
    config = farm_status.SSH_DIR / cluster
    if not config.is_file():
        return {'error': 'no ssh config'}
    try:
        proc = subprocess.run(['ssh', '-F', str(config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                               cluster, 'python3 -'], input=f'PARAMS = {params!r}\n{script}', text=True,
                              capture_output=True, timeout=timeout)
        return json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else \
            {'error': f'ssh rc={proc.returncode}'}
    except (subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        return {'error': type(exc).__name__}


def all_jobs():
    import sky
    rows, *_ = sky.get(sky.jobs.queue_v2(refresh=False, skip_finished=True,
                                         fields=['job_id', 'job_name', 'status', 'current_cluster_name', 'pool']))
    return [dict(job_id=r.job_id, name=r.job_name or '', status=getattr(r.status, 'value', str(r.status)),
                 cluster=r.current_cluster_name, pool=r.pool) for r in rows]


def trainer_row(job):
    row = dict(pool=(job.get('pool') or '-').replace('tpuswarm-', ''), job=job['job_id'], name=job['name'],
               state=job['status'], worker=(job.get('cluster') or '-').rsplit('-', 1)[-1])
    if job['status'] == 'RUNNING' and job.get('cluster'):
        p = ssh_python(job['cluster'], dict(name=job['name'], noisy=sorted(NOISY)), TRAINER_REMOTE)
        row.update(p)
    return row


def local_runs(root):
    rows = []
    for d in sorted(Path(root).glob('rq4-notrain-local-*')):
        if d.name.endswith('-smoke'):
            continue
        run = d.name
        R = d / 'runs' / run
        if not (R / 'local.jsonl').exists():
            continue
        try:
            alive = Path(f'/proc/{(d / "runner.pid").read_text().strip()}').exists()
        except OSError:
            alive = False
        log = R / 'client' / 'tinker_log' / run
        step = best = correct = None
        try:
            step = json.loads((log / 'global_step.json').read_text())['global_step']
        except (OSError, ValueError, KeyError):
            pass
        try:
            last = (log / 'metrics.jsonl').read_text().strip().splitlines()[-1]
            m = json.loads(last)
            best = m.get('pool/best_value')
            correct = next((v for k, v in m.items() if k.endswith('env/all/correctness')), None)
        except (OSError, ValueError, IndexError):
            pass
        boot = 0
        try:
            boot = sum('bootstrap_group_complete' in l for l in open(R / 'bootstrap.log', errors='replace'))
        except OSError:
            pass
        farms = []
        try:
            with open(R / 'inference-events.jsonl', 'rb') as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 262144))
                for line in reversed(f.read().splitlines()):
                    if b'"borrow_reconcile"' in line:
                        farms = json.loads(line)['held']
                        break
        except (OSError, ValueError):
            pass
        rows.append(dict(run=run.replace('rq4-notrain-local-', ''), alive=alive, step=step, bootstrap_groups=boot,
                         best=best, correct=correct, farm_urls=farms))
    return rows


class Collector:
    def __init__(self, local_root):
        self.local_root = local_root
        self.snapshot = dict(time=None, farms=[], trainers=[], local=[], error='collecting first snapshot')
        self.lock = threading.Lock()

    def collect(self):
        started = time.time()
        jobs = all_jobs()
        farm_jobs = [dict(job_id=j['job_id'], name=j['name'], status=j['status'], cluster=j['cluster'],
                          pool=j['pool']) for j in jobs if 'inference-farm' in j['name']]
        others = [j for j in jobs if 'inference-farm' not in j['name']]
        with ThreadPoolExecutor(max_workers=64) as pool:
            farms = list(pool.map(lambda r: farm_status.probe(r, 5.0, 60.0), farm_jobs))
            trainers = list(pool.map(trainer_row, others))
        farms = sorted((farm_status.summarize(r) for r in farms), key=lambda r: (r['pool'], r['model'], r['farm']))
        local = local_runs(self.local_root) if self.local_root else []
        url_name = {}
        try:
            registry = json.loads((Path(self.local_root).parent / 'farms.json').read_text())
            url_name = {f['url']: f['name'].replace('inference-farm-', '') for f in registry.get('farms', [])}
        except (OSError, ValueError, TypeError):
            pass
        for r in local:
            r['farms'] = [url_name.get(u, u) for u in r.pop('farm_urls')]
        return dict(time=time.time(), seconds=round(time.time() - started, 1), farms=farms,
                    trainers=sorted(trainers, key=lambda r: (r['pool'], r['name'])), local=local, error=None)

    def loop(self):
        while True:
            try:
                snap = self.collect()
            except Exception as exc:
                snap = dict(self.snapshot, error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-2000:])
            with self.lock:
                self.snapshot = snap
            time.sleep(REFRESH)


def fmt(v, digits=None):
    if v is None or v == '':
        return '-'
    if isinstance(v, float) and digits is not None:
        return f'{v:.{digits}f}'
    return html.escape(str(v))


def table(headers, rows):
    head = ''.join(f'<th>{html.escape(h)}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td class="{c}">{v}</td>' for c, v in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def health_class(h):
    return 'ok' if h == 'healthy' else ('warn' if h.startswith(('starting', 'no worker')) else 'bad')


def render(snap):
    age = f'{time.time() - snap["time"]:.0f}s ago (took {snap["seconds"]}s)' if snap.get('time') else 'not yet'
    parts = [f'<h1>Inference farms &amp; trainers</h1><p class="meta">Updated {age}. Refreshes every {REFRESH}s. '
             f'Host {html.escape(os.uname().nodename)}.</p>']
    if snap.get('error'):
        parts.append(f'<p class="bad">Collector: {html.escape(snap["error"])}</p>')
    pools = {}
    for f in snap.get('farms', []):
        pools.setdefault(f['pool'], []).append(f)
    parts.append('<h2>Farms by pool</h2>')
    for pool, items in pools.items():
        healthy = sum(f['health'] == 'healthy' for f in items)
        tok = sum(f['gen_tok_s'] or 0 for f in items)
        parts.append(f'<h3>{html.escape(pool)} <span class="meta">{len(items)} farms, {healthy} healthy, '
                     f'{tok:,} gen tok/s</span></h3>')
        parts.append(table(['model', 'farm', 'job', 'health', 'leased to', 'engines', 'gen tok/s', 'running',
                            'waiting', 'KV max %'],
                           [[('', fmt(f['model'])), ('', fmt(f['farm'])), ('num', fmt(f['job'])),
                             (health_class(f['health']), fmt(f['health'])),
                             ('free' if f['leased_to'] == 'free' else '', fmt(f['leased_to'])),
                             ('num', fmt(f['engines'])), ('num', fmt(f['gen_tok_s'])), ('num', fmt(f['running'])),
                             ('num', fmt(f['waiting'])), ('num', fmt(f['kv_max']))] for f in items]))
    parts.append('<h2>Trainers (managed jobs)</h2>')
    rows = []
    for t in snap.get('trainers', []):
        total = t.get('num_epochs')
        step = t.get('global_step', t.get('metric_rows'))
        progress = (f'{step}/{total}' if step is not None and total else fmt(step))
        if t.get('bootstrap_done') is False:
            progress = f'bootstrap ({t.get("bootstrap_groups", 0)} groups)'
        state = t['state']
        if state == 'RUNNING' and t.get('error'):
            state = f'RUNNING (unreachable: {t["error"]})'
        elif state == 'RUNNING' and t.get('found') is False:
            state = 'RUNNING (no run dir)'
        rows.append([('', fmt(t['pool'])), ('', fmt(t['name'])), ('num', fmt(t['job'])),
                     ('ok' if t['state'] == 'RUNNING' else 'warn', fmt(state)), ('num', fmt(t['worker'])),
                     ('num', fmt(progress)),
                     ('num', fmt(t.get('best'), 7)), ('num', fmt(t.get('correct'), 3)),
                     ('num', fmt(round(t['step_seconds'] / 60, 1) if t.get('step_seconds') else None)),
                     ('', fmt(t.get('last_event')))])
    parts.append(table(['pool', 'job name', 'job', 'state', 'worker', 'step', 'best value', 'correct',
                        'last step min', 'last event'], rows))
    if snap.get('local'):
        parts.append('<h2>RQ4 no-train runs (local, this host)</h2>')
        parts.append(table(['run', 'runner', 'step', 'bootstrap groups', 'best value', 'correct', 'farms held'],
                           [[('', fmt(r['run'])), ('ok' if r['alive'] else 'bad', 'up' if r['alive'] else 'DOWN'),
                             ('num', fmt(f'{r["step"]}/25' if r['step'] is not None else None)),
                             ('num', fmt(r['bootstrap_groups'])), ('num', fmt(r['best'], 7)),
                             ('num', fmt(r['correct'], 3)),
                             ('', fmt(', '.join(r['farms'])) if r['farms'] else '<span class="warn">none</span>')]
                            for r in snap['local']]))
    style = ('<style>:root{color-scheme:light dark;--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#e5e5ea;'
             '--ok:#1a7f37;--warn:#9a6700;--bad:#cf222e;--free:#0969da}'
             '@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#f2f2f7;--muted:#a1a1a6;--line:#2c2c2e;'
             '--ok:#3fb950;--warn:#d29922;--bad:#f85149;--free:#58a6ff}}'
             'body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:24px;}'
             'h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}h3{font-size:14px;margin:18px 0 6px}'
             '.meta{color:var(--muted);font-weight:normal}table{border-collapse:collapse;width:100%;margin-bottom:8px}'
             'th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}'
             'th{color:var(--muted);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}'
             '.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.free{color:var(--free)}</style>')
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="{REFRESH}">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>Farm status</title>{style}'
            f'</head><body>{"".join(parts)}</body></html>')


def serve(port, local_root=None, host='127.0.0.1'):
    collector = Collector(local_root)
    threading.Thread(target=collector.loop, name='dashboard-collector', daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            with collector.lock:
                snap = collector.snapshot
            if self.path.startswith('/api/status.json'):
                body, ctype = json.dumps(snap, indent=1).encode(), 'application/json'
            elif self.path in ('/', '/index.html'):
                body, ctype = render(snap).encode(), 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, name='dashboard-http', daemon=True).start()
    return server
