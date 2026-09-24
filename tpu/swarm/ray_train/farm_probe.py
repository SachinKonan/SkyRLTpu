"""Stage-1 checks against one relaunched grading farm, driven from the login node.

Every call runs a small stdlib script on the farm head over the pool's SSH
config (the same transport the supervisor uses), hitting the ingress on
127.0.0.1:24800. Nothing here touches other jobs.

Actions:
  status     /health, /status, /v1/models, Ray resources on the head
  lease      acquire a lease (owner 'probe-<ts>') and print it
  capacity   /skyrl/v1/grading/capacity under the current probe lease
  grade      submit N AC2 candidates (fast, slow/timeout, crash), poll, print
  cancel     submit one long candidate, cancel it, check the unit is gone
  expire     hold a lease with a running grade, stop heartbeating, watch the farm
             cancel the work and return to 'unleased' without quarantine
  release    release the probe lease
The lease id is cached in --state so successive actions share it.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid

REMOTE = r'''
import json, subprocess, sys, time, urllib.request, urllib.error
P = PARAMS
def call(method, path, body=None, headers=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request('http://127.0.0.1:24800' + path, data=data, method=method,
        headers={'Content-Type': 'application/json', **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try: return e.code, json.load(e)
        except Exception: return e.code, {'raw': e.read().decode(errors='replace')[:400]}
H = {'X-Lease-ID': P.get('lease_id')} if P.get('lease_id') else {}
out = {}
a = P['action']
if a == 'status':
    out['health'] = call('GET', '/health')
    s, st = call('GET', '/status')
    st.pop('capabilities_contract', None)
    if isinstance(st.get('capabilities'), dict): st['capabilities'].pop('contract', None)
    out['status'] = (s, st)
    out['models'] = call('GET', '/v1/models')
    try:
        import glob, os
        # Without an explicit --ray, find the run's controller venv on the host.
        ray = os.path.expanduser(P['ray']) if P.get('ray') else next(
            iter(sorted(glob.glob(os.path.expanduser('~/.cache/*/envs/controller/bin/ray')))), 'ray')
        out['ray_status'] = subprocess.run([ray, 'status'], capture_output=True, text=True, timeout=30).stdout[-1500:]
    except Exception as e:
        out['ray_status'] = repr(e)
    out['units'] = subprocess.run(['systemctl', 'list-units', '--no-legend', 'ac2-grade-*'], capture_output=True, text=True).stdout
elif a == 'lease':
    out['acquire'] = call('POST', '/acquire_lease', {'owner_run': P['owner'], 'ttl_seconds': P.get('ttl', 300),
                                                    'compatibility_sha256': P.get('sha')})
elif a == 'renew':
    out['renew'] = call('POST', '/acquire_lease', {'owner_run': P['owner'], 'ttl_seconds': P.get('ttl', 300),
                                                  'lease_id': P['lease_id']})
elif a == 'capacity':
    out['capacity'] = call('GET', '/skyrl/v1/grading/capacity', headers=H)
elif a == 'submit':
    out['submit'] = [call('POST', '/skyrl/v1/grading/submit', body, headers=H) for body in P['bodies']]
elif a == 'result':
    out['result'] = [call('GET', '/skyrl/v1/grading/result/%s?wait=%d' % (rid, P.get('wait', 0)), headers=H, timeout=60)
                     for rid in P['ids']]
elif a == 'cancel':
    out['cancel'] = call('POST', '/skyrl/v1/grading/cancel/' + P['id'], headers=H)
    time.sleep(2)
    out['units'] = subprocess.run(['systemctl', 'list-units', '--no-legend', 'ac2-grade-*'], capture_output=True, text=True).stdout
elif a == 'units':
    out['units'] = subprocess.run(['systemctl', 'list-units', '--no-legend', 'ac2-grade-*'], capture_output=True, text=True).stdout
    out['status'] = call('GET', '/status')[1].get('state'), call('GET', '/status')[1].get('grading')
elif a == 'release':
    out['release'] = call('POST', '/release_lease', {'lease_id': P['lease_id']}, headers=H)
elif a == 'events':
    import glob, os
    paths = glob.glob(os.path.expanduser('~/.cache/*/runs/*/inference-events.jsonl'))
    lines = []
    for p in paths:
        with open(p) as f: lines += f.readlines()[-P.get('n', 40):]
    out['events'] = [json.loads(l) for l in lines if any(k in l for k in P.get('keys', ['grading', 'lease']))][-P.get('n', 40):]
print(json.dumps(out, default=str))
'''

CANDIDATES = {
    'fast': 'import numpy as np\ndef construct_function():\n    return list(np.linspace(0.1, 1.0, 1000))\n',
    'slow': 'import time\ndef construct_function():\n    time.sleep(%d)\n    return [1.0] * 1000\n',
    'crash': 'def construct_function():\n    raise ValueError("boom")\n',
    'chatty': 'def construct_function():\n    print("x" * 100000)\n    return [1.0] * 1000\n',
}


def rpc(args, params):
    config = Path(args.ssh_dir) / args.cluster
    proc = subprocess.run(['ssh', '-F', str(config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', args.cluster,
                           'python3 -'], input='PARAMS = ' + repr(params) + '\n' + REMOTE, text=True,
                          capture_output=True, timeout=args.timeout)
    if proc.returncode:
        raise SystemExit(f'ssh failed ({proc.returncode}): {proc.stderr[-800:]}')
    return json.loads(proc.stdout)


def body(kind, timeout=60, sleep=30):
    code = CANDIDATES[kind] % sleep if kind == 'slow' else CANDIDATES[kind]
    return dict(request_id=uuid.uuid4().hex, task='ac2', owner_run='probe', scope={'kind': kind},
                spec=dict(program_code=code, function_name='construct_function', eval_timeout_seconds=timeout,
                          admission_timeout_s=timeout))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['status', 'lease', 'renew', 'capacity', 'grade', 'cancel', 'expire',
                                           'release', 'events', 'units'])
    parser.add_argument('--cluster', required=True)
    parser.add_argument('--ssh-dir', required=True)
    parser.add_argument('--state', default='/tmp/farm-probe-state.json')
    parser.add_argument('--ray', default='', help="the run's controller ray binary; found on the host when omitted")
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--sha')
    parser.add_argument('--count', type=int, default=4)
    args = parser.parse_args()
    state = json.loads(Path(args.state).read_text()) if Path(args.state).exists() else {}
    lease_id = state.get('lease_id')
    owner = state.get('owner') or f'probe-{int(time.time())}'

    def show(result):
        print(json.dumps(result, indent=1, default=str))

    if args.action == 'status':
        show(rpc(args, dict(action='status', ray=args.ray)))
    elif args.action == 'lease':
        result = rpc(args, dict(action='lease', owner=owner, sha=args.sha))
        show(result)
        code, ack = result['acquire']
        if code == 200:
            state.update(lease_id=ack['lease_id'], owner=owner)
            Path(args.state).write_text(json.dumps(state))
    elif args.action == 'renew':
        show(rpc(args, dict(action='renew', owner=owner, lease_id=lease_id)))
    elif args.action == 'capacity':
        show(rpc(args, dict(action='capacity', lease_id=lease_id)))
    elif args.action == 'grade':
        bodies = [body('fast') for _ in range(args.count)] + [body('slow', timeout=8, sleep=30), body('crash'), body('chatty')]
        submitted = rpc(args, dict(action='submit', lease_id=lease_id, bodies=bodies))
        show(submitted)
        ids = [b['request_id'] for b in bodies]
        for _ in range(12):
            results = rpc(args, dict(action='result', lease_id=lease_id, ids=ids, wait=20))['result']
            states = [r[1].get('state') for r in results]
            print('states', states, flush=True)
            if all(s in ('done', 'failed', 'cancelled') for s in states):
                break
        for kind, (code, view) in zip(['fast'] * args.count + ['slow', 'crash', 'chatty'], results):
            result = view.get('result') or {}
            print(kind, code, view.get('state'), 'error=', (result.get('error') or '')[:80],
                  'len(result)=', len(result.get('result') or []) if isinstance(result.get('result'), list) else result.get('result'),
                  'stdout_bytes=', len(result.get('stdout') or ''), 'host=', view.get('host'),
                  'admission_wait=', (result.get('metrics') or {}).get('admission_wait_seconds'),
                  'candidate_s=', (result.get('metrics') or {}).get('candidate_seconds'))
    elif args.action == 'cancel':
        long = body('slow', timeout=600, sleep=500)
        show(rpc(args, dict(action='submit', lease_id=lease_id, bodies=[long])))
        time.sleep(8)
        print('before:', rpc(args, dict(action='units'))['units'])
        show(rpc(args, dict(action='cancel', lease_id=lease_id, id=long['request_id'])))
        show(rpc(args, dict(action='result', lease_id=lease_id, ids=[long['request_id']])))
    elif args.action == 'expire':
        # Short lease, long candidate, no renewals: the farm must cancel the
        # unit itself after the cancel grace and return to 'unleased'.
        result = rpc(args, dict(action='lease', owner=owner + '-expire', sha=args.sha, ttl=30))
        show(result)
        code, ack = result['acquire']
        if code != 200:
            raise SystemExit('could not acquire a short lease')
        short = ack['lease_id']
        long = body('slow', timeout=600, sleep=500)
        show(rpc(args, dict(action='submit', lease_id=short, bodies=[long])))
        for t in (10, 25, 40, 55):
            time.sleep(15 if t > 10 else 10)
            u = rpc(args, dict(action='units'))
            print(f't+{t}s state={u["status"][0]} grading={u["status"][1]} units={u["units"].strip()!r}', flush=True)
        show(rpc(args, dict(action='events', keys=['lease_inflight_cancelled', 'grading_cancel', 'quarantine'], n=10)))
        show(rpc(args, dict(action='lease', owner=owner + '-after', sha=args.sha, ttl=60)))
    elif args.action == 'release':
        show(rpc(args, dict(action='release', lease_id=lease_id)))
        Path(args.state).unlink(missing_ok=True)
    elif args.action == 'events':
        show(rpc(args, dict(action='events', n=40)))
    elif args.action == 'units':
        show(rpc(args, dict(action='units')))


if __name__ == '__main__':
    main()
