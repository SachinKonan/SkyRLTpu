"""Controller-local discovery and push; never claims leases or submits jobs.

Run with the existing SkyPilot environment and SSH configs. Only explicitly
listed training job IDs or opted-in jobs in listed pools may receive updates.
Farms are discovered by job name across pools, plus an optional legacy pool.
--dry-run performs no HTTP writes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import time


# Private TPU addresses are reachable from the TPU hosts, not necessarily from
# this controller. Use its existing SSH access for localhost control requests.
REMOTE = r'''
import ipaddress, json, urllib.request, urllib.error
def call(path, body=None):
    request = urllib.request.Request('http://127.0.0.1:'+str(PARAMS['port'])+path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)
try:
    if PARAMS['action'] == 'farm':
        call('/health')
        status = call('/status')
        count = status.get('expected_engines')
        if (type(count) is not int or count != 4 or status.get('exhausted')
                or status.get('updating') or len(status.get('replicas', [])) != count
                or status.get('state') not in ('unleased', 'ready', 'awaiting_adapter', 'expired')):
            raise ValueError('not a healthy lease-capable farm')
        names = [item['id'] for item in call('/v1/models')['data']
                 if item['id'] not in status.get('versions', [])]
        request = urllib.request.Request(
            'http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip',
            headers={'Metadata-Flavor': 'Google'})
        with urllib.request.urlopen(request, timeout=5) as response:
            ip = str(ipaddress.ip_address(response.read().decode().strip()))
        if ip not in status.get('expected', []):
            raise ValueError('farm head is not in the expected engine set')
        result = dict(models=names, url='http://'+ip+':'+str(PARAMS['port']),
                      owner_run=status.get('owner_run'), state=status.get('state'),
                      active=status.get('active', 0), capabilities=status.get('capabilities'))
    else:
        result = call('/skyrl/v1/borrowing/services', PARAMS.get('body'))
    print(json.dumps(dict(ok=True, result=result)))
except Exception as exc:
    print(json.dumps(dict(ok=False, error=type(exc).__name__,
                         http_status=exc.code if isinstance(exc, urllib.error.HTTPError) else None)))
'''


def rpc(ssh_dir, cluster, action, port, body=None):
    if not isinstance(cluster, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]+', cluster):
        return {'ok': False, 'error': 'invalid_cluster'}
    config = Path(ssh_dir) / cluster
    if not config.is_file():
        return {'ok': False, 'error': 'missing_ssh_config'}
    params = dict(action=action, port=port)
    if body is not None:
        params['body'] = body
    try:
        proc = subprocess.run(['ssh', '-F', str(config), '-o', 'BatchMode=yes',
                               '-o', 'ConnectTimeout=8', cluster, 'python3 -'],
            input='PARAMS = '+repr(params)+'\n'+REMOTE, text=True,
            capture_output=True, timeout=50)
        if proc.returncode:
            return {'ok': False, 'error': 'ssh_failed', 'exit_code': proc.returncode}
        return json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {'ok': False, 'error': type(exc).__name__}


def inventory():
    import sky
    rows, *_ = sky.get(sky.jobs.queue_v2(refresh=False, skip_finished=True,
        fields=['job_id', 'job_name', 'status', 'current_cluster_name', 'pool', 'priority']))
    return [dict(job_id=r.job_id, status=getattr(r.status, 'value', str(r.status)),
                 cluster=r.current_cluster_name, run_id=r.job_name, pool=r.pool,
                 priority=r.priority or 0) for r in rows]


def tick(rows, farm_pool, trainer_ids, call, *, dry_run=False, trainer_pools=(),
         run_scoped_only=False, farm_name_contains='inference-farm', farm_pool_max_job_id=None):
    """One replace-list update per target, using this tick's observed farms."""
    running = {r['job_id']: r for r in rows if r['status'] == 'RUNNING' and r.get('cluster')}
    farm_rows = [r for r in running.values()
                 if (farm_pool and re.fullmatch(re.escape(farm_pool)+r'-\d+', r['cluster'])
                     and (farm_pool_max_job_id is None or
                          (r.get('pool') == farm_pool and r['job_id'] <= farm_pool_max_job_id)))
                 or (farm_name_contains and farm_name_contains.casefold()
                     in (r.get('run_id') or '').casefold())]
    def probe(row):
        result = call(row['cluster'], 'farm')
        return row, result
    with ThreadPoolExecutor(max_workers=8) as workers:
        probes = list(workers.map(probe, farm_rows))
    farms = []
    unavailable = []
    for row, result in probes:
        if result.get('ok'):
            farms.append(dict(job_id=row['job_id'], **result['result']))
        else:
            unavailable.append(dict(job_id=row['job_id'], **result))

    trainer_ids = list(dict.fromkeys([*trainer_ids, *[r['job_id'] for r in rows
        if r.get('pool') in trainer_pools]]))
    descriptors = {}
    with ThreadPoolExecutor(max_workers=8) as workers:
        jobs = [job for job in trainer_ids if job in running]
        values = workers.map(lambda job: call(running[job]['cluster'], 'target'), jobs)
        descriptors.update(zip(jobs, values))
    reserving = {job: value['result'] for job, value in descriptors.items()
                 if value.get('ok') and value['result'].get('enabled') is True
                 and value['result'].get('run_id') == running[job].get('run_id')
                 and value['result'].get('lease_scope') == 'run'}
    from .farm_admission import assignments
    assigned = assignments([r for r in rows if r['job_id'] in trainer_ids], farms, reserving)

    def update(job_id):
        row = running.get(job_id)
        if not row:
            return dict(job_id=job_id, state='not_running')
        descriptor = descriptors[job_id]
        if not descriptor.get('ok'):
            return dict(job_id=job_id, state='unreachable_or_not_supported', detail=descriptor)
        target = descriptor['result']
        if target.get('run_id') != row.get('run_id') or not row.get('run_id'):
            return dict(job_id=job_id, state='target_identity_mismatch')
        if target.get('enabled') is not True:
            return dict(job_id=job_id, state='not_opted_in')
        if run_scoped_only and target.get('lease_scope') != 'run':
            return dict(job_id=job_id, state='legacy_scope_unchanged')
        urls = (assigned[job_id] if job_id in assigned else
                sorted({f['url'] for f in farms if target['model'] in f['models']})[:2])
        if target.get('urls') == urls:
            return dict(job_id=job_id, state='unchanged', model=target['model'], urls=urls)
        body = {k: target[k] for k in ('model', 'run_id', 'instance')}
        body['urls'] = urls
        if dry_run:
            return dict(job_id=job_id, state='would_update', model=target['model'], urls=urls)
        result = call(row['cluster'], 'target', body)
        return dict(job_id=job_id, state='updated' if result.get('ok') else 'update_failed',
                    model=target['model'], urls=urls, detail=result if not result.get('ok') else None)
    with ThreadPoolExecutor(max_workers=8) as workers:
        targets = list(workers.map(update, trainer_ids))
    return dict(farms=farms, unavailable=unavailable, targets=targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--farm-pool', help='Also discover farms in this legacy pool')
    parser.add_argument('--farm-pool-max-job-id', type=int,
                        help='Inclusive job-ID ceiling for the legacy pool exception only')
    parser.add_argument('--farm-name-contains', default='inference-farm',
                        help='Case-insensitive job-name substring across all pools (default: inference-farm)')
    parser.add_argument('--trainer-job-id', type=int, action='append', default=[])
    parser.add_argument('--trainer-pool', action='append', default=[])
    parser.add_argument('--ssh-config-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=24800)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--lock-file', type=Path)
    parser.add_argument('--run-scoped-only', action='store_true')
    args = parser.parse_args()
    if args.farm_pool_max_job_id is not None and (not args.farm_pool or args.farm_pool_max_job_id < 1):
        parser.error('--farm-pool-max-job-id requires --farm-pool and a positive job ID')
    if not args.trainer_job_id and not args.trainer_pool:
        parser.error('at least one explicit trainer job or pool is required')
    if not args.farm_pool and not args.farm_name_contains.strip():
        parser.error('a farm pool or nonempty farm name selector is required')
    if args.interval < 10 or not 1 <= args.port <= 65535:
        parser.error('interval must be >=10 seconds and port must be valid')
    # Prevent two supervisors under this login from competing to replace lists.
    import fcntl
    lock_path = args.lock_file or Path.home() / '.cache/skyrl/borrowing-supervisor.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('another borrowing supervisor is already running under this login')
    def call(cluster, action, body=None):
        return rpc(args.ssh_config_dir, cluster, action, args.port, body)
    while True:
        try:
            result = tick(inventory(), args.farm_pool, args.trainer_job_id, call,
                          dry_run=args.dry_run, trainer_pools=args.trainer_pool,
                          run_scoped_only=args.run_scoped_only,
                          farm_name_contains=args.farm_name_contains,
                          farm_pool_max_job_id=args.farm_pool_max_job_id)
            print(json.dumps(dict(time=time.time(), **result)), flush=True)
        except Exception as exc:
            # An inventory outage must not look like an empty farm pool. Keep
            # the last list; borrowers still independently verify every acquire.
            print(json.dumps(dict(time=time.time(), error=type(exc).__name__)), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
