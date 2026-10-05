"""Recover explicitly identified failed pool workers without replacing TPUs.

Default is read-only inspection. --run bootstraps one existing slice through
SkyPilot's synchronous execution path, then hands readiness back to the pool.
It neither starts nor restarts an API/controller. Run with the durable API's
HOME, gcloud/ADC identity, configuration, and SKYPILOT_USER_ID=7bfcb694.
"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import pickle
import sqlite3
import subprocess
import time

POOL = 'tpuswarm-v5p32-east5a-erdos'
PROJECT = 'vision-mix'
ZONE = 'us-east5-a'
ACCOUNT = '289186856710-compute@developer.gserviceaccount.com'
HOME = Path('/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32')
NODES = {
    189: 'tpuswarm-v5p32-east5a-e-59-7bfcb694-head-ae0j8glu-tpu',
    197: 'tpuswarm-v5p32-east5a-e-9w-7bfcb694-head-9fls3fui-tpu',
    198: 'tpuswarm-v5p32-east5a-8gb6-7bfcb694-head-8arj4ny8-tpu',
}


class RecoveryRefused(BaseException):
    """Do not let provisioning exception handlers turn refusal into teardown."""


def require(condition, message):
    if not condition:
        raise RecoveryRefused(message)


def cloud_describe(kind, name):
    result = subprocess.run(
        ['gcloud', 'compute', 'tpus', kind, 'describe', name,
         f'--project={PROJECT}', f'--zone={ZONE}', '--format=json'],
        capture_output=True, text=True, timeout=45, check=True)
    return json.loads(result.stdout)


def validate_snapshot(worker, node, queued):
    expected = f'projects/{PROJECT}/locations/{ZONE}/nodes/{NODES[worker]}'
    require(node['name'] == expected, 'Unexpected TPU identity')
    require(bool(node.get('createTime')) and bool(queued.get('createTime')),
            'Missing provider creation timestamps')
    require(node.get('state') == 'READY' and node.get('health') == 'HEALTHY',
            'TPU is not READY/HEALTHY; leave it untouched')
    require(node.get('acceleratorType') == 'v5p-32', 'Wrong TPU topology')
    require(node.get('labels', {}).get('skypilot-user') == 'sk7524',
            'Unexpected TPU owner')
    require(len(node.get('networkEndpoints', [])) == 4, 'Expected four hosts')
    require(queued.get('state', {}).get('state') == 'ACTIVE',
            'Queued resource is not ACTIVE; leave it untouched')
    require(queued['name'].endswith('/queuedResources/' + NODES[worker] + '-q'),
            'Wrong queued resource')
    specs = queued.get('tpu', {}).get('nodeSpec', [])
    require(len(specs) == 1 and specs[0].get('nodeId') == NODES[worker],
            'Queued resource does not reference the expected TPU')


def snapshot(worker):
    node = cloud_describe('tpu-vm', NODES[worker])
    queued = cloud_describe('queued-resources', NODES[worker] + '-q')
    validate_snapshot(worker, node, queued)
    return node, queued


def connect(name, readonly=True):
    path = HOME / '.sky' / name
    return sqlite3.connect(f'file:{path}?mode={"ro" if readonly else "rw"}',
                           uri=True, timeout=15)


def replica_blob(worker):
    with connect('serve/services.db') as db:
        row = db.execute('SELECT replica_info FROM replicas WHERE '
                         'service_name=? AND replica_id=?', (POOL, worker)).fetchone()
    require(row is not None, 'Missing existing replica record')
    return row[0]


def check_no_operation(worker):
    with connect('api_server/requests.db') as db:
        rows = db.execute(
            "SELECT name,status FROM requests WHERE cluster_name=? "
            "AND status IN ('PENDING','RUNNING','WAITING')",
            (f'{POOL}-{worker}',)).fetchall()
    require(not rows, f'Existing operation on worker: {rows}')


def check_idle(node, key):
    script = '''import glob,json,subprocess
paths=glob.glob('/dev/vfio/*')+glob.glob('/dev/accel*')
assert paths, 'No TPU device paths'
r=subprocess.run(['sudo','-n','lsof','-t',*paths],capture_output=True,text=True)
assert not r.stderr.strip(), r.stderr
assert r.returncode in (0,1), r.returncode
print(json.dumps({'owners': r.stdout.split()}))
'''
    for endpoint in node['networkEndpoints']:
        ip = endpoint['accessConfig']['externalIp']
        alias = 'recovery-' + node['name'].split('/')[-1] + '-' + ip
        result = subprocess.run(
            ['ssh', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
             '-o', 'StrictHostKeyChecking=accept-new', '-o', f'HostKeyAlias={alias}',
             'gcpuser@' + ip, 'python3 -'], input=script,
            capture_output=True, text=True, timeout=30, check=True)
        require(not json.loads(result.stdout)['owners'],
                f'TPU is in use on {ip}; refusing bootstrap')


@contextlib.contextmanager
def existing_only(worker, original_node):
    from sky import provision
    from sky.provision import common, gcp, provisioner
    from sky.provision.gcp import instance, instance_utils

    def refuse(*args, **kwargs):
        del args, kwargs
        raise RecoveryRefused('Resource mutation refused: preserve existing TPU')

    def reuse(region, cluster_name, cluster_name_on_cloud, config):
        require(region == 'us-east5' and cluster_name == f'{POOL}-{worker}',
                'Unexpected cluster passed to provisioner')
        require(cluster_name_on_cloud == original_node['labels']['ray-cluster-name'],
                'Cloud cluster identity changed')
        require(config.count == 1, 'Unexpected slice count')
        require(config.provider_config['project_id'] == PROJECT and
                config.provider_config['availability_zone'] == ZONE,
                'Unexpected project or zone passed to provisioner')
        node, _ = snapshot(worker)
        require(node.get('createTime') == original_node.get('createTime'),
                'TPU was replaced during recovery')
        print('REUSE_EXISTING_TPU', NODES[worker], flush=True)
        return common.ProvisionRecord(
            provider_name='gcp', region=region, zone=ZONE,
            cluster_name=cluster_name_on_cloud,
            head_instance_id=NODES[worker], resumed_instance_ids=[],
            created_instance_ids=[])

    changes = []

    def replace(obj, name, value):
        changes.append((obj, name, name in vars(obj), vars(obj).get(name)))
        setattr(obj, name, value)

    try:
        replace(gcp, 'run_instances', reuse)
        replace(instance, 'run_instances', reuse)
        for module in [provision, gcp, instance]:
            for name in ['stop_instances', 'terminate_instances']:
                replace(module, name, refuse)
        replace(provisioner, 'teardown_cluster', refuse)
        for name in ['create_instances', 'start_instances', 'start_instance',
                     'stop', 'terminate', 'delete_queued_resource']:
            replace(instance_utils.GCPTPUVMInstance, name, staticmethod(refuse))
        yield
    finally:
        for obj, name, owned, old in reversed(changes):
            if owned:
                setattr(obj, name, old)
            else:
                delattr(obj, name)


def activate(worker, before):
    from sky.serve import serve_state
    from sky.utils import common_utils
    info = pickle.loads(before)
    require(info.cluster_name == f'{POOL}-{worker}', 'Replica identity mismatch')
    require(info.status == serve_state.ReplicaStatus.FAILED_CLEANUP,
            'Only FAILED_CLEANUP records may be recovered')
    info.status_property.sky_launch_status = common_utils.ProcessStatus.SUCCEEDED
    info.status_property.sky_down_status = None
    # Keep readiness false: the normal controller must independently probe job 1.
    info.status_property.service_ready_now = False
    info.status_property.first_ready_time = None
    info.first_not_ready_time = None
    info.consecutive_failure_times = []
    after = pickle.dumps(info)
    check_no_operation(worker)
    with connect('serve/services.db', readonly=False) as db:
        cursor = db.execute('UPDATE replicas SET replica_info=? WHERE '
                            'service_name=? AND replica_id=? AND replica_info=?',
                            (after, POOL, worker, before))
        require(cursor.rowcount == 1, 'Replica changed concurrently; not activated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('worker', type=int, choices=sorted(NODES))
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--ssh-key', required=True)
    args = parser.parse_args()
    require(Path.home() == HOME, 'Wrong durable HOME')
    require(os.environ.get('SKYPILOT_USER_ID') == '7bfcb694', 'Wrong SkyPilot user')
    require(os.environ.get('SKYPILOT_DISABLE_LOCAL_API_SERVER') == '1',
            'Local API auto-start must be disabled')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    import google.auth
    from google.auth.transport.requests import Request
    from sky import backends, execution, task as task_lib
    from sky.serve import serve_state
    import yaml

    credentials, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform'])
    require(project == PROJECT and
            getattr(credentials, 'service_account_email', None) == ACCOUNT,
            'Wrong ADC identity')
    credentials.refresh(Request())
    account = subprocess.run(
        ['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'],
        capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    require(account == ACCOUNT, 'gcloud and ADC identity mismatch')
    before = replica_blob(args.worker)
    info = pickle.loads(before)
    require(info.cluster_name == f'{POOL}-{args.worker}', 'Replica identity mismatch')
    require(info.status == serve_state.ReplicaStatus.FAILED_CLEANUP and
            not info.status_property.user_app_failed and
            not info.status_property.preempted and
            not info.status_property.purged, 'Not a recoverable outage record')
    check_no_operation(args.worker)
    node, queued = snapshot(args.worker)
    check_idle(node, args.ssh_key)
    with connect('state.db') as db:
        row = db.execute('SELECT autostop,to_down FROM clusters WHERE name=?',
                         (info.cluster_name,)).fetchone()
        require(row == (-1, 0), 'Missing cluster or enabled autostop')
        raw = db.execute('SELECT yaml FROM cluster_yaml WHERE cluster_name=?',
                         (info.cluster_name,)).fetchone()
    require(raw is not None, 'Missing original SkyPilot cluster YAML')
    require(yaml.safe_load(raw[0])['cluster_name'] ==
            node['labels']['ray-cluster-name'], 'Cluster handle identity mismatch')
    print('PREFLIGHT_OK', args.worker, node['name'], flush=True)
    if not args.run:
        return
    audit = HOME / '.sky' / 'recovery-audit' / f'{args.worker}-{time.time_ns()}'
    audit.mkdir(parents=True)
    (audit / 'replica-before.pickle').write_bytes(before)
    (audit / 'provider-before.json').write_text(json.dumps(
        {'node': node, 'queued': queued}, indent=2))
    with (HOME / '.sky' / 'existing-worker-recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(replica_blob(args.worker) == before, 'Replica changed after preflight')
        check_no_operation(args.worker)
        task_path = HOME / '.sky/serve' / POOL.replace('-', '_') / f'task_v{info.version}.yaml'
        task = task_lib.Task.from_yaml(str(task_path))
        with existing_only(args.worker, node):
            job_id, handle = execution.launch(
                task, cluster_name=info.cluster_name, retry_until_up=False,
                down=False, _is_launched_by_sky_serve_controller=True)
            backend = backends.CloudVmRayBackend()
            deadline = time.monotonic() + 600
            while True:
                statuses = backend.get_job_status(handle, [1], stream_logs=False)
                state = str(statuses.get(1))
                print('BOOTSTRAP_JOB', args.worker, job_id, state, flush=True)
                if state.endswith('SUCCEEDED'):
                    break
                require(not any(s in state for s in ['FAILED', 'CANCELLED']) and
                        time.monotonic() < deadline, 'Bootstrap job 1 did not succeed')
                time.sleep(10)
        after_node, after_queued = snapshot(args.worker)
        require(after_node.get('createTime') == node.get('createTime') and
                after_queued.get('createTime') == queued.get('createTime'),
                'Provider identity changed; do not reactivate')
        activate(args.worker, before)
        (audit / 'activated.json').write_text(json.dumps(
            {'worker': args.worker, 'job_id': job_id, 'time': time.time()}))
        print('BOOTSTRAPPED_AWAITING_CONTROLLER_PROBE', args.worker, flush=True)


if __name__ == '__main__':
    try:
        main()
    except RecoveryRefused as error:
        raise SystemExit(str(error))
