"""Check, launch and stop jobs on on-demand nodes over plain SSH: the v5p-64 (shu-v5p-64-on-demand-machine1)
and workers 0,1 of the on-demand v4-64 pool, where a trainer runs next to the pool's SkyPilot job.

These nodes are on-demand and must never be relinquished. This module only SSHes to their hosts as gcpuser,
installs, starts and stops our own systemd units there, and copies code archives to GCS. It never runs a
gcloud compute/tpu command, SkyPilot, or anything that could stop, reset, delete or recreate the VM.
Host addresses come from static maps (HOSTS_FILES). There is no preemption on an on-demand node, so the
only recovery is a process restart (the unit's Restart=on-failure and the runtime's own restarts).

Before anything is started, every host the job uses is checked with host_claim: other users' keys are on
this machine, and a host where another user's process holds the TPU (or a live claim by another run) is
refused. Stopping a job means stopping its unit on each host; nothing is killed by name. On the v4-64 the
pool job shares the hosts, so a launch is also refused unless the TPU devices are free, none of the job's ports
are in use and the hosts hold the job's memory plus PARTITION_MARGIN_GIB.

    python -m tpu.swarm.ray_train.assigned_launch check  <profile>
    python -m tpu.swarm.ray_train.assigned_launch launch <profile> --unit NAME --output DIR
    python -m tpu.swarm.ray_train.assigned_launch stop   --unit NAME --hosts IP [IP ...]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from . import host_claim
from .config import Config

HOSTS_FILES = tuple(Path(p) for p in os.environ.get('OD_HOSTS_FILE', ':'.join((
    '/scratch/gpfs/ZHUANGL/sk7524/rq4-local/od-v5p64-hosts.json',
    '/scratch/gpfs/ZHUANGL/sk7524/rq4-local/od-v4-64-hosts.json'))).split(':'))
KEY = '/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/clients/7bfcb694/ssh/sky-key'
USER = 'gcpuser'
UNIT_DIR = '.local/share/skyrl-od'
# MemAvailable left over once this job holds its cache and reserve: what the pool job may still grow into.
PARTITION_MARGIN_GIB = 32
# Free home disk: this job's environments, code and checkpoints share it with the pool job (no cleanup runs here).
PARTITION_DISK_GIB = 25  # measured: the OD box run dir incl. trainer env is ~6 GB per host (2026-11-02)


def external_ips():
    """Internal -> external address from the static host maps."""
    nodes = [json.loads(path.read_text()) for path in HOSTS_FILES if path.exists()]
    return {e['ipAddress']: e['accessConfig']['externalIp'] for node in nodes for e in node['networkEndpoints']}


def ssh(host, code, *, sudo=False, timeout=120):
    """Run a Python script on an internal host address; returns stdout."""
    remote = ('sudo -n python3 -' if sudo else 'python3 -')
    proc = subprocess.run(['ssh', '-i', KEY, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
                           '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=no',
                           '-o', 'UserKnownHostsFile=/dev/null', '-o', 'LogLevel=ERROR',
                           '%s@%s' % (USER, external_ips()[host]), remote],
                          input=code, text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError('ssh %s failed (%d): %s' % (host, proc.returncode, (proc.stderr or proc.stdout)[-800:]))
    return proc.stdout


def check(hosts, run_id):
    return host_claim.check_hosts(hosts, lambda h, script, sudo: ssh(h, script, sudo=sudo), run_id, USER)


GATE = r'''
import json, os, shutil, subprocess
from pathlib import Path
memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
devices = [str(p) for p in Path('/dev/vfio').glob('*') if p.name != 'vfio'] + [str(p) for p in Path('/dev').glob('accel*')]
owners = subprocess.run(['sudo', '-n', 'fuser', *devices], capture_output=True, text=True)
listening = subprocess.run(['ss', '-Hltn'], capture_output=True, text=True, check=True).stdout.splitlines()
ports = {int(line.split()[3].rsplit(':', 1)[1]) for line in listening}
print(json.dumps(dict(available_gib=memory['MemAvailable'] / 1024 ** 2, devices=len(devices),
                      disk_free_gib=shutil.disk_usage(Path.home()).free / 1024 ** 3,
                      device_owners=owners.stdout.split(), ports_in_use=sorted(ports & set(PARAMS['ports'])),
                      cpus=len(os.sched_getaffinity(0)))))
'''


def gate_verdict(probe, needed_gib):
    """Why the pool job's host cannot take this partition now, or None."""
    if not probe['devices']:
        return 'no TPU devices'
    if probe['device_owners']:
        return 'TPU devices are held by ' + ' '.join(probe['device_owners'])
    if probe['ports_in_use']:
        return 'ports in use: %s' % probe['ports_in_use'][:8]
    if probe['available_gib'] < needed_gib:
        return '%.1f GiB available, need %d' % (probe['available_gib'], needed_gib)
    if probe['disk_free_gib'] < PARTITION_DISK_GIB:
        return '%.1f GiB free on home, need %d' % (probe['disk_free_gib'], PARTITION_DISK_GIB)
    return None


def partition_gate(config):
    """{host: reason or None} and what each host reported; read-only."""
    from .bootstrap import workload_ports
    params = dict(ports=[*workload_ports(config), config.systemd_port])
    needed = config.cache.trainer_gib + config.cache.reserve_gib + PARTITION_MARGIN_GIB
    probes = {host: json.loads(ssh(host, 'PARAMS = %r\n' % params + GATE)) for host in config.assigned_host_ips}
    return {host: gate_verdict(probe, needed) for host, probe in probes.items()}, probes


INSTALL = r'''
import json, subprocess
from pathlib import Path
unit, text, script = PARAMS['unit'], PARAMS['unit_text'], PARAMS['script']
d = Path.home() / PARAMS['unit_dir'] / unit
d.mkdir(parents=True, exist_ok=True)
state = subprocess.run(['systemctl', 'is-active', unit], capture_output=True, text=True).stdout.strip()
if state not in ('inactive', 'failed', 'unknown', ''):
    raise SystemExit('unit %s is %s; stop it first' % (unit, state))
launch = d / 'launch.sh'
launch.write_text(script)
launch.chmod(0o700)
f = d / (unit + '.service')
f.write_text(text.replace('@LAUNCH@', str(launch)))
subprocess.run(['sudo', '-n', 'install', '-m', '644', str(f), '/etc/systemd/system/' + f.name], check=True)
subprocess.run(['sudo', '-n', 'systemctl', 'daemon-reload'], check=True)
subprocess.run(['sudo', '-n', 'systemctl', 'reset-failed', unit], capture_output=True)
print(json.dumps(dict(installed=unit)))
'''

UNIT_TEXT = '''[Unit]
Description={description}
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=3600
StartLimitBurst=3
[Service]
Type=exec
User=gcpuser
Group=gcpuser
WorkingDirectory=/home/gcpuser
Environment=HOME=/home/gcpuser
ExecStart=/bin/bash @LAUNCH@
Restart=on-failure
RestartSec=60
KillMode=control-group
TimeoutStopSec=1000
TasksMax=infinity
LimitNOFILE=1048576
LimitMEMLOCK=infinity
[Install]
WantedBy=multi-user.target
'''


def archive(profile, output):
    """Code archive for the profile, built exactly as for a managed job. tpu/science stays frozen here (farms hash
    it), and its packager has no direct mode on this branch, so science profiles stop at the guard in build()."""
    from .build import build
    path, uri, _ = build(profile, output, direct=True)
    return path, uri, hashlib.sha256(path.read_bytes()).hexdigest()


def scripts(config, profile_rel, unit, uri, digest):
    from .build import runtime_start
    from tpu.science.package_training import science_preparation
    from .regional_preflight import check_paths
    prep_env, prep = science_preparation(config)
    run = runtime_start(config, profile_rel).replace('exec python3 -m tpu.swarm.ray_train.bootstrap',
                                                     prep + 'exec python3 -m tpu.swarm.ray_train.bootstrap')
    out = []
    for rank in range(config.hosts):
        env = dict(prep_env, RAY_TRAIN_CODE=uri, RAY_TRAIN_CODE_SHA256=digest, SCIENCE_ACCELERATOR=config.accelerator,
                   SKYPILOT_NODE_IPS=' '.join(config.assigned_host_ips), SKYPILOT_NODE_RANK=str(rank),
                   SKYPILOT_TASK_ID=unit, SKYPILOT_DISABLE_LOCAL_API_SERVER='1')
        check_paths(config.to_dict(), config.effective_zone.rsplit('-', 1)[0], task=env)
        out.append('#!/bin/bash\nset -euo pipefail\nsudo -n prlimit --pid $$ --memlock=unlimited:unlimited\n'
                   + ''.join('export %s=%s\n' % (k, shlex.quote(str(v))) for k, v in env.items()) + run)
    return out


def launch(profile, unit, output, *, dry_run=False):
    profile = Path(profile).resolve()
    repo = Path(__file__).resolve().parents[3]
    config = Config.load(profile)
    config.validate()
    if not config.assigned_host_ips:
        raise ValueError('assigned_launch is only for assigned-host profiles')
    hosts = config.assigned_host_ips
    unknown = set(hosts) - set(external_ips())
    if unknown:
        raise ValueError('hosts not in the static map: %s' % sorted(unknown))
    path, uri, digest = archive(profile, output)
    texts = scripts(config, str(profile.relative_to(repo)), unit, uri, digest)
    record = dict(time=time.time(), run_id=config.run_id, unit=unit, hosts=hosts, uri=uri, sha256=digest,
                  profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                  script_sha256=[hashlib.sha256(t.encode()).hexdigest() for t in texts])
    Path(output, 'launch-record.json').write_text(json.dumps(record, indent=2) + '\n')
    for i, text in enumerate(texts):
        Path(output, 'launch-%d.sh' % i).write_text(text)
    if dry_run:
        return record
    if config.od_v4_64_partition:
        reasons = {host: reason for host, reason in partition_gate(config)[0].items() if reason}
        if reasons:
            raise SystemExit('the hosts cannot take this partition, refusing: %s' % json.dumps(reasons))
    claim = host_claim.claim_hosts(hosts, lambda h, script, sudo: ssh(h, script, sudo=sudo), config.run_id, unit,
                                   USER, time.time())
    if not claim['claimed']:
        raise SystemExit('hosts are held, refusing: %s' % json.dumps(claim['reasons']))
    if subprocess.run(['gcloud', 'storage', 'ls', uri], capture_output=True).returncode:
        subprocess.run(['gcloud', 'storage', 'cp', '--no-clobber', str(path), uri], check=True)
    text = UNIT_TEXT.format(description='SkyRL %s (%s)' % (config.run_id, unit))
    for host, script in zip(hosts, texts):
        ssh(host, 'PARAMS = %r\n' % dict(unit=unit, unit_text=text, script=script, unit_dir=UNIT_DIR) + INSTALL)
    start = 'import subprocess\nsubprocess.run(["sudo", "-n", "systemctl", "enable", "--now", %r], check=True)\n' % unit
    with ThreadPoolExecutor(len(hosts)) as pool:
        list(pool.map(lambda h: ssh(h, start), hosts))
    record['started'] = time.time()
    Path(output, 'launch-record.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


STOP = r'''
import json, subprocess
unit = PARAMS['unit']
subprocess.run(['sudo', '-n', 'systemctl', 'disable', '--now', unit], capture_output=True, text=True, timeout=1100)
subprocess.run(['sudo', '-n', 'systemctl', 'reset-failed', unit], capture_output=True)
state = subprocess.run(['systemctl', 'is-active', unit], capture_output=True, text=True).stdout.strip()
print(json.dumps(dict(unit=unit, state=state)))
'''


def stop(unit, hosts):
    """Stop (and disable) our unit on each host; the unit's control group takes every process with it."""
    with ThreadPoolExecutor(len(hosts)) as pool:
        return list(pool.map(lambda h: json.loads(ssh(h, 'PARAMS = %r\n' % dict(unit=unit) + STOP, timeout=1200)),
                             hosts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check')
    c.add_argument('profile')
    la = sub.add_parser('launch')
    la.add_argument('profile')
    la.add_argument('--unit', required=True)
    la.add_argument('--output', required=True)
    la.add_argument('--dry-run', action='store_true')
    s = sub.add_parser('stop')
    s.add_argument('--unit', required=True)
    s.add_argument('--hosts', nargs='+', required=True)
    args = parser.parse_args()
    if args.command == 'check':
        config = Config.load(args.profile)
        print(json.dumps(check(config.assigned_host_ips, config.run_id), indent=2))
        if config.od_v4_64_partition:
            print(json.dumps(partition_gate(config), indent=2))
    elif args.command == 'launch':
        Path(args.output).mkdir(parents=True, exist_ok=True)
        print(json.dumps(launch(args.profile, args.unit, args.output, dry_run=args.dry_run), indent=2))
    else:
        print(json.dumps(stop(args.unit, args.hosts), indent=2))


if __name__ == '__main__':
    main()
