"""Move the bounded SkyPilot API + its controllers to another della head node.

State lives on GPFS and is shared, but processes are not: in consolidation mode
the API host runs every pool and managed-job controller. Flock on GPFS is
node-local and the DBs are SQLite in WAL mode, so the old host must be fully
stopped before the new one starts; no lock file protects across hosts.

Phases (each asserts which host it runs on):

  prep    [source] audit dir, GCP inventory, DB backups, target preflight
  lock    [source] hold ~/.sky/api_server/.creation.lock on the source host
  stop    [source] pause pool + jobs controllers, freeze the API tree, refuse
                   if a sky.exec is in flight (thaw everything); then SIGKILL
                   the API tree, every controller group and the nightly loop,
                   and verify nothing SkyPilot-side is left on the source
  db      [source] cancel stale non-launch rows, write the replay manifest
  start   [target] verify the source host is empty over ssh, start the
                   bounded API with the manifest; HA recovery respawns the
                   pool and jobs controllers on the target
  tunnel  [source] forward source 127.0.0.1:46580 -> target, so clients still
                   on the source host keep working unchanged
  resume  [target] release the source creation lock, start the nightly loop
  verify  [target] controllers, pools, active requests, jobs, GCP inventory

Rollback before `start` succeeds: run `start` on the source host instead (it
checks the target is empty the same way).
"""
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api_plain_restart as plain  # noqa: E402
import api_reattach_restart as ra  # noqa: E402
import api_vacuum_restart as base  # noqa: E402

SOURCE = 'della-vis2'
TARGET = 'della-milan'
PTR = Path('/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/api-restart/api_move_host.current')
base.PTR = ra.PTR = plain.PTR = PTR
log, audit, load_state, save_state = base.log, base.audit, base.load_state, base.save_state
NIGHTLY = base.STATE_ROOT / 'maintenance/nightly.sh'
TUNNEL_LOG = base.STATE_ROOT / 'api-restart/api-tunnel.log'
POOL_PORTS = (20001, 20002, 20003, 20004, 20005)
# Bracketed first letters keep pgrep from matching the ssh/bash running it.
SKY_PATTERNS = ('[s]ky.server.server', '[s]kypilot_preserving_restart',
                '[s]ky.serve.service', '[s]ky.jobs.controller',
                '[m]aintenance/nightly.sh')


def host():
    return socket.gethostname().split('.')[0]


def on(expected):
    assert host() == expected, f'run this phase on {expected}, not {host()}'


def remote_sky_processes(other):
    """SkyPilot API/controller/nightly processes of ours on `other`."""
    pattern = '|'.join(SKY_PATTERNS)
    r = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', other,
         f"pgrep -u $(id -u) -af '{pattern}' | cut -c1-160; echo __ok__"],
        capture_output=True, text=True, timeout=300)
    assert '__ok__' in r.stdout, f'could not inspect {other}: {r.stderr[-300:]}'
    return [l for l in r.stdout.splitlines() if l.strip() and l != '__ok__']


def local_sky_processes():
    pattern = '|'.join(SKY_PATTERNS)
    r = subprocess.run(['pgrep', '-u', str(os.getuid()), '-af', pattern],
                       capture_output=True, text=True)
    return [l for l in r.stdout.splitlines()
            if l.strip() and str(os.getpid()) != l.split()[0]]


def preflight_target():
    """Target-side checks that need no downtime."""
    r = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', TARGET,
         f'bash {base.REPO}/tpu/swarm/start_skypilot_bounded.sh --check && '
         f'ss -ltn | grep -cE ":(46580|{"|".join(map(str, POOL_PORTS))}) " || true'],
        capture_output=True, text=True, timeout=600)
    lines = r.stdout.strip().splitlines()
    assert r.returncode == 0 and lines, r.stderr[-500:]
    assert lines[-1].strip() == '0', f'ports busy on {TARGET}: {lines[-1]}'
    assert not remote_sky_processes(TARGET), f'SkyPilot already running on {TARGET}'
    log(f'{TARGET} preflight ok: {lines[-2] if len(lines) > 1 else lines[-1]}')


# ---------------------------------------------------------------- phases
def prep():
    on(SOURCE)
    ts = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    a = base.SKY / f'recovery-audit/api-move-host-{ts}'
    a.mkdir(parents=True)
    os.chmod(a, 0o2700)
    PTR.parent.mkdir(parents=True, exist_ok=True)
    PTR.write_text(str(a))
    log(f'audit dir {a}; {SOURCE} -> {TARGET}')
    assert base.api_root() is not None, 'no API on the source host'
    preflight_target()
    base.gcp_inventory('before')
    for name in ('state.db', 'spot_jobs.db', 'serve/services.db'):
        base.backup_db(base.SKY / name, a / (name.replace('/', '-') + '.before'))
    c = sqlite3.connect(f'file:{base.SKY}/spot_jobs.db?mode=ro', uri=True, timeout=60)
    jobs = dict(c.execute(
        "select status, count(distinct spot_job_id) from spot where status in "
        "('PENDING','SUBMITTED','STARTING','RUNNING','RECOVERING','CANCELLING',"
        "'FAILED_CONTROLLER_RECOVERING') group by status").fetchall())
    c.close()
    log(f'backed up state/spot_jobs/services; nonterminal jobs {jobs}')
    save_state(prep_done=True, source=SOURCE, target=TARGET, jobs_before=jobs)


def lock():
    on(SOURCE)
    base.lock()


def stop():
    on(SOURCE)
    s = load_state()
    assert s.get('prep_done') and psutil.pid_exists(s.get('lock_pid', -1)), 'prep+lock first'
    root = base.api_root()
    tree = [root] + root.children(recursive=True)
    ids = {p.pid for p in tree}
    pcs, jcs = base.pool_controllers(), base.jobs_controllers()
    assert len(pcs) == 5 and not ids & {p.pid for p in pcs + jcs}, len(pcs)
    groups = sorted({os.getpgid(p.pid) for p in pcs + jcs})
    assert os.getpgid(root.pid) not in groups
    paused = [{'pid': p.pid, 'created_at': p.create_time(), 'pgid': g}
              for g in groups for p in base.mine() if base._pgid(p) == g]
    try:
        ra._signal_groups(groups, signal.SIGSTOP)
        log(f'paused {len(groups)} controller groups ({len(paused)} procs)')
        api_procs = [{'pid': p.pid, 'created_at': p.create_time(), 'ppid': p.ppid()}
                     for p in tree if p.is_running()]
        save_state(paused=paused, api_processes=api_procs)
        for p in tree:
            try:
                p.send_signal(signal.SIGSTOP)
            except psutil.NoSuchProcess:
                pass
        time.sleep(1)
        c = sqlite3.connect(f'file:{base.REQ_DB}?mode=ro', uri=True, timeout=30)
        execs = c.execute("select count(*) from requests where name='sky.exec' and "
                          "status in ('RUNNING','WAITING')").fetchone()[0]
        c.close()
        if execs:
            raise base.Abort(f'{execs} sky.exec in flight')
    except BaseException as e:
        base.thaw_all(tree, groups)
        raise SystemExit(f'ABORTED, thawed, nothing changed: {e!r}') from e
    # Point of no return: kill the API first so nothing new is dispatched,
    # then every controller group (they cannot move; the target respawns them).
    for r in api_procs:
        try:
            psutil.Process(r['pid']).kill()
        except psutil.NoSuchProcess:
            pass
    ra._signal_groups(groups, signal.SIGKILL)
    for p in base.mine():
        if 'maintenance/nightly.sh' in base.cmdline(p):
            p.kill()
    for _ in range(60):
        left = local_sky_processes()
        if not left and not [r for r in api_procs if base._alive(r)] and base.health() != '200':
            break
        time.sleep(1)
    left = local_sky_processes()
    assert not left, f'still running on {SOURCE}: {left}'
    c = sqlite3.connect(f'file:{base.REQ_DB}?mode=ro', uri=True, timeout=30)
    rows = c.execute('select count(*) from requests').fetchone()[0]
    c.close()
    log(f'killed API tree ({len(api_procs)} procs), {len(groups)} controller groups and '
        f'the nightly loop; {SOURCE} has no SkyPilot processes; rows={rows}')
    save_state(stopped_at=time.time(), recovered_rows=rows)


def db():
    on(SOURCE)
    assert not local_sky_processes()
    plain.db()


def start():
    here = host()
    s = load_state()
    assert here in (s['target'], s['source']), here
    other = s['source'] if here == s['target'] else s['target']
    assert s.get('db_done') and base.health() != '200'
    left = remote_sky_processes(other)
    assert not left, f'refusing: SkyPilot still running on {other}: {left}'
    env = dict(base.ENV, SKYPILOT_RESTART_MANIFEST=str(audit() / 'manifest.json'))
    r = subprocess.run(['bash', str(base.REPO / 'tpu/swarm/start_skypilot_bounded.sh'), '--start'],
                       cwd=base.REPO, env=env, capture_output=True, text=True, timeout=600)
    (audit() / f'launcher-output-{here}.txt').write_text(r.stdout + r.stderr)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    logp = [l.split(': ', 1)[1] for l in r.stdout.splitlines() if l.startswith('Startup log:')][0]
    log(f'launched on {here}; log {logp}')
    for i in range(180):
        if base.health() == '200':
            break
        time.sleep(2)
    text = Path(logp).read_text(errors='replace')
    log(f'health={base.health()} after ~{2*i}s')
    for key in ('Preserved request database', 'Requeued'):
        line = [l for l in text.splitlines() if key in l]
        log(f'  {line[-1] if line else "MISSING: " + key}')
    assert base.health() == '200'
    save_state(started=True, start_log=logp, api_host=here)


def tunnel():
    """Forward the source host's API port to the API host for local clients."""
    s = load_state()
    on(s['source'])
    assert s.get('started') and s['api_host'] != s['source']
    assert base.health() != '200', 'something already listens on 46580 here'
    loop = ('while :; do ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes '
            '-o ServerAliveInterval=30 -o ServerAliveCountMax=3 '
            f'-L 127.0.0.1:46580:127.0.0.1:46580 {s["api_host"]}; '
            'echo "[$(date -Is)] tunnel exited rc=$?, retrying in 5s"; sleep 5; done')
    with open(TUNNEL_LOG, 'ab') as out:
        p = subprocess.Popen(['bash', '-c', loop], stdin=subprocess.DEVNULL, stdout=out,
                             stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(30):
        if base.health() == '200':
            break
        time.sleep(1)
    log(f'tunnel loop pid {p.pid} (log {TUNNEL_LOG}); health via {s["source"]} '
        f'localhost={base.health()}')
    assert base.health() == '200'
    save_state(tunnel_pid=p.pid)


def resume():
    s = load_state()
    on(s['api_host'])
    assert s.get('started') and base.health() == '200'
    (audit() / 'release-lock').write_text('done')
    for _ in range(60):
        if (audit() / 'lock-released').exists():
            log('source creation lock released')
            break
        time.sleep(1)
    else:
        log('WARNING: source lock holder has not confirmed release')
    assert not [p for p in base.mine() if 'maintenance/nightly.sh' in base.cmdline(p)]
    subprocess.Popen(['nohup', 'bash', str(NIGHTLY)], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    log(f'nightly maintenance loop started on {s["api_host"]}')
    save_state(resumed=True)


def verify():
    s = load_state()
    on(s['api_host'])
    root = base.api_root()
    log(f'API pid={root.pid if root else None} health={base.health()} '
        f'affinity={root.cpu_affinity() if root else None}')
    pcs, jcs = base.pool_controllers(), base.jobs_controllers()
    log(f'pool controllers {len(pcs)} (want 5), jobs controller procs {len(jcs)}')
    c = sqlite3.connect(f'file:{base.SKY}/serve/services.db?mode=ro', uri=True)
    for name, port in c.execute('select name,controller_port from services order by name'):
        r = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', '--max-time',
                            '8', f'http://127.0.0.1:{port}/autoscaler/info'],
                           capture_output=True, text=True)
        log(f'  {name}: autoscaler/info {r.stdout}')
    c.close()
    c = sqlite3.connect(f'file:{base.SKY}/spot_jobs.db?mode=ro', uri=True, timeout=60)
    jobs = dict(c.execute(
        "select status, count(distinct spot_job_id) from spot where status in "
        "('PENDING','SUBMITTED','STARTING','RUNNING','RECOVERING','CANCELLING',"
        "'FAILED_CONTROLLER_RECOVERING') group by status").fetchall())
    c.close()
    log(f'nonterminal jobs now {jobs}; before {s.get("jobs_before")}')
    rq = sqlite3.connect(f'file:{base.REQ_DB}?mode=ro', uri=True, timeout=60)
    log('active requests: ' + str(rq.execute(
        "select name,status,count(*) from requests where status in "
        "('PENDING','RUNNING','WAITING') group by name,status").fetchall()))
    rq.close()
    after = base.gcp_inventory('after')
    before = json.loads((audit() / 'gcp-before.json').read_text())
    for z in base.ZONES:
        b = {i['name']: i['createTime'] for i in before.get(z, [])}
        a = {i['name']: i['createTime'] for i in after[z]}
        log(f'  {z}: before={len(b)} after={len(a)} gone={len(set(b) - set(a))} '
            f'new={len(set(a) - set(b))}')
    other = s['source'] if s['api_host'] == s['target'] else s['target']
    log(f'{other} SkyPilot processes: {remote_sky_processes(other) or "none"}')


PHASES = {'prep': prep, 'lock': lock, 'stop': stop, 'db': db, 'start': start,
          'tunnel': tunnel, 'resume': resume, 'verify': verify}

if __name__ == '__main__':
    PHASES[sys.argv[1]]()
