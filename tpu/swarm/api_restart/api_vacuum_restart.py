"""Bounded SkyPilot API restart + requests.db VACUUM (2026-10-04).

Follows tpu/swarm/SKYPILOT_BOUNDED_RESTART.md. One subcommand per phase so each
can be verified before the next:

  prep    audit dir, GCP inventory, consistent backups of state/spot_jobs/services
  lock    start a detached holder of ~/.sky/api_server/.creation.lock
  stop    pause pool controllers (SIGSTOP), kill idle jobs controllers (0 jobs),
          SIGKILL the audited API tree (no cancellation handlers run)
  db      cancel stale non-launch rows, write manifest, logical backup,
          in-place VACUUM with auto_vacuum=INCREMENTAL
  start   preserving start via the bounded launcher; wait for health + requeue
  resume  SIGCONT pool controllers, release the creation lock
  verify  post-restart checks

State for later phases is kept in <audit>/state.json.
"""
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import psutil

REPO = Path('/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost')
STATE_ROOT = Path('/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state')
HOME = STATE_ROOT / 'sky-home-v6e32'
SKY = HOME / '.sky'
REQ_DB = SKY / 'api_server/requests.db'
LOCK = SKY / 'api_server/.creation.lock'
PTR = Path('/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/api-restart/api_vacuum_restart.current')
OWNER = '7bfcb694'
ZONES = ('us-east5-a', 'us-east5-b', 'us-central2-b', 'us-central1-b')
GCLOUD = '/scratch/gpfs/ZHUANGL/sk7524/google-cloud-sdk/bin/gcloud'
ENV = dict(os.environ,
           HOME=str(HOME),
           SKYPILOT_CONFIG=str(STATE_ROOT / 'skypilot-config.yaml'),
           CLOUDSDK_CONFIG='/home/sk7524/.config/gcloud-tpuswarm-compute-sa-v6e32',
           GOOGLE_APPLICATION_CREDENTIALS=
           '/home/sk7524/.config/gcloud/vision-mix-compute-sa-key.json',
           SKYPILOT_DISABLE_LOCAL_API_SERVER='1',
           SKYPILOT_API_SERVER_ENDPOINT='http://127.0.0.1:46580')


def log(msg):
    print(f'{time.strftime("%H:%M:%SZ", time.gmtime())} {msg}', flush=True)


def audit():
    return Path(PTR.read_text().strip())


def load_state():
    p = audit() / 'state.json'
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(**kw):
    s = load_state()
    s.update(kw)
    (audit() / 'state.json').write_text(json.dumps(s, indent=1))


def cmdline(p):
    try:
        return ' '.join(p.cmdline())
    except (psutil.Error,):
        return ''


def mine():
    me = os.getpid()
    out = []
    for p in psutil.process_iter(['username']):
        if p.info['username'] == 'sk7524' and p.pid != me and p.pid != os.getppid():
            out.append(p)
    return out


def api_root():
    for p in mine():
        c = cmdline(p)
        if '--deploy' in c and 'port=46580' in c and (
                'sky.server.server' in c or 'skypilot_preserving_restart.py' in c):
            return p
    return None


def pool_controllers():
    return [p for p in mine()
            if cmdline(p).startswith(str(REPO)) and
            '-m sky.serve.service --service-name' in cmdline(p)]


def jobs_controllers():
    return [p for p in mine()
            if cmdline(p).startswith(str(REPO)) and
            '-m sky.jobs.controller' in cmdline(p).replace('-msky', '-m sky')]


class Abort(Exception):
    """A pre-kill check failed; nothing has been killed yet."""


def thaw_all(procs, pgids):
    """Best-effort SIGCONT of paused processes and groups; never raises."""
    for proc in procs:
        try:
            proc.send_signal(signal.SIGCONT)
        except psutil.Error:
            pass
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGCONT)
        except (ProcessLookupError, PermissionError):
            pass


def health():
    r = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}',
                        '--max-time', '5', 'http://127.0.0.1:46580/api/health'],
                       capture_output=True, text=True)
    return r.stdout


def nonterminal_jobs():
    c = sqlite3.connect(f'file:{SKY}/spot_jobs.db?mode=ro', uri=True, timeout=60)
    return c.execute(
        "select count(*) from spot where status in ('PENDING','SUBMITTED',"
        "'STARTING','RUNNING','RECOVERING','CANCELLING','FAILED_CONTROLLER_RECOVERING')"
    ).fetchone()[0]


def gcp_inventory(tag):
    out = {}
    for z in ZONES:
        r = subprocess.run([GCLOUD, 'compute', 'tpus', 'queued-resources', 'list',
                            f'--zone={z}', '--project=vision-mix', '--format=json'],
                           capture_output=True, text=True, env=ENV, timeout=200)
        assert r.returncode == 0, (z, r.stderr[-500:])
        items = json.loads(r.stdout or '[]')
        ours = [{'name': i['name'].rsplit('/', 1)[-1], 'state': i['state']['state'],
                 'createTime': i.get('createTime')}
                for i in items if f'-{OWNER}-' in i['name']]
        out[z] = ours
    (audit() / f'gcp-{tag}.json').write_text(json.dumps(out, indent=1))
    log('GCP ' + tag + ': ' + ', '.join(f'{z}={len(v)}' for z, v in out.items()))
    return out


def backup_db(src, dst):
    s = sqlite3.connect(f'file:{src}?mode=ro', uri=True, timeout=120)
    d = sqlite3.connect(dst)
    s.backup(d)
    d.close()
    s.close()


# ---------------------------------------------------------------- phases
def prep():
    ts = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    a = SKY / f'recovery-audit/api-vacuum-restart-{ts}'
    a.mkdir(parents=True)
    os.chmod(a, 0o2700)
    PTR.parent.mkdir(parents=True, exist_ok=True)
    PTR.write_text(str(a))
    log(f'audit dir {a}')
    assert api_root() is not None, 'no running API found'
    gcp_inventory('before')
    for name in ('state.db', 'spot_jobs.db', 'serve/services.db'):
        t = time.time()
        backup_db(SKY / name, a / (name.replace('/', '-') + '.before'))
        log(f'backed up {name} in {time.time()-t:.0f}s')
    save_state(prep_done=True)


def lock():
    holder = audit() / 'lock-holder.py'
    release = audit() / 'release-lock'
    holder.write_text(f'''
import filelock, os, time, pathlib
l = filelock.FileLock({str(LOCK)!r})
l.acquire(timeout=60)
pathlib.Path({str(audit() / 'lock-held')!r}).write_text(str(os.getpid()))
while not os.path.exists({str(release)!r}):
    time.sleep(1)
l.release()
pathlib.Path({str(audit() / 'lock-released')!r}).write_text(str(time.time()))
''')
    py = str(REPO / 'third_party/TPUSwarm/.venv/bin/python')
    p = subprocess.Popen([py, str(holder)], stdin=subprocess.DEVNULL,
                         stdout=open(audit() / 'lock-holder.log', 'ab'),
                         stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        if (audit() / 'lock-held').exists():
            log(f'creation lock held by pid {p.pid}')
            save_state(lock_pid=p.pid)
            return
        time.sleep(1)
    raise SystemExit('lock holder did not acquire the creation lock')


def stop():
    s = load_state()
    assert s.get('prep_done') and s.get('lock_pid'), 'run prep and lock first'
    assert psutil.pid_exists(s['lock_pid']), 'lock holder died'
    n = nonterminal_jobs()
    assert n == 0, f'{n} nonterminal managed jobs; refusing to kill jobs controllers'
    root = api_root()
    assert root is not None, 'no API'
    tree = [root] + root.children(recursive=True)
    api_procs = [{'pid': p.pid, 'created_at': p.create_time(), 'ppid': p.ppid()}
                 for p in tree if p.is_running()]
    pcs = pool_controllers()
    jcs = jobs_controllers()
    assert len(pcs) == 5, [cmdline(p) for p in pcs]
    tree_ids = {p.pid for p in tree}
    assert not tree_ids & {p.pid for p in pcs + jcs}
    pgids = sorted({os.getpgid(p.pid) for p in pcs})
    assert os.getpgid(root.pid) not in pgids
    paused = []
    try:
        for g in pgids:
            members = [p for p in mine() if _pgid(p) == g]
            os.killpg(g, signal.SIGSTOP)
            paused += [{'pid': p.pid, 'created_at': p.create_time(), 'pgid': g} for p in members]
        log(f'paused {len(pgids)} pool controller groups ({len(paused)} procs)')
        save_state(paused=paused, api_processes=api_procs)
        killed_jc = []
        for p in jcs:
            g = os.getpgid(p.pid)
            assert g != os.getpgid(root.pid) and g not in pgids
            kids = p.children(recursive=True)
            for q in [p] + kids:
                try:
                    q.kill()
                    killed_jc.append(q.pid)
                except psutil.NoSuchProcess:
                    pass
        log(f'killed {len(jcs)} idle jobs controllers ({len(killed_jc)} procs)')
        for rec in api_procs:
            try:
                psutil.Process(rec['pid']).kill()
            except psutil.NoSuchProcess:
                pass
        for _ in range(30):
            alive = [r['pid'] for r in api_procs if _alive(r)]
            if not alive and health() != '200':
                break
            time.sleep(1)
        alive = [r['pid'] for r in api_procs if _alive(r)]
        assert not alive, f'API processes still alive: {alive}'
    except BaseException as e:
        # If the API is still up, nothing is lost by aborting: resume the pool
        # controllers. (Killed jobs controllers were idle; the API respawns
        # them.) If the API is already dead, keep them paused so the restart
        # can continue with `db` and `start`.
        if any(_alive(r) for r in api_procs) or health() == '200':
            thaw_all([], pgids)
            raise SystemExit(f'ABORTED, controllers thawed: {e!r}') from e
        raise
    log(f'API tree killed ({len(api_procs)} procs); health now={health() or "down"}')
    save_state(stopped_at=time.time(), killed_jobs_controllers=killed_jc)


def _pgid(p):
    try:
        return os.getpgid(p.pid)
    except ProcessLookupError:
        return None


def _alive(rec):
    try:
        p = psutil.Process(rec['pid'])
        return p.create_time() == rec['created_at'] and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def db():
    s = load_state()
    assert s.get('stopped_at') and health() != '200'
    for rec in s['api_processes']:
        assert not _alive(rec), rec
    from sky.server import daemons  # pylint: disable=import-outside-toplevel
    dids = {d.id for d in daemons.INTERNAL_REQUEST_DAEMONS}
    c = sqlite3.connect(str(REQ_DB), timeout=60)
    c.row_factory = sqlite3.Row
    c.execute('BEGIN IMMEDIATE')
    rows = [dict(r) for r in c.execute(
        "select * from requests where status in ('PENDING','RUNNING','WAITING')")]
    stale = [r for r in rows if r['request_id'] not in dids and
             r['name'] not in ('sky.launch', 'sky.exec')]
    for r in stale:
        assert r['name'] in ('sky.jobs.queue_v2', 'sky.jobs.logs', 'sky.jobs.pool_status',
                             'sky.down', 'sky.status', 'sky.jobs.queue', 'sky.api_status',
                             'sky.serve.status', 'sky.jobs.pool_status_v2'), r['name']
    c.executemany("update requests set status='CANCELLED', finished_at=? where request_id=?",
                  [(time.time(), r['request_id']) for r in stale])
    c.commit()
    (audit() / 'stale-cancelled.json').write_text(json.dumps(
        [{k: r[k] for k in ('request_id', 'name', 'status', 'created_at', 'cluster_name')}
         for r in stale], indent=1))
    log(f'marked {len(stale)} stale non-launch rows CANCELLED: '
        + str(sorted({r["name"] for r in stale})))
    active = [dict(r) for r in c.execute(
        "select request_id,name,status,pid,cluster_name,user_id,created_at from requests "
        "where status in ('PENDING','RUNNING','WAITING')")]
    c.close()
    launches = [r for r in active if r['request_id'] not in dids]
    assert all(r['name'] in ('sky.launch', 'sky.exec') for r in launches)
    assert not [r for r in launches if r['name'] == 'sky.exec' and r['status'] != 'PENDING']
    manifest = {'home': str(HOME), 'requests': active,
                'api_processes': s['api_processes'],
                'controllers': s['paused'], 'snapshot_at': time.time()}
    (audit() / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    log(f'manifest: {len(launches)} launch/exec to replay, '
        f'{len(active)-len(launches)} daemon rows')
    # Logical backup first (free pages hold nothing), then compact in place.
    os.environ['SQLITE_TMPDIR'] = str(audit())
    c = sqlite3.connect(str(REQ_DB), timeout=60)
    before = {'bytes': REQ_DB.stat().st_size,
              'rows': c.execute('select count(*) from requests').fetchone()[0],
              'active': len(active)}
    t = time.time()
    c.execute('VACUUM INTO ?', (str(audit() / 'requests.db.logical-backup'),))
    log(f'logical backup written in {time.time()-t:.0f}s '
        f'({(audit()/"requests.db.logical-backup").stat().st_size/1e9:.2f} GB)')
    b = sqlite3.connect(str(audit() / 'requests.db.logical-backup'))
    assert b.execute('pragma integrity_check').fetchone()[0] == 'ok'
    assert b.execute('select count(*) from requests').fetchone()[0] == before['rows']
    b.close()
    t = time.time()
    c.execute('PRAGMA auto_vacuum=INCREMENTAL')
    c.execute('VACUUM')
    c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    log(f'in-place VACUUM done in {time.time()-t:.0f}s')
    after = {'bytes': REQ_DB.stat().st_size,
             'rows': c.execute('select count(*) from requests').fetchone()[0],
             'auto_vacuum': c.execute('pragma auto_vacuum').fetchone()[0],
             'integrity': c.execute('pragma integrity_check').fetchone()[0],
             'freelist': c.execute('pragma freelist_count').fetchone()[0]}
    c.close()
    assert after['rows'] == before['rows'] and after['integrity'] == 'ok'
    assert after['auto_vacuum'] == 2, after
    log(f'requests.db {before["bytes"]/1e9:.1f} GB -> {after["bytes"]/1e9:.2f} GB, '
        f'rows {after["rows"]}, auto_vacuum=INCREMENTAL, integrity ok')
    save_state(db_done=True, vacuum_before=before, vacuum_after=after)


def start():
    s = load_state()
    assert s.get('db_done') and health() != '200'
    assert psutil.pid_exists(s['lock_pid']), 'creation lock holder died'
    env = dict(ENV, SKYPILOT_RESTART_MANIFEST=str(audit() / 'manifest.json'))
    r = subprocess.run(['bash', str(REPO / 'tpu/swarm/start_skypilot_bounded.sh'), '--start'],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    (audit() / 'launcher-output.txt').write_text(r.stdout + r.stderr)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    logp = [l.split(': ', 1)[1] for l in r.stdout.splitlines() if l.startswith('Startup log:')][0]
    log(f'launched; log {logp}')
    for i in range(180):
        if health() == '200':
            break
        time.sleep(2)
    text = Path(logp).read_text(errors='replace')
    log(f'health={health()} after ~{2*i}s')
    for key in ('Preserved request database', 'Requeued'):
        line = [l for l in text.splitlines() if key in l]
        log(f'  {line[-1] if line else "MISSING: " + key}')
    assert health() == '200'
    save_state(started=True, start_log=logp)


def resume():
    s = load_state()
    assert s.get('started') and health() == '200'
    for g in sorted({r['pgid'] for r in s['paused']}):
        try:
            os.killpg(g, signal.SIGCONT)
        except ProcessLookupError:
            log(f'group {g} gone')
    log('pool controllers resumed')
    (audit() / 'release-lock').write_text('done')
    for _ in range(30):
        if (audit() / 'lock-released').exists():
            log('creation lock released')
            break
        time.sleep(1)
    save_state(resumed=True)


def verify():
    s = load_state()
    root = api_root()
    log(f'API pid={root.pid if root else None} health={health()} '
        f'affinity={root.cpu_affinity() if root else None}')
    tree = [root] + root.children(recursive=True)
    roles = {}
    for p in tree:
        c = cmdline(p)
        k = ('long' if 'executor:long' in c else 'short' if 'executor:short' in c else 'other')
        roles[k] = roles.get(k, 0) + 1
    log(f'API tree {roles}')
    stopped = [p for p in pool_controllers() if p.status() == psutil.STATUS_STOPPED]
    log(f'pool controllers {len(pool_controllers())}, stopped={len(stopped)}; '
        f'jobs controllers {len(jobs_controllers())}')
    c = sqlite3.connect(f'file:{SKY}/serve/services.db?mode=ro', uri=True)
    for name, port in c.execute('select name,controller_port from services order by name'):
        r = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', '--max-time',
                            '8', f'http://127.0.0.1:{port}/autoscaler/info'],
                           capture_output=True, text=True)
        log(f'  {name}: autoscaler/info {r.stdout}')
    rq = sqlite3.connect(f'file:{REQ_DB}?mode=ro', uri=True, timeout=60)
    log('active requests: ' + str(rq.execute(
        "select name,status,count(*) from requests where status in "
        "('PENDING','RUNNING','WAITING') group by name,status").fetchall()))
    log(f'requests.db {REQ_DB.stat().st_size/1e9:.2f} GB')
    after = gcp_inventory('after')
    before = json.loads((audit() / 'gcp-before.json').read_text())
    for z in ZONES:
        b = {i['name']: i['createTime'] for i in before[z]}
        a = {i['name']: i['createTime'] for i in after[z]}
        gone = sorted(set(b) - set(a))
        changed = [n for n in set(a) & set(b) if a[n] != b[n]]
        log(f'  {z}: before={len(b)} after={len(a)} gone={len(gone)} '
            f'new={len(set(a)-set(b))} recreated={len(changed)}')


if __name__ == '__main__':
    globals()[sys.argv[1]]()
