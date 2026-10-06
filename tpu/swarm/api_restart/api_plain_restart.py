"""Plain bounded API restart with live jobs, to load new API-side code.

Same phases as api_reattach_restart.py, but the request DB is already the live
file: stop pauses pool + jobs controllers, freezes the API tree, refuses if a
sky.exec is in flight (thawing everything), then SIGKILLs; db cancels stale
non-launch rows and writes the manifest in place.
"""
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api_reattach_restart as ra  # noqa: E402
import api_vacuum_restart as base  # noqa: E402

PTR = Path('/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/api-restart/api_plain_restart.current')
base.PTR = ra.PTR = PTR
log, audit, load_state, save_state = base.log, base.audit, base.load_state, base.save_state


def prep():
    ts = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    a = base.SKY / f'recovery-audit/api-plain-restart-{ts}'
    a.mkdir(parents=True)
    os.chmod(a, 0o2700)
    PTR.parent.mkdir(parents=True, exist_ok=True)
    PTR.write_text(str(a))
    log(f'audit dir {a}')
    base.gcp_inventory('before')
    for name in ('state.db', 'spot_jobs.db', 'serve/services.db'):
        base.backup_db(base.SKY / name, a / (name.replace('/', '-') + '.before'))
    log('backed up state/spot_jobs/services')
    save_state(prep_done=True)


def stop():
    s = load_state()
    assert s.get('prep_done') and psutil.pid_exists(s.get('lock_pid', -1))
    root = base.api_root()
    tree = [root] + root.children(recursive=True)
    ids = {p.pid for p in tree}
    pcs, jcs = base.pool_controllers(), base.jobs_controllers()
    assert len(pcs) == 5 and not ids & {p.pid for p in pcs + jcs}
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
        # Resume everything if anything fails before the API is killed.
        base.thaw_all(tree, groups)
        raise SystemExit(f'ABORTED, thawed, nothing changed: {e!r}') from e
    for r in api_procs:
        try:
            psutil.Process(r['pid']).kill()
        except psutil.NoSuchProcess:
            pass
    for _ in range(30):
        if not [r for r in api_procs if base._alive(r)] and base.health() != '200':
            break
        time.sleep(1)
    assert not [r for r in api_procs if base._alive(r)]
    c = sqlite3.connect(f'file:{base.REQ_DB}?mode=ro', uri=True, timeout=30)
    rows = c.execute('select count(*) from requests').fetchone()[0]
    c.close()
    log(f'froze, no exec in flight, killed API tree ({len(api_procs)} procs); rows={rows}')
    save_state(stopped_at=time.time(), recovered_rows=rows)


def db():
    s = load_state()
    assert s.get('stopped_at') and base.health() != '200'
    from sky.server import daemons  # pylint: disable=import-outside-toplevel
    dids = {d.id for d in daemons.INTERNAL_REQUEST_DAEMONS}
    c = sqlite3.connect(str(base.REQ_DB), timeout=60)
    c.row_factory = sqlite3.Row
    c.execute('BEGIN IMMEDIATE')
    act = [dict(r) for r in c.execute(
        "select * from requests where status in ('PENDING','RUNNING','WAITING')")]
    stale = [r for r in act if r['request_id'] not in dids and
             r['name'] not in ('sky.launch', 'sky.exec')]
    assert all(r['name'] in ra.OK_STALE for r in stale), {r['name'] for r in stale}
    c.executemany("update requests set status='CANCELLED', finished_at=? where request_id=?",
                  [(time.time(), r['request_id']) for r in stale])
    c.commit()
    active = [dict(r) for r in c.execute(
        "select request_id,name,status,pid,cluster_name,user_id,created_at from requests "
        "where status in ('PENDING','RUNNING','WAITING')")]
    c.close()
    launches = [r for r in active if r['request_id'] not in dids]
    assert all(r['name'] in ('sky.launch', 'sky.exec') for r in launches)
    assert all(r['user_id'] == base.OWNER and (r['cluster_name'] or '').startswith('tpuswarm-')
               for r in launches)
    import json  # pylint: disable=import-outside-toplevel
    (audit() / 'manifest.json').write_text(json.dumps(
        {'home': str(base.HOME), 'requests': active, 'api_processes': s['api_processes'],
         'controllers': s['paused'], 'snapshot_at': time.time()}, indent=1))
    log(f'cancelled {len(stale)} stale rows {sorted({r["name"] for r in stale})}; '
        f'manifest: {len(launches)} launches to replay')
    save_state(db_done=True)


PHASES = {'prep': prep, 'lock': base.lock, 'stop': stop, 'db': db,
          'start': base.start, 'resume': base.resume, 'verify': ra.verify}

if __name__ == '__main__':
    PHASES[sys.argv[1]]()
