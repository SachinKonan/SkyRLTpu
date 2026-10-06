"""Reattach the orphaned requests.db and restart the bounded API (2026-10-05).

A stray client auto-start at 2026-10-04 21:40Z ran reset_db_and_logs: the live
requests.db was unlinked and an empty one created, while the running API kept
working on the deleted inode. This recovers the live file through /proc and
restarts with the preserving entrypoint. Reuses api_vacuum_restart.py phases
(prep, lock, start, resume, verify); replaces stop and db:

  stop  pause pool AND jobs controllers (live jobs exist), freeze the API tree,
        copy the deleted requests.db(+wal) out via /proc, validate the copy;
        only then SIGKILL the tree. On any validation failure everything is
        thawed and nothing has changed.
  db    move the empty file aside, install the recovered one, cancel stale
        non-launch rows, write the manifest (no full VACUUM: already
        auto_vacuum=INCREMENTAL; the nightly job trims it).
"""
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api_vacuum_restart as base  # noqa: E402

PTR = Path('/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/api-restart/api_reattach_restart.current')
base.PTR = PTR
log, audit, load_state, save_state = base.log, base.audit, base.load_state, base.save_state
REQ_DB = base.REQ_DB
OK_STALE = ('sky.jobs.queue_v2', 'sky.jobs.logs', 'sky.jobs.pool_status', 'sky.down',
            'sky.status', 'sky.jobs.queue', 'sky.check', 'sky.api_status',
            'sky.jobs.pool_status_v2', 'sky.serve.status')


def prep():
    ts = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    a = base.SKY / f'recovery-audit/api-reattach-restart-{ts}'
    a.mkdir(parents=True)
    os.chmod(a, 0o2700)
    PTR.parent.mkdir(parents=True, exist_ok=True)
    PTR.write_text(str(a))
    log(f'audit dir {a}')
    assert base.api_root() is not None
    base.gcp_inventory('before')
    for name in ('state.db', 'spot_jobs.db', 'serve/services.db'):
        base.backup_db(base.SKY / name, a / (name.replace('/', '-') + '.before'))
    log('backed up state/spot_jobs/services')
    save_state(prep_done=True)


def _deleted_fds(pid):
    db = wal = None
    for fd in os.listdir(f'/proc/{pid}/fd'):
        try:
            t = os.readlink(f'/proc/{pid}/fd/{fd}')
        except OSError:
            continue
        if t == f'{REQ_DB} (deleted)':
            db = fd
        elif t == f'{REQ_DB}-wal (deleted)':
            wal = fd
    return db, wal


def _signal_groups(pgids, sig):
    for g in pgids:
        try:
            os.killpg(g, sig)
        except ProcessLookupError:
            pass


def stop():
    s = load_state()
    assert s.get('prep_done') and psutil.pid_exists(s.get('lock_pid', -1)), 'prep+lock first'
    root = base.api_root()
    tree = [root] + root.children(recursive=True)
    tree_ids = {p.pid for p in tree}
    pcs, jcs = base.pool_controllers(), base.jobs_controllers()
    assert len(pcs) == 5 and len(jcs) >= 1
    assert not tree_ids & {p.pid for p in pcs + jcs}
    ctl_pgids = sorted({os.getpgid(p.pid) for p in pcs + jcs})
    assert os.getpgid(root.pid) not in ctl_pgids
    paused = []
    for g in ctl_pgids:
        members = [p for p in base.mine() if base._pgid(p) == g]
        paused += [{'pid': p.pid, 'created_at': p.create_time(), 'pgid': g} for p in members]
    _signal_groups(ctl_pgids, signal.SIGSTOP)
    log(f'paused {len(ctl_pgids)} controller groups (5 pool + {len(jcs)} jobs; {len(paused)} procs)')
    api_procs = [{'pid': p.pid, 'created_at': p.create_time(), 'ppid': p.ppid()}
                 for p in tree if p.is_running()]
    save_state(paused=paused, api_processes=api_procs)

    def thaw(reason):
        for p in tree:
            try:
                p.send_signal(signal.SIGCONT)
            except psutil.NoSuchProcess:
                pass
        _signal_groups(ctl_pgids, signal.SIGCONT)
        raise SystemExit(f'ABORTED, everything thawed, nothing changed: {reason}')

    for p in tree:
        try:
            p.send_signal(signal.SIGSTOP)
        except psutil.NoSuchProcess:
            pass
    time.sleep(1)
    log(f'froze API tree ({len(tree)} procs)')
    holder = None
    for p in tree:
        db, wal = _deleted_fds(p.pid)
        if db and wal:
            holder = (p.pid, db, wal)
            break
    if holder is None:
        thaw('no API process holds the deleted requests.db + wal')
    rec = audit() / 'recovered'
    rec.mkdir()
    pid, db, wal = holder
    shutil.copyfile(f'/proc/{pid}/fd/{db}', rec / 'requests.db')
    shutil.copyfile(f'/proc/{pid}/fd/{wal}', rec / 'requests.db-wal')
    log(f'copied deleted db+wal from pid {pid} '
        f'({(rec/"requests.db").stat().st_size/1e9:.2f} GB + '
        f'{(rec/"requests.db-wal").stat().st_size/1e6:.1f} MB wal)')
    # Validate on a scratch copy so the recovered pair stays byte-exact.
    chk = audit() / 'recovered-check'
    shutil.copytree(rec, chk)
    try:
        c = sqlite3.connect(str(chk / 'requests.db'), timeout=30)
        integ = c.execute('pragma integrity_check').fetchone()[0]
        rows = c.execute('select count(*) from requests').fetchone()[0]
        newest = c.execute('select max(created_at) from requests').fetchone()[0]
        exec_running = c.execute("select count(*) from requests where name='sky.exec' "
                                 "and status in ('RUNNING','WAITING')").fetchone()[0]
        bad = [r for r in c.execute(
            "select name from requests where status in ('PENDING','RUNNING','WAITING') "
            "and user_id != 'skypilot-system' and name not in ('sky.launch','sky.exec')")
               if r[0] not in OK_STALE]
        c.close()
    except Exception as e:  # pylint: disable=broad-except
        thaw(f'recovered copy unreadable: {e}')
    log(f'recovered copy: integrity={integ} rows={rows} newest {time.time()-newest:.0f}s old, '
        f'running exec={exec_running}')
    if integ != 'ok':
        thaw('integrity check failed')
    if time.time() - newest > 600:
        thaw('recovered copy is stale (newest row > 10 min old)')
    if exec_running:
        thaw(f'{exec_running} sky.exec RUNNING/WAITING; replaying could double-dispatch')
    if bad:
        thaw(f'unexpected active request types {sorted(set(r[0] for r in bad))}')
    shutil.rmtree(chk)
    for rec_ in api_procs:
        try:
            psutil.Process(rec_['pid']).kill()
        except psutil.NoSuchProcess:
            pass
    for _ in range(30):
        if not [r for r in api_procs if base._alive(r)] and base.health() != '200':
            break
        time.sleep(1)
    alive = [r['pid'] for r in api_procs if base._alive(r)]
    assert not alive, alive
    log(f'API tree killed ({len(api_procs)} procs); health={base.health() or "down"}')
    save_state(stopped_at=time.time(), recovered_rows=rows)


def db():
    s = load_state()
    assert s.get('stopped_at') and base.health() != '200'
    for r in s['api_processes']:
        assert not base._alive(r), r
    aside = audit() / 'empty-requests-db-replaced'
    aside.mkdir()
    for suf in ('', '-wal', '-shm'):
        f = Path(f'{REQ_DB}{suf}')
        if f.exists():
            shutil.move(str(f), aside / f.name)
    rec = audit() / 'recovered'
    shutil.copyfile(rec / 'requests.db', REQ_DB)
    shutil.copyfile(rec / 'requests.db-wal', f'{REQ_DB}-wal')
    os.chmod(REQ_DB, 0o640)
    c = sqlite3.connect(str(REQ_DB), timeout=60)
    c.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchall()
    assert c.execute('pragma integrity_check').fetchone()[0] == 'ok'
    rows = c.execute('select count(*) from requests').fetchone()[0]
    assert rows == s['recovered_rows'], (rows, s['recovered_rows'])
    log(f'installed recovered requests.db: {rows} rows, '
        f'{REQ_DB.stat().st_size/1e9:.2f} GB, auto_vacuum={c.execute("pragma auto_vacuum").fetchone()[0]}')
    c.close()
    from sky.server import daemons  # pylint: disable=import-outside-toplevel
    dids = {d.id for d in daemons.INTERNAL_REQUEST_DAEMONS}
    c = sqlite3.connect(str(REQ_DB), timeout=60)
    c.row_factory = sqlite3.Row
    c.execute('BEGIN IMMEDIATE')
    act = [dict(r) for r in c.execute(
        "select * from requests where status in ('PENDING','RUNNING','WAITING')")]
    stale = [r for r in act if r['request_id'] not in dids and
             r['name'] not in ('sky.launch', 'sky.exec')]
    assert all(r['name'] in OK_STALE for r in stale), {r['name'] for r in stale}
    c.executemany("update requests set status='CANCELLED', finished_at=? where request_id=?",
                  [(time.time(), r['request_id']) for r in stale])
    c.commit()
    log(f'cancelled {len(stale)} stale non-launch rows {sorted({r["name"] for r in stale})}')
    active = [dict(r) for r in c.execute(
        "select request_id,name,status,pid,cluster_name,user_id,created_at from requests "
        "where status in ('PENDING','RUNNING','WAITING')")]
    c.close()
    launches = [r for r in active if r['request_id'] not in dids]
    assert all(r['name'] in ('sky.launch', 'sky.exec') for r in launches)
    assert not [r for r in launches if r['name'] == 'sky.exec' and r['status'] != 'PENDING']
    assert all(r['user_id'] == base.OWNER and (r['cluster_name'] or '').startswith('tpuswarm-')
               for r in launches), 'foreign launch rows'
    (audit() / 'manifest.json').write_text(json.dumps(
        {'home': str(base.HOME), 'requests': active, 'api_processes': s['api_processes'],
         'controllers': s['paused'], 'snapshot_at': time.time()}, indent=1))
    log(f'manifest: {len(launches)} launch/exec to replay, {len(active)-len(launches)} daemon rows')
    save_state(db_done=True)


def verify():
    base.verify()
    root = base.api_root()
    n = d = 0
    for p in root.children(recursive=True):
        for fd in os.listdir(f'/proc/{p.pid}/fd'):
            try:
                t = os.readlink(f'/proc/{p.pid}/fd/{fd}')
            except OSError:
                continue
            if t.startswith(str(REQ_DB)):
                n += 1
                d += t.endswith('(deleted)')
    log(f'API children requests.db handles: {n}, on a deleted file: {d}')


PHASES = {'prep': prep, 'lock': base.lock, 'stop': stop, 'db': db,
          'start': base.start, 'resume': base.resume, 'verify': verify}

if __name__ == '__main__':
    PHASES[sys.argv[1]]()
