"""storage_reconcile: reclaim other tasks' disk, never the current root, live paths, locks or journals."""
import fcntl
import json
import os
from types import SimpleNamespace

from tpu.swarm.ray_train import storage_reconcile as sr


def fill(path, size=4096):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'x' * size)
    return path


def manifest(run, **record):
    run.mkdir(parents=True, exist_ok=True)
    (run / sr.RUN_MANIFEST).write_text(json.dumps(dict(version=1, task_id='t-old', **record)))


def tree(tmp_path):
    home = tmp_path / 'home'
    cache = home / '.cache'
    current = cache / 'inference-farm-rq4-gemma-001'
    for root in (current, cache / 'old-farm', cache / 'old-trainer'):
        (root / 'owner.lock').parent.mkdir(parents=True, exist_ok=True)
        (root / 'owner.lock').touch()
        fill(root / 'envs/controller/bin/python')
        fill(root / 'sources/abc/.source-complete', 1)
    fill(current / 'runs/cur/client/state.db')
    fill(current / 'runs/cur/part.gstmp')  # current root: never touched, even partials
    fill(cache / 'old-farm/runs/farm/uploads/a.tar')
    fill(cache / 'old-farm/runs/farm/engine.log')
    manifest(cache / 'old-farm/runs/farm', run_id='farm', inference_only=True)
    fill(cache / 'old-trainer/runs/train/client/puct.json')
    fill(cache / 'old-trainer/runs/train/loras/v1/adapter.bin')
    manifest(cache / 'old-trainer/runs/train', run_id='train', inference_only=False)
    fill(cache / 'old-trainer/runs/train/client/request-journal.jsonl')
    fill(cache / 'old-trainer/base-download.tar.gz')
    fill(cache / 'skyrl-ray-code/aaa/code.tar.gz')
    fill(cache / 'skyrl-ray-code/cur/code.tar.gz')
    fill(cache / 'skyrl-ray-code/bbb/code.tar.gz.incomplete')
    fill(cache / 'uv/wheels/x.whl')
    return home, current


def paths(items):
    return {os.path.relpath(i['path'] if isinstance(i, dict) else i) for i in items}


def test_plan_tiers_current_root_and_journal_are_protected(tmp_path, monkeypatch):
    home, current = tree(tmp_path)
    monkeypatch.chdir((home / '.cache').resolve())
    items = sr.plan(home, current, 'cur', used=set(), mounts=set())
    got = {(i['tier'], os.path.relpath(i['path'])) for i in items}
    assert (1, 'skyrl-ray-code/bbb/code.tar.gz.incomplete') in got
    assert (2, 'skyrl-ray-code/aaa') in got and not any('skyrl-ray-code/cur' in p for _, p in got)
    assert (2, 'old-trainer/base-download.tar.gz') in got
    assert (3, 'old-farm/runs/farm/uploads') in got
    assert (4, 'old-farm/envs') in got and (4, 'old-trainer/sources') in got
    assert (5, 'old-farm/runs/farm') in got  # finished inference-only run: no durable state
    assert (6, 'uv') in got
    # A trainer run is never removed wholesale, and anything holding a client journal is kept.
    assert not any(p == 'old-trainer/runs/train' or 'journal' in p for _, p in got)
    assert (3, 'old-trainer/runs/train/loras') in got  # no journal inside loras
    assert not any(p.startswith('inference-farm-rq4-gemma-001') for _, p in got)


def test_live_paths_held_locks_and_tmpfs_mounts_are_skipped(tmp_path, monkeypatch):
    home, current = tree(tmp_path)
    cache = home / '.cache'
    used = {str(cache / 'skyrl-ray-code/aaa/code.tar.gz')}  # a live process has it open
    mounts = {str(cache / 'old-farm/envs')}                 # a tmpfs mount point
    fd = os.open(cache / 'old-trainer/owner.lock', os.O_RDWR)
    pid = os.fork()
    if pid == 0:  # another process holds the old trainer root's bootstrap lock
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.read(os.open(os.devnull, os.O_RDONLY), 1)
        import time
        time.sleep(30)
        os._exit(0)
    try:
        import time
        deadline = time.time() + 10
        while not sr.lock_is_held(cache / 'old-trainer/owner.lock') and time.time() < deadline:
            time.sleep(0.05)
        monkeypatch.chdir(cache.resolve())
        got = paths(sr.plan(home, current, 'cur', used=used, mounts=mounts))
        assert 'skyrl-ray-code/aaa' not in got
        assert 'old-farm/envs' not in got
        assert not any(p.startswith('old-trainer/') and p != 'old-trainer/base-download.tar.gz' for p in got)
    finally:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        os.close(fd)


def test_dry_run_deletes_nothing_and_logs_each_item(tmp_path):
    home, current = tree(tmp_path)
    log = tmp_path / 'events.jsonl'
    before = sorted(str(p) for p in home.rglob('*'))
    report = sr.reconcile(home, current, 'cur', target_free_bytes=10**18, startup=True, dry_run=True, log=log,
                          used=set(), mounts=set())
    assert sorted(str(p) for p in home.rglob('*')) == before
    assert report['dry_run'] and report['removed'] and report['freed_bytes'] > 0
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert {e['event'] for e in events} == {'storage_item_planned', 'storage_reconciled'}
    assert all({'item', 'reason', 'bytes', 'tier'} <= set(e) for e in events if e['event'] == 'storage_item_planned')


def test_startup_always_clears_dead_items_but_keeps_reusable_ones_when_disk_is_fine(tmp_path):
    home, current = tree(tmp_path)
    plenty = lambda _: SimpleNamespace(free=10**15)
    report = sr.reconcile(home, current, 'cur', target_free_bytes=10, startup=True, used=set(), mounts=set(),
                          disk_usage=plenty)
    cache = home / '.cache'
    assert not (cache / 'skyrl-ray-code/aaa').exists() and not (cache / 'old-farm/runs/farm/uploads').exists()
    assert (cache / 'old-farm/envs').exists() and (cache / 'uv').exists()  # reusable, and not short
    assert (cache / 'old-trainer/runs/train/client/request-journal.jsonl').exists()
    assert (current / 'runs/cur/part.gstmp').exists()
    assert report['stopped_early'] is None


def test_short_disk_reclaims_reusable_tiers_until_target(tmp_path):
    home, current = tree(tmp_path)
    free = {'n': 0}
    def usage(_):
        return SimpleNamespace(free=free['n'])
    removed = []
    real_remove = sr.remove
    def remove(path):
        removed.append(path)
        real_remove(path)
        if str(path).endswith('/envs'):
            free['n'] = 10**15  # first rebuildable env freed enough
    sr_remove, sr.remove = sr.remove, remove
    try:
        sr.reconcile(home, current, 'cur', target_free_bytes=10**12, used=set(), mounts=set(), disk_usage=usage)
    finally:
        sr.remove = sr_remove
    assert any(str(p).endswith('/envs') for p in removed)
    assert (home / '.cache/uv').exists()  # target met before the download cache


def test_time_budget_and_stop_request_end_a_pass_early(tmp_path):
    home, current = tree(tmp_path)
    report = sr.reconcile(home, current, 'cur', target_free_bytes=10**18, startup=True, used=set(), mounts=set(),
                          stopping=lambda: True)
    assert report['stopped_early'] == 'stopping' and not report['removed']
    report = sr.reconcile(home, current, 'cur', target_free_bytes=10**18, startup=True, used=set(), mounts=set(),
                          timeout=-1)
    assert report['stopped_early'] == 'timeout'


def test_live_paths_reads_proc(tmp_path):
    proc = tmp_path / 'proc'
    me = proc / '42'
    (me / 'fd').mkdir(parents=True)
    os.symlink('/data/old/run', me / 'cwd')
    os.symlink('/data/x.tar', me / 'fd' / '3')
    (me / 'maps').write_text('7f00-7f01 r-xp 0 00:00 1 /data/lib.so\n')
    (me / 'cmdline').write_bytes(b'python\0/data/script.py\0--flag\0')
    used = sr.live_paths(proc=proc, uid=os.getuid(), me=1)
    assert {'/data/old/run', '/data/x.tar', '/data/lib.so', '/data/script.py'} <= used


def test_mount_points_are_tmpfs_only(tmp_path):
    mounts = tmp_path / 'mounts'
    mounts.write_text('/dev/sda1 / ext4 rw 0 0\ntmpfs /home/u/.cache/r/ram tmpfs rw 0 0\n/dev/sdb /home ext4 rw 0 0\n')
    assert sr.mount_points(mounts) == {'/home/u/.cache/r/ram'}


def bare_host(tmp_path, **config):
    import threading
    import time
    from tpu.swarm.ray_train.host import Host
    host = Host.__new__(Host)
    host.config = SimpleNamespace(storage_min_free_gib=30, storage_reconcile_timeout=5, run_id='r',
                                  inference_only=True, retired_task_ids=[], retired_processes={},
                                  checkpoint_cleanup_timeout=0, **config)
    host.root = tmp_path / 'root'
    host.run = host.root / 'runs/r'
    host.run.mkdir(parents=True)
    host.log = host.run / 'host-0.jsonl'
    host.rank, host.stopping, host.lock = 0, threading.Event(), threading.RLock()
    host.last_storage_check, host.storage_thread = time.monotonic() - 1000, None
    host.role = host.phase = host.last_sync = host.sync_error = None
    host.processes, host.last_heartbeat = {}, time.monotonic()
    host.gcs = None
    return host


def test_one_storage_entry_point_reclaims_verified_checkpoints_then_disk(tmp_path, monkeypatch):
    from tpu.swarm.ray_train.host import Host
    import tpu.swarm.ray_train.checkpoint_retention as cr
    host = bare_host(tmp_path)
    calls = []
    monkeypatch.setattr(cr, 'reclaim_checkpoints', lambda *a, **kw: calls.append('checkpoints'))
    monkeypatch.setattr(sr, 'reconcile', lambda *a, **kw: calls.append(('disk', kw['startup'])) or {})
    Host.reconcile_storage(host, startup=True)
    assert calls == ['checkpoints', ('disk', True)]
    calls.clear()
    host.config.storage_min_free_gib = 0  # disk reclamation off: verified checkpoint reclaim still runs
    Host.reconcile_storage(host, startup=False)
    assert calls == ['checkpoints']


def test_run_manifest_records_inference_only(tmp_path):
    import tpu.swarm.ray_train.checkpoint_retention as cr
    config = SimpleNamespace(run_id='farm', run_gcs='gs://b/farm', inference_only=True)
    fd = cr.acquire_run_lease(config, tmp_path)
    try:
        record = json.loads((tmp_path / 'runs/farm' / cr.MANIFEST).read_text())
        assert record['inference_only'] is True
    finally:
        os.close(fd)


def test_heartbeat_starts_a_background_pass_only_when_disk_is_low(tmp_path, monkeypatch):
    from tpu.swarm.ray_train import host as host_mod
    host = bare_host(tmp_path)
    ran = []
    host.reconcile_storage = lambda startup: ran.append(startup)
    monkeypatch.setattr(host_mod.shutil, 'disk_usage', lambda p: SimpleNamespace(free=10**15))
    host._maybe_reclaim_storage()
    assert ran == []
    host.last_storage_check -= 1000
    monkeypatch.setattr(host_mod.shutil, 'disk_usage', lambda p: SimpleNamespace(free=1))
    host._maybe_reclaim_storage()
    host.storage_thread.join(5)
    assert ran == [False]
    host._maybe_reclaim_storage()  # rate limited: no second pass within the interval
    assert ran == [False]
