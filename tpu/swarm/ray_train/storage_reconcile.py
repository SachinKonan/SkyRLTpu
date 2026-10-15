"""Reclaim host disk that no live job can use: a manifest of what is in use, fast deletion of the rest.

Runs at job start (bootstrap preflight, so it also runs when a preempted job recovers on a new attempt) and
from the host heartbeat when free disk falls below a threshold. Pool workers run one job at a time, so
everything outside the current job's manifest belongs to a finished, cancelled or preempted task.

In use, never touched:
- the current executor root's code bundle, controller env, source tree and run directory;
- any path a live process of ours holds open (cwd, open file, mapped file) or names on its command line;
- any executor root whose bootstrap lock (owner.lock) or run lease (.checkpoint-owner.lock) is held;
- tmpfs mounts (model weights, compile caches): cache_admission.py owns those;
- local checkpoint archives: checkpoint_retention.py deletes them only after verifying the GCS copy;
- client/PUCT state, databases and logs of trainer runs, unless the run manifest says inference_only.

Reclaimable, in this order (cheapest to recreate first). Tiers 1-3 are dead and always go at startup; tiers
4-6 can be reused, so they go only while free disk is below the target:
1. partial transfers: *.gstmp, *.incomplete, *-partial staging directories;
2. code bundles of other tasks (~/.cache/skyrl-ray-code/<sha> other than the current one) and downloaded base
   archives whose source tree is complete;
3. derived serving artifacts of other runs: runs/*/loras, runs/*/uploads, runs/*/*.reload.tar;
4. controller environments and source trees of other, idle executor roots (rebuilt by bootstrap);
5. whole run directories of other roots whose run manifest (.checkpoint-owner.json, written by
   checkpoint_retention.acquire_run_lease: the one record of what a run is) says inference_only;
6. the uv download cache, only if still short of the target.
`dry_run=True` returns the same plan without deleting anything.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import time

PARTIAL_SUFFIXES = ('.gstmp', '.incomplete')
# Client journals (in-flight request/lease records) must survive restarts: nothing named like one, and no
# directory containing one, is ever reclaimed.
JOURNAL_MARKER = 'journal'
OWNER_LOCKS = ('owner.lock',)
RUN_LEASE = '.checkpoint-owner.lock'
RUN_MANIFEST = '.checkpoint-owner.json'
ALWAYS_AT_STARTUP = 3


def size_of(path):
    total = 0
    try:
        info = path.lstat()
    except FileNotFoundError:
        return 0
    if stat.S_ISDIR(info.st_mode):
        for dirpath, dirnames, filenames in os.walk(path, onerror=lambda _: None):
            for name in filenames:
                try:
                    total += os.lstat(os.path.join(dirpath, name)).st_blocks * 512
                except OSError:
                    pass
        return total
    return info.st_blocks * 512


def live_paths(proc=Path('/proc'), uid=None, me=None):
    """Resolved paths our live processes use: cwd, open files, mapped files, path-like arguments."""
    uid = os.getuid() if uid is None else uid
    me = os.getpid() if me is None else me
    used = set()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            used.add(os.readlink(entry / 'cwd'))
            for fd in (entry / 'fd').iterdir():
                try:
                    used.add(os.readlink(fd))
                except OSError:
                    pass
            for line in (entry / 'maps').read_text().splitlines():
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].startswith('/'):
                    used.add(parts[5].replace(' (deleted)', ''))
            for arg in (entry / 'cmdline').read_bytes().split(b'\0'):
                if arg.startswith(b'/'):
                    used.add(arg.decode(errors='replace'))
        except (FileNotFoundError, PermissionError, ProcessLookupError, NotADirectoryError):
            continue
    return {p for p in used if p.startswith('/')}


def is_journal(name):
    # Source modules (generation_journal.py) are code, not journals: otherwise every code bundle is kept forever.
    return JOURNAL_MARKER in name and not name.endswith(('.py', '.pyc'))


def holds_journal(path):
    if is_journal(path.name):
        return True
    if not path.is_dir() or path.is_symlink():
        return False
    for _, dirnames, filenames in os.walk(path, onerror=lambda _: None):
        if any(is_journal(n) for n in filenames) or any(JOURNAL_MARKER in n for n in dirnames):
            return True
    return False


def lock_is_held(path):
    """True if another process holds the flock (or the lock cannot be inspected safely)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def mount_points(mounts_file=Path('/proc/self/mounts')):
    """tmpfs mount points (model weights, compile caches); cache_admission.py, not this module, owns them."""
    try:
        lines = mounts_file.read_text().splitlines()
    except OSError:
        return set()
    return {f[1] for f in (line.split() for line in lines) if len(f) > 2 and f[2] == 'tmpfs'}


class Planner:
    def __init__(self, home, current_root, current_code, *, used, mounts):
        self.home = Path(home).resolve()
        self.current_root = Path(current_root).resolve() if current_root else None
        self.current_code = current_code
        self.used = used
        self.mounts = mounts
        self.items = []
        self.seen = set()

    def busy(self, path):
        """Held open, named by a live process, a mount (or containing one), or inside the current root."""
        p = str(path)
        if self.current_root and (path == self.current_root or path.is_relative_to(self.current_root)):
            return True
        if any(u == p or u.startswith(p + '/') for u in self.used):
            return True
        return any(m == p or m.startswith(p + '/') or p.startswith(m + '/') for m in self.mounts)

    def add(self, tier, path, reason):
        path = Path(path)
        if path in self.seen or path.is_symlink() or not path.exists():
            return
        try:
            if path.lstat().st_uid != os.getuid():
                return
        except FileNotFoundError:
            return
        if self.busy(path) or holds_journal(path):
            return
        self.seen.add(path)
        self.items.append(dict(tier=tier, path=str(path), reason=reason, bytes=size_of(path)))


def executor_roots(cache):
    """Executor roots under ~/.cache: directories with a bootstrap owner.lock or a runs/ tree."""
    if not cache.is_dir():
        return []
    roots = []
    for child in sorted(cache.iterdir()):
        if child.is_symlink() or not child.is_dir():
            continue
        if (child / 'owner.lock').exists() or (child / 'runs').is_dir() and (child / 'envs').is_dir():
            roots.append(child)
    return roots


def root_is_idle(root):
    if any(lock_is_held(root / name) for name in OWNER_LOCKS):
        return False
    runs = root / 'runs'
    if runs.is_dir():
        for run in runs.iterdir():
            if not run.is_symlink() and lock_is_held(run / RUN_LEASE):
                return False
    return True


def plan(home, current_root=None, current_code=None, *, used=None, mounts=None):
    home = Path(home).resolve()
    cache = home / '.cache'
    used = live_paths() if used is None else used
    mounts = mount_points() if mounts is None else mounts
    p = Planner(home, current_root, current_code, used=used, mounts=mounts)
    roots = executor_roots(cache)
    code_cache = cache / 'skyrl-ray-code'

    # 1. Partial transfers and staging anywhere in our caches.
    for base in [code_cache, *roots]:
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base, onerror=lambda _: None):
            here = Path(dirpath)
            if here.is_symlink():
                dirnames[:] = []
                continue
            for name in filenames:
                if name.endswith(PARTIAL_SUFFIXES):
                    p.add(1, here / name, 'partial transfer')
            for name in list(dirnames):
                if name.endswith('-partial'):
                    p.add(1, here / name, 'staging directory')
                    dirnames.remove(name)
                elif name in ('ram',):  # tmpfs mount points belong to cache_admission
                    dirnames.remove(name)

    # 2. Other tasks' code bundles; base archives whose source tree finished extracting.
    if code_cache.is_dir():
        for bundle in sorted(code_cache.iterdir()):
            if bundle.name != current_code:
                p.add(2, bundle, 'code bundle of another task')
    for root in roots:
        archive = root / 'base-download.tar.gz'
        sources = root / 'sources'
        if archive.exists() and sources.is_dir() and any(sources.glob('*/.source-complete')):
            p.add(2, archive, 'base archive already extracted')

    other = [r for r in roots if r.resolve() != p.current_root and root_is_idle(r)]
    # 3. Derived serving artifacts (adapters, uploads) of other runs.
    for root in other:
        for run in sorted((root / 'runs').glob('*')) if (root / 'runs').is_dir() else []:
            for name in ('loras', 'uploads'):
                p.add(3, run / name, 'serving artifacts of a finished run')
            for tar in run.glob('*.reload.tar'):
                p.add(3, tar, 'serving artifacts of a finished run')
    # 4. Rebuildable environments and source trees of idle roots.
    for root in other:
        for name in ('envs', 'sources'):
            p.add(4, root / name, 'rebuildable environment of an idle root')
    # 5. Whole run directories of finished inference-only runs (farms keep no durable state).
    for root in other:
        for run in sorted((root / 'runs').glob('*')) if (root / 'runs').is_dir() else []:
            try:
                record = json.loads((run / RUN_MANIFEST).read_text())
            except (OSError, ValueError):
                continue
            if record.get('inference_only') is True:
                p.add(5, run, 'finished inference-only run')
    # 6. Download caches.
    p.add(6, cache / 'uv', 'uv download cache')
    return sorted(p.items, key=lambda item: (item['tier'], -item['bytes']))


def remove(path):
    path = Path(path)
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def reconcile(home, current_root=None, current_code=None, *, target_free_bytes=0, startup=False,
              dry_run=False, timeout=120, log=None, used=None, mounts=None, disk_usage=shutil.disk_usage,
              stopping=lambda: False):
    """Delete planned items until free space reaches the target (at startup tiers 1-3 always go).

    Returns a report: planned items, what was removed, bytes freed, free bytes before/after, and whether the
    time budget or a stop request ended it early. Nothing is deleted when dry_run is set.
    """
    home = Path(home).resolve()
    started = time.monotonic()
    items = plan(home, current_root, current_code, used=used, mounts=mounts)
    free_before = disk_usage(home).free
    removed, freed, early = [], 0, None
    for item in items:
        if stopping():
            early = 'stopping'
            break
        if time.monotonic() - started > timeout:
            early = 'timeout'
            break
        short = disk_usage(home).free < target_free_bytes
        # Dead items (partials, other tasks' bundles and adapters) always go at startup; rebuildable
        # environments, finished farm runs and download caches only while short of the target.
        if not short and not (startup and item['tier'] <= ALWAYS_AT_STARTUP):
            continue
        if not dry_run:
            remove(item['path'])
        removed.append(item)
        freed += item['bytes']
        if log is not None:
            from .events import emit
            emit(log, 'storage_item_planned' if dry_run else 'storage_item_removed', item=item['path'],
                 reason=item['reason'], tier=item['tier'], bytes=item['bytes'])
    report = dict(dry_run=dry_run, planned=items, removed=[i['path'] for i in removed], freed_bytes=freed,
                  free_before=free_before, free_after=disk_usage(home).free, stopped_early=early,
                  seconds=round(time.monotonic() - started, 2))
    if log is not None:
        from .events import emit
        emit(log, 'storage_reconciled', dry_run=dry_run, removed=len(removed), freed_bytes=freed,
             free_before=free_before, free_after=report['free_after'], stopped_early=early)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Plan (default) or apply host storage reclamation.')
    parser.add_argument('--home', default=str(Path.home()))
    parser.add_argument('--current-root')
    parser.add_argument('--current-code', default=os.environ.get('RAY_TRAIN_CODE_SHA256'))
    parser.add_argument('--target-free-gib', type=float, default=0)
    parser.add_argument('--apply', action='store_true', help='delete; without it this is a dry run')
    args = parser.parse_args(argv)
    report = reconcile(args.home, args.current_root, args.current_code,
                       target_free_bytes=int(args.target_free_gib * 1024**3), startup=True, dry_run=not args.apply)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
