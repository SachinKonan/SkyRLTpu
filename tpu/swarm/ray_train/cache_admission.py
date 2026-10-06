"""Reconcile retired executor tmpfs mounts before reserving another RAM cache."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import subprocess

from .cache import mount_cache
from .checkpoint_retention import MANIFEST, LEASE, private_path, run_is_active as unprivileged_run_is_active
from .events import emit

# Only disposable cache contents may be moved or discarded. Unknown files
# cause preservation, rather than assuming every tmpfs is ours.
CACHE_ENTRIES = {'hf', 'orbax', 'compile', 'compile.prefix', 'compile-upload',
                 'download', 'metadata', 'role.json'}

# Root can inspect the same-user sshd/PAM processes whose /proc environment is
# unreadable to the workload UID. Return only a boolean, never environment data.
PROCESS_AUDIT = r'''
import json, os, sys
from pathlib import Path
uid, task_id, run_id, run = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
caller = int(sys.argv[5])
audit_pids = {os.getpid()}
parent = os.getppid()
# sudo retains the caller's real UID and its argv contains the audited run.
# Exclude only our own helper/transport ancestry up to the known caller.
while parent > 1 and parent != caller:
    audit_pids.add(parent)
    parent_status = dict(line.split(':', 1) for line in Path(f'/proc/{parent}/status').read_text().splitlines())
    parent = int(parent_status['PPid'].strip())
if parent != caller:
    raise RuntimeError('cannot verify process-audit ancestry')
active = False
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) in audit_pids:
        continue
    try:
        status = dict(line.split(':', 1) for line in (proc/'status').read_text().splitlines())
        if int(status['Uid'].split()[0]) != uid or status['State'].strip().startswith('Z'):
            continue
        env = dict(entry.split('=', 1) for entry in (proc/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in entry)
        args = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        if (env.get('SKYPILOT_TASK_ID') == task_id or env.get('RAY_NAMESPACE') == run_id
                or env.get('TTD_RUN_DIR') == run + '/client' or any(run in arg for arg in args)):
            active = True
            break
    except (FileNotFoundError, ProcessLookupError):
        continue
    except Exception:
        active = True
        break
print(json.dumps({'active': active}))
'''


def run_is_active(record, run):
    if not unprivileged_run_is_active(record, run):
        return False
    result = subprocess.run(['sudo', '-n', 'python3', '-c', PROCESS_AUDIT, str(os.getuid()),
                             record['task_id'], record['run_id'], str(run), str(os.getpid())],
                            capture_output=True, text=True, timeout=20)
    if result.returncode:
        return True
    try:
        return json.loads(result.stdout).get('active') is not False
    except (ValueError, AttributeError):
        return True


@contextmanager
def exclusive_file(path, *, create=False):
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0), 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise ValueError('cache lease is not an owned private regular file')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        os.close(fd)


def tmpfs_mounts():
    data = json.loads(subprocess.check_output(
        ['findmnt', '--json', '--types', 'tmpfs', '--output', 'TARGET'], text=True))
    def walk(items):
        for item in items:
            yield Path(item['target'])
            yield from walk(item.get('children', []))
    return set(walk(data['filesystems']))


def unused_mount(mount):
    result = subprocess.run(['sudo', '-n', 'fuser', '-m', str(mount)],
                            capture_output=True, text=True, timeout=15)
    # No users is exit 1 with no diagnostics; permission errors are not idle.
    return result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip()


@contextmanager
def retired_root(root, mounts):
    """Hold bootstrap and every run lease; reject live/orphaned users."""
    parent = root.parent
    private_path(parent, root)
    mount = private_path(root, root / 'ram')
    if root.stat().st_uid != os.getuid() or mount not in mounts:
        raise ValueError('not an owned executor tmpfs')
    if any(other != mount and other.is_relative_to(mount) for other in mounts):
        raise ValueError('nested mounts cannot be reclaimed')
    if any(p.name not in CACHE_ENTRIES or p.is_symlink() for p in mount.iterdir()):
        raise ValueError('unknown cache contents')
    with ExitStack() as stack:
        stack.enter_context(exclusive_file(private_path(root, root / 'owner.lock')))
        runs = private_path(root, root / 'runs')
        records = []
        for run in sorted(runs.iterdir()):
            private_path(root, run)
            if not run.is_dir():
                raise ValueError('unknown run entry')
            stack.enter_context(exclusive_file(private_path(root, run / LEASE)))
            record = json.loads(private_path(root, run / MANIFEST).read_text())
            if (record.get('version') != 1 or record.get('root') != str(root)
                    or record.get('run_id') != run.name
                    or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}', run.name)
                    or not re.fullmatch(r'sky-managed-[^\s]+_\d+-\d+', record.get('task_id', ''))
                    or not re.fullmatch(r'gs://[a-z0-9._-]+/ray-training/' + re.escape(run.name)
                                        + r'/checkpoints', record.get('checkpoint_gcs', ''))):
                raise ValueError('invalid executor ownership record')
            records.append((record, run))
        if not records or any(run_is_active(record, run) for record, run in records):
            raise RuntimeError('root has live or uninspectable run processes')
        if not unused_mount(mount):
            raise RuntimeError('RAM mount has users or cannot be inspected')
        yield mount


def compatible(mount, config, role):
    """Choose a reusable model/role; normal restore still verifies revisions."""
    if role is None:
        return False
    try:
        if json.loads((mount / 'role.json').read_text())['role'] != role:
            return False
        key = 'models--' + config.model.replace('/', '--')
        if not (mount / 'hf/hub' / key).is_dir():
            return False
        return role == 'inference' or (mount / 'orbax' / config.trainer.maxtext_model).is_dir()
    except (OSError, ValueError, KeyError):
        return False


def admit_cache(config, root, role, log):
    """Reuse one compatible mount, evict only idle owned caches, then reserve.

    Bootstrap holds current root/owner.lock. This additional lock serializes
    cross-root admission on the host. It never terminates another workload or
    deletes durable run state. No lazy/forced unmounts are used.
    """
    if role not in ('trainer', 'inference', None):
        raise ValueError('unknown cache role')
    root = Path(root).resolve()
    target = root / 'ram'
    private_path(root, target)
    target.mkdir(exist_ok=True)
    with exclusive_file(root.parent / '.skyrl-ram-admission.lock', create=True):
        mounts = tmpfs_mounts()
        # A previous interrupted admission can leave an empty reservation.
        if target in mounts and not any(target.iterdir()) and unused_mount(target):
            subprocess.run(['sudo', '-n', 'umount', str(target)], check=True, timeout=30)
            mounts = tmpfs_mounts()
        candidates = [p.parent for p in mounts if p.name == 'ram'
                      and p.parent.parent == root.parent and p.parent != root]
        candidates.sort(key=lambda p: (not compatible(p / 'ram', config, role), str(p)))
        for old in candidates:
            try:
                with retired_root(old, mounts) as mount:
                    # Recheck mount identity immediately before the operation.
                    current = tmpfs_mounts()
                    if mount not in current or not unused_mount(mount):
                        raise RuntimeError('cache mount changed or acquired users')
                    reuse = target not in current and compatible(mount, config, role)
                    if reuse:
                        if any(target.iterdir()):
                            raise RuntimeError('refusing to move a cache over existing files')
                        # MS_MOVE is forbidden beneath the shared root mount
                        # on TPU VMs. A bind followed by a normal unmount keeps
                        # the same tmpfs/pages under the new path instead.
                        subprocess.run(['sudo', '-n', 'mount', '--bind', str(mount), str(target)],
                                       check=True, timeout=30)
                        try:
                            if target not in tmpfs_mounts() or target.stat().st_dev != mount.stat().st_dev:
                                raise RuntimeError('cache bind does not reference the original tmpfs')
                            subprocess.run(['sudo', '-n', 'umount', str(mount)], check=True, timeout=30)
                        except (OSError, RuntimeError, subprocess.SubprocessError):
                            subprocess.run(['sudo', '-n', 'umount', str(target)], check=True, timeout=30)
                            raise
                        if target not in tmpfs_mounts() or mount in tmpfs_mounts():
                            raise RuntimeError('cache move verification failed')
                        emit(log, 'ram_cache_reused', previous_root=str(old), root=str(root), role=role)
                    else:
                        info = os.statvfs(mount)
                        used = (info.f_blocks - info.f_bfree) * info.f_frsize
                        subprocess.run(['sudo', '-n', 'umount', str(mount)], check=True, timeout=30)
                        if mount in tmpfs_mounts():
                            raise RuntimeError('cache unmount verification failed')
                        emit(log, 'ram_cache_reclaimed', previous_root=str(old), bytes=used)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError,
                    subprocess.SubprocessError) as exc:
                emit(log, 'ram_cache_preserved', previous_root=str(old), reason=str(exc))
        if role is None:
            return None
        cap = config.cache.trainer_gib if role == 'trainer' else config.cache.inference_gib
        return mount_cache(target, cap, config.cache.reserve_gib)
