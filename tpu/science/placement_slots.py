"""Ray TPU assignment and placement admission; never initializes a TPU."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat


def validate_chip(chip):
    if type(chip) is not int or chip not in range(4):
        raise ValueError('placement chip must be an integer from 0 to 3')
    return chip


def chips_from_env(env):
    chips = tuple(int(c) for c in env.get('PLACEMENT_TPU_CHIPS', '0').split(','))
    if not chips or len(set(chips)) != len(chips):
        raise ValueError('placement chips must be nonempty and unique')
    for chip in chips:
        validate_chip(chip)
    return chips


def host_resources(chips):
    count = len(chips)
    return {'TPU': count, 'placement_tpu_host': count}


def task_resources():
    return {'TPU': 1, 'placement_tpu_host': 1}


def assigned_chip(accelerator_ids):
    """Use Ray's assignment, never a driver-selected ID or inherited env var."""
    ids = accelerator_ids.get('TPU', [])
    if not isinstance(ids, (list, tuple)) or len(ids) != 1:
        raise RuntimeError(f'expected exactly one Ray-assigned TPU ID, got {ids!r}')
    raw = ids[0]
    if type(raw) is int:
        chip = raw
    elif isinstance(raw, str) and raw in ('0', '1', '2', '3'):
        chip = int(raw)
    else:
        raise RuntimeError(f'invalid Ray-assigned physical TPU ID: {raw!r}')
    return validate_chip(chip)


def chip_cpus(chip):
    start = 16 + 4 * validate_chip(chip)
    return list(range(start, start + 4))


def grading_nodes(nodes):
    return sorted((node for node in nodes if node['Alive'] and
                   node['Resources'].get('placement_tpu_host', 0) >= 1),
                  key=lambda node: node['NodeID'])


def grading_bundles(node):
    resources = node['Resources']
    count = resources['placement_tpu_host']
    if count not in (1, 2, 3, 4) or resources.get('TPU', 0) < count:
        raise ValueError('grading host must advertise matching TPU and grading capacity')
    # Ray's automatic node resource pins this group to this exact grading host.
    node_resource = 'node:' + node['NodeManagerAddress']
    if resources.get(node_resource, 0) < .001 * count:
        raise ValueError('grading node is missing its Ray node resource')
    return [dict(CPU=4, memory=16*1024**3, **task_resources(), **{node_resource: .001})
            for _ in range(int(count))]


@contextmanager
def chip_lock(chip, directory=None):
    # Independent of payload/run directories so separate runs cannot use the
    # same physical chip concurrently. Locks are never unlinked on release.
    directory = Path(directory) if directory else Path('/tmp') / f'science-placement-locks-{os.getuid()}'
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('placement lock directory must be private and user-owned')
    fd = os.open(directory / f'chip-{validate_chip(chip)}.lock',
                 os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise RuntimeError('invalid placement chip lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def device_paths(chip, accelerator, dev_root=Path('/dev')):
    """Expose one accelerator device; the VFIO control node is not a chip."""
    validate_chip(chip)
    if accelerator in ('tpu-v4-64', 'tpu-v4-32'):
        paths = [dev_root / f'accel{chip}']
    elif accelerator == 'tpu-v5p-32':
        paths = [dev_root / 'vfio' / str(chip), dev_root / 'vfio' / 'vfio']
    else:
        raise ValueError('unsupported placement accelerator')
    for path in paths:
        if not stat.S_ISCHR(path.stat().st_mode):
            raise RuntimeError(f'placement device is not a character device: {path}')
    return paths


def tpu_environment(chip, accelerator):
    validate_chip(chip)
    if accelerator not in ('tpu-v4-64', 'tpu-v4-32', 'tpu-v5p-32'):
        raise ValueError('unsupported placement accelerator')
    port = str(8476 + chip)
    return dict(JAX_PLATFORMS='tpu', TPU_VISIBLE_CHIPS=str(chip),
                TPU_PROCESS_BOUNDS='1,1,1', TPU_CHIPS_PER_PROCESS_BOUNDS='1,1,1',
                TPU_PROCESS_ADDRESSES='localhost:' + port, TPU_PROCESS_PORT=port,
                CLOUD_TPU_TASK_ID='0', TPU_SKIP_MDS_QUERY='true',
                TPU_ACCELERATOR_TYPE=accelerator.removeprefix('tpu-'),
                TPU_HOST_BOUNDS='1,1,1', TPU_CHIPS_PER_HOST_BOUNDS='2,2,1',
                TPU_WORKER_HOSTNAMES='localhost', TPU_WORKER_ID='0',
                TPU_TOPOLOGY_ALT='false', TPU_TOPOLOGY_WRAP='false,false,false')
