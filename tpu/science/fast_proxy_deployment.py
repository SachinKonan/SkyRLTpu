"""Trusted verification and minimal read-only mounts for the CPU helper."""
import hashlib
import json
from pathlib import Path


def verified_mounts(root):
    folder = Path(root) / 'tpu/science/fast_proxy'
    record = json.loads((folder / 'build-manifest.json').read_text())
    names = ('__init__.py', 'congestion.c', 'libproxy.so')
    for name in names:
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != record[name]:
            raise RuntimeError('fast proxy build checksum mismatch: ' + name)
    return [(folder / name, '/fast_proxy/' + name) for name in ('__init__.py', 'libproxy.so')], record
