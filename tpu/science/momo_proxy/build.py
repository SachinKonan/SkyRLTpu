"""Trusted setup only: compile before candidate timing, never from candidates."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile


def build():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=root) as folder:
        output = Path(folder) / 'libmomoproxy.so'
        subprocess.run(['cc', '-O3', '-std=c11', '-fPIC', '-shared', '-ffp-contract=off',
                        str(root / 'proxy.c'), '-lm', '-o', str(output)], check=True)
        output.replace(root / 'libmomoproxy.so')
    record = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ('__init__.py', 'proxy.c', 'libmomoproxy.so')}
    (root / 'build-manifest.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    build()
