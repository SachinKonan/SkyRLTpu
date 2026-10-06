"""Reclaim canary inference weights on an idle host assigned to this trainer."""

import argparse
from pathlib import Path


def prune(root):
    payloads = []
    markers = set()
    for canary in Path(root).glob('ray-serve-canary-v4-64-*'):
        model = canary/'model'
        if canary.is_symlink() or model.is_symlink():
            raise ValueError(f'Refusing a linked canary cache: {canary}')
        if not model.is_dir():
            continue
        # prepare_cache.py stores ordinary/hardlinked weight files here and
        # redownloads missing files on the next run. Keep tokenizer metadata,
        # results, logs, source, environments, and compilation caches intact.
        for path in model.glob('*.safetensors'):
            if path.is_file() and not path.is_symlink():
                payloads.append(path)
                markers.add(canary/'model-ready')
    total = sum(path.stat().st_size for path in payloads)
    # Invalidate readiness before removing any payload, including on interruption.
    for marker in markers:
        marker.unlink(missing_ok=True)
    for path in payloads:
        path.unlink()
    return len(payloads), total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('host_home', type=Path)
    args = parser.parse_args()
    count, size = prune(args.host_home)
    print(f'Canary inference cache reclaimed: {count} weight files, {size} bytes', flush=True)
