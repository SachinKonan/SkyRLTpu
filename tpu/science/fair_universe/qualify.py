"""Synthetic plumbing check only. Never report this score as benchmark evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from .contract import VENDOR, VERSION
from .evaluate import evaluate


def fixture(path):
    path = Path(path)
    if (path/'manifest.json').exists(): return
    rng = np.random.default_rng(4291)
    mask = np.ones((1424, 176), bool)
    mask[:10] = False
    for name in ['fit/train', 'fit/val', 'judge/development']:
        folder = path/name; folder.mkdir(parents=True, exist_ok=True)
        maps = rng.normal(0, .003, (16, int(mask.sum()))).astype(np.float16)
        labels = rng.uniform(.1, 1, (16, 5)).astype(np.float32)
        np.save(folder/'maps.npy', maps); np.save(folder/'labels.npy', labels); np.save(folder/'mask.npy', mask)
    (path/'manifest.json').write_text(json.dumps(dict(version=VERSION, synthetic_plumbing_only=True)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--chip', type=int, required=True)
    p.add_argument('--python', required=True)
    a = p.parse_args(); root = Path(a.root)
    fixture(root/'fixture')
    result = evaluate(VENDOR/'candidates/baseline_jax.py', root/'fixture', root/f'qualify-chip-{a.chip}',
                      a.python, chip=a.chip, steps=2, batch_size=4, max_ind=8,
                      train_timeout=240, reward_timeout=240)
    print(json.dumps(result, allow_nan=False))
    raise SystemExit(0 if result.get('correctness') else 1)
