"""Candidate entrypoint. No judge arrays, split metadata, or labels at prediction."""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from .data import DataView
from .contract import checked_scores


def fixed_predict(mod, model, maps, mask, config):
    """One fixed XLA shape prevents batch-size-dependent FFT/dot rounding.

    Padding is constant and its scores are discarded. The independence gate
    still varies real companions, order, and repeated calls to catch coupling.
    """
    if not 0 < len(maps) <= 32:
        raise ValueError('prediction batches must contain 1 through 32 maps')
    padded = np.zeros((32, *maps.shape[1:]), dtype=maps.dtype)
    padded[:len(maps)] = maps
    return checked_scores(mod.predict(model, padded, mask, config), 32)[:len(maps)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['fit', 'predict'])
    a = p.parse_args()
    cfg = json.loads(Path('/config.json').read_text())
    sys.path.insert(0, '/vendor')
    spec = importlib.util.spec_from_file_location('candidate_program', '/candidate.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(mod)
        if a.phase == 'fit':
            info = mod.fit(DataView('/fit/train'), DataView('/fit/val'), cfg, Path('/artifacts'))
            Path('/artifacts/fit_info.json').write_text(json.dumps(info, default=str))
            return
        model = mod.load_model(Path('/artifacts'), cfg)
        mask = np.load('/mask.npy', allow_pickle=False)
    print(json.dumps({'ready': True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('close'):
            return
        with contextlib.redirect_stdout(sys.stderr):
            maps = np.load('/input/maps.npy', allow_pickle=False)
            scores = fixed_predict(mod, model, maps, mask, cfg)
        print(json.dumps({'scores': scores.tolist()}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
