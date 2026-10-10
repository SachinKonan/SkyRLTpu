"""Versioned inputs and reproducible candidate identities."""
import hashlib
import json
from pathlib import Path

VERSION = 'fair-universe-phase2-v1'
ARCHIVE_SHA256 = '4c74709ea2a168faffe5581627d7f0c06eb3942bb1a47586e0b24c1d1edda60a'
TEST_NAME = 'WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy'
TEST_COUNT = 10000
VENDOR = Path(__file__).with_name('vendor')
SPLIT_COUNTS = dict(train=160, val=32, development=32, locked=32)
NOISE_SIGMA = .4 / (2 * 30 * 2**2)**.5


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def bundle_identity(candidate, config, manifest):
    # All allowed imported candidate helpers and runtime code participate.
    root = Path(__file__).parent
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.suffix in ('.py', '.json', '.toml') and '__pycache__' not in p.parts}
    payload = dict(version=VERSION, candidate=digest(candidate), runtime=files,
                   config=config, data=manifest)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def checked_scores(scores, count):
    import numpy as np
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (count,) or not np.isfinite(scores).all():
        raise ValueError('predict must return one finite score per map')
    return scores


def check_independence(predict, maps):
    """Metamorphic gate on validation inputs; never inspect official test data."""
    import numpy as np
    n = len(maps)
    batched = checked_scores(predict(maps), n)
    singles = np.array([checked_scores(predict(maps[i:i+1]), 1)[0] for i in range(n)])
    reverse = checked_scores(predict(maps[::-1]), n)[::-1]
    repeat = checked_scores(predict(maps), n)
    # Compare within a fixed precision tolerance; no rank-only exemption.
    for name, scores in [('singleton', singles), ('permutation', reverse), ('repeat', repeat)]:
        if not np.allclose(batched, scores, rtol=1e-5, atol=1e-7):
            raise ValueError(f'prediction independence failed: {name}')
    return dict(passed=True, samples=n, rtol=1e-5, atol=1e-7)
