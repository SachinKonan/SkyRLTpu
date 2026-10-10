"""Paired fresh-noise and storage-precision checks, from training maps only."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parents[1]/'fair_r2'))
sys.path.insert(2, str(Path(__file__).resolve().parents[1]/'fair_r3'))
sys.path.insert(3, str(Path(__file__).resolve().parents[1]/'fair_ae'))
from common import splits, write_json, sha


def main(a):
    from train import load_training, NOISE
    from diagnostics import shift
    from morphology_probes import iaaft
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    maps, mask, labels, index = load_training(a.raw)
    h, w = mask.shape; count = mask.reshape(h//2, 2, w//2, 2).sum((1, 3)); support = count >= 3
    normal_ids = index['audit']; probe_ids = normal_ids.reshape(101, 32)[:, [4, 9, 18, 27]].ravel()
    families = [('spectral_low', .2), ('spectral_high', .2), ('quadratic_peak', .2),
                ('quadratic_peak', -.2), ('gaussian_mixture', .5), ('anisotropic', .2),
                ('spatial_modulation', .5), ('phase_iaaft', 1.)]
    groups = {}; offset = 0
    for name, n in [('normal_float32', len(normal_ids)), ('normal_float16', len(normal_ids))] + [
            (f'{name}:{strength}', len(probe_ids)) for name, strength in families]:
        groups[name] = dict(offset=offset, count=n); offset += n
    pooled = np.lib.format.open_memmap(out/'observed.npy', mode='w+', dtype=np.float32,
                                      shape=(offset, h//2, w//2))
    source_ids = np.empty(offset, np.int32)
    if a.features:
        from features import make_predict
        from spatial_features import SpatialSummary
        physical = make_predict(mask, 16); spatial = SpatialSummary(mask, 16)
        features = np.empty((offset, 165), np.float32)
    start = time.monotonic(); iaaft_errors = []

    def save(name, lo, x, ids):
        row = groups[name]['offset']+lo
        pooled[row:row+len(x)] = (x.reshape(len(x), h//2, 2, w//2, 2).sum((2, 4)) /
                                  np.maximum(count, 1)*support).astype(np.float32)
        source_ids[row:row+len(x)] = ids
        if a.features: features[row:row+len(x)] = np.column_stack([physical(x), spatial(x)])

    def raw_batch(ids):
        x = np.zeros((len(ids), h, w), np.float32); x[:, mask] = maps[ids]
        noise = np.stack([np.random.default_rng(991000+int(i)).normal(0, NOISE, (h, w))
                          for i in ids]).astype(np.float32)*mask
        return x, noise

    for lo in range(0, len(normal_ids), 16):
        ids = normal_ids[lo:lo+16]; x, noise = raw_batch(ids); observed = x+noise
        save('normal_float32', lo, observed, ids)
        save('normal_float16', lo, observed.astype(np.float16).astype(np.float32), ids)
    print(json.dumps(dict(stage='normals', seconds=time.monotonic()-start)), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for lo in range(0, len(probe_ids), 16):
            ids = probe_ids[lo:lo+16]; x, noise = raw_batch(ids)
            ref_ids = np.asarray([index['train'][160*(i//256)+(i%160)] for i in ids])
            reference = np.zeros_like(x); reference[:, mask] = maps[ref_ids]
            for family, strength in families:
                if family == 'phase_iaaft':
                    transformed = list(pool.map(lambda pair: iaaft(pair[1], mask, 861004+int(pair[0]), 60), zip(ids, x)))
                    y = np.stack([r[0] for r in transformed]); iaaft_errors.extend([r[1] for r in transformed])
                else: y = shift(x, mask, family, strength, reference)
                save(f'{family}:{strength}', lo, (y+noise).astype(np.float16).astype(np.float32), ids)
            if lo%64 == 0: print(json.dumps(dict(stage='probes', count=lo+len(ids), seconds=time.monotonic()-start)), flush=True)
    pooled.flush(); np.save(out/'source_ids.npy', source_ids)
    if a.features: np.save(out/'features165.npy', features)
    digest = hashlib.sha256()
    with (out/'observed.npy').open('rb') as f:
        for block in iter(lambda: f.read(16*1024*1024), b''): digest.update(block)
    manifest = dict(kind='FROZEN_FRESH_NOISE_REUSED_NORMAL_MAPS_AND_SYNTHETIC_FAMILIES',
                    public_test_read=False, model_fitting=False, groups=groups, shape=list(pooled.shape),
                    source_sha256=sha(__file__), observations_sha256=digest.hexdigest(),
                    ids_sha256=sha(out/'source_ids.npy'), seconds=time.monotonic()-start,
                    features_generated=a.features, iaaft_errors=iaaft_errors)
    write_json(out/'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'iaaft_errors'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True); p.add_argument('--out', required=True)
    p.add_argument('--features', action='store_true')
    main(p.parse_args())
