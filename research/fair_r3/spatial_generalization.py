"""Frozen regional specialist on fresh noise and previously untested modulation patterns.

Clean audit realizations are reused. This is a controlled generalization check,
not a fresh simulator holdout and not official challenge accuracy.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'fair_r2'))
sys.path.insert(2, str(Path(__file__).resolve().parents[1] / 'fair_ae'))
from spatial_features import SpatialSummary
from spatial_residuals import load, predict
from calibrate_spatial import regress
from common import write_json, sha, low_fpr_metric
from flow import predictor
from train import load_training, NOISE
from features import make_predict


def match(y, x, mask):
    target, source = x[:, mask], y[:, mask]
    return ((y - source.mean(1)[:, None, None]) *
            (target.std(1) / np.maximum(source.std(1), 1e-8))[:, None, None] +
            target.mean(1)[:, None, None]) * mask


def main(a):
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=False)
    controls = root / 'spatial-generalization-controls'
    model_path = controls / 'regional-residual-model.npz'
    regression_path = controls / 'normal-regressors.npz'
    models = [Path(a.baseline) / f'flow-cosmology-asinh-s{i}' for i in range(3)]
    new_models = [root / f'flow-noise4-long-s{i}' for i in range(3)]
    manifest = dict(kind='FROZEN_REGIONAL_GENERALIZATION_PROTOCOL', count=1024,
        public_test_read=False, model_fitting=False,
        map_source='Reused normal audit realizations; fresh shape noise and new modulation patterns',
        id_seed=90321, noise_seed_base=984000, phase_seed_base=943902,
        batch=16, amplitude=.25,
        patterns=['normal', 'sine_original', 'sine_random_phase', 'sine_two_cycles', 'linear_gradient'],
        pattern_rule='Multiply clean signal by 1+0.25*wave, then match its observed mean and standard deviation to the original clean map; add the same fresh Gaussian noise to each paired member.',
        regional_model_sha256=sha(model_path), normal_regression_sha256=sha(regression_path),
        flow_checkpoints={p.name: sha(p / 'best.msgpack') for p in models + new_models},
        selection='No fitting or selection is performed on the new patterns.')
    write_json(out / 'frozen-protocol.json', manifest)
    maps, mask, labels, splits = load_training(a.raw)
    ids = np.random.default_rng(manifest['id_seed']).permutation(splits['audit'])[:1024]
    old = [predictor(p) for p in models]; new = [predictor(p) for p in new_models]
    spatial = SpatialSummary(mask, 16); physical = make_predict(mask, 16)
    regional = load(model_path)
    with np.load(regression_path) as f:
        mean = {k.removeprefix('mean_'): f[k] for k in f.files if k.startswith('mean_')}
    results = {name: {} for name in ('04-summary', 'noise-long-summary', 'regional-raw', 'regional-centered')}
    feature_cache = {}; start = time.monotonic()
    h = mask.shape[0]; coordinate = np.arange(h) / h
    for lo in range(0, len(ids), 16):
        ix = ids[lo:lo + 16]; x = np.zeros((len(ix), *mask.shape), np.float32); x[:, mask] = maps[ix]
        rng = np.random.default_rng(manifest['noise_seed_base'] + lo)
        noise = rng.normal(0, NOISE, x.shape).astype(np.float32) * mask
        phases = np.asarray([np.random.default_rng(manifest['phase_seed_base'] + int(i)).uniform(0, 2*np.pi)
                             for i in ix])
        patterns = {
            'sine_original': np.broadcast_to(np.sin(2*np.pi*coordinate), (len(ix), h)),
            'sine_random_phase': np.sin(2*np.pi*coordinate[None] + phases[:, None]),
            'sine_two_cycles': np.sin(4*np.pi*coordinate[None] + phases[:, None]),
            'linear_gradient': np.broadcast_to(2*coordinate - 1, (len(ix), h))}
        for key in manifest['patterns']:
            clean = x if key == 'normal' else match(x * (1 + .25*patterns[key][..., None]), x, mask)
            observed = (clean + noise).astype(np.float32)
            features = np.column_stack([physical(observed), spatial(observed)]).astype(np.float64)
            feature_cache.setdefault(key, []).append(features.astype(np.float32))
            raw = predict(features, regional)
            values = {'04-summary': np.mean([fn(features[:, :72])['minimum_base_energy'] for fn in old], axis=0),
                      'noise-long-summary': np.mean([fn(features[:, :72])['minimum_base_energy'] for fn in new], axis=0),
                      'regional-raw': raw,
                      'regional-centered': np.log(np.maximum(raw, 1e-12)) - regress(features[:, :72], mean)}
            for name, value in values.items(): results[name].setdefault(key, []).append(value)
        if lo % 256 == 0: print(json.dumps(dict(completed=lo+len(ix), seconds=time.monotonic()-start)), flush=True)
    np.savez(out / 'features.npz', ids=ids, **{k: np.concatenate(v) for k, v in feature_cache.items()})
    report = dict(kind='FRESH_NOISE_NEW_PATTERNS_REUSED_AUDIT_REALIZATIONS', public_test_read=False,
                  model_fitting=False, frozen_protocol_sha256=sha(out / 'frozen-protocol.json'), results={})
    for name, result in results.items():
        values = {k: np.concatenate(v) for k, v in result.items()}
        with np.load(controls / (name + '.npz')) as f: cal = f['calibration']
        assert all(np.isfinite(v).all() for v in values.values())
        np.savez(out / (name + '.npz'), calibration=cal, ids=ids, **values)
        thresholds = np.quantile(cal, [.95, .99, .999])
        report['results'][name] = dict(
            scores={k: low_fpr_metric(cal, v) for k, v in values.items()},
            normal_count=len(ids), normal_exceedances=[int((values['normal'] > t).sum()) for t in thresholds])
    report['seconds'] = time.monotonic() - start
    write_json(out / 'report.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'baseline', 'raw', 'out'): p.add_argument('--' + name, required=True)
    main(p.parse_args())
