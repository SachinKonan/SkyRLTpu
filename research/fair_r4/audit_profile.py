"""Evaluate a frozen per-map physical profile; no training or public-test input."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from continuous_profile import Profile
from common import load_labels, splits, sha, write_json, score_report, low_fpr_metric


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    model = Profile(a.run); labels = load_labels(a.raw); index = splits()
    start = time.monotonic(); diagnostics = {}; normal = {}

    def evaluate(name, features, theta=None):
        tick = time.monotonic(); result = model(features, theta)
        assert all(np.isfinite(v).all() for v in result.values())
        violation = result['continuous_theta'] @ model.equations[:, :2].T + model.equations[:, 2]
        assert violation.max() < 3e-6
        assert np.all(result['continuous_energy'] <= result['grid_energy'] + 1e-4)
        reduction = result['grid_energy'] - result['continuous_energy']
        entry = dict(count=len(features), seconds=time.monotonic()-tick,
                     max_hull_violation=float(violation.max()),
                     reduction_quantiles=np.quantile(reduction, [0, .5, .9, .99, 1]).tolist())
        diagnostics[name] = entry
        print(json.dumps(dict(dataset=name, **entry)), flush=True)
        return result

    for name in ('val', 'calibration', 'audit'):
        features = np.load(Path(a.features)/(name+'_latent.npy'), allow_pickle=False)
        theta = labels[index[name], :2]
        result = evaluate(name, features, theta)
        np.savez(out/(name+'.npz'), ids=index[name], true_theta=theta, **result)
        normal[name] = {k: v for k, v in result.items() if k.endswith('_energy')}

    probe_metrics = {}
    for path in sorted(Path(a.probes).glob('*.npz')):
        combined = {}; metrics = {}
        with np.load(path, allow_pickle=False) as f:
            for name in f.files:
                if name == 'ids':
                    combined['ids'] = f[name]
                    continue
                result = evaluate(path.stem+'/'+name, f[name])
                combined.update({name+'__'+k: v for k, v in result.items()})
                metrics[name] = {key: low_fpr_metric(normal['calibration'][key], result[key])
                                 for key in ('grid_energy', 'continuous_energy')}
        np.savez(out/(path.stem+'.npz'), **combined)
        probe_metrics[path.stem] = metrics

    write_json(out/'report.json', dict(
        kind='NORMAL_AND_REUSED_SYNTHETIC_NOT_OFFICIAL_OOD', public_test_read=False,
        network_weights_frozen=True, checkpoint_sha256=sha(Path(a.run)/'best.msgpack'),
        runtime_sha256=sha(Path(__file__).with_name('continuous_profile.py')),
        steps=model.steps, starts=model.starts, step_size=model.step_size,
        normal=score_report(normal['calibration'], normal['audit']),
        diagnostics=diagnostics, probes=probe_metrics, seconds=time.monotonic()-start,
        limitations=['Known true cosmology is diagnostic only.',
                     'No guaranteed global optimum; interpolated null family is enlarged.',
                     'Audit maps and synthetic families have been used in previous research rounds.']))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('run', 'raw', 'features', 'probes', 'out'):
        p.add_argument('--'+name, required=True)
    main(p.parse_args())
