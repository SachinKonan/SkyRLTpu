"""Numerical and normal-data pilot; no synthetic or public test inference."""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np

from field_latents import FrozenNoiseField, ode_latent
from noise_statistics import statistics
from field_evaluate import predictor
from field_flow import NOISE
from common import splits, write_json, sha


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    model = FrozenNoiseField(a.run); norm = model.norm; support = model.support
    ids = np.random.default_rng(6831).permutation(splits()['val'])[:a.count]
    clean = np.load(Path(a.cache)/'clean.npy', mmap_mode='r')[ids].astype(np.float32)
    x = (clean+np.random.default_rng(96831).normal(size=clean.shape)*NOISE/
         np.sqrt(np.maximum(norm['count'],1)))*support
    x = x.astype(np.float32)
    numerical = {}; outputs = {}
    for steps in (16,32,64):
        if steps != 32: model = FrozenNoiseField(a.run, steps=steps)
        else: model = FrozenNoiseField(a.run)
        start = time.monotonic(); latent = model(x); first = time.monotonic()-start
        start = time.monotonic(); repeat = model(x); warm = time.monotonic()-start
        assert np.array_equal(latent, repeat)
        values = statistics(latent, support); outputs[steps] = values
        np.savez(out/f'statistics-{steps}.npz', ids=ids, **values)
        numerical[str(steps)] = dict(compile_and_first_seconds=first, warm_seconds=warm,
                                     maps_per_second=len(x)/warm)
        if steps == 32:
            reference = predictor(a.run, steps=32)(x)['base_energy']
            energy_error = float(np.max(abs(values['energy']-reference)))
            assert np.allclose(values['energy'],reference,rtol=3e-6,atol=.03)
            perm = model(x[::-1])[::-1]
            chunks = np.concatenate([model(x[:1]),model(x[1:7]),model(x[7:])])
            missing = x.copy(); missing[:,~support] = np.nan
            errors = [float(np.max(abs(latent-v))) for v in (perm,chunks,model(missing))]
            assert max(errors) < 1e-5
            assert model(x[:0]).shape == x[:0].shape
            bad = x[:1].copy(); bad[:,support] = np.nan
            try: model(bad)
            except ValueError: pass
            else: raise AssertionError('Nonfinite observed input accepted')
            np.savez(out/'example-fields.npz', ids=ids[:4], observed=x[:4], latent=latent[:4], support=support)
        print(json.dumps(dict(steps=steps, **numerical[str(steps)])),flush=True)
        del model; jax.clear_caches()
    for steps in (16,32):
        numerical[str(steps)]['statistic_error_vs64'] = {
            k: dict(median_absolute=float(np.median(abs(v-outputs[64][k]))),
                    maximum_absolute=float(np.max(abs(v-outputs[64][k]))),
                    median_over_normal_sd=float(np.median(abs(v-outputs[64][k]))/max(float(outputs[64][k].std()),1e-8)))
            for k,v in outputs[steps].items()}
    m = j.asarray([[1,1,0],[1,0,1]],j.float32); sigma=1.7
    g = j.asarray(np.random.default_rng(17).normal(size=(3,2,3)),j.float32)*m
    def velocity(v,t): return (1-sigma)/(sigma+(1-sigma)*t[:,None,None])*v
    check = jax.jit(lambda v: ode_latent(velocity,v,m,64))(g)
    gaussian_error = float(j.max(abs(check-g/sigma))); assert gaussian_error < 1e-3
    gaussian = np.random.default_rng(90252).normal(size=(a.count,*support.shape))*support
    gaussian_stats = statistics(gaussian,support)
    shuffled = gaussian.copy(); shuffled[:,support] = gaussian[:,support][:,::-1]
    assert np.allclose(statistics(shuffled,support)['ad'],gaussian_stats['ad'])
    np.savez(out/'gaussian-statistics.npz',**gaussian_stats)
    report = dict(kind='R6_NORMAL_ONLY_NUMERICAL_PILOT', public_test_read=False, synthetic_anomalies_read=False,
        count=len(ids), checkpoint_sha256=sha(Path(a.run)/'best.msgpack'), source_sha256=sha(__file__),
        gaussian_solver_max_error=gaussian_error, original_energy_max_error=energy_error,
        pointwise_errors=errors, numerical=numerical,
        normal_medians={k:float(np.median(v)) for k,v in outputs[32].items()},
        gaussian_medians={k:float(np.median(v)) for k,v in gaussian_stats.items()},
        interpretation='Comparison to masked Gaussian fields is a numerical and model-fit diagnostic, not an OOD result or a theoretical false-positive guarantee.')
    write_json(out/'report.json',report);print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('run','cache','out'): p.add_argument('--'+name,required=True)
    p.add_argument('--count',type=int,default=16)
    main(p.parse_args())
