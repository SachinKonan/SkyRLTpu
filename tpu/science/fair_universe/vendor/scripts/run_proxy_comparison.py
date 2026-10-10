"""Reproducible CPU-sized synthetic comparison, not FAIR leaderboard results.

Train on normal maps only. Select a scoring variant on development anomalies.
Evaluate that frozen selection on independently seeded holdout maps and three
additional anomaly families. Save all scores, checkpoints, configs and hashes.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse, hashlib, json, platform, time
import numpy as np
import jax
from fairdiscover.config import read_config, noise_sigma
from fairdiscover.data import stage, MapStore, DataView
from fairdiscover.shifts import proxy_shift, FAMILIES
from fairdiscover.metric import per_family_scores
from fairdiscover.worker import load_candidate, code_hash, checked_predict
from fairdiscover.submission import export_submission


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def make_fields(out, seed, nreal, h=64, w=32):
    """Toy masked Gaussian/lognormal random fields; NOT cosmology simulations."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    mask = np.ones((h, w), bool)
    mask[:3] = False; mask[-3:] = False; mask[:, :2] = False; mask[:, -2:] = False
    mask[17:22, 11:15] = False; mask[38:42, 23:26] = False
    ky = np.fft.fftfreq(h)[:, None]; kx = np.fft.fftfreq(w)[None, :]
    k2 = ky*ky + kx*kx
    fields = np.empty((16, nreal, int(mask.sum())), np.float16)
    labels = np.zeros((16, nreal, 5), np.float32)
    for c in range(16):
        a, b = divmod(c, 4)
        amplitude = .014 + .003*a; slope = 1.1 + .3*b
        kernel = (k2 + .04**2)**(-slope/4.) * np.exp(-k2/.35**2)
        kernel[0, 0] = 0.; kernel /= np.sqrt(np.mean(kernel**2))
        nonlinearity = .24 + .09*a
        for start in range(0, nreal, 64):
            m = min(64, nreal-start)
            white = rng.normal(size=(m,h,w))
            g = np.fft.ifft2(np.fft.fft2(white)*kernel).real
            lognormal = (np.expm1(nonlinearity*g)-np.expm1(.5*nonlinearity**2))/nonlinearity
            signal = amplitude*lognormal + .002*rng.normal(size=(m,h,w))
            fields[c,start:start+m] = signal[:,mask]
        labels[c,:,:] = [.22+.04*a, .70+.06*b, 7.5, .01, 0.]
    np.save(out/'sampled_WIDE12H_bin2_2arcmin_kappa.npy', fields)
    np.save(out/'sampled_label.npy', labels)
    np.save(out/'WIDE12H_bin2_2arcmin_mask.npy', mask)
    return dict(seed=seed, shape=list(fields.shape),
                map_sha256=digest(out/'sampled_WIDE12H_bin2_2arcmin_kappa.npy'))


def extra_shift(clean, mask, family, rng):
    from scipy.ndimage import gaussian_filter
    x = clean.astype(np.float32)
    if family == 'anisotropic_smoothing':
        y = gaussian_filter(x,(2.8,.5))/np.maximum(gaussian_filter(mask.astype(np.float32),(2.8,.5)),.05)
    elif family == 'localized_peak':
        rows, cols = np.where(mask); c = rng.integers(len(rows)); hh, ww = np.indices(mask.shape)
        bump = np.exp(-((hh-rows[c])**2+(ww-cols[c])**2)/8.)
        y = x + 5.*max(float(x[mask].std()),1.e-6)*bump
    elif family == 'phase_randomization':
        ph = np.fft.rfft2(rng.normal(size=x.shape)); ph /= np.maximum(np.abs(ph),1.e-12)
        y = np.fft.irfft2(np.abs(np.fft.rfft2(x))*ph,s=x.shape)
    else: raise ValueError(family)
    y = (y-y[mask].mean())*(max(float(x[mask].std()),1.e-7)/max(float(y[mask].std()),1.e-7))+x[mask].mean()
    return (y*mask).astype(np.float32)


def evaluate(store, ids, models, cfg, families, seed, output):
    rng = np.random.default_rng(seed)
    names = ['autoencoder','spectral','global','conditional','neighbor','ensemble','max_ensemble']
    scores = {n:{f:[] for f in ['ind']+list(families)} for n in names}
    timings = {'autoencoder':0.,'spectral':0.,'multiscale_shared':0.}
    def append(batch, family):
        for n in ['autoencoder','spectral']:
            mod, model = models[n]; t = time.perf_counter()
            val = checked_predict(mod, model, batch, store.mask, cfg)
            timings[n] += time.perf_counter()-t; scores[n][family].extend(val.tolist())
        mod, model = models['multiscale']; t = time.perf_counter()
        vals = mod.all_scores(model,batch,store.mask,cfg)
        timings['multiscale_shared'] += time.perf_counter()-t
        for n,val in vals.items():
            n = 'global' if n == 'global_score' else n
            if not np.isfinite(val).all(): raise ValueError('Non-finite output')
            scores[n][family].extend(np.asarray(val).tolist())
    bs = int(cfg['infer_batch_size'])
    for start in range(0,len(ids),bs):
        selected = ids[start:start+bs]; clean = store.get_clean(selected)
        append(store.apply_noise(clean,rng,cfg),'ind')
        for fi,f in enumerate(families):
            altered = np.empty_like(clean)
            for j,pair in enumerate(selected):
                r = np.random.default_rng(np.random.SeedSequence([seed,830+fi,int(pair[0]),int(pair[1])]))
                altered[j] = proxy_shift(clean[j],store.mask,f,r,1.) if f in FAMILIES else extra_shift(clean[j],store.mask,f,r)
            append(store.apply_noise(altered,rng,cfg),f)
        print(f'EVALUATE {output.name} {min(start+bs,len(ids))}/{len(ids)} normal maps',flush=True)
    report = {}; packed = {}
    for n,data in scores.items():
        vals = per_family_scores(data['ind'],{f:data[f] for f in families})
        report[n] = dict(mean_score=float(np.mean(list(vals.values()))),per_family=vals)
        for f,a in data.items(): packed[n+'__'+f] = np.asarray(a,np.float64)
    report['_timing_seconds'] = timings
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'scores.npz',**packed)
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    return report,packed


def paired_bootstrap(scores, winner, families, repetitions=300):
    rng = np.random.default_rng(615923); n = len(scores[winner+'__ind'])
    means = {m:[] for m in [winner,'spectral','autoencoder']}
    for _ in range(repetitions):
        ids = rng.integers(0,n,n)
        for model in means:
            vals = per_family_scores(scores[model+'__ind'][ids],{f:scores[model+'__'+f][ids] for f in families})
            means[model].append(np.mean(list(vals.values())))
    result = dict(method='paired-map percentile bootstrap, not multiple-testing corrected',repetitions=repetitions)
    for n,v in means.items(): result[n+'_95pct_interval'] = np.quantile(v,[.025,.975]).tolist()
    for n in ['spectral','autoencoder']:
        result['winner_minus_'+n+'_95pct_interval'] = np.quantile(np.asarray(means[winner])-np.asarray(means[n]),[.025,.975]).tolist()
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--data-dir',type=Path,required=True)
    a = p.parse_args(); out = a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    data = a.data_dir.resolve(); data.mkdir(parents=True,exist_ok=True)
    # Generate and hash both datasets BEFORE fitting/selecting any candidate.
    manifest = dict(development=make_fields(data/'development_raw',70001,512),holdout=make_fields(data/'holdout_raw',90001,64))
    stage(str(data/'development_raw'),str(data/'development_stage'),mode='sample',seed=402)
    stage(str(data/'holdout_raw'),str(data/'holdout_stage'),mode='sample',seed=403)
    dev = MapStore(data/'development_stage'); test = MapStore(data/'holdout_stage')
    cfg = read_config(); cfg.update(steps=250,batch_size=64,infer_batch_size=128,hidden_dim=128,calibration_max=1024,seed=31415,evolved_mode='conditional')
    (out/'config.json').write_text(json.dumps(cfg,indent=2)); (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    models = {}; fit_reports = {}
    for name,filename in [('autoencoder','baseline_jax.py'),('spectral','spectral_jax.py'),('multiscale','multiscale_jax.py')]:
        candidate = ROOT/'candidates'/filename; mod = load_candidate(candidate)
        run = out/'runs'/name; run.mkdir(parents=True,exist_ok=True)
        (run/'trial_config.json').write_text(json.dumps(cfg,indent=2)); t = time.perf_counter()
        info = mod.fit(DataView(dev,'train',cfg),DataView(dev,'val',cfg),cfg,run/'artifacts')
        meta = dict(candidate_sha256=code_hash(candidate),train_seconds=time.perf_counter()-t,fit_info=info,train_count=len(dev.ids('train')),val_count=len(dev.ids('val')))
        (run/'train_meta.json').write_text(json.dumps(meta,indent=2)); fit_reports[name] = meta
        models[name] = (mod,mod.load_model(run/'artifacts',cfg))
        print('TRAINED',name,json.dumps(meta),flush=True)
    development,_ = evaluate(dev,dev.ids('reward'),models,cfg,FAMILIES,12001,out/'development')
    names = [n for n in development if not n.startswith('_')]
    winner = max(names,key=lambda n:development[n]['mean_score'])
    decision = dict(selected=winner,selection_criterion='highest development mean',development_score=development[winner]['mean_score'],all_development_scores={n:development[n]['mean_score'] for n in names},holdout_consulted_for_selection=False)
    (out/'selection.json').write_text(json.dumps(decision,indent=2))
    print('FROZEN SELECTION',json.dumps(decision),flush=True)
    ids = np.asarray([(c,r) for c in range(16) for r in range(64)],np.int32)
    families = tuple(FAMILIES)+('anisotropic_smoothing','localized_peak','phase_randomization')
    holdout,packed = evaluate(test,ids,models,cfg,families,22001,out/'holdout')
    uncertainty = paired_bootstrap(packed,winner,FAMILIES)
    (out/'bootstrap.json').write_text(json.dumps(uncertainty,indent=2))
    summary = dict(selection=decision,fits=fit_reports,uncertainty=uncertainty,
        hardware=dict(python=sys.version,platform=platform.platform(),jax=jax.__version__,devices=[str(d) for d in jax.devices()]),
        test_cases=dict(normal_dev=len(dev.ids('reward')),normal_holdout=len(ids),map_shape=list(test.mask.shape),train=len(dev.ids('train')),val=len(dev.ids('val')),selection_anomaly_families=list(FAMILIES),novel_stress_test_families=list(families[4:]),noise_sigma=noise_sigma(cfg)),
        result_type='SYNTHETIC PROXY ONLY; not FAIR data or leaderboard results',
        holdout={n:dict(same_family_mean=float(np.mean([holdout[n]['per_family'][f] for f in FAMILIES])),novel_family_mean=float(np.mean([holdout[n]['per_family'][f] for f in families[4:]])),per_family=holdout[n]['per_family']) for n in names})
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    final_name = winner if winner in ['autoencoder','spectral'] else 'multiscale'
    filename = dict(autoencoder='baseline_jax.py',spectral='spectral_jax.py',multiscale='multiscale_jax.py')[final_name]
    run = out/'runs'/final_name; selected_cfg = dict(cfg,evolved_mode=winner)
    (run/'trial_config.json').write_text(json.dumps(selected_cfg,indent=2))
    noisy = test.get_noisy(ids[:16],np.random.default_rng(631),cfg)
    test_file = data/'synthetic_submission_inputs.npy'; np.save(test_file,noisy)
    package = export_submission(ROOT/'candidates'/filename,run,data/'holdout_stage',test_file,out/'DO_NOT_SUBMIT_synthetic_format_check.zip',batch_size=8,expected_count=16)
    (out/'submission_format_check.json').write_text(json.dumps(package,indent=2))
    print('DONE',json.dumps(summary),flush=True)

if __name__ == '__main__': main()
