"""Normal-only covariance audit and post-fit calibration of conditional flows.

Covariance/mean fitting uses normal validation latents at known training
cosmologies. Shrinkage selection uses five held-cosmology validation folds.
Calibration/audit maps and synthetic probes only evaluate the resulting score.
No public-test inputs are used. This is not an exact chi-square test.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from flax import serialization
from flow import AffineFlow
from common import load_labels, splits, score_report, low_fpr_metric, write_json, sha


def fit_gaussian(z, alpha):
    mu = z.mean(0)
    cov = np.cov(z, rowvar=False)
    cov = (1-alpha)*cov + alpha*np.eye(z.shape[1])
    chol = np.linalg.cholesky(cov)
    return mu, np.linalg.inv(chol.T), 2*np.log(np.diag(chol)).sum()


def make_model(run):
    cfg = json.loads((run/'config.json').read_text())
    with np.load(run/'normalization.npz') as f:
        norm = {k:f[k] for k in f.files}
    assert cfg['condition'] == 'cosmology' and cfg['transform'] == 'asinh'
    params = jax.tree.map(j.asarray, serialization.msgpack_restore((run/'best.msgpack').read_bytes()))
    net = AffineFlow(cfg['dim'],cfg['layers'],cfg['width'],cfg['scale_bound'])
    apply = jax.jit(lambda x,c:net.apply({'params':params},x,c,return_latent=True))
    def encode(features, theta):
        value = np.arcsinh((features-norm['target_offset'])/norm['target_scale'])
        x = (value-norm['mean'])@norm['whiten']
        c = (theta-norm['condition_mean'])/norm['condition_std']
        return np.asarray(apply(j.asarray(x,j.float32),j.asarray(c,j.float32)))
    return encode, norm['cosmology_grid']


def main(a):
    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    labels = load_labels(a.raw)
    ix = splits()
    features = {k:np.load(root/'features72'/f'{k}_latent.npy').astype(np.float64)
                for k in ('val','calibration','audit')}
    for prefix,path in [('development',root/'probes72/development.npz'),
                        ('reused',root/'probes72/r1_novel_now_reused.npz'),
                        ('morphology',root/'morphology-probes/features72/morphology_r2.npz')]:
        with np.load(path) as f:
            features.update({prefix+'/'+k:f[k].astype(np.float64) for k in f.files if k!='ids'})
    fold_assignment = np.empty(101,int)
    fold_assignment[np.random.default_rng(782).permutation(101)] = np.arange(101)%5
    folds = fold_assignment[ix['val']//256]
    results = []
    reports = []
    start = time.monotonic()
    for seed in range(3):
        run = root/f'flow-cosmology-asinh-s{seed}'
        encode, grid = make_model(run)
        val = np.concatenate([encode(features['val'][lo:lo+128], labels[ix['val'][lo:lo+128],:2])
                              for lo in range(0,len(ix['val']),128)])
        candidates = []
        for alpha in (0.,.01,.05,.1,.25,.5,1.):
            losses = np.empty(len(val))
            for fold in range(5):
                train = folds != fold
                mu, w, logdet = fit_gaussian(val[train],alpha)
                losses[~train] = .5*(((val[~train]-mu)@w)**2).sum(1)+.5*logdet
            candidates.append(dict(alpha=alpha,normal_cv_nll=float(losses.mean())))
        alpha = min(candidates,key=lambda r:r['normal_cv_nll'])['alpha']
        mu,w,logdet = fit_gaussian(val,alpha)
        np.savez(out/f'calibration-s{seed}.npz',mean=mu,whiten=w,alpha=alpha)
        seed_result = {}
        for key,f in features.items():
            if key == 'val':
                continue
            values = []
            for lo in range(0,len(f),32):
                b = f[lo:lo+32]
                z = encode(np.repeat(b,len(grid),axis=0),np.tile(grid,(len(b),1)))
                energy = (((z.astype(np.float64)-mu)@w)**2).sum(1).reshape(len(b),len(grid)).min(1)
                values.append(energy)
            seed_result[key] = np.concatenate(values)
        np.savez(out/f'scores-s{seed}.npz',**seed_result)
        eigen = np.linalg.eigvalsh(np.cov(val,rowvar=False))
        corr = np.corrcoef(val,rowvar=False)
        record = dict(seed=seed,model_sha256=sha(run/'best.msgpack'),selected_alpha=alpha,
                      cv=candidates,latent_mean_norm=float(np.linalg.norm(val.mean(0))),
                      covariance_eigenvalue_range=[float(eigen.min()),float(eigen.max())],
                      offdiagonal_correlation_rms=float(np.sqrt(np.mean(corr[~np.eye(len(corr),dtype=bool)]**2))),
                      expected_covariance='Identity for a correctly estimated conditional Gaussianizing flow',
                      seconds=time.monotonic()-start)
        reports.append(record)
        results.append(seed_result)
        print(json.dumps(record),flush=True)
        del encode
        jax.clear_caches()
    values = {k:np.mean([r[k] for r in results],axis=0) for k in results[0]}
    np.savez(out/'ensemble_scores.npz',**values)
    report = dict(kind='NORMAL_COVARIANCE_CALIBRATION_WITH_REUSED_SYNTHETIC_DIAGNOSTICS',
                  fit_source='3232 NORMAL validation maps at their known cosmology',models=reports,
                  normal=score_report({'score':values['calibration']},{'score':values['audit']})['score'],
                  probes={k:low_fpr_metric(values['calibration'],v) for k,v in values.items() if '/' in k},
                  limitations=['Synthetic probes have been reused in research.',
                               'Normal-validation post-fitting and shrinkage can remove genuine anomaly directions.',
                               'Per-map scores profile the discrete training cosmology grid, not a continuous posterior.'])
    write_json(out/'report.json',report)
    print(json.dumps(report['probes']),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','raw','out'):
        p.add_argument('--'+name,required=True)
    main(p.parse_args())
