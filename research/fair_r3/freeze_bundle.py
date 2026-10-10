"""Freeze R3 artifacts using only recorded normal/synthetic selection evidence."""
import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scale(path):
    with np.load(path) as f: med, upper = np.quantile(f['calibration'], [.5,.9])
    return [float(med), float(upper-med)]


def main(a):
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=False)
    source = root/'research'; evidence = root/'.fair/r3-evidence'
    r2 = root/'.fair/deliverables/fair-r2-hybrid-2026-10-09/reproduce'
    old = json.loads((r2/'hybrid-manifest.json').read_text())
    report = json.loads((evidence/'final-comparison-001/report.json').read_text())
    model_names = [v['selected'] for v in report['noise_training_selection']]
    assert model_names == [f'flow-noise4-long-s{i}' for i in range(3)]
    for name in model_names:
        src = evidence/'long-training-001'/name
        dst = out/'models'/name; dst.mkdir(parents=True)
        for filename in ('config.json','normalization.npz','best.msgpack','best.json','source.py'):
            shutil.copy2(src/filename,dst)
    for name in ('field-s0','field-s1'):
        shutil.copytree(r2/'models'/name,out/'models'/name)
    for src, name in ((r2/'mask.npy','mask.npy'),
                      (r2/'nll-normal-regressors.npz','field-regressors.npz'),
                      (evidence/'spatial-residuals-001/regional-residual-model.npz','spatial-residual.npz'),
                      (evidence/'regional-calibration-001/normal-regressors.npz','spatial-centering.npz')):
        shutil.copy2(src,out/name)
    code = {
        'fair_ae':['features.py'],
        'fair_r2':['flow.py','field_flow.py','field_evaluate.py','common.py','condition_field_scores.py'],
        'fair_r3':['frozen_detector.py','export_frozen.py','predict_frozen.py','spatial_features.py',
                   'spatial_residuals.py','calibrate_spatial.py','compare.py']}
    for folder, names in code.items():
        dst = out/'code'/folder; dst.mkdir(parents=True)
        for name in names: shutil.copy2(source/folder/name,dst/name)
    files = {str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    manifest = dict(format='FAIR_R3_POINTWISE_V1',frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        summary_models=model_names,summary_weights=[1/3]*3,field_models=['field-s0','field-s1'],
        ode_steps=32,trace_seed=451,trace_probes=1,
        conditional_std_floor=old['conditional_std_floor'],
        summary_normal_center_scale=scale(evidence/'final-comparison-001/noise-long-summary.npz'),
        field_normal_center_scale=old['field_normal_center_scale'],
        hybrid_normal_center_scale=scale(evidence/'final-comparison-001/noise-long-field.npz'),
        regional_normal_center_scale=scale(evidence/'regional-calibration-001/regional-centered.npz'),
        public_test_sha256=old['test_sha256'],
        frozen_field_cache_sha256=digest(root/'.fair/r2-evidence/checkpoint-005/hybrid-exports/component_scores.npz'),
        files=files,
        score_definitions={
            '06':'Equal average of normal-scaled noise-trained summary energy and frozen power-conditioned absolute field-likelihood residual.',
            '07':'Maximum of normal-scaled candidate 06 and the normal-centered regional Mahalanobis residual score.'},
        selection='Noise models selected by normal validation NLL; spatial model and centering use normal-only fitting/CV; ensemble choices examined reused synthetic diagnostics and fresh-noise pattern tests. No public test fitting, pooled statistics or selection.',
        official_score=None,submitted=False,
        limitations=['Synthetic shifts are not the unknown competition physical alternatives.',
                     'The private-evaluation interface is not yet validated.',
                     'Runtime tested on the authorized TPU environment; a clean reinstall and CPU numerical equivalence remain unverified.',
                     'Candidate 07 trades some sensitivity on other shifts for stronger spatial nonstationarity detection.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    print(json.dumps(dict(bundle=str(out),files=len(files),manifest_sha256=digest(out/'manifest.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
