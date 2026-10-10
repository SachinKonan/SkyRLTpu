"""Research-only plots and paired sampling intervals; no public predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import low_fpr_metric, write_json


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def main(a):
    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    summary = load(root/'comparison-003/flow-cosmology__mean_energy.npz')
    baseline = load(root/'comparison-003/r1__physics-distance.npz')
    for values, name in [(summary,'ensemble-cosmology'), (baseline,'r1-physics-distance')]:
        values.update({'morphology/'+k:v for k,v in
                       load(root/f'checkpoint-004/morphology-evaluations/{name}.npz').items()})
        values['reused/anisotropic:0.2'] = values['r1_novel_now_reused/anisotropic:0.2']
    raw = load(root/'field-comparison-001/field-ensemble__nll.npz')
    corrected = load(root/'field-conditioned-001/nll__hetero-absolute.npz')
    hybrid = load(root/'field-summary-ensembles-001/nll__hetero-absolute-equal.npz')
    families = ['development/spectral_low:0.2','development/spectral_high:0.2',
                'development/quadratic_peak:0.2','reused/anisotropic:0.2',
                'morphology/phase_iaaft:1.0','morphology/negative_peak:0.2']
    labels = ['R1 cluster distance','04: summary-flow ensemble','Raw field likelihood',
              'Power-conditioned field residual','05: equal summary + field ensemble']
    values = [baseline,summary,raw,corrected,hybrid]
    matrix = np.array([[low_fpr_metric(v['calibration'],v[f]) for f in families] for v in values])
    fig, ax = plt.subplots(figsize=(12,4.5),layout='constrained')
    im = ax.imshow(matrix,vmin=0,vmax=.8,cmap='viridis')
    ax.set_xticks(range(6),['Mild smoothing','Mild sharpening','Enhanced peaks',
                            'Mild anisotropy','IAAFT structure','Reduced peaks'],rotation=18,ha='right')
    ax.set_yticks(range(5),labels)
    for i in range(5):
        for j in range(6):
            ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',
                    color='black' if matrix[i,j]>.42 else 'white')
    ax.set_title('Synthetic probes: field likelihood helps only after background control',pad=14)
    fig.colorbar(im,ax=ax,label='Mean TPR at FPR 0.001–0.05; NOT leaderboard scores')
    fig.savefig(out/'field-ensemble-comparison.png',dpi=180)
    fig.savefig(out/'field-ensemble-comparison.pdf')
    plt.close(fig)
    rng = np.random.default_rng(202610095)
    records = []
    for f in families:
        differences = []
        for _ in range(400):
            ci = (np.arange(101)[:,None]*32+rng.integers(32,size=(101,32))).ravel()
            pi = rng.integers(len(summary[f]),size=len(summary[f]))
            differences.append(low_fpr_metric(hybrid['calibration'][ci],hybrid[f][pi])-
                               low_fpr_metric(summary['calibration'][ci],summary[f][pi]))
        records.append(dict(family=f,summary=low_fpr_metric(summary['calibration'],summary[f]),
                            hybrid=low_fpr_metric(hybrid['calibration'],hybrid[f]),
                            delta_95pct_interval=np.quantile(differences,[.025,.975]).tolist()))
    write_json(out/'report.json',dict(kind='REUSED_SYNTHETIC_PROBES_NOT_OFFICIAL_OOD',
                                      labels=labels,families=families,matrix=matrix.tolist(),
                                      resamples=400,paired_intervals=records,
                                      limitations='Sampling intervals condition on the fitted models, observed cosmologies and chosen artificial shifts. They exclude selection bias and unknown OOD physics.'))
    print(json.dumps(records,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--out',required=True)
    main(p.parse_args())
