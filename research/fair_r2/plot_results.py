"""Export research figures and paired uncertainty, with explicit proxy labels."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import low_fpr_metric,write_json

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--comparison',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root);comparison=Path(a.comparison);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
r=json.loads((comparison/'report.json').read_text())['results'];m=json.loads((root/'morphology-evaluations/report.json').read_text())['results']
names=['r1/physics-distance','flow-cosmology/mean_energy','flow88-cosmology/mean_energy','flow216-cosmology/mean_energy']
labels=['R1: cluster distance','R2: 3-flow ensemble','R2: +16 wavelet PCs (1 seed)','R2: +144 wavelets (1 seed)']
families=['development/spectral_low:0.2','development/spectral_high:0.2','development/quadratic_peak:0.2','r1_novel_now_reused/anisotropic:0.2']
columns=['Mild smoothing','Mild sharpening','Enhanced peaks','Mild anisotropy','IAAFT structure','Reduced peaks']
matrix=[]
for name in names:
    row=[r[name]['probes'][f] for f in families]
    mn={'r1/physics-distance':'r1-physics-distance','flow-cosmology/mean_energy':'ensemble-cosmology','flow88-cosmology/mean_energy':'flow88-cosmology-s0','flow216-cosmology/mean_energy':'flow216-cosmology-s0'}[name]
    for f in ('phase_iaaft:1.0','negative_peak:0.2'):
        v=m[mn][f];row.append(v['minimum_base_energy'] if isinstance(v,dict) else v)
    matrix.append(row)
matrix=np.asarray(matrix);fig,ax=plt.subplots(figsize=(11,4.1),layout='constrained');im=ax.imshow(matrix,cmap='viridis',vmin=0,vmax=.6)
ax.set_xticks(np.arange(len(columns)),columns,rotation=20,ha='right');ax.set_yticks(np.arange(len(labels)),labels)
for y in range(len(labels)):
    for x in range(len(columns)):ax.text(x,y,f'{matrix[y,x]:.3f}',ha='center',va='center',color='white' if matrix[y,x]<.32 else 'black')
ax.set_title('Synthetic probe detection — not competition performance',pad=12)
ax.axvline(3.5,color='white',ls='--',lw=2)
fig.colorbar(im,ax=ax,label='Mean TPR at 100 log-spaced FPRs (0.001–0.05)')
fig.savefig(out/'proxy-comparison.png',dpi=180);fig.savefig(out/'proxy-comparison.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,3.8),layout='constrained')
for name,label in zip(names[:2],labels[:2]):
    values=np.asarray(r[name]['normal']['per_cosmology_counts_at_1pct']);ax.plot(np.arange(101),values,'o-',ms=3,lw=.7,label=label)
ax.set(xlabel='Cosmology index',ylabel='Flagged normal maps out of 32',title='Reused normal audit: false positives at a calibration-set 1% threshold')
ax.axhline(.32,ls='--',c='black',lw=1,label='Expected average if calibrated');ax.legend();fig.savefig(out/'normal-cosmology-audit.png',dpi=180);plt.close(fig)

base=np.load(comparison/'r1__physics-distance.npz');flow=np.load(comparison/'flow-cosmology__mean_energy.npz')
bm=np.load(root/'morphology-evaluations/r1-physics-distance.npz');fm=np.load(root/'morphology-evaluations/ensemble-cosmology.npz')
rows=[];rng=np.random.default_rng(29101)
for name in families+['phase_iaaft:1.0','negative_peak:0.2']:
    b,f=(base[name],flow[name]) if '/' in name else (bm[name],fm[name]);samples=[]
    for _ in range(400):
        # Same calibration and positive resamples for both models. Cosmology
        # proportions in the normal reference remain exactly 32/3232 each.
        ci=(np.arange(101)[:,None]*32+rng.integers(32,size=(101,32))).ravel();pi=rng.integers(len(b),size=len(b))
        samples.append(low_fpr_metric(flow['calibration'][ci],f[pi])-low_fpr_metric(base['calibration'][ci],b[pi]))
    rows.append(dict(family=name,baseline=low_fpr_metric(base['calibration'],b),ensemble=low_fpr_metric(flow['calibration'],f),paired_delta_95pct_percentile_interval=np.quantile(samples,[.025,.975]).tolist()))
write_json(out/'paired-bootstrap.json',dict(kind='CONDITIONAL_UNCERTAINTY_ON_SYNTHETIC_PROBES',resamples=400,limitations='Does not cover model selection, repeated-audit use, simulator mismatch, or unknown official OOD physics.',results=rows))
print(json.dumps(rows,indent=2))
