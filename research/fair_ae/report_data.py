"""Aggregate observed artifacts and paired cosmology-block bootstrap diagnostics."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def metric(normal,anomaly):
    s=np.r_[normal,anomaly];y=np.r_[np.zeros(len(normal)),np.ones(len(anomaly))]
    ix=np.argsort(-s,kind='stable');s=s[ix];y=y[ix]
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];tp=np.cumsum(y)[ends];fp=ends+1-tp
    return float(np.interp(np.geomspace(.001,.05,100),np.r_[0,fp/len(normal)],np.r_[0,tp/len(anomaly)]).mean())

p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
base=Path(a.snapshot);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
runs={}
for host in [0,1,2]:
    for run in (base/f'host{host}').iterdir():
        if run.is_dir() and (run/'density/report.json').exists() and not run.name.startswith('screen-'):
            runs[run.name]=run
rows=[];proxy=[]
for name,run in runs.items():
    cfg=json.loads((run/'config.json').read_text());b=json.loads((run/'best.json').read_text()) if (run/'best.json').exists() else {};complete=json.loads((run/'complete.json').read_text()) if (run/'complete.json').exists() else {}
    density=json.loads((run/'density/report.json').read_text());probe=json.loads((run/'linear_probe.json').read_text()) if (run/'linear_probe.json').exists() else {}
    cal=np.load(run/'density/calibration_scores.npy');normal=np.load(run/'density/audit_scores.npy');thresholds=np.quantile(cal,[.95,.99,.999])
    row=dict(name=name,architecture=cfg.get('architecture','global'),best_step=b.get('step'),trained_steps=complete.get('steps'),training_seconds=complete.get('seconds'),reconstruction_nmse=b.get('reconstruction_nmse') if cfg.get('reconstruction_weight',1)>0 else None,summary_nmse=b.get('clean_summary_nmse'),cluster_k=density['selected_k'],effective_rank=probe.get('correlation_effective_rank'))
    for i,q in enumerate(['95','99','999']):
        row['normal_audit_exceedance_'+q]=int((normal>thresholds[i]).sum())
    row['max_cosmology_count_at_1pct']=int((normal.reshape(101,32)>thresholds[1]).sum(1).max())
    pm=probe.get('audit_normalized_mse',[None]*5);row.update(probe_omega_m_mse=pm[0],probe_s8_mse=pm[1])
    for phase in ['development','novel']:
        # Exported selected copies include the once-only held-out results.
        selected=base/'host0/final-components'/name
        report=run/f'{phase}_diagnostics.json'
        if not report.exists() and (selected/report.name).exists():report=selected/report.name
        if report.exists():
            d=json.loads(report.read_text())
            if phase=='development':row['noise_rank_correlation']=d['noise_rank_correlation']
            for f in d['families']:proxy.append(dict(name=name,phase=phase,**f))
    rows.append(row)
ensemble_path=base/'host0/ensemble-audit.json'
if ensemble_path.exists():
    e=json.loads(ensemble_path.read_text());row={'name':'complementary-ensemble','architecture':'ensemble'}
    for q,n in zip(['95','99','999'],e['normal_audit_exceedance_counts']):row['normal_audit_exceedance_'+q]=n
    rows.append(row)
    for phase in ['development','novel']:
        for key,value in e[phase].items():
            if key not in ['normal','second']:proxy.append(dict(name='complementary-ensemble',phase=phase,family=key,one_sided=value))
for name,data in [('normal-and-training.csv',rows),('synthetic-diagnostics.csv',proxy)]:
    fields=list(dict.fromkeys(k for r in data for k in r))
    with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)

# Paired block resampling across the 101 cosmologies; not training-seed variance.
normal_paths=[runs['physics'],runs['physics-distance']]
cal=[np.load(q/'density/calibration_scores.npy') for q in normal_paths]
with np.load(base/'host0/ae-z32/splits.npz') as f:splits={k:f[k] for k in f.files}
intervals=[]
for phase,seed,split in [('development',474,'val'),('novel',475,'audit')]:
    scores=[np.load(q/f'{phase}_diagnostic_scores.npz') for q in normal_paths]
    ids=np.random.default_rng(seed).permutation(splits[split])[:512];groups=ids//256
    for family in scores[0].files:
        if family in ['normal','second']:continue
        observed=[metric(c,s[family]) for c,s in zip(cal,scores)]
        for run,value in zip(normal_paths,observed):
            expected=next(x['one_sided'] for x in json.loads((run/f'{phase}_diagnostics.json').read_text())['families'] if x['family']==family)
            if not np.isclose(value,expected,rtol=0,atol=1e-12):raise ValueError('Recomputed metric differs from original artifact')
        rng=np.random.default_rng(882);diff=[]
        for _ in range(300):
            selected=rng.integers(0,101,101);ci=(selected[:,None]*32+np.arange(32)).ravel();ai=np.concatenate([np.flatnonzero(groups==g) for g in selected])
            m=[metric(c[ci],s[family][ai]) for c,s in zip(cal,scores)];diff.append(m[1]-m[0])
        intervals.append(dict(phase=phase,family=family,physical_likelihood=observed[0],cluster_distance=observed[1],difference=observed[1]-observed[0],difference_interval_95=np.quantile(diff,[.025,.975]).tolist()))
(out/'paired-bootstrap.json').write_text(json.dumps({'kind':'300 paired cosmology-block bootstrap replicates; synthetic probes only; not seed variance or official test uncertainty','results':intervals},indent=2))
(out/'summary.json').write_text(json.dumps({'models':rows,'synthetic':proxy},indent=2))
print(json.dumps({'models':len(rows),'synthetic_rows':len(proxy),'report_dir':str(out)}))
