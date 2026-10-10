"""Frozen fresh-noise scores, measured against previously frozen normal references."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(1, str(Path(__file__).resolve().parents[1]/'fair_r3'))
from compare import read_npz, paired_bootstrap
from common import low_fpr_metric, write_json, sha
from condition_field_scores import predict


def main(a):
    root, r4, r3, out = map(Path,(a.root,a.r4,a.r3,a.out));out.mkdir(parents=True,exist_ok=False)
    cache=root/'closure-cache';manifest=json.loads((cache/'manifest.json').read_text())
    for path in cache.glob('host*-manifest.json'):
        other=json.loads(path.read_text())
        assert other['observations_sha256']==manifest['observations_sha256']
        assert other['ids_sha256']==manifest['ids_sha256']
    group=manifest['groups'];features=np.load(cache/'features165.npy')[:,:72].astype(np.float64)
    n=len(features);nll=[];sources={}
    for model in range(2):
        full=np.empty(n);seen=np.zeros(n,int)
        for shard in range(8):
            path=root/f'field-s{model}-{shard}'
            r=json.loads((path/'complete.json').read_text());v=read_npz(path/'scores.npz')
            assert sha(path/'scores.npz')==r['scores_sha256']
            assert r['count']==len(v['indices'])
            full[v['indices']]=v['nll'];seen[v['indices']]+=1;sources[path.name]=r
        assert np.all(seen==1) and np.isfinite(full).all();nll.append(full)
    nll=np.mean(nll,axis=0)
    bundle=Path(a.bundle);b=json.loads((bundle/'base/manifest.json').read_text());m=json.loads((bundle/'r4-manifest.json').read_text())
    with np.load(bundle/'base/field-regressors.npz',allow_pickle=False) as f:
        mean={k.removeprefix('mean_'):f[k] for k in f.files if k.startswith('mean_')}
        var={k.removeprefix('variance_'):f[k] for k in f.files if k.startswith('variance_')}
    mean['kind']=str(mean['kind']);var['kind']=str(var['kind'])
    sd=np.maximum(np.exp(np.clip(predict(features,var),-30,30)/2),b['conditional_std_floor'])
    field=abs((nll-predict(features,mean))/sd)
    comp=read_npz(root/'closure-summary/scores.npz')
    def norm(x,scale):return (x-scale[0])/scale[1]
    fn=norm(field,b['field_normal_center_scale']);region=norm(comp['regional'],b['regional_normal_center_scale'])
    values={'06-grid-hybrid':.5*(norm(comp['grid'],b['summary_normal_center_scale'])+fn)}
    values['07-grid-spatial']=np.maximum(norm(values['06-grid-hybrid'],b['hybrid_normal_center_scale']),region)
    for mode in ('continuous','centered','scaled'):
        values[mode+'-hybrid']=.5*(norm(comp[mode],m['normal_center_scale'][mode])+fn)
        values[mode+'-spatial']=np.maximum(norm(values[mode+'-hybrid'],m['normal_center_scale'][mode+'-hybrid']),region)
    calibration={
        '06-grid-hybrid':read_npz(r3/'final-comparison-001/noise-long-field.npz')['calibration'],
        '07-grid-spatial':read_npz(r3/'regional-calibration-001/hybrid+centered-max.npz')['calibration']}
    for mode in ('continuous','centered','scaled'):
        for suffix in ('hybrid','spatial'):
            name=mode+'-'+suffix;calibration[name]=read_npz(r4/'score-calibration-001'/f'{name}.npz')['calibration']
    def group_values(value,g):return value[group[g]['offset']:group[g]['offset']+group[g]['count']]
    parts={name:{'calibration':calibration[name],**{g:group_values(v,g) for g in group}} for name,v in values.items()}
    results={};precision={};differences={}
    for name,v in parts.items():
        threshold=np.quantile(v['calibration'],[.95,.99,.999])
        results[name]=dict(normal={g:dict(count=len(v[g]),exceedances=[int((v[g]>t).sum()) for t in threshold])
                                  for g in ('normal_float32','normal_float16')},
                           probes={g:low_fpr_metric(v['calibration'],v[g]) for g in group if ':' in g})
        error=v['normal_float16']-v['normal_float32']
        precision[name]=dict(absolute_error_quantiles=np.quantile(abs(error),[.5,.9,.99,1]).tolist(),
                             spearman=float(spearmanr(v['normal_float16'],v['normal_float32']).statistic),
                             flag_changes_at_1pct=int(np.sum((v['normal_float16']>threshold[1])!=(v['normal_float32']>threshold[1]))))
        np.savez(out/(name+'.npz'),**v)
    for name,control in [('centered-hybrid','06-grid-hybrid'),('scaled-hybrid','06-grid-hybrid'),
                         ('centered-spatial','07-grid-spatial'),('scaled-spatial','07-grid-spatial')]:
        differences[name]={g:paired_bootstrap(parts[control],parts[name],g,a.bootstrap) for g in group if ':' in g}
    rows=[dict(probe=g,**{name:r['probes'][g] for name,r in results.items()}) for g in group if ':' in g]
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(out/'report.json',dict(kind='FROZEN_FRESH_NOISE_REUSED_NORMAL_AND_SYNTHETIC',
        public_test_read=False,model_fitting=False,normalization_recalculated=False,
        results=results,paired_differences=differences,storage_precision=precision,field_sources=sources,
        iaaft_final_amplitude_error_quantiles=np.quantile([r['final_relative_amplitude_error'] for r in manifest['iaaft_errors']],[0,.5,.9,1]).tolist(),
        limitations=['Clean audit maps and synthetic families were reused; these are new noise realizations, not new physical simulations.',
                     'Bootstrap conditions on fixed models and observed empirical distributions. No unknown-OOD guarantee.',
                     'IAAFT does not preserve masked Fourier amplitudes exactly.']))
    np.savez(out/'components.npz',nll=nll,field=field,**comp)
    print(json.dumps(dict(results=results,precision=precision),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','r4','r3','out','bundle','raw'):p.add_argument('--'+name,required=True)
    p.add_argument('--bootstrap',type=int,default=500);main(p.parse_args())
