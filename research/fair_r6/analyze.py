"""Compare frozen field-noise statistics with existing paired R4 scores."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'fair_r3'))
from compare import read_npz,normalized,paired_bootstrap
from common import low_fpr_metric,write_json,sha,load_labels
from check_noise_hybrid import groups as parameter_groups


def quantile_transform(x,reference):
    reference=np.sort(np.asarray(reference,np.float64))
    return (np.searchsorted(reference,x,side='right')+.5)/(len(reference)+1)


def main(a):
    root,r4,out=map(Path,(a.root,a.r4,a.out));out.mkdir(parents=True,exist_ok=False)
    launch=json.loads((root/'launch-001.json').read_text())
    normal=json.loads((root/'normal-manifest.json').read_text())
    closure=json.loads((root/'closure-manifest.json').read_text())
    group=dict(normal['groups'])
    group.update({k:dict(offset=v['offset']+6464,count=v['count']) for k,v in closure['groups'].items()})
    def grouped(v):return {k:v[d['offset']:d['offset']+d['count']] for k,d in group.items()}
    sources={};all_models=[]
    for model in range(2):
        full={};seen=np.zeros(16160,np.int32)
        for shard in range(8):
            run=root/f'field-s{model}-{shard}';report=json.loads((run/'report.json').read_text())
            part=read_npz(run/'statistics.npz');ids=part.pop('ids')
            assert report['output_sha256']==sha(run/'statistics.npz')
            assert report['steps']==64 and report['count']==len(ids)
            assert report['normal_cache_sha256']==normal['observations_sha256']
            assert report['closure_cache_sha256']==closure['observations_sha256']
            assert report['checkpoint_sha256']==launch['pilots'][str(model)]['pilot']['checkpoint_sha256']
            assert np.array_equal(ids,np.arange(shard,16160,8))
            for k,v in part.items():
                assert v.shape==ids.shape and np.isfinite(v).all()
                full.setdefault(k,np.empty(16160))[ids]=v
            seen[ids]+=1;sources[run.name]=report
        assert np.all(seen==1)
        all_models.append(full);np.savez(out/f'model-{model}-statistics.npz',**full)
    component={k:.5*(all_models[0][k]+all_models[1][k]) for k in all_models[0]}
    np.savez(out/'mean-statistics.npz',**component)
    selected=('energy','ad','ad_shape','psd_cv','correlation_energy')
    parts={k:grouped(component[k]) for k in selected}
    references={k:v['val'] for k,v in parts.items()}
    np.savez(out/'normal-reference-statistics.npz',**references)
    quantiles={k:{g:quantile_transform(v,references[k]) for g,v in part.items()} for k,part in parts.items()}
    for name,statistics in [('sitn-inspired',('ad','psd_cv')),
                            ('shape-spatial',('ad_shape','psd_cv','correlation_energy'))]:
        parts[name]={g:-np.log1p(-np.max(np.stack([quantiles[k][g] for k in statistics]),axis=0)) for g in group}
    for number,path in [('08','scaled-hybrid.npz'),('09','scaled-spatial.npz')]:
        parts[number]=read_npz(r4/'closure-comparison-001'/path)
    bn=normalized(parts['08'])
    for name in ('sitn-inspired','shape-spatial'):
        other=normalized(parts[name])
        parts['08+'+name+'-mean']={k:.5*(v+other[k]) for k,v in bn.items()}
        parts['08+'+name+'-max']={k:np.maximum(v,other[k]) for k,v in bn.items()}
    results={};precision={};bins={};labels=load_labels(a.raw)
    for name,v in parts.items():
        thresholds=np.quantile(v['calibration'],[.95,.99,.999])
        results[name]=dict(normal={g:dict(count=len(v[g]),exceedances=[int((v[g]>t).sum()) for t in thresholds])
                                   for g in ('normal_float32','normal_float16')},
                           probes={g:low_fpr_metric(v['calibration'],v[g]) for g in group if ':' in g})
        error=abs(v['normal_float16']-v['normal_float32'])
        precision[name]=dict(absolute_error_quantiles=np.quantile(error,[.5,.9,.99,1]).tolist(),
                             spearman=float(spearmanr(v['normal_float16'],v['normal_float32']).statistic),
                             flag_changes_at_1pct=int(np.sum((v['normal_float16']>thresholds[1])!=(v['normal_float32']>thresholds[1]))))
        bins[name]=parameter_groups(dict(calibration=v['calibration'],audit=v['normal_float16']),labels)
        np.savez(out/(name+'.npz'),**v)
    rows=[dict(probe=g,**{name:r['probes'][g] for name,r in results.items()}) for g in group if ':' in g]
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(stage='scores_written',results=results)),flush=True)
    differences={name:{g:paired_bootstrap(parts['08'],v,g,a.bootstrap) for g in group if ':' in g}
                 for name,v in parts.items() if name.startswith('08+')}
    saturated={name:{g:int(np.sum(q>len(references[name])/(len(references[name])+1))) for g,q in values.items()}
               for name,values in quantiles.items()}
    report=dict(kind='R6_REUSED_DIAGNOSTICS_NOT_OFFICIAL_OOD',public_test_read=False,neural_retraining=False,
                results=results,paired_differences_from08=differences,storage_precision=precision,
                known_parameter_false_positives=bins,reference_upper_saturation_counts=saturated,
                sources=sources,
                normal_mean_statistic_quantiles={k:{g:np.quantile(v[g],[.01,.5,.99]).tolist() for g in ('val','calibration','normal_float16')}
                                                for k,v in parts.items() if k in selected},
                limitations=['The R4 observations/families are reused for exploration; these are not a blind test.',
                             'Validation realizations previously selected neural checkpoints; statistic CDFs use only these normal maps.',
                             'Finite normal-reference CDFs saturate and create score ties. No public ranks or public adaptation are used.',
                             'Paired IID bootstrap excludes repeated selection, simulation dependence and unknown physical alternatives.',
                             'SITN-inspired statistics on masked pooled cosmological fields are an adaptation, not a reproduction of the cited experiments or theoretical guarantees.'])
    write_json(out/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','r4','raw','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--bootstrap',type=int,default=500);main(p.parse_args())
