"""Compare trained field models with matched original models; no test reads."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'fair_r3'))
from compare import read_npz,normalized,paired_bootstrap
from common import low_fpr_metric,sha,splits,load_labels,write_json
from condition_field_scores import fit_ridge,predict
from check_noise_hybrid import groups as parameter_groups


def fit_background(x,y,ids):
    order=np.random.default_rng(330).permutation(101);assign=np.empty(101,int);assign[order]=np.arange(101)%5
    folds=assign[ids//256];trials=[];best=np.inf
    for kind in ('power','log','both'):
        for alpha in (.0001,.001,.01,.1,1.):
            oof=np.empty_like(y)
            for fold in range(5):
                train=folds!=fold;oof[~train]=predict(x[~train],fit_ridge(x[train],y[train],kind,alpha))
            error=float(np.sqrt(np.mean((y-oof)**2))/y.std())
            trials.append(dict(kind=kind,alpha=alpha,relative_oof_rmse=error))
            if error<best:best=error;chosen=(kind,alpha);best_oof=oof.copy()
    mean=fit_ridge(x,y,*chosen);floor=.05*y.std()
    variance=fit_ridge(x,np.log((y-best_oof)**2+floor**2),'both',.1)
    return mean,variance,floor,dict(selected=dict(kind=chosen[0],alpha=chosen[1],relative_oof_rmse=best),trials=trials)


def main(a):
    root,r2,r3,r4,out=map(Path,(a.root,a.r2,a.r3,a.r4,a.out));out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'evaluation-cache-manifest.json').read_text());groups=manifest['groups'];n=manifest['shape'][0]
    launch=json.loads((root/'evaluation-launch-001.json').read_text());ids=np.load(root/'evaluation-source_ids.npy')
    def split(v):return {g:v[d['offset']:d['offset']+d['count']] for g,d in groups.items()}
    source_ids=split(ids);assert np.array_equal(source_ids['calibration'],splits()['calibration'])
    for g in ('normal_float16','normal_float32'):assert np.array_equal(source_ids[g],splits()['audit'])
    fit_features=read_npz(r2/'checkpoint-004/probes72/development.npz')
    assert np.array_equal(source_ids['fit_normal'],fit_features['ids'])
    features={'fit_normal':fit_features['normal'],'calibration':np.load(r2/'checkpoint-004/features216/calibration_latent.npy')[:,:72]}
    closure_manifest=json.loads((r4/'closure-001/closure-cache/manifest.json').read_text())
    cx=np.load(r4/'closure-001/closure-cache/features165.npy')[:,:72].astype(np.float64)
    for g,d in closure_manifest['groups'].items():features[g]=cx[d['offset']:d['offset']+d['count']]
    per_model={};sources={}
    for name in ('old-s0','old-s1','new-s0','new-s1'):
        y=np.empty(n);seen=np.zeros(n,int)
        for shard in range(4):
            path=root/f'eval-{name}-{shard}';record=json.loads((path/'complete.json').read_text());v=read_npz(path/'scores.npz')
            assert record['checkpoint_sha256']==launch['models'][name]['sha256'] and record['ode_steps']==64
            assert record['scores_sha256']==sha(path/'scores.npz') and record['count']==len(v['indices'])
            y[v['indices']]=v['nll'];seen[v['indices']]+=1;sources[path.name]=record
        assert np.all(seen==1) and np.isfinite(y).all();per_model[name]=split(y)
    comp=read_npz(r4/'closure-comparison-001/components.npz')
    summary={'calibration':read_npz(r4/'score-calibration-001/scaled-summary.npz')['calibration']}
    region={'calibration':read_npz(r3/'regional-calibration-001/regional-centered.npz')['calibration']}
    for g,d in closure_manifest['groups'].items():
        sl=slice(d['offset'],d['offset']+d['count']);summary[g]=comp['scaled'][sl];region[g]=comp['regional'][sl]
    sn,rn=normalized(summary),normalized(region)
    scores={'08':read_npz(r4/'closure-comparison-001/scaled-hybrid.npz'),
            '09':read_npz(r4/'closure-comparison-001/scaled-spatial.npz')}
    old_norm=normalized(scores['08'])
    old09_error={g:float(np.max(abs(scores['09'][g]-np.maximum(old_norm[g],rn[g])))) for g in scores['09']}
    assert max(old09_error.values())<1e-7
    fitted={};pointwise={}
    for tag in ('old','new'):
        raw={g:np.mean([per_model[f'{tag}-s{s}'][g] for s in range(2)],axis=0) for g in groups}
        mean,variance,floor,fit=fit_background(features['fit_normal'],raw['fit_normal'],source_ids['fit_normal'])
        def residual(x,y):
            sd=np.maximum(np.exp(np.clip(predict(x,variance),-30,30)/2),floor)
            return abs((y-predict(x,mean))/sd)
        field={g:residual(features[g],v) for g,v in raw.items()}
        x=features['calibration'][:27];y=raw['calibration'][:27];base=field['calibration'][:27]
        checks=[residual(x[::-1],y[::-1])[::-1],np.r_[residual(x[:1],y[:1]),residual(x[1:],y[1:])]]
        pointwise[tag]=[float(np.max(abs(v-base))) for v in checks];assert max(pointwise[tag])<1e-7
        fitted[tag]=dict(**fit,conditional_std_floor=float(floor),fit_count=len(raw['fit_normal']))
        np.savez(out/(tag+'64-regressors.npz'),**{'mean_'+k:v for k,v in mean.items()},
                 **{'variance_'+k:v for k,v in variance.items()},conditional_std_floor=floor)
        fn=normalized(field)
        scores[tag+'64-nll']=raw
        scores[tag+'64-centered-nll']={g:abs(v-raw['fit_normal'].mean()) for g,v in raw.items()}
        scores[tag+'64-field']=field
        hybrid={g:.5*(sn[g]+fn[g]) for g in sn};scores[tag+'64-hybrid']=hybrid
        hn=normalized(hybrid);scores[tag+'64-spatial']={g:np.maximum(hn[g],rn[g]) for g in hn}
    reports={};bins={};labels=load_labels(a.raw)
    for name,v in scores.items():
        threshold=np.quantile(v['calibration'],[.95,.99,.999])
        reports[name]=dict(normal={g:dict(count=len(v[g]),exceedances=[int((v[g]>t).sum()) for t in threshold])
                                  for g in ('normal_float32','normal_float16')},
                           probes={g:low_fpr_metric(v['calibration'],v[g]) for g in groups if ':' in g})
        bins[name]=parameter_groups(dict(calibration=v['calibration'],audit=v['normal_float16']),labels)
        np.savez(out/(name+'.npz'),**v)
    rows=[dict(probe=g,**{k:v['probes'][g] for k,v in reports.items()}) for g in reports['08']['probes']]
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(stage='scores_written',fits=fitted,results=reports)),flush=True)
    differences={}
    for alt,control in [('new64-hybrid','old64-hybrid'),('new64-spatial','old64-spatial'),
                        ('new64-hybrid','08'),('new64-spatial','09'),('old64-hybrid','08'),('old64-spatial','09')]:
        differences[alt+'-minus-'+control]={g:paired_bootstrap(scores[control],scores[alt],g,a.bootstrap) for g in groups if ':' in g}
    write_json(out/'report.json',dict(kind='R8_MATCHED_FIELD_TRAINING_REUSED_SYNTHETIC_DIAGNOSTICS',
        public_test_read=False,selection=launch['selection'],model_sources=launch['models'],field_sources=sources,
        conditioning=fitted,pointwise_errors=pointwise,candidate09_reproduction=old09_error,
        results=reports,paired_differences=differences,known_parameter_false_positives=bins,
        limitations=['All clean maps and anomaly families are reused; no official OOD score is measured.',
                     'Normal validation losses select architecture/checkpoints; this comparison does not cover every training architecture on OOD.',
                     'The 512 normal conditioning examples also participated in earlier normal-data research.',
                     'Power conditioning can remove genuine anomaly signals; these residuals are not exact likelihood ratios.',
                     'IID paired bootstrap excludes model selection and simulator dependence.']))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','r2','r3','r4','raw','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--bootstrap',type=int,default=500);main(p.parse_args())
