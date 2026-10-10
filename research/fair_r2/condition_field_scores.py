"""Normal-only background control for field scores, using 512 validation maps.

Regress score expectations on observed power summaries. Fit and choose ridge
strengths by held-cosmology CV on NORMAL validation maps only; the 3232-map
calibration and audit sets do not fit this regression. This is a residual score,
not an exact likelihood ratio or conditional probability density.
"""
import argparse,json
from pathlib import Path
import numpy as np
from common import low_fpr_metric,score_report,write_json


def basis(x,kind):
    log=x[:,:20];power=np.exp(np.clip(log,-30,30))
    return {'log':log,'power':power,'both':np.column_stack([log,power])}[kind]


def fit_ridge(x,y,kind,alpha):
    b=basis(x,kind);mean=b.mean(0);std=np.maximum(b.std(0),1e-9);z=(b-mean)/std;ym=y.mean(0)
    w=np.linalg.solve(z.T@z+alpha*len(z)*np.eye(z.shape[1]),z.T@(y-ym))
    return dict(mean=mean,std=std,w=w,ym=ym,kind=kind,alpha=alpha)


def predict(x,m):return (basis(x,m['kind'])-m['mean'])/m['std']@m['w']+m['ym']


def main(a):
    root=Path(a.root);field=Path(a.field);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    features={k:np.load(root/'features216'/(k+'_latent.npy'))[:,:72] for k in ('calibration','audit')}
    ids=None
    for prefix,path in [('development',root/'probes72/development.npz'),('reused',root/'probes72/r1_novel_now_reused.npz'),('morphology',root/'morphology-probes/features72/morphology_r2.npz')]:
        with np.load(path) as f:
            features.update({prefix+'/'+k:f[k] for k in f.files if k!='ids'})
            if prefix=='development':ids=f['ids']
    rng=np.random.default_rng(330);order=rng.permutation(101);fold_for_cosmo=np.empty(101,int);fold_for_cosmo[order]=np.arange(101)%5;fold=fold_for_cosmo[ids//256]
    reports={};scores={};fits={}
    for key in ('nll','base_energy'):
        with np.load(field/('field-ensemble__'+key+'.npz')) as f:values={k:f[k] for k in f.files}
        x=features['development/normal'];y=values['development/normal'];candidates=[];best=np.inf
        for kind in ('power','log','both'):
            for alpha in (.0001,.001,.01,.1,1.):
                oof=np.empty_like(y)
                for f in range(5):
                    tr=fold!=f;te=~tr;m=fit_ridge(x[tr],y[tr],kind,alpha);oof[te]=predict(x[te],m)
                error=float(np.sqrt(np.mean((y-oof)**2))/y.std());candidates.append(dict(kind=kind,alpha=alpha,relative_oof_rmse=error))
                if error<best:best=error;chosen=(kind,alpha,oof.copy())
        kind,alpha,oof=chosen;m=fit_ridge(x,y,kind,alpha)
        # Heteroskedastic scaling is a fixed normal-only control, not tuned on probes.
        residual=y-oof;variance_floor=(.05*y.std())**2
        vm=fit_ridge(x,np.log(residual**2+variance_floor),'both',.1)
        def sigma(x):return np.maximum(np.exp(np.clip(predict(x,vm),-30,30)/2),np.sqrt(variance_floor))
        for hetero in (False,True):
            result={g:(v-predict(features[g],m))/(sigma(features[g]) if hetero else y.std()) for g,v in values.items()}
            for tail in ('upper','absolute'):
                name=key+('/hetero-' if hetero else '/residual-')+tail
                scores[name]={g:np.abs(v) if tail=='absolute' else v for g,v in result.items()}
        fits[key]=dict(selected=dict(kind=kind,alpha=alpha,relative_oof_rmse=best),cv=candidates,fit_count=len(y),fit_source='NORMAL development/validation maps',split='5 folds grouped by cosmology')
        np.savez(out/(key+'-normal-regressors.npz'),**{'mean_'+k:v for k,v in m.items()},**{'variance_'+k:v for k,v in vm.items()})
    for name,v in scores.items():
        norm=score_report({'score':v['calibration']},{'score':v['audit']})['score'];probes={g:low_fpr_metric(v['calibration'],s) for g,s in v.items() if '/' in g}
        reports[name]=dict(normal=norm,probes=probes);np.savez(out/(name.replace('/','__')+'.npz'),**v)
        print(name,json.dumps(dict(worst_cosmology_1pct=norm['max_cosmology_count_at_1pct'],**probes)))
    write_json(out/'report.json',dict(kind='EXPLORATORY_NORMAL_CONDITIONED_FIELD_SCORE_NOT_OFFICIAL_OOD',fits=fits,results=reports,limitations=['Residualization is not an exact likelihood ratio.', 'The 512 normal fitting maps also underlie the reused development probes; the audit and morphology source maps do not fit these regressors.', 'Power-conditioning can remove sensitivity to genuine spectral anomalies.', 'Model choices use normal-only regression CV; score diagnostics reuse engineered probes.']))
    print('CV',json.dumps(fits))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--field',required=True);p.add_argument('--out',required=True);main(p.parse_args())
