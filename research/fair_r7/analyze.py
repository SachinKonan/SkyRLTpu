"""Joint normal-reference study over existing paired field-model statistics."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from joint_density import gaussian_fit,mahalanobis,kde_fit,kde_logpdf,choose_bandwidth
sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'fair_r3'))
from compare import read_npz,normalized,paired_bootstrap
from common import low_fpr_metric,write_json,sha,splits,load_labels
from condition_field_scores import fit_ridge,predict
from check_noise_hybrid import groups as parameter_groups


def original_models(path):
    a=read_npz(path)
    m={k.removeprefix('mean_'):v for k,v in a.items() if k.startswith('mean_')}
    v={k.removeprefix('variance_'):x for k,x in a.items() if k.startswith('variance_')}
    m['kind']=str(m['kind']);v['kind']=str(v['kind'])
    return m,v


def scale(x,model,floor):
    return np.maximum(np.exp(np.clip(predict(x,model),-30,30)/2),floor)


def crossfit(x,y,folds,kind,alpha):
    result=np.empty(len(y));details=[]
    for outer in range(5):
        train=folds!=outer;test=~train;mu=fit_ridge(x[train],y[train],kind,alpha)
        inner_prediction=np.empty(train.sum());xx=x[train];yy=y[train];ff=folds[train]
        for inner in np.unique(ff):
            use=ff!=inner
            inner_prediction[~use]=predict(xx[~use],fit_ridge(xx[use],yy[use],kind,alpha))
        floor=.05*yy.std()
        vm=fit_ridge(xx,np.log((yy-inner_prediction)**2+floor*floor),'both',.1)
        result[test]=(y[test]-predict(x[test],mu))/scale(x[test],vm,floor)
        details.append(dict(fold=outer,train=int(train.sum()),held_out=int(test.sum()),floor=float(floor)))
    return result,details


def main(a):
    r2,r4,out=map(Path,(a.r2,a.r4,a.out));out.mkdir(parents=True,exist_ok=False)
    snapshot=r2/'checkpoint-004';features={k:np.load(snapshot/'features216'/(k+'_latent.npy'))[:,:72]
                                          for k in ('calibration','audit')}
    for phase,path in [('development',snapshot/'probes72/development.npz'),
                       ('reused',snapshot/'probes72/r1_novel_now_reused.npz'),
                       ('morphology',snapshot/'morphology-probes/features72/morphology_r2.npz')]:
        values=read_npz(path);ids=values.pop('ids')
        if phase=='development':fit_ids=ids
        features.update({phase+'/'+k:v for k,v in values.items()})
    assert set(fit_ids).issubset(set(splits()['val'])) and len(fit_ids)==512
    order=np.random.default_rng(330).permutation(101);assignment=np.empty(101,int);assignment[order]=np.arange(101)%5
    folds=assignment[fit_ids//256]
    selected=json.loads((r2/'field-conditioned-001/report.json').read_text())['fits']
    raw={k:read_npz(r2/'field-comparison-001'/('field-ensemble__'+k+'.npz')) for k in ('nll','base_energy')}
    closure_root=r4/'closure-001';manifest=json.loads((closure_root/'closure-cache/manifest.json').read_text())
    cx=np.load(closure_root/'closure-cache/features165.npy')[:,:72].astype(np.float64)
    comp=read_npz(r4/'closure-comparison-001/components.npz');energies=[]
    for model in range(2):
        e=np.empty(len(cx));seen=np.zeros(len(cx),int)
        for shard in range(8):
            run=closure_root/f'field-s{model}-{shard}';record=json.loads((run/'complete.json').read_text())
            assert sha(run/'scores.npz')==record['scores_sha256']
            v=read_npz(run/'scores.npz');e[v['indices']]=v['base_energy'];seen[v['indices']]+=1
        assert np.all(seen==1);energies.append(e)
    cy={'nll':comp['nll'],'base_energy':np.mean(energies,axis=0)}
    original={};closure={};fitted={};outer=[];errors={};crossfit_details={}
    for key in ('nll','base_energy'):
        mu,vm=original_models(r2/'field-conditioned-001'/(key+'-normal-regressors.npz'))
        y=raw[key]['development/normal'];floor=.05*y.std()
        z={g:(v-predict(features[g],mu))/scale(features[g],vm,floor) for g,v in raw[key].items()}
        stored=read_npz(r2/'field-conditioned-001'/(key+'__hetero-upper.npz'))
        errors[key]={g:float(np.max(abs(v-stored[g]))) for g,v in z.items()}
        assert all(np.allclose(v,stored[g],rtol=1e-7,atol=1e-7) for g,v in z.items())
        original[key]=z;closure[key]=(cy[key]-predict(cx,mu))/scale(cx,vm,floor)
        choice=selected[key]['selected']
        cf,detail=crossfit(features['development/normal'],y,folds,choice['kind'],choice['alpha'])
        outer.append(cf);crossfit_details[key]=detail
        fitted[key]=dict(mean=mu,variance=vm,floor=floor)
    refs={'oof':np.column_stack(outer),'in-sample':np.column_stack([original[k]['development/normal'] for k in ('nll','base_energy')])}
    np.savez(out/'normal-references.npz',ids=fit_ids,folds=folds,**refs)
    values={g:np.column_stack([original[k][g] for k in ('nll','base_energy')]) for g in raw['nll']}
    pair={'calibration':values['calibration'],'audit':values['audit']}
    pair.update({'legacy/'+g:v for g,v in values.items() if '/' in g})
    cp=np.column_stack([closure[k] for k in ('nll','base_energy')])
    for g,d in manifest['groups'].items():
        name='closure/'+g if ':' in g else g
        pair[name]=cp[d['offset']:d['offset']+d['count']]
    np.savez(out/'paired-residuals.npz',**pair)
    fields={'abs_nll':{g:abs(v[:,0]) for g,v in pair.items()},'abs_energy':{g:abs(v[:,1]) for g,v in pair.items()}}
    model_reports={};pointwise={}
    for ref_name,reference in refs.items():
        model_reports[ref_name]=dict(mean=reference.mean(0).tolist(),cov=np.cov(reference,rowvar=False).tolist(),
                                     correlation=float(np.corrcoef(reference,rowvar=False)[0,1]),models={})
        for method in ('diagonal','gaussian_joint','kde_independent','kde_joint'):
            name=ref_name+'/'+method
            if method.startswith('kde_'):
                model,cv=choose_bandwidth(reference,folds,independent=method=='kde_independent')
                fn=lambda x,m=model:-kde_logpdf(x,m)
                info=dict(bandwidth_cv=cv,selected_factor=model['factor'],bandwidth=model['bandwidth'])
            else:
                model=gaussian_fit(reference,diagonal=method=='diagonal')
                fn=lambda x,m=model:mahalanobis(x,m)
                info=dict(covariance_condition=float(np.linalg.cond(model['cov'])))
            fields[name]={g:fn(v) for g,v in pair.items()}
            x=pair['calibration'][:31];base=fields[name]['calibration'][:31]
            checks=[fn(x[::-1])[::-1],np.r_[fn(x[:1]),fn(x[1:13]),fn(x[13:])]]
            pointwise[name]=[float(np.max(abs(v-base))) for v in checks]
            assert max(pointwise[name])<1e-8
            model_reports[ref_name]['models'][method]=info
            np.savez(out/(name.replace('/','__')+'-model.npz'),**model)
    old_summary=read_npz(r4/'score-calibration-001/scaled-summary.npz')
    old_08=read_npz(r4/'score-calibration-001/scaled-hybrid.npz');old_09=read_npz(r4/'score-calibration-001/scaled-spatial.npz')
    def translated(k):return k.replace('reused/','r1_novel_now_reused/').replace('morphology/','morphology_r2/')
    summary={g:old_summary[g] for g in ('calibration','audit')}
    scores={n:{g:v[g] for g in ('calibration','audit')} for n,v in [('08',old_08),('09',old_09)]}
    for g in values:
        if '/' in g:
            summary['legacy/'+g]=old_summary[translated(g)]
            scores['08']['legacy/'+g]=old_08[translated(g)];scores['09']['legacy/'+g]=old_09[translated(g)]
    closure_base={n:read_npz(r4/'closure-comparison-001'/p) for n,p in [('08','scaled-hybrid.npz'),('09','scaled-spatial.npz')]}
    for g,d in manifest['groups'].items():
        name='closure/'+g if ':' in g else g
        summary[name]=comp['scaled'][d['offset']:d['offset']+d['count']]
        for n in ('08','09'):scores[n][name]=closure_base[n][g]
    sn=normalized(summary);scores['summary_only']=summary
    for name,v in fields.items():
        scores['field/'+name]=v;vn=normalized(v)
        scores['hybrid/'+name]={g:.5*(sn[g]+value) for g,value in vn.items()}
    reproduction={g:float(np.max(abs(v-scores['hybrid/abs_nll'][g]))) for g,v in scores['08'].items()}
    assert all(np.allclose(v,scores['hybrid/abs_nll'][g],rtol=1e-5,atol=1e-5) for g,v in scores['08'].items()),reproduction
    results={};bins={};labels=load_labels(a.raw)
    for name,v in scores.items():
        thresholds=np.quantile(v['calibration'],[.95,.99,.999])
        results[name]=dict(normal={g:dict(count=len(v[g]),exceedances=[int((v[g]>t).sum()) for t in thresholds])
                                   for g in ('audit','normal_float32','normal_float16')},
                           probes={g:low_fpr_metric(v['calibration'],value) for g,value in v.items() if ':' in g})
        bins[name]=parameter_groups(dict(calibration=v['calibration'],audit=v['normal_float16']),labels)
        np.savez(out/(name.replace('/','__')+'.npz'),**v)
    rows=[dict(probe=g,**{name:r['probes'][g] for name,r in results.items()}) for g in results['08']['probes']]
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(stage='scores_written',normal_references=model_reports,results=results)),flush=True)
    differences={name:{g:paired_bootstrap(scores['08'],scores[name],g,a.bootstrap) for g in results['08']['probes'] if g.startswith('closure/')}
                 for name in ('hybrid/oof/diagonal','hybrid/oof/gaussian_joint','hybrid/oof/kde_independent','hybrid/oof/kde_joint')}
    matched={name:{g:paired_bootstrap(scores[control],scores[name],g,a.bootstrap) for g in results['08']['probes'] if g.startswith('closure/')}
             for name,control in [('hybrid/oof/gaussian_joint','hybrid/oof/diagonal'),('hybrid/oof/kde_joint','hybrid/oof/kde_independent')]}
    report=dict(kind='R7_REUSED_JOINT_STATISTICS_DIAGNOSTICS_NOT_OFFICIAL_OOD',public_test_read=False,neural_retraining=False,
                normal_reference_count=512,normal_references=model_reports,crossfit_details=crossfit_details,
                original_scalar_reproduction=errors,candidate08_reproduction=reproduction,pointwise_errors=pointwise,
                results=results,paired_differences_from08=differences,joint_minus_independence=matched,
                known_parameter_false_positives=bins,
                limitations=['All diagnostic families and clean maps are reused.',
                             'Outer/inner residual cross-fitting uses fixed scalar hyperparameters previously selected on this validation pool; it is not fully nested whole-pipeline validation.',
                             'KDE bandwidth CV operates on cross-fitted residuals whose prediction models share training observations; its log-density is an approximate selection diagnostic.',
                             'Paired IID bootstrap excludes repeated selection and simulator dependence.',
                             'This is a conditional-residual adaptation motivated by DoSE, not a reproduction of its models or performance.'])
    write_json(out/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('r2','r4','raw','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--bootstrap',type=int,default=500);main(p.parse_args())
