"""Merge exact field shards and compare fixed scores on paired study inputs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from common import low_fpr_metric,score_report,write_json,sha


def main(a):
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    cache=Path(a.cache);meta=json.loads((cache/'manifest.json').read_text());total=meta['shape'][0]
    scores={};provenance={}
    for seed in (0,1):
        values={};seen=np.zeros(total,int);configs=[]
        for shard in range(4):
            path=root/f'field-score-s{seed}-{shard}';cfg=json.loads((path/'complete.json').read_text())
            assert cfg['shard']==shard and cfg['shards']==4
            assert cfg['cache_manifest_sha256']==sha(cache/'manifest.json') and cfg['scores_sha256']==sha(path/'scores.npz')
            with np.load(path/'scores.npz') as f:
                ids=f['indices'];seen[ids]+=1
                for k in f.files:
                    if k!='indices':values.setdefault(k,np.empty(total))[ids]=f[k]
            configs.append(cfg)
        assert (seen==1).all()
        assert len({c['checkpoint_sha256'] for c in configs})==1
        provenance[f'seed{seed}']=configs
        for k,v in values.items():
            assert np.isfinite(v).all()
            scores[f'field-s{seed}/{k}']={name:v[group['offset']:group['offset']+group['count']] for name,group in meta['groups'].items()}
    for key in ('nll','base_energy','integrated_divergence'):
        a0=scores['field-s0/'+key];a1=scores['field-s1/'+key]
        scores['field-ensemble/'+key]={g:(a0[g]+a1[g])/2 for g in a0}
        if key=='nll':scores['field-ensemble/waic']={g:(a0[g]+a1[g])/2+(a0[g]-a1[g])**2/4 for g in a0}
    for name,v in list(scores.items()):
        if name.endswith('integrated_divergence'):continue
        q10,mid,q90=np.quantile(v['calibration'],[.1,.5,.9]);up=max(q90-mid,1e-8);down=max(mid-q10,1e-8)
        scores[name+'-two-sided']={g:np.where(s>=mid,(s-mid)/up,(mid-s)/down) for g,s in v.items()}
    reports={};rows=[]
    for name,v in scores.items():
        normal=score_report({'score':v['calibration']},{'score':v['audit']})['score']
        probes={g:low_fpr_metric(v['calibration'],s) for g,s in v.items() if '/' in g}
        repeat={phase:float(spearmanr(v[phase+'/normal'],v[phase+'/second']).statistic) for phase in ('development','reused')}
        reports[name]=dict(normal=normal,probes=probes,noise_repeatability=repeat)
        rows.append(dict(name=name,audit_1pct_count=normal['exceedances'][1],worst_cosmology_1pct=normal['max_cosmology_count_at_1pct'],**probes))
        np.savez(out/(name.replace('/','__')+'.npz'),**v)
        print(json.dumps(rows[-1]),flush=True)
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'report.json',dict(kind='FIELD_DENSITY_ON_REUSED_NORMAL_AND_SYNTHETIC_PROBES_NOT_OFFICIAL_OOD',provenance=provenance,results=reports,limitations=['32-step Heun approximation with one fixed Rademacher trace vector.', 'Two-sided score scales use only normal calibration maps.', 'Normal/synthetic study inputs were reused, and OOD shifts are engineered.', 'Field inputs have fixed 2x2 pooling; the R1 physical controls also retain full-resolution information.']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);main(p.parse_args())
