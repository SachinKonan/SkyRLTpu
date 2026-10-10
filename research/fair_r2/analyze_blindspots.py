"""Evaluate additional families using the already frozen candidate-05 rule."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from common import sha,write_json,low_fpr_metric
from condition_field_scores import predict


def load(path):
    with np.load(path) as f:
        return {k:f[k] for k in f.files}


def main(a):
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    cache=root/'blindspots-cache';meta=json.loads((cache/'manifest.json').read_text())
    manifest=json.loads(Path(a.manifest).read_text());count=meta['shape'][0]
    assert meta['frozen_hybrid_manifest_sha256']==sha(a.manifest)
    field=[]
    for seed in (0,1):
        seen=np.zeros(count,int);values=np.empty(count)
        for shard in range(4):
            path=root/f'field-blind-s{seed}-{shard}'
            report=json.loads((path/'complete.json').read_text())
            assert report['scores_sha256']==sha(path/'scores.npz')
            assert report['cache_manifest_sha256']==sha(cache/'manifest.json')
            assert report['checkpoint_sha256']==manifest['field_models'][seed]['files']['best.msgpack']
            v=load(path/'scores.npz');seen[v['indices']]+=1;values[v['indices']]=v['nll']
        assert (seen==1).all();field.append(values)
    nll=np.mean(field,axis=0)
    assert sha(a.regression)==manifest['regression_sha256']
    f=load(a.regression)
    m={k.removeprefix('mean_'):v for k,v in f.items() if k.startswith('mean_')}
    vm={k.removeprefix('variance_'):v for k,v in f.items() if k.startswith('variance_')}
    m['kind']=str(m['kind']);vm['kind']=str(vm['kind'])
    features=np.load(cache/'features.npy')
    sd=np.maximum(np.exp(np.clip(predict(features,vm),-30,30)/2),manifest['conditional_std_floor'])
    residual=np.abs((nll-predict(features,m))/sd)
    summary=load(a.summary);hybrid_reference=load(a.hybrid_reference)
    sm,ss=manifest['summary_normal_center_scale'];fm,fs=manifest['field_normal_center_scale']
    rows=[];scores={}
    for name,g in meta['groups'].items():
        original=name.replace('reused/','r1_novel_now_reused/')
        s=summary[original];r=residual[g['offset']:g['offset']+g['count']]
        values=.5*((s-sm)/ss+(r-fm)/fs)
        rows.append(dict(family=name,count=len(s),summary=low_fpr_metric(summary['calibration'],s),
                         hybrid=low_fpr_metric(hybrid_reference['calibration'],values)))
        scores[name]=values
    np.savez(out/'hybrid_scores.npz',**scores)
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'report.json',dict(kind='ADDITIONAL_FAMILY_CHECK_AFTER_FREEZE_NOT_OFFICIAL_OOD',
               results=rows,frozen_manifest_sha256=sha(a.manifest),
               interpretation='These shifts were not part of field/hybrid score selection. Normal maps and earlier summary-model research are reused; this is not a new official OOD validation set. The candidate is unchanged.'))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','out','manifest','regression','summary','hybrid-reference'):
        p.add_argument('--'+name,required=True)
    main(p.parse_args())
