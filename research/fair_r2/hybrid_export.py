"""Freeze and export the fixed summary/field ensemble, pointwise inference only."""
import argparse,json,time,zipfile,hashlib,sys
from pathlib import Path
import numpy as np
from common import sha,write_json


def prepare(a):
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path(a.manifest).read_text())
    for m in manifest['field_models']:
        for name,digest in m['files'].items():
            if sha(root/m['name']/name)!=digest:raise ValueError('Changed frozen field model')
    if sha(root/'exports/04-conditional-flow-ensemble.zip')!=manifest['summary_archive_sha256']:raise ValueError('Changed summary component')
    # Validate the official v2 input, without deriving any cohort statistics.
    h=hashlib.sha256()
    with Path(a.test).open('rb') as f:
        for block in iter(lambda:f.read(2**24),b''):h.update(block)
    if h.hexdigest()!=manifest['test_sha256']:raise ValueError('Wrong test version')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
    from features import make_predict
    mask=np.load(Path(a.raw)/'WIDE12H_bin2_2arcmin_mask.npy');h,w=mask.shape
    count=mask.reshape(h//2,2,w//2,2).sum((1,3));support=count>=3;feature=make_predict(mask,16)
    test=np.load(a.test,mmap_mode='r');assert test.shape==(10000,int(mask.sum()))
    pooled=np.lib.format.open_memmap(out/'observed.npy',mode='w+',dtype=np.float32,shape=(10000,h//2,w//2))
    features=np.empty((10000,72),np.float32);start=time.monotonic()
    for i in range(0,len(test),16):
        data=test[i:i+16];x=np.zeros((len(data),h,w),np.float32);x[:,mask]=data
        features[i:i+len(x)]=feature(x)
        pooled[i:i+len(x)]=(x.reshape(len(x),h//2,2,w//2,2).sum((2,4))/np.maximum(count,1)*support).astype(np.float32)
        if i%2048==0:print('prepared',i+len(x),time.monotonic()-start,flush=True)
    pooled.flush();np.save(out/'features.npy',features)
    write_json(out/'manifest.json',dict(kind='FROZEN_PUBLIC_INFERENCE_INPUTS_NO_FITTING',shape=list(pooled.shape),frozen_model_manifest_sha256=sha(a.manifest),source_test_sha256=manifest['test_sha256'],per_map_preprocessing='Fixed survey mask and fixed 2x2 supported pooling; no test normalization or fitting',seconds=time.monotonic()-start))


def combine(a):
    root=Path(a.root);cache=Path(a.cache);manifest=json.loads(Path(a.manifest).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    if sha(root/'exports/04-conditional-flow-ensemble.zip')!=manifest['summary_archive_sha256']:raise ValueError('Changed summary scores')
    if sha(root/'hybrid-frozen/nll-normal-regressors.npz')!=manifest['regression_sha256']:raise ValueError('Changed normal regression')
    if json.loads((cache/'manifest.json').read_text())['frozen_model_manifest_sha256']!=sha(a.manifest):raise ValueError('Wrong inference input manifest')
    field=[];sources=[]
    for seed in (0,1):
        v=np.empty(10000);seen=np.zeros(10000,int)
        for shard in range(4):
            folder=root/f'field-test-s{seed}-{shard}';record=json.loads((folder/'complete.json').read_text());sources.append(record)
            assert record['checkpoint_sha256']==manifest['field_models'][seed]['files']['best.msgpack']
            assert record['cache_manifest_sha256']==sha(cache/'manifest.json') and record['scores_sha256']==sha(folder/'scores.npz')
            with np.load(folder/'scores.npz') as f:ids=f['indices'];seen[ids]+=1;v[ids]=f['nll']
        assert (seen==1).all();field.append(v)
    nll=np.mean(field,axis=0)
    with np.load(root/'hybrid-frozen/nll-normal-regressors.npz') as f:
        mean={k.removeprefix('mean_'):f[k] for k in f.files if k.startswith('mean_')}
        variance={k.removeprefix('variance_'):f[k] for k in f.files if k.startswith('variance_')}
    from condition_field_scores import predict
    mean['kind']=str(mean['kind']);variance['kind']=str(variance['kind'])
    x=np.load(cache/'features.npy');sd=np.maximum(np.exp(np.clip(predict(x,variance),-30,30)/2),manifest['conditional_std_floor'])
    field_score=np.abs((nll-predict(x,mean))/sd)
    with zipfile.ZipFile(root/'exports/04-conditional-flow-ensemble.zip') as z:summary=np.asarray(json.loads(z.read('result.json'))['ood_scores'])
    sm,ss=manifest['summary_normal_center_scale'];fm,fs=manifest['field_normal_center_scale']
    result=.5*((summary-sm)/ss+(field_score-fm)/fs)
    assert result.shape==(10000,) and np.isfinite(result).all()
    archive=out/'05-summary-field-ensemble.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('result.json',json.dumps({'ood_scores':result.tolist()},allow_nan=False))
    np.savez(out/'component_scores.npz',summary=summary,field_nll=nll,field_conditioned_score=field_score)
    receipt=dict(count=len(result),sha256=sha(archive),manifest_sha256=sha(a.manifest),rule='Fixed equal average of normal-scaled summary energy and absolute power-conditioned field likelihood residual',submitted=False,field_inference_sources=sources,independence='Fixed per-map functions; field/summary components numerically checked separately. Full hybrid API equivalence pending.')
    write_json(out/'05-summary-field-ensemble.receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='field_inference_sources'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','combine']);p.add_argument('--root',required=True);p.add_argument('--raw');p.add_argument('--test');p.add_argument('--cache');p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args();{'prepare':prepare,'combine':combine}[a.mode](a)
