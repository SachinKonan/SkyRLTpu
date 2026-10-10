"""Freeze and export a pointwise 3-seed flow ensemble; never fit on test maps."""
import argparse,datetime,hashlib,json,sys,time,zipfile
from pathlib import Path
import numpy as np
from common import write_json,sha
from flow import predictor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
from features import make_predict

TEST_SHA='1b854da91b9414797109eb62ae08837af517a7832a721d3e7e48508592e6f6b7'


def main(a):
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    names=[f'flow-cosmology-asinh-s{seed}' for seed in range(3)]
    models=[dict(name=name,files={file:sha(root/name/file) for file in ('config.json','normalization.npz','best.msgpack')}) for name in names]
    manifest=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=models,weights=[1/3]*3,score='arithmetic mean of minimum_base_energy across the three seeds, with each seed minimized over the fixed 101 training cosmologies',selection='Selected on normal validation and engineered synthetic probes, including reused audit probes. No public test statistics, ranks, labels, or scores used.',runtime={str(p.name):sha(p) for p in (Path(__file__),Path(__file__).parent/'flow.py',Path(__file__).parent/'common.py',Path(__file__).resolve().parents[1]/'fair_ae/features.py')})
    write_json(out/'manifest.json',manifest)
    # The model choices, weights and runtime are fixed before test data is opened.
    test=Path(a.test);h=hashlib.sha256()
    with test.open('rb') as f:
        for block in iter(lambda:f.read(2**24),b''):h.update(block)
    if h.hexdigest()!=TEST_SHA:raise ValueError('Wrong public test version')
    maskpath=Path(a.raw)/'WIDE12H_bin2_2arcmin_mask.npy'
    if hashlib.md5(maskpath.read_bytes()).hexdigest()!='29b8bbb8a9ad41d3dacfaf61008a8268':raise ValueError('Wrong mask')
    mask=np.load(maskpath,allow_pickle=False);data=np.load(test,mmap_mode='r',allow_pickle=False)
    if data.shape!=(10000,int(mask.sum())):raise ValueError('Unexpected test shape')
    features=make_predict(mask,16);models=[predictor(root/name) for name in names]
    def predict(observed):
        maps=np.zeros((len(observed),*mask.shape),np.float32);maps[:,mask]=observed
        f=features(maps)
        return np.mean([fn(f)['minimum_base_energy'] for fn in models],axis=0)
    ix=np.asarray([0,1,17,113,509,1021,4321,9999]);x=data[ix];reference=predict(x)
    alone=predict(x[:1]);reversed_values=predict(x[::-1])[::-1];chunks=np.concatenate([predict(x[:3]),predict(x[3:])])
    errors=dict(singleton=float(np.max(np.abs(alone-reference[:1]))),reversal=float(np.max(np.abs(reversed_values-reference))),chunking=float(np.max(np.abs(chunks-reference))))
    if not all(np.allclose(v,w,rtol=1e-5,atol=1e-4) for v,w in [(alone,reference[:1]),(reversed_values,reference),(chunks,reference)]):raise ValueError('Pointwise independence check failed')
    start=time.monotonic();values=[]
    for i in range(0,len(data),16):
        values.append(predict(data[i:i+16]))
        if i%1024==0:print('inferred',min(i+16,len(data)),'seconds',time.monotonic()-start,flush=True)
    scores=np.concatenate(values)
    if scores.shape!=(10000,) or not np.isfinite(scores).all():raise ValueError('Invalid predictions')
    archive=out/'04-conditional-flow-ensemble.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('result.json',json.dumps({'ood_scores':scores.tolist()},allow_nan=False))
    receipt=dict(archive=archive.name,count=len(scores),sha256=sha(archive),manifest_sha256=sha(out/'manifest.json'),test_sha256=TEST_SHA,independence_verified=True,independence_max_abs_errors=errors,seconds=time.monotonic()-start,submitted=False)
    write_json(out/'04-conditional-flow-ensemble.receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--raw',required=True);p.add_argument('--test',required=True);p.add_argument('--out',required=True);main(p.parse_args())
