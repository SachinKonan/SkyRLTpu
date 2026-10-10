"""Sharded field likelihood evaluation on explicitly prepared map caches.

The prepare mode reads only official normal training maps, with noise paired
to the summary benchmarks. Score mode also supports frozen public-inference
caches produced by hybrid_export.py; it performs no fitting or calibration.
Cache manifests distinguish those inputs from training-derived probes.
"""
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from common import write_json,sha


def prepare(a):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
    from train import load_training,NOISE
    from diagnostics import shift
    from morphology_probes import iaaft
    maps,mask,labels,splits=load_training(a.raw);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    h,w=mask.shape;counts=mask.reshape(h//2,2,w//2,2).sum((1,3));support=counts>=3
    groups={};offset=0
    definition=[('calibration',3232),('audit',3232),('development/normal',512),('development/second',512),('development/spectral_low:0.2',512),('development/spectral_high:0.2',512),('development/quadratic_peak:0.2',512),('reused/normal',512),('reused/second',512),('reused/anisotropic:0.2',512),('morphology/normal',256),('morphology/phase_iaaft:1.0',256),('morphology/negative_peak:0.2',256)]
    for name,n in definition:groups[name]=dict(offset=offset,count=n);offset+=n
    target=np.lib.format.open_memmap(out/'observed.npy',mode='w+',dtype=np.float32,shape=(offset,h//2,w//2));source_ids=np.empty(offset,np.int32);start=time.monotonic()
    def save(name,i,x,ids):
        g=groups[name];lo=g['offset']+i;hi=lo+len(x)
        if hi>g['offset']+g['count']:raise ValueError('Group overflow')
        target[lo:hi]=(x.reshape(len(x),h//2,2,w//2,2).sum((2,4))/np.maximum(counts,1)*support).astype(np.float32);source_ids[lo:hi]=ids
    for name,noise_seed in [('calibration',302),('audit',303)]:
        ids=splits[name];rng=np.random.default_rng(noise_seed)
        for i in range(0,len(ids),16):
            ix=ids[i:i+16];x=np.zeros((len(ix),h,w),np.float32);x[:,mask]=maps[ix];x+=rng.normal(0,NOISE,x.shape).astype(np.float32)*mask;save(name,i,x,ix)
        print('prepared',name,time.monotonic()-start,flush=True)
    for phase,seed,source,families in [('development',474,'val',['spectral_low','spectral_high','quadratic_peak']),('reused',475,'audit',['anisotropic'])]:
        ids=np.random.default_rng(seed).permutation(splits[source])[:512]
        for i in range(0,len(ids),16):
            ix=ids[i:i+16];x=np.zeros((len(ix),h,w),np.float32);x[:,mask]=maps[ix]
            rng=np.random.default_rng(7300+i+int(source=='audit'));noise=rng.normal(0,NOISE,x.shape).astype(np.float32)*mask
            save(phase+'/normal',i,x+noise,ix);save(phase+'/second',i,x+rng.normal(0,NOISE,x.shape).astype(np.float32)*mask,ix)
            for family in families:save(phase+'/'+family+':0.2',i,shift(x,mask,family,.2)+noise,ix)
        print('prepared',phase,time.monotonic()-start,flush=True)
    ids=np.random.default_rng(20261009).permutation(splits['audit'])[:256]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0,len(ids),8):
            ix=ids[i:i+8];x=np.zeros((len(ix),h,w),np.float32);x[:,mask]=maps[ix]
            noise=np.random.default_rng(962000+i).normal(0,NOISE,x.shape).astype(np.float32)*mask
            y=np.stack([v[0] for v in pool.map(lambda pair:iaaft(pair[1],mask,90123+int(pair[0]),60),zip(ix,x))])
            save('morphology/normal',i,x+noise,ix);save('morphology/phase_iaaft:1.0',i,y+noise,ix);save('morphology/negative_peak:0.2',i,shift(x,mask,'quadratic_peak',-.2)+noise,ix)
    target.flush();np.save(out/'source_ids.npy',source_ids)
    write_json(out/'manifest.json',dict(kind='PAIRED_NORMAL_AND_SYNTHETIC_NOT_OFFICIAL_OOD',shape=list(target.shape),groups=groups,seconds=time.monotonic()-start,source_sha256=sha(__file__),paired_noise='Original batch-16 normal/probe draws; original batch-8 morphology draws; then fixed 2x2 pooling.',raw_test_read=False))
    print('COMPLETE',time.monotonic()-start,flush=True)


def score(a):
    from field_evaluate import predictor
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cache=Path(a.cache)
    manifest=json.loads((cache/'manifest.json').read_text());maps=np.load(cache/'observed.npy',mmap_mode='r');fn=predictor(a.run,a.ode_steps,batch=a.batch)
    record=dict(model=str(a.run),checkpoint_sha256=sha(Path(a.run)/'best.msgpack'),cache_manifest_sha256=sha(cache/'manifest.json'),shard=a.shard,shards=a.shards,batch=a.batch,ode_steps=a.ode_steps,trace_seed=451,trace_probes=1)
    write_json(out/'config.json',record);values={};indices=[];start=time.monotonic();batches=0
    for block,lo in enumerate(range(0,len(maps),a.batch)):
        if block%a.shards!=a.shard:continue
        ix=np.arange(lo,min(lo+a.batch,len(maps)));v=fn(np.asarray(maps[ix]))
        if not all(np.isfinite(x).all() for x in v.values()):raise ValueError('Nonfinite likelihood')
        indices.append(ix)
        for k,x in v.items():values.setdefault(k,[]).append(x)
        batches+=1
        if batches%4==0:
            progress=dict(count=sum(len(x) for x in indices),seconds=time.monotonic()-start)
            write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    np.savez(out/'scores.npz',indices=np.concatenate(indices),**{k:np.concatenate(v) for k,v in values.items()})
    write_json(out/'complete.json',dict(**record,count=sum(len(x) for x in indices),seconds=time.monotonic()-start,scores_sha256=sha(out/'scores.npz')));print('COMPLETE',time.monotonic()-start,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','score']);p.add_argument('--raw');p.add_argument('--cache');p.add_argument('--run');p.add_argument('--out',required=True);p.add_argument('--batch',type=int,default=32);p.add_argument('--ode-steps',type=int,default=32);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4);a=p.parse_args();{'prepare':prepare,'score':score}[a.mode](a)
