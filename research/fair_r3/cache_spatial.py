"""Cache paired normal/synthetic spatial descriptors; no public inputs."""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_r2'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
from train import load_training,NOISE
from diagnostics import shift,DEVELOPMENT,NOVEL
from morphology_probes import iaaft
from common import write_json,sha
from spatial_features import SpatialSummary


def main(a):
    root=Path(a.baseline_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    maps,mask,labels,splits=load_training(a.raw);fn=SpatialSummary(mask,16)
    start=time.monotonic();ids=splits['val'][:32]
    x=np.zeros((len(ids),*mask.shape),np.float32);x[:,mask]=maps[ids]
    x+=np.random.default_rng(44).normal(0,NOISE,x.shape).astype(np.float32)*mask
    v=fn(x);first=time.monotonic()-start
    tick=time.monotonic();repeat=fn(x);warm=time.monotonic()-tick
    assert np.isfinite(v).all() and np.array_equal(v,repeat)
    assert np.allclose(v[:1],fn(x[:1]),rtol=1e-6,atol=1e-6)
    assert np.allclose(v,fn(x[::-1])[::-1],rtol=1e-6,atol=1e-6)
    write_json(out/'smoke.json',dict(**fn.description,finite=True,independence_verified=True,
                                    first_32_seconds=first,warm_32_seconds=warm,
                                    warm_maps_per_second=len(x)/warm))
    print((out/'smoke.json').read_text(),flush=True)
    if a.smoke:return
    for name,ids in splits.items():
        rng=np.random.default_rng(300+list(splits).index(name));values=[]
        for i in range(0,len(ids),16):
            ix=ids[i:i+16];x=np.zeros((len(ix),*mask.shape),np.float32);x[:,mask]=maps[ix]
            x+=rng.normal(0,NOISE,x.shape).astype(np.float32)*mask;values.append(fn(x))
            if i%4096==0:print('normal',name,i+len(ix),time.monotonic()-start,flush=True)
        base=np.load(root/'features72'/f'{name}_latent.npy')
        np.save(out/f'{name}_latent.npy',np.column_stack([base,np.concatenate(values)]))
        print('COMPLETE normal',name,len(ids),time.monotonic()-start,flush=True)
    probes=out/'probes';probes.mkdir()
    for phase,seed,source,families in [('development',474,'val',DEVELOPMENT),
                                     ('r1_novel_now_reused',475,'audit',NOVEL)]:
        with np.load(root/'probes72'/f'{phase}.npz') as f:previous={k:f[k] for k in f.files}
        ids=previous['ids'];values={}
        assert np.array_equal(ids,np.random.default_rng(seed).permutation(splits[source])[:512])
        for i in range(0,len(ids),16):
            ix=ids[i:i+16];x=np.zeros((len(ix),*mask.shape),np.float32);x[:,mask]=maps[ix]
            ri=np.array([splits['train'][160*(n//256)+(n%160)] for n in ix])
            ref=np.zeros_like(x);ref[:,mask]=maps[ri]
            rng=np.random.default_rng(7300+i+int(source=='audit'))
            noise=rng.normal(0,NOISE,x.shape).astype(np.float32)*mask
            values.setdefault('normal',[]).append(fn(x+noise))
            values.setdefault('second',[]).append(fn(x+rng.normal(0,NOISE,x.shape).astype(np.float32)*mask))
            for family in families:
                for strength in (.2,.5):
                    values.setdefault(f'{family}:{strength}',[]).append(fn(shift(x,mask,family,strength,ref)+noise))
        result={k:np.column_stack([previous[k],np.concatenate(v)]) for k,v in values.items()}
        np.savez(probes/f'{phase}.npz',ids=ids,**result)
        print('COMPLETE probes',phase,time.monotonic()-start,flush=True)
    with np.load(root/'morphology-probes/features72/morphology_r2.npz') as f:previous={k:f[k] for k in f.files}
    ids=previous['ids'];values={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0,len(ids),8):
            ix=ids[i:i+8];x=np.zeros((len(ix),*mask.shape),np.float32);x[:,mask]=maps[ix]
            rng=np.random.default_rng(962000+i);noise=rng.normal(0,NOISE,x.shape).astype(np.float32)*mask
            values.setdefault('normal',[]).append(fn(x+noise))
            values.setdefault('second',[]).append(fn(x+rng.normal(0,NOISE,x.shape).astype(np.float32)*mask))
            y=np.stack([v[0] for v in pool.map(lambda pair:iaaft(pair[1],mask,90123+int(pair[0]),60),zip(ix,x))])
            values.setdefault('phase_iaaft:1.0',[]).append(fn(y+noise))
            for strength in (-.2,-.5):
                values.setdefault(f'negative_peak:{abs(strength)}',[]).append(fn(shift(x,mask,'quadratic_peak',strength)+noise))
            mu=x[:,mask].mean(1);y=(2*mu[:,None,None]-x)*mask
            values.setdefault('sign_reflection:1.0',[]).append(fn(y+noise))
            print('morphology',i+len(ix),time.monotonic()-start,flush=True)
    result={k:np.column_stack([previous[k],np.concatenate(v)]) for k,v in values.items()}
    np.savez(probes/'morphology_r2.npz',ids=ids,**result)
    write_json(out/'manifest.json',dict(kind='NORMAL_AND_REUSED_SYNTHETIC_NO_PUBLIC_TEST',
        definition=fn.description,seconds=time.monotonic()-start,
        source_sha256=sha(Path(__file__).parent/'spatial_features.py'),
        source_baseline=str(root),feature_hashes={p.name:sha(p) for p in out.glob('*.npy')},
        probe_hashes={p.name:sha(p) for p in probes.glob('*.npz')},
        grouping='First72 original features, next45 digital morphology, last48 regional statistics.'))
    print('COMPLETE ALL',time.monotonic()-start,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('raw','baseline-root','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--smoke',action='store_true')
    main(p.parse_args())
