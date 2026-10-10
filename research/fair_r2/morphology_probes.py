"""New engineered morphology probes on reused normal audit maps.

IAAFT keeps the valid-pixel histogram exact and the masked Fourier amplitudes
approximately fixed. Its spectral error is measured, not assumed zero. These
are NOT organizer OOD examples and are never generated from public test maps.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys,time
import numpy as np
from scipy.fft import rfft2,irfft2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
from train import load_training,NOISE
from features import make_predict
from diagnostics import shift
from wavelets import WaveletSummary
from common import write_json,sha


def iaaft(x,mask,seed,iterations):
    rng=np.random.default_rng(seed);hist=np.sort(x[mask]);amp=np.abs(rfft2(x))
    y=np.zeros_like(x);y[mask]=rng.permutation(x[mask])
    initial=float(np.sqrt(np.sum((np.abs(rfft2(y))-amp)**2)/np.sum(amp**2)))
    for _ in range(iterations):
        f=rfft2(y);v=irfft2(amp*f/np.maximum(np.abs(f),1e-15),s=x.shape)
        order=np.argsort(v[mask]);values=np.empty_like(hist);values[order]=hist;y[mask]=values
    error=float(np.sqrt(np.sum((np.abs(rfft2(y))-amp)**2)/np.sum(amp**2)))
    assert np.array_equal(np.sort(y[mask]),hist)
    assert np.all(y[~mask]==0) and np.isfinite(y).all()
    return y,dict(initial_relative_amplitude_error=initial,final_relative_amplitude_error=error,histogram_exact=True)


def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    maps,mask,labels,splits=load_training(a.raw);base=make_predict(mask,8);wave=WaveletSummary(mask,8)
    ids=np.random.default_rng(20261009).permutation(splits['audit'])[:a.count]
    outputs={};quality=[]
    def features(x):return np.column_stack([base(x),wave(x)])
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0,len(ids),8):
            ix=ids[i:i+8];x=np.zeros((len(ix),*mask.shape),np.float32);x[:,mask]=maps[ix]
            rng=np.random.default_rng(962000+i);noise=rng.normal(0,NOISE,x.shape).astype(np.float32)*mask
            outputs.setdefault('normal',[]).append(features(x+noise))
            outputs.setdefault('second',[]).append(features(x+rng.normal(0,NOISE,x.shape).astype(np.float32)*mask))
            result=list(pool.map(lambda pair:iaaft(pair[1],mask,90123+int(pair[0]),a.iterations),zip(ix,x)))
            y=np.stack([v[0] for v in result]);quality.extend([dict(id=int(idx),**v[1]) for idx,v in zip(ix,result)])
            outputs.setdefault('phase_iaaft:1.0',[]).append(features(y+noise))
            for strength in (-.2,-.5):
                y=shift(x,mask,'quadratic_peak',strength)
                outputs.setdefault(f'negative_peak:{abs(strength)}',[]).append(features(y+noise))
            mean=x[:,mask].mean(1);y=(2*mean[:,None,None]-x)*mask
            outputs.setdefault('sign_reflection:1.0',[]).append(features(y+noise))
            print('maps',i+len(ix),'seconds',time.monotonic()-start,flush=True)
    result={k:np.concatenate(v) for k,v in outputs.items()}
    for name,d in [('features72',72),('features216',216)]:
        (out/name).mkdir();np.savez(out/name/'morphology_r2.npz',ids=ids,**{k:v[:,:d] for k,v in result.items()})
    write_json(out/'manifest.json',dict(kind='NEW_SYNTHETIC_FAMILIES_ON_REUSED_NORMAL_AUDIT_NOT_OFFICIAL_OOD',count=len(ids),iaaft_iterations=a.iterations,noise_sigma=float(NOISE),quality=quality,seconds=time.monotonic()-start,source_sha256=sha(__file__),feature_hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*.npz')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=256);p.add_argument('--iterations',type=int,default=60);main(p.parse_args())
