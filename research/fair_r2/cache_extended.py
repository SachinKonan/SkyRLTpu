"""Add spatial wavelet summaries to the original physical features, normal data only."""
import argparse,sys,time,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
from train import load_training,NOISE
from diagnostics import shift,DEVELOPMENT,NOVEL
from common import write_json,sha
from wavelets import WaveletSummary

p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--features',required=True);p.add_argument('--old-probes',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False);maps,mask,labels,splits=load_training(a.raw);fn=WaveletSummary(mask,8);start=time.monotonic()
x=np.zeros((4,*mask.shape),np.float32);x[:,mask]=maps[splits['val'][:4]];x+=np.random.default_rng(44).normal(0,NOISE,x.shape).astype(np.float32)*mask
v=fn(x);alone=fn(x[:1]);reverse=fn(x[::-1])[::-1]
assert np.isfinite(v).all() and np.allclose(v[:1],alone,rtol=1e-5,atol=1e-5) and np.allclose(v,reverse,rtol=1e-5,atol=1e-5)
write_json(out/'smoke.json',dict(**fn.description,finite=True,independence_verified=True,seconds=time.monotonic()-start));print((out/'smoke.json').read_text(),flush=True)
if a.smoke:raise SystemExit(0)
for name,ids in splits.items():
    rng=np.random.default_rng(300+list(splits).index(name));values=[]
    # Noise draws use the original batch=16 sequence for exact pairing.
    for i in range(0,len(ids),16):
        ix=ids[i:i+16];x=np.zeros((len(ix),*mask.shape),np.float32);x[:,mask]=maps[ix];x+=rng.normal(0,NOISE,x.shape).astype(np.float32)*mask;values.append(fn(x))
    wave=np.concatenate(values);base=np.load(Path(a.features)/f'{name}_latent.npy');np.save(out/f'{name}_latent.npy',np.column_stack([base,wave]));print(name,len(wave),time.monotonic()-start,flush=True)
probes=out/'probes';probes.mkdir()
for phase,seed,source,families in [('development',474,'val',DEVELOPMENT),('r1_novel_now_reused',475,'audit',NOVEL)]:
    previous=np.load(Path(a.old_probes)/(phase+'.npz'));ids=previous['ids'];values={}
    for i in range(0,len(ids),16):
        ix=ids[i:i+16];x=np.zeros((len(ix),*mask.shape),np.float32);x[:,mask]=maps[ix]
        ridx=np.array([splits['train'][160*(n//256)+(n%160)] for n in ix]);ref=np.zeros_like(x);ref[:,mask]=maps[ridx]
        rng=np.random.default_rng(7300+i+int(source=='audit'));noise=rng.normal(0,NOISE,x.shape).astype(np.float32)*mask
        values.setdefault('normal',[]).append(fn(x+noise));values.setdefault('second',[]).append(fn(x+rng.normal(0,NOISE,x.shape).astype(np.float32)*mask))
        for family in families:
            for strength in (.2,.5):values.setdefault(f'{family}:{strength}',[]).append(fn(shift(x,mask,family,strength,ref)+noise))
    result={k:np.column_stack([previous[k],np.concatenate(v)]) for k,v in values.items()};np.savez(probes/(phase+'.npz'),ids=ids,**result);print('probes',phase,time.monotonic()-start,flush=True)
write_json(out/'manifest.json',dict(kind='normal_and_synthetic_only',definition=fn.description,source_sha256=sha(Path(__file__).parent/'wavelets.py'),seconds=time.monotonic()-start,feature_hashes={p.name:sha(p) for p in out.glob('*.npy')}))
