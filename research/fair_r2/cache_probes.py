"""Cache descriptors of training-derived probes; never reads the public test."""
import argparse,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
from train import load_training,NOISE
from features import make_predict
from diagnostics import shift,DEVELOPMENT,NOVEL
from common import write_json,sha

p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=512);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False);maps,mask,labels,split=load_training(a.raw);predict=make_predict(mask,16);start=time.monotonic()
for phase,seed,source,families in [('development',474,'val',DEVELOPMENT),('r1_novel_now_reused',475,'audit',NOVEL)]:
    ids=np.random.default_rng(seed).permutation(split[source])[:a.count];result={}
    for i in range(0,len(ids),16):
        ix=ids[i:i+16];x=np.zeros((len(ix),*mask.shape),np.float32);x[:,mask]=maps[ix]
        ridx=np.array([split['train'][160*(n//256)+(n%160)] for n in ix]);ref=np.zeros_like(x);ref[:,mask]=maps[ridx]
        rng=np.random.default_rng(7300+i+int(source=='audit'));noise=rng.normal(0,NOISE,x.shape).astype(np.float32)*mask
        result.setdefault('normal',[]).append(predict(x+noise));result.setdefault('second',[]).append(predict(x+rng.normal(0,NOISE,x.shape).astype(np.float32)*mask))
        for family in families:
            for strength in (.2,.5):result.setdefault(f'{family}:{strength}',[]).append(predict(shift(x,mask,family,strength,ref)+noise))
    result={k:np.concatenate(v) for k,v in result.items()};np.savez(out/f'{phase}.npz',ids=ids,**result)
    print(phase,{k:v.shape for k,v in result.items()},flush=True)
write_json(out/'manifest.json',dict(kind='normal_data_and_synthetic_probes_NOT_OFFICIAL_OOD',count=a.count,reused_audit=True,seconds=time.monotonic()-start,files={p.name:sha(p) for p in out.glob('*.npz')}))
