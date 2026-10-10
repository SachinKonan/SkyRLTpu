"""Fit a multiscale linear denoiser on training maps; evaluate only normal holdouts."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from train import load_training,NOISE

p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True);a=p.parse_args()
maps,mask,labels,splits=load_training(a.raw)
sigmas=(0.,.5,1.,2.,4.)
coverage={s:gaussian_filter(mask.astype(np.float32),s) for s in sigmas if s}
mu=float(np.asarray(maps[splits['train'][::32]],np.float64).mean())

def data(split,seed,count=512):
    rng=np.random.default_rng(seed); ids=rng.permutation(splits[split])[:count]
    for i in range(0,len(ids),8):
        clean=np.zeros((len(ids[i:i+8]),*mask.shape),np.float32)
        clean[:,mask]=maps[ids[i:i+8]]
        y=(clean[:,mask]-mu).astype(np.float64)
        noisy=(clean-mu+rng.normal(0,NOISE,clean.shape).astype(np.float32))*mask
        f=np.stack([(noisy if s==0 else gaussian_filter(noisy,(0,s,s))/np.maximum(coverage[s],.05))[:,mask] for s in sigmas],axis=-1).astype(np.float64)
        yield f.reshape((-1,5)),y.ravel()

xx=np.zeros((5,5));xy=np.zeros(5);yy=0.;n=0
for f,y in data('train',744):xx+=f.T@f;xy+=f.T@y;yy+=y@y;n+=len(y)
w=np.linalg.solve(xx+np.eye(5)*np.trace(xx)*1e-6,xy)
err=base=0.;nv=0
for f,y in data('val',745):err+=np.sum((f@w-y)**2);base+=y@y;nv+=len(y)
report=dict(kind='normal-only-linear-denoising-reference',train_maps=512,val_maps=512,sigmas=sigmas,weights=w.tolist(),val_mse=err/nv,val_mean_map_mse=base/nv,val_relative_to_mean=err/base,train_mean=mu)
Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
