"""Frozen multiscale physical summary. All operations are per map."""
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as j

@lru_cache(maxsize=4)
def extractor(h,w):
    fy=np.fft.fftfreq(h)[:,None]; fx=np.fft.rfftfreq(w)[None,:]
    r=np.sqrt(fy**2+fx**2)
    bins=j.asarray(np.clip(np.digitize(r.ravel(),np.geomspace(.002,.72,21))-1,0,19),j.int32)
    counts=j.maximum(j.bincount(bins,length=20),1)
    kernels=j.asarray(np.stack([np.exp(-2*np.pi**2*s*s*r*r) for s in (1.,2.,4.,8.)]),j.float32)
    @jax.jit
    def extract(x,mask):
        x=x*mask; count=mask.sum(); f=j.fft.rfft2(x)
        power=j.abs(f)**2/count
        spec=jax.vmap(lambda p:j.log(j.bincount(bins,weights=p.ravel(),length=20)/counts+1e-9))(power)
        smask=j.fft.irfft2(j.fft.rfft2(mask)[None]*kernels,s=(h,w))
        smooth=j.fft.irfft2(f[:,None]*kernels[None],s=(h,w))/j.maximum(smask[None],.1)
        fields=[x]+[smooth[:,i] for i in range(4)]
        feats=[spec]; standardized=[]
        vy=(mask*j.roll(mask,1,0)).at[0,:].set(0)
        vx=(mask*j.roll(mask,1,1)).at[:,0].set(0)
        for q in fields:
            mu=(q*mask).sum((1,2))/count
            c=(q-mu[:,None,None])*mask
            sd=j.sqrt((c*c).sum((1,2))/count+1e-10)
            z=c/sd[:,None,None]; standardized.append(z)
            dx=(q-j.roll(q,1,axis=2))*vx
            dy=(q-j.roll(q,1,axis=1))*vy
            feats.append(j.stack([mu,j.log(sd),(z**3*mask).sum((1,2))/count,(z**4*mask).sum((1,2))/count,((z>2)*mask).sum((1,2))/count,((z<-2)*mask).sum((1,2))/count,j.log((dx*dx).sum((1,2))/vx.sum()+1e-10),j.log((dy*dy).sum((1,2))/vy.sum()+1e-10)],axis=1))
        for a,b in [(0,1),(1,2),(2,3),(3,4)]:
            za,zb=standardized[a],standardized[b]
            feats.append(j.stack([(za*zb*mask).sum((1,2))/count,(za*za*zb*mask).sum((1,2))/count,(za*za*zb*zb*mask).sum((1,2))/count],axis=1))
        return j.concatenate(feats,axis=1)
    return extract


def make_predict(mask,batch=16):
    fn=extractor(*mask.shape); m=j.asarray(mask,j.float32)
    def predict(x):
        out=[]
        for i in range(0,len(x),batch):
            p=x[i:i+batch]; n=len(p)
            if n<batch:p=np.pad(p,((0,batch-n),(0,0),(0,0)))
            out.append(np.asarray(fn(j.asarray(p,j.float32),m))[:n])
        return np.concatenate(out)
    return predict
