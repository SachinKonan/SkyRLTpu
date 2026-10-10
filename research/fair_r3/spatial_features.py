"""Fixed per-map digital morphology and regional descriptors.

These use the same survey mask for every map and do not fit to any cohort.
They are inspired by Minkowski-functional analyses, not a reproduction of an
unbiased continuum estimator. Boundary contributions remain part of the fixed
masked statistic and are handled empirically by normal-only density fitting.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
import jax
import jax.numpy as j

jax.config.update('jax_default_matmul_precision','highest')


def digital_geometry(binary):
    """Area, edge perimeter, and Euler characteristic (4-connected foreground).

    Accept [B,H,W]. Pad the image with background so boundary-touching objects
    are counted. Background has dual 8-connectivity when counting holes.
    """
    b=j.pad(binary.astype(j.int32),((0,0),(1,1),(1,1)))
    area=b.sum((1,2))
    perimeter=j.abs(b[:,1:]-b[:,:-1]).sum((1,2))+j.abs(b[:,:,1:]-b[:,:,:-1]).sum((1,2))
    a,c,d,e=b[:,:-1,:-1],b[:,:-1,1:],b[:,1:,:-1],b[:,1:,1:]
    n=a+c+d+e
    diagonal=((a==1)&(e==1)&(c==0)&(d==0))|((a==0)&(e==0)&(c==1)&(d==1))
    chi=((n==1).sum((1,2))-(n==3).sum((1,2))+2*diagonal.sum((1,2)))/4.
    return j.stack([area,perimeter,chi],axis=-1)


class SpatialSummary:
    def __init__(self,mask,batch=16):
        self.mask=np.asarray(mask,bool);self.batch=batch
        h,w=self.mask.shape
        if h%4:
            raise ValueError('Four longitudinal regions require divisible height')
        self.scales=(1.,3.,6.)
        ph=int(np.ceil((h+64)/128)*128);pw=int(np.ceil((w+64)/128)*128)
        py=(ph-h)//2;px=(pw-w)//2
        fy=np.fft.fftfreq(ph)[:,None];fx=np.fft.rfftfreq(pw)[None,:]
        radius2=fy*fy+fx*fx
        kernel=np.stack([np.exp(-2*np.pi**2*s*s*radius2) for s in self.scales])
        padded=np.pad(self.mask,((py,ph-h-py),(px,pw-w-px)))
        denominator=np.fft.irfft2(np.fft.rfft2(padded)[None]*kernel,s=(ph,pw))[:,py:py+h,px:px+w]
        distance=distance_transform_edt(np.pad(self.mask,1))[1:-1,1:-1]
        supports=np.stack([self.mask&(distance>s) for s in self.scales])
        supports_all=np.concatenate([self.mask[None],supports],axis=0)
        if min(supports.sum((1,2)))<100:
            raise ValueError('Too little supported survey area')
        for support in supports_all:
            if min(support.reshape(4,h//4,w).sum((1,2)))<20:
                raise ValueError('Empty or tiny longitudinal region')
        threshold=j.asarray([-1.,0.,1.,2.,3.],j.float32)
        kernel=j.asarray(kernel,j.float32);den=j.asarray(denominator,j.float32)
        support_arrays=[j.asarray(s,j.float32) for s in supports_all]
        crop=lambda x:x[...,py:py+h,px:px+w]
        @jax.jit
        def extract(x):
            p=j.pad(x,((0,0),(py,ph-h-py),(px,pw-w-px)))
            smooth=crop(j.fft.irfft2(j.fft.rfft2(p)[:,None]*kernel[None],s=(ph,pw)))/j.maximum(den[None],1e-4)
            fields=[x]+[smooth[:,i] for i in range(3)]
            moments=[]
            for q,support in zip(fields,support_arrays):
                count=support.sum();mu=(q*support).sum((1,2))/count
                centered=q-mu[:,None,None]
                sd=j.sqrt((centered*centered*support).sum((1,2))/count+1e-12)
                moments.append((mu,sd,centered/sd[:,None,None]))
            morphology=[]
            for i in (1,2,3):
                support=support_arrays[i];z=moments[i][2]
                # Process thresholds separately to keep intermediate memory small.
                for t in threshold:
                    b=(z>t)&(support[None]>0)
                    morphology.append(digital_geometry(b)/support.sum())
            regional=[]
            for i in (0,2,3):
                q=fields[i];support=support_arrays[i];mu,sd,_=moments[i]
                qr=q.reshape(len(q),4,h//4,w);mr=support.reshape(4,h//4,w)
                count=mr.sum((1,2));local_mu=(qr*mr).sum((2,3))/count
                c=qr-local_mu[:,:,None,None]
                local_sd=j.sqrt((c*c*mr).sum((2,3))/count+1e-12)
                z=c/local_sd[:,:,None,None]
                v=j.stack([(local_mu-mu[:,None])/sd[:,None],
                           j.log(local_sd/sd[:,None]),
                           (z**3*mr).sum((2,3))/count,
                           (z**4*mr).sum((2,3))/count],axis=-1)
                regional.append(v.reshape(len(q),16))
            return j.concatenate(morphology+regional,axis=-1)
        self._extract=extract
        self.description=dict(dimensions=93,morphology_dimensions=45,regional_dimensions=48,
            thresholds=[-1,0,1,2,3],morphology_smoothing_pixels=list(self.scales),
            regional_smoothing_pixels=[0,3,6],regional_count=4,
            mask_shape=[h,w],fft_shape=[ph,pw],support_counts=supports.sum((1,2)).tolist(),
            geometry='Four-connected foreground; edge-count perimeter; fixed-mask digital statistics.',
            input_normalization='Per-map supported mean and standard deviation; no cohort fitting.')

    def __call__(self,x):
        x=np.asarray(x)
        if x.ndim!=3 or x.shape[1:]!=self.mask.shape:
            raise ValueError('Expected full map batch')
        if not len(x):
            return np.empty((0,93),np.float32)
        if not np.isfinite(x[:,self.mask]).all():
            raise ValueError('Nonfinite observed pixels')
        x=np.where(self.mask,x,0.).astype(np.float32)
        result=[]
        for lo in range(0,len(x),self.batch):
            b=x[lo:lo+self.batch];n=len(b)
            if n<self.batch:b=np.pad(b,((0,self.batch-n),(0,0),(0,0)))
            result.append(np.asarray(self._extract(j.asarray(b)))[:n])
        return np.concatenate(result)
