"""Masked first/second-order Morlet summaries; deterministic, per map, no fitting."""
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as j

class WaveletSummary:
    def __init__(self,mask,batch=8):
        self.batch=batch;self.mask=mask.astype(np.float32);h,w=mask.shape;self.original_shape=(h,w)
        counts=mask.reshape(h//2,2,w//2,2).sum((1,3));m=(counts>=3).astype(np.float32);h,w=m.shape
        ky=2*np.pi*np.fft.fftfreq(h)[:,None];kx=2*np.pi*np.fft.fftfreq(w)[None,:]
        psi=[];gauss=[];scales=[];angles=[]
        for level,sigma in enumerate([1.,2.,4.,8.]):
            for angle in np.arange(4)*np.pi/4:
                k0x=2.5*np.cos(angle)/sigma;k0y=2.5*np.sin(angle)/sigma
                g=np.exp(-.5*sigma*sigma*(kx*kx+ky*ky))
                # A complex Morlet filter, with exactly zero DC and unit L2 norm.
                wave=np.exp(-.5*sigma*sigma*((kx-k0x)**2+(ky-k0y)**2))-np.exp(-.5*2.5**2)*g
                wave[0,0]=0;wave/=np.sqrt(np.square(wave).sum()/(h*w))
                psi.append(wave);gauss.append(g);scales.append(level);angles.append(int(round(angle/(np.pi/4))))
        psi=np.asarray(psi,np.float32);gauss=np.asarray(gauss,np.float32);fm=np.fft.fft2(m)
        coverage=np.fft.ifft2(fm[None]*gauss).real.astype(np.float32)
        mask_wave=np.fft.ifft2(fm[None]*psi).astype(np.complex64)
        spatial=np.fft.ifft2(psi);energy=np.fft.fft2(np.abs(spatial)**2)
        noise_var=np.maximum(np.fft.ifft2(fm[None]*energy).real,.05).astype(np.float32)
        weight=(m[None]*(coverage>.65)*(noise_var>.15)).astype(np.float32)
        # Exclude centers whose primary support crosses the image boundary.
        for index,level in enumerate(scales):
            border=int(4*(2**level));weight[index,:border,:]=0;weight[index,-border:,:]=0;weight[index,:,:border]=0;weight[index,:,-border:]=0
        if np.any(weight.sum((1,2))<100):raise ValueError('Insufficient supported wavelet pixels')
        pairs=[(a,b) for a in range(16) for b in range(16) if scales[b]>scales[a] and (angles[b]-angles[a])%4 in [0,2]]
        src=np.array([a for a,b in pairs]);dst=np.array([b for a,b in pairs])
        self.description={'kind':'masked_morlet_L1_L2_kurtosis_and_two_layer_modulus','input_pooling':2,'scales_pooled_pixels':[1,2,4,8],'angles':4,'first_order_filters':16,'second_order_pairs':len(pairs),'dimensions':3*16+2*len(pairs),'supported_pixel_counts':weight.sum((1,2)).tolist()}
        pm=j.asarray(self.mask);counts=j.asarray(np.maximum(counts,1));m=j.asarray(m);psi=j.asarray(psi);gauss=j.asarray(gauss);coverage=j.asarray(np.maximum(coverage,.1));mask_wave=j.asarray(mask_wave);noise_sd=j.sqrt(j.asarray(noise_var));weight=j.asarray(weight);denom=weight.sum((1,2));src=j.asarray(src);dst=j.asarray(dst)
        @jax.jit
        def extract(x):
            x=(x*pm).reshape(len(x),h,2,w,2).sum((2,4))/counts
            x=x*m;f=j.fft.fft2(x)
            local=j.fft.ifft2(f[:,None]*gauss[None]).real/coverage
            response=(j.fft.ifft2(f[:,None]*psi[None])-local*mask_wave)/noise_sd
            amp=j.abs(response)
            first=(amp*weight).sum((2,3))/denom
            second=(amp**2*weight).sum((2,3))/denom
            fourth=(amp**4*weight).sum((2,3))/denom
            af=j.fft.fft2(amp*m)
            alocal=j.fft.ifft2(af[:,src]*gauss[dst]).real/coverage[dst]
            response2=(j.fft.ifft2(af[:,src]*psi[dst])-alocal*mask_wave[dst])/noise_sd[dst]
            mag=j.abs(response2);s1=(mag*weight[dst]).sum((2,3))/denom[dst];s2=(mag*mag*weight[dst]).sum((2,3))/denom[dst]
            return j.concatenate([j.log(first+1e-8),j.log(second+1e-12),j.log(fourth/(second**2+1e-20)+1e-8),j.log((s1+1e-8)/(first[:,src]+1e-8)),j.log((s2+1e-12)/(first[:,src]**2+1e-12))],axis=1)
        self.extract=extract
    def __call__(self,x):
        values=[]
        for start in range(0,len(x),self.batch):
            part=x[start:start+self.batch];n=len(part)
            if n<self.batch:part=np.pad(part,((0,self.batch-n),(0,0),(0,0)))
            values.append(np.asarray(self.extract(j.asarray(part,j.float32)))[:n])
        return np.concatenate(values)
