"""Extract frozen Gaussianized coordinates at recorded per-map physical fits."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from flax import serialization

sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'fair_r4'))
from continuous_profile import Profile
from flow import AffineFlow
from common import load_labels,splits,write_json,sha


class FrozenLatents:
    def __init__(self,run,batch=32):
        self.profile=Profile(run);self.batch=batch
        cfg=self.profile.cfg
        params=jax.tree.map(j.asarray,serialization.msgpack_restore((Path(run)/'best.msgpack').read_bytes()))
        net=AffineFlow(cfg['dim'],cfg['layers'],cfg['width'],cfg['scale_bound'])
        self.apply=jax.jit(lambda x,c:net.apply({'params':params},x,c,return_latent=True))

    def __call__(self,features,coordinates):
        x=self.profile.features(features);c=np.asarray(coordinates,np.float32)
        if c.shape!=(len(x),2):raise ValueError('Mismatched physical coordinates')
        if not len(x):return np.empty((0,self.profile.cfg['dim']),np.float32)
        values=[]
        for lo in range(0,len(x),self.batch):
            b=x[lo:lo+self.batch];cc=c[lo:lo+self.batch];count=len(b)
            if count<self.batch:
                b=np.pad(b,((0,self.batch-count),(0,0)));cc=np.pad(cc,((0,self.batch-count),(0,0)))
            values.append(np.asarray(self.apply(j.asarray(b),j.asarray(cc)))[:count])
        result=np.concatenate(values)
        if not np.isfinite(result).all():raise ValueError('Nonfinite latent')
        return result


def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);model=FrozenLatents(a.run)
    index=splits();labels=load_labels(a.raw);start=time.monotonic();reports={}
    def extract(name,features,coordinates,expected):
        z=model(features,coordinates);energy=(z.astype(np.float64)**2).sum(1)
        error=float(np.max(abs(energy-expected)))
        assert np.allclose(energy,expected,rtol=3e-6,atol=2e-4),(name,error)
        reports[name]=dict(count=len(z),max_energy_error=error)
        return z
    for phase in ('val','calibration','audit'):
        features=np.load(Path(a.features)/(phase+'_latent.npy'),allow_pickle=False)
        with np.load(Path(a.profile)/(phase+'.npz'),allow_pickle=False) as p:
            assert np.array_equal(p['ids'],index[phase])
            c=p['continuous_theta'];z=extract(phase,features,c,p['continuous_energy'])
        true_c=(labels[index[phase],:2]-model.profile.norm['condition_mean'])/model.profile.norm['condition_std']
        truth=model(features,true_c)
        np.savez(out/(phase+'.npz'),ids=index[phase],latent=z,true_latent=truth,coordinates=c)
        if phase=='val':
            checks=[model(features[:19][::-1],c[:19][::-1])[::-1],
                    np.concatenate([model(features[:1],c[:1]),model(features[1:7],c[1:7]),model(features[7:19],c[7:19])])]
            errors=[float(np.max(abs(v-z[:19]))) for v in checks]
            assert max(errors)==0
    for path in sorted(Path(a.probes).glob('*.npz')):
        output={}
        with np.load(path,allow_pickle=False) as features,np.load(Path(a.profile)/(path.stem+'.npz'),allow_pickle=False) as p:
            if 'ids' in features.files:output['ids']=features['ids']
            for key in features.files:
                if key=='ids':continue
                c=p[key+'__continuous_theta'];expected=p[key+'__continuous_energy']
                output[key+'__latent']=extract(path.stem+'/'+key,features[key],c,expected)
                output[key+'__coordinates']=c
        np.savez(out/(path.stem+'.npz'),**output)
    report=dict(kind='FROZEN_LATENT_EXTRACTION_NORMAL_AND_REUSED_SYNTHETIC',
                public_test_read=False,model_fitting=False,checkpoint_sha256=sha(Path(a.run)/'best.msgpack'),
                source_sha256=sha(__file__),pointwise_errors=errors,datasets=reports,seconds=time.monotonic()-start)
    write_json(out/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('run','profile','features','probes','raw','out'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
