"""Frozen per-map inference API for the R2 three-seed conditional-flow ensemble.

The eventual organizer private-evaluation wrapper may differ; this class is a
reusable model API, not a claim of compatibility with an unpublished contract.
It reads model artifacts and the fixed survey mask, never training/test cohorts.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
from flow import predictor
from features import make_predict


class FrozenFlowDetector:
    def __init__(self,models,mask_file,manifest_file):
        models=Path(models);mask_file=Path(mask_file)
        if hashlib.md5(mask_file.read_bytes()).hexdigest()!='29b8bbb8a9ad41d3dacfaf61008a8268':
            raise ValueError('The fixed competition survey mask is required')
        self.mask=np.load(mask_file,allow_pickle=False)
        manifest=json.loads(Path(manifest_file).read_text())
        expected=[f'flow-cosmology-asinh-s{seed}' for seed in range(3)]
        if [m['name'] for m in manifest['models']]!=expected or manifest['weights']!=[1/3]*3:
            raise ValueError('This API supports the frozen R2 three-seed ensemble')
        for m in manifest['models']:
            for name,digest in m['files'].items():
                if hashlib.sha256((models/m['name']/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError(f'Model file changed: {m["name"]}/{name}')
        self.features=make_predict(self.mask,16)
        self.models=[predictor(models/name) for name in expected]

    def predict(self,observed):
        """Return one score per noisy map. Larger means more anomalous.

        Accept a single full/valid-pixel map, [N,1424,176], or [N,132019].
        Batch composition and order do not enter the score definition.
        """
        x=np.asarray(observed)
        if x.shape==self.mask.shape:x=x[None]
        if x.ndim==1:x=x[None]
        if x.ndim==2 and x.shape[1]==int(self.mask.sum()):
            full=np.zeros((len(x),*self.mask.shape),np.float32);full[:,self.mask]=x
        elif x.ndim==3 and x.shape[1:]==self.mask.shape:
            full=np.where(self.mask,np.asarray(x,np.float32),0.)
        else:raise ValueError('Expected masked images or valid-pixel vectors')
        if not len(full):return np.empty(0,dtype=np.float32)
        if not np.isfinite(full[:,self.mask]).all():raise ValueError('Nonfinite observed pixels')
        result=[]
        for start in range(0,len(full),16):
            features=self.features(full[start:start+16])
            result.append(np.mean([m(features)['minimum_base_energy'] for m in self.models],axis=0))
        scores=np.concatenate(result)
        if not np.isfinite(scores).all():raise ValueError('Nonfinite model score')
        return scores
