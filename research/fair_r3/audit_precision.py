"""Normal-only check of the public file's float16 storage precision."""
import argparse
import sys
import json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_r2'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
from train import load_training,NOISE
from features import make_predict
from flow import predictor
from spatial_features import SpatialSummary
from common import write_json


def main(a):
    root=Path(a.root);maps,mask,labels,splits=load_training(a.raw)
    ids=np.random.default_rng(60019).permutation(splits['val'])[:128]
    x=np.zeros((len(ids),*mask.shape),np.float32);x[:,mask]=maps[ids]
    x+=np.random.default_rng(61900).normal(0,NOISE,x.shape).astype(np.float32)*mask
    rounded=x.astype(np.float16).astype(np.float32)
    base=make_predict(mask,16);extra=SpatialSummary(mask,16)
    f=base(x);fr=base(rounded);s=extra(x);sr=extra(rounded)
    reference=np.load(a.spatial_training).astype(np.float64)
    sd=np.maximum(reference.std(0),1e-12)
    relative=(np.column_stack([fr,sr])-np.column_stack([f,s]))/sd
    scores=[];scores_rounded=[]
    for seed in range(3):
        fn=predictor(root/f'flow-cosmology-asinh-s{seed}')
        scores.append(fn(f)['minimum_base_energy']);scores_rounded.append(fn(fr)['minimum_base_energy'])
    values=np.mean(scores,axis=0);other=np.mean(scores_rounded,axis=0)
    report=dict(kind='NORMAL_ONLY_FLOAT16_PRECISION_AUDIT',count=len(ids),
        public_data_read=False,model_fitting=False,
        median_absolute_summary_score_change=float(np.median(abs(values-other))),
        max_absolute_summary_score_change=float(np.max(abs(values-other))),
        physical_feature_rms_change_in_training_sd=float(np.sqrt(np.mean(relative[:,:72]**2))),
        morphology_feature_rms_change_in_training_sd=float(np.sqrt(np.mean(relative[:,72:117]**2))),
        regional_feature_rms_change_in_training_sd=float(np.sqrt(np.mean(relative[:,117:]**2))),
        limitation='Checks 128 normal maps; not a proof of negligible tail effects on all possible maps.')
    write_json(a.out,report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','raw','spatial-training','out'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
