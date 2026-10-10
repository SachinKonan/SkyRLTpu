"""Isolate backend differences on normal development inputs, not test selection."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import jax
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
from features import make_predict
from flow import predictor
from common import write_json

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--raw',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.root);raw=Path(a.raw)
maps=np.load(raw/'WIDE12H_bin2_2arcmin_kappa_newrealization.npy',mmap_mode='r').reshape(25856,-1);mask=np.load(raw/'WIDE12H_bin2_2arcmin_mask.npy')
with np.load(root/'probes72/development.npz') as f:ids=f['ids'][:16];ref=f['normal'][:16]
x=np.zeros((len(ids),*mask.shape),np.float32);x[:,mask]=maps[ids];x+=np.random.default_rng(7300).normal(0,.4/np.sqrt(240),x.shape).astype(np.float32)*mask
features=make_predict(mask,16)(x);delta=features-ref
record=dict(devices=[str(d) for d in jax.devices()],max_feature_abs_error=float(np.max(abs(delta))),per_feature_max_abs_error=np.max(abs(delta),axis=0).tolist(),flow={})
for seed in range(3):
    run=root/f'flow-cosmology-asinh-s{seed}';fn=predictor(run);cpu_all=fn(features)['minimum_base_energy'];cpu_fixed=fn(ref)['minimum_base_energy']
    with np.load(run/'development_scores.npz') as f:target=f['normal__minimum_base_energy'][:16]
    record['flow'][str(seed)]=dict(full_pipeline_max_abs_error=float(np.max(abs(cpu_all-target))),fixed_tpu_features_max_abs_error=float(np.max(abs(cpu_fixed-target))),feature_change_effect_max_abs=float(np.max(abs(cpu_all-cpu_fixed))))
write_json(a.out,record);print(json.dumps(record),flush=True)
