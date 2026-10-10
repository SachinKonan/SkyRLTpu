"""Three additional shape-noise views of FITTING realizations only."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_r2'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
from common import write_json,sha
from train import load_training,NOISE
from features import make_predict


def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    maps,mask,labels,splits=load_training(a.raw);ids=splits['train']
    fn=make_predict(mask,16);start=time.monotonic();seeds=[9301,9302,9303]
    result=np.lib.format.open_memmap(out/'training_views.npy',mode='w+',dtype=np.float32,
                                     shape=(len(seeds),len(ids),72))
    for view,seed in enumerate(seeds):
        rng=np.random.default_rng(seed)
        for lo in range(0,len(ids),16):
            ix=ids[lo:lo+16];x=np.zeros((len(ix),*mask.shape),np.float32);x[:,mask]=maps[ix]
            x+=rng.normal(0,NOISE,x.shape).astype(np.float32)*mask
            result[view,lo:lo+len(ix)]=fn(x)
            if lo%4096==0:print('view',view,'count',lo+len(ix),'seconds',time.monotonic()-start,flush=True)
    result.flush();np.save(out/'ids.npy',ids)
    assert np.isfinite(result).all()
    write_json(out/'manifest.json',dict(kind='FITTING_REALIZATIONS_ONLY_NEW_NOISE_VIEWS',
        distinct_simulator_realizations=len(ids),additional_noise_views=len(seeds),noise_seeds=seeds,
        noise_sigma=float(NOISE),shape=list(result.shape),seconds=time.monotonic()-start,
        views_sha256=sha(out/'training_views.npy'),ids_sha256=sha(out/'ids.npy'),source_sha256=sha(__file__),
        validation_calibration_audit_or_public_read=False,
        clarification='More observations of the same clean training maps; not additional independent simulator realizations.'))
    print('COMPLETE',time.monotonic()-start,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
