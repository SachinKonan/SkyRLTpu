"""Frozen pointwise inference only, using a verified existing field-score cache."""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
import numpy as np


def main(a):
    bundle=Path(a.bundle);sys.path.insert(0,str(bundle.resolve()/'code/fair_r4'))
    from detector import FrozenR4Detector
    from common import write_json,sha
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    model=FrozenR4Detector(bundle);b=model.base
    assert sha(a.test)==b.manifest['public_test_sha256']
    assert sha(a.field_cache)==b.manifest['frozen_field_cache_sha256']
    maps=np.load(a.test,mmap_mode='r',allow_pickle=False)
    assert maps.shape==(10000,int(b.mask.sum())) and maps.dtype==np.float16
    with np.load(a.field_cache,allow_pickle=False) as f:field=f['field_conditioned_score']
    assert field.shape==(10000,) and np.isfinite(field).all()
    start=time.monotonic();features=[]
    for lo in range(0,len(maps),16):
        full=b._full(maps[lo:lo+16])
        features.append(np.column_stack([b.physical(full),b.spatial(full)]))
        if lo%512==0:
            write_json(out/'progress.json',dict(stage='fixed_features',count=lo+len(full),seconds=time.monotonic()-start))
    features=np.concatenate(features);comp=model.summary_components(features)
    combined=model.combine(comp,field)
    write_json(out/'progress.json',dict(stage='raw_api_checks',count=len(maps),seconds=time.monotonic()-start))
    # Compare only to verify the unchanged legacy implementation and row order.
    # These differences are not used to choose any model or score setting.
    legacy_error={}
    for variant,filename in [('06','06-noise-trained-hybrid.zip'),('07','07-noise-hybrid-spatial.zip')]:
        with zipfile.ZipFile(Path(a.legacy_exports)/filename) as z:
            old=np.asarray(json.loads(z.read('result.json'))['ood_scores'])
        legacy_error[variant]=float(np.max(abs(old-combined[variant])))
        assert np.allclose(old,combined[variant],rtol=1e-6,atol=2e-5)
    receipts={}
    for variant,name in [('08','08-cosmology-corrected-hybrid.zip'),('09','09-cosmology-corrected-spatial.zip')]:
        value=np.asarray(combined[variant]);assert value.shape==(10000,) and np.isfinite(value).all()
        path=out/name
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('result.json',json.dumps({'ood_scores':value.tolist()},allow_nan=False))
        with zipfile.ZipFile(path) as z:
            assert z.namelist()==['result.json']
            assert np.array_equal(json.loads(z.read('result.json'))['ood_scores'],value)
        receipts[variant]=dict(filename=name,count=len(value),sha256=sha(path),submitted=False)
    np.savez(out/'component_scores.npz',field_score=field,**comp,score08=combined['08'],score09=combined['09'])
    # Fixed indices, selected before inference and unrelated to scores.
    ids=np.asarray([0,1,17,101,1007,4095,7001,9999]);raw=np.asarray(maps[ids])
    actual=model.predict_both(raw)
    raw_errors={k:float(np.max(abs(v-combined[k][ids]))) for k,v in actual.items()}
    assert max(raw_errors.values())<1e-4
    tests={}
    alone=model.predict_both(raw[:1]);reverse=model.predict_both(raw[::-1])
    left=model.predict_both(raw[:3]);right=model.predict_both(raw[3:])
    full=b._full(raw);full[:,~b.mask]=np.nan;unobserved=model.predict_both(full)
    for k,v in actual.items():
        tests[k]=dict(single=float(np.max(abs(v[:1]-alone[k]))),
                      reverse=float(np.max(abs(v-reverse[k][::-1]))),
                      chunks=float(np.max(abs(v-np.r_[left[k],right[k]]))),
                      outside_mask_nan=float(np.max(abs(v-unobserved[k]))))
        assert max(tests[k].values())<1e-4
    assert all(len(v)==0 for v in model.predict_both(raw[:0]).values())
    invalid=raw[:1].astype(np.float32);invalid[0,0]=np.nan
    try:model.predict_both(invalid)
    except ValueError:pass
    else:raise AssertionError('Nonfinite observed pixel was accepted')
    np.save(out/'api-check-input.npy',raw);np.savez(out/'api-check-expected.npz',**actual)
    write_json(out/'report.json',dict(kind='FROZEN_POINTWISE_PUBLIC_INFERENCE_NOT_LEADERBOARD_SCORE',
        public_test_use='Independent frozen inference and numerical verification only; no fitting, normalization, ranking or selection.',
        manifest_sha256=sha(bundle/'r4-manifest.json'),field_cache_reused=True,
        legacy_reproduction_errors=legacy_error,raw_api_errors=raw_errors,pointwise_errors=tests,
        api_check_indices=ids.tolist(),empty_and_nonfinite_guards_checked=True,
        seconds=time.monotonic()-start,receipts=receipts,
        timing_limit='Reuses prior field inference; timing is not the full cost of fresh inference on 10000 maps.',
        submitted=False,official_score=None))
    print(json.dumps(dict(receipts=receipts,raw_api_errors=raw_errors,pointwise_errors=tests,
                          seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('bundle','test','field-cache','legacy-exports','out'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
