"""API equivalence checks against frozen predictions; no fitting or selection."""
import argparse,json,zipfile
from pathlib import Path
import numpy as np
from detector import FrozenFlowDetector
from common import write_json,sha

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mask',required=True);p.add_argument('--manifest',required=True);p.add_argument('--test',required=True);p.add_argument('--zip',required=True);p.add_argument('--out',required=True);a=p.parse_args()
detector=FrozenFlowDetector(a.root,a.mask,a.manifest);data=np.load(a.test,mmap_mode='r',allow_pickle=False)
ids=np.asarray([0,1,17,113,509,1021,4321,9999]);valid=np.asarray(data[ids]);values=detector.predict(valid)
with zipfile.ZipFile(a.zip) as z:expected=np.asarray(json.loads(z.read('result.json'))['ood_scores'])[ids]
full=np.full((len(ids),*detector.mask.shape),np.nan,np.float32);full[:,detector.mask]=valid
forms=dict(full_with_unobserved_nan=detector.predict(full),reversed=detector.predict(valid[::-1])[::-1],chunked=np.concatenate([detector.predict(valid[:3]),detector.predict(valid[3:])]))
assert np.allclose(detector.predict(valid[0]),values[:1],rtol=1e-5,atol=1e-4)
assert np.allclose(detector.predict(full[0]),values[:1],rtol=1e-5,atol=1e-4)
assert detector.predict(np.empty((0,valid.shape[1]))).shape==(0,)
for value in forms.values():assert np.allclose(value,values,rtol=1e-5,atol=1e-4)
reference_match=bool(np.allclose(values,expected,rtol=1e-5,atol=1e-3))
invalid=valid[:1].astype(np.float32).copy();invalid[0,0]=np.nan
try:detector.predict(invalid)
except ValueError:pass
else:raise AssertionError('Accepted a nonfinite observed pixel')
write_json(a.out,dict(kind='FROZEN_INFERENCE_API_EQUIVALENCE',reference_archive_sha256=sha(a.zip),detector_source_sha256=sha(Path(__file__).parent/'detector.py'),count=len(ids),reference_match=reference_match,max_absolute_difference_from_frozen_tpu_export=float(np.max(abs(values-expected))),max_relative_difference_from_frozen_tpu_export=float(np.max(abs(values-expected)/np.maximum(abs(expected),1e-8))),form_errors={k:float(np.max(abs(v-values))) for k,v in forms.items()},invalid_observed_pixels_rejected=True,empty_input_supported=True,model_or_test_fitting=False))
print(Path(a.out).read_text())
assert reference_match,'CPU/device mismatch against frozen reference'
