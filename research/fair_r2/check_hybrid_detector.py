"""Verify frozen candidate 05 against fixed inputs; never fit or select on test."""
import argparse
import json
import zipfile
from pathlib import Path
import numpy as np
from common import write_json, sha
from hybrid_detector import FrozenHybridDetector


def main(a):
    root = Path(a.root)
    detector = FrozenHybridDetector(root, a.mask, root/'exports/manifest.json',
                                    root/'hybrid-frozen/manifest.json',
                                    root/'hybrid-frozen/nll-normal-regressors.npz')
    ids = np.asarray([0, 1, 17, 113, 509, 1021, 4321, 9999])
    valid = np.asarray(np.load(a.test, mmap_mode='r', allow_pickle=False)[ids])
    values = detector.predict(valid)
    with zipfile.ZipFile(a.zip) as z:
        expected = np.asarray(json.loads(z.read('result.json'))['ood_scores'])[ids]
    full = np.full((len(ids), *detector.mask.shape), np.nan, np.float32)
    full[:, detector.mask] = valid
    forms = dict(full=detector.predict(full),
                 reversed=detector.predict(valid[::-1])[::-1],
                 chunked=np.concatenate([detector.predict(valid[:3]),
                                         detector.predict(valid[3:])]))
    for x in (valid[0], full[0]):
        assert np.allclose(detector.predict(x), values[:1], rtol=1e-6, atol=1e-5)
    assert detector.predict(np.empty((0, valid.shape[1]))).shape == (0,)
    for v in forms.values():
        assert np.allclose(v, values, rtol=1e-6, atol=1e-5)
    invalid = valid[:1].astype(np.float32).copy()
    invalid[0, 0] = np.nan
    try:
        detector.predict(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted nonfinite observed pixels')
    reference_match = bool(np.allclose(values, expected, rtol=1e-6, atol=1e-5))
    write_json(a.out, dict(kind='FROZEN_HYBRID_API_EQUIVALENCE', count=len(ids),
                           reference_archive_sha256=sha(a.zip), reference_match=reference_match,
                           max_absolute_difference=float(np.max(abs(values-expected))),
                           form_errors={k: float(np.max(abs(v-values))) for k, v in forms.items()},
                           invalid_observed_pixels_rejected=True, empty_input_supported=True,
                           source_sha256=sha(Path(__file__).parent/'hybrid_detector.py'),
                           model_or_test_fitting=False))
    print(Path(a.out).read_text(), flush=True)
    assert reference_match, 'Mismatch with frozen candidate 05 predictions'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'mask', 'test', 'zip', 'out'):
        p.add_argument('--'+name, required=True)
    main(p.parse_args())
