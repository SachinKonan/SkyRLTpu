"""Freeze research candidates before fresh-noise closure or public inference."""
import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=False)
    base = root/'.fair/deliverables/fair-r3-2026-10-09/reproduce'
    shutil.copytree(base, out/'base')
    shutil.copytree(base/'code', out/'code')
    runtime = out/'code/fair_r4'; runtime.mkdir()
    for name in ('continuous_profile.py','score_correction.py','detector.py'):
        shutil.copy2(root/'research/fair_r4'/name, runtime/name)
    comparison = root/'.fair/r4-evidence/score-calibration-001'
    shutil.copy2(comparison/'normal-score-regressors.npz',out)
    scales = {}
    for mode in ('continuous', 'centered', 'scaled'):
        for suffix in ('summary', 'hybrid'):
            with np.load(comparison/f'{mode}-{suffix}.npz', allow_pickle=False) as f:
                mid, upper = np.quantile(f['calibration'], [.5, .9])
            scales[mode if suffix=='summary' else mode+'-hybrid'] = [float(mid),float(upper-mid)]
    manifest = dict(format='FAIR_R4_POINTWISE_V1', frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    normal_center_scale=scales, files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
                    score_definitions={'08':'Equal normal-scaled hybrid with continuous energy corrected by normal-only inferred-cosmology mean and scale.',
                                       '09':'Maximum of normal-scaled 08 and the unchanged regional specialist.'},
                    fit_scope='Normal validation location/scale regression; normal calibration ensemble scales; public data not read.',
                    closure_status='Pending frozen fresh-noise check; staged research candidates only.',
                    official_score=None,submitted=False,
                    limitations=['Reused synthetic diagnostic tradeoffs are not an official score.',
                                 'Per-map physical coordinate fitting is frozen inference; network weights never adapt.',
                                 'Private evaluation interface and clean installation are unverified.'])
    (out/'r4-manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    print(json.dumps(dict(path=str(out),files=len(manifest['files']),manifest_sha256=sha(out/'r4-manifest.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
