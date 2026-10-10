"""Assemble checked predictions, frozen weights, research evidence and a CLI."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    root,out=Path(a.root).resolve(),Path(a.out).resolve();ev=root/'.fair/r4-evidence'
    export=ev/'exports-001/exports';report=json.loads((export/'report.json').read_text())
    cli=json.loads((ev/'cli-check.json').read_text());assert cli['max_absolute_error']<1e-6
    assert max(report['raw_api_errors'].values())<1e-4
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(ev/'frozen-candidates-v2',out/'reproduce',ignore=shutil.ignore_patterns('__pycache__'))
    for record in report['receipts'].values():
        p=export/record['filename'];assert sha(p)==record['sha256'];shutil.copy2(p,out)
    for name in ('RESEARCH.json','PLAN.json','CALIBRATION_PLAN.json','CLOSURE_PLAN.json'):
        shutil.copy2(root/'research/fair_r4'/name,out)
    shutil.copy2(root/'research/fair_r4/predict_frozen.py',out/'predict.py')
    shutil.copytree(root/'research/fair_r4',out/'research-source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ev/'figures-001',out/'figures')
    for name in ('comparison-001','score-calibration-001','closure-comparison-001'):
        d=out/'reports'/name;d.mkdir(parents=True)
        for file in ('report.json','comparison.csv'):shutil.copy2(ev/name/file,d)
    shutil.copy2(export/'report.json',out/'reports/export-checks.json')
    shutil.copy2(ev/'cli-check.json',out/'reports/cli-check.json')
    shutil.copy2(root/'.fair/deliverables/fair-r3-2026-10-09/ENVIRONMENT.json',out/'ENVIRONMENT.json')
    # The base snapshot hash is evidence of the unchanged prior model artifacts.
    wt=dict(root=str(root),branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            source_state='Uncommitted research; exact packaged files covered by SHA256SUMS.json.')
    (out/'WORKTREE.json').write_text(json.dumps(wt,indent=2))
    (out/'README.txt').write_text('''FAIR Universe R4 candidates 08 and 09 — 2026-10-09

These are prepared, unsubmitted predictions. No official leaderboard score has
been observed. Synthetic diagnostics do not establish competition performance.

08-cosmology-corrected-hybrid.zip combines three frozen summary flows with two
frozen field flows. It infers each map's two physical coordinates continuously
within the training cosmology hull, then corrects the summary score using a
normal-validation location/scale regression. All neural weights stay fixed.

09-cosmology-corrected-spatial.zip adds the unchanged regional specialist using
a maximum of normal-scaled scores. Each ZIP contains only result.json with
10,000 finite ood_scores, in the original v2 test-file order.

On a frozen new-noise check, 08 improved seven of eight synthetic point estimates
over 06. Both 08 and 09 lost IAAFT morphology sensitivity. Neither is uniformly
better. The clean audit maps and shift families were reused; new measurement
noise does not provide independent cosmological simulations.

At the nominal 1% threshold, 08 flagged 40/3232 normal float16 observations and
09 flagged 43/3232. Float16 roundtrip changed one decision for 08 and none for 09.
The threshold and ensemble scales came from earlier normal calibration data.

Run from this directory in the tested JAX TPU environment:

  python predict.py --bundle reproduce --input observed.npy --output result.json --variant 08

Choose 09 for the spatial variant. Inputs are already noisy observations:
valid-pixel vectors (N,132019) or full survey maps (N,1424,176). A single map is
also accepted. The CLI does not add noise, fit weights, calculate test-cohort
statistics, or rank test samples. Unobserved pixels are ignored. No public data
or training data are required beyond the input maps and packaged fixed assets.

The raw Python API is FrozenR4Detector(bundle).predict(observed, variant='08')
from reproduce/code/fair_r4/detector.py. The portable CLI loads this exact runtime.
The numerical checks covered eight fixed public maps, independent recomputation
of the field component, singleton/reverse/chunk calls, unobserved NaNs, empty
inputs and rejection of nonfinite observed pixels. Singleton/reordering/chunk
checks were exact; raw-vs-cached differences were below 0.000004. The CLI was
also run from an unrelated working directory.

ENVIRONMENT.json records the tested Python/packages. A clean install and CPU
numerical equivalence have not been validated. This API is not claimed to match
an unverified private-evaluation contract.

reproduce/r4-manifest.json was frozen before closure and therefore records its
then-pending closure status. Completed checks are in reports/. Model definition
and score scales did not change after the frozen fresh-noise experiment.

RESEARCH.json and reports/ contain full results, regressions, descriptive paired
bootstrap intervals, caveats and primary-source research links. figures/ holds
standalone PNG/PDF plots. The bootstrap excludes selection uncertainty and the
unknown physical alternatives in the competition. research-source/ records the
research scripts; large regeneration caches are intentionally excluded.

Public test data were used only for frozen pointwise inference and numerical
verification. The export reused verified, previously frozen field scores; its
191-second timing is not the cost of fresh field inference on all 10,000 maps.
Prior candidate ZIP files 04/05/06/07 remain unchanged.
''')
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(files,indent=2))
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=out.name)
    with tarfile.open(archive) as t:
        for relative,digest in files.items():
            assert hashlib.sha256(t.extractfile(out.name+'/'+relative).read()).hexdigest()==digest
    receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files_verified=len(files),
                 candidates=report['receipts'],submitted=False,official_score=None)
    archive.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);main(p.parse_args())
