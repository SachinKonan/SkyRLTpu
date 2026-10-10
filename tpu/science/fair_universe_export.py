"""Export an evaluated, frozen detector without fitting on official test inputs.

Run using the same frozen runtime tree as the evaluation. Public submission
contains only result.json. The detector package is a separate reproducibility
artifact, not a claim about the unpublished private-phase submission format.
"""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from tpu.science.fair_universe.contract import (
    TEST_NAME, TEST_COUNT, bundle_identity, checked_scores, digest,
)
from tpu.science.fair_universe.evaluate import Predictor, artifact_hashes, command
from tpu.science.fair_universe.vendor.fairdiscover.submission import load_test_batch


def write_result(scores, output):
    scores = checked_scores(scores, TEST_COUNT)
    output = Path(output)
    payload = json.dumps({'ood_scores': scores.tolist()}, allow_nan=False, separators=(',', ':'))
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('result.json', payload)
    return dict(count=len(scores), sha256=digest(output))


def export(candidate, dataset, evaluation, test_file, output, python, chip=0):
    candidate, dataset, evaluation, test_file, output = map(Path,
        (candidate, dataset, evaluation, test_file, output))
    result = json.loads((evaluation/'result.json').read_text())
    config = json.loads((evaluation/'config.json').read_text())
    manifest = json.loads((dataset/'manifest.json').read_text())
    if result.get('correctness') != 1 or not result.get('independence', {}).get('passed'):
        raise ValueError('Detector must pass evaluation and prediction independence first')
    if digest(candidate) != result['candidate_sha256'] or bundle_identity(candidate, config, manifest) != result['identity']:
        raise ValueError('Candidate, helper code, config, or data changed after evaluation')
    if artifact_hashes(evaluation/'artifacts') != result['artifacts']:
        raise ValueError('Detector weights changed after evaluation')
    if test_file.name != TEST_NAME:
        raise ValueError('Use the current official Phase-2 test release')
    test = np.load(test_file, mmap_mode='r', allow_pickle=False)
    if len(test) != TEST_COUNT:
        raise ValueError('Official test release must contain exactly 10000 maps')
    output.mkdir(parents=True, exist_ok=False)
    run = output/'predict'
    for directory in ('cache', 'input'):
        (run/directory).mkdir(parents=True)
    shutil.copy2(evaluation/'config.json', run/'config.json')
    shutil.copytree(evaluation/'artifacts', run/'artifacts')
    mask = np.load(dataset/'fit/val/mask.npy', allow_pickle=False)
    predictor = Predictor(command(candidate, run, dataset, python, 'predict', chip), run, 3600)
    scores = []
    try:
        for start in range(0, TEST_COUNT, 32):
            maps = load_test_batch(test, start, min(start+32, TEST_COUNT), mask)
            # Official test maps are already noisy. No augmentation, fitting,
            # calibration, rank transformation, or cross-sample normalization.
            scores.extend(predictor(maps).tolist())
    finally:
        predictor.close()
    if artifact_hashes(run/'artifacts') != result['artifacts']:
        raise ValueError('Prediction modified detector artifacts')
    if bundle_identity(candidate, config, manifest) != result['identity']:
        raise ValueError('Runtime changed while predicting')
    info = write_result(scores, output/'submission.zip')
    info.update(test_sha256=digest(test_file), detector_identity=result['identity'],
                official_score=None, uploaded=False)
    (output/'export.json').write_text(json.dumps(info, indent=2)+'\n')
    return info


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for field in ('candidate', 'dataset', 'evaluation', 'test-file', 'output', 'python'):
        p.add_argument('--'+field, required=True)
    p.add_argument('--chip', type=int, default=0)
    print(json.dumps(export(**vars(p.parse_args()))))
