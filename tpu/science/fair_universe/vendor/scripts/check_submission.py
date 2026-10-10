"""Validate the Phase-2 prediction ZIP contract before uploading to Codabench."""
import argparse
import json
import math
import zipfile


def check(path, expected_count=None):
    with zipfile.ZipFile(path, 'r') as zf:
        if zf.namelist() != ['result.json']:
            raise ValueError('ZIP must contain ONLY root-level result.json')
        data = json.loads(zf.read('result.json'))
    if not isinstance(data, dict) or list(data) != ['ood_scores']:
        raise ValueError('Expected object with only ood_scores key')
    scores = data['ood_scores']
    if not isinstance(scores, list) or not scores:
        raise ValueError('Scores must be a nonempty list')
    if expected_count is not None and len(scores) != expected_count:
        raise ValueError(f'Expected {expected_count} scores, got {len(scores)}')
    if any(not isinstance(s, (int, float)) or not math.isfinite(s) for s in scores):
        raise ValueError('Scores must be finite numbers')
    return {'valid': True, 'count': len(scores), 'min': min(scores), 'max': max(scores)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zip', required=True)
    p.add_argument('--expected-count', type=int)
    a = p.parse_args()
    print(json.dumps(check(a.zip, a.expected_count), indent=2))


if __name__ == '__main__':
    main()
