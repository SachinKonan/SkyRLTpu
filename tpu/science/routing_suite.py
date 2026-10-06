"""Explicit suite selection and rescoring of independently verified case counts."""
import hashlib
import json
import math
from pathlib import Path

from .rewards import qubit, valid


def validate_suite(name):
    if name not in ('full', 'q20'):
        raise ValueError('routing suite must be full or q20')
    return name


def manifest():
    return json.loads(Path(__file__).with_name('manifests').joinpath('routing-v1.json').read_text())


def select_suite(root, work, name, suite=None):
    validate_suite(name)
    root = Path(root).resolve()
    original = Path(suite or root / 'python/benchmarks/sabre_suite.json').resolve()
    if name == 'full':
        return original
    if suite is not None:
        raise ValueError('Q20 selection cannot override the pinned suite')
    pinned = manifest()
    if hashlib.sha256(original.read_bytes()).hexdigest() != pinned['suite_sha256']:
        raise ValueError('routing suite differs from pinned manifest')
    payload = json.loads(original.read_text())
    cases = [dict(c) for c in payload['cases'] if c['id'].endswith('_q20')]
    if len(cases) != 24 or {c['id'] for c in cases} != {c['id'] for c in pinned['cases'] if c['id'].endswith('_q20')}:
        raise ValueError('Q20 requires exactly 24 pinned cases')
    for case in cases:
        for key in ('qasm3_path', 'topology_path'):
            case[key] = str((original.parent / case[key]).resolve())
    payload['cases'] = cases
    output = Path(work) / 'q20-suite.json'
    output.write_text(json.dumps(payload, indent=2) + '\n')
    return output


def rescore_verified(row):
    """Only full-suite successes with complete trusted case diagnostics qualify."""
    if row['correctness'] != 1 or not row.get('code'):
        raise ValueError('only verified valid programs can be rescored')
    metrics = row['metrics']
    digest = hashlib.sha256(row['code'].encode()).hexdigest()
    if metrics.get('source_sha256') != digest:
        raise ValueError('source hash disagrees with verified program')
    specs = manifest()['cases']
    cases = metrics['cases']
    by_id = {c['case']: c for c in cases}
    if len(cases) != 72 or len(by_id) != 72 or set(by_id) != {c['id'] for c in specs}:
        raise ValueError('incomplete or duplicate full-suite verification')
    for case in cases:
        if (type(case['swaps']) is not int or case['swaps'] < 0
                or case['added_cnots'] != 3 * case['swaps']):
            raise ValueError('invalid verified SWAP count')
    def score(selected):
        return qubit([c['original_cnot_added'] for c in selected],
                     [by_id[c['id']]['added_cnots'] for c in selected],
                     [c['weight'] for c in selected])
    old, _ = score(specs)
    if not math.isclose(old, row['reward'], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('original reward disagrees with verified counts')
    selected = [c for c in specs if c['id'].endswith('_q20')]
    reward, result = score(selected)
    result.update(cases=[by_id[c['id']] for c in selected], case_count=24,
                  routing_suite='q20', source_sha256=digest,
                  provenance='Rescored trusted full-suite verification; no new execution')
    return valid(reward, result)
