"""Audit Hyra's exported circuits with the cached SimpleTES routing verifier."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time


def verify_case(root, row):
    start = time.monotonic()
    path = Path(root) / 'evaluator.py'
    spec = importlib.util.spec_from_file_location('hyra_audit_verifier', path)
    verifier = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = verifier
    spec.loader.exec_module(verifier)
    specs = verifier._load_suite_metadata(Path(root) / 'python/benchmarks/sabre_suite.json')
    def timeout(*_):
        raise TimeoutError('case verification exceeded 240 seconds')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(240)
    try:
        from qiskit import qasm3
        error = verifier._validate_case_routing(row, specs[row['id']])
        circuit = qasm3.loads(row['output_circuit'])
        swaps = int(circuit.count_ops().get('swap', 0))
        original = qasm3.loads(specs[row['id']].qasm3_path.read_text())
        assert not original.count_ops().get('swap', 0)
        return {'id': row['id'], 'valid': error is None, 'error': error,
                'swaps': swaps, 'seconds': time.monotonic() - start}
    except Exception as exc:
        return {'id': row['id'], 'valid': False, 'error': repr(exc),
                'seconds': time.monotonic() - start}
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--solution', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    suite_path = Path(args.root) / 'python/benchmarks/sabre_suite.json'
    suite = json.loads(suite_path.read_text())
    manifest = json.loads(args.manifest.read_text())
    assert hashlib.sha256(suite_path.read_bytes()).hexdigest() == manifest['suite_sha256']
    assert suite['cases'] == manifest['cases']
    cases = json.loads(args.solution.read_text())['cases']
    ids = [r['id'] for r in cases]
    assert len(ids) == len(set(ids))
    known = {x['id']: x for x in suite['cases']}
    assert set(ids) <= set(known)
    missing = [dict(id=k, baseline_swaps=v['original_cnot_added'] // 3)
               for k, v in known.items() if k not in ids]
    assert all(known[x['id']]['original_cnot_added'] % 3 == 0 for x in missing)
    verified = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(verify_case, args.root, row) for row in cases]
        with (args.output / 'verification-progress.jsonl').open('w') as log:
            for future in as_completed(futures):
                row = future.result()
                verified.append(row)
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(json.dumps({'completed': len(verified), **row}), flush=True)
    totals = {}
    for topology in ('q20', 'heron_fez', 'willow'):
        exported = [r for r in verified if r['id'].endswith('_' + topology)]
        omitted = [r for r in missing if r['id'].endswith('_' + topology)]
        swaps = sum(x.get('swaps', 0) for x in exported)
        fallback = sum(x['baseline_swaps'] for x in omitted)
        totals[topology] = {'exported_cases': len(exported), 'missing_cases': len(omitted),
                            'exported_swaps': swaps, 'missing_baseline_swaps': fallback,
                            'baseline_completed_swaps': swaps + fallback}
    result = {'suite_sha256': manifest['suite_sha256'],
              'solution_sha256': hashlib.sha256(args.solution.read_bytes()).hexdigest(),
              'verifier_sha256': hashlib.sha256((Path(args.root) / 'evaluator.py').read_bytes()).hexdigest(),
              'validated_exported_cases': sum(x['valid'] for x in verified),
              'exported_cases': len(verified), 'missing': missing,
              'cases': sorted(verified, key=lambda x: x['id']), 'topologies': totals,
              'note': 'Missing cases use cached SABRE counts for accounting only; Hyra does not export their circuits.'}
    (args.output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}), flush=True)
    if not all(x['valid'] for x in verified):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
