"""Publish and verify mandatory Xplace starting layouts for new placement runs."""
import argparse
import hashlib
import json
from pathlib import Path

from .challenge_contract import CASES

DESTINATION = 'tpu/science/results/placement/xplace-start-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_inputs(root):
    folder = Path(root)/DESTINATION
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['schema'] != 'xplace-start-v1' or set(manifest['cases']) != set(CASES):
        raise ValueError('Incomplete Xplace starting-layout manifest')
    paths = {}
    for case in CASES:
        row = manifest['cases'][case]
        path = folder/f'{case}-problem.npz'
        if digest(path) != row['problem_sha256'] or row['valid'] is not True:
            raise ValueError(f'Invalid Xplace starting layout: {case}')
        native = Path(root)/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/case
        for name, expected in row['native_sha256'].items():
            if digest(native/name) != expected:
                raise ValueError(f'Xplace starting layout has stale native input: {case}/{name}')
        paths[case] = path
    return paths


def publish(root, source):
    import numpy as np
    root, source = Path(root), Path(source)
    target = root/DESTINATION
    target.mkdir(parents=True, exist_ok=True)
    if (target/'manifest.json').exists():
        raise FileExistsError('Starting layouts are immutable; use a new version')
    manifest = {'schema': 'xplace-start-v1', 'method': 'Archgen route-aware Xplace + deterministic CPU legalization',
                'candidate_budget_excludes_precomputation': True, 'cases': {}}
    staged = {}
    for case in CASES:
        run = source/f'xplace-{case}'
        report = json.loads((run/'report.json').read_text())
        if not report['valid'] or report['method'] != 'xplace' or report['case'] != case:
            raise ValueError(f'Xplace has not passed grading: {case}')
        original = root/'tpu/science/results/placement/scaffold-pilot'/f'{case}-problem.npz'
        with np.load(original, allow_pickle=False) as data:
            problem = {k: data[k].copy() for k in data.files}
        positions = np.load(run/'positions.npy', allow_pickle=False)
        if positions.shape != problem['initial_positions'].shape or not np.isfinite(positions).all():
            raise ValueError(f'Invalid numeric starting layout: {case}')
        if not np.array_equal(positions[problem['fixed']], problem['initial_positions'][problem['fixed']]):
            raise ValueError(f'Fixed objects changed: {case}')
        problem['initial_positions'] = positions.copy()
        staged[case] = problem
        native = root/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/case
        manifest['cases'][case] = dict(
            valid=True, positions_sha256=digest(run/'positions.npy'), source_problem_sha256=digest(original),
            raw_positions_sha256=digest(run/'raw_positions.npy'), report_sha256=digest(run/'report.json'),
            native_sha256={name:digest(native/name) for name in ['netlist.pb.txt','initial.plc']},
            metrics=report['metrics'], precomputation_seconds=report['candidate_wall_seconds'],
            gpu=report['candidate']['gpu'], xplace_commit=report['xplace_commit'],
            adapter_commit=report['repository_commit'])
    for case, problem in staged.items():
        path = target/f'{case}-problem.npz'
        np.savez_compressed(path, **problem)
        manifest['cases'][case]['problem_sha256'] = digest(path)
    costs = [v['metrics']['proxy_cost'] for v in manifest['cases'].values()]
    manifest['mean_proxy_cost'] = sum(costs)/len(costs)
    manifest['reward'] = 1/(1+manifest['mean_proxy_cost'])
    temporary = target/'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(target/'manifest.json')
    verified_inputs(root)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    publish(Path(__file__).resolve().parents[2], args.source)
