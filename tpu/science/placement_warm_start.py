"""Publish and verify mandatory Xplace starting layouts for new placement runs."""
import argparse
import hashlib
import json
from pathlib import Path

from .challenge_contract import CASES, SUITE, GRADING_LIMIT_SECONDS

DESTINATION = 'tpu/science/results/placement/xplace-start-ibm17-v2'
SCHEMA = 'xplace-start-ibm17-v2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_inputs(root, *, folder=None):
    folder = Path(root)/DESTINATION if folder is None else Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    if (manifest['schema'] != SCHEMA or manifest.get('benchmark_suite') != SUITE
            or set(manifest['cases']) != set(CASES)):
        raise ValueError('Incomplete Xplace starting-layout manifest')
    paths = {}
    for case in CASES:
        row = manifest['cases'][case]
        path = folder/f'{case}-problem.npz'
        if digest(path) != row['problem_sha256'] or row['valid'] is not True:
            raise ValueError(f'Invalid Xplace starting layout: {case}')
        native = Path(root)/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/case
        if set(row['native_sha256']) != {'netlist.pb.txt', 'initial.plc'}:
            raise ValueError(f'Incomplete native input hashes: {case}')
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
    manifest = {'schema': SCHEMA, 'benchmark_suite': SUITE, 'method': 'Archgen route-aware Xplace + deterministic CPU legalization',
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


def publish_full17(root, inputs):
    """Revalidate the audited 17 starts with the unchanged trusted scorer."""
    from concurrent.futures import ThreadPoolExecutor
    import os
    import subprocess
    import numpy as np
    root, inputs = Path(root).resolve(), Path(inputs).resolve()
    target = root/DESTINATION
    target.mkdir(parents=True, exist_ok=True)
    if (target/'manifest.json').exists():
        raise FileExistsError('Starting layouts are immutable; use a new version')
    python = root/'.science/venv-cuda/bin/python'
    def prepare(case):
        source = inputs/f'{case}.npz'
        provenance = json.loads((inputs/f'{case}.json').read_text())
        if digest(source) != provenance['problem_sha256']:
            raise ValueError(f'Full-suite input hash changed: {case}')
        with np.load(source, allow_pickle=False) as f:
            problem = {k:f[k].copy() for k in f.files}
        positions = target/f'{case}-positions.npy'
        np.save(positions, problem['initial_positions'], allow_pickle=False)
        score = target/f'{case}-score.json'
        with (target/f'{case}-verification.log').open('wb') as log:
            subprocess.run([str(python), str(root/'tpu/science/challenge_score_child.py'),
                '--root',str(root),'--case',case,'--positions',str(positions),'--result',str(score)],
                check=True,timeout=GRADING_LIMIT_SECONDS+30,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
        metrics = json.loads(score.read_text())
        assert metrics['overlap_count'] == 0
        path = target/f'{case}-problem.npz'
        path.write_bytes(source.read_bytes())
        native = root/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/case
        return case, dict(valid=True,problem_sha256=digest(path),positions_sha256=digest(positions),
            native_sha256={n:digest(native/n) for n in ['netlist.pb.txt','initial.plc']},
            metrics=metrics, source_provenance=provenance)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = dict(pool.map(prepare, CASES))
    mean = sum(row['metrics']['proxy_cost'] for row in rows.values())/len(CASES)
    manifest = dict(schema=SCHEMA,benchmark_suite=SUITE,case_count=len(CASES),cases=rows,
        method='Pinned Xplace starts from full17 audit; includes documented ibm14 numerical recovery',
        candidate_budget_excludes_precomputation=True,mean_proxy_cost=mean,reward=1/(1+mean),
        leaderboard_verified=False,grader_sha256=digest(root/'tpu/science/challenge_score_child.py'))
    temporary=target/'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest,indent=2)+'\n');temporary.replace(target/'manifest.json')
    verified_inputs(root)
    print(json.dumps(dict(destination=str(target),case_count=len(rows),mean_proxy_cost=mean)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source')
    parser.add_argument('--full17-inputs')
    args = parser.parse_args()
    if bool(args.source) == bool(args.full17_inputs):
        parser.error('choose exactly one source')
    if args.full17_inputs:
        publish_full17(Path(__file__).resolve().parents[2], args.full17_inputs)
    else:
        publish(Path(__file__).resolve().parents[2], args.source)
