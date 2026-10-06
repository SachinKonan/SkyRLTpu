"""Publish three verified AbuPlace GP variants beside the frozen fallback."""
import argparse
import json
from pathlib import Path

from .abuplace_starts import ABUPLACE_COMMIT, VARIANTS
from .placement_warm_start import CASES, SUITE, digest, verified_inputs

SCHEMA = 'abuplace-xplace-three-starts-v1'
DESTINATION = 'tpu/science/results/placement/abuplace-xplace-three-starts-v1'
SCORE_COLUMNS = ('proxy_cost', 'wirelength_cost', 'density_cost', 'congestion_cost')


def publish(root, queue, destination):
    import numpy as np
    root, queue, destination = Path(root), Path(queue), Path(destination)
    starts = verified_inputs(root)
    # Stage every case in memory first; a partial suite is never published.
    staged, records = {}, {}
    for case in CASES:
        with np.load(starts[case], allow_pickle=False) as data:
            problem = {k: data[k].copy() for k in data.files}
        layouts, scores, provenance = [], [], []
        for variant in VARIANTS:
            done = json.loads((queue/'tasks'/f'xplace-{variant}-{case}'/'done.json').read_text())
            report_path = Path(done['report'])
            report = json.loads(report_path.read_text())
            gp = json.loads((report_path.parent/'gp.json').read_text())
            if (report.get('valid') is not True or report.get('case') != case
                    or report.get('variant') != variant or report.get('method') != 'xplace-abu'
                    or report.get('repository_commit') != ABUPLACE_COMMIT
                    or gp.get('source_commit') != ABUPLACE_COMMIT
                    or gp.get('variant') != variant or gp.get('stage') != 'global_placement_only'):
                raise ValueError(f'unverified GP variant: {case}/{variant}')
            path = report_path.parent/'positions.npy'
            positions = np.load(path, allow_pickle=False)
            if (positions.shape != problem['initial_positions'].shape
                    or not np.isfinite(positions).all()
                    or not np.array_equal(positions[problem['fixed']],
                                          problem['initial_positions'][problem['fixed']])):
                raise ValueError(f'invalid variant coordinates: {case}/{variant}')
            score = [float(report['metrics'][key]) for key in SCORE_COLUMNS]
            if not np.isfinite(score).all():
                raise ValueError(f'nonfinite variant score: {case}/{variant}')
            layouts.append(positions)
            scores.append(score)
            provenance.append(dict(variant=variant, report=str(report_path),
                report_sha256=digest(report_path), positions_sha256=digest(path), gp=gp,
                preprocessing_seconds=report['candidate_wall_seconds']))
        problem.update(starting_layouts=np.stack(layouts), starting_scores=np.asarray(scores),
                       starting_names=np.asarray(VARIANTS), starting_score_columns=np.asarray(SCORE_COLUMNS))
        staged[case] = problem
        native = root/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/case
        records[case] = dict(valid=True, variants=provenance,
            fallback_problem_sha256=digest(starts[case]),
            native_sha256={name:digest(native/name) for name in ('netlist.pb.txt', 'initial.plc')})
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'manifest.json').exists():
        raise FileExistsError('starting layouts are immutable; use a new version')
    for case, problem in staged.items():
        path = destination/f'{case}-problem.npz'
        np.savez_compressed(path, **problem)
        records[case]['problem_sha256'] = digest(path)
    manifest = dict(schema=SCHEMA, benchmark_suite=SUITE, cases=records,
                    variants=list(VARIANTS), candidate_budget_excludes_precomputation=True,
                    fallback='unchanged frozen Xplace initial_positions')
    temp = destination/'manifest.json.tmp'
    temp.write_text(json.dumps(manifest, indent=2)+'\n')
    temp.replace(destination/'manifest.json')
    verified_inputs(root, folder=destination)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', required=True)
    parser.add_argument('--output', default=DESTINATION)
    args = parser.parse_args()
    publish(Path(__file__).resolve().parents[2], args.queue, args.output)
