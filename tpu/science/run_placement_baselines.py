"""CPU reproduction pilot; downloaded code is isolated from the final scorer.

Run in a Slurm allocation with four cores and 16 GiB. This script does not
allocate resources or launch Ray jobs. Native GPU leaderboard reproduction is
a separate experiment; these executions exercise upstream CPU fallbacks.
"""
import argparse
import contextlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from isolation import Limits, python_mounts, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='+', default=['archgen', 'abuplace'])
    p.add_argument('--cases', nargs='+', default=['ibm01', 'ibm18'])
    p.add_argument('--seconds', type=int, default=180)
    p.add_argument('--out', required=True)
    p.add_argument('--reuse-completed', action='store_true',
                   help='Regrade saved numeric outputs in this exact pilot directory')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    probe = root / '.science/challenge-probe'
    sys.path.insert(0, str(probe))
    import numpy as np
    import torch
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = {'archgen': ('submissions/final_placer.py', 'OptimalPlacer'),
               'abuplace': ('abuplace/placer.py', 'XplacePlacer')}
    report = {'host': platform.node(), 'cpus': 4, 'memory_gib': 16,
              'wall_limit_seconds': args.seconds,
              'profile': 'upstream_cpu_fallback_pilot',
              'challenge_commit': '996193c83eaad151e5ae3fb166cedbd83372d060',
              'evaluator_commit': '45a721d01dfe56fc4800b95da52e9b4193b76d04',
              'runs': []}
    for method in args.methods:
        checkout = root / '.science' / method
        source, cls = sources[method]
        commit = subprocess.check_output(
            ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        for name in args.cases:
            case_out = out / f'{method}-{name}'
            case_out.mkdir(exist_ok=True)
            row = {'method': method, 'case': name, 'commit': commit,
                   'valid': False, 'reward': 0.0}
            # Public time knobs reduce search budget. Xplace is unavailable on
            # CPU; disabling its optional Archgen lane is explicitly recorded.
            search = max(1, args.seconds - 40)
            env = {'PYTHONPATH': '/eval:/baseline', 'CUDA_VISIBLE_DEVICES': '',
                   'MPLCONFIGDIR': '/tmp/matplotlib', 'MPLBACKEND': 'Agg',
                   'XDG_CACHE_HOME': '/tmp/cache', 'XDG_CONFIG_HOME': '/tmp/config',
                   'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4',
                   'MKL_NUM_THREADS': '4', 'NUMEXPR_NUM_THREADS': '4'}
            if method == 'archgen':
                env.update(EVO_XRA_ENABLE='0', PAD80_CD_TIME=str(search),
                           EVO_TARGET_TOTAL_TIME_S=str(search),
                           AUTODMP_TOTAL_TIME_LIMIT=str(search),
                           EVO_UNDER60_DEADLINE_S=str(search),
                           EVO_UNDER60_SCORE_SAFETY_S='10')
            row['environment'] = env
            started = time.monotonic()
            try:
                if args.reuse_completed and (case_out/'child.json').exists() and (case_out/'positions.npy').exists():
                    row['reused_completed_output'] = True
                else:
                    row['isolated_seconds'] = run(
                        [sys.executable, '/runner.py', name, '/baseline/' + source, cls],
                        limits=Limits(args.seconds, 16, 4), log=case_out/'candidate.log',
                        readonly=python_mounts(sys.executable) + [
                            (probe, '/eval'),
                            (Path(__file__).with_name('placement_baseline_child.py'), '/runner.py'),
                        ] + ([(Path('/etc/alternatives/ld'), '/etc/alternatives/ld')]
                             if Path('/etc/alternatives/ld').exists() else []),
                        writable=[(checkout, '/baseline'), (case_out, '/output')],
                        env=env, cwd='/eval')
                row.update(json.loads((case_out/'child.json').read_text()))
                positions = torch.from_numpy(np.load(case_out/'positions.npy', allow_pickle=False))
                # Fresh canonical input/scorer, outside the baseline process.
                grade_start = time.monotonic()
                with (case_out/'grader.log').open('w') as log, contextlib.redirect_stdout(log):
                    benchmark, plc = load_benchmark_from_dir(str(
                        probe/'external/MacroPlacement/Testcases/ICCAD04'/name))
                    valid, errors = validate_placement(positions, benchmark)
                    row.update(valid=valid, violations=errors)
                    if valid:
                        metrics = compute_proxy_cost(positions, benchmark, plc)
                        if not all(np.isfinite(float(v)) for v in metrics.values()):
                            raise ValueError('nonfinite grader metric')
                        if metrics['overlap_count']:
                            row.update(valid=False, violations=['scorer reports hard overlaps'])
                        row['metrics'] = {k: float(v) for k, v in metrics.items()}
                        if row['valid']:
                            row['reward'] = max(1e-6, 1/(1+float(metrics['proxy_cost'])))
                row['grading_seconds'] = time.monotonic() - grade_start
            except Exception as exc:
                row.update(valid=False, reward=0.0, error=f'{type(exc).__name__}: {exc}')
            row['total_seconds'] = time.monotonic() - started
            report['runs'].append(row)
            (out/'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
