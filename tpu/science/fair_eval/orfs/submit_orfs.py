"""Submit the Tier-2 OpenROAD runs: each method's repeat-1 NG45 placement from the fair evaluation,
plus ORFS's own rtl_macro_placer (no placement given) as a common reference. One Slurm job per run;
runs that already have a result directory or a queued job are skipped.

Usage: submit_orfs.py [--methods m1,m2] [--designs d1,d2] [--dry]
"""
import argparse
import json
import subprocess
from pathlib import Path

F = Path('/scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval')
SBATCH = Path(__file__).resolve().parent / 'orfs_job.sbatch'
NG45 = ['ariane133', 'ariane136', 'mempool_tile', 'nvdla']
METHODS = ['momofill', 'momo300', 'archgen', 'abuplace', 'rtlmp']


def positions(method, design):
    spec = json.loads((F / 'pli-queue/done' / f'eval-{method}-{design}-r1.json').read_text())
    att = Path(spec['out']) / spec['attempts'][-1]['attempt']
    rep = json.loads((att / 'report.json').read_text())
    assert rep['valid_official'], f'{method} {design}: not a valid placement'
    return att / 'positions.npy'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--designs', default=','.join(NG45))
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    queued = subprocess.run(['squeue', '-h', '-u', 'sk7524', '-o', '%j'], capture_output=True, text=True).stdout.split()
    for m in a.methods.split(','):
        for d in a.designs.split(','):
            name = 'rtlmp' if m == 'rtlmp' else f'{m}-r1'
            job = f'orfs-{name}-{d}'
            if job in queued or (F / 'orfs/runs' / name / d).exists():
                print('skip', job)
                continue
            pos = 'none' if m == 'rtlmp' else str(positions(m, d))
            cmd = ['sbatch', f'--export=ALL,NAME={name},DESIGN={d},POS={pos}', '-J', job, str(SBATCH)]
            print(' '.join(cmd))
            if not a.dry:
                subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
