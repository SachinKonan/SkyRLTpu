"""Queue the fair evaluation: every method x design x repeat, drained by the PLI queue jobs.

Each task runs eval_one.py in the method's own repository and environment and re-scores the
result with the pinned official challenge repository. Every time-filling method gets the same
3,000 s budget per design (the challenge allows 1 hour; ArchGen's README asks for 3,300 s); MOMO's
budget covers its netlist reload, Xplace starts, legalization and search together.

Designs: the 17 IBM designs (Tier 1) and the 4 public NG45 designs (Tier 2; proxy cost only).
Repeat r of every method has priority (100 - r), so first repeats of all methods finish first.
Repeats rerun the same seeds (MOMO 42 + Xplace 0, ArchGen 42, AbuPlace 42): they measure runtime
nondeterminism, not seed variance.

Usage: make_eval_tasks.py [--queue DIR] [--runs DIR] [--methods m1,m2] [--repeats N] [--cases ...]
                          [--smoke [--budget S]]
"""
import argparse
import json
from pathlib import Path

F = Path('/scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval')
W = Path(__file__).resolve().parents[3]
CUDA_PY = str(F / 'env/cuda/bin/python')
OFFICIAL = str(F / 'src/macro-place-challenge-2026')
EVAL_ONE = str(W / 'tpu/science/fair_eval/eval_one.py')
MOMO_PLACER = str(W / 'tpu/science/fair_eval/submission/placer.py')
BUDGET = 3000
IBM = [f'ibm{i:02d}' for i in range(1, 19) if i != 5]
NG45 = ['ariane133', 'ariane136', 'mempool_tile', 'nvdla']

# est_seconds: an upper estimate of the wall time, used to start a task only when it can finish
# within the job. gpu_share: fraction of a GPU the method keeps busy (MOMO uses one only for its
# ~1 min Xplace step; ArchGen and AbuPlace use theirs throughout).
METHODS = {
    'momo300': dict(python=CUDA_PY, repo=OFFICIAL, placer=MOMO_PLACER,
                    env=dict(MOMO_XPLACE_ROOT=str(F / 'src/Xplace'), MOMO_SEARCH_SECONDS='300',
                             MOMO_TOTAL_SECONDS=str(BUDGET)),
                    est=900, timeout=3330, repeats=3, gpu_share=0.05),
    'momofill': dict(python=CUDA_PY, repo=OFFICIAL, placer=MOMO_PLACER,
                     env=dict(MOMO_XPLACE_ROOT=str(F / 'src/Xplace'), MOMO_SEARCH_SECONDS='fill',
                              MOMO_TOTAL_SECONDS=str(BUDGET)),
                     est=BUDGET + 200, timeout=3330, repeats=2, gpu_share=0.05),
    'archgen': dict(python=CUDA_PY, repo=str(F / 'src/archgen'), placer='submissions/final_placer.py',
                    env=dict(EVO_XRA_XPLACE_ROOT=str(F / 'src/Xplace-archgen'), XPLACE_PYTHON=CUDA_PY,
                             EVO_XRA_ENABLE='1', EVO_FORCE_FULL_TIME='1', EVO_TARGET_TOTAL_TIME_S=str(BUDGET),
                             AUTODMP_TOTAL_TIME_LIMIT=str(BUDGET), EVO_UNDER60_DEADLINE_S=str(BUDGET + 120),
                             EVO_UNDER60_SCORE_SAFETY_S='120', EVO_RETURN_GUARD_S='10', OMP_NUM_THREADS='1',
                             MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1'),
                    est=BUDGET + 200, timeout=3330, repeats=2, gpu_share=0.9),
    # AbuPlace's own wall guard (XP_RUDY_BO_BUDGET_S) defaults to 2,500 s.
    'abuplace': dict(python=str(F / 'src/AbuPlace/.venv/bin/python'), repo=str(F / 'src/AbuPlace'),
                     placer='abuplace/placer.py', env={}, est=2700, timeout=3330, repeats=2, gpu_share=0.9),
}

# Private copy per task: AbuPlace builds its C helpers with -march=native on first use and keeps a
# LEF/DEF cache in benchmarks/xplace_cache; Xplace writes logs/caches into its own tree.
PRIVATE = {
    'abuplace': ('rm -rf "$TASK_WORK/repo" && mkdir -p "$TASK_WORK/repo" && '
                 'rsync -a --exclude .venv --exclude external --exclude .git --exclude benchmarks/xplace_cache '
                 f'{F}/src/AbuPlace/ "$TASK_WORK/repo/" && '
                 f'ln -s {F}/src/AbuPlace/external "$TASK_WORK/repo/external" && '
                 'find "$TASK_WORK/repo/abuplace" -path "*extensions*" -name "*.so" -delete'),
    'archgen': ('rm -rf "$TASK_WORK/repo" "$TASK_WORK/xplace" && mkdir -p "$TASK_WORK/repo" "$TASK_WORK/xplace" && '
                f'rsync -a --exclude .git --exclude external {F}/src/archgen/ "$TASK_WORK/repo/" && '
                f'ln -s {F}/src/archgen/external "$TASK_WORK/repo/external" && '
                f'rsync -a --exclude .git --exclude "build*" {F}/src/Xplace-archgen/ "$TASK_WORK/xplace/" && '
                'export EVO_XRA_XPLACE_ROOT="$TASK_WORK/xplace"'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', default=str(F / 'pli-queue'))
    ap.add_argument('--runs', default=str(F / 'runs/pli-eval'), help='output root for task results')
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--repeats', type=int)
    ap.add_argument('--cases', nargs='*', default=IBM + NG45)
    ap.add_argument('--smoke', action='store_true', help='one repeat, tagged smoke, run first')
    ap.add_argument('--budget', type=int, help='override the time-filling budget (smoke tests only)')
    a = ap.parse_args()
    if a.budget and not a.smoke:
        ap.error('--budget only for smoke tests')
    q = Path(a.queue)
    (q / 'pending').mkdir(parents=True, exist_ok=True)
    added = 0
    for m in a.methods.split(','):
        cfg = METHODS[m]
        if a.budget:
            env = {k: (str(a.budget) if k in ('EVO_TARGET_TOTAL_TIME_S', 'AUTODMP_TOTAL_TIME_LIMIT', 'MOMO_TOTAL_SECONDS')
                       else str(a.budget + 120) if k == 'EVO_UNDER60_DEADLINE_S' else v) for k, v in cfg['env'].items()}
            cfg = dict(cfg, env=env, est=min(cfg['est'], a.budget + 400))
        reps = 1 if a.smoke else (a.repeats or cfg['repeats'])
        for r in range(1, reps + 1):
            for case in a.cases:
                tag = 'smoke' if a.smoke else f'r{r}'
                tid = f'eval-{m}-{case}-{tag}'
                if any((q / d / f'{tid}.json').exists() for d in ('pending', 'running', 'done', 'failed')):
                    continue
                out = Path(a.runs) / m / tag / case
                run = [cfg['python'], '-u', EVAL_ONE, '--placer', cfg['placer'], '--case', case,
                       '--out', str(out), '--official-python', CUDA_PY, '--official-repo', OFFICIAL]
                cmd, cwd = run, cfg['repo']
                if m in PRIVATE:
                    cmd, cwd = ['bash', '-c', PRIVATE[m] + ' && cd "$TASK_WORK/repo" && exec ' + ' '.join(map(repr, run))], None
                spec = dict(id=tid, kind='eval', method=m, case=case, repeat=tag, cmd=cmd, cwd=cwd, env=dict(cfg['env']),
                            out=str(out), est_seconds=cfg['est'], timeout_seconds=cfg['timeout'],
                            gpu_share=cfg['gpu_share'], cpus=12, memory_gb=64,
                            priority=200 if a.smoke else 100 - r)
                (q / 'pending' / f'{tid}.json').write_text(json.dumps(spec, indent=1))
                added += 1
    print(added, 'added;', len(list((q / 'pending').glob('*.json'))), 'pending')


if __name__ == '__main__':
    main()
