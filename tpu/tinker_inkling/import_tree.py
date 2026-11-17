"""Hand-off: start an Inkling run from another run's PUCT search tree (e.g. a Qwen seed's step-12 tree).

  import_tree.py RUN_NAME SNAPSHOT [--keep-visits]
    SNAPSHOT  gs://.../puct_sampler_step_000012.json or a local path

Writes the tree as RUNS/RUN/tinker_log/RUN/puct_sampler_step_000000.json, so a fresh Inkling run starts from it:
the client resumes the snapshot with the most states (rl/resume.py); with no checkpoint and resume step 0 it keeps that
tree (the step-0 reset only triggers for snapshots beyond step 0) and creates fresh LoRA weights. PUCT visit counts are
reset (puct_n = {}, puct_T = 0) unless --keep-visits: states, values, code, constructions and puct_m are kept, as in the
clone-base design. States keep their original timesteps (only lineage metadata / stats use them).
Refuses to overwrite an existing run dir. Records provenance in RUNS/RUN/imported_tree.json.
"""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path

RUNS = Path(os.environ.get('GRID_RUNS', '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs'))
GSUTIL = '/scratch/gpfs/ZHUANGL/sk7524/google-cloud-sdk/bin/gsutil'
REQUIRED = ('states', 'initial_states', 'puct_m', 'puct_n', 'puct_T')


def main():
    run, src = sys.argv[1], sys.argv[2]
    keep = '--keep-visits' in sys.argv
    rd = RUNS / run
    if rd.exists():
        sys.exit(f'refusing: {rd} already exists')
    if src.startswith('gs://'):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmp = f.name
        subprocess.run([GSUTIL, '-q', 'cp', src, tmp], check=True)
        tree = json.load(open(tmp))
        os.unlink(tmp)
    else:
        tree = json.load(open(src))
    missing = [k for k in REQUIRED if k not in tree]
    if missing:
        sys.exit(f'not a PUCT snapshot (missing {missing})')
    vals = [s['value'] for s in tree['states'] if s.get('value') is not None]
    best = max(vals) if vals else None
    src_step = tree.get('step')
    if not keep:
        tree['puct_n'], tree['puct_T'] = {}, 0
    tree['step'] = 0
    lg = rd / 'tinker_log' / run
    lg.mkdir(parents=True)
    (lg / 'puct_sampler_step_000000.json').write_text(json.dumps(tree))
    prov = dict(source=src, source_step=src_step, imported_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                n_states=len(tree['states']), best_c5=(-best if best is not None else None),
                visit_counts='kept' if keep else 'reset')
    (rd / 'imported_tree.json').write_text(json.dumps(prov, indent=1))
    print(json.dumps(prov))


if __name__ == '__main__':
    main()
