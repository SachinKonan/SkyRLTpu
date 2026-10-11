"""Track MOMO's FAIR candidates from the graders' archive (evaluator-side; includes the hidden-shift block).

  python factory/momo_watch.py [--sync] [--top 15]

The farm graders archive every full result to gs://sk7524-tinker-tpu-us-east5/fair-factory/results/<host>/.
This syncs them to fair-data/factory/momo_results/, dedups by candidate sha (+mode, stack), and prints:
  - counts (graded / ok / failure reasons),
  - the top candidates by the proposer-visible reward,
  - the ACCEPTED set: reward > 0.006, every visible fold gain > 0, hidden_ok true.
Accepted candidates are written to fair-data/factory/momo_results/accepted/<sha>.py for a full re-grade on della
(factory/grade.py --mode full, the torch grader) before anything goes into a submission.
"""
import argparse
import collections
import glob
import json
import os
import subprocess
from pathlib import Path

GCS = 'gs://sk7524-tinker-tpu-us-east5/fair-factory/results'
OUT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/factory/momo_results')
GCLOUD = '/scratch/gpfs/ZHUANGL/sk7524/google-cloud-sdk/bin/gcloud'
THRESH = 0.006


def sync():
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([GCLOUD, 'storage', 'rsync', '-r', GCS, str(OUT / 'raw')], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load():
    best = {}
    for p in glob.glob(str(OUT / 'raw' / '**' / '*.json'), recursive=True):
        try:
            r = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        key = (r['sha'], r['mode'], r['stack'])
        if key not in best or r['t'] > best[key]['t']:
            r['host'] = Path(p).parent.name
            best[key] = r
    return list(best.values())


def accepted(res):
    v = res.get('visible') or {}
    folds = v.get('per_fold_gain') or []
    return (res.get('ok') and (res.get('reward') or -1) > THRESH and folds and min(folds) > 0
            and (res.get('hidden') or {}).get('hidden_ok'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sync', action='store_true')
    ap.add_argument('--top', type=int, default=15)
    args = ap.parse_args()
    if args.sync:
        sync()
    rows = load()
    ok = [r for r in rows if r['result'].get('ok')]
    fails = collections.Counter(((r['result'].get('contract') or {}).get('fail') or r['result'].get('error') or '?')[:60]
                                for r in rows if not r['result'].get('ok'))
    print(f'graded {len(rows)} unique (sha, mode, stack); ok {len(ok)}')
    for reason, n in fails.most_common(8):
        print(f'  fail x{n}: {reason}')
    ok.sort(key=lambda r: -(r['result'].get('reward') or -1))
    print(f'\ntop {args.top} by visible reward:')
    for r in ok[:args.top]:
        res = r['result']; v = res.get('visible') or {}; h = res.get('hidden') or {}
        print(f"  {r['sha']} {r['mode']}/{r['stack']} reward {res['reward']:+.4f} folds {v.get('per_fold_gain')} "
              f"hidden_ok {h.get('hidden_ok')} all19 {h.get('all19_cand')} rt {(res.get('runtime') or {}).get('total_s')}s")
    acc = [r for r in ok if accepted(r['result'])]
    (OUT / 'accepted').mkdir(parents=True, exist_ok=True)
    for r in acc:
        (OUT / 'accepted' / f"{r['sha']}.py").write_text(r['source'])
        (OUT / 'accepted' / f"{r['sha']}.json").write_text(json.dumps(r['result'], indent=1))
    print(f'\naccepted {len(acc)} -> {OUT / "accepted"}')


if __name__ == '__main__':
    main()
