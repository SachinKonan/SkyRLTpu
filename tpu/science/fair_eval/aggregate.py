"""Aggregate the fair evaluation: per-task results, coverage, validity, fallbacks, leaderboard check.

For every task in the queue, the result is taken from its final attempt only:
  done      -> <out>/<attempt>/report.json (official rescoring of what the placer returned)
  failed    -> counted as a failure (timeout, memory cap, crash, repeated interruption)
  pending/running -> missing
Each attempt's log (<queue>/work/<id>/<attempt>/stdout.log) is scanned for known signs of a
degraded run (a baseline or MOMO silently falling back), which are reported, never hidden.

Means are reported only where every method has a valid result for every design and repeat
(coverage is printed first). Two validity criteria are reported (see eval_one.py); the primary one
is valid_official, the criterion the leaderboard ranked by. Repeats reuse the same seeds.

Usage: aggregate.py [--queue DIR] [--csv OUT.csv] [--json OUT.json]
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import statistics as st

F = Path('/scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval')
IBM = [f'ibm{i:02d}' for i in range(1, 19) if i != 5]
NG45 = ['ariane133', 'ariane136', 'mempool_tile', 'nvdla']
LEADERBOARD_IBM_MEAN = {'archgen': 0.9507, 'abuplace': 0.9522}   # judge-verified, 17 IBM designs

# Log lines that mean a run did not take its intended path.
DEGRADED = {
    'archgen': [r'PLC unavailable', r'helper import failed', r'CUDA unavailable', r'no legal seed candidates',
                r'xra_fallback', r'using fallback', r'\bfallback_\w+'],
    'abuplace': [r'emergency legal', r'off\(fallback\)', r"trajectory '[^']+' FAILED", r'\[BO2\] skip',
                 r'CUDA not available', r'config failed'],
    'momo300': [], 'momofill': [],
    '*': [r'Traceback', r'undefined symbol', r'No module named', r'Killed', r'out of memory'],
}


def scan(log, method):
    if not log.exists():
        return {}
    text = log.read_text(errors='replace')
    hits = {}
    for pat in DEGRADED.get(method, []) + DEGRADED['*']:
        n = len(re.findall(pat, text))
        if n:
            hits[pat] = n
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', default=str(F / 'pli-queue'))
    ap.add_argument('--csv')
    ap.add_argument('--json')
    a = ap.parse_args()
    q = Path(a.queue)
    rows = []
    for state in ('done', 'failed', 'pending', 'running'):
        for f in sorted((q / state).glob('*.json')):
            spec = json.loads(f.read_text())
            if spec.get('kind') != 'eval' or spec.get('repeat') == 'smoke':
                continue
            last = (spec.get('attempts') or [{}])[-1]
            row = dict(id=spec['id'], method=spec['method'], case=spec['case'], repeat=spec['repeat'], state=state,
                       attempt=last.get('attempt'), reason=last.get('reason'), host=last.get('host'),
                       peak_rss_gb=last.get('peak_rss_gb'), cotenants=last.get('gpu_cotenants_at_start'),
                       n_attempts=len(spec.get('attempts', [])))
            if state == 'done' and last.get('attempt'):
                rep = Path(spec['out']) / last['attempt'] / 'report.json'
                if rep.exists():
                    r = json.loads(rep.read_text())
                    row.update(proxy=r['metrics']['proxy_cost'], wirelength=r['metrics']['wirelength_cost'],
                               density=r['metrics']['density_cost'], congestion=r['metrics']['congestion_cost'],
                               overlaps=r['metrics']['overlap_count'], valid_official=r['valid_official'],
                               valid_strict=r['valid_strict'], max_bound_excess=r['max_bound_excess'],
                               placer_seconds=r['placer_seconds'], within_1h=r['within_1h'],
                               dtype=r.get('returned_dtype'))
                    momo_log = Path(spec['out']) / last['attempt'] / 'momo_log.json'
                    if spec['method'].startswith('momo') and momo_log.exists():
                        ml = json.loads(momo_log.read_text())
                        row['momo_fallback'] = ml.get('fallback')
                        row['momo_best_start'] = ml.get('best_start')
                else:
                    row['state'] = 'done-no-report'
            row['degraded'] = scan(q / 'work' / spec['id'] / str(last.get('attempt')) / 'stdout.log', spec['method'])
            rows.append(row)

    methods = sorted({r['method'] for r in rows})
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['method'], r['case'])].append(r)

    print('== coverage (valid_official results / tasks) and problems')
    for m in methods:
        mr = [r for r in rows if r['method'] == m]
        ok = [r for r in mr if r.get('valid_official')]
        print(f"{m:10s} tasks {len(mr):3d}  valid {len(ok):3d}  failed {sum(r['state']=='failed' for r in mr):3d}  "
              f"missing {sum(r['state'] in ('pending','running','done-no-report') for r in mr):3d}  "
              f"invalid {sum(r['state']=='done' and not r.get('valid_official') for r in mr):3d}  "
              f"strict-invalid {sum(r['state']=='done' and r.get('valid_official') and not r.get('valid_strict') for r in mr):3d}  "
              f"over-1h {sum(r.get('within_1h') is False for r in mr):3d}  "
              f"degraded-log {sum(bool(r['degraded']) for r in mr):3d}  momo-fallback {sum(bool(r.get('momo_fallback')) for r in mr):3d}")
    for r in rows:
        if r['state'] != 'done' or r['degraded'] or r.get('momo_fallback') or not r.get('valid_official', True):
            print(f"   ! {r['id']}: state={r['state']} reason={r.get('reason')} valid={r.get('valid_official')} "
                  f"degraded={r['degraded']} momo_fallback={r.get('momo_fallback')}")

    summary = {}
    for group, cases in (('ibm', IBM), ('ng45', NG45)):
        complete = all(by.get((m, c)) and all(r.get('valid_official') for r in by[(m, c)]) for m in methods for c in cases)
        print(f"\n== {group.upper()} ({len(cases)} designs): mean proxy cost per method (mean over repeats, then designs)"
              + ('' if complete else '  [INCOMPLETE: means over available designs only, not comparable]'))
        per = {}
        for m in methods:
            vals = {c: st.mean(r['proxy'] for r in by[(m, c)] if r.get('valid_official'))
                    for c in cases if any(r.get('valid_official') for r in by.get((m, c), []))}
            per[m] = vals
            if vals:
                spread = [max(r['proxy'] for r in by[(m, c)] if r.get('valid_official')) -
                          min(r['proxy'] for r in by[(m, c)] if r.get('valid_official')) for c in vals]
                line = f"  {m:10s} {st.mean(vals.values()):.4f} over {len(vals)} designs; max repeat spread {max(spread):.4f}"
                if group == 'ibm' and m in LEADERBOARD_IBM_MEAN and len(vals) == len(cases):
                    gap = st.mean(vals.values()) / LEADERBOARD_IBM_MEAN[m] - 1
                    line += f"; leaderboard {LEADERBOARD_IBM_MEAN[m]:.4f} ({gap:+.2%}{'  CHECK' if abs(gap) > 0.005 else ''})"
                print(line)
        summary[group] = dict(complete=complete, per_design=per)
        base = [m for m in ('archgen', 'abuplace') if m in per]
        for m in [m for m in methods if m.startswith('momo')]:
            common = [c for c in cases if c in per[m] and all(c in per[b] for b in base)]
            if base and common:
                d = [per[m][c] - min(per[b][c] for b in base) for c in common]
                print(f"  {m} vs best baseline per design: wins {sum(x < 0 for x in d)}/{len(d)}, mean diff {st.mean(d):+.4f}")

    if a.csv:
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in rows)
    if a.json:
        Path(a.json).write_text(json.dumps(dict(rows=rows, summary=summary), indent=1))


if __name__ == '__main__':
    main()
