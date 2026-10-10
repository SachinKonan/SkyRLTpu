"""Markdown tables from fit_v3 report.json files: normal tail + per-shift scores."""
import json, sys
from pathlib import Path
rows = []
for spec in sys.argv[1:]:
    path, keys = spec.split('=')
    r = json.loads(Path(path).read_text())
    for k in keys.split(';'):
        rows.append((k, r[k]))
shifts = list(rows[0][1]['shifts'])
print('| model | q99 | q99.5 | q99.9 | shift_mean | ' + ' | '.join(shifts) + ' |')
print('|' + '---|' * (5 + len(shifts)))
for k, v in rows:
    q = v['normal_quantiles']
    print(f'| {k} | {q[3]:.2f} | {q[4]:.2f} | {q[5]:.2f} | {v["shift_mean"]:.4f} | ' + ' | '.join(f'{v["shifts"][s]:.3f}' for s in shifts) + ' |')
