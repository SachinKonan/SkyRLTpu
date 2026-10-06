"""Read-only combined view of preserved A100 and queued Ray task results."""
import json
from pathlib import Path
root = Path(__file__).resolve().parent
plan = json.loads((root/'plan.json').read_text())
summary = {}
for method in ['abuplace', 'archgen']:
    rows = {}
    for task in plan['jobs']:
        if task['method'] != method:
            continue
        name = method+'-'+task['case']
        pointer = root/'tasks'/name/'done.json'
        if not pointer.exists() and method in plan.get('ignore_legacy_methods', []):
            continue
        report = (Path(json.loads(pointer.read_text())['report']) if pointer.exists()
                  else Path(plan['legacy'])/name/'report.json')
        if report.exists():
            r = json.loads(report.read_text())
            rows[task['case']] = dict(valid=r.get('valid', False),
                proxy_cost=r.get('metrics', {}).get('proxy_cost'), report=str(report))
    valid = [r['proxy_cost'] for r in rows.values() if r['valid']]
    summary[method] = dict(completed=len(rows), valid=len(valid), required=len(plan['cases']),
        mean_proxy_cost=sum(valid)/len(valid) if len(valid)==len(plan['cases']) else None,
        cases=rows)
print(json.dumps(summary, indent=2))
