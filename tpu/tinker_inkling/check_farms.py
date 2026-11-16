"""Print readiness and Erdős capacity of the 6 gradefarm hosts through the local forwards 27861-6."""
import json, urllib.request
for i in range(1, 7):
    try:
        req = urllib.request.Request(f'http://127.0.0.1:2786{i}/skyrl/v1/grading/capacity',
                                     headers={'X-Grading-Run': 'check', 'X-Grading-Token': 'check'})
        c = json.load(urllib.request.urlopen(req, timeout=8))
        m = c['families']['math']
        print(f'host {i}: ready {c["ready"]} math slots {m["total"]} cpus {m["cpus"]} running {m["running"]} queued {m["queued"]}')
    except Exception as e:
        print(f'host {i}: unreachable ({type(e).__name__}: {e})')
