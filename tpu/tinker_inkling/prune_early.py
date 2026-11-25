"""Prune each live run's early Tinker checkpoints (user 2026-11-25): delete the weights + sampler_weights saved for steps
below KEEP_FROM (5); keep step 5 onward, 'final', and ALWAYS the run's two newest numbered steps (resume safety).
Runs and their keys come from RUNS below; each run's Tinker training-run id is read from its checkpoints.jsonl.
Appends every deletion to tinker-inkling-runs/deleted_checkpoints_early.jsonl. usage: prune_early.py [--dry]"""
import json, os, re, subprocess, sys, time
R = '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs'
KEEP_FROM = 5
RUNS = {'mn9-s1': '2', 'mn9-s2': '2', 'mn9-s3': '3', 'mn9-s4': '1', 'mn9-s5': '1',
        'mc-s1': 'TAI', 'mc-s2': 'TAI', 'mc-s3': '3', 'mc-s4': 'HARISH', 'mc-s5': 'HARISH'}
dry = '--dry' in sys.argv


def key_env(k):
    var = 'TINKER_API_KEY' if k == '1' else f'TINKER_API_KEY_{k}'
    return subprocess.run(['bash', '-ic', f'printf %s "${var}"'], capture_output=True, text=True).stdout


def ledger(run):
    p = f'{R}/{run}/tinker_log/{run}/member_inkling/checkpoints.jsonl'
    rows = []
    if os.path.exists(p):
        for l in open(p):
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return rows


by_key = {}
for run, k in RUNS.items():
    rows = ledger(run)
    steps = sorted({int(r['name']) for r in rows if str(r.get('name', '')).isdigit()})
    if not steps:
        continue
    protect = set(steps[-2:])                                   # newest two numbered steps: resume needs them
    for r in rows:
        name = str(r.get('name', ''))
        if not name.isdigit() or int(name) >= KEEP_FROM or int(name) in protect:
            continue
        for kind in ('state_path', 'sampler_path'):
            if r.get(kind):
                by_key.setdefault(k, []).append((run, int(name), r[kind]))
import tinker
rec = open(f'{R}/deleted_checkpoints_early.jsonl', 'a')
for k, items in by_key.items():
    os.environ['TINKER_API_KEY'] = key_env(k)
    rc = tinker.ServiceClient().create_rest_client()
    existing = set()
    off = 0
    while True:                                                  # only delete what still exists
        page = rc.list_user_checkpoints(limit=100, offset=off).result().checkpoints
        existing |= {c.tinker_path for c in page}
        if len(page) < 100:
            break
        off += 100
    todo = [x for x in items if x[2] in existing]
    print(f'key {k}: {len(todo)} early checkpoints to delete ' + str(sorted({(r, s) for r, s, _ in todo})))
    for run, step, path in todo:
        if dry:
            continue
        try:
            rc.delete_checkpoint_from_tinker_path(path).result()
            rec.write(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), run=run, step=step,
                                      path=path, key=k)) + '\n'); rec.flush()
        except Exception as e:  # noqa: BLE001
            print('FAILED', path, str(e)[:120])
