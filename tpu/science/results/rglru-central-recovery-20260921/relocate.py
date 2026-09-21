"""Relocate the displaced Gemma RG-LRU job with its repaired immutable bundle."""
import fcntl
import hashlib
import importlib.util
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('ops', HERE.parent / 'reallocation-10step-20260921/operations.py')
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)


def save(name, value):
    path = HERE / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    lock = (HERE / 'relocate.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (HERE / 'submission.json').exists(), 'existing intent; reconcile before retrying'
    row = json.loads((HERE / 'prepared.json').read_text())
    assert row['old_job_id'] == 1343 and row['priority'] == 110
    assert o.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)']).strip() == '289186856710-compute@developer.gserviceaccount.com'
    pool = o.sky('jobs', 'pool', 'status', '-a')
    idle = [l for l in pool.splitlines() if l.startswith('tpuswarm-v6e32-central1b')
            and 'READY' in l and l.split()[-1] == '-']
    assert idle, 'no idle central capacity; leave recovery untouched'
    save('central-idle-before.json', dict(time=time.time(), rows=idle))
    queue = o.queue()
    job = next(r for r in queue if r['job_id'] == 1343)
    assert job['status'] in {'PENDING', 'RECOVERING'} and job['cluster'] == row['old_cluster'], 'recovery changed; leave it untouched'
    assert job['run_id'] == row['run_id']
    assert hashlib.sha256((ROOT / row['task']).read_bytes()).hexdigest() == row['task_sha256']
    save('before-cancel.json', job)
    receipt = dict(time=time.time(), run_id=row['run_id'], old_job_id=1343,
                   pool=row['pool'], state='cancelling', checkpoint=0)
    save('submission.json', receipt)
    (HERE / 'cancel.txt').write_text(o.sky('jobs', 'cancel', '1343', '--yes'))
    deadline = time.monotonic() + 180
    while True:
        queue = o.queue()
        old = next(r for r in queue if r['job_id'] == 1343)
        if old['status'] == 'CANCELLED':
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('cancellation unconfirmed; no replacement submitted')
        time.sleep(3)
    save('cancelled.json', old)
    assert not any(r['run_id'] == row['run_id'] and r['status'] in
                   {'PENDING', 'RECOVERING', 'RUNNING', 'STARTING', 'SUBMITTED', 'CANCELLING'}
                   for r in queue), 'another active writer exists'
    receipt['state'] = 'submitting'
    save('submission.json', receipt)
    result = o.sky('jobs', 'launch', str(ROOT / row['task']), '--pool', row['pool'], '--yes', '--detach-run')
    (HERE / 'submission.txt').write_text(result)
    ids = re.findall(r'(?:Job ID|Job id|job ID|job id)[:\s]+(\d+)', result)
    if not ids:
        raise RuntimeError('uncertain launch; reconcile before retrying')
    receipt.update(state='submitted', job_id=int(ids[-1]))
    save('submission.json', receipt)
    queue = o.queue()
    job = next(r for r in queue if r['job_id'] == receipt['job_id'])
    assert job['run_id'] == row['run_id'] and job['pool'] == row['pool']
    save('queue-after.json', [r for r in queue if r['run_id'] == row['run_id']])
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
