"""Resume the one prepared Qwen AC2 job on available central capacity."""
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
    if (HERE / 'submissions.json').exists():
        raise RuntimeError('existing migration intent; reconcile before any retry')
    assert o.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)']).strip() == '289186856710-compute@developer.gserviceaccount.com'
    rows = json.loads((HERE / 'prepared.json').read_text())['jobs']
    assert {r['old_job_id'] for r in rows} == {1334} and len(rows) == 1
    # Availability is advisory; the pool scheduler remains the placement authority.
    pool = o.sky('jobs', 'pool', 'status', '-a')
    idle = [line for line in pool.splitlines() if line.startswith('tpuswarm-v6e32-central1b')
            and 'READY' in line and line.split()[-1] == '-']
    assert len(idle) >= 1, 'central capacity changed; leave original recoveries untouched'
    save('central-idle-before.json', dict(time=time.time(), rows=idle))
    proof = json.loads((HERE / 'state-verification.json').read_text())
    assert proof['database_quick_check'] == 'ok' and proof['epochs'] == 10
    assert proof['checkpoint']['batch'] == rows[0]['checkpoint']['batch']
    receipts = []
    for row in rows:
        queue = o.queue()
        job = next(r for r in queue if r['job_id'] == row['old_job_id'])
        if job['status'] not in {'PENDING', 'RECOVERING'} or job['cluster'] != row['old_cluster']:
            print('Original recovery changed; leaving it alone:', job, flush=True)
            continue
        assert job['run_id'] == row['run_id']
        assert hashlib.sha256((ROOT / row['task']).read_bytes()).hexdigest() == row['task_sha256']
        receipt = dict(run_id=row['run_id'], old_job_id=row['old_job_id'],
                       old_cluster=row['old_cluster'], pool=row['pool'],
                       checkpoint=row['checkpoint']['batch'], state='cancelling', time=time.time())
        receipts.append(receipt)
        save('submissions.json', receipts)
        save(row['model'] + '-before-cancel.json', job)
        result = o.sky('jobs', 'cancel', str(row['old_job_id']), '--yes')
        (HERE / (row['model'] + '-cancel.txt')).write_text(result)
        deadline = time.monotonic() + 180
        while True:
            queue = o.queue()
            old = next(r for r in queue if r['job_id'] == row['old_job_id'])
            if old['status'] == 'CANCELLED':
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('cancellation unconfirmed; no replacement submitted')
            time.sleep(3)
        save(row['model'] + '-cancelled.json', old)
        assert not any(r['run_id'] == row['run_id'] and r['status'] in
                       {'PENDING', 'RECOVERING', 'RUNNING', 'STARTING', 'SUBMITTED', 'CANCELLING'}
                       for r in queue), 'another active writer exists'
        receipt['state'] = 'submitting'
        save('submissions.json', receipts)
        result = o.sky('jobs', 'launch', str(ROOT / row['task']), '--pool', row['pool'], '--yes', '--detach-run')
        (HERE / (row['model'] + '-submission.txt')).write_text(result)
        ids = re.findall(r'(?:Job ID|Job id|job ID|job id)[:\s]+(\d+)', result)
        if not ids:
            raise RuntimeError('uncertain launch; reconcile before retrying')
        receipt.update(state='submitted', job_id=int(ids[-1]))
        save('submissions.json', receipts)
        print('Submitted', receipt, flush=True)
    queue = o.queue()
    for receipt in receipts:
        job = next(r for r in queue if r['job_id'] == receipt['job_id'])
        assert job['run_id'] == receipt['run_id'] and job['pool'] == receipt['pool']
    save('queue-after.json', [r for r in queue if r['run_id'] in {x['run_id'] for x in rows}])


if __name__ == '__main__':
    main()
