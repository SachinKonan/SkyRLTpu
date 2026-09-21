"""Submit the six explicitly authorized v5p farms, preserving launch receipts.

Run with the existing operations environment. An uncertain submission is never
retried automatically; reconcile it with the SkyPilot queue first.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from google.cloud import storage
import sky

HERE = Path(__file__).resolve().parent
SKY = '/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/sky'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    lock = (HERE / 'submit.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepared = json.loads((HERE / 'prepared.json').read_text())
    assert len(prepared) == 6 and len({r['run_id'] for r in prepared}) == 6
    check = json.loads((HERE / 'preflight.json').read_text())
    assert check['adc_identity_match'] and check['storage_read']
    client = storage.Client(project='vision-mix')
    assert client._credentials.service_account_email == '289186856710-compute@developer.gserviceaccount.com'
    queue, *_ = sky.get(sky.jobs.queue_v2(refresh=True, skip_finished=True,
                       fields=['job_id', 'job_name', 'status']))
    active = {r.job_name: r.job_id for r in queue}
    path = HERE / 'launches.json'
    receipts = json.loads(path.read_text()) if path.exists() else []
    for row in prepared:
        previous = [r for r in receipts if r['run_id'] == row['run_id']]
        if previous:
            assert previous[-1]['state'] == 'submitted', 'Reconcile uncertain submission first'
            continue
        assert row['run_id'] not in active, 'Matching active job already exists'
        archive = Path(row['archive'])
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == row['archive_sha256']
        assert hashlib.sha256(Path(row['task']).read_bytes()).hexdigest() == row['task_sha256']
        bucket, name = row['code_uri'][5:].split('/', 1)
        blob = client.bucket(bucket).blob(name)
        if not blob.exists():
            blob.upload_from_filename(archive, if_generation_match=0, timeout=180)
        blob.reload()
        assert hashlib.sha256(blob.download_as_bytes(if_generation_match=blob.generation,
                             timeout=180)).hexdigest() == row['archive_sha256']
        receipt = dict(run_id=row['run_id'], pool=row['pool'], state='submitting',
                       time=time.time(), code_uri=row['code_uri'], generation=blob.generation,
                       archive_sha256=row['archive_sha256'])
        receipts.append(receipt)
        save(path, receipts)
        result = subprocess.run([SKY, 'jobs', 'launch', row['task'], '--pool', row['pool'],
                                 '--yes', '--detach-run'], capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        (HERE / (row['run_id'] + '-submission.txt')).write_text(output)
        assert result.returncode == 0, 'Submission failed; inspect saved output and reconcile queue'
        ids = re.findall(r'(?i)job id[:\s]+(\d+)', re.sub(r'\x1b\[[0-9;]*m', '', output))
        assert ids, 'Submission outcome uncertain; reconcile queue'
        receipt.update(job_id=int(ids[-1]), state='submitted')
        save(path, receipts)
        print('SUBMITTED', receipt['job_id'], row['run_id'], flush=True)


if __name__ == '__main__':
    main()
