"""Submit one hardware variant in priority order, with durable attempt receipts.

An uncertain submission is never automatically retried. Reconcile its run_id
against SkyPilot before changing the attempt receipt.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import google.auth
from google.auth.transport.requests import Request
from google.cloud import storage

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ACCOUNT = '289186856710-compute@developer.gserviceaccount.com'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as out:
        json.dump(value, out, indent=2)
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', choices=('v4', 'v5p'), required=True)
    parser.add_argument('--stage', choices=('math', 'science', 'rglru', 'all'), default='all')
    parser.add_argument('--gcloud', default='/scratch/gpfs/ZHUANGL/sk7524/google-cloud-sdk/bin/gcloud')
    parser.add_argument('--sky', default=str(ROOT.parent / 'SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/sky'))
    args = parser.parse_args()
    active = subprocess.check_output([args.gcloud, 'auth', 'list', '--filter=status:ACTIVE',
                                      '--format=value(account)'], text=True).strip()
    creds, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    creds.refresh(Request())
    if active != ACCOUNT or getattr(creds, 'service_account_email', '') != ACCOUNT:
        raise RuntimeError('gcloud and ADC must both use the approved compute service account')
    storage_client = storage.Client(project='vision-mix', credentials=creds)
    rows = json.loads((HERE / 'jobs.json').read_text())['jobs']
    for row in rows:
        if row['hardware'] != args.hardware:
            continue
        stage = ('math' if row['task'] in ('ac2','cp26') else
                 'rglru' if row['task'] == 'rglru' else 'science')
        if args.stage != 'all' and args.stage != stage:
            continue
        folder = ROOT / row['package_dir']
        receipt = folder / 'submission.json'
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous.get('state') == 'submitted':
                print(json.dumps(previous), flush=True)
                continue
            raise RuntimeError(f'Unreconciled submission: {receipt}')
        archive, task = ROOT / row['archive'], ROOT / row['task_yaml']
        if hashlib.sha256(archive.read_bytes()).hexdigest() != row['archive_sha256']:
            raise RuntimeError('archive changed after packaging')
        if hashlib.sha256(task.read_bytes()).hexdigest() != row['yaml_sha256']:
            raise RuntimeError('task YAML changed after packaging')
        bucket_name, object_name = row['code_uri'][5:].split('/', 1)
        existing = storage_client.list_blobs(bucket_name, prefix=f"ray-training/{row['run_id']}/", max_results=1)
        if next(iter(existing), None) is not None:
            raise RuntimeError(f"Fresh run already has durable output: {row['run_id']}")
        blob = storage_client.bucket(bucket_name).blob(object_name)
        if not blob.exists():
            blob.upload_from_filename(archive, if_generation_match=0, timeout=180)
        if hashlib.sha256(blob.download_as_bytes(timeout=180)).hexdigest() != row['archive_sha256']:
            raise RuntimeError('uploaded archive checksum mismatch')
        record = dict(run_id=row['run_id'], pool=row['pool'], priority=row['priority'],
                      state='attempting', code_uri=row['code_uri'], archive_sha256=row['archive_sha256'],
                      submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(receipt, record)
        # The explicit attempt is durable before dispatch, preventing duplicate
        # jobs if the CLI times out after the controller accepts the launch.
        proc = subprocess.run([args.sky, 'jobs', 'launch', '-p', row['pool'], str(task),
                               '--priority', str(row['priority']), '-y', '-d'],
                              capture_output=True, text=True, timeout=360)
        log = proc.stdout + proc.stderr
        (folder / 'submit.log').write_text(log)
        match = re.search(r'Managed Job ID:\s*(\d+)', log)
        record.update(exit_code=proc.returncode, state='submitted' if match else 'unreconciled')
        if match:
            record['job_id'] = int(match.group(1))
        save(receipt, record)
        print(json.dumps(record), flush=True)
        if proc.returncode != 0 or not match:
            raise RuntimeError(f'Submission needs reconciliation: {receipt}')


if __name__ == '__main__':
    main()
