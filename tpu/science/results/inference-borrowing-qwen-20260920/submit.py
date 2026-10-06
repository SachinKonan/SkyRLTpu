"""Submit this one authorized v6e circuit borrowing trial, with a durable receipt."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import google.auth
import yaml
from google.auth.transport.requests import AuthorizedSession
from google.auth.transport.requests import Request
from google.cloud import storage

HERE = Path(__file__).resolve().parent
RUN = 'science-circuit-v6e-qwen-borrow-supervised-20260920'
POOL = 'tpuswarm-v6e32-central1b'
ACCOUNT = '289186856710-compute@developer.gserviceaccount.com'
GCLOUD = '/scratch/gpfs/ZHUANGL/sk7524/google-cloud-sdk/bin/gcloud'
SKY = '/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/sky'


def save(path, record):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def main():
    receipt = HERE / 'submission.json'
    if receipt.exists():
        raise RuntimeError('Existing submission receipt: reconcile instead of submitting twice')
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    credentials.refresh(Request())
    active = subprocess.check_output([GCLOUD, 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'], text=True).strip()
    assert active == ACCOUNT == credentials.service_account_email
    records = [json.loads(line) for line in (HERE.parent / 'inference-borrowing-qwen-20260919/batch-probe.log').read_text().splitlines() if line.startswith('{')]
    assert any(r.get('event') == 'passed' and r.get('total_choices') == 128 for r in records)
    assert records[-1].get('event') == 'released'
    client = storage.Client(project='vision-mix', credentials=credentials)
    bucket_name = 'sk7524-tinker-tpu-us-central1'
    assert next(iter(client.list_blobs(bucket_name, prefix='ray-training/' + RUN + '/', max_results=1)), None) is None
    session = AuthorizedSession(credentials)
    response = session.get('https://tpu.googleapis.com/v2/projects/vision-mix/locations/us-central1-b/nodes', timeout=30)
    response.raise_for_status()
    assert any(n.get('acceleratorType') == 'v6e-32' and n.get('state') == 'READY' for n in response.json().get('nodes', []))
    import sky
    pools = sky.get(sky.jobs.pool_status([POOL]))
    assert any(str(r.get('status')).endswith('READY') and not r.get('used_by') for p in pools for r in p.get('replica_info', [])), 'No idle worker; do not duplicate a pending experiment'
    archive = HERE / 'bundle/science-training.tar.gz'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    task = HERE / 'bundle' / (RUN + '.yaml')
    doc = yaml.safe_load(task.read_text())
    check = json.loads((HERE / 'artifact-check.json').read_text())
    assert check['passed'] and check['archive_sha256'] == digest
    assert doc['envs']['RAY_TRAIN_CODE_SHA256'] == digest
    assert 'prepare_placement_host' in doc['run']
    uri = doc['envs']['RAY_TRAIN_CODE']
    assert uri.startswith('gs://' + bucket_name + '/code-bundles/science-training-')
    object_name = uri.removeprefix('gs://' + bucket_name + '/')
    blob = client.bucket(bucket_name).blob(object_name)
    if not blob.exists():
        blob.upload_from_filename(str(archive), if_generation_match=0, timeout=180)
    assert hashlib.sha256(blob.download_as_bytes(timeout=180)).hexdigest() == digest
    task = HERE / 'bundle' / (RUN + '.yaml')
    record = dict(run_id=RUN, pool=POOL, state='attempting',
        code_uri='gs://' + bucket_name + '/' + object_name, archive_sha256=digest,
        task_sha256=hashlib.sha256(task.read_bytes()).hexdigest(),
        external_discovery_pool='tpuswarm-v4-32-central2-smoke', supersedes_job_id=1270,
        submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(receipt, record)
    rows, *_ = sky.get(sky.jobs.queue_v2(refresh=False, skip_finished=False,
        fields=['job_id', 'job_name', 'status', 'current_cluster_name']))
    old = next(r for r in rows if r.job_id == 1270)
    assert old.job_name == 'science-circuit-v6e-qwen-borrow173-grpo-20260919-fix1'
    if getattr(old.status, 'value', str(old.status)) not in ('CANCELLED', 'FAILED', 'SUCCEEDED'):
        cancel = subprocess.run([SKY, 'jobs', 'cancel', '1270', '-y'], capture_output=True, text=True, timeout=120)
        (HERE / 'cancel-1270.log').write_text(cancel.stdout + cancel.stderr)
        if cancel.returncode:
            raise RuntimeError('Old trial cancellation failed; reconcile receipt before launch')
    proc = subprocess.run([SKY, 'jobs', 'launch', '-p', POOL, str(task), '--priority', '50', '-y', '-d'],
                          capture_output=True, text=True, timeout=360)
    text = proc.stdout + proc.stderr
    (HERE / 'submit.log').write_text(text)
    match = re.search(r'Managed Job ID:\s*(\d+)', text)
    record.update(exit_code=proc.returncode, state='submitted' if match else 'unreconciled')
    if match:
        record['job_id'] = int(match.group(1))
    save(receipt, record)
    print(json.dumps(record), flush=True)
    if not match or proc.returncode:
        raise RuntimeError('Submission needs reconciliation; do not blindly retry')


if __name__ == '__main__':
    main()
