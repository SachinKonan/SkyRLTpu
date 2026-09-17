"""Wait for named seed shards, verify/merge them, then submit pinned training."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time

import yaml

from .bootstrap import identity, read, save
from .seed_pool import merge
from tpu.swarm.ray_train.config import Config


def seed_job_status(sky, shards, env):
    """An unavailable control plane is unknown, never success or job failure."""
    try:
        queue = subprocess.run([sky, 'jobs', 'queue', '-a'], env=env,
                               capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None, 'SkyPilot queue request timed out; will retry'
    if queue.returncode:
        return None, 'SkyPilot queue request failed; will retry: ' + queue.stderr[-1000:]
    pending = []
    for shard in shards:
        row = next((line for line in queue.stdout.splitlines()
                    if re.match(r'^\s*' + str(shard['job_id']) + r'\s', line)), '')
        if re.search(r'\b(?:FAILED\w*|CANCELLED)\b', row):
            raise RuntimeError(f"seed job {shard['job_id']} failed or was cancelled; no training submitted")
        if not re.search(r'\bSUCCEEDED\b', row):
            pending.append('job-' + str(shard['job_id']))
    return pending, None


def bind_bundle(template, output, digest):
    """Bind only the imported pool hash; all executable bytes remain pinned."""
    template, output = Path(template), Path(output)
    manifest = read(template / 'manifest.json')
    archive = template / 'science-training.tar.gz'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('training template checksum mismatch')
    if output.exists():
        raise ValueError('training package already exists; reconcile before retry')
    output.mkdir(parents=True)
    name = manifest['run_id']
    profile = f'tpu/swarm/ray_train/profiles/{name}.json'
    changed = 0
    with tarfile.open(archive) as source, tarfile.open(output / 'science-training.tar.gz', 'w:gz') as dest:
        for member in source:
            stream = source.extractfile(member) if member.isfile() else None
            if member.name == profile:
                config = json.load(stream)
                if config.get('seed_pool_sha256') != '0' * 64:
                    raise ValueError('training template is already bound to a seed pool')
                config['seed_pool_sha256'] = digest
                Config.from_dict(config).validate()
                data = (json.dumps(config, indent=2) + '\n').encode()
                member.size = len(data)
                stream = io.BytesIO(data)
                changed += 1
            dest.addfile(member, stream)
    if changed != 1:
        raise ValueError('expected exactly one training profile')
    code_hash = hashlib.sha256((output / 'science-training.tar.gz').read_bytes()).hexdigest()
    uri = manifest['code_uri'].rsplit('/', 1)[0] + '/science-training-' + code_hash + '.tar.gz'
    doc = yaml.safe_load((template / (name + '.yaml')).read_text())
    doc['envs'].update(RAY_TRAIN_CODE=uri, RAY_TRAIN_CODE_SHA256=code_hash)
    task = output / (name + '.yaml')
    task.write_text(yaml.safe_dump(doc, sort_keys=False))
    manifest.update(sha256=code_hash, code_uri=uri, seed_pool_sha256=digest,
                    template_sha256=manifest['sha256'], task_sha256=hashlib.sha256(task.read_bytes()).hexdigest())
    save(output / 'manifest.json', manifest)


def run(plan_path):
    plan = read(plan_path)
    root = Path(plan['directory'])
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **plan['environment'])
    gcloud = plan['gcloud']

    def command(args, *, allow_missing=False):
        result = subprocess.run([gcloud, *args], env=env, capture_output=True, timeout=300)
        if result.returncode:
            error = result.stderr.decode(errors='replace')
            if allow_missing and any(s in error.lower() for s in (
                    'no urls matched', 'matched no objects or files', 'not found', 'does not exist')):
                return None
            raise RuntimeError('gcloud operation failed: ' + error[-2000:])
        return result.stdout

    def unchanged():
        for path, digest in plan['source_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('handoff implementation changed: ' + path)

    unchanged()
    configs = [Config.load(s['profile']) for s in plan['shards']]
    target = Config.load(plan['training_profile'])
    until = time.monotonic() + plan.get('deadline_seconds', 129600)
    while True:
        if time.monotonic() >= until:
            raise TimeoutError('seed shards did not finish; no training submitted')
        pending, unavailable = seed_job_status(plan['sky'], plan['shards'], env)
        if unavailable:
            status = dict(event='queue_status_unavailable', detail=unavailable, time=time.time())
            save(root / 'status.json', status)
            print(json.dumps(status), flush=True)
            time.sleep(60)
            continue
        for config in configs:
            if command(['storage', 'cat', config.run_gcs + '/client/bootstrap/complete.json'], allow_missing=True) is None:
                pending.append(config.run_id)
        status = dict(event='waiting_for_seed_shards', pending=pending, time=time.time())
        save(root / 'status.json', status)
        print(json.dumps(status), flush=True)
        if not pending:
            break
        if time.monotonic() >= until:
            raise TimeoutError('seed shards did not finish; no training submitted')
        time.sleep(60)
    unchanged()
    clients = []
    for config in configs:
        client = root / 'downloads' / config.run_id / 'client'
        client.mkdir(parents=True, exist_ok=True)
        command(['storage', 'rsync', '--recursive', config.run_gcs + '/client/', str(client)])
        if Config.from_dict(read(client / 'bootstrap/contract.json')['contract']['config']) != config:
            raise ValueError('downloaded shard differs from the submitted profile')
        clients.append(client)
    result = merge(clients, target, root / 'merged')
    pool = root / 'merged/puct_sampler_step_000000.json'
    uri = target.run_gcs + '/client/tinker_log/' + target.run_id + '/' + pool.name
    command(['storage', 'cp', '--no-clobber', str(pool), uri])
    if identity(json.loads(command(['storage', 'cat', uri]))) != result['pool_sha256']:
        raise ValueError('uploaded training seed pool checksum mismatch')
    command(['storage', 'cp', '--no-clobber', str(root / 'merged/seed-import.json'),
             target.run_gcs + '/client/seed-import.json'])
    bind_bundle(plan['training_template'], plan['training_package'], result['pool_sha256'])
    unchanged()
    subprocess.run([plan['submit_python'], plan['submit_script'], plan['training_package'], plan['training_pool']],
                   env=env, check=True, timeout=900)
    status = dict(event='training_submitted', time=time.time(), seeds=result,
                  submission=read(Path(plan['training_package']) / 'submission.json'))
    save(root / 'status.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    args = parser.parse_args()
    try:
        run(args.plan)
    except Exception as exc:
        plan = read(args.plan)
        save(Path(plan['directory']) / 'status.json', dict(event='handoff_failed', error=str(exc), time=time.time()))
        raise
