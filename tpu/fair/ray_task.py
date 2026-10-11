"""FAIR feature-family grading as farm Ray tasks: one TPU chip per grade, fresh child process per candidate.

Same pattern as the placement and RG-LRU TPU graders: Ray assigns exactly one chip (resources
{'TPU': 1, 'grading_fair': 1}, which exist only on the farm's non-engine hosts), the candidate runs in a
fresh child under process.py's guard holding a per-chip lock, and no generated code runs in the Ray worker.

Host preparation (`prepare_host`) stages the grader's Python env and frozen inputs from the run's
same-region bucket (FAIR_FACTORY_GCS, default gs://sk7524-tinker-tpu-us-east5/fair-factory/v1) into
/dev/shm/fair (manifest.json with sha256 per file), once per host under a file lock.

The returned result is the grader's public view: the hidden-shift block stays in
/dev/shm/fair/factory/results/<host>.jsonl on the grading host and never reaches the trainer.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import ray

SHM = Path('/dev/shm/fair')
FACTORY = SHM / 'factory'
GCS = os.environ.get('FAIR_FACTORY_GCS', 'gs://sk7524-tinker-tpu-us-east5/fair-factory/v1')
CODE = Path(__file__).resolve().parent          # tpu/fair: tpu_grader/ is vendored next to this file
CHILD_TIMEOUT_S = 900


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def _gcloud(*args):
    env = dict(os.environ, CLOUDSDK_STORAGE_CHECK_HASHES='never')   # v5p images lack crc32c; we check sha256
    subprocess.run(['gcloud', 'storage', *args], check=True, env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, timeout=3600)


def stage():
    """Download and verify the grader env + frozen inputs once per host. Returns the READY record."""
    SHM.mkdir(parents=True, exist_ok=True)
    ready = SHM / 'READY.json'
    with open(SHM / '.stage.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ready.is_file():
            return json.loads(ready.read_text())
        t0 = time.time()
        _gcloud('cp', f'{GCS}/manifest.json', str(SHM / 'manifest.json'))
        manifest = json.loads((SHM / 'manifest.json').read_text())
        for item in manifest['files']:
            dest = SHM / item['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(3):
                if dest.is_file() and _sha256(dest) == item['sha256']:
                    break
                _gcloud('cp', f'{GCS}/{item["path"]}', str(dest))
            else:
                raise RuntimeError(f'sha256 mismatch after 3 downloads: {item["path"]}')
        for item in manifest.get('untar', []):
            subprocess.run(['tar', '-xzf', str(SHM / item['archive']), '-C', str(SHM / item['into'])], check=True)
        record = dict(staged_s=round(time.time() - t0, 1), files=len(manifest['files']), version=manifest.get('version'))
        ready.write_text(json.dumps(record))
        return record


@ray.remote(num_cpus=0.1, max_retries=0)
def prepare_host():
    """Readiness probe for one grading host (scheduled with node affinity by the farm's warmup)."""
    try:
        record = stage()
        python = SHM / 'venv/bin/python'
        if not python.is_file():
            return dict(ok=False, detail='grader venv missing after staging')
        return dict(ok=True, host=socket.gethostname(), **record)
    except Exception as exc:  # noqa: BLE001
        return dict(ok=False, detail=f'{type(exc).__name__}: {exc}'[:500])


def _chip():
    from tpu.science.placement_slots import assigned_chip
    return assigned_chip(ray.get_runtime_context().get_accelerator_ids())


def grade(spec):
    """Grade one candidate on this task's Ray-assigned chip; returns the public result dict."""
    stage()
    chip = _chip()
    host = socket.gethostname()
    (FACTORY / 'work').mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='grade-', dir=FACTORY / 'work'))
    cand = work / 'candidate.py'
    cand.write_text(spec['source'])
    out = work / 'out'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('RAY_', 'TPU_', 'JAX_', 'CLOUD_TPU', 'XLA_'))}
    env.update(PYTHONPATH=str(CODE), FAIR_FACTORY_ROOT=str(FACTORY), JAX_PLATFORMS='tpu',
               TPU_VISIBLE_CHIPS=str(chip), TPU_PROCESS_BOUNDS='1,1,1', TPU_CHIPS_PER_PROCESS_BOUNDS='1,1,1',
               TPU_PROCESS_PORT=str(8476 + chip), TPU_PROCESS_ADDRESSES=f'localhost:{8476 + chip}',
               CLOUD_TPU_TASK_ID='0', OMP_NUM_THREADS='16', OPENBLAS_NUM_THREADS='16')
    child = [str(SHM / 'venv/bin/python'), '-m', 'tpu_grader.grade', str(cand), '--out', str(out),
             '--mode', spec['mode'], '--stack', spec.get('stack', 'v2')]
    locks = SHM / 'chip-locks'
    locks.mkdir(exist_ok=True)
    process_py = Path(__file__).resolve().parents[1] / 'swarm' / 'ray_train' / 'process.py'
    command = [sys.executable, str(process_py), 'guard', str(os.getpid()), json.dumps(child), '15',
               str(locks / f'{chip}.lock')]
    t0 = time.time()
    log = work / 'grade.log'
    with log.open('wb') as fh:
        proc = subprocess.Popen(command, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=35)
            rc = 'timeout'
    wall = round(time.time() - t0, 1)
    result_path = out / 'result.json'
    if result_path.is_file():
        result = json.loads(result_path.read_text())
    else:
        tail = log.read_bytes()[-2000:].decode(errors='replace')
        result = dict(ok=False, reward=None, error=f'grader exited {rc} without a result: {tail}')
    results = FACTORY / 'results'
    results.mkdir(parents=True, exist_ok=True)
    with open(results / f'{host}.jsonl', 'a') as fh:
        fh.write(json.dumps(dict(t=time.time(), chip=chip, mode=spec['mode'], source=spec['source'],
                                 result=result)) + '\n')
    public = {k: v for k, v in result.items() if k != 'hidden'}
    public['correctness'] = 1.0 if result.get('ok') else 0.0
    public['metrics'] = dict(public.get('metrics') or {}, host=host, chip=chip, wall_s=wall)
    return public


@ray.remote(max_retries=0)
def grade_fair(spec):
    return grade(spec)
