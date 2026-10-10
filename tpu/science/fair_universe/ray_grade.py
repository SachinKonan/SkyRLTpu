"""A complete detector fit/evaluation occupies one exclusively assigned chip."""
import json
import os
from pathlib import Path
import pwd
import subprocess
import time
import uuid
import ray
from .contract import VERSION, digest


def validate_spec(spec):
    source = spec.get('source')
    if not isinstance(source, str) or not source or len(source.encode()) > 1024**2:
        raise ValueError('fair_universe requires a source module of at most 1 MiB')
    seed, steps = spec.get('seed', 0), spec.get('steps', 250)
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError('baseline seed must be 0, 1, or 2')
    if type(steps) is not int or steps not in (250, 1000):
        raise ValueError('baseline tier must be 250 or 1000 steps')
    if set(spec) - {'source', 'seed', 'steps', 'admission_timeout_s'}:
        raise ValueError('candidate may not override the evaluation contract')
    return dict(source=source, seed=seed, steps=steps)


@ray.remote(num_cpus=1, max_retries=0)
def readiness(dataset, python):
    root = Path(dataset)
    marker = root / 'qualification.json'
    if not marker.is_file():
        return dict(ready=False, reason='actual TPU isolation and evaluator qualification required')
    data = json.loads(marker.read_text())
    ok = (Path(python).is_file() and data.get('version') == VERSION
          and data.get('manifest_sha256') == digest(root/'manifest.json')
          and data.get('chips_passed') == [0, 1, 2, 3]
          and data.get('independence_passed') is True)
    return dict(ready=ok, qualification=data)


@ray.remote(num_cpus=4, memory=16*1024**3,
            resources={'TPU': 1, 'grading_fair_universe': 1}, max_calls=1, max_retries=0)
def grade(spec, dataset, python, root, memory_gib=16):
    from tpu.science.placement_slots import assigned_chip, chip_lock, chip_cpus, device_paths
    from tpu.science.worker import process_identity
    spec = validate_spec(spec)
    chip = assigned_chip(ray.get_runtime_context().get_accelerator_ids())
    root = Path(root).expanduser().resolve()
    work = root/'fair-trials'/uuid.uuid4().hex
    work.mkdir(parents=True)
    source = work/'candidate.py'; source.write_text(spec['source'])
    run = work/'evaluation'
    unit = 'fair-trial-'+work.name
    cpus = chip_cpus(chip)
    if not set(cpus) <= os.sched_getaffinity(0):
        raise RuntimeError('infrastructure: assigned grading CPUs unavailable')
    code = Path(__file__).resolve().parents[3]
    command = ['sudo', '-n', 'systemd-run', '--unit='+unit, '--wait', '--collect', '--pipe', '--quiet',
        '--uid='+pwd.getpwuid(os.getuid()).pw_name, '--gid='+str(os.getgid()),
        '--property=MemoryMax='+str(memory_gib)+'G', '--property=MemorySwapMax=0',
        '--property=LimitMEMLOCK=infinity', '--property=CPUQuota=400%',
        '--property=AllowedCPUs='+','.join(map(str, cpus)), '--property=TasksMax=1024',
        '--property=RuntimeMaxSec=5460', '--property=KillMode=control-group',
        '--property=TimeoutStopSec=2', '--property=OOMPolicy=stop',
        '--setenv=PYTHONPATH='+str(code), '--setenv=JAX_PLATFORMS=cpu',
        '--setenv=OPENBLAS_NUM_THREADS=4', '--setenv=OMP_NUM_THREADS=4',
        str(python), '-m', 'tpu.science.fair_universe.evaluate', '--candidate', str(source),
        '--dataset', dataset, '--run', str(run), '--python', str(python), '--chip', str(chip),
        '--seed', str(spec['seed']), '--steps', str(spec['steps']),
        '--owner-pid', str(os.getpid()), '--owner-start', process_identity(os.getpid())]
    started = time.monotonic()
    with chip_lock(chip):
        try:
            busy = subprocess.run(['sudo', '-n', 'fuser', str(device_paths(chip, 'tpu-v5p-32')[0])],
                                  capture_output=True, timeout=10)
            if busy.returncode != 1 or busy.stdout.strip() or busy.stderr.strip():
                raise RuntimeError('infrastructure: grading chip is occupied or ownership check failed')
            with (work/'worker.log').open('wb') as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    rc = proc.wait(timeout=5500)
                finally:
                    if proc.poll() is None:
                        proc.terminate(); proc.wait(timeout=10)
            if rc or not (run/'result.json').is_file():
                raise RuntimeError('infrastructure: trial supervisor failed without a verdict: '
                                   +(work/'worker.log').read_text(errors='replace')[-3000:])
            result = json.loads((run/'result.json').read_text())
        finally:
            subprocess.run(['sudo', '-n', 'systemctl', 'stop', unit], capture_output=True, timeout=15)
    result['metrics'] = dict(host=__import__('socket').gethostname(), physical_chip_id=chip,
                              seconds=time.monotonic()-started, artifact_directory=str(work),
                              ray_node_id=ray.get_runtime_context().get_node_id())
    return result
