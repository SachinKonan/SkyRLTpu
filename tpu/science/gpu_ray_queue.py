"""Four-GPU allocation-local Ray workers backed by a durable GPFS task queue.

Each task owns a flock throughout execution. Persistent Slurm ownership also
prevents reclaiming an orphaned child until its allocation has ended. Results,
including invalid solutions, are terminal; infrastructure interruptions retry.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

_JOB_STATUS = {}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def job_alive(job):
    # Fail closed on scheduler communication errors.
    now = time.monotonic()
    cached = _JOB_STATUS.get(job)
    if cached and now - cached[0] < 30:
        return cached[1]
    try:
        r = subprocess.run(['squeue', '-h', '-j', job, '-o', '%T'],
                           capture_output=True, text=True, timeout=15)
    except (subprocess.TimeoutExpired, OSError):
        _JOB_STATUS[job] = (now, True)
        return True
    if r.returncode and 'Invalid job id' in r.stderr:
        alive = False
    else:
        alive = r.returncode != 0 or bool(r.stdout.strip())
    _JOB_STATUS[job] = (now, alive)
    return alive


class Queue:
    def __init__(self, root, alive=job_alive):
        self.root = Path(root)
        self.plan = json.loads((self.root / 'plan.json').read_text())
        self.alive = alive

    def terminal(self, task):
        name = task['method'] + '-' + task['case']
        done = self.root / 'tasks' / name / 'done.json'
        if done.exists():
            return True
        legacy = Path(self.plan['legacy']) / name / 'report.json'
        if legacy.exists() and task['method'] not in self.plan.get('ignore_legacy_methods', []):
            json.loads(legacy.read_text())  # A partial JSON is never completion.
            return True
        return False

    def claim(self, owner):
        for task in self.plan['jobs']:
            name = task['method'] + '-' + task['case']
            folder = self.root / 'tasks' / name
            folder.mkdir(parents=True, exist_ok=True)
            lock = (folder / 'lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            try:
                if self.terminal(task):
                    lock.close()
                    continue
                legacy_job = self.plan.get('legacy_active', {}).get(name)
                if legacy_job and self.alive(legacy_job):
                    lock.close()
                    continue
                state = folder / 'claim.json'
                previous = json.loads(state.read_text()) if state.exists() else {}
                if previous.get('job') and self.alive(previous['job']):
                    lock.close()
                    continue
                attempt = folder / ('attempt-' + uuid.uuid4().hex[:12])
                attempt.mkdir()
                atomic_json(state, dict(owner, task=task, attempt=str(attempt),
                                        claimed_unix=time.time()))
                return task, attempt, lock
            except BaseException:
                lock.close()
                raise
        return None


def has_time(deadline, now, task_seconds=3300, cleanup_seconds=30):
    return deadline - now >= task_seconds + cleanup_seconds


def run_task(root, queue, task, attempt, gpu_uuid, cpus, task_seconds):
    started = time.monotonic()
    xp = attempt / 'xplace'
    xp.mkdir()
    original = Path(queue.plan['build_xplace'])
    for name in ['src', 'utils', 'cpp_to_py', 'data', 'thirdparty', 'tool', 'main.py', 'CMakeLists.txt']:
        source = original / name
        if source.is_dir():
            shutil.copytree(source, xp / name, symlinks=True)
        else:
            shutil.copy2(source, xp / name)
    method = task['method']
    source_root = Path(queue.plan.get('source_root', root))
    variant = task.get('variant')
    if variant is not None:
        from .abuplace_starts import VARIANTS
        if variant not in VARIANTS or method != 'xplace-' + variant:
            raise ValueError('invalid Xplace variant task')
    repo = source_root / '.science' / ('abuplace' if method == 'abuplace' or variant else 'archgen-cuda-run')
    output = attempt / 'evaluation'
    cmd = [queue.plan['candidate_python'],
           str(root / 'tpu/science/placement_gpu_suite.py'), '--method', 'xplace-abu' if variant else method,
           '--case', task['case'], '--repository', str(repo), '--xplace-root', str(xp),
           '--output', str(output), '--cpus', str(cpus), '--seconds', str(queue.plan.get('candidate_seconds', 3180))]
    if variant:
        cmd += ['--variant', variant]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, OMP_NUM_THREADS=str(cpus),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    proc = None
    timed_out = False
    try:
        with (attempt / 'wrapper.log').open('wb') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            try:
                proc.wait(timeout=max(0, task_seconds - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    report = output / 'report.json'
    if not report.exists():
        output.mkdir(exist_ok=True)
        atomic_json(report, dict(task, valid=False, reward=0.0,
                    error='total task wall timeout' if timed_out else 'wrapper exited without report',
                    task_wall_seconds=time.monotonic() - started, exit_code=proc.returncode))
    result = json.loads(report.read_text())
    atomic_json(attempt.parent / 'done.json', {
        'report': str(report), 'valid': result.get('valid', False),
        'job': os.environ['SLURM_JOB_ID'], 'gpu_uuid': gpu_uuid,
        'cpus': cpus, 'task_seconds_limit': task_seconds,
        'wall_seconds': time.monotonic() - started, 'finished_unix': time.time(),
    })
    return result.get('valid', False)


def worker_loop(root, queue_path, slot, affinity, deadline, task_seconds):
    root = Path(root)
    os.sched_setaffinity(0, affinity)
    queue = Queue(queue_path)
    # Ray sets CUDA_VISIBLE_DEVICES per worker. Convert it to a stable UUID
    # before entering the bwrap device namespace.
    probe = ('import setuptools; import torch; from torch.utils.cpp_extension import load; '
             'assert torch.cuda.device_count()==1; '
             'assert "A100" in torch.cuda.get_device_name(0); '
             'print("GPU-"+str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-"))')
    gpu = subprocess.check_output([queue.plan['candidate_python'], '-c', probe],
                                  text=True, timeout=45).strip()
    owner = dict(job=os.environ['SLURM_JOB_ID'], slot=slot, host=os.uname().nodename,
                 gpu_uuid=gpu, pid=os.getpid())
    atomic_json(Path(queue_path) / 'allocations' / owner['job'] / f'worker-{slot}.json',
                dict(owner, affinity=affinity, ready_unix=time.time()))
    completed = []
    while has_time(deadline, time.time(), task_seconds):
        claim = queue.claim(owner)
        if claim is None:
            break
        task, attempt, lock = claim
        try:
            print(json.dumps(dict(event='claimed', task=task, **owner)), flush=True)
            valid = run_task(root, queue, task, attempt, gpu, len(affinity), task_seconds)
            completed.append(dict(task, valid=valid))
            print(json.dumps(dict(event='completed', task=task, valid=valid, **owner)), flush=True)
        finally:
            lock.close()
    return dict(owner, completed=completed, remaining_seconds=deadline-time.time())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', required=True)
    p.add_argument('--deadline', required=True, type=float)
    p.add_argument('--gpus', type=int, default=4)
    p.add_argument('--cpus-per-task', type=int, default=8)
    p.add_argument('--task-seconds', type=int, default=3300)
    args = p.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'allocation signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    import ray
    root = Path(__file__).resolve().parents[2]
    queue = Path(args.queue).resolve()
    job = os.environ['SLURM_JOB_ID']
    affinity = sorted(os.sched_getaffinity(0))
    count = args.gpus * args.cpus_per_task
    if len(affinity) < count:
        raise RuntimeError(f'Need {count} allocated CPUs, got {len(affinity)}')
    allocation = queue / 'allocations' / job
    allocation.mkdir(parents=True, exist_ok=True)
    temp = f'/tmp/cray-{job}-{os.getpid()}'
    atomic_json(allocation / 'start.json', dict(job=job, host=os.uname().nodename,
                deadline=args.deadline, started_unix=time.time(), affinity=affinity[:count],
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), ray_version=ray.__version__))
    ray.init(address='local', num_cpus=count, num_gpus=args.gpus, include_dashboard=False,
             object_store_memory=512*1024**2, _memory=56*1024**3,
             _temp_dir=temp, _node_ip_address='127.0.0.1')
    results, errors = [], []
    try:
        remote = ray.remote(num_gpus=1, num_cpus=args.cpus_per_task,
                            memory=12*1024**3, max_retries=0)(worker_loop)
        pending = [remote.remote(str(root), str(queue), slot,
                   affinity[slot*args.cpus_per_task:(slot+1)*args.cpus_per_task],
                   args.deadline, args.task_seconds) for slot in range(args.gpus)]
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=20)
            for ref in ready:
                try:
                    results.append(ray.get(ref))
                except Exception as exc:
                    errors.append(repr(exc))
            atomic_json(allocation / 'progress.json', dict(results=results, errors=errors))
            if time.time() >= args.deadline-15:
                raise TimeoutError('Allocation cleanup deadline reached')
    finally:
        ray.shutdown()
        atomic_json(allocation / 'finish.json', dict(results=results, errors=errors,
                                                    finished_unix=time.time()))
    if errors:
        raise RuntimeError(errors)


if __name__ == '__main__':
    main()
