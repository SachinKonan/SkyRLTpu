"""Drain a file-based work queue with a single-node Ray cluster inside one Slurm job.

Queue layout (QUEUE=<dir>):
  pending/<id>.json   task specs waiting to run
  running/<id>.json   claimed by the current job
  done/<id>.json      finished (exit code 0), spec + result
  failed/<id>.json    finished with a nonzero exit or exceeded its own timeout
Each task writes its logs and outputs under <QUEUE>/work/<id>/.

Task spec (JSON):
  {"id": str, "cmd": [str, ...], "cwd": str, "env": {..}, "est_seconds": float,
   "timeout_seconds": float, "gpus": 1, "cpus": 12, "memory_gb": 200, "priority": int}

A slot starts a task only if the job's remaining time covers est_seconds plus a margin.
When the job's deadline arrives, running tasks are killed and returned to pending/ for the
next array job; done tasks are never rerun. Only one array job runs at a time (%1), so this
process is the only writer.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

import ray


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(path)


@ray.remote
def run_task(spec, work, deadline):
    """Run one task; returns a result dict. CUDA_VISIBLE_DEVICES is set by Ray."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in spec.get('env', {}).items()})
    env['TASK_WORK'] = str(work)
    threads = str(spec.get('cpus', 12))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env.setdefault(key, threads)
    limit = min(float(spec['timeout_seconds']), deadline - time.time())
    started = time.time()
    with open(work / 'stdout.log', 'ab') as log:
        log.write(f"\n=== {time.strftime('%F %T')} host={socket.gethostname()} "
                  f"gpus={env.get('CUDA_VISIBLE_DEVICES')} limit={limit:.0f}s ===\n".encode())
        log.flush()
        proc = subprocess.Popen(spec['cmd'], cwd=spec.get('cwd') or str(work), env=env, stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = proc.wait(timeout=max(1.0, limit))
            status = 'done' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = None
            # Hitting the job deadline is an infrastructure interruption; hitting the task's
            # own timeout is a task failure.
            status = 'interrupted' if deadline - time.time() < 5 else 'failed'
    return dict(status=status, exit_code=code, seconds=time.time() - started,
                host=socket.gethostname(), gpus=env.get('CUDA_VISIBLE_DEVICES'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', required=True)
    ap.add_argument('--wall-seconds', type=float, default=3600)
    ap.add_argument('--reserve-seconds', type=float, default=150, help='kept free at the end of the job')
    ap.add_argument('--start-margin', type=float, default=60, help='extra slack required to start a task')
    ap.add_argument('--gpus', type=int, default=4)
    ap.add_argument('--cpus', type=int, default=48)
    a = ap.parse_args()
    q = Path(a.queue)
    for sub in ('pending', 'running', 'done', 'failed', 'work'):
        (q / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    deadline = t0 + a.wall_seconds - a.reserve_seconds
    job = os.environ.get('SLURM_JOB_ID', 'local')

    # Tasks a killed job left in running/ go back to pending/.
    for f in (q / 'running').glob('*.json'):
        spec = load(f)
        spec.setdefault('interruptions', []).append(dict(job=spec.get('last_job'), reason='stale'))
        save(q / 'pending' / f.name, spec)
        f.unlink()

    # A login-shell RAY_RUNTIME_ENV_HOOK (from a newer Ray/uv setup) breaks this Ray version.
    for key in [k for k in os.environ if k.startswith('RAY_RUNTIME_ENV')]:
        os.environ.pop(key)
    ray.init(num_cpus=a.cpus, num_gpus=a.gpus, include_dashboard=False, log_to_driver=False)
    inflight = {}   # ObjectRef -> (spec, path)
    stop = False

    def handle(sig, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, handle)

    def free_gpus():
        return a.gpus - sum(s.get('gpus', 1) for s, _ in inflight.values())

    while True:
        now = time.time()
        if not stop and now < deadline:
            pending = sorted((load(f) | {'_path': str(f)} for f in (q / 'pending').glob('*.json')),
                             key=lambda s: (-s.get('priority', 0), -s.get('est_seconds', 0), s['id']))
            for spec in pending:
                if free_gpus() < spec.get('gpus', 1):
                    break
                if now + spec.get('est_seconds', 0) + a.start_margin > deadline:
                    continue   # too long for what is left of this job; the next job takes it
                src = Path(spec.pop('_path'))
                spec['last_job'] = job
                path = q / 'running' / src.name
                save(path, spec)
                src.unlink()
                ref = run_task.options(num_gpus=spec.get('gpus', 1), num_cpus=spec.get('cpus', 12),
                                       memory=int(spec.get('memory_gb', 0) * 1024**3) or None
                                       ).remote(spec, str(q / 'work' / spec['id']), deadline)
                inflight[ref] = (spec, path)
                print(f"[{time.time()-t0:7.0f}s] start {spec['id']} (est {spec.get('est_seconds', 0):.0f}s)", flush=True)
        if not inflight:
            if stop or time.time() >= deadline or not any((q / 'pending').glob('*.json')):
                break
            # nothing fits in the remaining time
            if all(time.time() + load(f).get('est_seconds', 0) + a.start_margin > deadline
                   for f in (q / 'pending').glob('*.json')):
                break
        ready, _ = ray.wait(list(inflight), num_returns=1, timeout=10)
        for ref in ready:
            spec, path = inflight.pop(ref)
            try:
                result = ray.get(ref)
            except Exception as exc:   # worker died
                result = dict(status='interrupted', error=repr(exc))
            spec.setdefault('attempts', []).append(dict(job=job, **result))
            dest = {'done': 'done', 'failed': 'failed'}.get(result['status'], 'pending')
            save(q / dest / path.name, spec)
            path.unlink(missing_ok=True)
            print(f"[{time.time()-t0:7.0f}s] {result['status']} {spec['id']} "
                  f"({result.get('seconds', 0):.0f}s, exit {result.get('exit_code')})", flush=True)
        if stop or time.time() >= deadline + 30:
            break

    # Anything still running is returned to the queue for the next job.
    for ref, (spec, path) in inflight.items():
        ray.cancel(ref, force=True)
        spec.setdefault('attempts', []).append(dict(job=job, status='interrupted', reason='job deadline'))
        save(q / 'pending' / path.name, spec)
        path.unlink(missing_ok=True)
        print(f"[{time.time()-t0:7.0f}s] requeued {spec['id']}", flush=True)
    ray.shutdown()
    left = len(list((q / 'pending').glob('*.json')))
    print(f"job {job} exits: done={len(list((q/'done').glob('*.json')))} failed={len(list((q/'failed').glob('*.json')))} pending={left}")


if __name__ == '__main__':
    main()
