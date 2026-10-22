"""Drain a shared file-based work queue with a single-node Ray cluster inside one Slurm job.

Many jobs may drain the same queue at once. A job keeps its GPUs for its whole wall time and
keeps starting tasks as slots free up, as long as a task's expected duration still fits.

Queue layout (QUEUE=<dir>):
  pending/<id>.json   waiting
  running/<id>.json   claimed; carries the owning job and its lease (the job's deadline)
  done/<id>.json      exit code 0          failed/<id>.json   nonzero exit or own timeout
  work/<id>/          logs and outputs of the task

Concurrency:
  * claim = atomic rename pending/ -> running/; a job that loses the race skips the task;
  * a running task whose lease expired (its job was killed) is returned to pending/ by any job;
  * at its own deadline a job kills what it still runs and returns those tasks to pending/.
Packing: a job is a pool of GPUs and cores. Each task declares a GPU share and a core count; the
runner starts every pending task that fits, giving it one GPU (CUDA_VISIBLE_DEVICES) and its own
slice of cores (CPU affinity), so a neighbouring task cannot take CPU time from a time-limited
search. A GPU-bound placer takes most of a GPU (gpu_share 0.9); a CPU-bound placer that needs a GPU
only briefly takes a sliver (e.g. 0.05) and shares that GPU. Method budgets are in the specs.

Task spec: {"id", "cmd": [...], "cwd", "env": {}, "est_seconds", "timeout_seconds",
            "gpu_share": 0.9, "cpus": 12, "priority"}
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import ray


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    tmp = Path(f'{path}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(path)


@ray.remote(num_cpus=0)
def run_task(spec, work, deadline, gpu, cores):
    """Run one task on the GPU and cores the driver assigned."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in spec.get('env', {}).items()})
    env['TASK_WORK'] = str(work)
    env['CUDA_VISIBLE_DEVICES'] = gpu
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env.setdefault(key, str(len(cores)))
    limit = min(float(spec['timeout_seconds']), deadline - time.time())
    gpu = subprocess.run(['nvidia-smi', '-i', env.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0],
                          '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    started = time.time()
    with open(work / 'stdout.log', 'ab') as log:
        log.write(f"\n=== {time.strftime('%F %T')} host={socket.gethostname()} job={os.environ.get('SLURM_JOB_ID')} "
                  f"gpu={env.get('CUDA_VISIBLE_DEVICES')} ({gpu}) cores={cores} limit={limit:.0f}s ===\n".encode())
        log.flush()
        proc = subprocess.Popen(spec['cmd'], cwd=spec.get('cwd') or str(work), env=env, stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                preexec_fn=lambda: os.sched_setaffinity(0, cores))
        try:
            code = proc.wait(timeout=max(1.0, limit))
            status = 'done' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = None
            # The job deadline is an infrastructure interruption; the task's own timeout is a failure.
            status = 'interrupted' if deadline - time.time() < 5 else 'failed'
    return dict(status=status, exit_code=code, seconds=time.time() - started, host=socket.gethostname(),
                gpu=gpu, gpu_index=env.get('CUDA_VISIBLE_DEVICES'), cores=cores)


def recover_stale(q, grace=120):
    """Return tasks whose owning job's lease expired (job killed) to pending/."""
    now = time.time()
    for f in (q / 'running').glob('*.json'):
        try:
            spec = load(f)
        except (OSError, ValueError):
            continue
        # A just-claimed task has no lease yet (the claimant writes it right after the rename);
        # judge such a file by its age instead.
        lease = spec.get('lease_deadline')
        if lease is None:
            try:
                lease = f.stat().st_mtime + 600
            except FileNotFoundError:
                continue
        if lease + grace < now:
            spec.setdefault('history', []).append(dict(job=spec.get('owner_job'), status='stale-lease'))
            spec.pop('lease_deadline', None)
            try:
                f.rename(q / 'pending' / f.name)   # atomic; another job may do it first
                save(q / 'pending' / f.name, spec)
            except FileNotFoundError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', required=True)
    ap.add_argument('--wall-seconds', type=float, default=3600)
    ap.add_argument('--reserve-seconds', type=float, default=120, help='kept free at the end of the job')
    ap.add_argument('--start-margin', type=float, default=60, help='extra slack required to start a task')
    ap.add_argument('--slice-cpus', type=int, default=12, help='cores per task slice')
    a = ap.parse_args()
    q = Path(a.queue)
    for sub in ('pending', 'running', 'done', 'failed', 'work'):
        (q / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    deadline = t0 + a.wall_seconds - a.reserve_seconds
    job = os.environ.get('SLURM_JOB_ID', f'local-{os.getpid()}')
    job_gpus = [g.strip() for g in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',') if g.strip()]
    job_cpus = sorted(os.sched_getaffinity(0))
    print(f'job {job} on {socket.gethostname()}: gpus {job_gpus}, cpus {len(job_cpus)}', flush=True)

    # A login-shell RAY_RUNTIME_ENV_HOOK (from a newer Ray/uv setup) breaks this Ray version.
    for key in [k for k in os.environ if k.startswith('RAY_RUNTIME_ENV')]:
        os.environ.pop(key)
    ray.init(num_cpus=len(job_cpus), num_gpus=0, include_dashboard=False, log_to_driver=False)
    inflight = {}   # ObjectRef -> (spec, running path, gpu, slice index)
    gpu_used = {g: 0.0 for g in job_gpus}
    slice_cpus = a.slice_cpus
    slices = [job_cpus[i * slice_cpus:(i + 1) * slice_cpus] for i in range(len(job_cpus) // slice_cpus)]
    free_slices = list(range(len(slices)))
    print(f'{len(slices)} core slices of {slice_cpus}; GPUs {job_gpus}', flush=True)
    stop = False

    def handle(sig, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, handle)

    def share(spec):
        return float(spec.get('gpu_share', spec.get('gpus', 1)))

    def pick_gpu(need):
        ok = [g for g in job_gpus if gpu_used[g] + need <= 1.0 + 1e-9]
        return min(ok, key=lambda g: (gpu_used[g], g)) if ok else None

    def fits(spec):
        return time.time() + spec.get('est_seconds', 0) + a.start_margin <= deadline

    while True:
        recover_stale(q)
        if not stop and time.time() < deadline and free_slices:
            pending = []
            for f in (q / 'pending').glob('*.json'):
                try:
                    pending.append((load(f), f))
                except (OSError, ValueError):
                    continue
            pending.sort(key=lambda t: (-t[0].get('priority', 0), -t[0].get('est_seconds', 0), t[0]['id']))
            for spec, src in pending:
                if not free_slices:
                    break
                if not fits(spec) or int(spec.get('cpus', slice_cpus)) > slice_cpus:
                    continue
                gpu = pick_gpu(share(spec))
                if gpu is None:
                    continue
                dst = q / 'running' / src.name
                try:
                    src.rename(dst)                 # atomic claim
                except FileNotFoundError:
                    continue                        # another job took it
                spec.update(owner_job=job, owner_host=socket.gethostname(), lease_deadline=deadline + 60)
                save(dst, spec)
                k = free_slices.pop(0)
                gpu_used[gpu] += share(spec)
                ref = run_task.remote(spec, str(q / 'work' / spec['id']), deadline, gpu, slices[k])
                inflight[ref] = (spec, dst, gpu, k)
                print(f"[{time.time()-t0:7.0f}s] start {spec['id']} (est {spec.get('est_seconds', 0):.0f}s)", flush=True)
        if not inflight:
            pending = list((q / 'pending').glob('*.json'))
            if stop or time.time() >= deadline or not pending:
                break
            if not any(fits(load(f)) for f in pending if f.exists()):
                break
        ready, _ = ray.wait(list(inflight), num_returns=1, timeout=15) if inflight else ([], [])
        if not inflight:
            time.sleep(15)
        for ref in ready:
            spec, path, gpu, k = inflight.pop(ref)
            gpu_used[gpu] -= share(spec)
            free_slices.append(k)
            free_slices.sort()
            try:
                result = ray.get(ref)
            except Exception as exc:   # worker died
                result = dict(status='interrupted', error=repr(exc))
            spec.setdefault('attempts', []).append(dict(job=job, **result))
            spec.pop('lease_deadline', None)
            dest = {'done': 'done', 'failed': 'failed'}.get(result['status'], 'pending')
            save(q / dest / path.name, spec)
            path.unlink(missing_ok=True)
            print(f"[{time.time()-t0:7.0f}s] {result['status']} {spec['id']} "
                  f"({result.get('seconds', 0):.0f}s, exit {result.get('exit_code')})", flush=True)
        if stop or time.time() >= deadline + 30:
            break

    for ref, (spec, path, gpu, k) in inflight.items():
        ray.cancel(ref, force=True)
        spec.setdefault('attempts', []).append(dict(job=job, status='interrupted', reason='job deadline'))
        spec.pop('lease_deadline', None)
        save(q / 'pending' / path.name, spec)
        path.unlink(missing_ok=True)
        print(f"[{time.time()-t0:7.0f}s] requeued {spec['id']}", flush=True)
    ray.shutdown()
    print(f"job {job} exits: done={len(list((q/'done').glob('*.json')))} failed={len(list((q/'failed').glob('*.json')))} "
          f"pending={len(list((q/'pending').glob('*.json')))} running={len(list((q/'running').glob('*.json')))}")


if __name__ == '__main__':
    main()
