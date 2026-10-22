"""Drain a shared file-based work queue with a single-node Ray cluster inside one Slurm job.

Many jobs may drain the same queue at once. A job keeps its GPUs for its whole wall time and
keeps starting tasks as slots free up, as long as a task's expected duration still fits.

Queue layout (QUEUE=<dir>):
  pending/<id>.json   waiting
  running/<id>.json   claimed; carries the owning job and its lease (the job's deadline)
  done/<id>.json      exit code 0          failed/<id>.json   nonzero exit or own timeout
  work/<id>/          logs and outputs of the task

Concurrency (several jobs drain one queue):
  * claim: write the leased spec to a temp file and hard-link it to running/<id>.json (link fails if
    the name exists), then remove pending/<id>.json; if that removal fails another job already
    claimed and finished the task, so the claim is rolled back;
  * a running task whose lease expired (its job was killed) is returned to pending/ by any job, by
    hard-linking a rewritten spec into pending/ and then removing the running file it read;
  * each attempt has its own work/output directory (work/<id>/<attempt>, TASK_ATTEMPT), so a retried
    or duplicated attempt never overwrites another;
  * at its deadline a job kills its tasks' process groups and returns them to pending/; a task
    interrupted MAX_INTERRUPTS times is marked failed instead of looping.
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


# Only these are inherited from the job; everything else a task sees is set explicitly below.
ENV_WHITELIST = ('PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'LANG', 'LC_ALL', 'TERM')


MAX_INTERRUPTS = 3


def tree_rss(pid):
    """Proportional set size (shared pages split, not double counted) of a process tree (bytes)."""
    total, todo, seen = 0, [pid], set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            for line in open(f'/proc/{p}/smaps_rollup'):
                if line.startswith('Pss:'):
                    total += int(line.split()[1]) * 1024
            for t in os.listdir(f'/proc/{p}/task'):
                todo.extend(int(c) for c in open(f'/proc/{p}/task/{t}/children').read().split())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return total


def _child_setup(cores):
    """In the task's process: pin cores and die with the Ray worker that started it."""
    import ctypes
    os.sched_setaffinity(0, cores)
    ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG


@ray.remote(num_cpus=0, max_retries=0)
def run_task(spec, work, deadline, gpu, cores):
    """Run one task on the GPU and cores the driver assigned, in a clean environment.

    Isolation: CPU affinity to the task's own core slice; one visible GPU; a private HOME, TMPDIR
    and compiler/kernel caches inside its work dir (no shared ~/.cache, Triton or Torch extension
    caches between tasks); a memory cap on the whole process tree (spec memory_gb, default 64, the
    challenge's eval_docker/run_eval.sh `--memory 64g`; the judges' machine has 100 GB), enforced by
    a watchdog on the process tree's PSS that kills the task. The task's process group id is written
    to <work>/pgid so the driver can kill the whole tree at its deadline.
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    private = {name: work / 'sandbox' / name for name in ('home', 'tmp', 'cache', 'triton', 'torch_ext')}
    for path in private.values():
        path.mkdir(parents=True, exist_ok=True)
    env = {k: os.environ[k] for k in ENV_WHITELIST if k in os.environ}
    env.update(HOME=str(private['home']), TMPDIR=str(private['tmp']), XDG_CACHE_HOME=str(private['cache']),
               TRITON_CACHE_DIR=str(private['triton']), TORCH_EXTENSIONS_DIR=str(private['torch_ext']),
               PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = str(len(cores))
    env.update({k: str(v) for k, v in spec.get('env', {}).items()})   # method settings win
    env['TASK_WORK'] = str(work)
    env['TASK_ATTEMPT'] = spec.get('attempt', 'single')
    env['CUDA_VISIBLE_DEVICES'] = gpu
    mem_cap = float(spec.get('memory_gb', 64)) * 1024**3
    limit = min(float(spec['timeout_seconds']), deadline - time.time())
    gpu_name = subprocess.run(['nvidia-smi', '-i', gpu, '--query-gpu=name', '--format=csv,noheader'],
                              capture_output=True, text=True).stdout.strip()
    started, peak, reason = time.time(), 0, None
    with open(work / 'stdout.log', 'ab') as log:
        log.write(f"\n=== {time.strftime('%F %T')} host={socket.gethostname()} job={os.environ.get('SLURM_JOB_ID')} "
                  f"gpu={gpu} ({gpu_name}) cores={cores} mem_cap={mem_cap/1024**3:.0f}GB limit={limit:.0f}s ===\n".encode())
        log.flush()
        proc = subprocess.Popen(spec['cmd'], cwd=spec.get('cwd') or str(work), env=env, stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                preexec_fn=lambda: _child_setup(cores))
        (work / 'pgid').write_text(str(proc.pid))
        while proc.poll() is None:
            peak = max(peak, tree_rss(proc.pid))
            if peak > mem_cap:
                reason = 'memory cap exceeded'
            elif time.time() - started > limit:
                reason = 'job deadline' if deadline - time.time() < 5 else 'task timeout'
            if reason:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                break
            time.sleep(1.0)
        code = proc.returncode
        # The job deadline is an infrastructure interruption; the task's own timeout or memory cap is a failure.
        status = 'interrupted' if reason == 'job deadline' else 'failed' if reason or code else 'done'
    return dict(status=status, exit_code=code, reason=reason, seconds=time.time() - started,
                peak_rss_gb=round(peak / 1024**3, 2), host=socket.gethostname(), gpu=gpu_name,
                gpu_index=gpu, cores=cores)


def recover_stale(q, grace=120):
    """Return tasks whose owning job's lease expired (job killed) to pending/."""
    now = time.time()
    for f in (q / 'running').glob('*.json'):
        try:
            st = f.stat()
            spec = load(f)
        except (OSError, ValueError):
            continue
        lease = spec.get('lease_deadline')
        if lease is None:   # files written by older runners: judge by inode change time
            lease = st.st_ctime + 3600
        if lease + grace >= now:
            continue
        spec.setdefault('attempts', []).append(dict(job=spec.get('owner_job'), attempt=spec.get('attempt'),
                                                    status='interrupted', reason='stale lease (job killed)'))
        for k in ('lease_deadline', 'owner_job', 'owner_host', 'attempt'):
            spec.pop(k, None)
        dest = q / ('failed' if interrupts(spec) >= MAX_INTERRUPTS else 'pending') / f.name
        tmp = q / 'running' / f'.recover-{f.stem}-{os.getpid()}.json'
        save(tmp, spec)
        try:
            os.link(tmp, dest)            # fails if another job already returned it
        except FileExistsError:
            tmp.unlink(missing_ok=True)
            continue
        tmp.unlink(missing_ok=True)
        try:
            if f.stat().st_ino == st.st_ino:
                f.unlink()
        except FileNotFoundError:
            pass


def interrupts(spec):
    return sum(1 for x in spec.get('attempts', []) if x.get('status') == 'interrupted')


def claim(q, src, spec, job, deadline):
    """Claim pending/<id>.json for this job; returns the running path or None."""
    if any((q / d / src.name).exists() for d in ('done', 'failed')):
        return None
    n = len(spec.get('attempts', [])) + 1
    spec = dict(spec, owner_job=job, owner_host=socket.gethostname(), lease_deadline=deadline + 60,
                attempt=f'a{n}-{job}')
    dst = q / 'running' / src.name
    tmp = q / 'running' / f'.claim-{src.stem}-{job}-{os.getpid()}.json'
    save(tmp, spec)
    try:
        os.link(tmp, dst)                 # atomic: fails if another job holds the task
    except FileExistsError:
        tmp.unlink(missing_ok=True)
        return None
    tmp.unlink(missing_ok=True)
    try:
        src.unlink()                      # we own it only if the pending file was still there
    except FileNotFoundError:
        try:
            if load(dst).get('attempt') == spec['attempt']:
                dst.unlink()
        except (OSError, ValueError):
            pass
        return None
    return dst, spec


def finish(q, spec, path, result):
    """Record an attempt and move the task to done/, failed/ or back to pending/."""
    attempt = spec.get('attempt')
    spec.setdefault('attempts', []).append(result)
    for k in ('lease_deadline', 'owner_job', 'owner_host', 'attempt'):
        spec.pop(k, None)
    status = result.get('status')
    if status == 'interrupted' and interrupts(spec) >= MAX_INTERRUPTS:
        status = 'failed'
        result['reason'] = f"{result.get('reason')}; interrupted {MAX_INTERRUPTS} times"
    dest = {'done': 'done', 'failed': 'failed'}.get(status, 'pending')
    save(q / dest / path.name, spec)
    try:
        if load(path).get('attempt') == attempt:   # only remove our own claim
            path.unlink()
    except (OSError, ValueError):
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
                got = claim(q, src, spec, job, deadline)
                if got is None:
                    continue
                dst, spec = got
                k = free_slices.pop(0)
                spec['gpu_cotenants_at_start'] = sorted(s2['id'] for s2, _, g2, _ in inflight.values() if g2 == gpu)
                gpu_used[gpu] += share(spec)
                work = q / 'work' / spec['id'] / spec['attempt']
                ref = run_task.remote(spec, str(work), deadline, gpu, slices[k])
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
            finish(q, spec, path, dict(job=job, attempt=spec['attempt'],
                                       gpu_cotenants_at_start=spec.pop('gpu_cotenants_at_start', []), **result))
            print(f"[{time.time()-t0:7.0f}s] {result['status']} {spec['id']} "
                  f"({result.get('seconds', 0):.0f}s, exit {result.get('exit_code')})", flush=True)
        if stop or time.time() >= deadline + 30:
            break

    for ref, (spec, path, gpu, k) in inflight.items():
        pgid_file = q / 'work' / spec['id'] / spec['attempt'] / 'pgid'
        try:
            os.killpg(int(pgid_file.read_text()), signal.SIGKILL)   # the whole task tree, not just the worker
        except (OSError, ValueError):
            pass
        ray.cancel(ref, force=True)
        finish(q, spec, path, dict(job=job, attempt=spec['attempt'], status='interrupted', reason='job deadline',
                                   gpu_cotenants_at_start=spec.pop('gpu_cotenants_at_start', [])))
        print(f"[{time.time()-t0:7.0f}s] requeued {spec['id']}", flush=True)
    ray.shutdown()
    print(f"job {job} exits: done={len(list((q/'done').glob('*.json')))} failed={len(list((q/'failed').glob('*.json')))} "
          f"pending={len(list((q/'pending').glob('*.json')))} running={len(list((q/'running').glob('*.json')))}")


if __name__ == '__main__':
    main()
