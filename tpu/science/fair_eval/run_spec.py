"""Run one task spec (one Slurm array element = one task) and record the outcome.

Spec format is the same as queue_runner.py. The result goes to <plan>/results/<id>.json and the
task's logs to <plan>/work/<id>/stdout.log.

Usage: run_spec.py --plan DIR --index N   (N = line of <plan>/index.txt, i.e. $SLURM_ARRAY_TASK_ID)
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--index', type=int, required=True)
    a = ap.parse_args()
    plan = Path(a.plan)
    tid = (plan / 'index.txt').read_text().split()[a.index]
    spec = json.loads((plan / 'specs' / f'{tid}.json').read_text())
    work = plan / 'work' / tid
    work.mkdir(parents=True, exist_ok=True)
    (plan / 'results').mkdir(exist_ok=True)
    env = dict(os.environ)
    for key in [k for k in env if k.startswith('RAY_RUNTIME_ENV')]:
        env.pop(key)
    env.update({k: str(v) for k, v in spec.get('env', {}).items()})
    env['TASK_WORK'] = str(work)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env.setdefault(key, str(spec.get('cpus', 12)))
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '')
    started = time.time()
    with open(work / 'stdout.log', 'ab') as log:
        log.write(f"\n=== {time.strftime('%F %T')} host={socket.gethostname()} job={os.environ.get('SLURM_JOB_ID')} "
                  f"gpu={gpu} cpu={cpu} limit={spec['timeout_seconds']}s ===\n".encode())
        log.flush()
        proc = subprocess.Popen(spec['cmd'], cwd=spec.get('cwd') or str(work), env=env, stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = proc.wait(timeout=float(spec['timeout_seconds']))
            status = 'done' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code, status = None, 'timeout'
    result = dict(id=tid, status=status, exit_code=code, seconds=time.time() - started,
                  host=socket.gethostname(), gpu=gpu, cpu=cpu, job=os.environ.get('SLURM_JOB_ID'),
                  array_job=os.environ.get('SLURM_ARRAY_JOB_ID'), array_index=a.index)
    (plan / 'results' / f'{tid}.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
