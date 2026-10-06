"""Local background supervisor: one A40, CPU affinity, RSS watchdog, saved status."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import sys

import psutil

root = Path(__file__).resolve().parents[2]
retry = '--run-only' in sys.argv
out = root/'tpu/science/results/placement'/('archgen-a40-device-fix' if retry else 'archgen-a40-once')
if '--output' in sys.argv:
    out = Path(sys.argv[sys.argv.index('--output')+1]).resolve()
out.mkdir(parents=True, exist_ok=True)
cpus = sorted(os.sched_getaffinity(0))[:16]
assert len(cpus) == 16
os.sched_setaffinity(0, cpus)
env = dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES='GPU-9a686fbc-3da4-ffee-0df8-d9fcb490a39f',
           PLACEMENT_GPU_NODE='/dev/nvidia0')
started = time.time()
with (out/'background.log').open('ab', buffering=0) as log:
    command = ([str(root/'.science/venv-cuda/bin/python'), 'tpu/science/archgen_cuda_once.py', '--output', str(out)]
               if retry else ['bash','tpu/science/archgen_a40_once.sh'])
    proc = subprocess.Popen(command, cwd=root,
                            env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    record = {'supervisor_pid':os.getpid(),'worker_pid':proc.pid,'started_unix':started,
              'gpu_index':0,'gpu_uuid':env['CUDA_VISIBLE_DEVICES'],'cpu_affinity':cpus,
              'memory_limit_gib':100,'memory_enforcement':'one-second process-tree RSS watchdog',
              'candidate_timeout_seconds':3600,'setup_and_run_timeout_seconds':14400,
              'cancelled_slurm_job':'13857007','state':'running_setup'}
    (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    peak = 0
    while proc.poll() is None:
        try:
            parent = psutil.Process(proc.pid)
            processes = [parent,*parent.children(recursive=True)]
            rss = 0
            for p in processes:
                try:
                    rss += p.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak,rss)
            if rss > 100*1024**3 or time.time()-started > 14400:
                record['stop_reason'] = 'memory_watchdog' if rss>100*1024**3 else 'overall_timeout'
                for p in reversed(processes):
                    try:p.kill()
                    except psutil.NoSuchProcess:pass
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(1)
    record.update(exit_code=proc.wait(),finished_unix=time.time(),peak_process_tree_rss_bytes=peak)
    record['state'] = 'finished' if record['exit_code']==0 else 'failed'
    (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
