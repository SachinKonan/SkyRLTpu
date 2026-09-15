"""Read-only, per-host gate before downloading code or starting workloads."""
import json
import os
from pathlib import Path
import shutil
import subprocess

home = Path.home()
storage = {}
for name, path, minimum in [('home', home, 30), ('tmp', Path('/tmp'), 10)]:
    usage = shutil.disk_usage(path)
    inode = os.statvfs(path)
    storage[name] = dict(free_gib=round(usage.free / 1024**3, 2), free_inodes=inode.f_favail)
    if usage.free < minimum * 1024**3 or inode.f_favail < 10000:
        raise RuntimeError(f'Insufficient storage on {name}: {storage[name]}')

devices = [str(p) for p in Path('/dev/vfio').glob('*') if p.name != 'vfio']
devices += [str(p) for p in Path('/dev').glob('accel*')]
if not devices:
    raise RuntimeError('No TPU devices found')
result = subprocess.run(['sudo', '-n', 'fuser', *devices], capture_output=True, text=True)
if result.returncode != 1 or result.stdout.strip() or result.stderr.strip():
    raise RuntimeError('TPU device already owned or audit failed: ' + result.stdout.strip() + result.stderr.strip())

modules = {'skyrl.tinker.api', 'skyrl.tinker.engine', 'skyrl.backends.rpc',
           'tpu.thinking_budget.server', 'tpu.swarm.ray_train.thinking_budget.server',
           'tpu.swarm.ray_train.grader_child', 'tpu.swarm.ray_train.bootstrap',
           'tpu.science.placement_task', 'tpu.science.challenge_score_child',
           'tpu.science.worker'}
suffixes = ('/vllm_tpu_server.py', '/thinking_budget/server.py', '/grader_child.py', '/runner.py')
strays = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        args = proc.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
    except FileNotFoundError:
        continue
    except PermissionError:
        continue  # TPU ownership is independently inspected as root above.
    matches = [arg for arg in args if arg in modules or arg.endswith(suffixes) or arg.startswith('VLLM::')]
    if matches:
        strays.append(dict(pid=int(proc.name), matched=matches))
if strays:
    raise RuntimeError('Existing workload processes; refusing to start: ' + json.dumps(strays))
units = subprocess.run(['systemctl', 'list-units', '--no-legend', '--plain',
                        '--state=active,activating,deactivating', 'placement-grade-*', 'science-grade-*'],
                       capture_output=True, text=True, timeout=10)
if units.returncode or units.stdout.strip():
    raise RuntimeError('Existing grading units or unit audit failed: ' + units.stdout.strip() + units.stderr.strip())
mem = {key: int(value.split()[0]) for key, value in
       (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
print(json.dumps(dict(event='native_sweep_clean_host_audit',
                     rank=os.environ.get('SKYPILOT_NODE_RANK'),
                     storage=storage, mem_available_gib=round(mem['MemAvailable'] / 1024**2, 2),
                     devices=devices, owners=[], stray_workloads=[])), flush=True)
