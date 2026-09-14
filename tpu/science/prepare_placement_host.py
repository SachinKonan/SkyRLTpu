"""Prepare pinned candidate and trusted grader environments under a pool lease."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path.cwd(); rank=int(os.environ['SKYPILOT_NODE_RANK'])
grading_ranks = [int(r) for r in os.environ.get('PLACEMENT_TPU_RANKS','5,6,7').split(',')]
if rank not in grading_ranks:raise SystemExit(0)
from .placement_slots import chips_from_env, chip_cpus, device_paths
chips = chips_from_env(os.environ)
accelerator = os.environ.get('SCIENCE_ACCELERATOR','tpu-v4-64')
for chip in chips:
    if not set(chip_cpus(chip)) <= os.sched_getaffinity(0):
        raise RuntimeError('placement CPU sets are unavailable')
    device_paths(chip, accelerator)
mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
if int(mem['MemAvailable'].split()[0]) < (16*len(chips)+8)*1024**2:
    raise RuntimeError('insufficient available host RAM for grading slots plus 8 GiB headroom')
uv=str(Path.home()/'.local/bin/uv')
for name,version,lock,extras in [
    ('venv','3.11.13','requirements-challenge-pilot.lock',['--extra-index-url','https://download.pytorch.org/whl/cpu','--index-strategy','unsafe-best-match']),
    ('candidate-venv','3.12.12','requirements-placement-candidate.lock',[])]:
    target=root/'.science'/name; req=root/'tpu/science'/lock
    digest=hashlib.sha256(req.read_bytes()).hexdigest();marker=target/'.placement-lock'
    if not marker.exists():
        subprocess.run([uv,'--no-config','venv','--python',version,str(target)],check=True)
        subprocess.run([uv,'--no-config','pip','sync','--python',str(target/'bin/python'),str(req),*extras],check=True)
        marker.write_text(digest)
    elif marker.read_text()!=digest:raise RuntimeError('environment lock mismatch; use a new runtime directory')
subprocess.run(['bwrap','--version'],check=True)
subprocess.run(['sudo','-n','systemctl','--version'],check=True)
(root/'.science/ready.json').write_text(json.dumps(dict(rank=rank,profile='placement_jax_v1',chips=chips,accelerator=accelerator)))
