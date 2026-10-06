"""Opt-in five-minute CPU placement with configurable host-wide case slots."""
from contextlib import ExitStack
import fcntl
import os
import json
from pathlib import Path
import stat
import time

VERSION='cpu300-4g-v1'

def contract(slots=32, *, cpus=4, memory_gib=4):
    if type(cpus) is not int or cpus not in (2, 4):
        raise ValueError("placement case CPUs must be 2 or 4")
    if type(memory_gib) is not int or memory_gib not in (2, 4):
        raise ValueError("placement case memory must be 2 or 4 GiB")
    if type(slots) is not int or not 1 <= slots <= 192 // cpus:
        raise ValueError("placement slots exceed 192 grading CPUs")
    return dict(version=VERSION,cpus=cpus,memory_gib=memory_gib,slots=slots,search_seconds=300,
                candidate_seconds=310,grading_seconds=180,envelope_seconds=510,seed=42)

def validate(value):
    if value!=contract(value.get("slots", 32), cpus=value.get("cpus", 4), memory_gib=value.get("memory_gib", 4)):raise ValueError('placement runtime contract mismatch')

def partition(slots=32, *, cpus_per_case=4):
    contract(slots, cpus=cpus_per_case)
    cpus=sorted(os.sched_getaffinity(0))
    count = cpus_per_case * slots
    if len(cpus)<count+32:raise RuntimeError('placement slots require four CPUs each plus 32 service CPUs')
    return cpus[-count:],cpus[:-count]

def acquire(slots,deadline_seconds=2400,root=None, *, cpus_per_case=4):
    contract(slots, cpus=cpus_per_case)
    cpus,_=partition(slots, cpus_per_case=cpus_per_case);root=Path(root or f'/tmp/science-cpu-locks-{os.getuid()}')
    root.mkdir(mode=0o700,exist_ok=True);info=root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077:
        raise RuntimeError('grading locks must be private and owned by this user')
    def lock(name,mode):
        fd=os.open(root/name,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);f=os.fdopen(fd,'r+')
        i=os.fstat(fd)
        if not stat.S_ISREG(i.st_mode) or i.st_uid!=os.getuid() or i.st_nlink!=1:
            f.close();raise RuntimeError('invalid admission lock')
        try:fcntl.flock(fd,mode|fcntl.LOCK_NB)
        except BaseException:f.close();raise
        return f
    end=time.monotonic()+deadline_seconds
    while time.monotonic()<end:
        held=ExitStack()
        try:
            for i in range(16):held.enter_context(lock(f'cpu-slot-{i}.lock',fcntl.LOCK_SH))
            for i in range(10):held.enter_context(lock(f'parallel-v2-{i}.lock',fcntl.LOCK_SH))
            with lock(VERSION+'-map.lock',fcntl.LOCK_EX):
                probes=[]
                try:
                    for i in range(96):
                        try:probes.append(lock(f'{VERSION}-{i}.lock',fcntl.LOCK_EX))
                        except BlockingIOError:break
                    path=root/(VERSION+'-map.json')
                    layout = cpus if cpus_per_case == 4 else dict(cpus=cpus, cpus_per_case=cpus_per_case)
                    if len(probes)==96:path.write_text(json.dumps(layout))
                    elif json.loads(path.read_text())!=layout:raise RuntimeError('active placement CPU map differs; drain all cases before changing resources')
                finally:
                    for f in probes:f.close()
                for i in range(slots):
                    try:f=lock(f'{VERSION}-{i}.lock',fcntl.LOCK_EX)
                    except BlockingIOError:continue
                    held.enter_context(f)
                    return i,cpus[i*cpus_per_case:(i+1)*cpus_per_case],held
        except BlockingIOError:pass
        except BaseException:held.close();raise
        held.close();time.sleep(.1)
    raise TimeoutError('placement admission timed out')
