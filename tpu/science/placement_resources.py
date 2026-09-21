"""Opt-in five-minute CPU placement with 32 host-wide case slots."""
from contextlib import ExitStack
import fcntl
import os
import json
from pathlib import Path
import stat
import time

VERSION='cpu300-4g-v1'

def contract():
    return dict(version=VERSION,cpus=4,memory_gib=4,slots=32,search_seconds=300,
                candidate_seconds=310,grading_seconds=180,envelope_seconds=510,seed=42)

def validate(value):
    if value!=contract():raise ValueError('placement runtime contract mismatch')

def partition():
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)<160:raise RuntimeError('32 placement slots require 128 grading CPUs plus 32 service CPUs')
    return cpus[-128:],cpus[:-128]

def acquire(slots,deadline_seconds=2400,root=None):
    if type(slots) is not int or not 1<=slots<=32:raise ValueError('invalid placement slots')
    cpus,_=partition();root=Path(root or f'/tmp/science-cpu-locks-{os.getuid()}')
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
                    for i in range(32):
                        try:probes.append(lock(f'{VERSION}-{i}.lock',fcntl.LOCK_EX))
                        except BlockingIOError:break
                    path=root/(VERSION+'-map.json')
                    if len(probes)==32:path.write_text(json.dumps(cpus))
                    elif json.loads(path.read_text())!=cpus:raise RuntimeError('active placement CPU map differs')
                finally:
                    for f in probes:f.close()
                for i in range(slots):
                    try:f=lock(f'{VERSION}-{i}.lock',fcntl.LOCK_EX)
                    except BlockingIOError:continue
                    held.enter_context(f)
                    return i,cpus[i*4:i*4+4],held
        except BlockingIOError:pass
        except BaseException:held.close();raise
        held.close();time.sleep(.1)
    raise TimeoutError('placement admission timed out')
