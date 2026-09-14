"""Verify the enclosing Slurm/systemd CPU task has a hard aggregate memory cap."""
from pathlib import Path


def envelope(memory_gib):
    lines=Path('/proc/self/cgroup').read_text().splitlines()
    path=next((x.split(':',2)[2] for x in lines if x.startswith('0::')),None)
    if path is None:raise RuntimeError('science worker requires cgroup v2 memory accounting')
    root=Path('/sys/fs/cgroup');current=root/path.lstrip('/')
    bounds=[]
    for directory in [current,*current.parents]:
        if not directory.is_relative_to(root):continue
        file=directory/'memory.max'
        if file.exists():
            value=file.read_text().strip()
            if value!='max':bounds.append((int(value),directory))
    if not bounds:raise RuntimeError('no hard cgroup memory limit: launch this CPU task with the Ray executor resource wrapper')
    limit,directory=min(bounds,key=lambda x:x[0])
    if limit>memory_gib*1024**3:raise RuntimeError(f'enclosing task memory cap {limit} exceeds {memory_gib} GiB')
    return directory,limit


def metrics(directory):
    result={}
    for name in ('memory.current','memory.peak'):
        file=directory/name
        if file.exists():result[name.replace('.','_')+'_mib']=int(file.read_text())/1024**2
    return result
