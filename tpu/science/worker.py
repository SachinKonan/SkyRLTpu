"""Trusted entrypoint inside a Ray-admitted, systemd-limited CPU task."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def process_identity(pid):
    # comm may contain spaces or parentheses; fields after its final ')' are stable.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]


def watch_owner(pid, identity):
    def watch():
        while True:
            try: alive=process_identity(pid)==identity
            except OSError: alive=False
            if not alive: os._exit(124)  # systemd kills every remaining child in this unit
            time.sleep(1)
    threading.Thread(target=watch,daemon=True).start()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--request',required=True);p.add_argument('--result',required=True)
    p.add_argument('--owner-pid',type=int,required=True);p.add_argument('--owner-start',required=True)
    args=p.parse_args();watch_owner(args.owner_pid,args.owner_start)
    from .rewards import invalid
    from .cgroup_limits import envelope,metrics
    request=json.loads(Path(args.request).read_text())
    root=Path(request['root']);memory_group,_=envelope(8)
    grader_digest=hashlib.sha256(b''.join(p.name.encode()+b'\0'+p.read_bytes()
        for p in sorted((root/'tpu/science').glob('*.py')))).hexdigest()
    lock_digest=hashlib.sha256((root/'tpu/science/requirements-cpu.lock').read_bytes()).hexdigest()
    started=time.monotonic()
    try:
        source=Path(request['source']).read_text()
        if request['task']=='portfolio':
            from .portfolio import evaluate
            result=evaluate(source,data=root/'.science/data/portfolio',work=request['work'])
        elif request['task']=='portfolio_v2':
            from .portfolio_v2_grader import evaluate
            result=evaluate(source,data=root/'.science/data/portfolio-v2-runtime',work=request['work'])
        elif request['task']=='routing':
            from .routing import evaluate
            result=evaluate(source,root=root/'.science/routing-task',work=request['work'],
                python=sys.executable,cargo_home=root/'.science/cargo',rustup_home=root/'.science/rustup',
                target_cache=root/'.science/router-target')
        else:raise ValueError('unknown science task')
    except Exception as exc:
        result=invalid(f'{type(exc).__name__}: {exc}')
    result['metrics'].update(allocation_memory=metrics(memory_group),worker_seconds=time.monotonic()-started,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        grader_sha256=grader_digest,library_lock_sha256=lock_digest)
    Path(args.result).write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')


if __name__=='__main__':main()
