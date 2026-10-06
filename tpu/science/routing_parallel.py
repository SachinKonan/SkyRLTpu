"""Compile once; run four independently limited and verified cases at a time."""
import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .isolation import Limits, python_mounts, run
from .routing_resources import CASE_CPUS, CASE_GIB, CASE_WORKERS, PROGRAM_GIB, CANDIDATE_SECONDS, contract, RoutingInfrastructureError
from .routing import scaffold, verify
from .contracts import rust_literal
from .cgroup_limits import envelope, metrics as memory_metrics
from .rewards import qubit, valid


def cpu_seconds():
    return sum(v.ru_utime + v.ru_stime for v in
               (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))


def partial_metrics(work, *, deadline_exceeded=False):
    """Recover trusted completed checks, even if the coordinator was killed.

    A started case without a verified result is unfinished, not a failed circuit.
    Missing cases never receive fabricated measurements or a partial reward.
    """
    from .routing_suite import manifest
    specs=manifest()['cases']; rows={}; states={}; errors={}
    progress=Path(work)/'case-progress.jsonl'
    if progress.exists():
        for line in progress.read_text().splitlines():
            try: event=json.loads(line)
            except json.JSONDecodeError: continue  # Interrupted last journal append.
            name=event.get('case')
            if event.get('event')=='started': states[name]='unfinished'
            if 'exit_code' in event:
                states[name]='failed';errors[name]=event.get('detail','')[-240:]
            result=event.get('result')
            if result and result.get('correctness')==1:
                verified=result.get('metrics',{}).get('cases',[])
                if len(verified)==1 and verified[0].get('case')==name:
                    rows[name]=verified[0];states[name]='passed'
    statuses=[]
    for spec in specs:
        name=spec['id'];state=states.get(name,'not_started')
        if state=='unfinished' and deadline_exceeded:state='deadline_unfinished'
        statuses.append(dict(case=name,status=state))
    return dict(cases=[rows[s['id']] for s in specs if s['id'] in rows],
        case_statuses=statuses,case_errors=errors,case_count=len(rows),required_case_count=len(specs),
        completed_cases=len(rows),suite_complete=False,routing_suite='full')


def remaining(deadline):
    value = deadline-time.monotonic()
    if value <= 0: raise TimeoutError('shared routing evaluation deadline exhausted')
    return value


def split_suite(original, destination):
    """Preserve case records, seeds and canonical ordering; only resolve paths."""
    original, destination = Path(original).resolve(), Path(destination)
    payload = json.loads(original.read_text()); cases = payload['cases']
    if not cases or len({r['id'] for r in cases}) != len(cases):
        raise ValueError('missing or duplicate suite case IDs')
    destination.mkdir()
    paths = []
    for index, raw in enumerate(cases):
        row = dict(raw)
        for key in ('qasm3_path', 'topology_path'):
            row[key] = str((original.parent / row[key]).resolve())
        target = destination / f'{index:03d}.json'
        target.write_text(json.dumps(dict(payload, cases=[row])))
        paths.append((raw['id'], target))
    return paths


def verified_case(name, result):
    if result.get('correctness') != 1 or len(result.get('metrics',{}).get('cases',[])) != 1:
        raise ValueError('invalid case result')
    row=result['metrics']['cases'][0]
    if row['case'] != name:raise ValueError('case result ID mismatch')
    return row


def aggregate(expected, results):
    if len(results) != len(expected) or set(results) != set(expected):
        raise ValueError('incomplete verified case set')
    cases = []
    for name in expected:
        cases.append(verified_case(name, results[name]))
    reward, metrics = qubit([r['baseline_added_cnots'] for r in cases],
                            [r['added_cnots'] for r in cases], [r['weight'] for r in cases])
    metrics.update(cases=cases, case_count=len(cases))
    return valid(reward, metrics)


class CaseGroups:
    """Child cgroups enforce aggregate limits for verifier plus router descendants."""
    def __init__(self, cpus):
        if len(cpus) != 10: raise RuntimeError('routing program requires ten CPUs')
        path = next(v.split(':', 2)[2] for v in Path('/proc/self/cgroup').read_text().splitlines()
                    if v.startswith('0::'))
        self.root = Path('/sys/fs/cgroup') / path.lstrip('/')
        if 'science-grade-' not in str(self.root) and 'routing-benchmark-' not in str(self.root):
            raise RuntimeError('case groups require an owned delegated grading service')
        self.cpus, self.groups = cpus, []
        self.coord = self.root / 'coordinator'; self.coord.mkdir()
        self.write(self.coord / 'cgroup.procs', str(os.getpid()))
        self.write(self.root / 'cgroup.subtree_control', '+cpu +cpuset +memory +pids')
        # Establish the hierarchy before spawning compiler/bwrap descendants.
        # They are born in this leaf, so their exit cannot keep the service root
        # populated while domain controllers are being enabled.
        self.write(self.coord / 'cpuset.cpus', ','.join(map(str, cpus)))
        self.write(self.coord / 'cpu.max', '1000000 100000')
        self.write(self.coord / 'memory.max', str(PROGRAM_GIB*1024**3))
        for index in range(CASE_WORKERS):
            group = self.root / f'case-{index}'; group.mkdir()
            self.write(group / 'cpuset.cpus', ','.join(map(str, cpus[index*2:index*2+2])))
            self.write(group / 'cpu.max', '200000 100000')
            self.write(group / 'memory.max', str(CASE_GIB*1024**3))
            self.write(group / 'memory.swap.max', '0')
            self.write(group / 'pids.max', '128')
            self.groups.append(group)

    def write(self, path, value):
        try:
            path.write_text(value)
        except OSError as exc:
            raise RoutingInfrastructureError(
                f'cannot configure {path.name} in {path.parent.name}: {exc}; '
                f'root processes={(self.root / "cgroup.procs").read_text().split()}') from exc

    def begin_cases(self):
        """Release the compilation allowance before any case is admitted."""
        self.write(self.coord / 'cpuset.cpus', ','.join(map(str, self.cpus[8:])))
        self.write(self.coord / 'cpu.max', '200000 100000')
        self.write(self.coord / 'memory.max', str(4*1024**3))

    def kill(self, slot):
        group = self.groups[slot]
        if (group / 'cgroup.kill').exists(): (group / 'cgroup.kill').write_text('1')
        else:
            for path in [group, *group.rglob('*')]:
                if path.is_dir() and (path/'cgroup.procs').exists():
                    for pid in (path/'cgroup.procs').read_text().split():
                        try: os.kill(int(pid), signal.SIGKILL)
                        except ProcessLookupError: pass


def case(request):
    started = time.monotonic(); deadline = request['deadline']; work = Path(request['work'])
    work.mkdir(); out = work/'out';out.mkdir(); root=Path(request['root']); suite=Path(request['suite'])
    mounts=python_mounts(sys.executable)
    route_started=time.monotonic()
    run(['/router','--suite',str(suite),'--out','/work/results.json'],
        limits=Limits(remaining(deadline), CASE_GIB, cpus=CASE_CPUS), log=work/'route.log',
        readonly=mounts+[(root/'python',root/'python'),(Path(request['binary']),'/router'),(suite,suite)],
        writable=[(out,'/work')],env=dict(PYTHONHOME=mounts[-1][0],
        PYTHONPATH=str(Path(sys.executable).parent.parent/'lib/python3.11/site-packages'),
        QUBIT_ROUTING_LAYOUT_TRIALS='20',QUBIT_ROUTING_ROUTING_TRIALS='20',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1'))
    route_seconds=time.monotonic()-route_started
    fd=os.open(out/'results.json',os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd) as f: output=json.load(f)
    remaining(deadline)
    checked=time.monotonic(); _,metrics=verify(root,suite,output,score=False)
    remaining(deadline)
    metrics.update(solve_seconds=route_seconds,verify_seconds=time.monotonic()-checked,
                   total_seconds=time.monotonic()-started,process_cpu_seconds=cpu_seconds())
    result=dict(correctness=1,metrics=metrics)
    (work/'result.json').write_text(json.dumps(result))
    # Verified compact counts persist; release large QASM output promptly.
    (out/'results.json').unlink()
    return result


def _evaluate(source, *, root, work, python, cargo_home, rustup_home, target_cache,
             seconds=CANDIDATE_SECONDS, case_workers=CASE_WORKERS, routing_suite='full', suite=None):
    if case_workers not in (1, CASE_WORKERS): raise ValueError('case workers must be one (paired benchmark) or four')
    if not 0 < seconds <= CANDIDATE_SECONDS: raise ValueError('invalid shared evaluation deadline')
    if routing_suite != 'full': raise ValueError('parallel-v2 requires the full routing suite')
    started=time.monotonic(); deadline=started+seconds; mem,_=envelope(PROGRAM_GIB)
    root,work=Path(root).resolve(),Path(work).resolve(); work.mkdir(parents=True,exist_ok=False)
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)!=10: raise RuntimeError('parallel routing requires an exact ten-CPU program allocation')
    groups=CaseGroups(cpus)
    # Preparation and compilation consume the same deadline as cases.
    rust=work/'rust';shutil.copytree(root/'rust',rust,ignore=shutil.ignore_patterns('target','.cargo_target'))
    target=work/'target';shutil.copytree(target_cache,target);remaining(deadline)
    (rust/'router_core/src/candidate.rs').write_text(scaffold(root,rust_literal(source)))
    from .routing_suite import select_suite,manifest
    original=select_suite(root,work,'full',suite)
    if hashlib.sha256(original.read_bytes()).hexdigest()!=manifest()['suite_sha256']:
        raise ValueError('parallel routing requires the pinned full suite')
    specs=split_suite(original,work/'specs');expected=[name for name,_ in specs]
    if expected!=[c['id'] for c in manifest()['cases']]:
        raise ValueError('parallel routing canonical case order differs from pinned manifest')
    mounts=python_mounts(python);cargo=Path(cargo_home).resolve();rustup=Path(rustup_home).resolve()
    readonly=mounts+[(rustup,rustup)]+[(cargo/n,cargo/n) for n in ('bin','registry','git') if (cargo/n).exists()]
    build_started=time.monotonic()
    build_seconds=run([str(cargo/'bin/cargo'),'build','--offline','--locked','--release','-j8','-p','router_cli'],
        limits=Limits(remaining(deadline), PROGRAM_GIB, cpus=10),log=work/'build.log',
        readonly=readonly,writable=[(work,'/work')],cwd='/work/rust',env=dict(CARGO_HOME=str(cargo),
        RUSTUP_HOME=str(rustup),PYO3_PYTHON=str(python),CARGO_TARGET_DIR='/work/target',
        CARGO_NET_OFFLINE='true',CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER='/usr/bin/gcc',PATH=f'{cargo}/bin:/usr/bin:/bin'))
    groups.begin_cases(); active={}; results={}; index=0; peaks={}; progress=work/'case-progress.jsonl'
    old_handler=signal.getsignal(signal.SIGTERM)
    def terminate(*_): raise InterruptedError('routing coordinator terminated')
    signal.signal(signal.SIGTERM,terminate)
    try:
        while index<len(specs) or active:
            remaining(deadline)
            for slot in range(case_workers):
                if slot in active or index>=len(specs): continue
                name,path=specs[index];folder=work/f'case-{index:03d}';request=dict(root=str(root),suite=str(path),
                    binary=str(target/'release/router_cli'),work=str(folder),deadline=deadline)
                rp=work/f'request-{index:03d}.json';rp.write_text(json.dumps(request));index+=1
                group=groups.groups[slot]
                with progress.open('a') as f:
                    f.write(json.dumps(dict(case=name,event='started',elapsed=time.monotonic()-started))+'\n')
                def enter(group=group, slot=slot):
                    (group/'cgroup.procs').write_text(str(os.getpid()))
                    os.sched_setaffinity(0,cpus[slot*2:slot*2+2])
                log=(work/f'case-{index-1:03d}.log').open('wb')
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                try:proc=subprocess.Popen([str(python),'-m',__name__,'--request',str(rp)],
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=enter,env=env)
                except BaseException: log.close();raise
                active[slot]=(name,folder,proc,log)
            for slot,(name,folder,proc,log) in list(active.items()):
                if proc.poll() is None: continue
                log.close();code=proc.returncode
                if code:
                    failure=dict(case=name,exit_code=code,elapsed=time.monotonic()-started,
                        memory=memory_metrics(groups.groups[slot]),
                        detail=(work/(folder.name+'.log')).read_text(errors='replace')[-4000:])
                    with progress.open('a') as f:f.write(json.dumps(failure)+'\n')
                    raise RuntimeError(f'{name}: case worker exited {code}; '+failure['detail'][-1500:])
                row=json.loads((folder/'result.json').read_text());results[name]=row
                peaks[slot]=memory_metrics(groups.groups[slot])
                with progress.open('a') as f:f.write(json.dumps(dict(case=name,result=row,elapsed=time.monotonic()-started))+'\n')
                del active[slot]
                # Reject immediately; never aggregate a verified subset.
                verified_case(name,row)
            if active:time.sleep(min(.05,remaining(deadline)))
        result=aggregate(expected,results);remaining(deadline)
        result['metrics'].update(routing_suite='full',resource_contract=contract(),case_workers=case_workers,
            build_seconds=build_seconds,prepare_seconds=build_started-started,total_seconds=time.monotonic()-started,
            solve_seconds=sum(r['metrics']['solve_seconds'] for r in results.values()),
            verify_seconds=sum(r['metrics']['verify_seconds'] for r in results.values()),
            process_cpu_seconds=cpu_seconds(),
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),allocation_memory=memory_metrics(mem),case_memory=peaks,
            layout_trials=20,routing_trials=20,hardware='cpu')
        (work/'verdict.json').write_text(json.dumps(result,indent=2)+'\n');return result
    except BaseException as exc:
        (work/'failure.json').write_text(json.dumps(dict(error=type(exc).__name__,detail=str(exc),
            elapsed=time.monotonic()-started,completed_cases=list(results),expected_cases=expected)))
        raise
    finally:
        for slot in range(CASE_WORKERS):groups.kill(slot)
        for _,_,proc,log in active.values():
            proc.wait(timeout=10);log.close()
        signal.signal(signal.SIGTERM,old_handler)


def evaluate(source, **kwargs):
    # Interrupt preparation, compilation and trusted verification as well as
    # subprocess waits. Cleanup is outside the candidate deadline.
    seconds = kwargs.get('seconds', CANDIDATE_SECONDS)
    if not 0 < seconds <= CANDIDATE_SECONDS:
        raise ValueError('invalid shared evaluation deadline')
    old = signal.getsignal(signal.SIGALRM)
    def expire(*_):
        raise TimeoutError('shared routing evaluation deadline exhausted')
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return _evaluate(source, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
    case(json.loads(Path(a.request).read_text()))
