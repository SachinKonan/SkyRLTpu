"""SimpleTES policy compilation, isolated routing, and trusted Qiskit replay."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import signal

from .contracts import rust_literal
from .isolation import Limits, python_mounts, run
from .rewards import qubit, valid
from .cgroup_limits import envelope, metrics as memory_metrics

START,END='// EVOLVE-BLOCK-START','// EVOLVE-BLOCK-END'


def load_verifier(root):
    path=Path(root)/'evaluator.py'
    spec=importlib.util.spec_from_file_location('science_simpletes_verifier',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def scaffold(root, body):
    source=(Path(root)/'rust/router_core/src/candidate.rs').read_text()
    if source.count(START)!=1 or source.count(END)!=1:raise ValueError('ambiguous pinned scaffold')
    if START in body or END in body:raise ValueError('return block contents without marker lines')
    prefix=source.split(START)[0];suffix=source.split(END)[1]
    # Reproducibility guard; isolation remains the security boundary.
    import re
    if re.search(r'\bunsafe\b|\b(?:std\s*::\s*)?(?:fs|process|net)\s*::|\b(?:include|include_str|include_bytes)\s*!',body):
        raise ValueError('unsafe code, external I/O/processes, and include macros are prohibited')
    return prefix+START+'\n'+body+'\n'+END+suffix


def verify(root, suite_path, output, progress=None):
    verifier=load_verifier(root)
    specs=verifier._load_suite_metadata(Path(suite_path))
    rows=output.get('cases')
    if not isinstance(rows,list) or len(rows)!=len(specs):raise ValueError('missing routing cases')
    by_id={r['id']:r for r in rows}
    if len(by_id)!=len(rows) or set(by_id)!=set(specs):raise ValueError('duplicate or unexpected routing cases')
    baseline=[];candidate=[];weights=[];diagnostics=[]
    from qiskit import qasm3
    for name,spec in specs.items():
        row=by_id[name]
        if row.get('ok') is not True:raise ValueError(f'{name}: routing failed')
        error=verifier._validate_case_routing(row,spec)
        if error:raise ValueError(f'{name}: {error}')
        # Do not trust a candidate-modifiable swap_count field. For this suite,
        # original circuits have no SWAPs; count actual validated output operations.
        original=qasm3.loads(spec.qasm3_path.read_text())
        if original.count_ops().get('swap',0):raise ValueError('v1 input circuits must not contain native SWAPs')
        routed=qasm3.loads(row['output_circuit'])
        swaps=int(routed.count_ops().get('swap',0))
        if type(row.get('swap_count')) is not int or row['swap_count']!=swaps:
            raise ValueError(f'{name}: reported SWAP count disagrees with independently parsed circuit')
        baseline.append(spec.original_cnot_added);candidate.append(3*swaps);weights.append(spec.weight)
        diagnostics.append(dict(case=name,swaps=swaps,added_cnots=3*swaps))
        if progress is not None:progress(diagnostics[-1],len(diagnostics))
    reward,metrics=qubit(baseline,candidate,weights)
    metrics.update(cases=diagnostics,case_count=len(rows))
    return reward,metrics


def evaluate(source, *, root, work, python, cargo_home, rustup_home, target_cache, suite=None, seconds=1800, routing_suite='full'):
    memory_group,_=envelope(8)
    root,work=Path(root).resolve(),Path(work).resolve()
    work.mkdir(parents=True,exist_ok=False)
    rust=work/'rust';shutil.copytree(root/'rust',rust,ignore=shutil.ignore_patterns('target','.cargo_target'))
    target=work/'target'
    # A private copy prevents candidate builds from modifying another candidate's cache.
    shutil.copytree(target_cache,target)
    from .routing_suite import select_suite
    suite=select_suite(root, work, routing_suite, suite)
    body=rust_literal(source);(rust/'router_core/src/candidate.rs').write_text(scaffold(root,body))
    started=time.monotonic();times={}
    py_mounts=python_mounts(python)
    rustup=Path(rustup_home).resolve();cargo=Path(cargo_home).resolve()
    env=dict(CARGO_HOME=str(cargo),RUSTUP_HOME=str(rustup),PYO3_PYTHON=str(python),
        CARGO_TARGET_DIR='/work/target',CARGO_NET_OFFLINE='true',
        CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER='/usr/bin/gcc',
        PATH=f'{cargo}/bin:/usr/bin:/bin')
    readonly=py_mounts+[(rustup,rustup)]
    readonly += [(cargo/name,cargo/name) for name in ('bin','registry','git') if (cargo/name).exists()]
    times['build_seconds']=run([str(cargo/'bin/cargo'),'build','--offline','--locked','--release','-j4','-p','router_cli'],
        limits=Limits(min(900,seconds),8),log=work/'build.log',readonly=readonly,writable=[(work,'/work')],env=env,cwd='/work/rust')
    # Inputs mounted separately; build output and grader data are not writable here.
    out=work/'out';out.mkdir()
    base_prefix=py_mounts[-1][0]
    runtime=dict(PYTHONHOME=base_prefix,PYTHONPATH=str(Path(python).parent.parent/'lib/python3.11/site-packages'),
                 QUBIT_ROUTING_LAYOUT_TRIALS='20',QUBIT_ROUTING_ROUTING_TRIALS='20')
    # Suite path uses the same absolute read-only directory as its relative circuits.
    remaining=min(900,seconds)-(time.monotonic()-started)
    if remaining<=0:raise TimeoutError('routing build exhausted budget')
    times['solve_seconds']=run(['/router','--suite',str(suite),'--out','/work/results.json'],
        limits=Limits(remaining,8),log=work/'route.log',readonly=py_mounts+[(root/'python',root/'python'),
        (target/'release/router_cli','/router')]+([] if suite.is_relative_to(root/'python') else [(suite,suite)]),
        writable=[(out,'/work')],env=runtime)
    # O_NOFOLLOW prevents forged output symlinks from crossing the trust boundary.
    fd=os.open(out/'results.json',os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd) as f:output=json.load(f)
    checked=time.monotonic()
    def progress(row,count):
        with (work/'verification-progress.jsonl').open('a') as log:
            log.write(json.dumps(dict(**row,verified_cases=count,seconds=time.monotonic()-checked))+'\n')
    def timeout(*_):raise TimeoutError('independent verification exhausted budget')
    old_handler=signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,max(.01,min(900,seconds-(checked-started))))
    try:reward,metrics=verify(root,suite,output,progress=progress)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
    times['verify_seconds']=time.monotonic()-checked
    times['total_seconds']=time.monotonic()-started
    if times['total_seconds']>seconds:raise TimeoutError('independent verification exhausted budget')
    metrics.update(routing_suite=routing_suite)
    metrics.update(times,seed=42,layout_trials=20,routing_trials=20,hardware='cpu',cpus=4,memory_gib=8,
                   source_sha256=hashlib.sha256(source.encode()).hexdigest(),allocation_memory=memory_metrics(memory_group))
    result=valid(reward,metrics);(work/'verdict.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--root',required=True);p.add_argument('--work',required=True)
    p.add_argument('--target-cache',required=True);p.add_argument('--suite');p.add_argument('--python',default=sys.executable)
    p.add_argument('--routing-suite', choices=('full','q20'), default='full')
    p.add_argument('--cargo-home',default=str(Path.home()/'.cargo'));p.add_argument('--rustup-home',default=str(Path.home()/'.rustup'))
    a=p.parse_args();kwargs=vars(a);source=Path(kwargs.pop('source')).read_text();print(json.dumps(evaluate(source,**kwargs),indent=2))
