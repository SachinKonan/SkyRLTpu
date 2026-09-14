"""Isolated CPU learning and causal historical evaluation; no TPU dependencies."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import time
import numpy as np
from .isolation import Limits,Session,python_mounts
from .portfolio_replay import replay
from .rewards import valid
from .cgroup_limits import envelope, metrics as memory_metrics

from .resources import PORTFOLIO_LIBRARIES, COMPUTATIONAL_STDLIB, library_environment

ALLOWED=set(PORTFOLIO_LIBRARIES+COMPUTATIONAL_STDLIB)


def check_imports(source):
    for node in ast.walk(ast.parse(source)):
        names=[]
        if isinstance(node,ast.Import):names=[a.name for a in node.names]
        elif isinstance(node,ast.ImportFrom):
            if node.level:raise ValueError('relative candidate imports are not allowed')
            names=[node.module or '']
        if any(n.split('.')[0] not in ALLOWED for n in names):raise ValueError('candidate import outside CPU allowlist: '+','.join(names))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('__import__','eval','exec','compile','open'):
            raise ValueError('dynamic imports/code and filesystem access are prohibited')


def evaluate(source, *, data, work, python=sys.executable, cpus=4,memory_gib=8):
    check_imports(source)
    memory_group,memory_limit=envelope(memory_gib)
    data=Path(data).resolve();work=Path(work).resolve();work.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest['hardware']!='cpu':raise ValueError('CPU manifest required')
    started=time.monotonic()
    inputs=work/'input';inputs.mkdir();(inputs/'candidate.py').write_text(source)
    for split in ('train','valid','discovery'):
        expected=manifest['splits'][split]['sha256']
        if hashlib.sha256((data/(split+'.npz')).read_bytes()).hexdigest()!=expected:raise RuntimeError('dataset checksum mismatch')
    for split in ('train','valid'):shutil.copyfile(data/(split+'.npz'),inputs/(split+'.npz'))
    model=work/'model';model.mkdir()
    # The child runner is the sole trusted script exposed; it has no scorer or labels.
    runner=Path(__file__).with_name('portfolio_child.py').resolve()
    readonly=python_mounts(python)+[(runner,'/runner.py'),(inputs,'/input')]
    env=library_environment(cpus)
    fit_start=time.monotonic()
    session=Session([str(python),'/runner.py','fit','--budget','180'],limits=Limits(180,memory_gib,cpus),
        log=work/'fit.log',readonly=readonly,writable=[(model,'/output')],env=env)
    try:fit_stats=session.finish()
    finally:session.close()
    fit_seconds=time.monotonic()-fit_start
    model_file=model/'model.pkl'
    if model_file.is_symlink() or not model_file.is_file() or model_file.stat().st_size>128*1024**2:
        raise ValueError('invalid or oversized serialized model')
    # Copy opaque bytes into a new read-only directory; never unpickle in grader.
    frozen=work/'frozen';frozen.mkdir();shutil.copyfile(model_file,frozen/'model.pkl')
    infer_inputs=work/'inference-input';infer_inputs.mkdir();shutil.copyfile(inputs/'candidate.py',infer_inputs/'candidate.py')
    inference_start=time.monotonic()
    session=Session([str(python),'/runner.py','act','--budget','60'],limits=Limits(60,memory_gib,cpus),
        log=work/'inference.log',readonly=python_mounts(python)+[(runner,'/runner.py'),(infer_inputs,'/input'),(frozen,'/model')],env=env)
    try:
        def act(obs):
            session.send({k:v.tolist() for k,v in obs.items()})
            response=session.receive()
            if not isinstance(response,dict) or set(response)!={'weights'}:raise ValueError('invalid action response')
            return response['weights']
        with np.load(data/'discovery.npz',allow_pickle=False) as f:market=dict(f)
        reward,metrics,trace=replay(market,act,manifest['fee_rate'])
        inference_stats=session.finish()
    finally:session.close()
    metrics.update(hardware='cpu',cpus=cpus,memory_gib=memory_gib,fit_seconds=fit_seconds,
        inference_seconds=time.monotonic()-inference_start,total_seconds=time.monotonic()-started,
        memory_enforcement='cgroup_v2',memory_limit_bytes=memory_limit,
        allocation_memory=memory_metrics(memory_group),
        model_bytes=model_file.stat().st_size,source_sha256=hashlib.sha256(source.encode()).hexdigest(),seed=1)
    if metrics['total_seconds']>300:raise TimeoutError('overall portfolio budget exhausted')
    result=valid(reward,metrics)
    (work/'verdict.json').write_text(json.dumps(result,indent=2)+'\n')
    (work/'trace.json').write_text(json.dumps(trace)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--data',required=True);p.add_argument('--work',required=True)
    p.add_argument('--cpus',type=int,default=4);p.add_argument('--memory-gib',type=int,default=8)
    a=p.parse_args();kwargs=vars(a);source=Path(kwargs.pop('source')).read_text();print(json.dumps(evaluate(source,**kwargs),indent=2))
