"""Isolated dataframe fit and close-time causal replay on a CPU worker."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

from .cgroup_limits import envelope,metrics as memory_metrics
from .isolation import Limits,Session,python_mounts
from .portfolio import check_imports
from .portfolio_v2_accounting import replay_v2
from .resources import library_environment
from .rewards import valid


def evaluate(source,*,data,work,python=sys.executable,cpus=4,memory_gib=8):
    check_imports(source)
    group,limit=envelope(memory_gib)
    data=Path(data).resolve();work=Path(work).resolve();work.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest['version']!='portfolio_v2_runtime_001' or manifest['hardware']!='cpu':raise ValueError('wrong runtime manifest')
    started=time.monotonic()
    required=['train.json.gz','valid.json.gz','discovery_observations.json.gz','discovery_market.npz']
    for name in required:
        if hashlib.sha256((data/name).read_bytes()).hexdigest()!=manifest['files'][name]['sha256']:raise ValueError('runtime checksum mismatch')
    inputs=work/'input';inputs.mkdir();(inputs/'candidate.py').write_text(source)
    for name in required[:2]:shutil.copyfile(data/name,inputs/name)
    model=work/'model';model.mkdir()
    runner=Path(__file__).with_name('portfolio_child.py').resolve()
    env=library_environment(cpus)
    fit_start=time.monotonic()
    session=Session([python,'/runner.py','fit','--budget','180','--dataframe'],
        limits=Limits(180,memory_gib,cpus),log=work/'fit.log',
        readonly=python_mounts(python)+[(runner,'/runner.py'),(inputs,'/input')],
        writable=[(model,'/output')],env=env)
    try:session.finish()
    finally:session.close()
    fit_seconds=time.monotonic()-fit_start
    model_file=model/'model.pkl'
    if model_file.is_symlink() or not model_file.is_file() or model_file.stat().st_size>128*1024**2:
        raise ValueError('invalid or oversized serialized model')
    frozen=work/'frozen';frozen.mkdir();shutil.copyfile(model_file,frozen/'model.pkl')
    infer_inputs=work/'inference-input';infer_inputs.mkdir();shutil.copyfile(inputs/'candidate.py',infer_inputs/'candidate.py')
    frames=pd.read_json(data/'discovery_observations.json.gz',orient='split')
    frames['session']=pd.to_datetime(frames.session)
    frames['decision_at']=pd.to_datetime(frames.decision_at,utc=True)
    observations=[f.reset_index(drop=True) for _,f in frames.groupby('session',sort=True)]
    if any(f.asset_id.tolist()!=manifest['assets'] for f in observations):raise ValueError('asset order mismatch')
    with np.load(data/'discovery_market.npz',allow_pickle=False) as f:market=dict(f)
    if len(observations)!=manifest['evaluation_days']:raise ValueError('evaluation length mismatch')
    inference_start=time.monotonic()
    session=Session([python,'/runner.py','act','--budget','60','--dataframe'],
        limits=Limits(60,memory_gib,cpus),log=work/'inference.log',
        readonly=python_mounts(python)+[(runner,'/runner.py'),(infer_inputs,'/input'),(frozen,'/model')],env=env)
    try:
        def act(obs):
            frame=json.loads(obs['market'].to_json(orient='split',date_format='iso',double_precision=15,index=False))
            session.send(dict(market=frame,current_weights=obs['current_weights'].tolist()))
            reply=session.receive()
            if not isinstance(reply,dict) or set(reply)!={'weights'}:raise ValueError('invalid action response')
            return reply['weights']
        reward,metrics,trace=replay_v2(observations,market,act,manifest['fee_rate'])
        session.finish()
    finally:session.close()
    metrics.update(hardware='cpu',cpus=cpus,memory_gib=memory_gib,fit_seconds=fit_seconds,
        inference_seconds=time.monotonic()-inference_start,total_seconds=time.monotonic()-started,
        allocation_memory=memory_metrics(group),memory_limit_bytes=limit,
        model_bytes=model_file.stat().st_size,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        runtime_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),seed=1,
        data_version=manifest['version'],evaluation_period='2025',holdings_observed_at='decision_close')
    if metrics['total_seconds']>300:raise TimeoutError('overall portfolio budget exhausted')
    result=valid(reward,metrics)
    (work/'verdict.json').write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')
    (work/'trace.json').write_text(json.dumps(trace,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--data',required=True);p.add_argument('--work',required=True)
    a=p.parse_args();print(json.dumps(evaluate(Path(a.source).read_text(),data=a.data,work=a.work),indent=2))
