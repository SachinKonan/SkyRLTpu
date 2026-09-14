"""CPU candidate sandbox entrypoint. Pickle is used only inside this namespace."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np
from joblib import parallel_config
from threadpoolctl import threadpool_limits


def load_candidate(source):
    spec=importlib.util.spec_from_file_location('candidate',source)
    candidate=importlib.util.module_from_spec(spec);sys.modules['candidate']=candidate
    spec.loader.exec_module(candidate)
    return candidate


def isolated_action(candidate,model,obs):
    # Each action forks the same frozen model/import state. Candidate changes to
    # globals or estimator state cannot persist to another historical decision.
    read_fd,write_fd=os.pipe();pid=os.fork()
    if pid==0:
        os.close(read_fd)
        try:
            action=np.asarray(candidate.act(model,obs))
            payload=json.dumps({'weights':action.tolist()},allow_nan=False).encode()
            if len(payload)>65536:raise ValueError('oversized action')
            with os.fdopen(write_fd,'wb') as f:f.write(payload)
            os._exit(0)
        except BaseException:
            os._exit(1)
    os.close(write_fd)
    with os.fdopen(read_fd,'rb') as f:payload=f.read(65537)
    _,status=os.waitpid(pid,0)
    if status or len(payload)>65536:raise RuntimeError('candidate action failed')
    return json.loads(payload)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','act']);p.add_argument('--budget',type=float,required=True)
    p.add_argument('--dataframe',action='store_true')
    args=p.parse_args();start=time.monotonic()
    transport=sys.stdout;sys.stdout=sys.stderr
    cpus=int(os.environ['SCIENCE_CPUS'])
    parallel=parallel_config(backend='threading',n_jobs=cpus)
    parallel.__enter__()
    native=threadpool_limits(limits=1)
    native.__enter__()
    candidate=load_candidate('/input/candidate.py')
    # Apply limits again after candidate imports load additional BLAS libraries.
    loaded_native=threadpool_limits(limits=1)
    loaded_native.__enter__()
    if args.mode=='fit':
        if args.dataframe:
            import pandas as pd
            train=pd.read_json('/input/train.json.gz',orient='split')
            valid=pd.read_json('/input/valid.json.gz',orient='split')
            for frame in [train,valid]:
                frame['session']=pd.to_datetime(frame.session)
                frame['decision_at']=pd.to_datetime(frame.decision_at,utc=True)
        else:
            with np.load('/input/train.npz',allow_pickle=False) as f:train=dict(f)
            with np.load('/input/valid.npz',allow_pickle=False) as f:valid=dict(f)
        remaining=args.budget-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError('candidate import exhausted fit allowance')
        model=candidate.fit(train,valid,np.random.default_rng(1),time_budget_s=remaining)
        # This object is never unpickled by a controller or trusted grader.
        with open('/output/model.pkl','wb') as f:pickle.dump(model,f,protocol=5)
        Path('/output/fit.json').write_text(json.dumps(dict(hardware='cpu',fit_seconds=time.monotonic()-start)))
    else:
        # Only returned state crosses between fit and the fresh inference sandbox.
        with open('/model/model.pkl','rb') as f:model=pickle.load(f)
        for line in sys.stdin:
            request=json.loads(line)
            if args.dataframe:
                import pandas as pd
                if set(request)!={'market','current_weights'}:raise ValueError('invalid observation')
                frame=pd.DataFrame(request['market']['data'],columns=request['market']['columns'])
                frame['session']=pd.to_datetime(frame.session)
                frame['decision_at']=pd.to_datetime(frame.decision_at,utc=True)
                obs=dict(market=frame,current_weights=np.asarray(request['current_weights']))
            else:
                obs={k:np.asarray(v) for k,v in request.items()}
                if set(obs)!={'features','current_weights'}:raise ValueError('invalid observation')
            transport.write(json.dumps(isolated_action(candidate,model,obs),allow_nan=False)+'\n');transport.flush()


if __name__=='__main__':main()
