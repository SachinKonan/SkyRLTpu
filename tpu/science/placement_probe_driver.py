"""Reference and adversarial checks through the actual Ray task envelope."""
import argparse
import json
from pathlib import Path
import time

import ray
from .placement_ray import PlacementPool, aggregate
from .placement_slots import grading_nodes
from .placement_task import CASES


def main():
    p=argparse.ArgumentParser()
    for k in ('address','namespace','root','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--accelerator',default='tpu-v4-64')
    p.add_argument('--expected-slots',type=int,default=3)
    a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=True)
    ray.init(address=a.address,namespace=a.namespace)
    nodes=grading_nodes(ray.nodes())
    count=sum(n['Resources']['placement_tpu_host'] for n in nodes)
    if count!=a.expected_slots:raise RuntimeError(f'expected {a.expected_slots} chip capacity, found {count}')
    pool=PlacementPool(nodes)
    try:
        run_checks(a,root,out,pool)
    finally:
        pool.close()
        ray.shutdown()


def run_checks(a,root,out,pool):
    pending={}
    for source_name in ('challenge_seed.py','challenge_seed_jax.py'):
        source=(root/'tpu/science'/source_name).read_text()
        for case in CASES:
            ref=pool.submit(source,case,str(root),accelerator=a.accelerator)
            pending[ref]=(source_name,case)
    results=[]
    while pending:
        ready,_=ray.wait(list(pending),num_returns=1,timeout=330)
        if not ready:raise TimeoutError('reference grading stalled')
        ref=ready[0];source,case=pending.pop(ref);row=ray.get(ref)
        row.update(source=source);results.append(row)
        with (out/'references.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(event='placement_reference',source=source,case=case,
                              correctness=row['correctness'],msg=row['msg'],metrics=row['metrics'])),flush=True)
    summaries={s:aggregate([r for r in results if r['source']==s]) for s in ('challenge_seed.py','challenge_seed_jax.py')}
    # A malformed result must earn zero even when compilation and TPU init work.
    bad="def place(problem, seed, *, time_budget_s):\n    return {'positions': [[float('nan'), 0.0]]}\n"
    rejected=ray.get(pool.submit(bad,CASES[0],str(root),accelerator=a.accelerator),timeout=330)
    report=dict(references=summaries,rejection=rejected)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(r['correctness']==1 for r in results), 'reference checks failed'
    assert rejected['reward']==0 and rejected['correctness']==0, 'invalid output was accepted'


if __name__=='__main__':main()
