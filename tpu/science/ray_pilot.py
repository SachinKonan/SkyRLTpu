"""Submit reference or generated programs to the workload's CPU Ray executor."""
import argparse
import json
from pathlib import Path
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from .ray_cpu import grade


def main():
    p=argparse.ArgumentParser();p.add_argument('--address',required=True);p.add_argument('--namespace',required=True)
    p.add_argument('--root',required=True);p.add_argument('--task',choices=['portfolio','portfolio_v2','routing'],required=True)
    p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--all-nodes',action='store_true');args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    ray.init(address=args.address,namespace=args.namespace,log_to_driver=False,
        runtime_env={'env_vars':{'PYTHONPATH':args.root}})
    sources=[(s,Path(s).read_text()) for s in args.sources]
    nodes=sorted([n for n in ray.nodes() if n['Alive']],key=lambda n:n['NodeManagerAddress'])
    tasks={}
    for i,(path,source) in enumerate(sources):
        targets=nodes if args.all_nodes else [nodes[i%len(nodes)]]
        for node in targets:
            ref=grade.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node['NodeID'],soft=False)).remote(args.task,source,args.root)
            tasks[ref]=dict(source=path,node_ip=node['NodeManagerAddress'])
    while tasks:
        ready,_=ray.wait(list(tasks),num_returns=1,timeout=30)
        for ref in ready:
            metadata=tasks.pop(ref)
            try:result=ray.get(ref)
            except Exception as exc:result=dict(reward=0,correctness=0,msg=str(exc),metrics={})
            row=dict(**metadata,**result)
            with (output/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    ray.shutdown()


if __name__=='__main__':main()
