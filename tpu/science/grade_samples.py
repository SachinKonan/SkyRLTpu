"""Grade completed native sample groups as soon as each task's answers arrive."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import ray
from .ray_cpu import grade
from .rewards import invalid


async def task(args,name):
    folder=Path(args.samples)/name;output=Path(args.output)/name
    output.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+7200
    while True:
        try:summary=json.loads((folder/'summary.json').read_text());break
        except (FileNotFoundError,json.JSONDecodeError):
            if time.monotonic()>deadline:raise TimeoutError(f'{name}: sample group did not finish')
            await asyncio.sleep(5)
    (output/'generation-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    async def one(row):
        index=row['index'];source_file=folder/f'candidate-{index:03d}.py'
        if not row['has_code']:
            result=invalid(row.get('format_error') or 'no complete answer code',phase='format')
            source_sha256=None
        else:
            source=source_file.read_text();source_sha256=hashlib.sha256(source.encode()).hexdigest()
            (output/source_file.name).write_text(source)
            ref=grade.options(scheduling_strategy='SPREAD').remote(name,source,args.root)
            try:result=await asyncio.wait_for(asyncio.wrap_future(ref.future()),timeout=7800)
            except Exception as exc:result=invalid(f'{type(exc).__name__}: {exc}',phase='ray')
            finally:ray.cancel(ref,force=False)
        result.update(candidate_index=index,source_sha256=source_sha256,generation=row)
        (output/f'verdict-{index:03d}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(event='graded',task=name,index=index,reward=result['reward'],
            correctness=result['correctness'],msg=result['msg'],metrics={k:v for k,v in result['metrics'].items() if k!='cases'})),flush=True)
        return result
    results=await asyncio.gather(*[one(row) for row in summary['candidates']])
    report=dict(task=name,samples=len(results),valid=sum(r['correctness']==1 for r in results),
        mean_reward=sum(r['reward'] for r in results)/len(results),best_reward=max(r['reward'] for r in results),
        generation_seconds=summary['generation_seconds'])
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report


async def main(args):
    ray.init(address=args.address,namespace=args.namespace,log_to_driver=False,
        runtime_env={'env_vars':{'PYTHONPATH':args.root}})
    try:
        results=await asyncio.gather(*[task(args,name) for name in args.tasks])
        print(json.dumps(dict(event='science_pilot_complete',tasks=results)),flush=True)
    finally:ray.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--address',required=True);p.add_argument('--namespace',required=True)
    p.add_argument('--root',required=True);p.add_argument('--samples',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tasks',nargs='+',choices=['portfolio','routing'],default=['portfolio','routing'])
    asyncio.run(main(p.parse_args()))
