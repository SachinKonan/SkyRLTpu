"""Bounded reference, native sampling, and Ray grading for one model pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import ray
import httpx
from .placement_ray import PlacementPool, aggregate
from .placement_slots import grading_nodes, chips_from_env
from .placement_task import CASES
from .rewards import invalid
from tpu.swarm.ray_train.config import Config


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',required=True)
    a=p.parse_args();config=Config.load(a.profile)
    root=Path.cwd();native=Path(config.root).expanduser();ips=os.environ['SKYPILOT_NODE_IPS'].split()
    out=root/('results-'+config.run_id);out.mkdir(exist_ok=False)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    status=None;failure=None;pool=None
    try:
        ray.init(address=ips[0]+':19679',namespace=config.run_id)
        deadline=time.monotonic()+7200
        while time.monotonic()<deadline:
            try:status=ray.get_actor('runtime-status',namespace=config.run_id)
            except ValueError:time.sleep(5);continue
            nodes=grading_nodes(ray.nodes())
            if (len(nodes)==len(config.placement_ranks) and
                all(n['Resources']['placement_tpu_host']==len(chips_from_env(config.client_env)) for n in nodes)):break
            time.sleep(5)
        else:raise TimeoutError('placement Ray hosts did not register')
        pool=PlacementPool(nodes)
        seed=(root/'tpu/science/challenge_seed_jax.py').read_text()
        rows=ray.get([pool.submit(seed,c,str(root),accelerator=config.accelerator)
                      for c in CASES],timeout=1200)
        reference=aggregate(rows)
        (out/'reference.json').write_text(json.dumps(reference,indent=2))
        if reference['correctness']!=1:raise RuntimeError('JAX reference failed on this slice')
        snapshot=None
        while time.monotonic()<deadline:
            log=native/'runs'/config.run_id/'host-1.jsonl'
            if log.exists():
                for line in log.read_text().splitlines():
                    try:r=json.loads(line)
                    except ValueError:continue
                    if r.get('event')=='host_prepared' and r.get('role')=='inference':snapshot=r['snapshot']
            ready=False
            try:
                catalog=ray.get_actor('inference-catalog',namespace=config.run_id)
                state=ray.get(catalog.snapshot.remote(),timeout=10)
                ready=(len(state['replicas'])==config.inference_hosts*config.engines_per_host and httpx.get('http://'+ips[0]+':19800/health',timeout=5).status_code==200)
            except (ValueError, ray.exceptions.RayError, OSError, httpx.HTTPError):pass
            if snapshot and ready:break
            time.sleep(10)
        else:raise TimeoutError('native inference service did not become ready')
        cmd=[str(native/'envs/serving/bin/python'),'-m','tpu.science.sample','--profile',a.profile,
             '--tokenizer',snapshot,'--prompt',str(root/'tpu/science/prompts'/('placement-jax-v5p.txt' if config.accelerator=='tpu-v5p-32' else 'placement-jax.txt')),
             '--base','http://'+ips[0]+':19800','--output',str(out/'samples'),'--samples','4']
        with (out/'sampling.log').open('wb') as log:
            subprocess.run(cmd,check=True,timeout=11000,env=env,stdout=log,stderr=subprocess.STDOUT)
        summary=json.loads((out/'samples/summary.json').read_text());pending={};verdicts={}
        for row in summary['candidates']:
            index=row['index']
            if not row['has_code']:
                verdicts[index]=invalid(row['format_error'] or 'missing code',phase='format');continue
            source=(out/'samples'/f'candidate-{index:03d}.py').read_text()
            for c in CASES:
                pending[pool.submit(source,c,str(root),
                                    accelerator=config.accelerator)]=(index,c)
        case_rows={}
        while pending:
            ready,_=ray.wait(list(pending),num_returns=1,timeout=330)
            if not ready:raise TimeoutError('candidate grading stalled')
            ref=ready[0];index,case=pending.pop(ref);row=ray.get(ref)
            case_rows.setdefault(index,[]).append(row)
            with (out/'case-results.jsonl').open('a') as f:f.write(json.dumps(dict(candidate=index,case=case,result=row))+'\n')
            if len(case_rows[index])==len(CASES):
                verdicts[index]=aggregate(case_rows[index])
                (out/f'candidate-{index:03d}-verdict.json').write_text(json.dumps(verdicts[index],indent=2))
                print(json.dumps(dict(event='placement_candidate_graded',candidate=index,
                                      reward=verdicts[index]['reward'],msg=verdicts[index]['msg'])),flush=True)
        (out/'summary.json').write_text(json.dumps(dict(run_id=config.run_id,reference=reference,candidates=verdicts),indent=2))
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}'
        (out/'failure.json').write_text(json.dumps(dict(error=failure)))
        raise
    finally:
        # Preserve diagnostics before asking this run's controller to stop.
        try:
            archive=root/(config.run_id+'-results.tar.gz')
            subprocess.run(['tar','-czf',str(archive),'-C',str(root),out.name],check=True,timeout=120)
            subprocess.run(['gcloud','storage','cp',str(archive),os.environ['SCIENCE_RESULT_URI']],check=True,timeout=120)
        finally:
            if pool:pool.close()
            if status:ray.get(status.request_stop.remote(),timeout=30)
            ray.shutdown()


if __name__=='__main__':main()
