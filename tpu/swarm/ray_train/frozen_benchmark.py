"""Finite 16x32 native-thinking replay with incremental generation/grading results."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time

from .config import Config
from .arena_sampling import extract_code
from tpu.swarm.bench.realistic_bench import render_prompt, scrape_metrics


def save(path, value):
    data=json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(data);tmp.replace(path)


def event(kind, **fields):
    print(json.dumps(dict(event=kind,time=time.time(),**fields)),flush=True)


def payload(config, render, n=32, warmup=False):
    prompt=len(render.prompt)
    thinking=config.client_phase1_max_tokens-prompt
    total=config.client_context_window-prompt-50
    if thinking<=0 or total<=thinking+128:
        raise ValueError(f'prompt {prompt} does not fit agreed thinking/answer budgets')
    return dict(model=config.model,prompt=render.prompt,n=n,max_tokens=64 if warmup else total,
                thinking_token_budget=8 if warmup else thinking,temperature=1.0,top_p=1.0,
                stop_token_ids=render.stop_ids,skip_special_tokens=False,add_special_tokens=False,
                logprobs=1,stream=False,return_token_ids=True,top_k=-1)


def check_choice(choice, request):
    audit=choice.get('thinking_budget',{})
    if not audit.get('enforced') or audit.get('budget_basis')!='phase1_generated_tokens':
        raise ValueError('missing new native thinking enforcement audit')
    if audit.get('counted_phase1_tokens',math.inf)>request['thinking_token_budget']:
        raise ValueError('native thinking cap exceeded')
    ids=choice.get('token_ids')
    if not isinstance(ids,list) or len(ids)>request['max_tokens']:
        raise ValueError('missing token IDs or context allowance exceeded')
    mask=choice.get('loss_mask')
    if not isinstance(mask,list) or len(mask)!=len(ids):
        raise ValueError('missing native forced-token mask')
    if any(mask[i]!=0 for i in choice.get('forced_token_positions',[])):
        raise ValueError('forced transition tokens must be masked')


def prepare_host(host):
    """Head owns the client; inference hosts remain independent TP4 engines."""
    from .cache import CacheStore, mount_cache
    if host.role == 'inference':
        if host.config.arena_samples:
            raise ValueError('TPU grader must have a dedicated host')
        if host.store is None or host.snapshot is None:
            raise RuntimeError('shared head requires completed inference preparation')
        host.install_client()
        host.phase='frozen_client_ready'
        return host.heartbeat()
    host.source_ready()
    ram=mount_cache(host.root/'ram',host.config.cache.inference_gib,host.config.cache.reserve_gib)
    host.store=CacheStore(ram,host.gcs)
    host.store.scope_compile(host.config.cache.inference_compile)
    host.snapshot=host.store.restore_hf(host.config.cache.hf,host.config.model,weights=False,
                                      layout=host.config.cache.hf_layout)
    host.install_client()
    if host.config.arena_samples:
        from .arena_sampling import prepare_host as prepare_judge
        prepare_judge(host)
    host.phase='frozen_client_ready'
    return host.heartbeat()


def start_host(host):
    from .commands import client_environment
    env=client_environment(host.config,host.root,host.ips[0],trainer_head=host.ips[0])
    package=Path(__file__).resolve().parents[3]
    env['PYTHONPATH']=f'{package}:{package / "tpu"}:{host.source / "third_party/discover"}'
    env['ARENA_QUEUE_URL']='http://127.0.0.1:8791'
    host.start('client',[str(host.root/'envs/client/bin/python'),'-m',
        'tpu.swarm.ray_train.frozen_benchmark','--config-json',json.dumps(host.config.to_dict()),
        '--source',str(host.source),'--snapshot',str(host.snapshot),'--head',host.ips[0]],env,host.root)
    return host.heartbeat()


async def run(config, source, snapshot, head):
    import httpx
    import ray
    from transformers import AutoTokenizer
    from ttt_discover import State
    spec=config.frozen_benchmark
    path=Path(__file__).with_name('frozen_workloads')/(spec['task']+'.json')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=spec['manifest_sha256']:
        raise ValueError('frozen workload checksum mismatch')
    manifest=json.loads(raw)
    parents=manifest['parents'][:spec.get('groups',16)]
    n=spec.get('group_size',32)
    folder=Path(config.root).expanduser()/'runs'/config.run_id/'client/frozen-benchmark'
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/'manifest.json',manifest);save(folder/'config.json',config.to_dict())
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    renderer=config.client_member_spec.split(':')[1]
    requests=[]
    for parent in parents:
        render=render_prompt(tokenizer,renderer,parent['question'],manifest['renderer_date'])
        requests.append(payload(config,render,n))
    save(folder/'requests.json',requests)
    base=f'http://{head}:{config.ports.inference}'
    ray.init(address='auto',namespace=config.run_id)
    evaluators=[]
    if spec['task']!='rglru':
        module,cls,ptype={'erdos':('erdos_min_overlap','ErdosMinOverlapEnv',''),
                         'ac2':('ac_inequalities','AutoCorrInequalityEnv','ac2'),
                         'packing32':('circle_packing','CirclePackingEnv','32')}[spec['task']]
        env_cls=getattr(importlib.import_module(f'examples.{module}.env'),cls)
        # Independent evaluator instances; reward evaluators retain mutable stdout.
        evaluators=[env_cls.reward_function(problem_type=ptype,log_dir=str(folder/f'grader-{i}'),
                    num_cpus_per_task=2,eval_timeout=1100 if spec['task']!='packing32' else 530,
                    eval_backend='ray') for i in range(32)]
    grading_slots=asyncio.Semaphore(32 if evaluators else 8)
    available=asyncio.Queue()
    for e in evaluators:available.put_nowait(e)
    rows=[];grades=[];group_times=[];started=None;generation_seconds=None
    stop=threading.Event()
    async with httpx.AsyncClient(timeout=config.inference.request_timeout) as http:
        before=(await http.get(base+'/status')).json();save(folder/'engines-before.json',before)
        replicas=before.get('replicas',[])
        expected=len(config.inference_only_ranks)
        if len(replicas)!=expected:raise RuntimeError(f'expected {expected} engines; got {len(replicas)}')
        def monitor():
            while not stop.is_set():
                rec={'time':time.time(),'engines':{r['ip']:scrape_metrics(f"http://{r['ip']}:{config.ports.engine}") for r in replicas}}
                with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                stop.wait(15)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        try:
            # Native compatibility is checked before any full 32-choice request.
            for engine_index in range(expected):
                warm=dict(requests[0],n=2,max_tokens=64,thinking_token_budget=8)
                rsp=await http.post(base+'/v1/completions',json=warm);rsp.raise_for_status();data=rsp.json()
                if len(data.get('choices',[]))!=2:raise ValueError('native n=2 preflight incomplete')
                for choice in data['choices']:check_choice(choice,warm)
                save(folder/f'native-preflight-{engine_index}.json',data)
            event('native_preflight_passed',run_id=config.run_id)
            if spec['task']=='rglru':
                from pallas_arena.judge.client import ArenaQueueClient
                from pallas_arena.rl.task import public_contract,translate_verdict
                cases=[k for k,_ in public_contract()[1]]
                queue= ArenaQueueClient('http://127.0.0.1:8791',timeout_s=30)
                def judge_kernel(code,tag):
                    wid=queue.submit('rg_lru',code,mode='full',smoke=False,cases=cases,enforce_pallas=True,tag=tag)
                    verdict=queue.wait([wid],timeout_s=14400).get(wid)
                    if verdict is None:raise TimeoutError(f'judge deadline: {wid}')
                    return {'grade':translate_verdict(verdict),'verdict':verdict,'work_id':wid}
                seed=(Path(__file__).resolve().parents[2]/'pallas_arena/rl/seed_rglru.py').read_text()
                control=await asyncio.to_thread(judge_kernel,seed,'seed-preflight')
                save(folder/'seed-preflight.json',control)
                if control['grade'].get('correctness')!=1:raise RuntimeError('RG-LRU seed grading preflight failed')

            def summary():
                done=[r for r in rows if 'choice' in r]
                graded=[r for r in rows if 'grade' in r]
                valid=[r for r in graded if r['grade'].get('correctness')==1]
                elapsed=time.monotonic()-started if started else 0
                gens=generation_seconds or elapsed
                return dict(run_id=config.run_id,task=spec['task'],model=config.model,groups=len(parents),group_size=n,
                    attempted=len(parents)*n,generated=len(done),groups_generated=len(group_times),graded=len(graded),valid=len(valid),
                    generation_complete=generation_seconds is not None,generation_seconds=gens,elapsed_seconds=elapsed,
                    generation_tokens=sum(len(r['choice']['token_ids']) for r in done),
                    generation_tokens_per_second=sum(len(r['choice']['token_ids']) for r in done)/max(gens,1e-9),
                    groups_per_minute=len(group_times)*60/max(gens,1e-9),valid_per_hour=len(valid)*3600/max(elapsed,1e-9),
                    truncated=sum(r['choice'].get('finish_reason')=='length' for r in done),
                    infrastructure_errors=sum('generation_error'in r or 'grading_error'in r for r in rows),
                    group_latency_median=statistics.median(group_times) if group_times else None,
                    inference_replicas=expected)

            async def grade(row,parent):
                async with grading_slots:
                    start=time.monotonic()
                    try:
                        if spec['task']=='rglru':
                            try:code=extract_code(row['choice']['text'],config.model_preset)
                            except ValueError as exc:
                                row['grade']={'reward':0.0,'correctness':0.0,'msg':str(exc)}
                            else:row.update(await asyncio.to_thread(judge_kernel,code,row['id']))
                        else:
                            e=await available.get()
                            try:row['grade']=await asyncio.to_thread(e.get_reward,row['choice']['text'],State.from_dict(parent['state']))
                            finally:available.put_nowait(e)
                    except Exception as exc:row['grading_error']=f'{type(exc).__name__}: {exc}'
                    row['grading_seconds']=time.monotonic()-start
                    save(folder/(row['id']+'.json'),row);save(folder/'summary.json',summary())
                    event('graded',run_id=config.run_id,id=row['id'],reward=row.get('grade',{}).get('reward'),error=row.get('grading_error'))

            async def group(index,parent,req):
                begin=time.monotonic()
                try:
                    response=await http.post(base+'/v1/completions',json=req);response.raise_for_status();data=response.json()
                    save(folder/f'group-{index:02d}-response.json',data)
                    choices=data.get('choices',[])
                    if len(choices)!=n:raise ValueError(f'expected {n} choices, got {len(choices)}')
                    for choice in choices:check_choice(choice,req)
                    wall=time.monotonic()-begin;group_times.append(wall)
                    for j,choice in enumerate(sorted(choices,key=lambda c:c['index'])):
                        row=dict(id=f'g{index:02d}-s{j:02d}',parent_id=parent['parent_id'],choice=choice,group_seconds=wall)
                        rows.append(row);save(folder/(row['id']+'.json'),row)
                        grades.append(asyncio.create_task(grade(row,parent)))
                    event('group_generated',run_id=config.run_id,group=index,seconds=wall,tokens=sum(len(c['token_ids']) for c in choices))
                except Exception as exc:
                    row=dict(id=f'g{index:02d}-error',generation_error=f'{type(exc).__name__}: {exc}')
                    rows.append(row);save(folder/(row['id']+'.json'),row);event('generation_error',run_id=config.run_id,**row)
                save(folder/'summary.json',summary())

            started=time.monotonic()
            event('sampling_started',run_id=config.run_id,groups=len(parents),group_size=n,engines=expected)
            await asyncio.gather(*(group(i,p,r) for i,(p,r) in enumerate(zip(parents,requests))))
            generation_seconds=time.monotonic()-started
            save(folder/'generation-summary.json',summary());event('generation_complete',**summary())
            await asyncio.gather(*grades)
            after=(await http.get(base+'/status')).json();save(folder/'engines-after.json',after)
            result=summary();result['engine_starts_before']=before.get('starts');result['engine_starts_after']=after.get('starts')
            result['stable_engines']=before.get('starts')==after.get('starts') and not after.get('exhausted')
            save(folder/'summary.json',result);event('benchmark_complete',**result)
            return int(bool(result['infrastructure_errors']) or result['generated']!=len(parents)*n or not result['stable_engines'])
        finally:
            stop.set();thread.join(timeout=25);ray.shutdown()


def main():
    parser=argparse.ArgumentParser()
    for flag in ['config-json','source','snapshot','head']:parser.add_argument('--'+flag,required=True)
    args=parser.parse_args();sys.path.insert(0,str(Path(args.source)/'third_party/discover'))
    config=Config.from_dict(json.loads(args.config_json))
    if not config.inference_only or not config.inference.native_thinking_budget:raise ValueError('requires inference-only native thinking')
    return asyncio.run(run(config,args.source,args.snapshot,args.head))

if __name__=='__main__':raise SystemExit(main())
