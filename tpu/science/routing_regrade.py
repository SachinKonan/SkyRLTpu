"""Freeze all historical drafts and import only new verified routing verdicts."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import time
from .routing_resources import contract

SOURCES = {
    'gemma': ('sk7524-tinker-tpu-us-east5','science-v6e-gemma-qubit-grpo-lr4e5-s1-20260920'),
    'qwen': ('sk7524-tinker-tpu-us-east5','science-v6e-qwen-qubit-grpo-lr15e4-s1-20260920'),
    'muse': ('sk7524-tinker-tpu-us-central1','capacity-v6e-muse-qubit-grpo-4e5-20260920'),
}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def freeze(model, destination):
    from google.cloud import storage
    bucket,run=SOURCES[model];client=storage.Client(project='vision-mix');b=client.bucket(bucket)
    destination=Path(destination)
    if destination.exists():raise ValueError('source archive destination already exists')
    destination.mkdir(parents=True)
    prefix='ray-training/'+run+'/client/bootstrap/'
    objects=list(client.list_blobs(b,prefix=prefix));grades=[v for v in objects if re.fullmatch(re.escape(prefix)+r'drafts/group-\d{3}/grade-\d{3}\.json',v.name)]
    expected={f'drafts/group-{group:03d}/grade-{index:03d}.json' for group in range(64) for index in range(16)}
    if {v.name.removeprefix(prefix) for v in grades}!=expected:
        raise ValueError('original archive must contain all 1024 draft grade records')
    def fetch(blob):
        from google.api_core.exceptions import NotFound, PreconditionFailed
        # Active run syncs can replace an unchanged archive object after listing.
        # Retry the current generation, but always record the generation fetched.
        for attempt in range(5):
            try:
                raw=blob.download_as_bytes(if_generation_match=blob.generation,timeout=120)
                break
            except (NotFound, PreconditionFailed):
                if attempt == 4: raise
                blob=b.get_blob(blob.name)
                if blob is None: raise ValueError('source object disappeared during freeze')
        record=json.loads(raw);relative=blob.name.removeprefix(prefix)
        save(destination/relative,record)
        return dict(uri='gs://'+bucket+'/'+blob.name,generation=str(blob.generation),
                    object_sha256=hashlib.sha256(raw).hexdigest(),relative=relative,record=record)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:records=list(ex.map(fetch,grades))
    metadata=[]
    for name in ['complete.json','contract.json','root.json']:
        blob=next((v for v in objects if v.name==prefix+name),None)
        if blob is None:raise ValueError('original archive missing '+name)
        row=fetch(blob);metadata.append({k:v for k,v in row.items() if k!='record'})
    summary=json.loads((destination/'complete.json').read_text())
    if summary.get('drafted',summary.get('total'))!=1024 or summary.get('optimizer_steps')!=0:
        raise ValueError('unexpected original generation budget')
    from .bootstrap import identity
    source_contract=json.loads((destination/'contract.json').read_text())
    if (identity(source_contract['contract'])!=source_contract['sha256'] or
            source_contract['sha256']!=summary['contract_sha256']):
        raise ValueError('original bootstrap contract checksum mismatch')
    programs={};ids=set();occurrences=[]
    for row in sorted(records,key=lambda r:r['relative']):
        old=row['record'];code=old.get('code','')
        if not isinstance(code,str):raise ValueError('draft code must be a string, including malformed/empty sources')
        if old['id'] in ids:raise ValueError('duplicate draft ID')
        ids.add(old['id']);sha=hashlib.sha256(code.encode()).hexdigest()
        program=programs.setdefault(sha,dict(source_sha256=sha,code=code,canonical_source=row['uri'],occurrences=[]))
        occurrence={k:v for k,v in row.items() if k!='record'}
        occurrence.update(draft_id=old['id'],source_sha256=sha,old_correctness=old['correctness'],
                          old_reward=old['reward'],old_message=old.get('message',old.get('msg','')))
        occurrences.append(occurrence);program['occurrences'].append(old['id'])
    manifest=dict(version=1,model=model,source_run=run,bucket=bucket,drafts=1024,unique_programs=len(programs),
                  old_valid=sum(v['old_correctness']==1 for v in occurrences),metadata=metadata,
                  occurrences=occurrences,programs=list(programs.values()),
                  prompt_provenance='Historical generations under the original bootstrap prompt; no new generations',
                  deduplication='Exact UTF-8 source bytes, one evaluation per canonical source; no best-of-duplicate replay')
    manifest['sha256']=digest(manifest);save(destination/'source-manifest.json',manifest)
    return manifest


def import_pool(manifest, verdicts, destination, *, target_run, evaluator_sha256):
    """All unique programs must have a verdict, including syntax/timeout failures."""
    from .feedback import observation
    from .bootstrap import identity
    expected=manifest['sha256'];unsigned={k:v for k,v in manifest.items() if k!='sha256'}
    if digest(unsigned)!=expected:raise ValueError('source manifest checksum mismatch')
    programs={p['source_sha256']:p for p in manifest['programs']}
    if len(programs)!=manifest['unique_programs'] or set(verdicts)!=set(programs):
        raise ValueError('incomplete or duplicate regrading outcomes')
    good=[]
    for sha,verdict in verdicts.items():
        if verdict.get('failure_class')=='infrastructure':
            raise ValueError('infrastructure failure is not a regrade outcome')
        m=verdict['metrics']
        if (m.get('source_sha256')!=sha or m.get('resource_contract')!=contract()
                or m.get('regrade_evaluator_sha256')!=evaluator_sha256):
            raise ValueError('regrade evaluator/source/resource identity mismatch')
        if verdict['correctness']==1:
            from .routing_suite import manifest as suite_manifest
            cases=m.get('cases',[])
            if len(cases)!=72 or {c['case'] for c in cases}!={c['id'] for c in suite_manifest()['cases']}:
                raise ValueError('successful regrade missing the exact 72 cases')
            from .routing_suite import rescore_verified
            # Recompute the full reward from trusted pinned weights/baselines.
            rescore_verified(dict(correctness=1,code=programs[sha]['code'],reward=verdict['reward'],metrics=m))
            good.append((sha,verdict))
        elif verdict['correctness']!=0 or verdict['reward']!=0:
            raise ValueError('invalid regrade must have zero reward')
    good.sort(key=lambda item:(-item[1]['reward'],item[0]));retained=good[:512]
    if not retained:raise ValueError('no valid regraded programs; do not launch training')
    states=[]
    for sha,verdict in retained:
        states.append(dict(type='State',id='regrade-'+sha,code=programs[sha]['code'],construction=None,
            value=verdict['reward'],observation=observation('routing',verdict),origin=manifest['model'],
            timestep=0,parents=[],parent_values=[]))
    pool=dict(step=0,states=states,initial_states=states,puct_n={},puct_m={},puct_T=0)
    valid_hashes={sha for sha,_ in good};old=manifest['occurrences']
    def timeout(message):return any(v in message.lower() for v in ['timeout','timed out','exhausted budget','deadline'])
    old_valid={v['source_sha256'] for v in old if v['old_correctness']==1}
    topology_minima={}
    for sha,verdict in good:
        from .feedback import routing_cases
        for name,row in routing_cases(verdict['metrics'])['topologies'].items():
            candidate=dict(source_sha256=sha,swaps=row['swaps'])
            if name not in topology_minima or (candidate['swaps'],sha)<(topology_minima[name]['swaps'],topology_minima[name]['source_sha256']):
                topology_minima[name]=candidate
    summary=dict(source_manifest_sha256=expected,evaluator_sha256=evaluator_sha256,resource_contract=contract(),
        target_run=target_run,model=manifest['model'],drafts=1024,unique_programs=len(programs),
        raw_valid_before=manifest['old_valid'],raw_valid_after=sum(v['source_sha256'] in valid_hashes for v in old),
        unique_valid_after=len(good),retained=len(states),pool_sha256=identity(pool),optimizer_steps=0,
        unique_valid_before=len(old_valid),newly_valid_unique=len(valid_hashes-old_valid),
        unique_timeouts_before=len({v['source_sha256'] for v in old if timeout(v['old_message'])}),
        unique_timeouts_after=sum(timeout(v.get('msg','')) for v in verdicts.values()),
        regrade_program_wall_seconds=sum(v['metrics'].get('worker_seconds',0) for v in verdicts.values()),
        regrade_cpu_seconds=sum(v['metrics'].get('process_cpu_seconds',0) for v in verdicts.values()),
        topology_minima=topology_minima,topology_minima_note='May be different policies; not one combined score vector',
        raw_timeouts_before=sum(timeout(v['old_message']) for v in old),
        raw_timeouts_after=sum(timeout(verdicts[v['source_sha256']].get('msg','')) for v in old),
        newly_valid_drafts=sum(v['old_correctness']==0 and v['source_sha256'] in valid_hashes for v in old),
        best_policy=dict(id=states[0]['id'],reward=retained[0][1]['reward'],
                         feedback=json.loads(states[0]['observation'])),
        prompt_provenance=manifest['prompt_provenance'],training_prompt='parallel-v2 Gemini target-informed',
        initialization='Fresh adapter, optimizer, and PUCT statistics; historical source regraded once per exact program')
    destination=Path(destination)
    if destination.exists():raise ValueError('refusing to overwrite seed import')
    destination.mkdir(parents=True);save(destination/'puct_sampler_step_000000.json',pool);save(destination/'seed-import.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',choices=tuple(SOURCES),required=True);p.add_argument('--destination',required=True)
    args=p.parse_args();r=freeze(args.freeze,args.destination);print(json.dumps({k:r[k] for k in ['model','drafts','unique_programs','old_valid','sha256']}))
