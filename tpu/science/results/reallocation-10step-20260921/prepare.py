"""Build ten-step replacements. No cloud writes, cancellation or submission."""
import concurrent.futures,hashlib,json
from pathlib import Path
import yaml
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.build import build
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).parent
PROFILES=ROOT/'tpu/swarm/ray_train/profiles'
OUT=ROOT/'.science/reallocation-10step/packages'

def read(name):return json.loads((PROFILES/(name+'.json')).read_text())
def hybrid(d):
    d['inference'].update(external_pool_updates=True,external_pool_lease_scope='run',external_pool_require_initial=True,
        external_pool_scheduler=True,external_pool_attestation=True,external_pool_engines=4,external_pool_max_n=32,
        external_pool_max_concurrent_requests=4,external_pool_prepare_timeout=1800,external_pool_rpc_timeout=10,
        external_pool_lease_seconds=300,external_pool_heartbeat_seconds=30,external_pool_health_grace_seconds=90)

def main():
    rows=[]
    def add(d,kind,model,pool,priority,minimum=0,old=(),label=None):
        d.update(systemd_runtime=True,max_restarts_on_errors=3,checkpoint_resume=True,resume_min_checkpoint_step=minimum)
        if kind!='farm':d['client_env']['NUM_EPOCHS']='10'
        d['root']='~/.cache/'+d['run_id']
        for role in ['trainer','inference']:
            if role+'_compile' in d['cache']:
                d['cache'].setdefault(role+'_compile_seed',d['cache'][role+'_compile'])
                d['cache'][role+'_compile']=d['bucket']+'/reallocation-10step-20260921/'+(label or d['run_id'])+'/'+role
        cfg=Config.from_dict(d)
        name=label or d['run_id']+'-10step-deployment'
        path=PROFILES/(name+'.json');path.write_text(json.dumps(d,indent=2)+'\n')
        rows.append(dict(kind=kind,model=model,run_id=d['run_id'],profile=str(path.relative_to(ROOT)),pool=pool,
            priority=priority,resume_min_checkpoint_step=minimum,supersedes=list(old),label=name,
            bucket=d['bucket'],epochs=None if kind=='farm' else 10))
    for model,number,old in [('qwen',1,1250),('gemma',1,1252),('gemma',2,1253),('muse',1,1254)]:
        d=read('qwen-multilora-farm-2-20260919' if model=='qwen' else f'{model}-multilora-farm-{number}-20260919')
        d['run_id']=f'farm10-{model}-{number}-20260921'
        d['inference'].update(external_pool_attestation=True,farm_drain_timeout=120)
        add(d,'farm',model,'tpuswarm-v4-32-central2-smoke',120,old=[old])
    ac2_sources={
        'qwen':('fresh-v4-qwen-ac2-grpo-lr15e4-s1-20260919',1),
        'gemma':('math-v6e-gemma-ac2-grpo-lr4e5-s1-20260920',1),
        'muse':('math-v6e-muse-ac2-grpo-lr4e5-s1-20260920',2)}
    for model,(source,step) in ac2_sources.items():
        orig=read(source)
        decision_path=ROOT/'.science/reallocation-10step/placement.json'
        decision=json.loads(decision_path.read_text()) if decision_path.exists() else {}
        region=decision.get(model, 'central1b' if model=='muse' else 'east5b')
        d=read(f'hybrid-v6e-{model}-ac2-canary-'+('central1b' if model=='muse' else 'east5b')+'-20260920')
        d.update(run_id=source,bucket=orig['bucket'],zone='us-central1-b' if region=='central1b' else 'us-east5-b')
        d['client_env']['TTD_SICK_MARKER']=f'/home/gcpuser/.cache/{source}/runs/{source}/ENGINE-SICK'
        hybrid(d)
        add(d,'ac2',model,'tpuswarm-v6e32-'+('central1b' if region=='central1b' else 'east5b-qwen35'),120,
            step,old={'qwen':[1318],'gemma':[],'muse':[1321]}[model])
    d=read('fresh-v6e-gemma-rglru-grpo-lr4e5-s1-20260919-fix1');hybrid(d)
    add(d,'rglru','gemma','tpuswarm-v6e32-east5b-qwen35',110,old=[1294])
    routing_sources={'qwen':('science-v6e-qwen-qubit-grpo-lr15e4-s1-20260920',2),
        'gemma':('science-v6e-gemma-qubit-grpo-lr4e5-s1-20260920',0),
        'muse':('capacity-v6e-muse-qubit-grpo-4e5-20260920',3)}
    for model,(source,step) in routing_sources.items():
        d=read(f'fresh-v4-{model}-qubit-grpo-lr'+('15e4' if model=='qwen' else '4e5')+'-s1-20260919')
        orig=read(source);d.update(run_id=source,bucket=orig['bucket'],bootstrap_layers=0,bootstrap_all_hosts=False,
            bootstrap_max_drafts=0,bootstrap_target_valid=0)
        d['client_env']['TTD_SICK_MARKER']=f'/home/gcpuser/.cache/{source}/runs/{source}/ENGINE-SICK'
        add(d,'qubit',model,'tpuswarm-v4-64-central2-qwen35-erdos',110,step,
            old={'qwen':[1287,1301,1322],'gemma':[1288,1302],'muse':[1289,1303,1323]}[model])
    for model in ['gemma','muse','qwen']:
        d=read(f'fresh-v5p-{model}-circuit-grpo-lr'+('15e4' if model=='qwen' else '4e5')+'-s1-20260919')
        d.update(accelerator='tpu-v5p-64',hosts=8,zone='us-central1-a')
        if model=='gemma':
            source=read('fresh-v4-gemma-circuit-grpo-lr4e5-s1-20260919')
            d.update(run_id=source['run_id'],bucket=source['bucket'],bootstrap_layers=0,bootstrap_all_hosts=False,
                bootstrap_max_drafts=0,bootstrap_target_valid=0)
        else:d.update(run_id=f'placement-v5p64-{model}-10step-20260921',bucket='gs://sk7524-tinker-tpu-us-central1')
        d['client_env']['TTD_SICK_MARKER']=f'/home/gcpuser/.cache/{d["run_id"]}/runs/{d["run_id"]}/ENGINE-SICK'
        add(d,'circuit',model,None,100,old={'gemma':[1291],'muse':[1292],'qwen':[1290,1324]}[model],label=f'placement-v5p64-{model}-10step-20260921')
    def build_row(row):
        folder=OUT/row['label'];archive,uri,task=build(ROOT/row['profile'],folder)
        doc=yaml.safe_load(task.read_text());doc['resources']['priority']=row['priority']
        task.write_text(yaml.safe_dump(doc,sort_keys=False))
        row.update(archive=str(archive.relative_to(ROOT)),code_uri=uri,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            task=str(task.relative_to(ROOT)),task_sha256=hashlib.sha256(task.read_bytes()).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(build_row, rows))
    (HERE/'prepared.json').write_text(json.dumps(dict(jobs=rows,defer_jobs=[1293],total_step_cap=10),indent=2)+'\n')
    print(json.dumps([dict(kind=r['kind'],model=r['model'],run=r['run_id'],pool=r['pool'],resume=r['resume_min_checkpoint_step']) for r in rows],indent=2))
if __name__=='__main__':main()
