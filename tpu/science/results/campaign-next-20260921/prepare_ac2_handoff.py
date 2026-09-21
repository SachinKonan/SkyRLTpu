"""Freeze a completed AC2 round, then build three shared-winner successors.

fetch: live completion checks and generation-pinned downloads (no cloud writes).
build: run on CPU allocation, verify the winner, reset pool history, build tasks.
Submission remains a separate receipt-backed operation after inspection.
"""
import argparse,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
DEST=ROOT/'tpu/science/results/ac2-shared-best-20260921'
OUT=ROOT/'.science/ac2-shared-best-20260921'
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def fetch():
    spec=importlib.util.spec_from_file_location('gate',HERE/'check_completion.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    gate.main();status=json.loads((HERE/'ac2-completion.json').read_text())
    if not status['ready']:
        print('WAIT: original AC2 round has not completed; no handoff prepared');return
    if (OUT/'snapshots.json').exists():
        print('Final source snapshot already frozen; refusing to replace it');return
    from google.cloud import storage
    client=storage.Client(project='vision-mix');snapshots=[]
    for r in status['runs']:
        uri=r['final_pool']['uri'];b,n=uri[5:].split('/',1);blob=client.bucket(b).blob(n,generation=r['final_pool']['generation'])
        data=blob.download_as_bytes(if_generation_match=r['final_pool']['generation'],timeout=180)
        snapshots.append(dict(model=r['model'],run_id=r['run_id'],status=r['status'],checkpoint=r['checkpoint'],uri=uri,generation=r['final_pool']['generation'],pool=json.loads(data)))
    write(OUT/'snapshots.json',snapshots);print('Downloaded all three pinned final pools; build next on CPU')
def build_tasks():
    from tpu.science.ac2_handoff import best_completed_candidate,recipient_profile
    from tpu.swarm.ray_train.build import build
    from tpu.swarm.ray_train.config import Config
    import yaml
    if (DEST/'submissions.json').exists():raise RuntimeError('Cannot rebuild submitted handoff')
    snapshots=json.loads((OUT/'snapshots.json').read_text());pool,provenance=best_completed_candidate(snapshots)
    original=json.loads((HERE.parent/'reallocation-10step-20260921/prepared.json').read_text())['jobs']
    seed=json.dumps(pool,indent=2).encode();seed_digest=hashlib.sha256(seed).hexdigest();jobs=[]
    for model in ['gemma','muse','qwen']:
        src=next(r for r in original if r['kind']=='ac2' and r['model']==model)
        run=f'ac2-shared-best-{model}-10step-20260921';zone='us-east5-b' if model=='qwen' else 'us-central1-b';bucket='gs://sk7524-tinker-tpu-us-central1'
        profile=recipient_profile(json.loads((ROOT/src['profile']).read_text()),run,zone,bucket);Config.from_dict(profile)
        path=ROOT/'tpu/swarm/ray_train/profiles'/(run+'.json');write(path,profile)
        folder=OUT/run;folder.mkdir(parents=True,exist_ok=True);(folder/'seed-pool.json').write_bytes(seed)
        archive,uri,task=build(path,folder);doc=yaml.safe_load(task.read_text());doc['resources']['priority']=120;task.write_text(yaml.safe_dump(doc,sort_keys=False))
        jobs.append(dict(run_id=run,model=model,kind='ac2',pool='tpuswarm-v6e32-'+('east5b-qwen35' if zone=='us-east5-b' else 'central1b'),priority=120,bucket=bucket,epochs=10,profile=str(path.relative_to(ROOT)),archive=str(archive.relative_to(ROOT)),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),code_uri=uri,task=str(task.relative_to(ROOT)),task_sha256=hashlib.sha256(task.read_bytes()).hexdigest(),seed_file=str((folder/'seed-pool.json').relative_to(ROOT)),seed_file_sha256=seed_digest,seed_pool_sha256=provenance['seed_pool_sha256'],seed_count=1,seed_destination=bucket+'/ray-training/'+run+'/client/tinker_log/'+run+'/puct_sampler_step_000000.json',optimizer='fresh',adapter='fresh'))
    write(DEST/'provenance.json',provenance);write(DEST/'prepared.json',dict(jobs=jobs));print(json.dumps(provenance,indent=2));print('Built three identical-seed AC2 successors; not submitted')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fetch','build']);a=p.parse_args();{'fetch':fetch,'build':build_tasks}[a.action]()
