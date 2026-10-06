"""Durable Gemma20 -> Qwen/Muse handoff, with at-most-once submissions."""
import argparse
import fcntl
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from google.api_core.exceptions import NotFound, PreconditionFailed
from tpu.science.bootstrap import identity
from tpu.science.qubit_gen1 import checkpoint_at_step, transfer_pool
from tpu.science.seed_handoff import bind_bundle
from tpu.science.seed_pool import verify_pool
from tpu.science.ops.routing_relaunch_watch import Rollout, BASE, save

FOLDER = BASE / 'gen1-gemma20'
DONOR = 'qubit-v4-gemma-parallel2-pwc-rho05-20260921-r2'
SOURCE_BUCKET = 'sk7524-tinker-tpu-us-central2'
DEST_BUCKET = 'sk7524-tinker-tpu-us-east5'
POOL = 'tpuswarm-v5p32-east5a-erdos'


def pinned_json(bucket, key):
    b = bucket.blob(key); b.reload(timeout=60)
    raw = b.download_as_bytes(if_generation_match=b.generation, timeout=120)
    return raw, dict(uri='gs://'+bucket.name+'/'+key, generation=b.generation,
                    size=b.size, sha256=hashlib.sha256(raw).hexdigest())


def copy_once(client, source, key, destination):
    src = client.bucket(source).blob(key); src.reload(timeout=60)
    if not src.size:
        raise ValueError('empty source artifact')
    target = client.bucket(DEST_BUCKET).blob(destination)
    try:
        token = None
        while True:
            token, _, _ = target.rewrite(src, token=token, if_generation_match=0,
                                         if_source_generation_match=src.generation, timeout=120)
            if token is None: break
    except PreconditionFailed:
        pass
    target.reload(timeout=60)
    if (target.size, target.crc32c) != (src.size, src.crc32c):
        raise ValueError('conflicting immutable regional artifact: '+destination)
    return dict(source='gs://'+source+'/'+key, source_generation=src.generation,
                uri='gs://'+DEST_BUCKET+'/'+destination, generation=target.generation,
                size=target.size, crc32c=target.crc32c)


def upload_once(client, uri, data):
    bucket, key = uri.removeprefix('gs://').split('/', 1)
    blob = client.bucket(bucket).blob(key)
    try: blob.upload_from_string(data, if_generation_match=0, timeout=180)
    except PreconditionFailed: pass
    if hashlib.sha256(blob.download_as_bytes(timeout=180)).digest() != hashlib.sha256(data).digest():
        raise ValueError('conflicting immutable artifact: '+uri)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    FOLDER.mkdir(exist_ok=True)
    lock=(FOLDER/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r=Rollout();client=r.storage;bucket=client.bucket(SOURCE_BUCKET)
    plan=json.loads((FOLDER/'plan.json').read_text())
    assert {x['model'] for x in plan}=={'qwen','muse'}
    def status(state, **fields):
        data=dict(time=time.time(),state=state,source_run=DONOR,source_step=20,pool=POOL,**fields)
        save(FOLDER/'status.json',data);print(json.dumps(data),flush=True)
    def tick():
        for item in plan:
            if not (FOLDER/(item['model']+'-template')/'manifest.json').exists():
                status('waiting_for_templates');return False
        frozen=FOLDER/'donor.json'
        if not frozen.exists():
            prefix='ray-training/'+DONOR+'/'
            log=prefix+'client/tinker_log/'+DONOR+'/'
            raw, ixmeta=pinned_json(bucket,log+'member_gemma/checkpoints.jsonl')
            checkpoints=[json.loads(l) for l in raw.splitlines() if l.strip()]
            raw, metmeta=pinned_json(bucket,log+'metrics.jsonl')
            metrics=[json.loads(l) for l in raw.splitlines() if l.strip()]
            ck=checkpoint_at_step(checkpoints,metrics)
            if ck is None:
                status('waiting_for_gemma20',latest_checkpoint=max((c['batch'] for c in checkpoints),default=0),
                       latest_metrics=max((m.get('step',0) for m in metrics),default=0),
                       donor_job=r.query('i.name=?',(DONOR,))[-1]);return False
            raw, poolmeta=pinned_json(bucket,log+'puct_sampler_step_000020.json')
            source=json.loads(raw);seed=transfer_pool(source)
            save(FOLDER/'source-step20.json',source);save(FOLDER/'seed-pool.json',seed)
            digest=identity(seed);verify_pool(FOLDER/'seed-pool.json',digest)
            model, checkpoint=ck['state_path'].removeprefix('tinker://').split('/weights/')
            # Archive Gemma's exact training/optimizer and sampler artifacts in
            # east5 for provenance. They are not loaded into the recipient models.
            copies=[]
            for suffix in [model+'/'+checkpoint+'.tar.gz',model+'/sampler_weights/'+checkpoint+'.tar.gz']:
                key=prefix+'checkpoints/'+suffix
                copies.append(copy_once(client,SOURCE_BUCKET,key,'qubit-gen1-gemma20-20260924/donor/checkpoints/'+suffix))
            raw_uri='gs://'+DEST_BUCKET+'/qubit-gen1-gemma20-20260924/donor/puct_sampler_step_000020.json'
            upload_once(client,raw_uri,raw)
            save(frozen,dict(source_run=DONOR,source_step=20,checkpoint=ck,index=ixmeta,metrics=metmeta,
                source_pool=poolmeta,regional_pool_uri=raw_uri,regional_checkpoints=copies,seed_pool_sha256=digest,
                programs=sum(bool(s.get('code')) for s in seed['states']),
                best_reward=max(s['value'] for s in seed['states'] if s.get('code')),
                recipe='fresh recipient base/LoRA/optimizer; full donor pool; reset PUCT visits and time',target_steps=10))
        donor=json.loads(frozen.read_text());verify_pool(FOLDER/'seed-pool.json',donor['seed_pool_sha256'])
        inventory=r.query("i.pool=? AND s.status IN ('PENDING','STARTING','RUNNING','RECOVERING')",(POOL,))
        save(FOLDER/'pool-before-launch.json',dict(time=time.time(),jobs=inventory))
        for item in plan:
            name=item['run_id'];model=item['model'];receipt=FOLDER/(model+'-submission.json')
            matches=r.query('i.name=?',(name,))
            if receipt.exists():
                old=json.loads(receipt.read_text())
                if old['state']=='submitted':continue
                if len(matches)==1 and matches[0]['pool']==POOL:
                    save(receipt,dict(state='submitted',job_id=matches[0]['job_id'],run_id=name,reconciled=True));continue
                raise RuntimeError('uncertain prior submission; manual reconciliation required: '+name)
            if matches:raise RuntimeError('unexpected preexisting branch: '+name)
            target=FOLDER/(model+'-bound')
            if not (target/'manifest.json').exists():
                if target.exists():shutil.rmtree(target)
                bind_bundle(FOLDER/(model+'-template'),target,donor['seed_pool_sha256'])
            manifest=json.loads((target/'manifest.json').read_text());arc=target/'science-training.tar.gz';task=target/(name+'.yaml')
            assert manifest['run_id']==name and manifest['seed_pool_sha256']==donor['seed_pool_sha256']
            assert hashlib.sha256(arc.read_bytes()).hexdigest()==manifest['sha256']
            assert hashlib.sha256(task.read_bytes()).hexdigest()==manifest['task_sha256']
            assert r.ops.run(['gcloud','auth','list','--filter=status:ACTIVE','--format=value(account)']).strip()=='289186856710-compute@developer.gserviceaccount.com'
            dest='gs://'+DEST_BUCKET+'/ray-training/'+name+'/client/'
            upload_once(client,dest+'tinker_log/'+name+'/puct_sampler_step_000000.json',(FOLDER/'seed-pool.json').read_bytes())
            upload_once(client,dest+'seed-import.json',frozen.read_bytes());upload_once(client,manifest['code_uri'],arc.read_bytes())
            with receipt.open('x') as f:
                json.dump(dict(state='submitting',run_id=name,time=time.time()),f);f.flush()
                import os
                os.fsync(f.fileno())
            result=r.ops.sky('jobs','launch',str(task),'--pool',POOL,'--yes','--detach-run')
            (FOLDER/(model+'-submission.txt')).write_text(result)
            ids=re.findall(r'(?:Job ID|Job id|job ID|job id)[:\s]+(\d+)',result)
            if not ids:raise RuntimeError('ambiguous launch response: '+name)
            jid=int(ids[-1]);actual=r.job(jid)
            assert actual['name']==name and actual['pool']==POOL
            save(receipt,dict(state='submitted',job_id=jid,run_id=name,target_steps=10,time=time.time()))
        status('both_gen1_submitted',submissions=[json.loads((FOLDER/(x['model']+'-submission.json')).read_text()) for x in plan]);return True
    while True:
        try:
            if tick():return
        except NotFound as exc:status('waiting_for_durable_artifact',detail=str(exc)[:500])
        except Exception as exc:status('blocked_or_retrying',detail=repr(exc)[:1000])
        if args.once:return
        time.sleep(60)

if __name__=='__main__':main()
