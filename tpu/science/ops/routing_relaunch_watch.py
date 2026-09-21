"""Bounded relaunch progression for three explicitly authorized v4-64 jobs.

A failed check stops progression. Existing workload cancellation is restricted to
1339 (old Gemma routing) and 1340 (old Muse routing), after replacement artifacts
and preserved checkpoint evidence exist. AC2/farms/circuit jobs are read-only.
"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'.science/routing-relaunch-20260921'
POOL='tpuswarm-v4-64-central2-qwen35-erdos'
BUCKET='sk7524-tinker-tpu-us-central2'
PYTHON='/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-hybrid-inference-migration/.venv/bin/python'
OPS='/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-hybrid-inference-migration/.science/reallocation-10step/ops.py'


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


class Rollout:
    def __init__(self):
        spec=importlib.util.spec_from_file_location('routing_ops',OPS)
        self.ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.ops)
        os.environ.update(self.ops.ops.env)
        from google.cloud import storage
        self.storage=storage.Client(project='vision-mix')
        assert self.storage._credentials.service_account_email=='289186856710-compute@developer.gserviceaccount.com'
        assert self.ops.run(['gcloud','auth','list','--filter=status:ACTIVE','--format=value(account)']).strip()==self.storage._credentials.service_account_email
        self.path=BASE/'rollout-state.json'
        self.state=json.loads(self.path.read_text()) if self.path.exists() else dict(phase='benchmark',jobs={},artifacts={},retired={})
        self.db=Path(self.ops.ops.env['HOME'])/'.sky/spot_jobs.db'

    def persist(self,**fields):
        self.state.update(fields,updated=time.time());save(self.path,self.state)
        print(json.dumps({'time':time.time(),'phase':self.state['phase'],'jobs':self.state['jobs'],**fields}),flush=True)

    def query(self,where,args):
        c=sqlite3.connect('file:'+str(self.db)+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
        try:return [dict(r) for r in c.execute('SELECT i.spot_job_id AS job_id,i.name,s.status,i.pool,i.current_cluster_name AS cluster FROM job_info i JOIN spot s ON s.spot_job_id=i.spot_job_id WHERE '+where,args)]
        finally:c.close()

    def job(self,jid):
        rows=self.query('i.spot_job_id=?',(jid,));assert len(rows)==1,rows
        return rows[0]

    def done(self,jid):
        row=self.job(jid)
        if row['status'].startswith('FAILED') or row['status']=='CANCELLED':
            raise RuntimeError('Affected new job stopped: '+str(row))
        return row['status']=='SUCCEEDED'

    def run(self,args):
        subprocess.run([str(a) for a in args],check=True,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)))

    def upload(self,path,uri):
        from google.api_core.exceptions import PreconditionFailed
        bucket,key=uri.removeprefix('gs://').split('/',1);blob=self.storage.bucket(bucket).blob(key)
        try:blob.upload_from_filename(str(path),if_generation_match=0)
        except PreconditionFailed:
            assert hashlib.sha256(blob.download_as_bytes()).digest()==hashlib.sha256(Path(path).read_bytes()).digest(),'immutable object conflict'

    def submit(self,key,task,name):
        known=self.state['jobs'].get(key)
        if known:return known
        existing=self.query('i.name=?',(name,))
        if existing:
            assert len(existing)==1,existing
            assert existing[0]['pool']==POOL,'same name belongs to another pool'
            self.state['jobs'][key]=existing[0]['job_id'];self.persist();return existing[0]['job_id']
        intents=self.state.setdefault('submit_intents',{})
        if key in intents:raise RuntimeError('Ambiguous prior submission; reconcile before repeating '+key)
        # Inventory immediately before mutation. Submission uses this existing
        # pool only; this program never changes its size or provisions directly.
        nodes=json.loads(self.ops.run(['gcloud','compute','tpus','tpu-vm','list','--zone','us-central2-b','--project','vision-mix','--format=json']))
        healthy=[{k:n.get(k) for k in ('name','state','health')} for n in nodes if n.get('acceleratorType')=='v4-64' and n.get('state')=='READY' and n.get('health')=='HEALTHY']
        assert healthy,'No healthy existing v4-64 capacity'
        inventory=self.query('i.pool=? AND s.status IN (\'RUNNING\',\'STARTING\',\'PENDING\',\'RECOVERING\')',(POOL,))
        intents[key]=dict(task=str(task),task_sha256=hashlib.sha256(Path(task).read_bytes()).hexdigest(),name=name,time=time.time(),inventory=inventory,provider=healthy)
        self.persist()
        raw=self.ops.sky('jobs','launch',str(task),'--pool',POOL,'--yes','--detach-run')
        (BASE/(key+'-launch.txt')).write_text(raw)
        ids=re.findall(r'(?:Job ID|Job id|job ID|job id)[:\s]+(\d+)',raw)
        if not ids:raise RuntimeError('Ambiguous submission; reconcile '+key)
        self.state['jobs'][key]=int(ids[-1]);self.persist();return int(ids[-1])

    def regrade(self,model):
        folder=BASE/'regrade-v2'/model;m=json.loads((folder/'manifest.json').read_text())
        assert hashlib.sha256((folder/'task.yaml').read_bytes()).hexdigest()==m['task_sha256']
        digest=hashlib.sha256(b''.join(p.name.encode()+b'\0'+p.read_bytes() for p in sorted((ROOT/'tpu/science').glob('*.py')))).hexdigest()
        assert digest==m['evaluator_sha256'],'grading source changed after frozen regrade bundle'
        return self.submit(model+'-regrade',folder/'task.yaml',m['run_id'])

    def prepare_training(self,model):
        if model in self.state['artifacts']:return self.state['artifacts'][model]
        from tpu.science.routing_regrade import import_pool
        from tpu.science.routing_relaunch_profile import build
        recipe=json.loads((BASE/'regrade-v2'/model/'manifest.json').read_text())
        source=json.loads((BASE/'sources-v2'/model/'source-manifest.json').read_text())
        bucket,key=recipe['results_uri'].removeprefix('gs://').split('/',1)
        verdicts={Path(b.name).stem:json.loads(b.download_as_bytes(if_generation_match=b.generation))
                  for b in self.storage.list_blobs(bucket,prefix=key+'/verdicts/') if b.name.endswith('.json')}
        seeds=BASE/'seeds'/model
        if not seeds.exists():import_pool(source,verdicts,seeds,target_run=recipe['target_train_run'],evaluator_sha256=recipe['evaluator_sha256'])
        report=json.loads((seeds/'seed-import.json').read_text())
        assert report['evaluator_sha256']==recipe['evaluator_sha256'] and report['source_manifest_sha256']==source['sha256']
        profile=ROOT/'tpu/swarm/ray_train/profiles'/(recipe['target_train_run']+'.json')
        if not profile.exists():
            assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'worktree changed; review before packaging'
            build(model,seeds,profile)
            self.run(['git','add',profile,str(profile)+'.changes.json'])
            self.run(['git','commit','-m','Prepare '+model+' routing relaunch with regraded seeds and farm borrowing'])
        else:
            assert json.loads(profile.read_text())['seed_pool_sha256']==report['pool_sha256']
            assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'uncommitted training profile; review'
        output=BASE/'training'/model
        self.run([PYTHON,'-m','tpu.science.package_training','--profile',profile,'--output',output])
        package=json.loads((output/'manifest.json').read_text());task=output/(recipe['target_train_run']+'.yaml')
        import yaml
        doc=yaml.safe_load(task.read_text());doc['resources']['priority']=110
        task.write_text(yaml.safe_dump(doc,sort_keys=False))
        package['task_sha256']=hashlib.sha256(task.read_bytes()).hexdigest()
        save(output/'manifest.json',package)
        self.upload(output/'science-training.tar.gz',package['code_uri'])
        target='gs://'+BUCKET+'/ray-training/'+recipe['target_train_run']
        self.upload(seeds/'puct_sampler_step_000000.json',target+'/client/tinker_log/'+recipe['target_train_run']+'/puct_sampler_step_000000.json')
        self.upload(seeds/'seed-import.json',target+'/seed-import.json')
        artifact=dict(run_id=recipe['target_train_run'],task=str(task),code_uri=package['code_uri'],code_sha256=package['sha256'],
            commit=package['main_commit'],seed_pool_sha256=report['pool_sha256'],task_sha256=hashlib.sha256(task.read_bytes()).hexdigest())
        self.state['artifacts'][model]=artifact;self.persist();return artifact

    def retire_old(self,model):
        old={'gemma':1339,'muse':1340}.get(model)
        if old is None:return True
        row=self.job(old)
        from tpu.science.routing_regrade import SOURCES
        bucket,run=SOURCES[model]
        assert row['name']==run and row['pool']==POOL,('old target changed',row)
        if row['status'] in ('SUCCEEDED','CANCELLED','FAILED'):return True
        if model not in self.state['retired']:
            assert model in self.state['artifacts'],'replacement must exist before retirement'
            blobs=[dict(name=b.name,generation=b.generation,size=b.size) for b in self.storage.list_blobs(bucket,prefix='ray-training/'+run+'/checkpoints/') if b.name.endswith('.tar.gz')]
            assert blobs,'No durable old checkpoints found; preserve and review'
            self.state['retired'][model]=dict(job=row,durable_checkpoints=blobs,replacement=self.state['artifacts'][model],time=time.time())
            self.persist();self.ops.sky('jobs','cancel',str(old),'--yes')
        return False

    def launch_training(self,model):
        artifact=self.state['artifacts'][model]
        assert hashlib.sha256(Path(artifact['task']).read_bytes()).hexdigest()==artifact['task_sha256']
        if not self.retire_old(model):return False
        self.submit(model+'-train',Path(artifact['task']),artifact['run_id']);return True

    def gemma_cycle(self):
        run=self.state['artifacts']['gemma']['run_id'];b=self.storage.bucket(BUCKET)
        prefix='ray-training/'+run+'/';client=prefix+'client/tinker_log/'+run+'/'
        def rows(key):
            from google.api_core.exceptions import NotFound, PreconditionFailed
            for attempt in range(3):
                blob=b.get_blob(key)
                if blob is None:return []
                try:
                    content=blob.download_as_text(if_generation_match=blob.generation)
                    return [json.loads(l) for l in content.splitlines() if l.strip()]
                except (NotFound,PreconditionFailed):
                    if attempt==2:raise
        metrics=rows(client+'metrics.jsonl');checkpoints=rows(client+'member_gemma/checkpoints.jsonl')
        if not metrics or not checkpoints:return False
        metric=next((m for m in metrics if m.get('progress/batch')==0 or m.get('life_step')==0),None)
        if metric is None:return False
        if metric.get('gemma/train_error',0) or metric.get('gemma/train_skipped',0):
            raise RuntimeError('Gemma first optimizer step failed or skipped; do not admit other models')
        if not any('time/train' in k and isinstance(v,(int,float)) and v>0 for k,v in metric.items()):return False
        if metric.get('gemma/rollout_groups_failed',0):raise RuntimeError('Gemma first batch has failed rollout groups')
        if (metric.get('gemma/puct/sampled_size')!=16 or
                metric.get('gemma/env/all/total_episodes')!=32):
            raise RuntimeError('Gemma first cycle does not attest the full 16 x 32 rollout batch')
        checkpoint=next((c for c in checkpoints if c.get('batch')==1),None)
        if not checkpoint or not checkpoint.get('sampler_path'):return False
        model_id,checkpoint_id=checkpoint['sampler_path'].removeprefix('tinker://').split('/',1)
        archive=b.get_blob(prefix+'checkpoints/'+model_id+'/000001.tar.gz')
        if archive is None or not archive.size:return False
        version=(model_id+'_'+checkpoint_id).replace('/','_').replace(':','_')
        proof=None;remote_generated=False
        for blob in self.storage.list_blobs(BUCKET,prefix=prefix+'logs/'):
            if not blob.name.endswith('inference-events.jsonl'):continue
            events=rows(blob.name)
            committed=[e for e in events if e.get('event')=='adapter_committed' and e.get('version')==version]
            remote_generated|=any(e.get('event')=='hybrid_generated' and e.get('route')=='remote' and e.get('output_tokens',0)>0 for e in events)
            if committed:
                after=committed[-1]['time']
                done=next((e for e in events if e.get('event')=='hybrid_generated' and e.get('time',0)>after and e.get('output_tokens',0)>0),None)
                if done:proof=dict(checkpoint=checkpoint,archive_generation=archive.generation,adapter=committed[-1],generation=done,metrics=metric)
        if proof and remote_generated:
            save(BASE/'gemma-first-cycle.json',proof);return True
        return False

    def tick(self):
        phase=self.state['phase']
        if 'gemma-train' in self.state['jobs']:
            # A real training failure holds later admissions even while CPU
            # regrades are still in flight. Recovery may continue normally.
            self.done(self.state['jobs']['gemma-train'])
        if phase=='benchmark':
            if not self.done(1439):return
            self.run([self.ops.ops.venv/'python',BASE/'collect_direct.py'])
            self.run([PYTHON,BASE/'benchmark_gate.py',BASE/'benchmark-v2/downloaded'])
            self.regrade('gemma');self.persist(phase='gemma_regrade')
        elif phase=='gemma_regrade':
            if not self.done(self.state['jobs']['gemma-regrade']):return
            self.prepare_training('gemma');self.persist(phase='gemma_launch')
        elif phase=='gemma_launch':
            if self.launch_training('gemma'):
                self.regrade('qwen');self.persist(phase='qwen_regrade')
        elif phase=='qwen_regrade':
            if not self.done(self.state['jobs']['qwen-regrade']):return
            self.prepare_training('qwen');self.regrade('muse');self.persist(phase='muse_regrade')
        elif phase=='muse_regrade':
            if not self.done(self.state['jobs']['muse-regrade']):return
            self.prepare_training('muse');self.persist(phase='gemma_cycle')
        elif phase=='gemma_cycle':
            if self.gemma_cycle():self.persist(phase='qwen_launch')
        elif phase=='qwen_launch':
            if self.launch_training('qwen'):self.persist(phase='muse_launch')
        elif phase=='muse_launch':
            if self.launch_training('muse'):self.persist(phase='submitted_all')


def main():
    os.chdir(ROOT)
    lock=(BASE/'rollout.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rollout=Rollout()
    if rollout.state.get('error'):raise RuntimeError('Review recorded error before resuming')
    while rollout.state['phase']!='submitted_all':
        try:rollout.tick();rollout.persist()
        except Exception as exc:
            rollout.persist(error=type(exc).__name__+': '+str(exc));raise
        time.sleep(30)


if __name__=='__main__':main()
