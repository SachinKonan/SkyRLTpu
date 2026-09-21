import importlib.util,json,os,sqlite3,tarfile
from pathlib import Path
s=importlib.util.spec_from_file_location('o','.science/reallocation-10step/ops.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o);os.environ.update(o.ops.env)
from google.cloud import storage
out=o.ROOT/'tpu/science/results/qwen-ac2-central-20260921';r=json.load(open(out/'prepared.json'))['jobs'][0];orig=next(r for r in json.load(open(o.HERE/'prepared.json'))['jobs'] if r['kind']=='ac2' and r['model']=='qwen')
with tarfile.open(o.ROOT/orig['archive']) as t:p=json.load(t.extractfile(orig['profile']))
assert p['checkpoint_resume'] and p['client_env']['NUM_EPOCHS']=='10' and p['run_id']==r['run_id']
meta=next(x for x in r['objects'] if x['name'].endswith('/tinker-backup.db'));c=storage.Client(project='vision-mix');b=c.bucket(orig['bucket'][5:]);path=o.ROOT/'.science/qwen-ac2-central-20260921/backup.db';b.blob(meta['name'],generation=meta['generation']).download_to_filename(path,if_generation_match=meta['generation'],timeout=120)
db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok';model=r['checkpoint']['state_path'].split('/')[2];rows=db.execute('select checkpoint_type,status from checkpoints where model_id=? and checkpoint_id=?',(model,r['checkpoint']['name'])).fetchall();assert ('TRAINING','COMPLETED') in rows and ('SAMPLER','COMPLETED') in rows
proof={'database_generation':meta['generation'],'database_quick_check':'ok','registrations':rows,'checkpoint':r['checkpoint'],'epochs':10,'checkpoint_resume':True};(out/'state-verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
