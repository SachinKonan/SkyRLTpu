"""Package four parallel math runs; never modify the existing v4 jobs."""
import copy
import hashlib
import json
from pathlib import Path
import yaml
from tpu.swarm.ray_train.build import build
from tpu.swarm.ray_train.config import Config
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
PROFILES=ROOT/'tpu/swarm/ray_train/profiles'
PLAN=[('gemma','ac2','us-central1-b',1228),('muse','ac2','us-central1-b',1229),
      ('gemma','cp26','us-central1-b',1231),('muse','cp26','us-east5-b',1232)]
rows=[]
for model,task,zone,old_id in PLAN:
    old=f'fresh-v4-{model}-{task}-grpo-lr4e5-s1-20260919'
    run=f'math-v6e-{model}-{task}-grpo-lr4e5-s1-20260920'
    c=json.loads((PROFILES/f'{old}.json').read_text())
    native=json.loads((PROFILES/f'fresh-v6e-{model}-rglru-grpo-lr4e5-s1-20260919-fix1.json').read_text())
    bucket='gs://sk7524-tinker-tpu-'+('us-central1' if zone=='us-central1-b' else 'us-east5')
    c.update(run_id=run,root=f'~/.cache/{run}',accelerator='tpu-v6e-32',zone=zone,bucket=bucket,
             trainer=copy.deepcopy(native['trainer']),inference=copy.deepcopy(native['inference']))
    c['client_env']['TTD_SICK_MARKER']=f'/home/gcpuser/.cache/{run}/runs/{run}/ENGINE-SICK'
    if zone=='us-central1-b':
        for field in ('hf','orbax'):
            c['cache'][field]=native['cache'][field].replace('sk7524-tinker-tpu-us-central2','sk7524-tinker-tpu-us-central1')
        for phase in ('trainer','inference'):
            c['cache'][phase+'_compile_seed']=bucket+f'/math-hedge-20260920/cache-seeds/{model}/{phase}'
    else:
        for field in ('hf','orbax','trainer_compile_seed'):
            c['cache'][field]=native['cache'][field]
        c['cache']['inference_compile_seed']=native['cache']['inference_compile']
    for phase in ('trainer','inference'):
        c['cache'][phase+'_compile']=bucket+f'/{run}-{phase}_compile-v1'
    Config.from_dict(c).validate()
    output=ROOT/'.science/packages/math-v6e-hedge-20260920'/model/task
    if (output/'submission.json').exists():raise RuntimeError('Do not repackage submitted runs')
    profile=PROFILES/f'{run}.json';profile.write_text(json.dumps(c,indent=2)+'\n')
    archive,_,task_yaml=build(profile,output)
    doc=yaml.safe_load(task_yaml.read_text())
    audit=(ROOT/'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text()
    doc['run']="set -euo pipefail\npython3 - <<'FRESH_CLEAN_HOST'\n"+audit+'\nFRESH_CLEAN_HOST\n'+doc['run']
    doc['resources']['priority']=100
    task_yaml.write_text(yaml.safe_dump(doc,sort_keys=False))
    row=dict(hardware='v6e',task=task,model=model,run_id=run,parallel_v4_job=old_id,
             profile=str(profile.relative_to(ROOT)),pool='tpuswarm-v6e32-central1b' if zone=='us-central1-b' else 'tpuswarm-v6e32-east5b-qwen35',
             zone=zone,priority=100,package_dir=str(output.relative_to(ROOT)),archive=str(archive.relative_to(ROOT)),
             task_yaml=str(task_yaml.relative_to(ROOT)),code_uri=doc['envs']['RAY_TRAIN_CODE'],
             archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),yaml_sha256=hashlib.sha256(task_yaml.read_bytes()).hexdigest())
    (output/'submission-manifest.json').write_text(json.dumps(row,indent=2)+'\n')
    rows.append(row);print(run,zone,flush=True)
(HERE/'jobs.json').write_text(json.dumps(dict(jobs=rows),indent=2)+'\n')
