"""Package isolated retries; preserve the original submitted runs and receipts."""
import hashlib
import json
from pathlib import Path
import yaml
from tpu.swarm.ray_train.build import build
from tpu.swarm.ray_train.config import Config
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
rows=[]
for model in ('qwen','gemma','muse'):
    lr='lr15e4' if model=='qwen' else 'lr4e5'
    original=f'fresh-v6e-{model}-rglru-grpo-{lr}-s1-20260919'
    run=original+'-fix1'
    profile=ROOT/'tpu/swarm/ray_train/profiles'/f'{run}.json'
    output=ROOT/'.science/packages/rglru-client-import-fix-20260919'/model
    if (output/'submission.json').exists():
        raise RuntimeError('Do not overwrite a submitted retry')
    config=json.loads((profile.parent/f'{original}.json').read_text())
    config.update(run_id=run,root=f'~/.cache/{run}')
    config['client_env']['TTD_SICK_MARKER']=f'/home/gcpuser/.cache/{run}/runs/{run}/ENGINE-SICK'
    for phase in ('trainer','inference'):
        config['cache'][phase+'_compile']=f"{config['bucket']}/{run}-{phase}_compile-v1"
    Config.from_dict(config).validate()
    profile.write_text(json.dumps(config,indent=2)+'\n')
    archive,_,task=build(profile,output)
    doc=yaml.safe_load(task.read_text())
    audit=(ROOT/'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text()
    doc['run']="set -euo pipefail\npython3 - <<'FRESH_CLEAN_HOST'\n"+audit+'\nFRESH_CLEAN_HOST\n'+doc['run']
    doc['resources']['priority']=50
    task.write_text(yaml.safe_dump(doc,sort_keys=False))
    row=dict(hardware='v6e',task='rglru',model=model,run_id=run,
             original_run_id=original,profile=str(profile.relative_to(ROOT)),
             pool='tpuswarm-v6e32-east5b-qwen35',priority=50,
             package_dir=str(output.relative_to(ROOT)),archive=str(archive.relative_to(ROOT)),
             task_yaml=str(task.relative_to(ROOT)),code_uri=doc['envs']['RAY_TRAIN_CODE'],
             archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
             yaml_sha256=hashlib.sha256(task.read_bytes()).hexdigest())
    (output/'submission-manifest.json').write_text(json.dumps(row,indent=2)+'\n')
    rows.append(row)
    print(run,flush=True)
(HERE/'jobs.json').write_text(json.dumps(dict(jobs=rows),indent=2)+'\n')
