"""Package three all-topology qubit hedges; preserve every existing run."""
import copy
import hashlib
import json
from pathlib import Path

import yaml
from tpu.science.package_training import package
from tpu.swarm.ray_train.config import Config

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PROFILES = ROOT / 'tpu/swarm/ray_train/profiles'
for model, lr, old_job in (('qwen', '15e4', 1233), ('gemma', '4e5', 1234), ('muse', '4e5', 1235)):
    run = f'science-v6e-{model}-qubit-grpo-lr{lr}-s1-20260920'
    output = ROOT / f'.science/packages/math-v6e-hedge-20260920/{model}/qubit'
    if (output / 'submission.json').exists():
        raise RuntimeError('Do not repackage a submitted run')
    c = json.loads((PROFILES / f'fresh-v4-{model}-qubit-grpo-lr{lr}-s1-20260919.json').read_text())
    native = json.loads((PROFILES / f'fresh-v6e-{model}-rglru-grpo-lr{lr}-s1-20260919-fix1.json').read_text())
    bucket = 'gs://sk7524-tinker-tpu-us-east5'
    c.update(run_id=run, root=f'~/.cache/{run}', accelerator='tpu-v6e-32',
             zone='us-east5-b', bucket=bucket, trainer=copy.deepcopy(native['trainer']),
             inference=copy.deepcopy(native['inference']))
    c['client_env']['TTD_SICK_MARKER'] = f'/home/gcpuser/.cache/{run}/runs/{run}/ENGINE-SICK'
    for field in ('hf', 'orbax', 'trainer_compile_seed'):
        c['cache'][field] = native['cache'][field]
    c['cache']['inference_compile_seed'] = native['cache']['inference_compile']
    for phase in ('trainer', 'inference'):
        c['cache'][phase + '_compile'] = bucket + f'/{run}-{phase}_compile-v1'
    Config.from_dict(c).validate()
    profile = PROFILES / (run + '.json')
    profile.write_text(json.dumps(c, indent=2) + '\n')
    task = package(profile, output)
    archive = output / 'science-training.tar.gz'
    doc = yaml.safe_load(task.read_text())
    doc['resources']['priority'] = 90
    task.write_text(yaml.safe_dump(doc, sort_keys=False))
    row = dict(hardware='v6e', task='qubit', model=model, run_id=run, parallel_v4_job=old_job,
               profile=str(profile.relative_to(ROOT)), pool='tpuswarm-v6e32-east5b-qwen35',
               zone='us-east5-b', priority=90, package_dir=str(output.relative_to(ROOT)),
               archive=str(archive.relative_to(ROOT)), task_yaml=str(task.relative_to(ROOT)),
               code_uri=doc['envs']['RAY_TRAIN_CODE'],
               archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
               yaml_sha256=hashlib.sha256(task.read_bytes()).hexdigest())
    science_manifest = json.loads((output / 'manifest.json').read_text())
    science_manifest['task_sha256'] = row['yaml_sha256']
    (output / 'manifest.json').write_text(json.dumps(science_manifest, indent=2) + '\n')
    (output / 'submission-manifest.json').write_text(json.dumps(row, indent=2) + '\n')
    manifest = json.loads((HERE / 'jobs.json').read_text())
    manifest['jobs'] = [r for r in manifest['jobs'] if r['run_id'] != run] + [row]
    (HERE / 'jobs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(row), flush=True)
