"""Build both hardware variants; never uploads or submits jobs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from tpu.swarm.ray_train.build import build
from tpu.swarm.ray_train.config import Config
from tpu.science.package_training import package

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PROFILES = ROOT / 'tpu/swarm/ray_train/profiles'
POOLS = {'v4': 'tpuswarm-v4-64-central2-qwen35-erdos', 'v5p': 'tpuswarm-v5p32-east5a-erdos'}


def profiles():
    rows = []
    for hardware in ('v4', 'v5p'):
        for task in ('ac2', 'cp26', 'qubit', 'circuit'):
            for model in ('qwen', 'gemma', 'muse'):
                lr = 'lr15e4' if model == 'qwen' else 'lr4e5'
                old_task = 'q20' if task == 'qubit' else task
                original = PROFILES / f'single-v4-{model}-{old_task}-grpo-{lr}-s1-20260919.json'
                config = json.loads(original.read_text())
                run = f'fresh-{hardware}-{model}-{task}-grpo-{lr}-s1-20260919'
                config.update(run_id=run, root=f'~/.cache/{run}')
                if hardware == 'v5p':
                    suffix = '-bwd256' if model == 'gemma' else ''
                    native_path = PROFILES / f'native-v5p-{model}-ac2-grpo-lr4e5-s1{suffix}-retryfix-004.json'
                    native = json.loads(native_path.read_text())
                    config.update(accelerator='tpu-v5p-32', hosts=4, zone='us-east5-a',
                                  bucket=native['bucket'], base_bundle=native['base_bundle'],
                                  trainer=copy.deepcopy(native['trainer']), inference=copy.deepcopy(native['inference']))
                    config['trainer']['request_timeout'] = 28800
                    # One four-chip engine per host, including Muse; its older
                    # TP2 profile used two engines per host and a different cache.
                    config['inference'].update(tp=4, prefix_caching=True, request_timeout=21600,
                                               memory_utilization=.75 if model == 'muse' else .8)
                    for key in ('hf', 'orbax'):
                        config['cache'][key] = native['cache'][key]
                    config['cache']['trainer_compile_seed'] = native['cache']['trainer_compile']
                    config['cache']['inference_compile_seed'] = (
                        native['cache']['inference_compile'] if native['inference']['tp'] == 4 else '')
                for phase in ('trainer', 'inference'):
                    config['cache'][phase + '_compile'] = f"{config['bucket']}/{run}-{phase}_compile-v1"
                config['client_env']['TTD_SICK_MARKER'] = f'/home/gcpuser/.cache/{run}/runs/{run}/ENGINE-SICK'
                if task == 'qubit':
                    config['client_env']['SCIENCE_ROUTING_SUITE'] = 'full'
                path = PROFILES / f'{run}.json'
                Config.from_dict(config).validate()
                path.write_text(json.dumps(config, indent=2) + '\n')
                rows.append(dict(hardware=hardware, task=task, model=model, run_id=run,
                                 profile=str(path.relative_to(ROOT)), pool=POOLS[hardware],
                                 priority=100 if task in ('ac2', 'cp26') else 50,
                                 package_dir=f'.science/packages/fresh-grpo-20260919/{hardware}/{model}/{task}'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles-only', action='store_true')
    args = parser.parse_args()
    rows = profiles()
    for row in rows:
        if args.profiles_only:
            continue
        profile, output = ROOT / row['profile'], ROOT / row['package_dir']
        if row['task'] in ('qubit', 'circuit'):
            task = package(profile, output)
            archive = output / 'science-training.tar.gz'
        else:
            archive, _, task = build(profile, output)
            doc = yaml.safe_load(task.read_text())
            audit = (ROOT / 'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text()
            doc['run'] = "set -euo pipefail\npython3 - <<'FRESH_CLEAN_HOST'\n" + audit + '\nFRESH_CLEAN_HOST\n' + doc['run']
            task.write_text(yaml.safe_dump(doc, sort_keys=False))
        doc = yaml.safe_load(task.read_text())
        # Priority belongs to resources in this deployed SkyPilot schema.
        doc['resources']['priority'] = row['priority']
        task.write_text(yaml.safe_dump(doc, sort_keys=False))
        row.update(archive=str(archive.relative_to(ROOT)), task_yaml=str(task.relative_to(ROOT)),
                   code_uri=doc['envs']['RAY_TRAIN_CODE'], archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   yaml_sha256=hashlib.sha256(task.read_bytes()).hexdigest())
        if row['task'] in ('qubit', 'circuit'):
            manifest_path = output / 'manifest.json'
            science_manifest = json.loads(manifest_path.read_text())
            science_manifest['task_sha256'] = row['yaml_sha256']
            manifest_path.write_text(json.dumps(science_manifest, indent=2) + '\n')
        (output / 'submission-manifest.json').write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps(dict(run_id=row['run_id'], packaged=True)), flush=True)
    (HERE / 'jobs.json').write_text(json.dumps(dict(jobs=rows, deferred=['rglru']), indent=2) + '\n')


if __name__ == '__main__':
    main()
