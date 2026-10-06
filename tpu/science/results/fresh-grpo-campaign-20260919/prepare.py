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
POOLS = {'v4': 'tpuswarm-v4-64-central2-qwen35-erdos', 'v5p': 'tpuswarm-v5p32-east5a-erdos', 'v6e': 'tpuswarm-v6e32-east5b-qwen35'}


def profiles(tasks=None, hardware_types=None):
    rows = []
    for hardware in (hardware_types or ('v4', 'v5p')):
        for task in ('ac2', 'cp26', 'qubit', 'circuit', 'rglru'):
            if (tasks and task not in tasks) or (hardware == 'v6e' and task != 'rglru'):
                continue
            for model in ('qwen', 'gemma', 'muse'):
                lr = 'lr15e4' if model == 'qwen' else 'lr4e5'
                old_task = 'ac2' if task == 'rglru' else 'q20' if task == 'qubit' else task
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
                if hardware == 'v6e':
                    native = json.loads((PROFILES / f'science-q20-v6e-{model}-grpo-hedge-20260918.json').read_text())
                    config.update(accelerator='tpu-v6e-32', zone='us-east5-b', bucket=native['bucket'],
                                  trainer=copy.deepcopy(native['trainer']), inference=copy.deepcopy(native['inference']))
                    for key in ('hf', 'orbax'):
                        config['cache'][key] = native['cache'][key]
                    for phase in ('trainer', 'inference'):
                        config['cache'][phase + '_compile_seed'] = native['cache'][phase + '_compile']
                for phase in ('trainer', 'inference'):
                    config['cache'][phase + '_compile'] = f"{config['bucket']}/{run}-{phase}_compile-v1"
                config['client_env']['TTD_SICK_MARKER'] = f'/home/gcpuser/.cache/{run}/runs/{run}/ENGINE-SICK'
                if task == 'qubit':
                    config['client_env']['SCIENCE_ROUTING_SUITE'] = 'full'
                if task == 'rglru':
                    config['arena_grader_rank'] = 1
                    config['client_env'].update(TTD_ENV='recurrent_gemma', TTD_PROBLEM_TYPE='rg_lru',
                                                ARENA_WAIT_TIMEOUT='14400', EVAL_TIMEOUT='14400')
                path = PROFILES / f'{run}.json'
                Config.from_dict(config).validate()
                encoded = json.dumps(config, indent=2) + '\n'
                receipt = ROOT / f'.science/packages/fresh-grpo-20260919/{hardware}/{model}/{task}/submission.json'
                if receipt.exists():
                    if not path.exists() or path.read_text() != encoded:
                        raise RuntimeError(f'Refusing to change submitted profile: {run}')
                else:
                    path.write_text(encoded)
                rows.append(dict(hardware=hardware, task=task, model=model, run_id=run,
                                 profile=str(path.relative_to(ROOT)), pool=POOLS[hardware],
                                 priority=100 if task in ('ac2', 'cp26') else 50,
                                 package_dir=f'.science/packages/fresh-grpo-20260919/{hardware}/{model}/{task}'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles-only', action='store_true')
    parser.add_argument('--task', action='append', choices=('ac2','cp26','qubit','circuit','rglru'),
                        help='Only prepare selected tasks; retain other indexed packages')
    parser.add_argument('--hardware', action='append', choices=('v4','v5p','v6e'))
    args = parser.parse_args()
    index_path = HERE / 'jobs.json'
    previous = {r['run_id']: r for r in json.loads(index_path.read_text())['jobs']} if index_path.exists() else {}
    rows = profiles(args.task, args.hardware)
    for row in rows:
        # Never repackage a submitted run: recovery uses its original archive.
        if (ROOT / row['package_dir'] / 'submission.json').exists():
            if row['run_id'] not in previous:
                raise RuntimeError('Submitted run is missing its package index')
            row.update(previous[row['run_id']])
            continue
        if args.profiles_only:
            if row['run_id'] in previous:
                row.update(previous[row['run_id']])
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
    previous.update({r['run_id']:r for r in rows})
    index_path.write_text(json.dumps(dict(jobs=list(previous.values()), deferred=[]), indent=2) + '\n')


if __name__ == '__main__':
    main()
