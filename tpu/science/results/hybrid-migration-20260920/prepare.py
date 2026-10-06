"""Build reviewed AC2 canaries and farm upgrades; never uploads or submits."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

from tpu.swarm.ray_train.build import build
from tpu.swarm.ray_train.config import Config

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent
PROFILES = ROOT / 'tpu/swarm/ray_train/profiles'
OUTPUT = ROOT / '.science/hybrid-migration/packages'
SEED_ROOT = Path('/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-science-placement')


def read(name):
    return json.loads((PROFILES / (name + '.json')).read_text())


def write_profile(data):
    path = PROFILES / (data['run_id'] + '.json')
    Config.from_dict(data)
    path.write_text(json.dumps(data, indent=2) + '\n')
    return path


def main():
    originals = {r['model']: r for r in json.loads(
        (HERE.parent / 'capacity-seven-20260920/jobs.json').read_text())['jobs'] if r['task'] == 'ac2'}
    records = []
    for model, region in [('qwen', 'east5b'), ('gemma', 'east5b'), ('muse', 'central1b')]:
        record = originals[model]
        if model == 'qwen':
            data = read('science-v6e-qwen-qubit-grpo-lr15e4-s1-20260920')
            original = read('capacity-v5p-qwen-ac2-grpo-15e4-20260920')
            data['client_env'] = copy.deepcopy(original['client_env'])
            data.pop('science_routing_slots_per_host', None)
        else:
            data = read(f'math-v6e-{model}-ac2-grpo-lr4e5-s1-20260920')
        run = f'hybrid-v6e-{model}-ac2-canary-{region}-20260920'
        bucket = 'gs://sk7524-tinker-tpu-us-' + ('east5' if region == 'east5b' else 'central1')
        data.update(run_id=run, root='~/.cache/' + run, bucket=bucket,
                    zone='us-east5-b' if region == 'east5b' else 'us-central1-b',
                    bootstrap_layers=0, bootstrap_all_hosts=False, bootstrap_max_drafts=0,
                    bootstrap_target_valid=0, checkpoint_resume=True, max_restarts_on_errors=3,
                    systemd_runtime=True)
        data['client_env'].update(NUM_EPOCHS='2', TTD_LEAGUE_PIPELINE='1',
            TTD_SICK_MARKER=f'/home/gcpuser/.cache/{run}/runs/{run}/ENGINE-SICK')
        data['inference'].update(external_pool_updates=True, external_pool_lease_scope='run',
            external_pool_require_initial=True, external_pool_scheduler=True, external_pool_attestation=True,
            external_pool_engines=4, external_pool_max_n=32, external_pool_max_concurrent_requests=4,
            external_pool_prepare_timeout=1800, external_pool_rpc_timeout=10,
            external_pool_lease_seconds=300, external_pool_heartbeat_seconds=30,
            external_pool_health_grace_seconds=90)
        for role in ('trainer', 'inference'):
            data['cache'][role + '_compile'] = bucket + '/' + run + '-' + role + '-compile-v1'
        profile = write_profile(data)
        output = OUTPUT / run
        output.mkdir(parents=True, exist_ok=True)
        seed = SEED_ROOT / record['seed_file']
        raw = seed.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record['seed_file_sha256']
        pool = json.loads(raw)
        assert hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest() == record['seed_pool_sha256']
        assert pool['step'] == 0 and len(pool['states']) == record['retained']
        shutil.copyfile(seed, output / 'seed-pool.json')
        records.append(dict(kind='training', model=model, run_id=run, profile=str(profile.relative_to(ROOT)),
            output=str(output.relative_to(ROOT)), pool='tpuswarm-v6e32-' + (
                'east5b-qwen35' if region == 'east5b' else 'central1b'), priority=120,
            seed_file=str((output / 'seed-pool.json').relative_to(ROOT)),
            seed_file_sha256=record['seed_file_sha256'], seed_pool_sha256=record['seed_pool_sha256'],
            seed_source_run=record['source_run_id'], retained=record['retained'],
            seed_destination=f'{bucket}/ray-training/{run}/client/tinker_log/{run}/puct_sampler_step_000000.json'))
        farm = read('qwen-multilora-farm-2-20260919' if model == 'qwen' else f'{model}-multilora-farm-1-20260919')
        farm_run = f'hybrid-v432-{model}-farm-20260920'
        farm.update(run_id=farm_run, root='~/.cache/' + farm_run, max_restarts_on_errors=3)
        farm['inference'].update(external_pool_attestation=True, farm_drain_timeout=120)
        farm['cache']['inference_compile_seed'] = farm['cache']['inference_compile']
        farm['cache']['inference_compile'] = farm['bucket'] + '/' + farm_run + '-inference-compile-v1'
        farm_profile = write_profile(farm)
        records.append(dict(kind='farm', model=model, run_id=farm_run, profile=str(farm_profile.relative_to(ROOT)),
            output=str((OUTPUT / farm_run).relative_to(ROOT)), pool='tpuswarm-v4-32-central2-smoke', priority=120,
            replaces_job={'qwen':1251, 'gemma':1252, 'muse':1254}[model]))
    # Build only after every profile exists so all archives contain the same set.
    for row in records:
        output = ROOT / row['output']
        archive, uri, task = build(ROOT / row['profile'], output)
        import yaml
        doc = yaml.safe_load(task.read_text())
        doc['resources']['priority'] = row['priority']
        audit = (ROOT / 'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text()
        doc['run'] = "set -euo pipefail\npython3 - <<'CLEAN_HOST'\n" + audit + '\nCLEAN_HOST\n' + doc['run']
        task.write_text(yaml.safe_dump(doc, sort_keys=False))
        row.update(archive=str(archive.relative_to(ROOT)), code_uri=uri,
                   archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   task=str(task.relative_to(ROOT)), task_sha256=hashlib.sha256(task.read_bytes()).hexdigest())
    (HERE / 'prepared.json').write_text(json.dumps(dict(submitted=False, jobs=records), indent=2) + '\n')
    print(json.dumps([dict(kind=r['kind'], model=r['model'], pool=r['pool']) for r in records], indent=2))


if __name__ == '__main__':
    main()
