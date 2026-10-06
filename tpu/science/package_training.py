"""Package native science training with pinned, isolated grading inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import yaml

from tpu.swarm.ray_train.build import build
from tpu.swarm.ray_train.config import Config


CPU_BUNDLE_SHA256 = '2445286f0b92ee4977d839f69359baf7b8d483523e4b3e172bb493c6163f9c50'
# A farm grades placement for whichever trainer leases it, so it carries the
# campaign's circuit grader exactly: CPU five-minute runtime, fast proxy
# helper and the three-start ibm17 inputs.
FARM_PLACEMENT_HELPER = 'fast_proxy_v1'
FARM_PLACEMENT_STARTS = 'abuplace-xplace-three-starts-v1'


def science_tasks(config):
    """Science grader environments this profile's hosts must prepare."""
    if config.inference_only:
        return sorted({'routing', 'placement'} & set(config.grading_families))
    return [config.science_task] if config.science_task else []


def placement_settings(config):
    if config.inference_only and 'placement' in config.grading_families:
        family = config.grading_families['placement']
        return dict(backend='cpu', runtime='cpu300-4g-v1', helper=FARM_PLACEMENT_HELPER,
                    starts=FARM_PLACEMENT_STARTS, slots=family['slots_per_host'],
                    cpus=family['cpus'], memory_gib=family['memory_gib'])
    return dict(backend=config.science_placement_backend, runtime=config.science_placement_runtime,
                helper=config.client_env.get('SCIENCE_PLACEMENT_HELPER', 'none'),
                starts=config.client_env.get('SCIENCE_PLACEMENT_STARTS', ''),
                slots=config.science_placement_slots_per_host, cpus=config.science_placement_cpus_per_case,
                memory_gib=config.science_placement_memory_gib)


def package(profile, output):
    config = Config.load(profile)
    # Inference farms that grade a science family need the same grader
    # environment as a science trainer; AC2-only farms use the generic bundle.
    tasks = science_tasks(config)
    task_name = '+'.join(tasks)
    if not tasks or (config.inference_only and not (config.bootstrap_only or config.grading_families)):
        raise ValueError('expected a science training profile or a farm with a science grading family')
    placement = placement_settings(config)
    root = Path(__file__).resolve().parents[2]
    out = Path(output).resolve()
    native, _, task = build(profile, out, _executor_only=True)
    files = {str(p.relative_to(root)): p for p in (root / 'tpu/science').glob('*.py')}
    for name in ('prepare_cpu_host.sh', 'requirements-cpu.lock', 'requirements-challenge-pilot.lock',
                 'requirements-placement-candidate.lock', 'prompts/rendered/routing.txt',
                 'prompts/placement-jax-v6e.txt', 'prompts/placement-jax-cpu.txt', 'prompts/placement-fast-proxy-cpu-v1.txt',
                 'prompts/placement-fast-proxy-cpu-ibm17-v2.txt', 'manifests/routing-v1.json'):
        files['tpu/science/' + name] = root / 'tpu/science' / name
    if 'placement' in tasks and placement['helper'] == 'fast_proxy_v1':
        for name in ('__init__.py', 'build.py', 'congestion.c', 'LICENSE-AbuPlace', 'README.md'):
            files['tpu/science/fast_proxy/' + name] = root / 'tpu/science/fast_proxy' / name
    if 'placement' in tasks:
        from .challenge_contract import CASES
        from .placement_warm_start import verified_inputs, DESTINATION
        folder = root / DESTINATION
        if placement['starts'] == 'abuplace-xplace-three-starts-v1':
            from .placement_start_portfolio import DESTINATION as PORTFOLIO
            folder = root / PORTFOLIO
        elif placement['starts']:
            raise ValueError('unknown placement starting layouts')
        starts = verified_inputs(root, folder=folder)
        files['.science/placement-inputs/manifest.json'] = folder / 'manifest.json'
        for path in (root / '.science/challenge-probe').rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and (
                    path.suffix == '.py' or path.name == 'sources.json' or set(CASES) & set(path.parts)):
                files[str(path.relative_to(root))] = path
        for case in CASES:
            for name in ('netlist.pb.txt', 'initial.plc'):
                path = root / '.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04' / case / name
                if not path.is_file():
                    raise FileNotFoundError(path)
            files[f'.science/placement-inputs/{case}-problem.npz'] = starts[case]
    archive = out / 'science-training.tar.gz'
    with tarfile.open(native) as old, tarfile.open(archive, 'w:gz') as bundle:
        for member in old.getmembers():
            if member.name not in files:
                bundle.addfile(member, old.extractfile(member) if member.isfile() else None)
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    doc = yaml.safe_load(task.read_text())
    doc['envs'].update(RAY_TRAIN_CODE=config.bucket + '/code-bundles/science-training-' + digest + '.tar.gz',
                       RAY_TRAIN_CODE_SHA256=digest, SCIENCE_ACCELERATOR=config.accelerator,
                       SCIENCE_PLACEMENT_BACKEND=placement['backend'],
                       SCIENCE_PLACEMENT_RUNTIME=placement['runtime'],
                       SCIENCE_PLACEMENT_HELPER=placement['helper'],
                       SCIENCE_PLACEMENT_SLOTS_PER_HOST=str(placement['slots']),
                       SCIENCE_PLACEMENT_CPUS_PER_CASE=str(placement['cpus']),
                       SCIENCE_PLACEMENT_MEMORY_GIB=str(placement['memory_gib']),
                       PLACEMENT_TPU_RANKS=','.join(map(str, range(config.hosts))))
    prep = '''cd "$code"
export SCIENCE_WORKER_ROOT="$code"
'''
    if 'routing' in tasks:
        # The runtime bundle lives in every region's bucket; never read another region.
        doc['envs'].update(
            SCIENCE_CPU_BUNDLE=f'{config.bucket}/code-bundles/science-cpu-{CPU_BUNDLE_SHA256}.tar.gz',
            SCIENCE_CPU_SHA256=CPU_BUNDLE_SHA256)
        prep += '''if [ ! -f "$code/.science/routing-prepared" ]; then
  # TPU images lack gcloud-crc32c for sliced downloads; the SHA-256 check below verifies the bundle.
  CLOUDSDK_STORAGE_CHECK_HASHES=never gcloud storage cp "$SCIENCE_CPU_BUNDLE" "$code/cpu-runtime.tar.gz"
  printf '%s  %s\\n' "$SCIENCE_CPU_SHA256" "$code/cpu-runtime.tar.gz" | sha256sum -c -
  tar -xzf "$code/cpu-runtime.tar.gz" -C "$code" .science/routing-task .science/rustup .science/cargo
  rm "$code/cpu-runtime.tar.gz"
  bash "$code/tpu/science/prepare_cpu_host.sh"
  touch "$code/.science/routing-prepared"
fi
'''
    if 'placement' in tasks:
        if 'routing' in tasks:
            # Placement pins differ from routing's; keep them in their own venv.
            doc['envs']['SCIENCE_PLACEMENT_VENV'] = 'venv-placement'
        # The physical grader rank is selected after topology probing. Prepare
        # dependencies on all hosts; CPU profiles grade on every host.
        prep += '''sudo -n apt-get -o DPkg::Lock::Timeout=600 update -qq
sudo -n apt-get -o DPkg::Lock::Timeout=600 install -y -qq bubblewrap build-essential pkg-config libssl-dev
PLACEMENT_TPU_CHIPS=0,1,2,3 python3 -m tpu.science.prepare_placement_host
'''
    audit = (root / 'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text()
    # Full training is bounded by NUM_EPOCHS; the old one-day smoke wrapper can
    # otherwise kill a valid bootstrap/optimizer run while grading is queued.
    executor = ('exec timeout --signal=TERM --kill-after=120s 24h python3 -m tpu.swarm.ray_train.bootstrap'
                if config.training_smoke and not config.bootstrap_layers else
                'exec python3 -m tpu.swarm.ray_train.bootstrap')
    doc['run'] = ('set -euo pipefail\npython3 - <<\'SCIENCE_CLEAN_HOST\'\n' + audit + '\nSCIENCE_CLEAN_HOST\n' +
                  doc['run'].replace('exec python3 -m tpu.swarm.ray_train.bootstrap',
                      prep + executor))
    task.write_text(yaml.safe_dump(doc, sort_keys=False))
    manifest = dict(run_id=config.run_id, task=task_name,
                    main_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                    discover_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD:third_party/discover'],text=True).strip(),
                    sha256=digest, code_uri=doc['envs']['RAY_TRAIN_CODE'], submitted=False,
                    task_sha256=hashlib.sha256(task.read_bytes()).hexdigest(),
                    files={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return task


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(package(args.profile, args.output))
