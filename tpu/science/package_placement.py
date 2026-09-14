"""Build local, reviewable placement probe/model tasks; never upload or submit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import yaml

from tpu.swarm.ray_train.build import build
from tpu.swarm.ray_train.config import Config
from .challenge_contract import CASES


def package(profile, output, *, probe=False):
    config = Config.load(profile)
    if not config.inference_only or not config.placement_ranks:
        raise ValueError('this packager prepares reference/sampling pilots, not RL training')
    root = Path(__file__).resolve().parents[2]
    out = Path(output).resolve()
    archive, _, task = build(profile, out)
    files = {}
    names = ('__init__.py sample.py rewards.py isolation.py worker.py cgroup_limits.py '
             'challenge_contract.py challenge_seed.py challenge_seed_jax.py '
             'challenge_candidate_child.py challenge_score_child.py placement_task.py '
             'placement_ray.py placement_slots.py placement_model_driver.py placement_model_host.py '
             'placement_probe_driver.py placement_probe_bootstrap.py prepare_placement_host.py '
             'requirements-challenge-pilot.lock requirements-placement-candidate.lock').split()
    for name in names:
        path = root/'tpu/science'/name
        files[str(path.relative_to(root))] = path
    prompt = (root/'tpu/science/prompts/placement-jax-v5p.txt' if config.accelerator == 'tpu-v5p-32'
              else root/'tpu/science/prompts/placement-jax.txt')
    files[str(prompt.relative_to(root))] = prompt
    challenge = root/'.science/challenge-probe'
    for path in challenge.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and (
                path.suffix == '.py' or path.name == 'sources.json' or set(CASES) & set(path.parts)):
            files[str(path.relative_to(root))] = path
    for case in CASES:
        for name in ('netlist.pb.txt', 'initial.plc'):
            if not (challenge/'external/MacroPlacement/Testcases/ICCAD04'/case/name).is_file():
                raise FileNotFoundError(f'missing pinned {case}/{name}')
        files[f'.science/placement-inputs/{case}-problem.npz'] = (
            root/'tpu/science/results/placement/scaffold-pilot'/f'{case}-problem.npz')
    for path in (root/'tpu/swarm/bench').glob('*.py'):
        files[str(path.relative_to(root))] = path
    destination = out/'placement-bundle.tar.gz'
    with tarfile.open(archive, 'r:gz') as old, tarfile.open(destination, 'w:gz') as bundle:
        for member in old.getmembers():
            if member.name not in files:
                bundle.addfile(member, old.extractfile(member) if member.isfile() else None)
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    doc = yaml.safe_load(task.read_text())
    run_id = config.run_id.replace('-pilot-', '-probe-') if probe else config.run_id
    doc['name'] = run_id
    doc['envs'].update(RAY_TRAIN_CODE=config.bucket+'/code-bundles/placement-'+digest+'.tar.gz',
        RAY_TRAIN_CODE_SHA256=digest, SCIENCE_RUN_ID=run_id, SCIENCE_ACCELERATOR=config.accelerator,
        SCIENCE_RESULT_URI=config.bucket+'/science-results/'+run_id+'.tar.gz',
        PLACEMENT_TPU_RANKS=','.join(map(str, config.placement_ranks)),
        PLACEMENT_TPU_CHIPS=config.client_env.get('PLACEMENT_TPU_CHIPS', '0'))
    audit = (root/'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text()
    prelude = "set -euo pipefail\npython3 - <<'PLACEMENT_CLEAN_HOST'\n"+audit+"\nPLACEMENT_CLEAN_HOST\n"
    if probe:
        entry = 'exec timeout --signal=TERM --kill-after=60s 30m python3 -m tpu.science.placement_probe_bootstrap'
        run = doc['run'][:doc['run'].index('exec python3 -m tpu.swarm.ray_train.bootstrap')]
        run += 'cd "$code"\npython3 -m tpu.science.prepare_placement_host\n'+entry+'\n'
    else:
        run = doc['run'].replace('exec python3 -m tpu.swarm.ray_train.bootstrap',
            'cd "$code"\npython3 -m tpu.science.prepare_placement_host\n'
            'exec timeout --signal=TERM --kill-after=120s 4h python3 -m tpu.science.placement_model_host')
    doc['run'] = prelude+run
    task = out/(run_id+'.yaml')
    task.write_text(yaml.safe_dump(doc, sort_keys=False))
    main_commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    discover_commit = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD:third_party/discover'], text=True).strip()
    (out/'manifest.json').write_text(json.dumps(dict(run_id=run_id,sha256=digest,
        main_commit=main_commit,discover_commit=discover_commit,
        accelerator=config.accelerator,grading_ranks=config.placement_ranks,
        grading_chips=doc['envs']['PLACEMENT_TPU_CHIPS'],submitted=False,
        files={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}), indent=2)+'\n')
    return task


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    print(package(args.profile, args.output, probe=args.probe))
