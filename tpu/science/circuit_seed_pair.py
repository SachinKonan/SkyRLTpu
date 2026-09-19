"""Promote one fully graded circuit bootstrap unchanged into a matched pair."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from .bootstrap import identity, read, save
from .seed_pool import sampling_contract, verify_pool
from .seed_handoff import bind_bundle, seed_job_status, storage_command
from tpu.swarm.ray_train.config import Config


def verify(client, source, targets):
    folder = Path(client) / 'bootstrap'
    record, summary = read(folder / 'contract.json'), read(folder / 'complete.json')
    contract = record['contract']
    from .training_env import task_prompt
    from .placement_suite_guard import validate_state
    expected_prompt = task_prompt('placement', environment=dict(source.client_env,
        SCIENCE_PLACEMENT_BACKEND=source.science_placement_backend, SCIENCE_ACCELERATOR=source.accelerator))
    if (Config.from_dict(contract['config']) != source
            or identity(contract) != record['sha256']
            or summary['contract_sha256'] != record['sha256']
            or contract['implementation_sha256'] != hashlib.sha256(Path(__file__).with_name('bootstrap.py').read_bytes()).hexdigest()
            or contract['prompt'] != expected_prompt
            or contract['policy'] != 'frozen-base-no-adapter'
            or not source.bootstrap_only or source.bootstrap_layers != 1
            or source.science_task != 'placement' or summary['layers'] != 1
            or summary['optimizer_steps'] != 0):
        raise ValueError('incompatible circuit bootstrap contract')
    def signature(c):
        s = sampling_contract(c)
        for k in ('TTD_ADV_ESTIMATOR', 'TTD_ADV_PIECEWISE_RHO', 'TTD_ADV_PIECEWISE_INVALID_REWARD'):
            s['env'].pop(k, None)
        return s
    if len(targets) != 2 or any(t.bootstrap_layers or t.bootstrap_only
            or t.seed_pool_sha256 != '0'*64 or signature(t) != signature(source) for t in targets):
        raise ValueError('training pair differs from bootstrap sampling contract')
    if {t.client_env['TTD_ADV_ESTIMATOR'] for t in targets} != {'mean_baseline', 'piecewise_valid_entropic_centered_adaptive'}:
        raise ValueError('expected GRPO/PWC pair')
    for t in targets:
        if t.has_adaptive_pwc_overlay and (t.client_env.get('TTD_ADV_PIECEWISE_RHO') != '0.5'
                or t.client_env.get('TTD_ADV_PIECEWISE_INVALID_REWARD') != '0'):
            raise ValueError('unexpected PWC parameters')
    roots = read(folder / 'roots.json')
    plan = read(folder / 'layer-0-plan.json')
    if len(roots) != 16 or len(plan) != 16 or contract['groups'] != 16 or contract['size'] != 32:
        raise ValueError('expected sixteen groups of thirty-two drafts')
    root_ids = {r['id'] for r in roots}
    if len(root_ids) != 16 or {p['root_id'] for p in plan} != root_ids:
        raise ValueError('duplicate or missing draft roots')
    rows = []
    for i, parent in enumerate(plan):
        paths = sorted((folder / 'layer-0' / f'group-{i:03d}').glob('grade-*.json'))
        if parent['repair_parent_id'] is not None or len(paths) != 32:
            raise ValueError('incomplete draft group or unexpected repair')
        for path in paths:
            r = read(path)
            if (r['root_id'] != parent['root_id'] or r['repair_parent_id'] is not None
                    or r['layer'] != 0 or r['correctness'] not in (0,1)
                    or not math.isfinite(r['reward'])
                    or (r['correctness'] == 1 and not (r['code'] and 0 < r['reward'] <= 1))
                    or (r['correctness'] == 0 and r['reward'] != 0)):
                raise ValueError('invalid draft grading record')
            rows.append(r)
    valid = {r['id']:r for r in rows if r['correctness'] == 1}
    if (len({r['id'] for r in rows}) != 512 or summary['total'] != 512
            or summary['valid'] != len(valid)
            or read(folder/'layer-0-summary.json') != dict(groups=16,total=512,valid=len(valid))):
        raise ValueError('draft grading totals disagree')
    path = Path(client)/'tinker_log'/source.run_id/'puct_sampler_step_000000.json'
    pool = verify_pool(path, summary['pool_sha256'])
    admitted = [s for s in pool['states'] if s.get('code')]
    if len(admitted) != summary['retained'] or not 1 <= len(admitted) <= 32:
        raise ValueError('invalid retained pool size')
    if len({s['code'] for s in admitted}) != len(admitted):
        raise ValueError('duplicate retained programs')
    for s in admitted:
        validate_state(s)
        r = valid.get(s['id'])
        if not r or s['code'] != r['code'] or s['value'] != r['reward']:
            raise ValueError('retained program has no matching valid grade')
    return path, summary


def run(plan_path):
    plan = read(plan_path); root = Path(plan['directory']); root.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ, **plan['environment'])
    source = Config.load(plan['bootstrap_profile'])
    targets = [Config.load(t['profile']) for t in plan['targets']]
    def pinned():
        for path,digest in plan['source_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('handoff source changed: '+path)
    def storage(args, missing=False):
        return storage_command(plan['gcloud'],args,env,allow_missing=missing)
    pinned(); until=time.monotonic()+plan.get('deadline_seconds',172800)
    while True:
        pending,error=seed_job_status(plan['sky'],[dict(job_id=plan['bootstrap_job'])],env)
        if not error and not pending:
            break
        save(root/'status.json',dict(event='waiting_for_bootstrap',pending=pending,error=error,time=time.time()))
        if time.monotonic()>until:raise TimeoutError('bootstrap completion deadline')
        time.sleep(60)
    pinned()
    client=root/'client';client.mkdir(exist_ok=True)
    storage(['storage','rsync','--recursive',source.run_gcs+'/client/',str(client)])
    pool,summary=verify(client,source,targets)
    report=dict(source_run=source.run_id,source_job=plan['bootstrap_job'],**summary)
    save(root/'seed-import.json',report)
    # Verify and stage BOTH arms before any submission.
    for target,spec in zip(targets,plan['targets']):
        uri=target.run_gcs+'/client/tinker_log/'+target.run_id+'/'+pool.name
        storage(['storage','cp','--no-clobber',str(pool),uri])
        if identity(json.loads(storage(['storage','cat',uri]))) != summary['pool_sha256']:
            raise ValueError('seed upload checksum mismatch')
        storage(['storage','cp','--no-clobber',str(root/'seed-import.json'),target.run_gcs+'/client/seed-import.json'])
        bind_bundle(spec['template'],spec['package'],summary['pool_sha256'])
    for spec in plan['targets']:
        pinned()
        subprocess.run([plan['submit_python'],plan['submit_script'],spec['package'],plan['pool']],env=env,check=True,timeout=900)
    save(root/'status.json',dict(event='pair_submitted',seeds=report,
        submissions=[read(Path(s['package'])/'submission.json') for s in plan['targets']],time=time.time()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('plan');args=parser.parse_args()
    try:run(args.plan)
    except Exception as exc:
        plan=read(args.plan);save(Path(plan['directory'])/'status.json',dict(event='failed',error=str(exc),time=time.time()))
        raise
