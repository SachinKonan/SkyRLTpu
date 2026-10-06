"""Add the approved fifth east workload without rebuilding active jobs."""
import hashlib,json
from pathlib import Path
import yaml
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.build import build
from importlib import import_module
base=import_module('tpu.science.results.reallocation-10step-20260921.prepare')

def main():
    run='fresh-v6e-qwen-rglru-grpo-lr15e4-s1-20260919-fix1'
    manifest=json.loads((base.HERE/'prepared.json').read_text())
    assert not any(r['run_id']==run for r in manifest['jobs']), 'fifth run already prepared'
    preflight=json.loads((base.HERE/'fifth-east-preflight.json').read_text())
    assert preflight['verified'] and preflight['run_id']==run and preflight['step']==5
    d=base.read(run);base.hybrid(d)
    # AC2 owns the scarce farms first. Local inference remains useful while
    # this run waits for a matching, exclusively leased farm to become free.
    d['inference']['external_pool_require_initial']=False
    d['client_env']['NUM_EPOCHS']='10'
    d.update(systemd_runtime=True,checkpoint_resume=True,resume_min_checkpoint_step=5,
             max_restarts_on_errors=3,bootstrap_layers=0,bootstrap_all_hosts=False,
             bootstrap_max_drafts=0,bootstrap_target_valid=0)
    for role in ['trainer','inference']:
        d['cache'][role+'_compile_seed']=d['cache'][role+'_compile']
        d['cache'][role+'_compile']=d['bucket']+'/reallocation-10step-20260921/'+run+'/'+role
    cfg=Config.from_dict(d)
    assert cfg.arena_grader_rank==1 and cfg.inference_hosts==3
    label=run+'-10step-deployment';profile=base.PROFILES/(label+'.json')
    profile.write_text(json.dumps(d,indent=2)+'\n')
    archive,uri,task=build(profile,base.OUT/label)
    doc=yaml.safe_load(task.read_text());doc['resources']['priority']=110
    task.write_text(yaml.safe_dump(doc,sort_keys=False))
    row=dict(kind='rglru',model='qwen',run_id=run,profile=str(profile.relative_to(base.ROOT)),
        pool='tpuswarm-v6e32-east5b-qwen35',priority=110,resume_min_checkpoint_step=5,
        supersedes=[1293],label=label,bucket=d['bucket'],epochs=10,archive=str(archive.relative_to(base.ROOT)),
        code_uri=uri,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        task=str(task.relative_to(base.ROOT)),task_sha256=hashlib.sha256(task.read_bytes()).hexdigest())
    manifest['jobs'].append(row);manifest['defer_jobs']=[j for j in manifest['defer_jobs'] if j!=1293]
    (base.HERE/'prepared.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(row,indent=2))
if __name__=='__main__':main()
