import json
from dataclasses import replace
from pathlib import Path
import pytest
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.farm_admission import assignments
from tpu.science.training_setup import split_roles

PROFILES = Path('tpu/swarm/ray_train/profiles')

@pytest.mark.parametrize('model,lr', [('qwen','15e4'),('muse','4e5'),('gemma','4e5')])
def test_v5p64_cpu_placement_bootstraps_and_trains(model, lr):
    raw=json.loads((PROFILES/f'fresh-v5p-{model}-circuit-grpo-lr{lr}-s1-20260919.json').read_text())
    raw.update(accelerator='tpu-v5p-64', hosts=8, zone='us-central1-a')
    raw['client_env']['NUM_EPOCHS']='10'
    cfg=Config.from_dict(raw)
    assert cfg.inference_hosts == 7
    assert split_roles('placement',[0],list(range(1,8)),placement_backend='cpu',accelerator=cfg.accelerator)==([0],list(range(1,8)),None)
    with pytest.raises(ValueError):
        replace(cfg, science_placement_backend='tpu').validate()
    with pytest.raises(ValueError):
        split_roles('placement',[0],[1,2,3],placement_backend='cpu',accelerator=cfg.accelerator)


def test_spare_gemma_farm_admits_rg_after_ac2_reservation():
    rows=[dict(job_id=i,run_id=n,status='RUNNING') for i,n in [(1,'gemma-rglru'),(2,'gemma-ac2'),(3,'qwen-ac2')]]
    targets={r['job_id']:dict(run_id=r['run_id'],model=r['run_id'].split('-')[0]) for r in rows}
    farms=[dict(job_id=i,models=['gemma'],url=f'g{i}',state='unleased') for i in [4,5]]
    assert assignments(rows,farms,targets)=={1:['g5'],2:['g4'],3:[]}
    assert assignments(rows,farms[:1],targets)=={1:[],2:['g4'],3:[]}
    rows[1]['status']='PENDING'
    targets.pop(2)
    assert assignments(rows,farms,targets)[1]==[]
    rows[1]['status']='SUCCEEDED'
    assert assignments(rows,farms,targets)[1]==['g4']


def test_attested_target_skips_legacy_and_incompatible_farms():
    rows=[dict(job_id=10,run_id='muse-ac2',status='RUNNING')]
    target={10:dict(run_id='muse-ac2',model='muse',compatibility_sha256='expected')}
    farms=[dict(job_id=i,models=['muse'],url=str(i),state='unleased',
                capabilities={'compatibility_sha256':sha} if sha else None)
           for i,sha in [(1,None),(2,'wrong'),(3,'expected')]]
    assert assignments(rows,farms,target)=={10:['3']}


def test_circuit_queue_advances_only_with_step_ten_and_no_live_hosts(tmp_path, monkeypatch):
    import importlib.util
    path=Path('tpu/science/results/reallocation-10step-20260921/circuit_queue.py')
    spec=importlib.util.spec_from_file_location('circuit_queue',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'HERE',tmp_path)
    state=dict(index=0,attempt=1,phase='launched',unit='owned-unit',external_ips=['host'])
    module.write(tmp_path/'circuit-queue-state.json',state)
    monkeypatch.setattr(module,'completed',lambda *args:True)
    monkeypatch.setattr(module,'parallel',lambda *args:[dict(exit=0,stdout=json.dumps(dict(ActiveState='active')))])
    rows=[dict(run_id='gemma'),dict(run_id='muse')]
    assert module.tick(None,rows)['index']==0
    monkeypatch.setattr(module,'parallel',lambda *args:[dict(exit=0,stdout=json.dumps(dict(ActiveState='inactive')))])
    assert module.tick(None,rows)==dict(index=1,attempt=0,phase='queued')


def test_circuit_queue_blocks_after_three_failed_attempts(tmp_path,monkeypatch):
    import importlib.util
    spec=importlib.util.spec_from_file_location('circuit_queue',Path('tpu/science/results/reallocation-10step-20260921/circuit_queue.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'HERE',tmp_path)
    module.write(tmp_path/'circuit-queue-state.json',dict(index=0,attempt=3,phase='launched',unit='owned-unit',external_ips=['host']))
    monkeypatch.setattr(module,'completed',lambda *args:False)
    monkeypatch.setattr(module,'parallel',lambda *args:[dict(exit=0,stdout=json.dumps(dict(ActiveState='failed')))])
    result=module.tick(None,[dict(run_id='gemma'),dict(run_id='muse')])
    assert result['index']==0 and result['phase']=='blocked' and 'unit' not in result
