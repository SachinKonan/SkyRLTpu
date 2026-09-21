"""Admission checks must hold other models on failed or incomplete evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpu.science.ops import routing_relaunch_watch as watch


def rollout():
    r=object.__new__(watch.Rollout)
    r.state=dict(phase='gemma_cycle',jobs={'gemma-train':1500},artifacts={'gemma':{'run_id':'gemma-new'}},retired={})
    return r


def test_recorded_phase_transition_is_resumable(tmp_path,capsys):
    r=rollout();r.path=tmp_path/'state.json'
    r.persist(phase='qwen_launch')
    assert json.loads(r.path.read_text())['phase']=='qwen_launch'
    assert json.loads(capsys.readouterr().out)['phase']=='qwen_launch'


@pytest.mark.parametrize('status',['FAILED','FAILED_SETUP','FAILED_CONTROLLER','CANCELLED'])
def test_failed_gemma_holds_admission(status):
    r=rollout();r.job=lambda jid:dict(job_id=jid,status=status)
    r.gemma_cycle=lambda:pytest.fail('must check training failure first')
    with pytest.raises(RuntimeError,match='Affected new job stopped'):r.tick()


def test_job_name_cannot_adopt_other_pool():
    r=rollout();r.query=lambda *args:[dict(job_id=999,pool='unrelated')]
    with pytest.raises(AssertionError,match='another pool'):r.submit('qwen-train',Path('unused'),'qwen-new')


class Blob:
    generation=12
    size=2048
    def __init__(self,name,rows):self.name=name;self.rows=rows
    def download_as_text(self,**kw):return '\n'.join(json.dumps(r) for r in self.rows)


def evidence(monkeypatch,tmp_path,*,train_error=0,skip=0,remote=True,reload=True,post_reload=True,checkpoint=True):
    monkeypatch.setattr(watch,'BASE',tmp_path)
    r=rollout();prefix='ray-training/gemma-new/';client=prefix+'client/tinker_log/gemma-new/'
    events=[dict(event='hybrid_generated',time=1,route='remote' if remote else 'local',output_tokens=20)]
    if reload:events.append(dict(event='adapter_committed',time=2,version='model_000001',sha256='abc'))
    if post_reload:events.append(dict(event='hybrid_generated',time=3,route='local',output_tokens=20))
    blobs={
      client+'metrics.jsonl':Blob('',[{'step':1,'progress/batch':0,'gemma/puct/sampled_size':16,
          'gemma/env/all/total_episodes':32,'gemma/time/train':3,'gemma/train_error':train_error,'gemma/train_skipped':skip}]),
      client+'member_gemma/checkpoints.jsonl':Blob('',[dict(batch=1,sampler_path='tinker://model/000001')]),
      prefix+'logs/inference-events.jsonl':Blob(prefix+'logs/inference-events.jsonl',events)}
    if checkpoint:blobs[prefix+'checkpoints/model/000001.tar.gz']=Blob('',[])
    bucket=SimpleNamespace(get_blob=lambda name:blobs.get(name))
    r.storage=SimpleNamespace(bucket=lambda name:bucket,list_blobs=lambda name,prefix:[v for k,v in blobs.items() if k.startswith(prefix)])
    return r


def test_complete_cycle_requires_optimizer_checkpoint_reload_and_remote_traffic(monkeypatch,tmp_path):
    assert evidence(monkeypatch,tmp_path).gemma_cycle()
    assert (tmp_path/'gemma-first-cycle.json').exists()


@pytest.mark.parametrize('missing',['remote','reload','post_reload','checkpoint'])
def test_incomplete_cycle_holds_admission(monkeypatch,tmp_path,missing):
    assert not evidence(monkeypatch,tmp_path,**{missing:False}).gemma_cycle()
    assert not (tmp_path/'gemma-first-cycle.json').exists()


@pytest.mark.parametrize('flag',['train_error','skip'])
def test_checkpoint_is_not_optimizer_proof(monkeypatch,tmp_path,flag):
    with pytest.raises(RuntimeError,match='failed or skipped'):
        evidence(monkeypatch,tmp_path,**{flag:1}).gemma_cycle()


@pytest.mark.parametrize('states,available',[
    (['RUNNING','RUNNING','RECOVERING'],False),
    (['RUNNING','STARTING','PENDING'],False),
    (['RUNNING','RUNNING','SUCCEEDED','FAILED_SETUP'],True),
])
def test_spare_slice_includes_pending_and_recovering_work(states,available):
    r=rollout();r.query=lambda *args:[dict(status=s) for s in states]
    assert r.spare_slice()==available


@pytest.mark.parametrize('spare',[True,False])
def test_muse_regrade_can_use_a_free_third_slice_while_qwen_is_running(spare):
    r=rollout();r.state['phase']='qwen_regrade';r.state['jobs']['qwen-regrade']=1501
    r.done=lambda jid:False;r.spare_slice=lambda:spare
    called=[];r.regrade=lambda model:called.append(model)
    r.tick()
    assert called==(['muse'] if spare else [])


@pytest.mark.parametrize('spare,cycle,expected',[(True,True,['qwen']),(True,False,[]),(False,True,[])])
def test_qwen_admission_during_muse_regrade_still_requires_cycle_and_capacity(spare,cycle,expected):
    r=rollout();r.state['phase']='muse_regrade';r.state['jobs']['muse-regrade']=1502
    r.done=lambda jid:False;r.spare_slice=lambda:spare;r.gemma_cycle=lambda:cycle
    called=[];r.launch_training=lambda model:called.append(model)
    r.tick();assert called==expected
