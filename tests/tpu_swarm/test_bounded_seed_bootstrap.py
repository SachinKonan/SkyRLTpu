import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tpu.swarm.ray_train import seed_bootstrap as b
from tpu.swarm.ray_train.config import Config

PROFILES = sorted(Path('tpu/swarm/ray_train/profiles').glob('single-v4-*-20260919.json'))

def cfg(**changes):
    return SimpleNamespace(**(dict(bootstrap_max_drafts=1024, bootstrap_target_valid=512,
        bootstrap_group_size=16, bootstrap_max_groups=32) | changes))

def row(g, i, valid=True, code=None):
    identifier = f'bootstrap-{g:03d}-{i:03d}'
    code = code or identifier
    return dict(id=identifier, correctness=int(valid), reward=.5, code=code,
        state=dict(id=identifier, timestep=0, code=code, value=g*16+i, construction=[1., 2.],
                   observation='graded output', parents=[{'id':'shared-root'}], parent_values=[0.]) if valid else None)

def run_collect(path, config, valid=lambda g,i:True, duplicate=False):
    generations=[];grades=[]
    async def generate(g):
        generations.append(g)
        return list(range(config.bootstrap_group_size))
    async def grade(g,i,c):
        grades.append((g,i))
        return row(g,i,valid(g,i),code='same' if duplicate else None)
    rows,drafted=asyncio.run(b.collect(config,path,generate,grade))
    return rows,drafted,generations,grades

def test_all_valid_stops_at_512(tmp_path):
    rows,drafted,generated,graded=run_collect(tmp_path,cfg())
    assert drafted==len(rows)==len(graded)==512
    assert len(generated)==32
    assert len(b.selected_states(rows,512))==512

def test_low_validity_reaches_cap_and_keeps_best(tmp_path):
    rows,drafted,generated,graded=run_collect(tmp_path,cfg(),lambda g,i:i==0)
    assert drafted==1024 and len(generated)==64
    selected=b.selected_states(rows,512)
    assert len(selected)==64 and selected[0]['value']==1008
    assert all(s['parents']==[] and s['parent_values']==[] for s in selected)
    assert selected[0]['construction']==[1.,2.] and selected[0]['observation']=='graded output'

def test_duplicates_do_not_satisfy_target(tmp_path):
    rows,drafted,_,_=run_collect(tmp_path,cfg(),duplicate=True)
    assert drafted==1024 and len(b.selected_states(rows,512))==1

def test_resume_only_missing_grades(tmp_path):
    c=cfg(bootstrap_max_drafts=32,bootstrap_target_valid=32,bootstrap_max_groups=2)
    b.save(tmp_path/'group-000/generation.json',dict(choices=list(range(16))))
    for i in range(5):b.save(tmp_path/f'group-000/grade-{i:03d}.json',row(0,i))
    rows,drafted,generated,graded=run_collect(tmp_path,c)
    assert drafted==len(rows)==32 and generated==[1]
    assert len(graded)==27 and all((0,i) not in graded for i in range(5))
    assert run_collect(tmp_path,c)[2:]==([],[])

def test_empty_and_nonfinite_are_not_seeds(tmp_path):
    rows,drafted,_,_=run_collect(tmp_path,cfg(bootstrap_max_drafts=32),lambda g,i:False)
    assert drafted==32 and b.selected_states(rows,512)==[]
    r=row(0,0);r['state']['value']=float('nan')
    assert b.selected_states([r],512)==[]

def test_seed_roots_survive_normal_top_two_flush(tmp_path):
    from ttt_discover.tinker_utils.sampler import PUCTSampler
    from examples.circle_packing.env import CirclePackingEnv
    states=b.selected_states([row(g,i) for g in range(32) for i in range(16)],512)
    b.save(tmp_path/'puct_sampler_step_000000.json',dict(step=0,states=states,initial_states=[]))
    sampler=PUCTSampler(str(tmp_path/'puct_sampler.json'),CirclePackingEnv,'26',resume_step=0)
    sampler.flush(step=1)
    assert len(sampler._states)==512 and sampler.topk_children==2

@pytest.mark.parametrize('profile',PROFILES,ids=lambda p:p.stem)
def test_campaign_contract(profile):
    c=Config.load(profile);c.validate()
    assert c.accelerator=='tpu-v4-64' and c.hosts==8 and c.trainer.hosts==4
    assert c.bootstrap_module=='tpu.swarm.ray_train.seed_bootstrap'
    assert (c.bootstrap_max_drafts,c.bootstrap_target_valid,c.bootstrap_max_groups,c.bootstrap_group_size)==(1024,512,32,16)
    assert c.bootstrap_layers==1 and c.bootstrap_all_hosts and not c.seed_pool_sha256
    assert c.resume_min_checkpoint_step==0 and c.checkpoint_resume
    assert c.inference.tp==4 and c.inference.hosts_per_engine==1
    assert c.client_env['GROUP_SIZE']=='32' and c.client_env['GROUPS_PER_BATCH']=='16'
    assert c.client_env['TTD_ADV_ESTIMATOR']=='mean_baseline'
    assert c.client_env['TTD_LOSS_FN']=='importance_sampling'
    assert float(c.client_learning_rate)==(1.5e-4 if c.client_env['TTD_ANSWER_MODEL_FAMILY']=='qwen' else 4e-5)
    for k in ('trainer_compile','inference_compile'):
        assert c.run_id in getattr(c.cache,k)
        assert getattr(c.cache,k)!=getattr(c.cache,k+'_seed')
    if c.science_task=='placement':
        assert c.client_env['SCIENCE_PLACEMENT_SUITE']=='ibm17-proxy-v1'
        assert c.science_placement_slots_per_host==16 and c.science_placement_backend=='cpu'
    elif c.science_task=='routing':assert c.client_env['SCIENCE_ROUTING_SUITE']=='q20'
    elif c.client_env['TTD_ENV']=='circle_packing':assert c.client_env['TTD_PROBLEM_TYPE']=='26'

def test_exactly_twelve_and_no_legacy_optin():
    assert len(PROFILES)==12
    c=Config.load('tpu/swarm/ray_train/profiles/science-q20-v4-qwen-grpo-clean-20260918.json')
    assert c.bootstrap_max_drafts==0 and c.bootstrap_module=='tpu.science.bootstrap'

def test_corrupt_resume_is_rejected(tmp_path):
    b.save(tmp_path/'group-000/generation.json',dict(choices=list(range(16))))
    b.save(tmp_path/'group-000/grade-000.json',row(1,0))
    with pytest.raises(ValueError,match='identity mismatch'):run_collect(tmp_path,cfg())

def test_interrupted_atomic_generation_retries(tmp_path):
    (tmp_path/'group-000').mkdir()
    (tmp_path/'group-000/generation.json.tmp').write_text('{')
    rows,drafted,generated,_=run_collect(tmp_path,cfg(bootstrap_max_drafts=16,bootstrap_target_valid=16))
    assert generated==[0] and drafted==len(rows)==16

@pytest.mark.parametrize('task',('ac2','cp26','circuit','q20'))
def test_actual_environment_prompt_and_state(task,tmp_path):
    from tpu.swarm.ray_train.commands import client_environment
    from ttt_discover.tinker_utils.dataset_builder import VerifyResult
    c=Config.load(next(p for p in PROFILES if f'qwen-{task}-' in p.name))
    with patch.dict(os.environ,client_environment(c,tmp_path,'127.0.0.1')):
        cls=b.environment_type(c);root=cls.create_initial_state(c.client_env['TTD_PROBLEM_TYPE'])
        env=b.make_environment(c,root,None,tmp_path/'eval')
        question=env.get_question()
        assert question.strip()
        state=env._create_next_state(0,'candidate',VerifyResult(.6,'valid',1.,3.2,[1.,2.],'feedback',{}))
        assert state.value==(3.2 if env.is_maximize() else -3.2)
        assert state.construction is not None and state.observation
        if task=='circuit':assert 'ALL 17 IBM' in question and 'ibm17' in question
        if task=='q20':assert 'Q20 ONLY' in question

@pytest.mark.parametrize('valid',(True,False))
def test_full_bootstrap_publication_resume_and_no_training(tmp_path,valid):
    from dataclasses import replace
    import httpx,ray,tinker,transformers
    from ttt_discover.tinker_utils import renderers
    from ttt_discover.tinker_utils.dataset_builder import VerifyResult
    from ttt_discover.tinker_utils.sampler import get_or_create_sampler_with_default
    from tpu.swarm.ray_train.commands import client_environment
    c=replace(Config.load(next(p for p in PROFILES if 'qwen-cp26-' in p.name)),
        bootstrap_max_drafts=32,bootstrap_target_valid=16,bootstrap_max_groups=1)
    calls=[]
    class Renderer:
        def build_generation_prompt(self,messages):return SimpleNamespace(to_ints=lambda:[1,2])
        def get_stop_sequences(self):return [3]
        def parse_response(self,ids):return {'content':''.join(map(chr,ids))},True
    class HTTP:
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def post(self,url,json):
            calls.append(json);choices=[]
            for i in range(json['n']):
                ids=list(map(ord,f'</think>\n```python\nx = {len(calls)*16+i}\n```'))
                choices.append(dict(token_ids=ids,loss_mask=[1]*len(ids),finish_reason='stop',
                    thinking_budget=dict(enforced=True,budget_basis='phase1_generated_tokens',counted_phase1_tokens=0)))
            return SimpleNamespace(raise_for_status=lambda:None,json=lambda:dict(choices=choices))
    async def grade(self,code,step):
        return VerifyResult(.6 if valid else 0,'OK',int(valid),3.2 if valid else 0,[1.,2.],'feedback',{})
    cls=b.environment_type(c)
    runtime_env=client_environment(c,tmp_path,'head');runtime_env['TTD_RUN_DIR']=str(tmp_path)
    with patch.dict(os.environ,runtime_env),patch.object(ray,'is_initialized',return_value=True), \
         patch.object(cls,'_safe_grade',grade),patch.object(renderers,'get_renderer',return_value=Renderer()), \
         patch.object(transformers.AutoTokenizer,'from_pretrained',return_value=SimpleNamespace(get_vocab=lambda:{'x':1})), \
         patch.object(httpx,'AsyncClient',return_value=HTTP()), \
         patch.object(tinker,'ServiceClient',side_effect=AssertionError('bootstrap must not train')):
        if not valid:
            with pytest.raises(RuntimeError,match='no valid seeds'):asyncio.run(b.run(c,'snapshot','head'))
            assert not (tmp_path/'bootstrap/complete.json').exists()
            return
        result=asyncio.run(b.run(c,'snapshot','head'))
        assert result['retained']==16 and result['optimizer_steps']==0 and len(calls)==1
        log=tmp_path/'tinker_log'/c.run_id
        sampler=get_or_create_sampler_with_default(str(log),cls,'26',16)
        assert len(sampler._states)==16 and sampler._current_step==0
        assert all(s.code.startswith('```python') and s.value==3.2 for s in sampler._states)
        assert asyncio.run(b.run(c,'snapshot','head'))==result
        (tmp_path/'bootstrap/complete.json').unlink()
        assert asyncio.run(b.run(c,'snapshot','head'))==result and len(calls)==1
