"""Opt-in routing contracts: no cloud, TPU allocation, or candidate execution."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import pytest
from tpu.science import routing_resources as resources
from tpu.science.routing_parallel import aggregate, split_suite, remaining, evaluate
from tpu.science.cpu_slots import acquire_slot
from tpu.science.feedback import routing_cases, observation
from tpu.science.routing_suite import manifest
from tpu.swarm.ray_train.config import Config


def test_split_preserves_case_seeds_and_order(tmp_path):
    rows=[{'id':'z','seed':42,'qasm3_path':'z.qasm','topology_path':'q20.json'},
          {'id':'a','qasm3_path':'a.qasm','topology_path':'willow.json'}]
    original=tmp_path/'suite.json';original.write_text(json.dumps({'cases':rows,'extra':'preserved'}))
    split=split_suite(original,tmp_path/'split')
    assert [v[0] for v in split]==['z','a']
    for raw,(_,path) in zip(rows,split):
        d=json.loads(path.read_text());row=d['cases'][0]
        assert d['extra']=='preserved'
        assert row.get('seed')==raw.get('seed')
        assert row['qasm3_path']==str(tmp_path/raw['qasm3_path'])
        assert ('seed' in row)==('seed' in raw)


def case(name,swaps=2):
    return dict(correctness=1,metrics={'cases':[dict(case=name,swaps=swaps,added_cnots=3*swaps,baseline_added_cnots=12,weight=.4)]})


def test_aggregate_order_and_partial_failure():
    result=aggregate(['a','b'],{'b':case('b',1),'a':case('a',2)})
    assert [r['case'] for r in result['metrics']['cases']]==['a','b']
    assert result['reward']==pytest.approx(24/(24+9))
    for rows in [{'a':case('a')},{'a':case('wrong'),'b':case('b')},
                 {'a':dict(correctness=0),'b':case('b')}]:
        with pytest.raises(ValueError):aggregate(['a','b'],rows)


def test_budget_includes_preparation_and_resets_alarm():
    def slow(*args,**kwargs):time.sleep(.5)
    with patch('tpu.science.routing_parallel._evaluate',side_effect=slow):
        start=time.monotonic()
        with pytest.raises(TimeoutError):evaluate('code',seconds=.05)
        assert time.monotonic()-start<.4
    import signal
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0
    with pytest.raises(TimeoutError):remaining(time.monotonic()-.1)


def test_dynamic_numa_partition(tmp_path):
    for i,cpus in enumerate(['0-59,120-179','60-119,180-239']):
        p=tmp_path/f'node{i}';p.mkdir();(p/'cpulist').write_text(cpus)
    grading,service=resources.host_partition(range(240),tmp_path)
    assert len(grading)==100 and len(service)==140
    assert not set(grading)&set(service)
    assert len(set(grading)&set(range(120,180)))==50
    with pytest.raises(RuntimeError):resources.host_partition(range(100),tmp_path)


def test_new_admission_blocks_legacy_both_directions(tmp_path):
    mapping=(list(range(100)),list(range(100,124)))
    with patch.object(resources,'host_partition',return_value=mapping):
        oldslot,old=acquire_slot(tmp_path,slots=16)
        with old:
            with pytest.raises(TimeoutError):resources.acquire(root=tmp_path,deadline_seconds=.02)
        slot,cpus,lease=resources.acquire(root=tmp_path,deadline_seconds=.1)
        with lease:
            assert len(cpus)==10
            with pytest.raises(TimeoutError):acquire_slot(tmp_path,slots=16,deadline_seconds=.02)
            slot2,cpus2,lease2=resources.acquire(root=tmp_path,deadline_seconds=.1)
            with lease2:assert not set(cpus)&set(cpus2)
        _,old=acquire_slot(tmp_path,slots=16,deadline_seconds=.1);old.close()


def test_feedback_targets_only_complete_topologies():
    rows=[dict(case=s['id'],swaps=0,baseline_added_cnots=s['original_cnot_added']) for s in manifest()['cases']]
    full=routing_cases({'cases':rows})
    assert len(full['cases'])==72
    for name,target in resources.GEMINI_TARGETS.items():
        assert full['topologies'][name]['gap_to_gemini']==-target
        assert full['topologies'][name]['complete']
    partial=routing_cases({'cases':rows[:1]})
    assert partial['topologies']['q20']['gap_to_gemini'] is None
    msg=observation('routing',dict(metrics={'cases':rows},reward=.5,msg='Valid',correctness=1))
    assert len(msg)<=12000 and len(json.loads(msg)['metrics']['cases'])==72


def test_profile_resource_contract_is_opt_in():
    old=Config.load('tpu/swarm/ray_train/profiles/science-v6e-gemma-qubit-grpo-lr4e5-s1-20260920-10step-deployment.json')
    assert old.science_routing_evaluator=='serial-v1'
    new=replace(old,science_routing_evaluator='parallel-v2',science_routing_slots_per_host=10,cache=replace(old.cache,reserve_gib=264))
    new.validate();assert new.ray_cpus_per_host==108
    for config in [replace(new,science_routing_slots_per_host=11),replace(new,systemd_runtime=False),
                   replace(new,client_env={**new.client_env,'EVAL_TIMEOUT':'1900'})]:
        with pytest.raises(ValueError):config.validate()


def test_target_prompt_has_one_consistent_deadline_for_initial_and_parent():
    from tpu.science.training_env import task_prompt, candidate_prompt
    env={'SCIENCE_ROUTING_EVALUATOR':'parallel-v2'}
    for starter in (True,False):
        prompt=task_prompt('routing',include_starter=starter,environment=env)
        assert '1,900-second' in prompt and '13,470' in prompt and '42,396' in prompt
        assert '900 seconds' not in prompt and '1800 seconds' not in prompt
    with patch.dict(os.environ,env):
        prompt=candidate_prompt('routing',code='parent source',feedback='parent feedback')
        assert '31,481' in str(prompt) and 'parent source' in str(prompt)
    legacy=task_prompt('routing',environment={})
    assert '1800 seconds' in legacy and 'SimpleTES Gemini' not in legacy


def test_timeout_feedback_preserves_50_verified_cases_and_all_72_statuses(tmp_path):
    from tpu.science.routing_parallel import partial_metrics
    from tpu.science.rewards import invalid
    specs=manifest()['cases'];events=[]
    for spec in specs[:50]:
        name=spec['id'];events.append(dict(case=name,event='started'))
        result=case(name);result['metrics']['cases'][0]['baseline_added_cnots']=spec['original_cnot_added']
        events.append(dict(case=name,result=result))
    for spec in specs[50:54]:events.append(dict(case=spec['id'],event='started'))
    (tmp_path/'case-progress.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n{"interrupted":')
    result=invalid('shared routing evaluation deadline exhausted')
    result['metrics'].update(partial_metrics(tmp_path,deadline_exceeded=True))
    text=observation('routing',result);feedback=json.loads(text)
    assert result['reward']==0 and result['correctness']==0
    assert len(text)<=12000 and feedback['metrics']['case_count']==50
    assert len(feedback['metrics']['cases'])==72
    assert feedback['metrics']['status_counts']==dict(passed=50,deadline_unfinished=4,not_started=18)
    rows=feedback['metrics']['cases']
    assert all(row[2]==2 for row in rows[:50])
    assert all(row[2] is None and row[4] is None for row in rows[50:])
    for topology in feedback['metrics']['topologies'].values():
        if not topology['complete']:assert topology['gap_to_gemini'] is None


def test_regrade_import_rejects_partial_rewards_and_resets_history(tmp_path):
    import hashlib
    from tpu.science.routing_regrade import digest,import_pool
    from tpu.science.rewards import valid,qubit,invalid
    from tpu.science.seed_pool import verify_pool
    programs=[];verdicts={};occurrences=[]
    for index,swaps in enumerate([2,1,None]):
        code=f'RUST_CODE = "fixture {index}"';sha=hashlib.sha256(code.encode()).hexdigest()
        programs.append(dict(source_sha256=sha,code=code))
        if swaps is None:r=invalid('compilation error')
        else:
            rows=[dict(case=s['id'],swaps=swaps,added_cnots=swaps*3,
                       baseline_added_cnots=s['original_cnot_added'],weight=s['weight']) for s in manifest()['cases']]
            reward,m=qubit([r['baseline_added_cnots'] for r in rows],[r['added_cnots'] for r in rows],[r['weight'] for r in rows])
            r=valid(reward,dict(m,cases=rows))
        r['metrics'].update(source_sha256=sha,resource_contract=resources.contract(),regrade_evaluator_sha256='pinned')
        verdicts[sha]=r;occurrences.append(dict(source_sha256=sha,old_correctness=0,old_message='timeout'))
    source=dict(programs=programs,unique_programs=3,model='gemma',occurrences=occurrences,old_valid=0,prompt_provenance='Historical fixture')
    source['sha256']=digest(source)
    summary=import_pool(source,verdicts,tmp_path/'import',target_run='fresh',evaluator_sha256='pinned')
    pool=verify_pool(tmp_path/'import/puct_sampler_step_000000.json',summary['pool_sha256'])
    assert summary['retained']==2 and summary['newly_valid_unique']==2
    assert pool['puct_n']=={} and pool['puct_T']==0 and pool['step']==0
    assert pool['states'][0]['code']==programs[1]['code']
    verdicts[programs[0]['source_sha256']]['metrics']['cases'].pop()
    with pytest.raises(ValueError,match='72'):
        import_pool(source,verdicts,tmp_path/'bad',target_run='fresh',evaluator_sha256='pinned')


def test_zero_baseline_case_is_verified_without_individual_reward():
    from tpu.science.routing_parallel import verified_case
    zero=case('zero',0);zero['metrics']['cases'][0]['baseline_added_cnots']=0
    assert verified_case('zero',zero)['swaps']==0
    r=aggregate(['zero','nonzero'],dict(zero=zero,nonzero=case('nonzero',1)))
    assert r['reward']==pytest.approx(12/(12+3))


@pytest.mark.parametrize('model',['gemma','qwen','muse'])
def test_relaunch_profiles_preserve_training_recipe_and_enable_farm(tmp_path,model):
    from tpu.science.routing_relaunch_profile import build,SOURCES
    from tpu.science.bootstrap import identity
    pool=dict(step=0,states=[dict(code='fixture',value=.5)],initial_states=[],puct_n={},puct_m={},puct_T=0)
    (tmp_path/'puct_sampler_step_000000.json').write_text(json.dumps(pool))
    (tmp_path/'seed-import.json').write_text(json.dumps(dict(model=model,pool_sha256=identity(pool),
        target_run='qubit-v4-'+model+'-parallel2-20260921',source_manifest_sha256='source',evaluator_sha256='grader')))
    fresh=build(model,tmp_path,tmp_path/'profile.json')
    old=json.loads((Path('tpu/swarm/ray_train/profiles')/SOURCES[model]).read_text())
    assert fresh['trainer']==old['trainer'] and fresh['client_learning_rate']==old['client_learning_rate']
    assert fresh['accelerator']=='tpu-v4-64' and fresh['hosts']==8
    for key in ['GROUP_SIZE','GROUPS_PER_BATCH','NUM_EPOCHS','TTD_LOSS_FN','TTD_ADV_ESTIMATOR','TTD_M0_LORA_SEED']:
        assert fresh['client_env'][key]==old['client_env'][key]
    for key in old['inference']:
        if not key.startswith('external_pool_'):assert fresh['inference'][key]==old['inference'][key]
    assert fresh['inference']['external_pool_updates'] and fresh['inference']['external_pool_lease_scope']=='run'
