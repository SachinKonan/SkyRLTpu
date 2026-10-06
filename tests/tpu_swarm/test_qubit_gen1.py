from copy import deepcopy
import json
import pytest
from tpu.science.qubit_gen1 import checkpoint_at_step, transfer_pool
from tpu.science.routing_suite import manifest


def fixture():
    cases = [[s['id'], s['original_cnot_added']//3] for s in manifest()['cases']]
    state = dict(id='s', code='program', parents=['root'], timestep=19, value=.5,
                 observation=json.dumps(dict(reward=.5, metrics=dict(case_columns=['case','swaps'],cases=cases))))
    return dict(step=20, states=[state],initial_states=[],puct_n={'s':3},puct_m={'s':2},puct_T=5)


@pytest.mark.parametrize('step',[19,21])
def test_no_substitute_boundary(step):
    p=fixture();p['step']=step
    with pytest.raises(ValueError):transfer_pool(p)


def test_preserve_programs_and_feedback_reset_only_search_clock():
    p=fixture();original=deepcopy(p);q=transfer_pool(p)
    assert p==original
    assert q['step']==0 and q['puct_n']=={} and q['puct_m']=={} and q['puct_T']==0
    assert q['states'][0]==dict(p['states'][0],timestep=0)


def test_new_branch_cannot_opt_into_donor_search_statistics():
    with pytest.raises(TypeError):
        transfer_pool(fixture(), preserve_search=True)


@pytest.mark.parametrize('kind',['missing','duplicate','reward'])
def test_reject_corrupt_feedback(kind):
    p=fixture();s=p['states'][0];obs=json.loads(s['observation'])
    if kind=='missing':obs['metrics']['cases'].pop()
    if kind=='duplicate':obs['metrics']['cases'][-1]=obs['metrics']['cases'][0]
    if kind=='reward':s['value']=.9
    s['observation']=json.dumps(obs)
    with pytest.raises(ValueError):transfer_pool(p)


def test_gate_requires_successful_exact_checkpoint_and_metrics():
    c=dict(batch=20,state_path='tinker://model_g/weights/000020')
    assert checkpoint_at_step([dict(c,batch=19)],[dict(step=19)]) is None
    assert checkpoint_at_step([c],[dict(step=19)]) is None
    assert checkpoint_at_step([dict(c,batch=21)],[dict(step=20)]) is None
    assert checkpoint_at_step([c],[dict(step=20)])==c
    with pytest.raises(ValueError):checkpoint_at_step([c],[dict(step=20,**{'gemma/train_error':True})])
