from copy import deepcopy
import json
from pathlib import Path
import pytest
from tpu.science.ac2_handoff import best_completed_candidate, recipient_profile
from tpu.swarm.ray_train.config import Config


def snapshots():
    return [dict(model=m, run_id=m+'-old', status='SUCCEEDED', checkpoint=10,
                 uri='gs://test/'+m, generation=123,
                 pool=dict(step=10, puct_T=200, states=[dict(id=m+'-best', code='raise RuntimeError("must not execute")',
                     construction=[1.0], value=2/3, timestep=9, parents=['parent'], parent_values=[0.5], origin=m)]))
            for m in ('qwen', 'gemma', 'muse')]


def test_verified_winner_is_fresh_root_without_running_generated_code():
    source=snapshots();before=deepcopy(source);pool,p=best_completed_candidate(source)
    assert source==before
    assert p['score']==pytest.approx(2/3)
    assert pool['step']==0 and pool['puct_T']==0 and not pool['puct_n'] and not pool['puct_m']
    root=pool['states'][0]
    assert root['timestep']==0 and root['parents']==[] and root['origin'] is None
    assert root['code'].startswith('raise RuntimeError')
    assert pool['initial_states']==pool['states']


@pytest.mark.parametrize('change',[{'status':'RUNNING'},{'checkpoint':9},{'status':'FAILED'}])
def test_unfinished_round_cannot_promote(change):
    rows=snapshots();rows[0].update(change)
    with pytest.raises(ValueError,match='not complete'):best_completed_candidate(rows)


def test_tampered_winning_score_cannot_promote():
    rows=snapshots();rows[0]['pool']['states'][0]['value']=0.99
    with pytest.raises(ValueError,match='does not reproduce'):best_completed_candidate(rows)


def test_fresh_recipient_preserves_recipe_and_does_not_modify_source():
    root=Path(__file__).resolve().parents[2]
    original=json.loads((root/'tpu/swarm/ray_train/profiles/fresh-v4-qwen-ac2-grpo-lr15e4-s1-20260919-10step-deployment.json').read_text())
    before=deepcopy(original);new=recipient_profile(original,'ac2-followup-test','us-central1-b','gs://test')
    Config.from_dict(new)
    assert original==before and new['trainer']==original['trainer']
    assert new['client_learning_rate']==original['client_learning_rate']
    assert new['resume_min_checkpoint_step']==0 and new['client_env']['NUM_EPOCHS']=='10'
    assert new['cache']['trainer_compile_seed']==original['cache']['trainer_compile']
    assert new['cache']['trainer_compile']!=original['cache']['trainer_compile']
