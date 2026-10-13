from copy import deepcopy

from tpu.swarm.ray_train.direct_discovery import candidates, stable_completion, tick


REG = dict(run_id='run', model='model', group='dedicated', stable_urls=['stable'], allow_spot=True)
TARGET = dict(run_id='run', model='model', instance='new-instance', enabled=True,
              borrowing={'owner_run': 'run:borrower-instance'},
              lease_scope='run', accepted_compatibility=['hash'], candidate_limit=16, urls=['stale'])


def farm(url='new', **kw):
    return dict(url=url, models=['model'], capabilities={'compatibility_sha256': 'hash'}, **kw)


def test_replacement_keeps_stable_and_removes_stale_address():
    writes = []
    def call(reg, body=None):
        if body:
            writes.append(body)
            return dict(ok=True, result=TARGET | body)
        return dict(ok=True, result=deepcopy(TARGET))
    assert tick([REG], [farm()], call)[0]['state'] == 'updated'
    assert writes == [dict(run_id='run', instance='new-instance', model='model',
                           group='dedicated', urls=['stable', 'new'])]


def test_farm_contract_and_exclusive_ownership():
    wrong = farm('wrong'); wrong['capabilities']['compatibility_sha256'] = 'wrong'
    other_model = farm('other-model'); other_model['models'] = ['other']
    assert candidates(TARGET, REG, [wrong, other_model, farm('busy', owner_run='another:x'),
                                   farm('own', owner_run='run:borrower-instance'), farm()]) == ['stable', 'new', 'own']


def test_fresh_trainer_uses_dedicated_contract_first():
    assert candidates(TARGET | {'accepted_compatibility': []}, REG, [farm()]) == ['stable']
    assert candidates(TARGET, REG | {'allow_spot': False}, [farm()]) == ['stable']


def test_wrong_run_and_disabled_group_never_receive_writes():
    for changes, state in [({'run_id': 'unrelated'}, 'identity_mismatch'),
                           ({'enabled': False}, 'not_opted_in')]:
        def call(reg, body=None):
            assert body is None
            return dict(ok=True, result=TARGET | changes)
        assert tick([REG], [farm()], call)[0]['state'] == state


def test_restart_between_read_and_push_is_not_success():
    def call(reg, body=None):
        return dict(ok=False, error='stale instance') if body else dict(ok=True, result=TARGET)
    assert tick([REG], [farm()], call)[0]['state'] == 'update_failed'


def test_unreachable_target_does_not_block_other_targets():
    def call(reg, body=None):
        if reg['run_id'] == 'down':
            raise TimeoutError('SSH timeout')
        return dict(ok=True, result=TARGET | {'urls': ['stable', 'new']})
    result = tick([REG | {'run_id': 'down'}, REG], [farm()], call)
    assert [r['state'] for r in result] == ['error', 'unchanged']


def test_empty_successful_inventory_preserves_stable_endpoint():
    assert candidates(TARGET, REG, []) == ['stable']


def test_spot_gate_requires_completed_adapter_request_on_same_stable_phase():
    ready = dict(event='borrow_ready', service='stable', phase='p', sha256='adapter')
    done = dict(event='borrow_generated', service='stable', phase='p')
    assert stable_completion([ready, done], ['stable'])
    for rows in [[done, ready], [ready], [ready, done | {'phase': 'different'}],
                 [ready | {'sha256': None}, done], [ready, done | {'service': 'other'}]]:
        assert stable_completion(rows, ['stable']) is None
