"""discovery --fair-share: split ready farms between trainers that are generating, never revoke a held lease."""
from tpu.swarm.ray_train import direct_discovery
from tpu.swarm.ray_train.discovery import fair_share_cap, trim_to_share

GEN = dict(borrowing=dict(phase='p1', held=[]))
HOLDS = dict(borrowing=dict(phase=None, held=['http://a']))
IDLE = dict(borrowing=dict(phase=None, held=[]))


def test_no_cap_unless_two_trainers_generate():
    assert fair_share_cap(3, [GEN]) is None
    assert fair_share_cap(3, [GEN, IDLE]) is None
    assert fair_share_cap(3, [GEN, {}]) is None


def test_cap_is_ceil_of_ready_over_generating():
    assert fair_share_cap(3, [GEN, GEN]) == 2
    assert fair_share_cap(16, [GEN, HOLDS]) == 8
    assert fair_share_cap(0, [GEN, GEN]) == 1


def test_trim_keeps_held_leases_first_even_over_cap():
    target = dict(borrowing=dict(held=['http://c', 'http://d', 'http://e']))
    urls = ['http://a', 'http://b', 'http://c', 'http://d', 'http://e']
    assert trim_to_share(target, urls, 2) == ['http://c', 'http://d', 'http://e']
    assert trim_to_share(dict(borrowing=dict(held=['http://b'])), urls, 2) == ['http://b', 'http://a']
    assert trim_to_share({}, urls, None) == urls


def test_model_caps_count_only_trainers_holding_or_acquiring():
    from tpu.swarm.ray_train.discovery import model_caps_from
    holds = dict(phase='p', held=['http://a'], preparing=False)
    acquiring = dict(phase='p', held=[], preparing=True)
    grading = dict(phase='p', held=[], preparing=False)  # phase id kept for the run, farms handed back
    # Live case 2026-11-10: 8 Qwen farms, 2 generating + 2 grading -> 4 each (was 2), one farm kept back.
    assert model_caps_from({'q': 8}, [('q', holds), ('q', acquiring), ('q', grading), ('q', grading)]) == {'q': 4}
    # One generating, two idle: everything but the reserve (was 3 with --model-fair-share).
    assert model_caps_from({'m': 8}, [('m', holds), ('m', grading), ('m', grading)]) == {'m': 7}
    # One generating, none idle: no cap; nobody generating: no cap.
    assert model_caps_from({'m': 8}, [('m', holds)]) == {}
    assert model_caps_from({'m': 8}, [('m', grading), ('m', grading)]) == {}
    # Two generating, none idle: plain split. Too few farms for a reserve: no reserve, floor of 1.
    assert model_caps_from({'g': 5}, [('g', holds), ('g', holds)]) == {'g': 3}
    assert model_caps_from({'g': 2}, [('g', holds), ('g', acquiring), ('g', grading)]) == {'g': 1}
    assert model_caps_from({}, [('x', holds), ('x', holds)]) == {'x': 1}


def test_direct_tick_applies_trim():
    target = dict(run_id='r', model='m', instance='i', enabled=True, lease_scope='run', urls=[],
                  compatibility_sha256='c', borrowing=dict(held=[]))
    farms = [dict(url=f'http://f{i}', models=['m'], capabilities=dict(compatibility_sha256='c')) for i in range(4)]
    reg = dict(run_id='r', model='m', group='g', stable_urls=[], allow_spot=True)
    posted = []

    def call(registration, body=None):
        if body is None:
            return dict(ok=True, result=target)
        posted.append(body['urls'])
        return dict(ok=True, result=dict(target, urls=body['urls']))
    out = direct_discovery.tick([reg], farms, call, trim=lambda t, urls: trim_to_share(t, urls, 2))
    assert posted == [['http://f0', 'http://f1']] and out[0]['state'] == 'updated'
