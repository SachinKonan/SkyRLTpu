"""Erdős multi-seed merge round 2: four ordinary single-model runs on one v5p-16 host each, resuming at batch 12 on
the union of the model's two seeds' step-12 trees (tpu/merge_seeds_setup.py, tpu/merge_seeds_profiles.py)."""
import json
import subprocess
from pathlib import Path

import pytest

from tpu import cosearch_merge_trees as mt
from tpu import merge_seeds_profiles as mp
from tpu import merge_seeds_setup as ms

ROOT = Path(__file__).resolve().parents[2]


def _have(ref):
    return subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', ref], capture_output=True).returncode == 0


def test_arms_pair_with_their_model_merge():
    assert set(ms.ARMS) == set(mp.ARMS)
    for arm, (key, run) in ms.ARMS.items():
        src = ms.SOURCES[key]
        assert key in ms.MERGES[src.tag] and mp.ARMS[arm][0] == run and mp.ARMS[arm][1] == src.tag
    # both arms of a model use the same merged tree; each seed's weights are used exactly once
    assert sorted(k for k, _ in ms.ARMS.values()) == sorted(k for pair in ms.MERGES.values() for k in pair)
    assert ms.SOURCES['qwen-s1'].batch == 5 and all(ms.SOURCES[k].batch == 12 for k in ('qwen-s3', 'muse-s1', 'muse-s3'))


def _tree(prefix, n, best, T):
    states = [dict(id=f'{prefix}-{i}', value=best - i * 1e-6, parents=[dict(id=f'{prefix}-root')] if i else [])
              for i in range(n)]
    return dict(step=12, states=states, initial_states=[dict(id=f'{prefix}-root')],
                puct_n={s['id']: 1 for s in states}, puct_m={s['id']: s['value'] for s in states}, puct_T=T)


def test_union_keeps_per_state_stats_and_sums_T():
    a, b = _tree('a', 5, -0.38085, 100), _tree('b', 4, -0.38087, 80)
    merged, report = mt.merge_snapshots([('s1', a), ('s3', b)], ms.BATCH)
    assert [s['id'] for s in merged['states']] == [s['id'] for s in a['states'] + b['states']]
    assert merged['puct_T'] == 180 and merged['step'] == 12
    assert merged['puct_n'] == {**a['puct_n'], **b['puct_n']} and merged['puct_m'] == {**a['puct_m'], **b['puct_m']}
    assert report['merged_best_value'] == -0.38085 and report['best_source'] == 's1'
    assert report['shared_identical'] == {'states': 0, 'initial_states': 0} and not report['over_buffer']


@pytest.mark.skipif(not _have(mp.COSEARCH_REF), reason='agent/cosearch commit not available')
@pytest.mark.parametrize('arm', sorted(mp.ARMS))
def test_profile_is_base_run_on_one_v5p_host(arm):
    run_id, tag, base_name, host = mp.ARMS[arm]
    p = mp.trainer(arm, host)
    config = mp.validate(arm, p)
    base = mp.local_profile(base_name)
    assert config.run_id == run_id and config.assigned_host_ips == [host]
    assert p['trainer']['backward_warmup'] is base['trainer']['backward_warmup'] is True
    assert p['client_env']['TTD_M0_LORA_SEED'] == base['client_env']['TTD_M0_LORA_SEED']
    assert 'ports' not in p and 'trainer_only' not in p and 'cosearch_peer' not in p
    written = mp.PROFILES / f'{run_id}.json'
    if written.exists():
        assert json.loads(written.read_text()) == p
