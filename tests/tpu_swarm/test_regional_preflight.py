import pytest

from tpu.swarm.ray_train.regional_preflight import check_paths


def profile():
    return dict(bucket='gs://sk7524-tinker-tpu-us-east5', zone='us-east5-b',
                single_bucket=True, cache={'hf': 'gs://sk7524-tinker-tpu-us-east5/model'})


def test_actual_bucket_location_must_match_even_if_name_looks_local():
    with pytest.raises(ValueError, match='location'):
        check_paths(profile(), 'US-CENTRAL2')


@pytest.mark.parametrize('field', ['client_env', 'cache', 'task'])
def test_hidden_cross_region_storage_is_rejected(field):
    p = profile()
    foreign = {'restore': 'gcloud storage cp gs://other-region/data /tmp/data'}
    task = foreign if field == 'task' else None
    if field != 'task':
        p[field] = foreign
    with pytest.raises(ValueError, match='unapproved bucket'):
        check_paths(p, 'US-EAST5', task=task)


def test_cross_region_farm_http_is_allowed():
    p = profile()
    p['inference'] = {'external_pool_urls': {'Qwen/Qwen3.8-27B': ['http://10.146.0.124:24800']}}
    assert check_paths(p, 'US-EAST5') == 'us-east5'


def test_opt_out_is_rejected():
    p = profile()
    p['single_bucket'] = False
    with pytest.raises(ValueError, match='single_bucket'):
        check_paths(p, 'US-EAST5')
