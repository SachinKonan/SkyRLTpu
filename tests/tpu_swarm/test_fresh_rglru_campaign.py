from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from tpu.swarm.ray_train.bootstrap import workload_resources
from tpu.swarm.ray_train.commands import client_environment
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.seed_bootstrap import environment_type, make_environment
from tpu.swarm.select_v4_64_topology import select_split

PROFILES = sorted(Path('tpu/swarm/ray_train/profiles').glob('fresh-*-rglru-*-20260919.json'))


@pytest.mark.parametrize('profile', PROFILES, ids=lambda p:p.stem)
def test_owned_grader_fresh_bootstrap_and_transport(profile):
    c = Config.load(profile)
    v5p = c.accelerator == 'tpu-v5p-32'
    assert c.arena_grader_rank == 1
    assert (c.hosts,c.trainer.hosts,c.inference_hosts) == ((4,1,2) if v5p else (8,4,3))
    assert c.bootstrap_module == 'tpu.swarm.ray_train.seed_bootstrap'
    assert (c.bootstrap_max_drafts,c.bootstrap_target_valid,c.bootstrap_max_groups,c.bootstrap_group_size) == (1024,512,32,16)
    assert c.bootstrap_all_hosts and not c.seed_pool_sha256 and c.resume_min_checkpoint_step == 0
    assert c.inference.tp == 4 and c.inference.max_sequences == 16
    assert c.systemd_runtime and c.checkpoint_resume
    for rank in range(c.hosts):
        resources = workload_resources(c,rank)
        assert resources['TPU'] == 4
        assert resources.get('arena_grader',0) == (4 if rank == 1 else 0)
    env = client_environment(c,Path('/runtime'),'head')
    assert env['ARENA_RAY_TASKS'] == '1' and not env['ARENA_QUEUE_URL']
    assert env['ARENA_RAY_ROOT'] == '/runtime'
    assert env['TTD_ADV_ESTIMATOR'] == 'mean_baseline' and env['TTD_LOSS_FN'] == 'importance_sampling'
    assert float(c.client_learning_rate) == (1.5e-4 if env['TTD_ANSWER_MODEL_FAMILY'] == 'qwen' else 4e-5)
    with patch.dict('os.environ',env):
        cls = environment_type(c)
        e = make_environment(c, cls.create_initial_state('rg_lru'), None, Path('/tmp/rg-prompt-check'))
        assert 'pallas_call' in e.get_question()
    with pytest.raises(ValueError):
        replace(c, arena_grader_rank=None).validate()
    with pytest.raises(ValueError):
        replace(c, client_env={**c.client_env,'ARENA_QUEUE_URL':'http://old-judge:8791'}).validate()


@pytest.mark.parametrize('shift',range(8))
def test_v4_topology_preserves_physical_trainer_row_and_excludes_grader(shift):
    records=[]
    for physical in range(8):
        y=(physical//4)*2
        records.append(dict(process_id=(physical+shift)%8,
                            coords=[(x,j,physical%4) for x in range(2) for j in (y,y+1)]))
    old_train,old_infer=select_split(records)
    assert old_train[0] == 0 and len(old_infer) == 4
    train,other=select_split(records,excluded_rank=1)
    inference=[r for r in other if r != 1]
    assert len(train) == 4 and len(inference) == 3
    assert 1 not in train and 1 not in inference
    assert set(train+inference+[1]) == set(range(8))
    by_rank={r['process_id']:r['coords'] for r in records}
    # All trainer hosts share the same x/y row; their z ordering remains valid.
    assert len({tuple(sorted({c[1] for c in by_rank[r]})) for r in train}) == 1
    zs=[by_rank[r][0][2] for r in train]
    assert all((b-a)%4 == 1 for a,b in zip(zs,zs[1:]))


def test_nine_rglru_profiles_and_reserved_rank_validation():
    assert len(PROFILES) == 9
    for bad in (-1,8,True):
        c=Config.load(PROFILES[0])
        with pytest.raises(ValueError):
            replace(c,arena_grader_rank=bad).validate()
