"""Extra MaxText arguments cannot silently replace the validated trainer mesh."""
from pathlib import Path

import pytest

from tpu.swarm.ray_train.commands import maxtext_kwargs
from tpu.swarm.ray_train.config import Config


PROFILE = Path('tpu/swarm/ray_train/profiles/science-placement-v6e-qwen-cpu-seeded-train-001.json')


@pytest.mark.parametrize('key,value', [
    ('ici_tensor_parallelism', 4), ('ici_fsdp_parallelism', 4),
    ('ici_context_parallelism', 2), ('ici_tensor_parallelism', '8'),
    ('ici_context_parallelism', True),
])
def test_conflicting_mesh_override_is_rejected(key, value):
    raw = Config.load(PROFILE).to_dict()
    raw['trainer']['maxtext_kwargs'][key] = value
    with pytest.raises(ValueError, match='conflicts with the trainer mesh'):
        Config.from_dict(raw)


def test_matching_explicit_mesh_remains_supported():
    raw = Config.load(PROFILE).to_dict()
    mesh = dict(ici_tensor_parallelism=8, ici_fsdp_parallelism=2, ici_context_parallelism=1)
    raw['trainer']['maxtext_kwargs'].update(mesh)
    cfg = Config.from_dict(raw)
    emitted = maxtext_kwargs(cfg, Path('/cache'))
    assert {key: emitted[key] for key in mesh} == mesh


@pytest.mark.parametrize('model', ['qwen', 'muse', 'gemma'])
def test_submitted_placement_mesh_matches_profile(model):
    cfg = Config.load(PROFILE.with_name(f'science-placement-v6e-{model}-cpu-seeded-train-001.json'))
    emitted = maxtext_kwargs(cfg, Path('/cache'))
    assert (emitted['ici_tensor_parallelism'], emitted['ici_fsdp_parallelism']) == (
        cfg.trainer.tp, cfg.trainer.fsdp)
