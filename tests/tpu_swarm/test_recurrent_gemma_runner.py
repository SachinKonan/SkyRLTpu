import dataclasses
from pathlib import Path
import tarfile

import pytest

from tpu.swarm.ray_train.build import build
from tpu.swarm.ray_train.commands import client_environment
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.overlay import ARENA_FILES, FILES, install, manifest

PROFILE = "tpu/swarm/ray_train/profiles/qwen35_recurrentgemma_v5p_32.json"


def test_single_adapter_arena_is_in_frozen_worker_source(tmp_path):
    config = Config.load(PROFILE)
    assert config.adapter_count == 1 and config.requires_source_overlay
    env = client_environment(config, tmp_path, "10.0.0.1")
    assert env["TTD_ENV"] == "recurrent_gemma" and env["EVAL_TIMEOUT"] == "3600"
    assert env["TTD_EVAL_BACKEND"] == "local"  # custom evaluator sends HTTP to TPU judges
    archive, _, _ = build(PROFILE, tmp_path / "build")
    with tarfile.open(archive) as bundle:
        bundle.extractall(tmp_path / "unpack", filter="data")
    overlay = tmp_path / "unpack/tpu/swarm/ray_train/source_overlay"
    install(overlay, tmp_path / "source")
    for name in ARENA_FILES:
        assert (tmp_path / "source" / name).read_bytes() == Path(name).read_bytes()


def test_multi_adapter_arena_combines_both_overlays():
    config = dataclasses.replace(Config.load(PROFILE), adapter_count=2)
    from tpu.swarm.ray_train.overlay import DATABASE_FILES
    assert set(manifest(Path.cwd(), config)) == set(ARENA_FILES) | set(FILES) | DATABASE_FILES


@pytest.mark.parametrize("changes", [
    {"ARENA_QUEUE_URL": ""}, {"ARENA_QUEUE_URL": "ftp://host"},
    {"ARENA_QUEUE_URL": "http://user:secret@host"},
    {"ARENA_WAIT_TIMEOUT": "3601"}, {"ARENA_WAIT_TIMEOUT": "nan"},
    {"TTD_PROBLEM_TYPE": "splash_attention"},
])
def test_invalid_arena_config_fails_before_build(changes):
    config = Config.load(PROFILE)
    config.client_env.update(changes)
    with pytest.raises(ValueError):
        config.validate()
