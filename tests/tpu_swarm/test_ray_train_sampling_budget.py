"""Keep prompt/thinking, answer, and training budgets consistent before launch."""

from dataclasses import replace
from pathlib import Path

import pytest

from tpu.swarm.ray_train.commands import client_environment
from tpu.swarm.ray_train.config import Config


@pytest.mark.parametrize("profile", ["qwen_v5p_32", "qwen_v4_64_retry",
                                    "qwen_v5p_32_budget", "qwen_v4_64_budget"])
def test_proven_two_phase_budget_leaves_answer_space(profile, monkeypatch):
    monkeypatch.setenv("TTD_M0_PHASE1_MAX_TOKENS", "20480")
    cfg = Config.load(f"tpu/swarm/ray_train/profiles/{profile}.json")
    env = client_environment(cfg, Path("/tmp/test"), "10.0.0.1")
    assert env["TTD_M0_PHASE1_MAX_TOKENS"] == "13824"
    assert env["CONTEXT_WINDOW"] == env["TTD_M0_CONTEXT_WINDOW"] == "18432"
    assert env["TTD_M0_TRAIN_MAX_SEQ"] == "18432"
    # A capped phase one includes the prompt, leaving the same answer reserve
    # regardless of prompt length; account for the completer's cue and buffer.
    for prompt in (1000, 4053, 8000):
        generated = int(env["TTD_M0_PHASE1_MAX_TOKENS"]) - prompt
        answer = int(env["TTD_M0_CONTEXT_WINDOW"]) - prompt - generated - 20 - 50
        assert answer == 4538


@pytest.mark.parametrize("overrides", [
    {"TTD_M0_PHASE1_MAX_TOKENS": "20480"},
    {"TTD_M0_PHASE1_MAX_TOKENS": "18432"},
    {"TTD_M0_PHASE1_MAX_TOKENS": "18400"},
    {"TTD_M0_PHASE1_MAX_TOKENS": "0"},
    {"TTD_M0_PHASE1_MAX_TOKENS": "bad"},
    {"TTD_M0_CONTEXT_WINDOW": "22528"},
    {"TTD_M0_TRAIN_MAX_SEQ": "22528"},
    {"CONTEXT_WINDOW": "32768"},
])
def test_invalid_effective_budget_is_rejected_before_client_start(overrides):
    cfg = Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32_budget.json")
    raw = cfg.to_dict()
    raw["client_env"] = overrides
    with pytest.raises(ValueError):
        Config.from_dict(raw)
    with pytest.raises(ValueError):
        client_environment(replace(cfg, client_env=overrides), Path("/tmp/test"), "10.0.0.1")


def test_explicit_valid_budget_overrides_are_preserved():
    cfg = Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32_budget.json")
    cfg = replace(cfg, client_env={"TTD_M0_PHASE1_MAX_TOKENS": "12288"})
    assert client_environment(cfg, Path("/tmp/test"), "10.0.0.1")["TTD_M0_PHASE1_MAX_TOKENS"] == "12288"


def test_corrected_runs_keep_hardware_layout_and_isolate_mutable_state():
    for profile, previous, trainer_shape in (
        ("qwen_v5p_32_budget", "qwen_v5p_32", (1, 1, 4, 18432)),
        ("qwen_v4_64_budget", "qwen_v4_64_retry", (4, 8, 2, 22528)),
    ):
        cfg = Config.load(f"tpu/swarm/ray_train/profiles/{profile}.json")
        old = Config.load(f"tpu/swarm/ray_train/profiles/{previous}.json")
        assert (cfg.trainer.hosts, cfg.trainer.tp, cfg.trainer.fsdp,
                cfg.trainer.sequence_length) == trainer_shape
        assert cfg.run_gcs != old.run_gcs
        assert cfg.cache == old.cache
        assert cfg.inference == old.inference
