"""RL integration: real queue protocol, state feedback, and seed gradients."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pallas_arena.rl.task import (
    ArenaInfrastructureError, build_prompt, public_contract, translate_verdict,
)


def good_verdict():
    names = [name for name, _ in public_contract()[1]]
    return dict(ok=True, passed=True, gate="all", reward=2.0, reward_with_bwd=1.0,
                score=2.0, baseline_mode="all", n_bwd_factors=len(names),
                per_case={name: 2.0 for name in names if "holdout" not in name},
                holdout={name: 2.0 for name in names if "holdout" in name},
                grad_scores={name: 0.5 for name in names})


def test_uses_combined_reward_and_preserves_raw_speedup():
    translated = translate_verdict(good_verdict())
    assert translated["reward"] == 1.0  # forward alone would pay 2.0
    assert translated["raw_score"] == pytest.approx(1.0)
    assert translated["correctness"] == 1.0


@pytest.mark.parametrize("change", [
    {"ok": False}, {"gate": "judge_fault"}, {"judge_fault": True},
    {"grad_scores": {}}, {"n_bwd_factors": 0}, {"baseline_mode": "xla"},
    {"excluded_cases": {"probe-a": "device busy"}}, {"reward_with_bwd": float("nan")},
    {"per_case": {}},
])
def test_partial_or_faulted_verdict_is_not_a_training_reward(change):
    with pytest.raises(ArenaInfrastructureError):
        translate_verdict({**good_verdict(), **change})


def test_candidate_failure_is_zero():
    out = translate_verdict(dict(ok=True, passed=False, gate="gradient", reward=0,
                                 violations=["wrong d/da"]))
    assert out["reward"] == out["correctness"] == 0
    assert "gradient" in out["stdout"].lower()


def test_prompt_client_imports_need_neither_jax_nor_fastapi():
    code = """
import sys
from pallas_arena.rl.task import build_prompt
from pallas_arena.judge.client import ArenaQueueClient
assert 'jax' not in sys.modules and 'fastapi' not in sys.modules
assert 'def _apply_reset' in build_prompt()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_prompt_matches_judge_case_contract():
    from pallas_arena.judge.problems.rg_lru import PROBLEM
    expected = [(c.name, c.dims) for c in PROBLEM.shape_cases() if c.probe and not c.tp]
    assert public_contract()[1] == expected
    assert PROBLEM.has_bwd and PROBLEM.bwd_gates and PROBLEM.require_pallas
    assert "both x and a" in build_prompt()


def test_queue_roundtrip_and_raw_artifact(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from pallas_arena.judge.queue import create_queue_app
    from pallas_arena.rl.env import RecurrentGemmaEnv, RecurrentGemmaRewardEvaluator
    from pallas_arena.judge.client import ArenaQueueClient

    queue = TestClient(create_queue_app())
    submitted = []

    def post(self, path, payload):
        response = queue.post(path, json=payload)
        response.raise_for_status()
        if path == "/submit":
            submitted.append(payload)
            lease = queue.get("/work?worker_id=test").json()
            queue.post("/result", json=dict(work_id=lease["work_id"], lease_id=lease["lease_id"],
                                            result=good_verdict())).raise_for_status()
        return response.json()

    monkeypatch.setattr(ArenaQueueClient, "_post", post)
    monkeypatch.setattr(ArenaQueueClient, "_get", lambda self, path: queue.get(path).json())
    monkeypatch.setenv("ARENA_QUEUE_URL", "http://test")
    monkeypatch.delenv("ARENA_WAIT_TIMEOUT", raising=False)
    seed = RecurrentGemmaEnv.create_initial_state("rg_lru")
    evaluator = RecurrentGemmaRewardEvaluator("rg_lru", tmp_path, eval_timeout=2)
    out = evaluator.get_reward("def kernel(x, a, reset): pass", seed)
    assert out["reward"] == 1.0
    assert submitted[0]["enforce_pallas"] is True
    assert submitted[0]["smoke"] is False
    assert len(submitted[0]["cases"]) == 6
    saved = json.loads(next((tmp_path / "arena").glob("*.json")).read_text())
    assert saved["result"] == good_verdict() and saved["parent_id"] == seed.id


def test_grading_exception_escapes_generic_zero_reward_wrapper():
    from pallas_arena.rl.env import RecurrentGemmaEnv
    env = object.__new__(RecurrentGemmaEnv)
    env.problem_type, env.log_path, env.state = "rg_lru", "unused", None

    def fail(*args):
        raise ArenaInfrastructureError("judge unavailable")

    env._run_verification = fail
    with pytest.raises(ArenaInfrastructureError, match="unavailable"):
        asyncio.run(env._safe_grade("code", 0))


@pytest.mark.parametrize("length,near_one", [(17, False), (65, True)])
def test_seed_forward_and_both_gradients_on_ragged_resets(length, near_one, monkeypatch):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from pallas_arena.rl import seed_rglru as seed
    from pallas_arena.judge.problems.rg_lru import rg_lru_scan_reference

    monkeypatch.setattr(seed, "INTERPRET", True)
    monkeypatch.setattr(seed, "BLOCK_T", 32)
    monkeypatch.setattr(seed, "BLOCK_D", 128)
    shape = (1, length, 16)
    x = jax.random.normal(jax.random.key(7), shape).astype(jnp.bfloat16)
    a = jnp.full(shape, 1 - 1e-6) if near_one else jax.random.uniform(jax.random.key(8), shape) * 0.9
    reset = (jnp.arange(length)[None, :] % 7 == 0)
    cotangent = jnp.cos(jnp.arange(x.size)).reshape(shape)
    ref, ref_vjp = jax.vjp(lambda xv, av: rg_lru_scan_reference(xv, av, reset), x, a)
    got, got_vjp = jax.vjp(lambda xv, av: seed.kernel(xv, av, reset), x, a)
    np.testing.assert_allclose(got, ref, rtol=2e-5, atol=2e-5)
    dx, da = got_vjp(cotangent)
    ref_dx, ref_da = ref_vjp(cotangent)
    np.testing.assert_allclose(np.asarray(dx, dtype=float), np.asarray(ref_dx, dtype=float), rtol=0.008, atol=0.008)
    np.testing.assert_allclose(da, ref_da, rtol=3e-5, atol=3e-4)
    np.testing.assert_array_equal(np.asarray(da)[:, ::7], 0)


@pytest.mark.parametrize("tile", [256, 512])
def test_seed_exports_forward_and_backward_for_tpu(tile, tmp_path):
    """Interpretation alone missed the unsupported rev in the old backward."""
    from pallas_arena.judge.grader import grade
    from pallas_arena.rl.task import ARENA

    source = (ARENA / "rl/seed_rglru.py").read_text().replace("BLOCK_T = 256", f"BLOCK_T = {tile}")
    signatures = []
    cases = public_contract()[1]
    for name, dims in cases:
        shape = [dims[k] for k in ("b", "t", "d")]
        args = [{"shape": shape, "dtype": "bfloat16"},
                {"shape": shape, "dtype": "float32"},
                {"shape": shape[:2], "dtype": "bool"}]
        signatures.extend({"name": name + "-" + kind, "kind": kind, "args": args}
                          for kind in ("forward", "grad"))
    result = grade("rg_lru", source, mode="aot_export", cases=[name for name, _ in cases],
        timeout_s=120, rlimit_gb=12, enforce_pallas=True,
        child_env={"JAX_PLATFORMS": "cpu", "PALLAS_INTERPRET": "0",
                   "ARENA_EXPORT_DEVICE_KIND": "TPU v5p", "ARENA_EXPORT_DEVICE_CORES": "2"},
        export_signatures=signatures, export_platforms=["tpu"],
        artifact_dir=str(tmp_path / "exports"), workdir=str(tmp_path / "work"))
    assert result.get("passed") is True, result
    assert len(result["artifacts"]) == 2 * len(cases)
