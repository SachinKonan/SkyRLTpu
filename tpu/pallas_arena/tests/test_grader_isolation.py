"""Grader isolation mechanics: fork-per-candidate, timeout, RLIMIT_AS,
seed secrecy, golden pass / subtly-wrong fail (test layers 1+2, CPU)."""

import json
from pathlib import Path

from pallas_arena.judge import grader
from pallas_arena.tests import candidates as cand


def test_honest_kernel_full_grade_passes(fast_grade_kwargs):
    r = grader.grade("rmsnorm", cand.HONEST_RMSNORM, **fast_grade_kwargs)
    assert r["ok"] and r["passed"], r
    assert r["gate"] == "all"
    assert r["score"] is not None and 0.2 < r["score"] < 5.0
    assert r["reward"] is not None and r["reward"] > 0
    assert r["baseline_available"]
    assert "tiny" in r["per_case"]
    assert r["backend"] == "cpu"


def test_holdout_logged_never_scored(fast_grade_kwargs):
    kw = dict(fast_grade_kwargs, cases=["tiny", "tiny-holdout"])
    r = grader.grade("rmsnorm", cand.HONEST_RMSNORM, **kw)
    assert r["passed"], r
    assert set(r["per_case"]) == {"tiny"}
    assert set(r["holdout"]) == {"tiny-holdout"}


def test_subtly_wrong_eps_fails_numerics(fast_grade_kwargs):
    """DESIGN layer 2: wrong-LN-epsilon must fail the calibrated tolerance."""
    r = grader.grade("rmsnorm", cand.WRONG_EPS_RMSNORM, **fast_grade_kwargs)
    assert r["ok"] and not r["passed"], r
    assert r["gate"] in ("correctness", "gradient")
    assert r["reward"] == 0.0


def test_no_kernel_function_fails(fast_grade_kwargs):
    r = grader.grade("rmsnorm", cand.NO_KERNEL, **fast_grade_kwargs)
    assert not r["passed"] and r["gate"] == "exec"


def test_seed_never_in_config_env_or_argv(fast_grade_kwargs, tmp_path):
    """The hidden seed crosses ONLY on stdin: the config file the child (and
    thus the candidate) can read contains no seed."""
    wd = tmp_path / "wd"
    r = grader.grade("rmsnorm", cand.HONEST_RMSNORM, workdir=str(wd), seed=123456789, **fast_grade_kwargs)
    assert r["passed"], r
    cfg_text = (wd / "config.json").read_text()
    assert "seed" not in json.loads(cfg_text)
    assert "123456789" not in cfg_text


def test_timeout_kills_process_group(fast_grade_kwargs):
    kw = dict(fast_grade_kwargs, timeout_s=20.0)
    r = grader.grade("rmsnorm", cand.SLEEPER, **kw)
    assert not r["passed"]
    assert r.get("timed_out") or r["gate"] == "timeout"


def test_rlimit_as_kills_memory_hog(fast_grade_kwargs):
    kw = dict(fast_grade_kwargs, rlimit_gb=3.0)
    r = grader.grade("rmsnorm", cand.MEMORY_HOG, **kw)
    assert not r["passed"]
    # either the child caught MemoryError while exec'ing the candidate, or
    # the allocator died outright — both must yield a failed, 0-reward grade
    assert r["gate"] in ("exec", "rlimit", "harness", "correctness")
    assert r.get("reward", 0.0) == 0.0


def test_deterministic_gates_across_regrades(fast_grade_kwargs):
    """Same candidate, two independent grades (fresh seeds): gate verdicts
    must agree; with a shared explicit seed, correctness is reproducible."""
    r1 = grader.grade("rmsnorm", cand.WRONG_EPS_RMSNORM, **fast_grade_kwargs)
    r2 = grader.grade("rmsnorm", cand.WRONG_EPS_RMSNORM, **fast_grade_kwargs)
    assert r1["passed"] == r2["passed"] is False
    assert r1["gate"] == r2["gate"]


def test_child_artifacts_in_workdir(fast_grade_kwargs, tmp_path):
    wd = tmp_path / "wd2"
    r = grader.grade("rmsnorm", cand.HONEST_RMSNORM, workdir=str(wd), **fast_grade_kwargs)
    assert r["passed"]
    assert (wd / "result.json").exists()
    assert (wd / "child.stdout").exists()
    assert (wd / "child.stderr").exists()
    assert Path(wd / "config.json").exists()


def test_cache_key_normalizes_whitespace():
    k1 = grader.cache_key("rmsnorm", "1", "full", "def kernel(x, g):\n    return x\n", False)
    k2 = grader.cache_key("rmsnorm", "1", "full", "def kernel(x, g):   \n    return x", False)
    k3 = grader.cache_key("rmsnorm", "1", "full", "def kernel(x, g):\n    return  x", False)
    assert k1 == k2
    assert k1 != k3
