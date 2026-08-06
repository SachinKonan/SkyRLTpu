"""hash->reward cache semantics + the CPU AOT pre-gate."""

from pallas_arena.judge import grader
from pallas_arena.judge.aot_gate import aot_pregate
from pallas_arena.judge.cache import RewardCache
from pallas_arena.tests import candidates as cand


def test_cache_hit_returns_identical_reward(fast_grade_kwargs, local_cache):
    kw = dict(fast_grade_kwargs, cache=local_cache)
    r1 = grader.grade("rmsnorm", cand.HONEST_RMSNORM, **kw)
    assert r1["passed"] and not r1["cache_hit"]
    r2 = grader.grade("rmsnorm", cand.HONEST_RMSNORM, **kw)
    assert r2["cache_hit"], "second grade must come from the cache"
    assert r2["reward"] == r1["reward"]
    assert r2["score"] == r1["score"]
    # whitespace-only variants hit the same entry (hash-dedup funnel)
    r3 = grader.grade("rmsnorm", cand.HONEST_RMSNORM + "\n\n   \n", **kw)
    assert r3["cache_hit"]


def test_cache_stores_failures_consistently(fast_grade_kwargs, local_cache):
    kw = dict(fast_grade_kwargs, cache=local_cache)
    r1 = grader.grade("rmsnorm", cand.WRONG_EPS_RMSNORM, **kw)
    assert not r1["passed"]
    r2 = grader.grade("rmsnorm", cand.WRONG_EPS_RMSNORM, **kw)
    assert r2["cache_hit"] and not r2["passed"]
    assert r2["gate"] == r1["gate"]


def test_cache_local_roundtrip(tmp_path):
    c = RewardCache(str(tmp_path / "rc"))
    assert c.get("ab" * 32) is None
    c.put("ab" * 32, {"reward": 1.25, "gate": "all"})
    assert c.get("ab" * 32) == {"reward": 1.25, "gate": "all"}


def test_cache_gcs_url_parsing():
    c = RewardCache("gs://bucket/prefix/")
    assert c.is_gcs
    assert c._path("ff" * 32).startswith("gs://bucket/prefix/ff/")


def test_pregate_accepts_honest_kernel(fast_grade_kwargs):
    ok, why = aot_pregate("rmsnorm", cand.HONEST_RMSNORM, smoke=True)
    assert ok, why


def test_pregate_rejects_syntax_error_instantly():
    ok, why = aot_pregate("rmsnorm", "def kernel(x, g:\n    return x")
    assert not ok and "syntax" in why.lower()


def test_pregate_rejects_banned_import_before_forking():
    ok, why = aot_pregate("rmsnorm", cand.ALIASED_REFERENCE_RMSNORM)
    assert not ok and "banned import" in why


def test_pregate_rejects_trace_time_failure(fast_grade_kwargs):
    bad = "import jax.numpy as jnp\n" "def kernel(x, g):\n" "    return x + undefined_name\n"
    ok, why = aot_pregate("rmsnorm", bad, smoke=True)
    assert not ok
    assert "pregate" in why or "exec" in why


def test_pregate_shape_mismatch_caught_at_trace(fast_grade_kwargs):
    bad = (
        "import jax.numpy as jnp\n" "def kernel(x, g):\n" "    return jnp.matmul(x, x)\n"
    )  # [32,64]@[32,64]: shape error
    ok, why = aot_pregate("rmsnorm", bad, smoke=True)
    assert not ok
