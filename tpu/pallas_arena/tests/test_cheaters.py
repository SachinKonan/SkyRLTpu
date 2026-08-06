"""The cheater battery: every known hack class MUST fail (or, for the timer
tamperer, must gain nothing). DESIGN.md test layer 1 + adversarial addendum.
Every grade here runs through the real fork-per-candidate grader."""

import pytest

from pallas_arena.judge import grader
from pallas_arena.tests import candidates as cand


def _grade(code, **kw):
    return grader.grade("rmsnorm", code, **kw)


def test_cached_output_fails_fresh_hidden_seeds(fast_grade_kwargs):
    r = _grade(cand.CACHED_OUTPUT_RMSNORM, **fast_grade_kwargs)
    assert r["ok"] and not r["passed"]
    assert r["gate"] == "correctness"
    assert r["reward"] == 0.0


def test_aliased_reference_import_is_ast_rejected(fast_grade_kwargs):
    r = _grade(cand.ALIASED_REFERENCE_RMSNORM, **fast_grade_kwargs)
    assert not r["passed"]
    assert r["gate"] == "ast"
    assert any("banned import" in v for v in r["violations"])


def test_obfuscated_import_hits_poison_stub(fast_grade_kwargs):
    """importlib.import_module("pallas_" + "arena...") is invisible to the
    AST gate but the child's sys.modules poisoning catches the access."""
    r = _grade(cand.OBFUSCATED_IMPORT_RMSNORM, **fast_grade_kwargs)
    assert not r["passed"]
    assert r["gate"] in ("poison_stub", "correctness")
    if r["gate"] == "correctness":
        # the poison must be the cause, not a numeric miss
        assert any("off-limits" in v or "ArenaBannedImport" in v for v in r["violations"])


def test_seed_reader_finds_nothing_and_fails(fast_grade_kwargs):
    r = _grade(cand.SEED_READER_RMSNORM, **fast_grade_kwargs)
    assert not r["passed"]
    assert r["gate"] == "correctness"


def test_timer_tamperer_gains_nothing(fast_grade_kwargs):
    """Patches time.perf_counter to 0 then runs slow-but-correct code. The
    harness's held reference times it honestly: it must NOT look fast."""
    kw = dict(fast_grade_kwargs, timeout_s=300.0)
    r = _grade(cand.TIMER_TAMPERER_RMSNORM, **kw)
    assert r["ok"], r
    assert r["passed"], r  # correct code — passes gates
    # 200 un-jitted dispatches vs one fused baseline: a real slowdown must
    # be measured. If the tampering had reached the timer, the ratio would
    # have exploded toward +inf.
    assert r["score"] is not None and r["score"] < 0.9, r["score"]
    assert r["reward"] < 1.0 + 1e-9


def test_python_memoizer_fails_on_fresh_inputs(fast_grade_kwargs):
    r = _grade(cand.MEMOIZER_RMSNORM, **fast_grade_kwargs)
    assert not r["passed"]
    assert r["gate"] == "correctness"


def test_split_personality_caught_on_timed_outputs(fast_grade_kwargs):
    """Correct through the gates, garbage during timing: killed only
    because correctness is verified on outputs of TIMED invocations."""
    r = _grade(cand.SPLIT_PERSONALITY_RMSNORM, **fast_grade_kwargs)
    assert r["ok"] and not r["passed"], r
    assert r["gate"] in ("timed_output_correctness", "correctness", "determinism", "gradient")


def test_wrong_backward_caught_by_gradient_contract(fast_grade_kwargs):
    r = _grade(cand.WRONG_GRAD_RMSNORM, **fast_grade_kwargs)
    assert not r["passed"]
    assert r["gate"] == "gradient"


def test_nondeterministic_kernel_fails_bitwise_check(fast_grade_kwargs):
    r = _grade(cand.NONDETERMINISTIC_RMSNORM, **fast_grade_kwargs)
    assert not r["passed"]
    assert r["gate"] == "determinism"


def test_pallas_requirement_rejects_plain_jax_when_enforced(fast_grade_kwargs):
    """splash requires a real pallas_call; the same naive-jax candidate is
    AST-rejected when enforced and passes gates when enforcement is off."""
    kw = dict(fast_grade_kwargs, cases=["tiny"], mode="gates")
    r = grader.grade("splash_attention", cand.NAIVE_SPLASH, enforce_pallas=True, **kw)
    assert not r["passed"]
    assert r["gate"] == "ast"
    assert any("pallas_call" in v for v in r["violations"])

    r2 = grader.grade("splash_attention", cand.NAIVE_SPLASH, enforce_pallas=False, **kw)
    assert r2["passed"], r2


def test_banned_attention_call_rejected():
    """jax.nn attention entry points are banned for attention tasks."""
    from pallas_arena.judge.gates import ast_gate
    from pallas_arena.judge.problems import get_problem

    p = get_problem("splash_attention")
    code = "import jax\n" "def kernel(q, k, v, seg):\n" "    return jax.nn.dot_product_attention(q, k, v)\n"
    v = ast_gate(
        code, banned_import_prefixes=p.all_banned_prefixes, banned_call_names=p.banned_call_names, require_pallas=False
    )
    assert any("banned call" in s for s in v)


@pytest.mark.parametrize(
    "snippet,expect",
    [
        ("import tpu_inference.kernels", "banned import"),
        ("from recurrentgemma.jax import pallas", "banned import"),
        ("from jax.experimental.pallas.ops.tpu import splash_attention", "banned import"),
        ("import skyrl.backends.tunix_backend", "banned import"),
        ("from jax.experimental import pallas as pl\n" "def kernel(x, g):\n    return pl.pallas_call\n", None),
    ],
)
def test_ast_gate_prefix_matrix(snippet, expect):
    from pallas_arena.judge.gates import ast_gate
    from pallas_arena.judge.problems import get_problem

    p = get_problem("splash_attention")
    v = ast_gate(
        snippet,
        banned_import_prefixes=p.all_banned_prefixes,
        banned_call_names=p.banned_call_names,
        require_pallas=False,
    )
    if expect is None:
        assert v == []
    else:
        assert any(expect in s for s in v)
