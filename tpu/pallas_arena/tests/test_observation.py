"""Curated observation strings — one test per gate the judge can emit.

The property under test is not "the string is pretty". It is: after the RL
side truncates to the LAST 500 characters (``ttt_discover.tinker_utils.
state.State.to_prompt``), the measured numbers are still there.
"""

from __future__ import annotations

import pytest

from pallas_arena.judge.observation import MAX_CHARS, attach_observation, build_observation

# The exact phrasing measured on silicon in phase 2/4 (see
# runs/pallas_arena/phase4-results-3646330.json).
VMEM_OOM = (
    "XlaRuntimeError: RESOURCE_EXHAUSTED: Error loading program: Attempting to allocate "
    "36.25M. That was not possible. There are 32.00M free.; (0x0x0_HBM0): while running "
    "replica 0 and partition 0 of a replicated computation (other replicas may have failed "
    "as well).\nCompileTimeScopedVmemOom:\nRan out of memory in memory space vmem while "
    "allocating on stack for %_fwd.1 = bf16[16384,6144]{1,0:T(8,128)(2,1)} "
    "custom-call(%args_0_.1, %bitcast.1), custom_call_target=\"tpu_custom_call\", "
    "operand_layout_constraints={bf16[16384,6144]{1,0}, f32[1,6144]{1,0}}, "
    "frontend_attributes={kernel_metadata={}}, metadata={op_name=\"jit(call)/call_exported/"
    "jit(kernel)/jit(_fwd)/pallas_call\" stack_frame_id=23}. Scoped allocation with size "
    "36.25M and limit 32.00M exceeded scoped vmem limit by 4.25M. It should not be possible "
    "to run out of scoped vmem - \nSee https://openxla.org/xla/errors/error_1001 for more "
    "details."
)


def tail(obs: str, n: int = 500) -> str:
    """What the RL prompt actually keeps."""
    return obs if len(obs) <= n else obs[-n:]


def _fail(gate: str, *violations, **extra) -> dict:
    return {"ok": True, "passed": False, "gate": gate, "reward": 0.0, "problem": "splash_attention",
            "violations": list(violations), **extra}


# --------------------------------------------------------------- invariants
ALL_GATES = [
    "ast", "poison_stub", "exec", "aot_export", "pregate", "candidate_compile",
    "compile_budget", "correctness", "timed_output_correctness", "gradient",
    "determinism", "budget", "timeout", "rlimit", "harness", "worker",
    "artifact_load", "fixtures",
]


@pytest.mark.parametrize("gate", ALL_GATES)
def test_every_gate_produces_a_bounded_observation_naming_itself(gate):
    obs = build_observation(_fail(gate, "some diagnostic text " * 40))
    assert obs.startswith(f"GATE {gate} |"), obs
    assert len(obs) <= MAX_CHARS
    assert obs == tail(obs), "must survive the 500-char tail truncation whole"


def test_pass_verdict_reports_score_and_where_the_time_went():
    obs = build_observation(
        {
            "passed": True, "gate": "all", "reward": 0.8261, "problem": "flce",
            "latencies": {"probe-4096x2880x151936": {"ref_median_s": 0.0121, "cand_median_s": 0.0146}},
            "peak_hbm_bytes": 18_400_000_000,
        }
    )
    assert "PASS reward=0.8261" in obs
    assert "12.100ms" in obs and "14.600ms" in obs
    assert "peak HBM 18.40GB" in obs
    assert len(obs) <= MAX_CHARS


# ------------------------------------------------------------ the numbers
def test_scoped_vmem_oom_keeps_the_two_numbers_that_matter():
    obs = build_observation(_fail("aot_export", VMEM_OOM))
    assert "CompileTimeScopedVmemOom: 36.25M requested vs 32.00M limit" in obs
    assert "over by 4.25M" in obs
    assert "bf16[16384,6144]" in obs
    assert "32MB" in obs  # the ceiling, stated
    assert len(obs) <= MAX_CHARS
    # the whole point: the numbers survive truncation
    assert "36.25M requested vs 32.00M limit" in tail(obs)
    # and the stack noise does NOT
    assert "operand_layout_constraints" not in obs
    assert "stack_frame_id" not in obs


def test_shape_mismatch_reports_both_shapes():
    obs = build_observation(
        _fail("correctness", "probe-holdout-h4-s2049#seed0: non-finite or malformed output "
              "(shape mismatch (4, 2048, 128) vs (4, 2049, 128))")
    )
    assert "output shape mismatch (4,2048,128) vs required (4,2049,128)" in obs
    assert "probe-holdout-h4-s2049#seed0" in obs
    assert len(obs) <= MAX_CHARS


def test_tolerance_miss_reports_error_and_tolerance_and_case():
    obs = build_observation(
        _fail("correctness", "probe-h8-s4096#seed1: per-element max error 3.100e-02 exceeds "
              "calibrated tolerance 8.000e-03")
    )
    assert "GATE correctness | splash_attention | probe-h8-s4096#seed1" in obs
    assert "max err 0.031 vs tol 0.008" in obs
    assert len(obs) <= MAX_CHARS


def test_q99_tail_miss_is_named_as_the_tail_not_the_max():
    obs = build_observation(
        _fail("correctness", "probe-h4-s2048#seed0: per-element q99 error tail 1.2e-02 exceeds "
              "calibrated tolerance 8.0e-03")
    )
    assert "q99 err 0.012 vs tol 0.008" in obs
    assert "max err" not in obs


def test_gradient_gate_names_the_custom_vjp_requirement():
    obs = build_observation(
        _fail("gradient", "per-element max error 4.400e-01 exceeds calibrated tolerance 1.100e-02")
    )
    assert obs.startswith("GATE gradient | splash_attention")
    assert "max err 0.44 vs tol 0.011" in obs
    assert "custom_vjp" in obs


def test_compile_budget_reports_elapsed_budget_and_unit():
    obs = build_observation(
        _fail("compile_budget", "candidate compile exceeded the 90s compile budget "
              "(90.0s inside a single un-cancellable XLA compile at unit 'fwd01'); judge restarting",
              compile_budget_s=90.0, candidate_compile_s=90.04)
    )
    assert "compile took 90.0s vs 90s budget at unit fwd01" in obs
    assert len(obs) <= MAX_CHARS


def test_determinism_gate_states_the_contract():
    obs = build_observation(_fail("determinism", "outputs not bitwise identical across 5 runs"))
    assert "not bitwise identical across 5 runs" in obs
    assert "bitwise identical" in obs.split("need:")[-1]


# --------------------------------------------------------- noisy diagnostics
def test_exec_traceback_is_reduced_to_the_exception_and_the_candidate_line():
    tb = (
        'Traceback (most recent call last):\n'
        '  File "/home/x/.venv/lib/python3.12/site-packages/pallas_arena/judge/child_runner.py", line 238, in _run\n'
        '    exec(compile(code, "<candidate>", "exec"), cand_mod.__dict__)\n'
        '  File "<candidate>", line 47, in <module>\n'
        '    _BLOCK = compute_block(seq)\n'
        '  File "<candidate>", line 31, in compute_block\n'
        '    return seq // 0\n'
        'ZeroDivisionError: integer division or modulo by zero\n'
    )
    obs = build_observation(_fail("exec", tb))
    assert "ZeroDivisionError: integer division or modulo by zero" in obs
    assert "candidate line 31 in compute_block" in obs
    assert "site-packages" not in obs
    assert "Traceback (most recent call last)" not in obs
    assert len(obs) <= MAX_CHARS


def test_missing_entrypoint_is_reported_as_such():
    obs = build_observation(_fail("exec", "no `kernel` function defined"))
    assert "no `kernel` function defined" in obs


def test_ast_gate_keeps_the_banned_name():
    obs = build_observation(
        _fail("ast", "banned import 'jax.experimental.pallas.ops.tpu.splash_attention' at line 4")
    )
    assert "jax.experimental.pallas.ops.tpu.splash_attention" in obs
    assert "line 4" in obs


def test_poison_stub_gate_names_the_module_reached_at_runtime():
    obs = build_observation(_fail("poison_stub", "ArenaBannedImport: banned module 'skyrl' imported at runtime"))
    assert "skyrl" in obs


def test_mosaic_not_implemented_keeps_only_the_message():
    raw = (
        "NotImplementedError: Mosaic failed to compile TPU kernel: Not implemented: "
        "unsupported shape cast from (8, 4096, 128) to (8, 32, 128, 128) "
        "at location loc(\"/dot_general\"(callsite(...))) " + "x" * 900
    )
    obs = build_observation(_fail("aot_export", raw))
    assert "unsupported shape cast from (8, 4096, 128)" in obs
    assert len(obs) <= MAX_CHARS
    assert obs == tail(obs)


def test_broadcast_shape_error_is_kept_verbatim():
    obs = build_observation(
        _fail("aot_export", "TypeError: Incompatible shapes for broadcast: (4, 18433, 128) and "
              "(4, 18432, 128)" + " trailing junk" * 60)
    )
    assert "Incompatible shapes for broadcast: (4, 18433, 128) and (4, 18432, 128)" in obs
    assert len(obs) <= MAX_CHARS


def test_a_giant_single_line_diagnostic_is_elided_in_the_MIDDLE():
    raw = "SomeError: " + "A" * 400 + "THE-TAIL-MATTERS"
    obs = build_observation(_fail("worker", raw))
    assert len(obs) <= MAX_CHARS
    assert "SomeError:" in obs
    assert "THE-TAIL-MATTERS" in obs
    assert "..." in obs


def test_timeout_and_rlimit_carry_their_limits():
    assert "timed out after 240s" in build_observation(_fail("timeout", "timed out after 240s"))
    assert "RLIMIT_AS 32G" in build_observation(_fail("rlimit", "child killed (rc=-9; likely RLIMIT_AS 32G exceeded)"))


# ------------------------------------------------------------------ safety
def test_observation_never_raises_and_is_attached_in_place():
    for bad in ({}, {"gate": None}, {"violations": "a bare string"}, {"violations": [None, 3]}):
        assert isinstance(build_observation(bad), str)
    r = {"gate": "ast", "violations": ["x"]}
    assert attach_observation(r) is r and r["observation"].startswith("GATE ast")


def test_hints_can_be_switched_off():
    obs = build_observation(_fail("gradient", "per-element max error 1e-1 exceeds calibrated tolerance 1e-3"),
                            hints=False)
    assert "need:" not in obs
    assert "max err 0.1 vs tol 0.001" in obs


def test_no_absolute_paths_leak_into_the_observation():
    obs = build_observation(_fail("exec", 'File "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/x.py", line 3\nValueError: bad'))
    assert "/n/fs/" not in obs
