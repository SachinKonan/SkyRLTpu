"""Unit tests for resume-after-preemption and deliberate-restart semantics.

Targets ttt_discover/rl/resume.py, which is deliberately import-light so these
can run without tinker/torch/wandb or a TPU.

Every case here corresponds to a failure that actually happened on the fleet:

  * a stale snapshot with FEWER states outranking the live one (nearly lost a
    world-record leaf)
  * one member's 404 dragging the whole ensemble back to step 0
  * a resumed wandb run silently dropping every point after a restart
  * restarting a still-productive run, and failing to restart a stalled one
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# Loaded by path, not by package import: ttt_discover/__init__.py pulls in chz
# and the rest of the discovery stack, which would defeat the point of keeping
# resume.py import-light and force these tests into the discover venv.
_RESUME_PY = (Path(__file__).resolve().parents[2]
              / "third_party" / "discover" / "ttt_discover" / "rl" / "resume.py")
_spec = importlib.util.spec_from_file_location("ttd_resume", _RESUME_PY)
assert _spec and _spec.loader, f"cannot load {_RESUME_PY}"
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def write_snapshot(d: Path, step: int, n_states: int) -> Path:
    p = d / f"puct_sampler_step_{step:06d}.json"
    p.write_text(json.dumps({"states": [{"id": i, "value": -0.38} for i in range(n_states)]}))
    return p


# ---------------------------------------------------------------------------
# Tree snapshot selection -- preemption must not lose discoveries
# ---------------------------------------------------------------------------

def test_picks_largest_tree_not_highest_step(tmp_path: Path):
    """The regression that nearly cost a world record.

    After a fresh-weight relaunch the batch counter restarts at 0 while the tree
    resumes higher, so flush() rewrites step_000001..N over the previous life's
    series. The highest-numbered file can be a stale, smaller tree.
    """
    write_snapshot(tmp_path, 3, 485)   # live tree
    write_snapshot(tmp_path, 5, 396)   # stale leftover, higher number
    step, path, n = R.pick_resume_snapshot(str(tmp_path))
    assert (step, n) == (3, 485)
    assert "step_000003" in path
    assert R.snapshot_is_shadowed(str(tmp_path)) is True


def test_ties_break_on_step_number(tmp_path: Path):
    write_snapshot(tmp_path, 2, 100)
    write_snapshot(tmp_path, 7, 100)
    step, _, n = R.pick_resume_snapshot(str(tmp_path))
    assert (step, n) == (7, 100)


def test_no_shadowing_flag_when_newest_is_largest(tmp_path: Path):
    write_snapshot(tmp_path, 1, 10)
    write_snapshot(tmp_path, 2, 40)
    assert R.snapshot_is_shadowed(str(tmp_path)) is False


def test_no_snapshots_returns_none(tmp_path: Path):
    assert R.pick_resume_snapshot(str(tmp_path)) is None


def test_corrupt_snapshot_does_not_win(tmp_path: Path):
    """A half-written snapshot must never be selected over a good one."""
    write_snapshot(tmp_path, 1, 50)
    (tmp_path / "puct_sampler_step_000002.json").write_text("{not json")
    step, _, n = R.pick_resume_snapshot(str(tmp_path))
    assert (step, n) == (1, 50)


# ---------------------------------------------------------------------------
# Step resolution -- resume at the SAME step, never silently at 0
# ---------------------------------------------------------------------------

def test_resumes_at_the_recorded_step(tmp_path: Path):
    assert R.resolve_start_batch([7, 7]) == 7


def test_single_member_404_is_a_hard_failure(tmp_path: Path):
    """Previously: min([10, 0]) == 0, so one 404 restarted the whole ensemble at
    step 0 while the other member kept step-10 weights."""
    with pytest.raises(R.ResumeError, match="failed to resume"):
        R.resolve_start_batch([10, None])


def test_member_disagreement_is_a_hard_failure():
    with pytest.raises(R.ResumeError, match="disagree"):
        R.resolve_start_batch([10, 9])


def test_non_strict_preserves_legacy_min_behaviour():
    assert R.resolve_start_batch([10, None], strict=False) == 0
    assert R.resolve_start_batch([10, 9], strict=False) == 9


def test_empty_member_list_raises():
    with pytest.raises(R.ResumeError):
        R.resolve_start_batch([])


# ---------------------------------------------------------------------------
# Monotonic global step -- wandb resume across lives
# ---------------------------------------------------------------------------

def test_global_step_survives_process_restart(tmp_path: Path):
    for _ in range(3):
        R.advance_global_step(str(tmp_path))
    assert R.read_global_step(str(tmp_path)) == 3
    # simulate a fresh process: only the file persists
    assert R.read_global_step(str(tmp_path)) == 3


def test_global_step_never_decreases_across_a_restart(tmp_path: Path):
    """The arm-R hazard: the per-life counter resets to 0, but wandb rejects
    out-of-order steps on a resumed run and would drop every later point."""
    steps = [R.advance_global_step(str(tmp_path)) for _ in range(5)]
    # deliberate restart: per-life batch counter goes back to 0
    after = [R.advance_global_step(str(tmp_path)) for _ in range(3)]
    assert steps == [1, 2, 3, 4, 5]
    assert after == [6, 7, 8]
    assert after == sorted(after) and after[0] > steps[-1]


def test_missing_state_file_starts_at_zero(tmp_path: Path):
    assert R.read_global_step(str(tmp_path / "nonexistent")) == 0


def test_seeds_from_existing_metrics_when_state_file_absent(tmp_path: Path):
    """A run that predates this counter has already logged N steps to its wandb
    run. Starting the counter at 0 would emit steps BELOW wandb's max, and wandb
    drops out-of-order steps on a resumed run -- silently losing everything."""
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps({"step": i}) + "\n" for i in range(9))
    )
    assert R.read_global_step(str(tmp_path)) == 9
    assert R.advance_global_step(str(tmp_path)) == 10


def test_state_file_takes_over_after_first_write(tmp_path: Path):
    """Seeded once from the 1 pre-existing metrics row, then the state file is
    authoritative -- so the count continues 2,3,4,5 rather than re-seeding."""
    (tmp_path / "metrics.jsonl").write_text('{"step": 0}\n')
    seen = [R.advance_global_step(str(tmp_path)) for _ in range(4)]
    assert seen == [2, 3, 4, 5]
    assert R.read_global_step(str(tmp_path)) == 5


def test_blank_lines_in_metrics_are_not_counted(tmp_path: Path):
    (tmp_path / "metrics.jsonl").write_text('{"step": 0}\n\n{"step": 1}\n\n')
    assert R.read_global_step(str(tmp_path)) == 2


# ---------------------------------------------------------------------------
# Arm R trigger -- validated against the four historical restart events
# ---------------------------------------------------------------------------

# (gains per step of the life, whether a restart was warranted, label)
HISTORICAL = [
    # ctrl life 0: decayed 692x by step 4; the restart there recovered 91x
    ([2.550e-04, 2.849e-04, 1.706e-05, 4.119e-07], True, "ctrl-0"),
    # main life 4: decayed to 1e-9..1e-11; trigger should fire by step 8
    ([8.751e-08, 2.817e-08, 5.262e-07, 1.565e-06, 1.134e-06,
      1.339e-07, 9.800e-08, 2.378e-08, 1.822e-09], True, "main-4"),
    # 4-agent life 1: 4.24e-06 -> 3.69e-09, recovered 48x
    ([4.239e-06, 3.686e-09], True, "4ag-1"),
    # 4-agent life 0: only 14x decay, still productive -- restarting gained 1.2x
    ([5.064e-05, 4.071e-05, 1.445e-05, 1.448e-05,
      7.597e-06, 4.079e-06, 3.687e-06], False, "4ag-0"),
]


@pytest.mark.parametrize("gains,should_fire,label", HISTORICAL)
def test_trigger_matches_historical_outcomes(gains, should_fire, label):
    assert R.should_restart(gains) is should_fire, label


def test_trigger_is_single_step_not_two_step_smoothed():
    """ctrl's step-3 gain (1.71e-05) sits well above peak/100, so a
    'two consecutive steps below threshold' rule would have missed the step-4
    restart that recovered 91x."""
    ctrl = [2.550e-04, 2.849e-04, 1.706e-05, 4.119e-07]
    assert R.should_restart(ctrl) is True
    assert R.should_restart(ctrl[:-1]) is False  # step 3 alone must not fire


def test_trigger_needs_a_prior_peak():
    assert R.should_restart([1.0]) is False
    assert R.should_restart([]) is False


def test_trigger_ignores_all_zero_life():
    """A life that never improved has no peak to decay from."""
    assert R.should_restart([0.0, 0.0, 0.0]) is False


def test_min_life_steps_guard():
    gains = [1.0, 1e-9]
    assert R.should_restart(gains) is True
    assert R.should_restart(gains, min_life_steps=3) is False


def test_gains_derive_from_running_best_series():
    """pool/best_value is already in metrics.jsonl -- no new instrumentation."""
    best = [-0.3810, -0.38095, -0.38095, -0.380872]
    gains = R.gains_from_best_values(best)
    assert gains[0] == 0.0
    assert gains[1] == pytest.approx(5e-05, rel=1e-6)
    assert gains[2] == 0.0            # no improvement that step
    assert gains[3] == pytest.approx(7.8e-05, rel=1e-6)


def test_gains_never_negative_on_regression():
    """best_value should be monotone, but a stale read must not yield a negative
    gain that corrupts the peak."""
    assert all(g >= 0 for g in R.gains_from_best_values([-0.381, -0.3805, -0.382]))


def test_discovery_jump_does_not_poison_the_peak():
    """A fresh run's family-discovery step gains ~0.1 while steady-state gains
    are ~1e-4. With peak = max, the trigger fires ONE STEP after the run's
    biggest discovery -- restarting a productive run, which historically gained
    nothing. The robust (second-largest) peak must hold instead."""
    gains = [0.0, 0.109, 3.3e-4, 1.2e-4]     # jump at step 1, healthy after
    assert R.should_restart(gains) is False
    # ...but a genuine collapse after the jump still fires
    assert R.should_restart([0.0, 0.109, 3.3e-4, 1.2e-4, 8e-5, 1e-7]) is True


def test_robust_peak_uses_max_below_three_observations():
    """With <3 prior gains there is no second-largest to trust; fall back to
    max (4-agent life 1 fired correctly off exactly two observations)."""
    assert R.should_restart([4.239e-06, 3.686e-09]) is True


# ---------------------------------------------------------------------------
# Life state -- must survive preemption
# ---------------------------------------------------------------------------

def test_life_state_round_trips(tmp_path: Path):
    state = {"gains": [0.0, 1e-4, 2e-5], "best": -0.3809, "restarts": 2}
    R.write_life_state(str(tmp_path), state)
    got = R.read_life_state(str(tmp_path))
    assert got["gains"] == [0.0, 1e-4, 2e-5]
    assert got["best"] == -0.3809
    assert got["restarts"] == 2


def test_life_state_missing_is_fresh(tmp_path: Path):
    assert R.read_life_state(str(tmp_path / "nope")) == {
        "gains": [], "best": None, "restarts": 0}


def test_life_state_survives_preemption_semantics(tmp_path: Path):
    """A preemption reloads the same life (history intact, so the trigger is
    not starved under churn); a deliberate restart clears gains but keeps the
    restart count."""
    R.write_life_state(str(tmp_path), {"gains": [0.0, 5e-5], "best": -0.381,
                                       "restarts": 0})
    life = R.read_life_state(str(tmp_path))          # preemption -> same life
    assert life["gains"] == [0.0, 5e-5]
    life["gains"] = []                                # deliberate restart
    life["restarts"] += 1
    R.write_life_state(str(tmp_path), life)
    after = R.read_life_state(str(tmp_path))
    assert after["gains"] == [] and after["restarts"] == 1
