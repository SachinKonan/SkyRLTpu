"""Regression tests for PerennialState selection.

The headline test is `test_visits_actually_damp`: it fails against the code that produced every
perennial/team/team-split cell of the RQ2 campaign, where `_select` looked up `vis.get(nid, 0)`
with an int while the dict is keyed by str, so the 1/(1+visits) term was permanently 1.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "client"))
import state_reuse as SR  # noqa: E402


def _mk(tmp_path, maximize=False, seed_score=1.0):
    return SR.PerennialState(tmp_path, maximize, "seed_program", seed_score,
                             n_agents=1, shared_buffer=True)


def _add(st, program, score, parent, rnd=1):
    """Commit a child under `parent` the way update() does, and bump the parent's visit count."""
    store = st.stores[0]
    nid = store.add(program, score, [parent], feedback="", rnd=rnd, summary="")
    st.visits["0"][str(parent)] = st.visits["0"].get(str(parent), 0) + 1
    return nid


def test_visits_are_written_as_strings(tmp_path):
    """The write side stringifies, and JSON forces strings on every reload."""
    st = _mk(tmp_path)
    _add(st, "p1", 0.9, 0)
    assert list(st.visits["0"]) == ["0"]
    (tmp_path / "visits.json").write_text(json.dumps(st.visits))
    reloaded = json.loads((tmp_path / "visits.json").read_text())
    assert all(isinstance(k, str) for k in reloaded["0"])


def _hot_cold(tmp_path, hot_visits):
    """`hot` is the BETTER node but heavily mined; `cold` is worse but untouched.

    Without damping the better node always wins, so which one `_select` returns is a direct
    readout of whether the 1/(1+visits) term is live.
    """
    st = _mk(tmp_path, maximize=True, seed_score=0.0)
    hot = _add(st, "hot", 0.52, 0)
    cold = _add(st, "cold", 0.50, 0)
    if hot_visits:
        st.visits["0"][str(hot)] = hot_visits
    return st, hot, cold


def test_visits_actually_damp(tmp_path):
    """The campaign bug: with the int lookup the denominator was always 1, so the higher-valued
    node won no matter how hard it had already been mined."""
    st, hot, cold = _hot_cold(tmp_path, hot_visits=16)
    picks = st._select(st.stores[0], st.visits["0"], 1, None)
    assert picks[0] == cold, (
        "a 16x-mined node still outranked an unmined one -- the visits denominator is inert")


def test_unmined_better_node_still_wins(tmp_path):
    """Guard against over-correcting: with no mining history the better node must win."""
    st, hot, cold = _hot_cold(tmp_path, hot_visits=0)
    picks = st._select(st.stores[0], st.visits["0"], 1, None)
    assert picks[0] == hot


def test_damping_is_monotone_in_visit_count(tmp_path):
    """More visits must never make a node more attractive; the flip happens once, not never."""
    winners = []
    for k, v in enumerate((0, 1, 2, 4, 8, 16)):
        st, hot, cold = _hot_cold(tmp_path / f"v{k}", hot_visits=v)
        winners.append(st._select(st.stores[0], st.visits["0"], 1, None)[0] == hot)
    assert winners[0] is True, "unmined better node should win"
    assert winners[-1] is False, "heavily mined better node should lose"
    # once it flips to cold it must stay flipped
    flipped = winners.index(False)
    assert all(w is False for w in winners[flipped:]), f"non-monotone: {winners}"


@pytest.mark.parametrize("n_agents,shared", [(1, True), (2, True), (2, False)])
def test_warm_start_from_copied_state(tmp_path, n_agents, shared):
    """The meta-driver's expansion primitive: copy a state dir, reopen, keep the buffer."""
    import shutil
    src = tmp_path / "parent"
    st = SR.PerennialState(src, False, "seed", 1.0, n_agents=n_agents, shared_buffer=shared)
    for i in range(3):
        _add(st, f"p{i}", 0.9 - i * 0.01, 0)
    for s in st.stores:
        s.save()
    (src / "visits.json").write_text(json.dumps(st.visits))
    before = sum(len(s.g.nodes) for s in st.stores)

    dst = tmp_path / "child"
    shutil.copytree(src, dst)
    child = SR.PerennialState(dst, False, "seed", 1.0, n_agents=n_agents, shared_buffer=shared)
    after = sum(len(s.g.nodes) for s in child.stores)
    assert after == before, "child must inherit the parent's buffer, not reseed"
    assert child.visits == st.visits
