"""Tests for the meta-search driver: value model, selection policies, continuation rules."""
import json
import sys
from pathlib import Path

import pytest

MAIN = Path("/n/fs/vision-mix/sk7524/SkyRLTpu")
sys.path.insert(0, str(MAIN / "tpu/rq2/meta"))
sys.path.insert(0, str(MAIN / "tpu/distill_ablation/portfolio"))
import driver as D  # noqa: E402
from store import Store  # noqa: E402

ERDOS = D.PROBLEMS["erdos"]      # minimise, reference 0.3808753
FC46 = D.PROBLEMS["fc46"]        # maximise, reference 0.234594


# ------------------------------------------------------------------ value model
def test_normalisation_compresses_the_raw_scale_gap():
    """Raw deltas span orders of magnitude with depth; normalised ones must not, or early
    expansions would dominate the statistics and the tree would always backtrack to the root."""
    early_raw, late_raw = 1.0e-3, 1.0e-5                    # 100x apart
    early = D.gain(0.3830, 0.3830 - early_raw, ERDOS)
    late = D.gain(0.38094, 0.38094 - late_raw, ERDOS)
    assert early_raw / late_raw == pytest.approx(100)
    assert early / late < 5, f"normalised gap still {early / late:.0f}x -- not compressed"


def test_a_tiny_late_gain_beats_a_big_early_one_when_it_closes_more_headroom():
    """The property that matters: fraction of the remaining gap, not absolute movement."""
    late = D.gain(0.38094, 0.38094 - 0.8 * (0.38094 - ERDOS["reference"]), ERDOS)   # closes 80%
    early = D.gain(0.3830, 0.3830 - 0.2 * (0.3830 - ERDOS["reference"]), ERDOS)     # closes 20%
    assert late == pytest.approx(0.8) and early == pytest.approx(0.2)
    assert late > early                       # despite early's raw delta being ~8x larger


def test_gain_sign_follows_problem_direction():
    assert D.gain(0.3820, 0.3810, ERDOS) > 0      # minimise: lower child is a gain
    assert D.gain(0.3810, 0.3820, ERDOS) < 0
    assert D.gain(0.20, 0.21, FC46) > 0           # maximise: higher child is a gain
    assert D.gain(0.21, 0.20, FC46) < 0


def test_gain_handles_passing_the_reference():
    assert D.gain(0.38080, 0.38070, ERDOS) == 1.0     # already past ref, still improving
    assert D.gain(0.38080, 0.38090, ERDOS) == 0.0     # already past ref, got worse
    assert D.gain(None, 0.3, ERDOS) == 0.0


# ------------------------------------------------------------------ selection
def _meta(*specs):
    """specs: (id, best, {model: expansions})"""
    return {"nodes": [{"id": i, "path": f"/tmp/{i}", "parent": None, "model": "qwen",
                       "round": 1, "best": b, "gain": None, "expansions": e}
                      for i, b, e in specs], "rounds_done": 1}


def test_greedy_picks_the_best_node_for_both_models():
    m = _meta(("a", 0.3812, {"qwen": 0}), ("b", 0.3810, {"qwen": 9}), ("c", 0.3815, {"qwen": 0}))
    for model in D.MODELS:
        assert D.select_node(m, model, "greedy", ERDOS)["id"] == "b"


def test_greedy_respects_maximise():
    m = _meta(("a", 0.20, {"qwen": 0}), ("b", 0.23, {"qwen": 0}))
    assert D.select_node(m, "qwen", "greedy", FC46)["id"] == "b"


def test_mcts_explores_an_unexpanded_node_greedy_would_ignore():
    """The best node has been hammered by this model; an untouched node must become attractive."""
    m = _meta(("hot", 0.3810, {"qwen": 12, "gemma": 0}),
              ("cold", 0.3813, {"qwen": 0, "gemma": 0}))
    assert D.select_node(m, "qwen", "greedy", ERDOS)["id"] == "hot"
    assert D.select_node(m, "qwen", "mcts", ERDOS)["id"] == "cold"


def test_mcts_sends_each_model_where_that_model_has_been_productive():
    """The meta-learning claim: qwen did badly from x and well from y, gemma the reverse, so the
    policy must split them. Requires observed per-model gains -- with none, Q is model-independent
    and only the exploration term can differ."""
    def child(i, parent, model, g, best):
        return {"id": i, "path": "", "parent": parent, "model": model, "round": 2,
                "best": best, "gain": g, "expansions": {"qwen": 5, "gemma": 5}}
    m = {"rounds_done": 2, "nodes": [
        {"id": "x", "path": "", "parent": None, "model": "qwen", "round": 1, "best": 0.3810,
         "gain": None, "expansions": {"qwen": 1, "gemma": 1}},
        {"id": "y", "path": "", "parent": None, "model": "gemma", "round": 1, "best": 0.3811,
         "gain": None, "expansions": {"qwen": 1, "gemma": 1}},
        child("c1", "x", "qwen", 0.01, 0.3815),      # qwen stalled from x
        child("c2", "y", "qwen", 0.60, 0.3816),      # qwen thrived from y
        child("c3", "x", "gemma", 0.60, 0.3817),     # gemma thrived from x
        child("c4", "y", "gemma", 0.01, 0.3818),     # gemma stalled from y
    ]}
    assert D.q_value(m, m["nodes"][0], "qwen", ERDOS) == pytest.approx(0.01)
    assert D.q_value(m, m["nodes"][0], "gemma", ERDOS) == pytest.approx(0.60)
    picks = {mo: D.select_node(m, mo, "mcts", ERDOS)["id"] for mo in D.MODELS}
    assert picks == {"qwen": "y", "gemma": "x"}, picks


def test_selection_ignores_nodes_with_no_score():
    m = {"nodes": [{"id": "dead", "path": "", "parent": None, "model": "qwen", "round": 1,
                    "best": None, "gain": None, "expansions": {}},
                   {"id": "live", "path": "", "parent": None, "model": "qwen", "round": 1,
                    "best": 0.3812, "gain": None, "expansions": {}}], "rounds_done": 1}
    assert D.select_node(m, "qwen", "greedy", ERDOS)["id"] == "live"
    assert D.select_node({"nodes": [], "rounds_done": 0}, "qwen", "greedy", ERDOS) is None


# ------------------------------------------------------------------ continuation rules
@pytest.fixture
def parent_state(tmp_path):
    st = tmp_path / "parent" / "state"
    st.mkdir(parents=True)
    s = Store(False, st / "graph.json")
    root = s.add("seed", 0.3830, [], rnd=0)
    prev = root
    for i in range(40):                      # a lineage so rule (c) has a tail to discard
        prev = s.add(f"prog{i}", 0.3829 - i * 1e-5, [prev], rnd=1 + i // 4)
    s.save()
    (st / "visits.json").write_text(json.dumps({"0": {"0": 16, "3": 4}}))
    (st / "MEMORY.md").write_text("accumulated lessons")
    (st / "agent_best.json").write_text(json.dumps({"0": 0.38251}))
    (tmp_path / "parent" / "trace.jsonl").write_text('{"step":1}\n')
    return st


def test_rule_a_drops_visits_keeps_tree(parent_state, tmp_path):
    child = tmp_path / "child" / "state"
    D.apply_continuation(parent_state, child, "a", maximize=False)
    assert (child / "graph.json").exists() and not (child / "visits.json").exists()
    assert (child / "MEMORY.md").read_text() == "accumulated lessons"
    assert len(Store.load(child / "graph.json").g.nodes) == 41


def test_rule_b_keeps_visits(parent_state, tmp_path):
    child = tmp_path / "child" / "state"
    D.apply_continuation(parent_state, child, "b", maximize=False)
    assert json.loads((child / "visits.json").read_text())["0"]["0"] == 16
    assert len(Store.load(child / "graph.json").g.nodes) == 41


def test_rule_c_keeps_only_top_k_as_roots(parent_state, tmp_path):
    child = tmp_path / "child" / "state"
    D.apply_continuation(parent_state, child, "c", maximize=False)
    g = Store.load(child / "graph.json").g
    assert len(g.nodes) == D.TOPK_C
    assert len(g.edges) == 0, "top-k states must be roots so lineage exclusion is vacuous"
    assert not (child / "visits.json").exists()
    kept = sorted(g.nodes[n]["r"] for n in g.nodes)
    assert kept[0] == pytest.approx(0.38251)   # the 16 best (lowest) of the parent's pool


def test_no_rule_copies_the_trace(parent_state, tmp_path):
    """trace.jsonl must never cross: its absence is what makes the child run a fresh 10 rounds."""
    for rule in ("a", "b", "c"):
        child = tmp_path / f"c_{rule}" / "state"
        D.apply_continuation(parent_state, child, rule, maximize=False)
        assert not (child.parent / "trace.jsonl").exists()


def test_unknown_rule_is_rejected(parent_state, tmp_path):
    with pytest.raises(ValueError):
        D.apply_continuation(parent_state, tmp_path / "x" / "state", "z", maximize=False)


# ------------------------------------------------------------------ bookkeeping
def test_mem_health_reads_the_trace(tmp_path):
    d = tmp_path / "cell"
    d.mkdir()
    (d / "trace.jsonl").write_text(
        json.dumps({"step": 1, "mem_ok": [True]}) + "\n" +
        json.dumps({"step": 2, "mem_ok": [False]}) + "\n")
    assert D.mem_health(d) == 0.5
    assert D.mem_health(tmp_path / "missing") is None


def test_expansion_done_requires_a_full_trace(tmp_path):
    d = tmp_path / "cell"
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"best_fast_score": 0.381}))
    (d / "trace.jsonl").write_text("\n".join(json.dumps({"step": i}) for i in range(3)))
    assert D.expansion_done(d) is False
    (d / "trace.jsonl").write_text("\n".join(json.dumps({"step": i}) for i in range(10)))
    assert D.expansion_done(d) is True
    assert D.read_best(d) == pytest.approx(0.381)
