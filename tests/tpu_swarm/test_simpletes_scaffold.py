"""SimpleTES arm of the scaffold experiment: budget, prompt, roots, warm start, eager commit."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpu.simpletes import tasks
from tpu.simpletes.backend import Caps


def test_fixed_budget_ignores_the_prompt_and_rejects_overflow():
    caps = Caps(context_window=32768, phase1_max_tokens=16384, fixed_thinking=12000, fixed_total=17000)
    assert caps.budget(3000) == caps.budget(15000) == (12000, 17000)
    assert caps.budget(32768 - 17000 - 50) == (12000, 17000)
    assert caps.budget(32768 - 17000 - 49) is None
    assert caps.max_prompt_tokens(4096) == 32768 - 17000 - 50 and caps.thinking == 12000


@pytest.fixture
def erdos(monkeypatch, tmp_path):
    monkeypatch.setenv("TTD_ERDOS_PROMPT", "v2")
    from tpu.simpletes.compat import import_simpletes
    import_simpletes()
    from tpu.simpletes.evaluator import RLEnvironmentEvaluator
    task = tasks.get_task("erdos")
    root = task.env_type().create_initial_state(task.problem_type, seed=0)
    return task, root, RLEnvironmentEvaluator(task, root, log_dir=tmp_path / "grading")


def node(score, code="x = 1", reflection="", parents=("p",), **metrics):
    from simpletes.node import Node, Status
    import uuid
    return Node(id=uuid.uuid4().hex, code=f"{tasks.EVOLVE_START}\n{code}\n{tasks.EVOLVE_END}\n",
                parent_ids=list(parents), status=Status.DONE, score=score, reflection=reflection,
                metrics=dict(combined_score=score, c5_bound=-score, validity=1.0, **metrics))


def arm_for(erdos, tmp_path, spec=None):
    from tpu.simpletes.scaffold import Arm
    task, root, evaluator = erdos
    engine = SimpleNamespace(_shared_construction_dir=str(tmp_path))
    return Arm(engine, task, evaluator, root, spec or {}, tmp_path, seed=1)


def test_prompt_is_the_shared_task_text_then_simpletes(erdos, tmp_path):
    arm = arm_for(erdos, tmp_path)
    question = erdos[2]._environment(erdos[1]).get_question()
    assert question.startswith(arm.text) and "beats it: C₅ < 0.3808586." in arm.text
    assert "You are iteratively optimizing" not in arm.text
    best, other = node(-0.3809, "best()", "Approach: a\nInsight: b"), node(-0.381, "other()")
    (tmp_path / f"{best.id}.json").write_text("[]")
    prompt = arm.build_prompt([other, best], {"TimeoutError: too slow": 0.125})
    assert prompt.startswith(f"Task: {arm.text}\n\nGeneration instruction (must follow exactly):\n1) Return")
    first, second = prompt.index("--- Inspiration 1 ---"), prompt.index("--- Inspiration 2 ---")
    assert prompt.index("best()") < second < prompt.index("other()") and first < prompt.index("Approach: a")
    assert "EVOLVE-BLOCK" not in prompt and "validity" not in prompt and "c5_bound: 0.380900" in prompt
    assert "\nYou may want to start your search from the best inspiration's construction" in prompt
    assert prompt.endswith("  - TimeoutError: too slow: 12.5%\n\nAnalyze the metrics above and create an improved "
                           "program with higher combined_score. Try diverse approaches to solve the problem. "
                           "Think outside the box. ")


def test_empty_root_prompt_shows_no_inspiration(erdos, tmp_path):
    from tpu.simpletes.scaffold import PLACEHOLDER
    arm = arm_for(erdos, tmp_path, dict(num_chains=2, start="none"))
    roots = arm.root_nodes()
    assert len(roots) == 2 and all(n.code == PLACEHOLDER and c for n, c in roots)
    assert roots[0][0].score != roots[1][0].score              # one random construction per chain
    (tmp_path / f"{roots[0][0].id}.json").write_text("[]")
    prompt = arm.build_prompt([roots[0][0]])
    assert "Inspiration" not in prompt and "no program yet" not in prompt
    assert "start your search from the current construction" in prompt


def test_bootstrap_roots_are_the_best_draft_of_each_group(tmp_path):
    from tpu.simpletes.scaffold import bootstrap_roots
    (tmp_path / "complete.json").write_text("{}")
    for g in range(4):
        group = tmp_path / "drafts" / f"group-{g:03d}"
        group.mkdir(parents=True)
        for i in range(4):
            value = -(1 + g * 4 + i) / 100
            state = dict(value=value, code=f"```python\nd{g}{i}()\n```", construction=[value])
            invalid = (g, i) == (0, 3)
            (group / f"grade-{i:03d}.json").write_text(json.dumps(
                dict(correctness=0.0 if invalid else 1.0, state=None if invalid else state, code="")))
    roots = bootstrap_roots(tmp_path, 2)
    # Chain 0 = drafts 0..7 (best -0.01), chain 1 = drafts 8..15 (best -0.09).
    assert [r[0] for r in roots] == [-0.01, -0.09] and "d00()" in roots[0][1]
    with pytest.raises(RuntimeError):
        bootstrap_roots(tmp_path, 3)


def test_eager_commit_finalizes_once_at_the_fraction():
    from tpu.simpletes.compat import import_simpletes
    import_simpletes()
    from simpletes.policies.rpucg import RpucgPolicy
    from tpu.simpletes.scaffold import eager
    policy = RpucgPolicy(num_chains=1, max_generations=64, k=32)
    eager(policy, 0.95)
    policy.register_batch(0, 0, ["root"], 32)
    pending = [policy.on_child_done(SimpleNamespace(gen_id=0, id=f"c{i}", metrics={}), []) for i in range(31)]
    assert [p is not None for p in pending] == [False] * 30 + [True]
    assert policy.on_child_done(SimpleNamespace(gen_id=0, id="late", metrics={}), []) is None
    assert policy.on_generation_failed(0) is None


def test_erdos_score_is_our_state_value(erdos):
    _, _, evaluator = erdos
    verdict = SimpleNamespace(correctness=1, reward=1 / 0.381, raw_score=0.381, msg="", metrics={},
                              result_construction=[0.5])
    metrics = evaluator.metrics(verdict)
    assert metrics["combined_score"] == -0.381 and metrics["c5_bound"] == 0.381


def test_specs_match_the_agreed_design():
    specs = Path(__file__).resolve().parents[2] / "tpu/simpletes/specs"
    for path in specs.glob("scaf-erdos-stes-*.json"):
        spec = json.loads(path.read_text())
        assert (spec["num_chains"], spec["k_candidates"], spec["max_generations"]) == (16, 32, 15 * 512)
        assert (spec["num_inspirations"], spec["eager"], spec["prompt"]) == (3, 0.95, "scaffold")
        assert ("bootstrap_from" in spec) == (spec["start"] == "boot")


def test_runner_skips_farms_shorter_than_the_run_context():
    from tpu.swarm.ray_train.local_notrain import candidates
    config = SimpleNamespace(model="m", run_id="r", client_context_window=32768)

    def farm(url, length):
        return dict(ok=True, tunnel_ok=True, models=["m"], state="unleased", url=url,
                    capabilities=dict(contract=dict(max_model_length=length)))
    assert candidates(dict(farms=[farm("old", 22528), farm("new", 32768)]), config, 4) == ["new"]


def test_v4_64_od_pool_takes_a_six_host_inference_farm_only():
    from tpu.swarm.ray_train.config import Config
    profile = Path(__file__).resolve().parents[2] / "tpu/swarm/ray_train/profiles/inference-farm-scaf-qwen38-central2-v464od-001.json"
    doc = json.loads(profile.read_text())
    assert Config.from_dict(doc).hosts == 6
    with pytest.raises(ValueError, match="host count"):
        Config.from_dict(dict(doc, hosts=5, inference_only_ranks=list(range(5))))


def test_v4_64_od_farm_job_matches_the_on_demand_pool(tmp_path):
    import yaml
    from tpu.swarm.ray_train.build import build
    profile = Path(__file__).resolve().parents[2] / "tpu/swarm/ray_train/profiles/inference-farm-scaf-qwen38-central2-v464od-001.json"
    _, _, path = build(profile, tmp_path)
    resources = yaml.safe_load(path.read_text())["resources"]
    assert (resources["use_spot"], resources["accelerator_args"]["gcp_queued_resource"], resources["disk_size"]) == (False, False, 300)


def test_previous_checkpoint_survives_until_the_next_is_written(tmp_path):
    from tpu.simpletes.driver import keep_previous_checkpoint
    old = tmp_path / "db_state_old"
    old.mkdir()
    seen = []

    class Manager:
        _last_checkpoint_path = str(old)

        async def snapshot(self):
            # Upstream would rmtree _last_checkpoint_path here; it must already be cleared.
            seen.append((self._last_checkpoint_path, old.exists()))
            (tmp_path / "db_state_new").mkdir()
            self._last_checkpoint_path = str(tmp_path / "db_state_new")

    manager = Manager()
    manager.snapshot = Manager.snapshot.__get__(manager)
    keep_previous_checkpoint(manager)
    asyncio.run(manager.snapshot())
    assert seen == [(None, True)] and not old.exists() and (tmp_path / "db_state_new").exists()


def test_resumed_run_extends_to_the_larger_budget():
    from tpu.simpletes.compat import import_simpletes
    import_simpletes()
    from simpletes.policies.rpucg import RpucgPolicy
    from tpu.simpletes.driver import extend_budget
    policy = RpucgPolicy(num_chains=2, max_generations=128, k=32)   # 2 batches per chain
    for chain in (0, 1):
        policy.chain_prompt_count[chain] = 2
        policy._ready_chains.remove(chain)
    assert extend_budget(policy, 256)                                # 25-step style extension: 4 per chain
    assert policy.prompt_budget == {0: 4, 1: 4} and sorted(policy._ready_chains) == [0, 1]
    assert not extend_budget(policy, 256)
