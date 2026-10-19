"""SimpleTES on the RQ4 farms: caps, answer extraction, science tasks, grading retries, prompt fit, runner."""
import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpu.simpletes import tasks
from tpu.simpletes.backend import Caps, answer_code, as_evolve_block

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "tpu/swarm/ray_train/profiles"


def test_caps_match_the_native_completer():
    caps = Caps(context_window=22528, phase1_max_tokens=16384)
    assert caps.budget(4000, transition_tokens=12) == (12384, 22528 - 4000 - 50)
    assert caps.budget(16384) is None                      # no thinking headroom
    # A full thinking budget must leave room for the forced transition and one answer token.
    tight = Caps(context_window=1000, phase1_max_tokens=900)
    assert tight.budget(100, transition_tokens=60) is None   # 850 < 800 + 60 + 1
    assert tight.budget(100, transition_tokens=0) == (800, 850)


@pytest.mark.parametrize("family,content", [
    ("qwen", "<think>plan ```python\nwrong()\n```</think>\nAnswer:\n```python\ndef place():\n    return 1\n```\n"),
    ("gemma", "thought ```python\nwrong()\n```<channel|>Final:\n```python\ndef place():\n    return 1\n```"),
    ("muse", "```python\ndef place():\n    return 1\n```"),
])
def test_simpletes_grades_the_code_rq4_extracts(family, content):
    """START + RQ4 code + END, through SimpleTES's extractor and our marker strip, is the RQ4 code."""
    from tpu.simpletes.compat import import_simpletes
    import_simpletes()
    from simpletes.utils.code_extract import EvolveBlockContext, extract_code_detailed
    from ttt_discover.tinker_utils.dataset_builder import answer_only_code
    message = dict(content=content)
    rq4 = answer_only_code(message, family, keep_separators=False)
    context = EvolveBlockContext.from_program(tasks.init_program(tasks.get_task("circuit"), "x = 1"))
    program, reason = extract_code_detailed(as_evolve_block(answer_code(message, family)), context)
    assert reason == "evolve_block_merged"
    assert tasks.candidate_source(tasks.get_task("circuit"), program) == rq4


def test_markers_written_by_the_model_are_not_graded():
    code = "# EVOLVE-BLOCK-START\ndef place():\n    return 2\n# EVOLVE-BLOCK-END"
    from tpu.simpletes.compat import import_simpletes
    import_simpletes()
    from simpletes.utils.code_extract import EvolveBlockContext, extract_code_detailed
    context = EvolveBlockContext.from_program(tasks.init_program(tasks.get_task("qubit"), "x = 1"))
    program, _ = extract_code_detailed(as_evolve_block(code), context)
    assert tasks.candidate_source(tasks.get_task("qubit"), program) == "def place():\n    return 2\n"


def rq4_environment(monkeypatch, problem):
    from tpu.swarm.ray_train.commands import client_environment
    from tpu.swarm.ray_train.config import Config
    config = Config.load(PROFILES / f"rq4-notrain-local-qwen35-{problem}-s1-20261007.json")
    for key, value in client_environment(config, Path(config.root), "127.0.0.1").items():
        monkeypatch.setenv(key, value)
    return config


def test_qubit_starter_is_the_rq4_answer_format(monkeypatch):
    rq4_environment(monkeypatch, "qubit")
    task = tasks.get_task("qubit")
    starter = tasks.science_starter(task)
    module = ast.parse(starter)
    assert len(module.body) == 1 and module.body[0].targets[0].id == "RUST_CODE"
    assert "choose_best_swap" in module.body[0].value.value
    instruction = tasks.science_instruction(task)
    assert tasks.STARTER_MARKERS["routing"] not in instruction and "RUST_CODE" in instruction
    assert task.timeout() == 72000


def test_circuit_starter_is_the_rq4_python_starter(monkeypatch):
    rq4_environment(monkeypatch, "circuit")
    task = tasks.get_task("circuit")
    ast.parse(tasks.science_starter(task))
    assert tasks.STARTER_MARKERS["placement"] not in tasks.science_instruction(task)
    assert task.timeout() == 172800


class Verdict(SimpleNamespace):
    pass


class FlakyEnv:
    calls = 0

    def __init__(self, renderer, initial_state, sampler, config):
        self.config = config

    def _should_keep_code_separators(self):
        return False

    async def _safe_grade(self, source, step):
        FlakyEnv.calls += 1
        if FlakyEnv.calls == 1:
            error = RuntimeError("farm lost")
            error.abort_training_step = True
            raise error
        return Verdict(correctness=1, reward=0.4, raw_score=0.4, msg="", metrics=dict(host="h", cpu_model="x"),
                       result_construction=None)


def test_infrastructure_failures_are_retried_not_charged(tmp_path, monkeypatch):
    from tpu.simpletes.evaluator import RLEnvironmentEvaluator
    task = tasks.Task("fake", __name__, "FlakyEnv", "routing", "score", None, 100, 4, science=True,
                      eval_backend="local")
    evaluator = RLEnvironmentEvaluator(task, object(), log_dir=tmp_path, infra_retry_seconds=0)
    outcome = asyncio.run(evaluator.evaluate("# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n"))
    assert outcome.metrics["combined_score"] == 0.4 and outcome.metrics["host"] == "h"
    assert len((tmp_path / "infra_retries.jsonl").read_text().splitlines()) == 1


def test_prompts_drop_lowest_scored_inspirations_to_fit(tmp_path):
    from tpu.simpletes.compat import import_simpletes
    import_simpletes()
    from tpu.simpletes.driver import fit_prompts
    nodes = [SimpleNamespace(score=s, metrics={}) for s in (0.1, 0.9, 0.5)]
    seen = []

    def build_prompt(inspirations, *args, **kwargs):
        seen.append([n.score for n in inspirations])
        return "x" * (1000 * len(inspirations))
    engine = SimpleNamespace(generator=SimpleNamespace(build_prompt=build_prompt))
    backend = SimpleNamespace(prompt_ids=lambda text: list(text))
    fit_prompts(engine, backend, 2000, tmp_path / "fit.jsonl")
    assert engine.generator.build_prompt(nodes) == "x" * 2000
    assert seen[-1] == [0.9, 0.5]
    assert json.loads((tmp_path / "fit.jsonl").read_text())["kept"] == 2


def test_runner_puts_the_simpletes_tree_last_on_the_child_path(tmp_path, monkeypatch):
    from tpu.swarm.ray_train import local_notrain
    from tpu.swarm.ray_train.config import Config
    monkeypatch.setenv("RAY_AUTH_TOKEN", "t")
    config = Config.load(PROFILES / "scaf-erdos-stes-boot-muse-s1.json")
    args = SimpleNamespace(hf_home=tmp_path, package=tmp_path / "pkg", source=tmp_path / "src", num_epochs=None,
                           driver="simpletes", simpletes_root=ROOT)
    env = local_notrain.child_environment(config, Path(config.root), args, "127.0.0.1:1")
    assert env["PYTHONPATH"].split(":")[0] == str(tmp_path / "pkg")
    assert env["PYTHONPATH"].split(":")[-1] == str(ROOT)
    assert env["TTD_FROZEN_BASE_URL"].endswith(":26880")


def test_search_runs_inside_one_borrowing_phase():
    """The remote-only ingress holds completions until a phase is open (the smoke hung without it)."""
    from contextlib import asynccontextmanager
    from tpu.simpletes.local_driver import in_generation_phase
    events = []

    @asynccontextmanager
    async def phase(expected_n):
        events.append(("begin", expected_n))
        yield
        events.append(("end", expected_n))

    async def work():
        events.append(("work", None))
        return 7
    assert asyncio.run(in_generation_phase(work, 16, phase=phase)) == 7
    assert events == [("begin", 16), ("work", None), ("end", 16)]
