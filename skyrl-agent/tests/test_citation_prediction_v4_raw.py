import asyncio
import json

from skyrl_agent.agents.citation_prediction_v4_raw import CitationPredictionV4RawAgent
from skyrl_agent.config.configuration_utils import TrajectoryConfig
from skyrl_agent.agents.mapping import AGENT_GENERATOR_REGISTRY, AGENT_TRAJECTORY_REGISTRY
from skyrl_agent.tasks.citation_prediction_v4.utils import (
    CitationPredictionV4RawTask,
    CitationPredictionV4Task,
)
from skyrl_agent.tools.citation_prediction_v4 import mark_citation_protocol_violation


class _Tokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
        text = "".join(f"{item['role']}:{item['content']}\n" for item in messages)
        if add_generation_prompt:
            text += "assistant:"
        return [ord(char) % 251 + 1 for char in text] + [self.eos_token_id]


class _Backend:
    async def async_generate_ids(self, **_kwargs):
        response = "<think>done</think><citation>2401.00001</citation><done></done>"
        return response, {"output_tokens": [11, 12, 13], "logprobs": [-0.1] * 3, "finish_reason": "stop"}


class _DoneEnv:
    tool_group = None

    def init(self, _instruction):
        return None

    def step(self, _response):
        return {"observations": [], "reward": 1.0, "done": True, "metadata": {}}


class _SearchThenDoneBackend:
    def __init__(self):
        self.calls = 0

    async def async_generate_ids(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            response = "<question>topic</question><search>query</search>"
        else:
            response = "<citation>2401.00001</citation><done></done>"
        return response, {"output_tokens": [11, 12, 13], "logprobs": [-0.1] * 3, "finish_reason": "stop"}


class _LargeObservationEnv:
    tool_group = None

    def init(self, _instruction):
        return None

    def step(self, response):
        if "<done>" in response:
            return {"observations": [], "reward": 1.0, "done": True, "metadata": {}}
        return {
            "observations": [{"role": "user", "content": "x" * 1800}],
            "reward": 0.0,
            "done": False,
            "metadata": {},
        }


def _instance():
    prompt = [
        {"role": "system", "content": "Use <search> tags."},
        {"role": "user", "content": "Find the paper."},
    ]
    return {
        "raw_prompt": json.dumps(prompt),
        "reward_spec": json.dumps({"ground_truth": {"targets": ["2401.00001"]}}),
        "max_predictions_ratio": 2.0,
    }


def test_raw_task_preserves_original_prompt_without_function_bridge():
    instruction = CitationPredictionV4RawTask.get_instruction(_instance())
    assert instruction[0]["content"] == "Use <search> tags."
    assert "citation_search" not in instruction[0]["content"]


def test_raw_agent_is_registered_with_react_runner():
    path = "skyrl_agent.agents.citation_prediction_v4_raw.CitationPredictionV4RawAgent"
    assert AGENT_GENERATOR_REGISTRY[path] == "skyrl_agent.agents.base.AgentRunner"
    assert AGENT_TRAJECTORY_REGISTRY[path] == "skyrl_agent.agents.react.ReActTrajectory"


def test_raw_agent_returns_plain_citation_transcript(monkeypatch):
    config = TrajectoryConfig(
        instance_id=0,
        trajectory_id=0,
        max_prompt_length=4096,
        sampling_params={"temperature": 0.6, "top_p": 1.0, "max_tokens": 4096},
        qwen3_enable_thinking=False,
        qwen3_acc_thinking=False,
        max_iterations=2,
        tools=[],
    )
    agent = CitationPredictionV4RawAgent(config, _Backend(), _Tokenizer())
    monkeypatch.setattr(agent, "_make_env", lambda _row: _DoneEnv())
    instruction = CitationPredictionV4RawTask.get_instruction(_instance())
    reason, result = asyncio.run(agent.run(instruction, _instance()))
    assert reason == "FINISH"
    assert result.endswith("<citation>2401.00001</citation><done></done>")
    assert agent.tool_params == []


def test_protocol_violation_forces_zero_reward():
    mark_citation_protocol_violation("example", 3, "malformed_search_tag")
    reward = asyncio.run(
        CitationPredictionV4Task.evaluate_result(
            "<citation>2401.00001</citation><done></done>",
            _instance(),
            "citation_prediction_v4",
            "example",
            3,
        )
    )
    assert reward == 0.0


def test_context_guard_replaces_large_observation_with_final_turn(monkeypatch):
    config = TrajectoryConfig(
        instance_id=0,
        trajectory_id=0,
        max_prompt_length=2048,
        sampling_params={"temperature": 0.6, "top_p": 1.0, "top_k": 20, "max_tokens": 128},
        qwen3_enable_thinking=False,
        qwen3_acc_thinking=False,
        max_iterations=3,
        tools=[],
    )
    agent = CitationPredictionV4RawAgent(config, _SearchThenDoneBackend(), _Tokenizer())
    monkeypatch.setattr(agent, "_make_env", lambda _row: _LargeObservationEnv())
    monkeypatch.setenv("CITATION_FORCE_FINAL_TURN_ON_MAX_INPUT_LENGTH", "1")
    monkeypatch.setenv("CITATION_FINAL_TURN_CONTEXT_RESERVE_TOKENS", "512")

    reason, result = asyncio.run(agent.run(CitationPredictionV4RawTask.get_instruction(_instance()), _instance()))

    assert reason == "FINISH"
    assert result.endswith("<citation>2401.00001</citation><done></done>")
    assert agent.context_final_turn_forced is True
    assert "Maximum context budget reached" in agent.get_messages()[-2]["content"]
    assert "x" * 1800 not in agent.get_messages()[-2]["content"]
