from __future__ import annotations

import os
import traceback
from typing import Any

from skyrl_agent.agents.react.react_agent import ReActAgent
from skyrl_agent.functional.utils import ContextWindowExceeded, StepException, StepResult
from skyrl_agent.tools.citation_prediction_v4 import mark_citation_protocol_violation
from skyrl_gym.envs.citation_prediction_v4.env import (
    CitationPredictionV4Env,
    CitationPredictionV4EnvConfig,
)


def _optional_int(name: str) -> int | None:
    value = os.getenv(name, "")
    return int(value) if value else None


def _optional_float(name: str) -> float | None:
    value = os.getenv(name, "")
    return float(value) if value else None


class CitationPredictionV4RawAgent(ReActAgent):
    """Run the citation-v4 text protocol without function-call conversion."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Keep MessageEncoder on the same plain chat template used by SFT/GPU RL.
        self.tool_params = []
        self.citation_env: CitationPredictionV4Env | None = None

    def _init_message(self, instruction: list[dict]) -> None:
        if not isinstance(instruction, list):
            raise ValueError("Instruction must be a list of messages.")
        self.history.initialize(instruction)

    def _make_env(self, instance: dict[str, Any]) -> CitationPredictionV4Env:
        reward_spec = instance.get("reward_spec", instance.get("reward_model"))
        config = CitationPredictionV4EnvConfig(
            search_url=os.environ["CITATION_RETRIEVER_URL"],
            topk=int(os.getenv("CITATION_TOP_K", "5")),
            timeout=int(os.getenv("CITATION_TIMEOUT", "30")),
            max_predictions_ratio=float(instance.get("max_predictions_ratio") or 2.0),
            max_question_segments=int(os.getenv("CITATION_MAX_QUESTION_SEGMENTS", "4")),
            max_searches_per_question=int(os.getenv("CITATION_MAX_SEARCHES_PER_QUESTION", "4")),
            rerank_retrieval_topk=_optional_int("CITATION_RERANK_RETRIEVAL_TOPK"),
            rerank_alpha=_optional_float("CITATION_RERANK_ALPHA"),
            rerank_final_topk=_optional_int("CITATION_RERANK_FINAL_TOPK"),
            rerank_norm=os.getenv("CITATION_RERANK_NORM", "rank"),
            citation_count_path=os.getenv("CITATION_COUNT_PATH", ""),
            citation_metric_beta=float(os.getenv("CITATION_METRIC_BETA", "1.0")),
            max_authors_in_result=_optional_int("CITATION_MAX_AUTHORS_IN_RESULT"),
            enable_question_coverage_reward=False,
            terminate_on_no_action=True,
        )
        return CitationPredictionV4Env(config, extras={"reward_spec": reward_spec})

    def _assistant_transcript(self) -> str:
        return "".join(
            str(message.get("content", ""))
            for message in self.history.messages
            if message.get("role") == "assistant"
        )

    async def step(self):
        self.step_count += 1
        print(f"[Citation Raw Step {self.step_count}] instance={self.instance_id} traj={self.trajectory_id}")
        result = None
        try:
            input_ids, sampling_params = self._prepare_llm_input()
            if self.response_token_len >= self.max_prompt_length:
                raise ContextWindowExceeded()

            response_str, meta_info = await self._generate_with_recording(
                input_ids=input_ids,
                sampling_params=sampling_params,
                request_id=self.agent_id,
            )
            stop_reason = meta_info["finish_reason"]
            print(f"[Citation Raw Step {self.step_count}] LLM response: {response_str}. Stop reason: {stop_reason}")
            self.history.add_assistant(response_str)
            if stop_reason == "length":
                raise ContextWindowExceeded()

            if self.citation_env is None:
                raise RuntimeError("citation environment was not initialized")
            env_result = self.citation_env.step(response_str)
            if env_result.get("metadata", {}).get("limit_violation"):
                reason = str(env_result["metadata"].get("limit_violation_reason", "protocol_violation"))
                mark_citation_protocol_violation(self.instance_id, self.trajectory_id, reason)

            if env_result["done"]:
                result = StepResult.finished("FINISH", self._assistant_transcript())
            else:
                for observation in env_result["observations"]:
                    self.history.add_user_guidance(str(observation["content"]))
                result = StepResult.continuing(response_str)
        except StepException as exc:
            result = exc.step_result
        except Exception as exc:
            print(f"[Citation Raw Step Error] {exc}")
            result = StepResult.finished(f"error: {exc}", None)
        return result.to_tuple()

    async def run(self, instruction: list[dict], instance: dict | None = None):
        if instance is None:
            raise ValueError("citation-v4 raw agent requires the dataset instance")
        self.instance = instance
        self.citation_env = self._make_env(instance)
        self.citation_env.init(instruction)
        self._init_message(instruction)
        result = None
        finish_reason = None
        try:
            while self.step_count < self.max_iterations:
                try:
                    done, finish_reason, result = await self.step()
                    if done:
                        break
                except Exception as exc:
                    finish_reason = f"error: {exc}"
                    print(f"[Citation Raw Run Error] {exc}")
                    print(traceback.format_exc())
                    break
            else:
                finish_reason = "max_iterations_reached"
            return finish_reason, result
        finally:
            tool_group = getattr(self.citation_env, "tool_group", None)
            session = getattr(tool_group, "session", None)
            if session is not None:
                session.close()
