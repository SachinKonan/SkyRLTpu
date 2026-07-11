from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from skyrl_agent.tasks.base import BaseTask

try:
    from skyrl_gym.envs.citation_prediction_v2.utils import normalize_arxiv_id
except Exception:  # pragma: no cover
    normalize_arxiv_id = None

from skyrl_agent.tools.citation_prediction_v4 import (
    get_citation_search_state,
    reset_citation_search_state,
)


def _as_dict(instance: Any) -> dict[str, Any]:
    if hasattr(instance, "to_dict"):
        return dict(instance.to_dict())
    if isinstance(instance, dict):
        return instance
    return dict(instance)


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
        if len(value) == 1:
            value = value[0]
    if isinstance(value, str):
        value = value.strip()
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
    return value


def _normalize_id(value: Any) -> str:
    text = str(value or "").strip()
    if normalize_arxiv_id is not None:
        return normalize_arxiv_id(text)
    match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", text)
    return match.group(1) if match else ""


def extract_citation_list(text: str) -> list[str]:
    ids: list[str] = []
    for match in re.finditer(r"<citation>(.*?)</citation>", text or "", re.DOTALL):
        for part in match.group(1).split(","):
            paper_id = _normalize_id(part)
            if paper_id:
                ids.append(paper_id)
    return ids


def _ground_truth_targets(instance: dict[str, Any]) -> set[str]:
    reward_spec = _parse_json_field(instance.get("reward_spec"))
    if reward_spec is None and "reward_model" in instance:
        reward_spec = instance.get("reward_model")
    if not isinstance(reward_spec, dict):
        return set()
    targets = reward_spec.get("ground_truth", {}).get("targets", [])
    return {paper_id for paper_id in (_normalize_id(item) for item in targets) if paper_id}


class CitationPredictionV4Task(BaseTask):
    @classmethod
    async def initialize_runtime(cls):
        pass

    @classmethod
    def get_instruction(cls, instance: Any) -> list[dict[str, str]]:
        row = _as_dict(instance)
        prompt = _parse_json_field(row.get("raw_prompt") if "raw_prompt" in row else row.get("prompt"))
        assert isinstance(prompt, list), f"Prompt must be a list of messages after JSON parsing, got {type(prompt)}"

        processed = []
        for message in prompt:
            assert isinstance(message, dict), f"Prompt message must be a dict, got {type(message)}"
            processed.append({"role": str(message["role"]), "content": str(message["content"])})

        bridge = (
            "\n\nFor this Tinker/ReAct run, retrieval is exposed as a function tool named `citation_search`. "
            "Use that tool when you would otherwise emit a `<search>...</search>` action. "
            "Keep the same literature-review protocol otherwise: reason with `<think>`, state sub-questions with "
            "`<question>`, read `<information>` observations, and call `finish` with final "
            "`<citation>arxiv_id</citation>` tags followed by `<done></done>`."
        )
        for message in processed:
            if message["role"] == "system":
                message["content"] += bridge
                break
        else:
            processed.insert(0, {"role": "system", "content": bridge.strip()})
        return processed

    @classmethod
    def complete_runtime(cls):
        pass

    @classmethod
    async def evaluate_result(
        cls,
        result: Any,
        instance: Any,
        data_source: str,
        instance_id: int | str,
        trajectory_id: int,
    ) -> float:
        state = get_citation_search_state(instance_id, trajectory_id)
        try:
            if state is not None and state.limit_violation:
                return 0.0

            if not isinstance(result, str) or not result.strip():
                return 0.0
            if "<done>" not in result or "</done>" not in result:
                return 0.0

            row = _as_dict(instance)
            gold = _ground_truth_targets(row)
            if not gold:
                return 0.0

            predicted_list = extract_citation_list(result)
            predicted = set(predicted_list)
            citation_budget = int(len(gold) * float(row.get("max_predictions_ratio", 2.0)))
            if len(predicted_list) > citation_budget:
                return 0.0
            return float(len(predicted & gold) / max(1, len(gold)))
        finally:
            reset_citation_search_state(instance_id, trajectory_id)


class CitationPredictionV4RawTask(CitationPredictionV4Task):
    """Citation-v4 task with the original SFT/GPU text protocol unchanged."""

    @classmethod
    def get_instruction(cls, instance: Any) -> list[dict[str, str]]:
        row = _as_dict(instance)
        prompt = _parse_json_field(row.get("raw_prompt") if "raw_prompt" in row else row.get("prompt"))
        assert isinstance(prompt, list), f"Prompt must be a list of messages after JSON parsing, got {type(prompt)}"
        return [{"role": str(message["role"]), "content": str(message["content"])} for message in prompt]
