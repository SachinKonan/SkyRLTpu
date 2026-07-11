from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from skyrl_agent.tools.base import BaseTool, register_tool

try:
    from skyrl_gym.envs.citation_prediction_v2.utils import normalize_arxiv_id
    from skyrl_gym.envs.citation_prediction_v4.env import (
        _extract_seen_arxiv_ids,
        _load_citation_counts,
        _rerank_retrievals,
    )
    from skyrl_gym.tools.search import call_search_api
except Exception:  # pragma: no cover - import failure is surfaced at runtime in call().
    normalize_arxiv_id = None
    _extract_seen_arxiv_ids = None
    _load_citation_counts = None
    _rerank_retrievals = None
    call_search_api = None


@dataclass
class CitationSearchState:
    total_searches: int = 0
    current_question_index: int = 0
    searches_in_current_question: int = 0
    question_segments: int = 0
    limit_violation: bool = False
    limit_violation_reason: str = ""
    retrieved_ids: set[str] = field(default_factory=set)


_STATE_LOCK = threading.Lock()
_TRAJ_STATES: dict[tuple[str, int], CitationSearchState] = {}


def _state_key(agent: Any | None, trajectory_id: int | None) -> tuple[str, int]:
    instance_id = getattr(agent, "instance_id", "unknown")
    return str(instance_id), int(trajectory_id or 0)


def reset_citation_search_state(instance_id: Any, trajectory_id: int) -> None:
    with _STATE_LOCK:
        _TRAJ_STATES.pop((str(instance_id), int(trajectory_id)), None)


def get_citation_search_state(instance_id: Any, trajectory_id: int) -> CitationSearchState | None:
    with _STATE_LOCK:
        return _TRAJ_STATES.get((str(instance_id), int(trajectory_id)))


def mark_citation_protocol_violation(instance_id: Any, trajectory_id: int, reason: str) -> None:
    key = (str(instance_id), int(trajectory_id))
    with _STATE_LOCK:
        state = _TRAJ_STATES.get(key)
        if state is None:
            state = CitationSearchState()
            _TRAJ_STATES[key] = state
        state.limit_violation = True
        state.limit_violation_reason = reason


def _get_or_create_state(agent: Any | None, trajectory_id: int | None) -> CitationSearchState:
    key = _state_key(agent, trajectory_id)
    with _STATE_LOCK:
        state = _TRAJ_STATES.get(key)
        if state is None:
            state = CitationSearchState()
            _TRAJ_STATES[key] = state
        return state


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return int(raw)


def _env_float(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return float(raw)


def _read_ready_file_url(path: str) -> str:
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("RETRIEVER_URL="):
                    return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        return ""
    return ""


def _truncate_author_list_in_contents(contents: str, max_authors: int | None) -> str:
    if not max_authors or max_authors <= 0:
        return contents
    match = re.search(r"(Authors:)(.*?)(\n\nAbstract:)", contents, flags=re.DOTALL)
    if not match:
        return contents
    parts = [part.strip() for part in match.group(2).strip().split(",") if part.strip()]
    if len(parts) <= max_authors:
        return contents
    truncated = ", ".join(parts[:max_authors]) + ", et al."
    return contents[: match.start(2)] + truncated + contents[match.end(2) :]


def _passages_to_string(retrieval_result: list[dict], max_authors: int | None = None) -> str:
    formatted = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"].strip()
        content = _truncate_author_list_in_contents(content, max_authors)
        formatted += f"Doc {idx + 1}: {content}\n"
    return formatted


def _count_question_segments(agent: Any | None) -> int:
    if agent is None or not hasattr(agent, "history"):
        return 0
    try:
        text = "".join(msg.get("content", "") for msg in agent.history.messages if msg.get("role") == "assistant")
    except Exception:
        return 0
    return len(re.findall(r"<question>.*?</question>", text, flags=re.DOTALL))


@register_tool("citation_search")
class CitationSearchTool(BaseTool):
    name = "citation_search"
    description = (
        "Search the local citation-prediction retriever for papers relevant to one sub-question. "
        "The query should be a concise literature-review search query."
    )
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query for the paper retriever."}},
        "required": ["query"],
    }

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.ready_file = os.getenv("CITATION_RETRIEVER_READY_FILE", "")
        self.search_url = os.getenv("CITATION_RETRIEVER_URL") or os.getenv("LOCAL_SEARCH_URL")
        if not self.search_url and self.ready_file:
            self.search_url = _read_ready_file_url(self.ready_file)
        if not self.search_url:
            raise ValueError("CITATION_RETRIEVER_URL or LOCAL_SEARCH_URL must be set for citation_search.")
        self.topk = _env_int("CITATION_TOP_K", _env_int("LOCAL_SEARCH_TOP_K", 10))
        self.timeout = _env_int("CITATION_TIMEOUT", 30)
        self.max_question_segments = _env_int("CITATION_MAX_QUESTION_SEGMENTS", 4)
        self.max_searches_per_question = _env_int("CITATION_MAX_SEARCHES_PER_QUESTION", 4)
        self.max_searches_total = _env_int(
            "CITATION_MAX_SEARCHES_TOTAL", self.max_question_segments * self.max_searches_per_question
        )
        self.max_authors_in_result = _env_int("CITATION_MAX_AUTHORS_IN_RESULT", 0) or None

        self.rerank_retrieval_topk = _env_int("CITATION_RERANK_RETRIEVAL_TOPK", 0) or None
        self.rerank_alpha = _env_float("CITATION_RERANK_ALPHA")
        self.rerank_final_topk = _env_int("CITATION_RERANK_FINAL_TOPK", 0) or self.topk
        self.rerank_norm = os.getenv("CITATION_RERANK_NORM", "rank")
        self.citation_count_path = os.getenv("CITATION_COUNT_PATH", "")
        self.citation_metric_beta = _env_float("CITATION_METRIC_BETA", 1.0) or 1.0
        self.use_rerank = (
            self.rerank_retrieval_topk is not None
            and self.rerank_retrieval_topk > self.rerank_final_topk
            and self.rerank_alpha is not None
        )

    def _protocol_violation(self, state: CitationSearchState, reason: str) -> str:
        state.limit_violation = True
        state.limit_violation_reason = reason
        return "\n<information>" + json.dumps({"result": f"Protocol violation: {reason}. Reward will be 0."}) + "</information>\n"

    def call(self, params: dict, **kwargs) -> str:
        if call_search_api is None:
            return "\n<information>" + json.dumps({"result": "Search tool import failed; reward will be 0."}) + "</information>\n"

        try:
            params = self._verify_json_format_args(params)
        except ValueError as exc:
            state = _get_or_create_state(kwargs.get("agent"), kwargs.get("trajectory_id"))
            return self._protocol_violation(state, f"invalid citation_search arguments: {exc}")

        query = str(params.get("query", "")).strip()
        state = _get_or_create_state(kwargs.get("agent"), kwargs.get("trajectory_id"))

        question_segments = _count_question_segments(kwargs.get("agent"))
        if question_segments > state.question_segments:
            state.current_question_index += question_segments - state.question_segments
            state.question_segments = question_segments
            state.searches_in_current_question = 0

        if state.question_segments > self.max_question_segments:
            return self._protocol_violation(
                state, f"too many question segments ({state.question_segments}>{self.max_question_segments})"
            )

        state.total_searches += 1
        state.searches_in_current_question += 1
        if state.total_searches > self.max_searches_total:
            return self._protocol_violation(state, f"too many total searches ({state.total_searches}>{self.max_searches_total})")
        if state.searches_in_current_question > self.max_searches_per_question:
            return self._protocol_violation(
                state,
                f"too many searches for one question ({state.searches_in_current_question}>{self.max_searches_per_question})",
            )
        if not query:
            return self._protocol_violation(state, "empty citation_search query")

        if self.ready_file:
            refreshed_url = _read_ready_file_url(self.ready_file)
            if refreshed_url:
                self.search_url = refreshed_url

        request_topk = self.rerank_retrieval_topk if self.use_rerank else self.topk
        api_response, error_msg = call_search_api(
            retrieval_service_url=self.search_url,
            query=query,
            topk=int(request_topk),
            return_scores=True,
            timeout=self.timeout,
            log_requests=False,
        )

        if error_msg:
            result_text = json.dumps({"result": f"Search error: {error_msg}"})
            return "\n<information>" + result_text + "</information>\n"

        retrievals = []
        if api_response:
            raw = api_response.get("result", [])
            if raw:
                retrievals = raw[0]

        if self.use_rerank and retrievals:
            citation_counts = _load_citation_counts(self.citation_count_path, self.citation_metric_beta)
            retrievals = _rerank_retrievals(
                retrievals,
                final_topk=self.rerank_final_topk,
                citation_counts=citation_counts,
                semantic_weight=float(self.rerank_alpha),
                citation_weight=1.0 - float(self.rerank_alpha),
                norm=self.rerank_norm,
            )

        formatted = _passages_to_string(retrievals, self.max_authors_in_result) if retrievals else "No search results found."
        if _extract_seen_arxiv_ids is not None:
            state.retrieved_ids.update(_extract_seen_arxiv_ids(formatted))
        return "\n<information>" + json.dumps({"result": formatted}) + "</information>\n"

    def get_system_prompt_prefix(self) -> str:
        return (
            "Citation-search tool instructions:\n"
            "- Use `citation_search` whenever you need to retrieve papers for a sub-question. "
            "This replaces the raw `<search>...</search>` action from older traces.\n"
            "- The tool returns observations inside `<information>{...}</information>` with paper metadata and arXiv ids.\n"
            f"- You may use at most {self.max_question_segments} `<question>` segments, at most "
            f"{self.max_searches_per_question} searches per question, and at most {self.max_searches_total} searches total.\n"
            "- Finish by calling the `finish` tool. The answer passed to `finish` must contain only your final "
            "`<citation>arxiv_id</citation>` tags followed by `<done></done>`."
        )
