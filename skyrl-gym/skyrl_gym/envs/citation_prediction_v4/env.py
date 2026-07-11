from __future__ import annotations

import json
import logging
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.citation_prediction_v2.utils import normalize_arxiv_id
from skyrl_gym.tools.search import SearchToolGroup, call_search_api

logger = logging.getLogger(__name__)

_CITATION_COUNTS_CACHE: dict[tuple[str, float], dict[str, float]] = {}
_CITATION_COUNTS_LOCK = Lock()
_RERANK_STATS_LOCK = Lock()
_RERANK_STATS = {"calls": 0, "candidate_count": 0, "missing_citation_count": 0}


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _extract_arxiv_id_from_doc(doc: dict) -> str:
    for key in ("id", "arxiv_id", "paper_id"):
        value = normalize_arxiv_id(str(doc.get(key, "")))
        if value:
            return value
    contents = str(doc.get("contents", ""))
    match = re.search(r"\[arxiv:([^\]]+)\]", contents, flags=re.IGNORECASE)
    if match:
        return normalize_arxiv_id(match.group(1))
    return ""


def _extract_seen_arxiv_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for raw in re.findall(r"\[arxiv:([^\]]+)\]", text or "", flags=re.IGNORECASE):
        normalized = normalize_arxiv_id(raw)
        if normalized:
            ids.add(normalized)
    for raw in re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", text or ""):
        normalized = normalize_arxiv_id(raw)
        if normalized:
            ids.add(normalized)
    return ids


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b)) if (a or b) else 0.0


def _pairwise_jaccard(candidate_sets: list[set[str]]) -> float:
    pairs = 0
    total = 0.0
    for i in range(len(candidate_sets)):
        for j in range(i + 1, len(candidate_sets)):
            total += _jaccard(candidate_sets[i], candidate_sets[j])
            pairs += 1
    return total / pairs if pairs else 0.0


def _load_citation_counts(path: str, citation_metric_beta: float) -> dict[str, float]:
    if not path:
        return {}

    cache_key = (path, citation_metric_beta)
    with _CITATION_COUNTS_LOCK:
        cached = _CITATION_COUNTS_CACHE.get(cache_key)
        if cached is not None:
            return cached

    sidecar = Path(path)
    if not sidecar.exists():
        raise FileNotFoundError(f"citation count sidecar not found: {path}")

    id_keys = ("paper_id", "arxiv_id", "id", "arxiv")
    count_keys = ("citation_count", "citationCount", "num_citations", "n_citations", "citations", "count")
    influential_count_keys = (
        "influential_citation_count",
        "influentialCitationCount",
        "influentialcitationcount",
        "num_influential_citations",
        "influential_citations",
    )

    def add_row(row: dict, counts: dict[str, float]) -> None:
        paper_id = ""
        for key in id_keys:
            paper_id = normalize_arxiv_id(str(row.get(key, "")))
            if paper_id:
                break
        if not paper_id:
            return

        citation_count = None
        for key in count_keys:
            citation_count = _maybe_float(row.get(key))
            if citation_count is not None:
                break

        influential_count = None
        for key in influential_count_keys:
            influential_count = _maybe_float(row.get(key))
            if influential_count is not None:
                break

        if citation_count is None and influential_count is None:
            return
        citation_count = max(0.0, citation_count or 0.0)
        influential_count = max(0.0, influential_count if influential_count is not None else citation_count)
        counts[paper_id] = max(0.0, citation_metric_beta * citation_count + (1.0 - citation_metric_beta) * influential_count)

    counts: dict[str, float] = {}
    suffix = sidecar.suffix.lower()
    if suffix == ".csv":
        with sidecar.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                add_row(row, counts)
    elif suffix == ".jsonl":
        with sidecar.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    add_row(json.loads(line), counts)
    elif suffix == ".json":
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            if all(not isinstance(value, dict) for value in raw.values()):
                for key, value in raw.items():
                    paper_id = normalize_arxiv_id(str(key))
                    count = _maybe_float(value)
                    if paper_id and count is not None:
                        counts[paper_id] = max(0.0, count)
            else:
                for value in raw.values():
                    if isinstance(value, dict):
                        add_row(value, counts)
        elif isinstance(raw, list):
            for row in raw:
                if isinstance(row, dict):
                    add_row(row, counts)
    else:
        raise ValueError(f"unsupported citation count sidecar format: {path}")

    logger.info("Loaded %s citation metric rows from %s using citation_metric_beta=%s", len(counts), path, citation_metric_beta)
    with _CITATION_COUNTS_LOCK:
        _CITATION_COUNTS_CACHE[cache_key] = counts
    return counts


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0] * len(values)
    return [(value - lo) / (hi - lo) for value in values]


def _rank_percentile(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.0]

    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        percentile = ((i + j - 1) / 2.0) / (len(values) - 1)
        for idx, _value in indexed[i:j]:
            ranks[idx] = percentile
        i = j
    return ranks


def _rerank_retrievals(
    retrievals: list[dict],
    final_topk: int,
    citation_counts: dict[str, float],
    semantic_weight: float,
    citation_weight: float,
    norm: str,
) -> list[dict]:
    if final_topk <= 0:
        return retrievals
    if len(retrievals) <= final_topk:
        return retrievals[:final_topk]

    semantic_scores = [_maybe_float(item.get("score")) or 0.0 for item in retrievals]
    citation_logs = []
    missing_citation_count = 0
    for item in retrievals:
        doc = item.get("document", item)
        paper_id = _extract_arxiv_id_from_doc(doc if isinstance(doc, dict) else {})
        if paper_id not in citation_counts:
            missing_citation_count += 1
        citation_logs.append(math.log1p(citation_counts.get(paper_id, 0.0)))

    with _RERANK_STATS_LOCK:
        _RERANK_STATS["calls"] += 1
        _RERANK_STATS["candidate_count"] += len(retrievals)
        _RERANK_STATS["missing_citation_count"] += missing_citation_count
        if _RERANK_STATS["calls"] % 1000 == 0:
            logger.info(
                "Rerank citation coverage: calls=%s candidates=%s missing=%s missing_rate=%.4f",
                _RERANK_STATS["calls"],
                _RERANK_STATS["candidate_count"],
                _RERANK_STATS["missing_citation_count"],
                _RERANK_STATS["missing_citation_count"] / max(1, _RERANK_STATS["candidate_count"]),
            )

    if norm == "rank":
        semantic_norm = _rank_percentile(semantic_scores)
        citation_norm = _rank_percentile(citation_logs)
    elif norm == "minmax":
        semantic_norm = _minmax(semantic_scores)
        citation_norm = _minmax(citation_logs)
    else:
        raise ValueError(f"unsupported rerank norm: {norm}")

    scored = []
    for idx, item in enumerate(retrievals):
        blend_score = semantic_weight * semantic_norm[idx] + citation_weight * citation_norm[idx]
        scored.append((blend_score, -idx, item))
    scored.sort(reverse=True)
    return [item for *_scores, item in scored[:final_topk]]


@dataclass
class CitationPredictionV4EnvConfig:
    log_requests: bool = False
    search_url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 10
    timeout: int = 30
    max_predictions_ratio: float = 2.0
    max_question_segments: int = 4
    max_searches_per_question: int = 4
    protocol_violation_reward: float = 0.0
    rerank_retrieval_topk: Optional[int] = None
    rerank_alpha: Optional[float] = None
    rerank_final_topk: Optional[int] = None
    rerank_norm: str = "rank"
    citation_count_path: str = ""
    citation_metric_beta: float = 1.0
    eval_topk: Optional[int] = None
    eval_rerank_retrieval_topk: Optional[int] = None
    eval_rerank_alpha: Optional[float] = None
    eval_rerank_final_topk: Optional[int] = None
    eval_rerank_norm: Optional[str] = None
    eval_citation_count_path: Optional[str] = None
    eval_citation_metric_beta: Optional[float] = None
    enable_question_coverage_reward: bool = False
    question_reward_mode: str = "final_correct_qhit"
    question_coverage_lambda: float = 0.2
    question_overlap_alpha: float = 0.1
    process_evidence_weight: float = 0.5
    process_expand_weight: float = 0.5
    unsupported_citation_alpha: float = 0.25
    terminate_on_no_action: bool = True
    max_authors_in_result: Optional[int] = None


class CitationPredictionV4Env(BaseTextEnv):
    """Private-vocab citation-prediction protocol used by v4 SFT trajectories."""

    def __init__(self, env_config: Union[CitationPredictionV4EnvConfig, DictConfig, dict], extras: Dict[str, Any] = {}):
        super().__init__()
        assert "reward_spec" in extras, "reward_spec field is required"
        reward_spec = extras["reward_spec"]
        if isinstance(reward_spec, str):
            reward_spec = json.loads(reward_spec)
        assert "ground_truth" in reward_spec, "ground_truth is required in reward_spec field"

        if isinstance(env_config, DictConfig):
            env_config = dict(env_config)
        if isinstance(env_config, dict):
            env_config = CitationPredictionV4EnvConfig(**env_config)

        self.ground_truth_ids = reward_spec["ground_truth"]["targets"]
        self.search_url = env_config.search_url
        self.topk = env_config.topk
        self.rerank_retrieval_topk = _coerce_optional_int(env_config.rerank_retrieval_topk)
        self.rerank_alpha = _coerce_optional_float(env_config.rerank_alpha)
        self.rerank_final_topk = _coerce_optional_int(env_config.rerank_final_topk) or self.topk
        self.rerank_norm = env_config.rerank_norm
        self.citation_counts = _load_citation_counts(env_config.citation_count_path, float(env_config.citation_metric_beta))
        self.use_rerank = (
            self.rerank_retrieval_topk is not None
            and self.rerank_retrieval_topk > self.rerank_final_topk
            and self.rerank_alpha is not None
        )
        if self.use_rerank:
            logger.info(
                "Citation rerank enabled: retrieval_topk=%s final_topk=%s alpha=%s citation_weight=%s norm=%s citation_rows=%s",
                self.rerank_retrieval_topk,
                self.rerank_final_topk,
                self.rerank_alpha,
                1.0 - self.rerank_alpha,
                self.rerank_norm,
                len(self.citation_counts),
            )
        self.timeout = env_config.timeout
        self.log_requests = env_config.log_requests
        self.max_predictions_ratio = env_config.max_predictions_ratio
        self.max_question_segments = env_config.max_question_segments
        self.max_searches_per_question = env_config.max_searches_per_question
        self.protocol_violation_reward = env_config.protocol_violation_reward
        self.enable_question_coverage_reward = bool(env_config.enable_question_coverage_reward)
        self.question_reward_mode = str(env_config.question_reward_mode)
        self.question_coverage_lambda = float(env_config.question_coverage_lambda)
        self.question_overlap_alpha = float(env_config.question_overlap_alpha)
        self.process_evidence_weight = float(env_config.process_evidence_weight)
        self.process_expand_weight = float(env_config.process_expand_weight)
        self.unsupported_citation_alpha = float(env_config.unsupported_citation_alpha)
        self.terminate_on_no_action = bool(env_config.terminate_on_no_action)
        self.max_authors_in_result = _coerce_optional_int(env_config.max_authors_in_result)
        self.max_search_turns = self.max_question_segments * self.max_searches_per_question
        self.max_turns = self.max_search_turns + 1
        self.citation_budget = int(len(self.ground_truth_ids) * self.max_predictions_ratio)

        self.tool_group = SearchToolGroup(
            search_url=self.search_url,
            topk=self.topk,
            timeout=self.timeout,
            log_requests=self.log_requests,
        )
        self.init_tool_groups([self.tool_group])

        self.chat_history: ConversationType = []
        self.question_segments = 0
        self.searches_in_current_question = 0
        self.total_searches = 0
        self.limit_violation = False
        self.limit_violation_reason: Optional[str] = None
        self.current_question_index: Optional[int] = None
        self.question_evidence_sets: list[set[str]] = []
        self.last_reward_metrics: dict[str, float] = {}

    def init(self, prompt: ConversationType):
        self.chat_history = [dict(message) for message in prompt]
        return prompt, {}

    @staticmethod
    def _last_tag_value(text: str, tag: str) -> Optional[str]:
        matches = list(re.finditer(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL))
        if not matches:
            return None
        return matches[-1].group(1).strip()

    @staticmethod
    def _normalize_arxiv_citation(text: str) -> str:
        match = re.fullmatch(r"(?:\[?\s*(?:arxiv:)?\s*)?(\d{4}\.\d{4,5})(?:v\d+)?\s*\]?", text.strip(), re.IGNORECASE)
        return match.group(1) if match else ""

    @classmethod
    def _extract_citations(cls, text: str) -> set[str]:
        ids: set[str] = set()
        for normalized in cls._extract_citation_list(text):
            ids.add(normalized)
        return ids

    @classmethod
    def _extract_citation_list(cls, text: str) -> list[str]:
        ids: list[str] = []
        for match in re.finditer(r"<citation>(.*?)</citation>", text, re.DOTALL):
            for part in match.group(1).split(","):
                normalized = cls._normalize_arxiv_citation(part)
                if normalized:
                    ids.append(normalized)
        return ids

    def _assistant_text(self) -> str:
        return "".join(item["content"] for item in self.chat_history if item.get("role") == "assistant")

    def _assistant_citations(self) -> set[str]:
        return self._extract_citations(self._assistant_text())

    def _assistant_citation_list(self) -> list[str]:
        return self._extract_citation_list(self._assistant_text())

    def _truncate_author_list_in_contents(self, contents: str) -> str:
        if self.max_authors_in_result is None or self.max_authors_in_result <= 0:
            return contents
        match = re.search(r"(Authors:)(.*?)(\n\nAbstract:)", contents, flags=re.DOTALL)
        if not match:
            return contents
        authors = match.group(2).strip()
        parts = [part.strip() for part in authors.split(",") if part.strip()]
        if len(parts) <= self.max_authors_in_result:
            return contents
        truncated = ", ".join(parts[: self.max_authors_in_result]) + ", et al."
        original = match.group(2)
        leading = original[: len(original) - len(original.lstrip())]
        trailing = original[len(original.rstrip()) :]
        return contents[: match.start(2)] + leading + truncated + trailing + contents[match.end(2) :]

    def _format_search_results(self, query: str) -> str:
        request_topk = self.rerank_retrieval_topk if self.use_rerank else self.topk
        api_response, error_msg = call_search_api(
            retrieval_service_url=self.search_url,
            query=query,
            topk=request_topk,
            timeout=self.timeout,
            log_requests=self.log_requests,
            session=self.tool_group.session,
        )
        if error_msg or not api_response:
            return "\n<information>" + json.dumps({"result": f"Search error: {error_msg}"}) + "</information>\n"

        raw_results = api_response.get("result", [])
        if not raw_results:
            return "\n<information>" + json.dumps({"result": "No search results found."}) + "</information>\n"

        formatted_parts = []
        for retrieval in raw_results:
            if self.use_rerank:
                retrieval = _rerank_retrievals(
                    retrieval,
                    final_topk=self.rerank_final_topk,
                    citation_counts=self.citation_counts,
                    semantic_weight=float(self.rerank_alpha),
                    citation_weight=1.0 - float(self.rerank_alpha),
                    norm=self.rerank_norm,
                )
            formatted = ""
            for idx, doc_item in enumerate(retrieval):
                content = doc_item["document"]["contents"].strip()
                content = self._truncate_author_list_in_contents(content)
                formatted += f"Doc {idx + 1}: {content}\n"
            formatted_parts.append(formatted)
        final_result = "\n---\n".join(formatted_parts)
        return "\n<information>" + json.dumps({"result": final_result}) + "</information>\n"

    def _update_session_counts(self, action: str) -> None:
        question_matches = list(re.finditer(r"<question>.*?</question>", action, re.DOTALL))
        if question_matches:
            self.question_segments += len(question_matches)
            self.searches_in_current_question = 0
            for _match in question_matches:
                self.question_evidence_sets.append(set())
            self.current_question_index = len(self.question_evidence_sets) - 1

        search_count = len(re.findall(r"<search>.*?</search>", action, re.DOTALL))
        self.searches_in_current_question += search_count
        self.total_searches += search_count
        self.turns = self.total_searches

    def _record_question_evidence(self, text: str) -> None:
        if self.current_question_index is None:
            return
        if self.current_question_index >= len(self.question_evidence_sets):
            return
        self.question_evidence_sets[self.current_question_index].update(_extract_seen_arxiv_ids(text))

    @staticmethod
    def _has_unbalanced_tag(text: str, tag: str) -> bool:
        return len(re.findall(rf"<{tag}>", text)) != len(re.findall(rf"</{tag}>", text))

    def _protocol_violation_reason(self, action: str) -> Optional[str]:
        for tag in ("question", "search", "citation", "done"):
            if self._has_unbalanced_tag(action, tag):
                return f"malformed_{tag}_tag"
        if self.question_segments > self.max_question_segments:
            return f"too_many_question_segments:{self.question_segments}>{self.max_question_segments}"
        if self.searches_in_current_question > self.max_searches_per_question:
            return f"too_many_searches_in_question:{self.searches_in_current_question}>{self.max_searches_per_question}"
        if self.total_searches > self.max_search_turns:
            return f"too_many_total_searches:{self.total_searches}>{self.max_search_turns}"
        return None

    def _limit_message(self) -> str:
        remaining_questions = max(0, self.max_question_segments - self.question_segments)
        remaining_searches = max(0, self.max_searches_per_question - self.searches_in_current_question)
        remaining_total_searches = max(0, self.max_search_turns - self.total_searches)
        num_cited = len(self._assistant_citation_list())
        if remaining_total_searches == 0:
            return (
                f"\n\nNo search turns remaining. Citations so far: {num_cited}/{self.citation_budget} max. "
                "Cite any remaining papers and write <done></done>."
            )
        if remaining_searches == 0:
            return (
                f"\n\nYou have used all {self.max_searches_per_question} searches for the current <question>. "
                "Do not emit another <search> in this <question>. "
                "Either open a new <question> for a different sub-question, or cite the final papers and write <done></done>. "
                f"Environment limits remaining: {remaining_total_searches} total search turns, "
                f"{remaining_questions} sub-questions. Citations so far: {num_cited}/{self.citation_budget} max."
            )
        return (
            f"\n\nEnvironment limits remaining: {remaining_total_searches} total search turns, "
            f"{remaining_questions} sub-questions, "
            f"{remaining_searches} searches in the current sub-question. "
            f"Citations so far: {num_cited}/{self.citation_budget} max."
        )

    def _is_done(self, action: str) -> bool:
        if "<done>" in action:
            return True
        return False

    def _get_reward(self, done: bool) -> float:
        if not done:
            return 0.0
        predicted = self._assistant_citations()
        predicted_list = self._assistant_citation_list()
        gt_set = {normalize_arxiv_id(gid) for gid in self.ground_truth_ids}
        gt_set.discard("")
        correct = predicted & gt_set
        root_recall = len(correct) / len(gt_set) if gt_set else 0.0
        citation_budget = self.max_predictions_ratio * len(gt_set)
        over_citation = len(predicted_list) > citation_budget
        retrieved_gold = {
            cid
            for evidence_ids in self.question_evidence_sets
            for cid in (evidence_ids & gt_set)
        }
        gold_evidence_recall = len(retrieved_gold) / len(gt_set) if gt_set else 0.0
        supported_correct = {
            cid
            for evidence_ids in self.question_evidence_sets
            for cid in (evidence_ids & correct)
        }
        final_correct_supported_rate = len(supported_correct) / len(correct) if correct else 0.0

        question_successes = [
            1.0 if correct and bool(evidence_ids & correct) else 0.0
            for evidence_ids in self.question_evidence_sets
        ]
        question_coverage = sum(question_successes) / len(question_successes) if question_successes else 0.0
        question_overlap = _pairwise_jaccard(self.question_evidence_sets)
        seen_gold: set[str] = set()
        additional_question_gain = 0.0
        wasted_overlap_values: list[float] = []
        for question_idx, evidence_ids in enumerate(self.question_evidence_sets):
            question_gold = evidence_ids & gt_set
            new_gold = question_gold - seen_gold
            delta = len(new_gold) / len(gt_set) if gt_set else 0.0
            if question_idx > 0:
                additional_question_gain += delta
                if delta == 0.0:
                    previous_overlaps = [
                        _jaccard(evidence_ids, previous_evidence)
                        for previous_evidence in self.question_evidence_sets[:question_idx]
                    ]
                    wasted_overlap_values.append(max(previous_overlaps) if previous_overlaps else 0.0)
            seen_gold.update(question_gold)
        wasted_question_overlap = (
            sum(wasted_overlap_values) / len(wasted_overlap_values) if wasted_overlap_values else 0.0
        )
        raw_process_quality = (
            self.process_evidence_weight * gold_evidence_recall
            + self.process_expand_weight * additional_question_gain
        )
        process_quality = raw_process_quality * (1.0 - self.question_overlap_alpha * wasted_question_overlap)
        process_bonus = self.question_coverage_lambda * (1.0 - root_recall) * process_quality
        unsupported_penalty = (
            self.unsupported_citation_alpha
            * root_recall
            * (1.0 - final_correct_supported_rate)
            if root_recall > 0.0
            else 0.0
        )

        shaped_reward = root_recall
        if self.enable_question_coverage_reward:
            if self.question_reward_mode == "final_correct_qhit":
                shaped_reward = (
                    (root_recall + self.question_coverage_lambda * question_coverage)
                    / (1.0 + self.question_coverage_lambda)
                    - self.question_overlap_alpha * question_overlap
                )
                shaped_reward = min(1.0, max(0.0, shaped_reward))
            elif self.question_reward_mode == "gold_evidence_delta":
                retrieved_credit = self.question_coverage_lambda * max(0.0, gold_evidence_recall - root_recall)
                unsupported_delta_penalty = (
                    self.unsupported_citation_alpha
                    * root_recall
                    * (1.0 - final_correct_supported_rate)
                    if root_recall > 0.0
                    else 0.0
                )
                shaped_reward = (
                    root_recall
                    + retrieved_credit
                    - self.question_overlap_alpha * question_overlap
                    - unsupported_delta_penalty
                )
                shaped_reward = min(1.0, max(0.0, shaped_reward))
            elif self.question_reward_mode == "marginal_gold_evidence":
                shaped_reward = root_recall + process_bonus - unsupported_penalty
            else:
                raise ValueError(f"unsupported question_reward_mode: {self.question_reward_mode}")
        if over_citation:
            shaped_reward = 0.0

        self.last_reward_metrics = {
            "root_recall": root_recall,
            "question_coverage": question_coverage,
            "question_overlap": question_overlap,
            "gold_evidence_recall": gold_evidence_recall,
            "additional_question_gain": additional_question_gain,
            "wasted_question_overlap": wasted_question_overlap,
            "raw_process_quality": raw_process_quality,
            "process_quality": process_quality,
            "process_bonus": process_bonus,
            "unsupported_citation_penalty": unsupported_penalty,
            "shaped_reward": shaped_reward,
            "final_correct_supported_rate": final_correct_supported_rate,
            "over_citation": float(over_citation),
            "num_citation_tags": float(len(predicted_list)),
            "num_unique_citations": float(len(predicted)),
            "num_duplicate_citation_tags": float(max(0, len(predicted_list) - len(predicted))),
        }
        if over_citation:
            return 0.0
        return shaped_reward

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.chat_history.append({"role": "assistant", "content": action})
        self._update_session_counts(action)
        violation_reason = self._protocol_violation_reason(action)
        if violation_reason is not None:
            self.limit_violation = True
            self.limit_violation_reason = violation_reason
            return BaseTextEnvStepOutput(
                observations=[],
                reward=float(self.protocol_violation_reward),
                done=True,
                metadata={
                    "tool_group": None,
                    "tool_name": None,
                    "tool_input": None,
                    "limit_violation": True,
                    "limit_violation_reason": violation_reason,
                },
            )

        done = self._is_done(action)
        reward = self._get_reward(done)
        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})

        query = self._last_tag_value(action, "search")
        if query is None:
            if self.terminate_on_no_action:
                self.limit_violation = True
                self.limit_violation_reason = "no_search_or_done_action"
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=float(self.protocol_violation_reward),
                    done=True,
                    metadata={
                        "tool_group": None,
                        "tool_name": None,
                        "tool_input": None,
                        "limit_violation": True,
                        "limit_violation_reason": self.limit_violation_reason,
                    },
                )
            observation = self._limit_message().strip()
            metadata = {"tool_group": None, "tool_name": None, "tool_input": None}
        else:
            observation = self._format_search_results(query) + self._limit_message()
            self._record_question_evidence(observation)
            metadata = {"tool_group": "SearchToolGroup", "tool_name": "search", "tool_input": [query]}

        obs = {"role": "user", "content": observation}
        self.chat_history.append(obs)
        return BaseTextEnvStepOutput(observations=[obs], reward=reward, done=False, metadata=metadata)

    def get_metrics(self) -> Dict[str, Any]:
        chat_history_str = "".join(item["content"] for item in self.chat_history)
        predicted = self._assistant_citations()
        predicted_list = self._assistant_citation_list()
        gt_set = {normalize_arxiv_id(gid) for gid in self.ground_truth_ids}
        correct = predicted & gt_set
        return {
            "num_predicted": len(predicted),
            "num_citation_tags": len(predicted_list),
            "num_duplicate_citation_tags": max(0, len(predicted_list) - len(predicted)),
            "num_ground_truth": len(gt_set),
            "num_correct": len(correct),
            "recall": len(correct) / len(gt_set) if gt_set else 0.0,
            "precision": len(correct) / len(predicted) if predicted else 0.0,
            "root_recall": self.last_reward_metrics.get("root_recall", len(correct) / len(gt_set) if gt_set else 0.0),
            "question_coverage": self.last_reward_metrics.get("question_coverage", 0.0),
            "question_overlap": self.last_reward_metrics.get("question_overlap", 0.0),
            "gold_evidence_recall": self.last_reward_metrics.get("gold_evidence_recall", 0.0),
            "additional_question_gain": self.last_reward_metrics.get("additional_question_gain", 0.0),
            "wasted_question_overlap": self.last_reward_metrics.get("wasted_question_overlap", 0.0),
            "raw_process_quality": self.last_reward_metrics.get("raw_process_quality", 0.0),
            "process_quality": self.last_reward_metrics.get("process_quality", 0.0),
            "process_bonus": self.last_reward_metrics.get("process_bonus", 0.0),
            "unsupported_citation_penalty": self.last_reward_metrics.get("unsupported_citation_penalty", 0.0),
            "shaped_reward": self.last_reward_metrics.get("shaped_reward", 0.0),
            "final_correct_supported_rate": self.last_reward_metrics.get("final_correct_supported_rate", 0.0),
            "over_citation": self.last_reward_metrics.get("over_citation", 0.0),
            "answered": int("<done>" in chat_history_str),
            "num_question_segments": self.question_segments,
            "num_searches": self.total_searches,
            "limit_violation": int(self.limit_violation),
            "limit_violation_reason": self.limit_violation_reason or "",
        }
