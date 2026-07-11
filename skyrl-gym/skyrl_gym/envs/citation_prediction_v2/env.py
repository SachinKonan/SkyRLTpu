from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.citation_prediction_v2.utils import extract_all_citations, compute_recall_reward, normalize_arxiv_id
from skyrl_gym.tools.search import call_search_api, SearchToolGroup
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from omegaconf import DictConfig
import json
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class CitationPredictionV2EnvConfig:
    log_requests: bool = False
    search_url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 3
    timeout: int = 30
    max_predictions_ratio: float = 2.0


class CitationPredictionV2Env(BaseTextEnv):
    """
    Environment for citation prediction v2: predict ALL citations in a
    Related Work subsection.

    The model searches over an arxiv corpus and submits citations using
    <citation>id1, id2, ...</citation> tags. When done, it writes <done></done>.

    Reward = recall with spam penalty:
      - recall = |predicted & ground_truth| / |ground_truth|
      - reward = 0 if |predicted| > 2 * |ground_truth| (spam penalty)

    Stop strings: ["</search>", "</done>"]
    """

    def __init__(self, env_config: Union[CitationPredictionV2EnvConfig, DictConfig, dict], extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        reward_spec = extras["reward_spec"]
        if isinstance(reward_spec, str):
            reward_spec = json.loads(reward_spec)
        assert "ground_truth" in reward_spec, "ground_truth is required in reward_spec field"

        self.ground_truth = reward_spec["ground_truth"]
        self.ground_truth_ids = self.ground_truth["targets"]  # list of arxiv IDs
        self.max_turns = extras.get("max_turns", 15)
        self.citation_budget = int(len(self.ground_truth_ids) * 2)  # 2x ground truth

        if isinstance(env_config, dict):
            env_config = CitationPredictionV2EnvConfig(**env_config)

        self.search_url = env_config.search_url
        self.topk = env_config.topk
        self.timeout = env_config.timeout
        self.log_requests = env_config.log_requests
        self.max_predictions_ratio = env_config.max_predictions_ratio

        self.tool_group = SearchToolGroup(
            search_url=self.search_url,
            topk=self.topk,
            timeout=self.timeout,
            log_requests=self.log_requests,
        )
        self.init_tool_groups([self.tool_group])

        self.chat_history: ConversationType = []

    def _parse_action(self, action: str) -> List[Optional[str]]:
        match = None
        if "<search>" in action and "</search>" in action:
            match = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
        return [match.group(1)] if match else [None]

    def _do_search(self, query: str) -> str:
        """Call retriever API and return formatted text for model."""
        api_response, error_msg = call_search_api(
            retrieval_service_url=self.search_url,
            query=query,
            topk=self.topk,
            timeout=self.timeout,
            log_requests=self.log_requests,
            session=self.tool_group.session,
        )

        if error_msg or not api_response:
            return "\n<information>" + json.dumps({"result": f"Search error: {error_msg}"}) + "</information>\n"

        raw_results = api_response.get("result", [])
        if not raw_results:
            return "\n<information>" + json.dumps({"result": "No search results found."}) + "</information>\n"

        pretty_parts = []
        for retrieval in raw_results:
            formatted = ""
            for idx, doc_item in enumerate(retrieval):
                content = doc_item["document"]["contents"].strip()
                formatted += f"Doc {idx+1}: {content}\n"
            pretty_parts.append(formatted)
        final_result = "\n---\n".join(pretty_parts)

        return "\n<information>" + json.dumps({"result": final_result}) + "</information>\n"

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<done>" in action

    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            return compute_recall_reward(
                chat_history_str,
                self.ground_truth_ids,
                max_ratio=self.max_predictions_ratio,
            )
        return 0.0

    def _validate_action(self, action: str):
        stop_tags = ["</search>", "</done>"]
        for tag in stop_tags:
            if tag in action:
                assert action.split(tag, 1)[1] == "", (
                    f"{tag} detected in the response but it is not the last string generated. "
                    f"Use {stop_tags} as stop strings in the configuration."
                )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self._validate_action(action)
        self.chat_history.append({"role": "assistant", "content": action})

        error = None

        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

        # Execute search if the model issued one
        observation = None
        query = self._parse_action(action)
        if query[0] is not None:
            try:
                observation = self._do_search(query[0])
            except Exception as e:
                error = str(e)
                observation = None

        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        # Count citations so far and compute remaining budget
        chat_history_str = "".join([item["content"] for item in self.chat_history])
        num_cited = len(extract_all_citations(chat_history_str))

        # Turn counter with citation budget
        remaining = self.max_turns - self.turns
        if remaining > 1:
            turn_msg = f"\n\n{remaining} turns remaining. Citations so far: {num_cited}/{self.citation_budget} max. You may cite fewer than the max — only cite papers you are confident belong in this subsection."
        elif remaining == 1:
            turn_msg = f"\n\nThis is your last turn. Citations so far: {num_cited}/{self.citation_budget} max. Cite any remaining papers and write <done></done>."
        else:
            turn_msg = ""

        if new_obs:
            if turn_msg:
                new_obs["content"] += turn_msg
        else:
            if turn_msg:
                new_obs = {"role": "user", "content": turn_msg.strip()}

        info = {
            "tool_group": "SearchToolGroup",
            "tool_name": "search",
            "tool_input": query,
        }

        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
        )

    def get_metrics(self) -> Dict[str, Any]:
        chat_history_str = "".join([item["content"] for item in self.chat_history])
        predicted = extract_all_citations(chat_history_str)
        gt_set = {normalize_arxiv_id(gid) for gid in self.ground_truth_ids}
        correct = predicted & gt_set

        recall = len(correct) / len(gt_set) if gt_set else 0.0
        precision = len(correct) / len(predicted) if predicted else 0.0

        answered = "<done>" in chat_history_str

        return {
            "num_predicted": len(predicted),
            "num_ground_truth": len(gt_set),
            "num_correct": len(correct),
            "recall": recall,
            "precision": precision,
            "answered": int(answered),
            "num_turns": self.turns,
        }
