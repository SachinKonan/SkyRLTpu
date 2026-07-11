from omegaconf import DictConfig

from skyrl_gym.envs.citation_prediction_v4.env import CitationPredictionV4Env


def make_env(targets=None, **config_overrides):
    targets = targets or ["2309.17080"]
    config = {
        "search_url": "http://127.0.0.1:8000/retrieve",
        "topk": 5,
        "timeout": 1,
        "log_requests": False,
        "max_predictions_ratio": 2.0,
    }
    config.update(config_overrides)
    return CitationPredictionV4Env(
        env_config=DictConfig(config),
        extras={
            "reward_spec": {
                "ground_truth": {
                    "targets": targets,
                }
            }
        },
    )


def set_assistant_text(env, text):
    env.chat_history = [{"role": "assistant", "content": text}]


def test_duplicate_citation_tags_count_against_budget():
    env = make_env(["2309.17080"])
    set_assistant_text(
        env,
        "".join(
            [
                "<citation>2309.17080</citation>",
                "<citation>2309.17080</citation>",
                "<citation>2309.17080</citation>",
                "<done></done>",
            ]
        ),
    )

    assert env._assistant_citations() == {"2309.17080"}
    assert env._assistant_citation_list() == ["2309.17080", "2309.17080", "2309.17080"]
    assert env._get_reward(done=True) == 0.0
    assert env.last_reward_metrics["root_recall"] == 1.0
    assert env.last_reward_metrics["over_citation"] == 1.0
    assert env.last_reward_metrics["num_citation_tags"] == 3.0
    assert env.last_reward_metrics["num_unique_citations"] == 1.0
    assert env.last_reward_metrics["num_duplicate_citation_tags"] == 2.0


def test_citation_budget_allows_raw_tags_at_or_below_limit():
    env = make_env(["2309.17080"])
    set_assistant_text(
        env,
        "".join(
            [
                "<citation>2309.17080</citation>",
                "<citation>2503.20314</citation>",
                "<done></done>",
            ]
        ),
    )

    assert env._get_reward(done=True) == 1.0
    assert env.last_reward_metrics["over_citation"] == 0.0
    assert env.last_reward_metrics["num_citation_tags"] == 2.0
    assert env.last_reward_metrics["num_unique_citations"] == 2.0


def test_author_truncation_limits_search_result_authors():
    env = make_env(max_authors_in_result=2)
    contents = "Title: Example\nAuthors: Ada, Bob, Chen, Devi\n\nAbstract: Test"

    assert env._truncate_author_list_in_contents(contents) == (
        "Title: Example\nAuthors: Ada, Bob, et al.\n\nAbstract: Test"
    )


def test_no_search_or_done_action_terminates_with_protocol_reward():
    env = make_env(["2309.17080"])
    env.init([{"role": "user", "content": "Find citations."}])

    result = env.step("I should probably search next.")

    assert result["done"] is True
    assert result["reward"] == 0.0
    assert result["observations"] == []
    assert result["metadata"]["limit_violation"] is True
    assert result["metadata"]["limit_violation_reason"] == "no_search_or_done_action"
    assert env.get_metrics()["limit_violation"] == 1
    assert env.get_metrics()["limit_violation_reason"] == "no_search_or_done_action"
