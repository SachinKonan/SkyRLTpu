AGENT_GENERATOR_REGISTRY = {
    "skyrl_agent.agents.oh_codeact.OHCodeActAgent": "skyrl_agent.agents.base.AgentRunner",
    "skyrl_agent.agents.react.ReActAgent": "skyrl_agent.agents.base.AgentRunner",
    "skyrl_agent.agents.citation_prediction_v4_raw.CitationPredictionV4RawAgent": "skyrl_agent.agents.base.AgentRunner",
}

AGENT_TRAJECTORY_REGISTRY = {
    "skyrl_agent.agents.oh_codeact.OHCodeActAgent": "skyrl_agent.agents.oh_codeact.CodeActTrajectory",
    "skyrl_agent.agents.react.ReActAgent": "skyrl_agent.agents.react.ReActTrajectory",
    "skyrl_agent.agents.citation_prediction_v4_raw.CitationPredictionV4RawAgent": (
        "skyrl_agent.agents.react.ReActTrajectory"
    ),
}
