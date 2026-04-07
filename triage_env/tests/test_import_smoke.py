def test_import_smoke_modules():
    import triage_env
    import triage_env.agents
    import triage_env.agents.llm_agent
    import triage_env.agents.rl_agents
    import triage_env.agents.q_learning_agents
    import triage_env.config
    import triage_env.evaluation.evaluator
    import triage_env.training.rollout
    import triage_env.training.state_encoder

    assert triage_env is not None
