from triage_env.agents.random_agent import RandomAgent
from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.training.rollout import run_episode


def test_run_episode_works_without_task_reset_kwarg():
    env = TriageEnvironment(max_steps=5)
    agent = RandomAgent()

    result = run_episode(env, agent, training=False)

    assert "total_reward" in result
    assert result["steps"] >= 1
