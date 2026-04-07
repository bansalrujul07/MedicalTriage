from triage_env.agents.random_agent import RandomAgent
from triage_env.evaluation.evaluator import evaluate, evaluate_agent
from triage_env.server.triage_env_environment import TriageEnvironment


def test_evaluate_agent_returns_summary_and_results():
    summary, results = evaluate_agent(TriageEnvironment, RandomAgent(), num_episodes=2, max_steps=5)

    assert summary["episodes"] == 2
    assert "avg_total_reward" in summary
    assert "avg_stabilization_rate" in summary
    assert len(results) == 2


def test_evaluate_wrapper_backwards_compatible():
    env = TriageEnvironment(max_steps=5)
    summary = evaluate(env, RandomAgent(), episodes=2, task="medium")

    assert summary["episodes"] == 2
