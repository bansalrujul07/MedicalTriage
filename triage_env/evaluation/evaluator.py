from triage_env.training.rollout import run_episode
from triage_env.evaluation.metrics import compute_metrics


def evaluate(env, agent, episodes=20):
    results = []

    for _ in range(episodes):
        episode_result = run_episode(env, agent)
        results.append(episode_result)

    metrics = compute_metrics(results)

    print("\nEvaluation Results:")
    print(metrics)

    return metrics