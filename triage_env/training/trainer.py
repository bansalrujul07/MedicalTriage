from triage_env.training.rollout import run_episode

def train(env, agent, episodes=10):
    results = []

    for episode in range(episodes):
        episode_result = run_episode(env, agent)
        results.append(episode_result)
        print(f"Episode {episode + 1}: {episode_result}")

    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    avg_alive = sum(r["alive_patients"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)

    summary = {
        "episodes": len(results),
        "avg_reward": avg_reward,
        "avg_alive_patients": avg_alive,
        "avg_steps": avg_steps,
    }

    print("\nTraining Summary:")
    print(summary)

    return summary