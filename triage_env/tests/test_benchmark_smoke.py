from triage_env.evaluation.benchmark import benchmark_agents


def test_benchmark_smoke_single_episode_task():
    rows = benchmark_agents(num_episodes=1, task="task1")

    assert len(rows) >= 3
    assert all(row["task"] == "task1" for row in rows)
    assert all("avg_total_reward" in row for row in rows)
