import argparse
from pathlib import Path

from triage_env.evaluation.benchmark import benchmark_agents, save_summary_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedicalTriage benchmarks")
    parser.add_argument("--tasks", default=None, help="Comma-separated tasks, e.g. task1,task2")
    parser.add_argument(
        "--agents",
        default=None,
        help="Comma-separated agents: RandomAgent,RuleBasedAgent,LLMAgent,RLAgent,TrainedQAgent",
    )
    parser.add_argument("--task", choices=["task1", "task2", "task3"], default=None)
    parser.add_argument(
        "--agent",
        choices=["RandomAgent", "RuleBasedAgent", "LLMAgent", "RLAgent", "TrainedQAgent"],
        default=None,
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else ([args.task] if args.task else [None])
    agents = [a.strip() for a in args.agents.split(",") if a.strip()] if args.agents else ([args.agent] if args.agent else [None])

    results = []
    for task_name in tasks:
        for agent_name in agents:
            results.extend(
                benchmark_agents(
                    num_episodes=args.episodes,
                    task=task_name,
                    agent_name=agent_name,
                )
            )

    current_task = None
    for row in results:
        if row["task"] != current_task:
            current_task = row["task"]
            print(f"\n{current_task}:")
        print(
            f"  {row['agent_name']}: "
            f"avg_reward={row['avg_total_reward']:.2f}, "
            f"survival_rate={row['survival_rate']:.2f}, "
            f"critical_survival_rate={row['critical_survival_rate']:.2f}, "
            f"success_rate={row['success_rate']:.2f}"
        )
        if row.get("checkpoint_warning"):
            print(f"  checkpoint_warning={row['checkpoint_warning']}")
        if row.get("checkpoint_status"):
            print(f"  checkpoint_status={row['checkpoint_status']}")

    package_root = Path(__file__).resolve().parents[1]
    output = Path(args.output) if args.output else (package_root / "evaluation" / "results" / "benchmark_summary.csv")
    save_summary_csv(results, str(output))
    print(f"Saved benchmark results to {output}")


if __name__ == "__main__":
    main()
