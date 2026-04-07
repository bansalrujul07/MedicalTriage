import argparse
import csv
from pathlib import Path

from triage_env.agents.random_agent import RandomAgent
from triage_env.agents.rule_based_agent import RuleBasedAgent
from triage_env.agents.llm_agent import LLMAgent
from triage_env.config import get_runtime_config
from triage_env.evaluation.benchmark import benchmark_agents
from triage_env.evaluation.task2_assessment import assess_task2_summary
from triage_env.training.train_rl import train_rl_agent


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    runtime = get_runtime_config()
    parser = argparse.ArgumentParser(description="Task 2 progression: train, compare, diagnose")
    parser.add_argument("--episodes", type=int, default=runtime.eval_episodes)
    parser.add_argument("--train", action="store_true", help="Train a fresh task2 RL model before comparing")
    parser.add_argument("--train-episodes", type=int, default=runtime.train_episodes)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    task2_rl_model = PACKAGE_ROOT / "training" / "triage_rl_qtable_task2.json"
    task1_warm_start = PACKAGE_ROOT / "training" / "triage_rl_qtable_task1_after_alignment.json"
    if args.train or not task2_rl_model.exists():
        print(f"Training task2 RL model for {args.train_episodes} episodes...")
        train_rl_agent(
            episodes=args.train_episodes,
            task="task2",
            save_path=str(task2_rl_model),
            warm_start_model_path=str(task1_warm_start) if task1_warm_start.exists() else None,
        )

    results = benchmark_agents(
        num_episodes=args.episodes,
        task="task2",
        rl_model_path=str(task2_rl_model),
    )

    rule_based_summary = next(row for row in results if row["agent_name"] == "RuleBasedAgent")
    rule_based_reward = float(rule_based_summary["avg_total_reward"])

    print("\nTASK 2 PROGRESSION")
    print("=" * 92)
    print("Agent           Reward      CritSurv  Success  VentUtil  Invalid  TargetStatus")
    print("-" * 92)

    assessments = []
    for row in results:
        assessment = assess_task2_summary(row, rule_based_reward=rule_based_reward)
        assessments.append(assessment)
        target_status = "PASS" if assessment.meets_targets else ",".join(assessment.failure_modes) or "UNKNOWN"
        print(
            f"{assessment.agent_name:<15} "
            f"{assessment.avg_total_reward:>8.2f} "
            f"{assessment.critical_survival_rate:>10.2f} "
            f"{assessment.success_rate:>8.2f} "
            f"{assessment.ventilator_utilization:>8.2f} "
            f"{assessment.invalid_action_count:>8.0f} "
            f"{target_status}"
        )

    print("\nTask 2 target profile:")
    print("- Critical survival minimum: 0.85")
    print("- Preferred critical band: 0.85 to 0.95")
    print("- Success rate minimum: 0.80")
    print("- Ventilator utilization band: 0.20 to 0.60")
    print("- Invalid actions: 0")
    print("- Reward: above RuleBasedAgent baseline")

    print("\nFailure modes:")
    for assessment in assessments:
        if assessment.failure_modes:
            print(f"- {assessment.agent_name}: {', '.join(assessment.failure_modes)}")
        else:
            print(f"- {assessment.agent_name}: none")

    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "agent_name",
                    "avg_total_reward",
                    "critical_survival_rate",
                    "success_rate",
                    "ventilator_utilization",
                    "invalid_action_count",
                    "meets_targets",
                    "failure_modes",
                ]
            )
            for assessment in assessments:
                writer.writerow(
                    [
                        assessment.agent_name,
                        f"{assessment.avg_total_reward:.4f}",
                        f"{assessment.critical_survival_rate:.4f}",
                        f"{assessment.success_rate:.4f}",
                        f"{assessment.ventilator_utilization:.4f}",
                        f"{assessment.invalid_action_count:.0f}",
                        assessment.meets_targets,
                        ";".join(assessment.failure_modes),
                    ]
                )
        print(f"\nSaved task2 progression report to {output_path}")


if __name__ == "__main__":
    main()
