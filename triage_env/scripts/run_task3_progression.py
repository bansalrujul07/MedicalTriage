import argparse
import csv
from pathlib import Path

from triage_env.evaluation.benchmark import benchmark_agents
from triage_env.evaluation.task3_assessment import assess_task3_summary
from triage_env.training.train_rl import train_rl_agent
from triage_env.tasks import TASK_TRAINING_DEFAULTS


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3 progression: train, compare, diagnose")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--train", action="store_true", help="Train a fresh task3 RL model before comparing")
    parser.add_argument("--train-episodes", type=int, default=int(TASK_TRAINING_DEFAULTS["task3"]["episodes"]))
    parser.add_argument("--warm-start-model-path", default=None)
    parser.add_argument("--epsilon-start", type=float, default=float(TASK_TRAINING_DEFAULTS["task3"]["epsilon_start"]))
    parser.add_argument("--epsilon-end", type=float, default=float(TASK_TRAINING_DEFAULTS["task3"]["epsilon_end"]))
    parser.add_argument("--epsilon-decay", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    task3_rl_model = PACKAGE_ROOT / "training" / "triage_rl_qtable_task3.json"
    task2_warm_start = PACKAGE_ROOT / "training" / "triage_rl_qtable_task2.json"
    if args.train or not task3_rl_model.exists():
        warm_start_model_path = args.warm_start_model_path
        if warm_start_model_path is None and task2_warm_start.exists():
            warm_start_model_path = str(task2_warm_start)
        print(f"Training task3 RL model for {args.train_episodes} episodes...")
        train_rl_agent(
            episodes=args.train_episodes,
            task="task3",
            save_path=str(task3_rl_model),
            warm_start_model_path=warm_start_model_path,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
        )

    results = benchmark_agents(
        num_episodes=args.episodes,
        task="task3",
        rl_model_path=str(task3_rl_model),
    )

    rule_based_summary = next(row for row in results if row["agent_name"] == "RuleBasedAgent")
    rule_based_reward = float(rule_based_summary["avg_total_reward"])

    print("\nTASK 3 PROGRESSION")
    print("=" * 126)
    print("Agent           Reward      SurvRate  CritSurv  AvgAlive  AvgHealth  Success  VentUtil  Invalid  Milestones  TargetStatus")
    print("-" * 126)

    assessments = []
    summaries_by_agent = {}
    for row in results:
        summaries_by_agent[row["agent_name"]] = row
        assessment = assess_task3_summary(row, rule_based_reward=rule_based_reward)
        assessments.append(assessment)
        milestones = [
            f"A={'PASS' if assessment.milestone_a else 'FAIL'}",
            f"B={'PASS' if assessment.milestone_b else 'FAIL'}",
            f"C={'PASS' if assessment.milestone_c else 'FAIL'}",
            f"F={'PASS' if assessment.meets_targets else 'FAIL'}",
        ]
        target_status = "PASS" if assessment.meets_targets else ",".join(assessment.failure_modes) or "UNKNOWN"
        print(
            f"{assessment.agent_name:<15} "
            f"{assessment.avg_total_reward:>8.2f} "
            f"{float(row['survival_rate']):>9.2f} "
            f"{assessment.critical_survival_rate:>10.2f} "
            f"{float(row['avg_survivors']):>8.2f} "
            f"{float(row['avg_health_alive']):>10.2f} "
            f"{assessment.success_rate:>8.2f} "
            f"{assessment.ventilator_utilization:>8.2f} "
            f"{assessment.invalid_action_count:>8.0f} "
            f"{' '.join(milestones):<11} "
            f"{target_status}"
        )

        if assessment.checkpoint_status or assessment.checkpoint_warning:
            print(
                f"  checkpoint_status={assessment.checkpoint_status or 'unknown'}"
                f" checkpoint_warning={assessment.checkpoint_warning or 'none'}"
            )
        checkpoint_metadata = row.get("checkpoint_metadata") or {}
        if checkpoint_metadata:
            warm_start_task = checkpoint_metadata.get("warm_start_task")
            warm_start_model_path = checkpoint_metadata.get("warm_start_model_path")
            if warm_start_task or warm_start_model_path:
                print(
                    f"  warm_start_task={warm_start_task or 'unknown'} "
                    f"warm_start_model_path={warm_start_model_path or 'unknown'}"
                )

    print("\nTask 3 target profile:")
    print("- Milestone A (primary): success_rate > 0.00, invalid_actions = 0, reward > RuleBased")
    print("- Milestone B: success_rate >= 0.15, critical_survival >= 0.50, ventilator_utilization >= 0.20")
    print("- Milestone C: success_rate >= 0.25, critical_survival >= 0.60")
    print("- Critical survival minimum: 0.60")
    print("- Success rate minimum: 0.40")
    print("- Ventilator utilization minimum: 0.20")
    print("- Invalid actions: 0")
    print("- Reward: above RuleBasedAgent baseline")

    print("\nFailure modes:")
    for assessment in assessments:
        row = summaries_by_agent.get(assessment.agent_name, {})
        if assessment.failure_reason_counts:
            ordered = sorted(assessment.failure_reason_counts.items(), key=lambda item: (-item[1], item[0]))
            failure_reason_text = ", ".join(f"{key}:{count}" for key, count in ordered)
        else:
            failure_reason_text = "none"
        if assessment.failure_modes:
            print(f"- {assessment.agent_name}: {', '.join(assessment.failure_modes)}")
        else:
            print(f"- {assessment.agent_name}: none")
        print(f"  terminal_failure_reasons: {failure_reason_text}")
        terminal_diag = row.get("terminal_diagnostics") or {}
        if terminal_diag:
            print(
                "  terminal_diag: "
                f"alive_count={terminal_diag.get('alive_count')}, "
                f"survival_rate={terminal_diag.get('survival_rate')}, "
                f"critical_survival_rate={terminal_diag.get('critical_survival_rate')}, "
                f"avg_health_alive={terminal_diag.get('avg_health_alive')}, "
                f"vent_occupancy={terminal_diag.get('ventilator_occupancy')}"
            )
        deaths_by_severity = row.get("deaths_by_severity")
        if deaths_by_severity:
            print(f"  deaths_by_severity: {deaths_by_severity}")

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
                    "milestone_a",
                    "milestone_b",
                    "milestone_c",
                    "failure_modes",
                    "failure_reason_counts",
                    "checkpoint_status",
                    "checkpoint_warning",
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
                        assessment.milestone_a,
                        assessment.milestone_b,
                        assessment.milestone_c,
                        ";".join(assessment.failure_modes),
                        ";".join(f"{key}:{value}" for key, value in sorted(assessment.failure_reason_counts.items())),
                        assessment.checkpoint_status or "",
                        assessment.checkpoint_warning or "",
                    ]
                )
        print(f"\nSaved task3 progression report to {output_path}")


if __name__ == "__main__":
    main()
