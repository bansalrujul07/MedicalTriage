import argparse

from triage_env.agents.rule_based_agent import RuleBasedAgent
from triage_env.config import get_runtime_config
from triage_env.evaluation.evaluator import run_single_episode
from triage_env.server.triage_env_environment import TriageEnvironment


def main() -> None:
    runtime = get_runtime_config()
    parser = argparse.ArgumentParser(description="Run rule-based agent")
    parser.add_argument("--task", choices=["task1", "task2", "task3"], default=runtime.default_task)
    args = parser.parse_args()

    env = TriageEnvironment(task=args.task)
    metrics = run_single_episode(env, RuleBasedAgent())
    print("Rule-based agent episode metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
