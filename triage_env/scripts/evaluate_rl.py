import argparse
import os
from pathlib import Path

from triage_env.agents.rl_agents import RLAgent
from triage_env.evaluation.evaluator import evaluate_agent
from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.tasks import TASK_CONFIGS


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL agent with optional exploration")
    parser.add_argument("--task", choices=["task1", "task2", "task3"], default="task1")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--model-path", default="triage_env/training/triage_rl_qtable.json")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"RL model not found at {model_path}")
        return

    agent = RLAgent()
    agent.load(str(model_path))
    agent.epsilon = max(0.0, min(1.0, args.epsilon))

    task_config = TASK_CONFIGS[args.task]

    summary, _ = evaluate_agent(
        env_class=TriageEnvironment,
        agent=agent,
        task=args.task,
        num_episodes=args.episodes,
        max_steps=task_config.max_steps,
    )
    summary["epsilon"] = agent.epsilon
    print(summary)


if __name__ == "__main__":
    main()