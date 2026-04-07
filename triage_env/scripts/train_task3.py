import argparse
from pathlib import Path

from triage_env.tasks import TASK_TRAINING_DEFAULTS
from triage_env.training.train_rl import train_rl_agent


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    defaults = TASK_TRAINING_DEFAULTS["task3"]
    parser = argparse.ArgumentParser(description="Train Task 3 RL agent with Task 2 warm start")
    parser.add_argument("--episodes", type=int, default=int(defaults["episodes"]))
    parser.add_argument("--warm-start-model-path", default=str(PACKAGE_ROOT / "training" / "triage_rl_qtable_task2.json"))
    parser.add_argument("--save-path", default=str(PACKAGE_ROOT / "training" / "triage_rl_qtable_task3.json"))
    parser.add_argument("--epsilon-start", type=float, default=float(defaults["epsilon_start"]))
    parser.add_argument("--epsilon-end", type=float, default=float(defaults["epsilon_end"]))
    parser.add_argument("--epsilon-decay", type=float, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    args = parser.parse_args()

    train_rl_agent(
        episodes=max(1, args.episodes),
        task="task3",
        save_path=args.save_path,
        warm_start_model_path=args.warm_start_model_path,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        print_every=args.print_every or max(50, max(1, args.episodes // 10)),
    )


if __name__ == "__main__":
    main()