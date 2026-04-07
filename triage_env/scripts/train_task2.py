from pathlib import Path

from triage_env.config import get_runtime_config
from triage_env.training.train_rl import train_rl_agent


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    runtime = get_runtime_config()
    save_path = PACKAGE_ROOT / "training" / "triage_rl_qtable_task2.json"
    task1_warm_start = PACKAGE_ROOT / "training" / "triage_rl_qtable_task1_after_alignment.json"
    train_rl_agent(
        episodes=runtime.train_episodes,
        task="task2",
        save_path=str(save_path),
        warm_start_model_path=str(task1_warm_start) if task1_warm_start.exists() else None,
    )


if __name__ == "__main__":
    main()
