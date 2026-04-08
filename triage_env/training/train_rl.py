from __future__ import annotations

import os
from pathlib import Path

from triage_env.agents.rl_agents import RLAgent
from triage_env.config import get_runtime_config
from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.tasks import TASK_CONFIGS


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def train_rl_agent(
    episodes: int = 500,
    task: str = "task2",
    save_path: str | None = None,
    warm_start_model_path: str | None = None,
    epsilon_start: float = 0.2,
    epsilon_end: float = 0.05,
    epsilon_decay: float | None = None,
    print_every: int | None = None,
) -> str:
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unsupported task: {task}")

    resolved_save_path = Path(save_path) if save_path else (PACKAGE_ROOT / "training" / "triage_rl_qtable.json")
    task_config = TASK_CONFIGS[task]

    decay = epsilon_decay
    if decay is None and episodes > 1 and epsilon_start > epsilon_end:
        decay = (epsilon_end / epsilon_start) ** (1 / max(1, episodes - 1))
    if decay is None:
        decay = 0.995

    agent = RLAgent(
        epsilon=epsilon_start,
        epsilon_min=epsilon_end,
        epsilon_decay=decay,
    )

    if warm_start_model_path and os.path.exists(warm_start_model_path):
        agent.load(warm_start_model_path)
        agent.epsilon = epsilon_start
        agent.epsilon_min = epsilon_end
        agent.epsilon_decay = decay

    effective_print_every = print_every or max(50, max(1, episodes // 10))

    for episode in range(episodes):
        env = TriageEnvironment(task=task, max_steps=task_config.max_steps)
        observation = env.reset(task=task)
        done = False

        while not done:
            action = agent.act(observation)
            next_observation = env.step(action)
            agent.update(observation, action, next_observation.reward, next_observation)
            observation = next_observation
            done = next_observation.done

        agent.decay_exploration()

        if (episode + 1) % effective_print_every == 0:
            print(f"Task {task}: completed episode {episode + 1}/{episodes}")

    os.makedirs(resolved_save_path.parent, exist_ok=True)
    metadata = {
        "task": task,
        "episodes": episodes,
        "training_version": 2,
        "warm_start_model_path": warm_start_model_path,
        "warm_start_task": task if not warm_start_model_path else None,
    }
    agent.save(str(resolved_save_path), metadata=metadata)
    print(f"Saved RL agent to {resolved_save_path}")
    return str(resolved_save_path)


def main() -> None:
    runtime = get_runtime_config()
    train_rl_agent(
        episodes=runtime.train_episodes,
        task="task3",
        save_path=str(PACKAGE_ROOT / "training" / "triage_rl_qtable.json"),
    )


if __name__ == "__main__":
    main()
