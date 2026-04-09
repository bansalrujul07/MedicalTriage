from __future__ import annotations

import json
import os
from pathlib import Path

from triage_env.agents.rl_agents import RLAgent
from triage_env.config import get_runtime_config
from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.tasks import TASK_CONFIGS


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _build_training_metadata(
    *,
    task: str,
    target_episodes: int,
    warm_start_model_path: str | None,
    last_completed_episode: int,
    agent: RLAgent,
    resumed_from_checkpoint: bool,
) -> dict:
    return {
        "task": task,
        "target_episodes": target_episodes,
        "last_completed_episode": last_completed_episode,
        "training_version": 3,
        "warm_start_model_path": warm_start_model_path,
        "warm_start_task": task if not warm_start_model_path else None,
        "resume_ready": True,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "exploration": {
            "epsilon": agent.epsilon,
            "epsilon_min": agent.epsilon_min,
            "epsilon_decay": agent.epsilon_decay,
        },
    }


def _atomic_save_agent(agent: RLAgent, save_path: Path, metadata: dict) -> None:
    os.makedirs(save_path.parent, exist_ok=True)
    temp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    agent.save(str(temp_path), metadata=metadata)
    os.replace(temp_path, save_path)


def _load_resume_episode(warm_start_model_path: str, task: str) -> int:
    try:
        with open(warm_start_model_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return 0

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        return 0

    if metadata.get("task") != task:
        return 0

    try:
        last_completed_episode = int(metadata.get("last_completed_episode", 0))
    except (TypeError, ValueError):
        return 0

    return max(0, last_completed_episode)


def train_rl_agent(
    episodes: int = 500,
    task: str = "task2",
    save_path: str | None = None,
    warm_start_model_path: str | None = None,
    epsilon_start: float = 0.2,
    epsilon_end: float = 0.05,
    epsilon_decay: float | None = None,
    print_every: int | None = None,
    checkpoint_every: int | None = None,
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

    start_episode = 0
    resumed_from_checkpoint = False
    if warm_start_model_path and os.path.exists(warm_start_model_path):
        agent.load(warm_start_model_path)
        start_episode = _load_resume_episode(warm_start_model_path, task)

        if start_episode > 0:
            resumed_from_checkpoint = True
            print(f"Resuming RL training from episode {start_episode}/{episodes} using {warm_start_model_path}")
        else:
            # Warm-start Q-table but reset exploration schedule for a fresh run.
            agent.epsilon = epsilon_start
            agent.epsilon_min = epsilon_end
            agent.epsilon_decay = decay

    effective_print_every = print_every or max(50, max(1, episodes // 10))
    effective_checkpoint_every = checkpoint_every or max(1, int(os.getenv("TRIAGE_RL_CHECKPOINT_EVERY", "50")))

    for episode in range(start_episode, episodes):
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

        if (episode + 1) % effective_checkpoint_every == 0:
            checkpoint_metadata = _build_training_metadata(
                task=task,
                target_episodes=episodes,
                warm_start_model_path=warm_start_model_path,
                last_completed_episode=episode + 1,
                agent=agent,
                resumed_from_checkpoint=resumed_from_checkpoint,
            )
            _atomic_save_agent(agent, resolved_save_path, checkpoint_metadata)
            print(f"Saved periodic RL checkpoint at episode {episode + 1}/{episodes} to {resolved_save_path}")

    metadata = _build_training_metadata(
        task=task,
        target_episodes=episodes,
        warm_start_model_path=warm_start_model_path,
        last_completed_episode=episodes,
        agent=agent,
        resumed_from_checkpoint=resumed_from_checkpoint,
    )
    _atomic_save_agent(agent, resolved_save_path, metadata)
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
