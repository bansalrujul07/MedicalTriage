import os
from pathlib import Path

from triage_env.agents.q_learning_agents import QLearningAgent
from triage_env.config import get_runtime_config
from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.tasks import TASK_CONFIGS
from triage_env.training.state_encoder import encode_observation


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def train_q_learning_agent(
    episodes: int = 500,
    task: str = "task2",
    save_path: str | None = None,
):
    resolved_save_path = Path(save_path) if save_path else (PACKAGE_ROOT / "training" / "q_agent.pkl")
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.2)
    task_config = TASK_CONFIGS[task]

    for episode in range(episodes):
        env = TriageEnvironment(task=task, max_steps=task_config.max_steps)
        observation = env.reset(task=task)
        done = False

        while not done:
            state = encode_observation(observation)
            action_obj = agent.act(observation)
            action_tuple = (action_obj.action_type, action_obj.patient_id)

            next_observation = env.step(action_obj)
            reward = next_observation.reward
            done = next_observation.done

            next_state = encode_observation(next_observation)
            next_valid_actions = agent.get_valid_actions(next_observation)

            agent.update(
                state=state,
                action=action_tuple,
                reward=reward,
                next_state=next_state,
                done=done,
                next_valid_actions=next_valid_actions,
            )

            observation = next_observation

        if (episode + 1) % 50 == 0:
            print(f"Completed episode {episode + 1}/{episodes}")

    os.makedirs(resolved_save_path.parent, exist_ok=True)
    agent.save(
        str(resolved_save_path),
        metadata={
            "task": task,
            "episodes": episodes,
            "training_version": 2,
        },
    )
    print(f"Saved Q-learning agent to {resolved_save_path}")


def main() -> None:
    runtime = get_runtime_config()
    resolved_save_path = PACKAGE_ROOT / "training" / "q_agent.pkl"
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.2)

    for task in ("task1", "task2", "task3"):
        task_config = TASK_CONFIGS[task]
        for episode in range(runtime.train_episodes):
            env = TriageEnvironment(task=task, max_steps=task_config.max_steps)
            observation = env.reset(task=task)
            done = False

            while not done:
                state = encode_observation(observation)
                action_obj = agent.act(observation)
                action_tuple = (action_obj.action_type, action_obj.patient_id)

                next_observation = env.step(action_obj)
                reward = next_observation.reward
                done = next_observation.done

                next_state = encode_observation(next_observation)
                next_valid_actions = agent.get_valid_actions(next_observation)

                agent.update(
                    state=state,
                    action=action_tuple,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    next_valid_actions=next_valid_actions,
                )

                observation = next_observation

            if (episode + 1) % 50 == 0:
                print(f"Task {task}: completed episode {episode + 1}/{runtime.train_episodes}")

    os.makedirs(resolved_save_path.parent, exist_ok=True)
    agent.epsilon = 0.0
    agent.save(
        str(resolved_save_path),
        metadata={
            "task": "task3",
            "episodes": runtime.train_episodes,
            "training_version": 2,
        },
    )
    print(f"Saved Q-learning agent to {resolved_save_path}")


if __name__ == "__main__":
    main()
