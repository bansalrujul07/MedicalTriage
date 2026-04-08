"""Rollout utilities for collecting experience from environment."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def run_episode(env: Any, agent: Any, training: bool = False, max_steps: int | None = None) -> Dict[str, Any]:
    """Backward-compatible single-episode runner used by tests/scripts.

    Supports both tuple-style Gym reset/step APIs and the triage observation-object API.
    """
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    if hasattr(agent, "reset"):
        agent.reset()

    total_reward = 0.0
    steps = 0
    step_limit = max_steps if max_steps is not None else getattr(env, "max_steps", 1000)

    while steps < step_limit:
        action = agent.act(obs) if hasattr(agent, "act") else env.action_space.sample()
        step_result = env.step(action)

        if isinstance(step_result, tuple):
            # Gym-style: obs, reward, done, truncated, info
            next_obs, reward, done, truncated, _info = step_result
            obs = next_obs
            is_done = bool(done or truncated)
        else:
            # Triage observation object style
            obs = step_result
            reward = float(getattr(step_result, "reward", 0.0))
            is_done = bool(getattr(step_result, "done", False))

        total_reward += float(reward)
        steps += 1

        if is_done:
            break

    return {
        "total_reward": float(total_reward),
        "steps": int(steps),
        "training": bool(training),
    }


def collect_rollout(
    agent: Any,
    env: Any,
    num_episodes: int = 1,
    max_steps: int = 1000
) -> Dict[str, Any]:
    """
    Collect rollout data from environment using an agent.
    
    Args:
        agent: Agent to collect rollouts from
        env: Environment to run rollouts in
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary with collected trajectories and statistics
    """
    trajectories = []
    total_reward = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": []
        }
        episode_reward = 0
        
        for step in range(max_steps):
            episode_trajectory["observations"].append(obs)
            
            # Get action from agent
            action = agent.act(obs) if hasattr(agent, 'act') else env.action_space.sample()
            episode_trajectory["actions"].append(action)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            episode_trajectory["rewards"].append(reward)
            episode_trajectory["dones"].append(done or truncated)
            episode_reward += reward
            
            if done or truncated:
                break
        
        trajectories.append(episode_trajectory)
        total_reward += episode_reward
        total_steps += step + 1
    
    return {
        "trajectories": trajectories,
        "mean_reward": total_reward / num_episodes,
        "mean_steps": total_steps / num_episodes,
        "num_episodes": num_episodes
    }
