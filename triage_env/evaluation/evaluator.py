import random
from collections import Counter
from statistics import mean

from triage_env.evaluation.metrics import compute_episode_metrics
from triage_env.tasks import resolve_task_name


def run_single_episode(env, agent):
    observation = env.reset()
    agent.reset()

    done = False
    total_reward = 0.0
    action_counts = {"wait": 0, "treat": 0, "allocate_ventilator": 0}

    while not done:
        action = agent.act(observation)
        if action.action_type in action_counts:
            action_counts[action.action_type] += 1
        observation = env.step(action)
        total_reward += observation.reward
        done = observation.done

    return compute_episode_metrics(observation, total_reward, action_counts)


def evaluate_agent(env_class, agent, task="task2", num_episodes=20, seed=42, verbose=False, **env_kwargs):
    """
    Creates a fresh environment per episode for clean evaluation.
    """
    episode_results = []

    if "task" not in env_kwargs:
        env_kwargs["task"] = task

    random.seed(seed)
    for episode_idx in range(num_episodes):
        random.seed(seed + episode_idx)
        env = env_class(**env_kwargs)
        result = run_single_episode(env, agent)
        episode_results.append(result)

        if verbose:
            print(
                f"task={task} reward={result.total_reward:.2f} "
                f"survival={result.survival_rate:.2f} "
                f"critical_survival={result.critical_survival_rate:.2f}"
            )

    summary = {
        "task": task,
        "agent_name": agent.name,
        "episodes": num_episodes,
        "avg_total_reward": mean(r.total_reward for r in episode_results),
        "avg_episode_length": mean(r.steps for r in episode_results),
        "avg_survivors": mean(r.survivors for r in episode_results),
        "avg_deaths": mean(r.deaths for r in episode_results),
        "survival_rate": mean(r.survival_rate for r in episode_results),
        "critical_survival_rate": mean(r.critical_survival_rate for r in episode_results),
        "avg_health_alive": mean(r.avg_health_alive for r in episode_results),
        "stabilization_rate": mean(r.stabilization_rate for r in episode_results),
        "invalid_action_count": mean(r.invalid_action_count for r in episode_results),
        "deaths_by_severity": {
            "critical": mean(r.deaths_by_severity["critical"] for r in episode_results),
            "severe": mean(r.deaths_by_severity["severe"] for r in episode_results),
            "moderate": mean(r.deaths_by_severity["moderate"] for r in episode_results),
            "mild": mean(r.deaths_by_severity["mild"] for r in episode_results),
        },
        "resource_utilization": {
            "medics": mean(r.resource_utilization["medics"] for r in episode_results),
            "ventilators": mean(r.resource_utilization["ventilators"] for r in episode_results),
        },
        "success_rate": mean(1.0 if r.success else 0.0 for r in episode_results),
        "action_distribution": {
            "wait": mean(r.action_distribution.get("wait", 0.0) for r in episode_results),
            "treat": mean(r.action_distribution.get("treat", 0.0) for r in episode_results),
            "allocate_ventilator": mean(
                r.action_distribution.get("allocate_ventilator", 0.0)
                for r in episode_results
            ),
        },
        # Backward-compatible aliases.
        "avg_steps": mean(r.steps for r in episode_results),
        "avg_stabilization_rate": mean(r.stabilization_rate for r in episode_results),
        "avg_action_distribution": {
            "wait": mean(r.action_distribution.get("wait", 0.0) for r in episode_results),
            "treat": mean(r.action_distribution.get("treat", 0.0) for r in episode_results),
            "allocate_ventilator": mean(
                r.action_distribution.get("allocate_ventilator", 0.0)
                for r in episode_results
            ),
        },
        "avg_unseen_states": mean((getattr(r, "unseen_states", 0) or 0) for r in episode_results),
    }

    failure_reason_counts = Counter(
        r.terminal_failure_reason for r in episode_results if r.terminal_failure_reason
    )
    summary["failure_reason_counts"] = dict(failure_reason_counts)
    summary["terminal_diagnostics"] = {
        "survival_rate": mean(r.terminal_diagnostics["survival_rate"] for r in episode_results),
        "critical_survival_rate": mean(r.terminal_diagnostics["critical_survival_rate"] for r in episode_results),
        "avg_health_alive": mean(r.terminal_diagnostics["avg_health_alive"] for r in episode_results),
        "alive_count": mean(r.terminal_diagnostics["alive_count"] for r in episode_results),
        "stabilization_rate": mean(r.terminal_diagnostics["stabilization_rate"] for r in episode_results),
        "ventilator_utilization": mean(r.terminal_diagnostics["ventilator_utilization"] for r in episode_results),
        "ventilator_occupancy": mean(r.terminal_diagnostics["ventilator_occupancy"] for r in episode_results),
    }

    checkpoint_metadata = getattr(agent, "checkpoint_metadata", None)
    if checkpoint_metadata is not None:
        summary["checkpoint_metadata"] = checkpoint_metadata
    checkpoint_warning = getattr(agent, "checkpoint_warning", None)
    if checkpoint_warning is not None:
        summary["checkpoint_warning"] = checkpoint_warning
    checkpoint_status = getattr(agent, "checkpoint_status", None)
    if checkpoint_status is not None:
        summary["checkpoint_status"] = checkpoint_status
    checkpoint_path = getattr(agent, "checkpoint_path", None)
    if checkpoint_path is not None:
        summary["checkpoint_path"] = checkpoint_path

    return summary, episode_results


def evaluate(env, agent, episodes=20, task=None, **_kwargs):
    """Backward-compatible wrapper around evaluate_agent.

    Accepts an environment instance and returns only the summary dict.
    Extra kwargs are ignored for compatibility with older callsites.
    """
    resolved_task = resolve_task_name(
        task=task if task in {"task1", "task2", "task3"} else None,
        difficulty=task if task not in {None, "task1", "task2", "task3"} else None,
    )
    summary, _ = evaluate_agent(
        env.__class__,
        agent,
        task=resolved_task or getattr(env, "task_name", "task2"),
        num_episodes=episodes,
        max_steps=env.max_steps,
    )
    return summary