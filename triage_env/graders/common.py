from __future__ import annotations

import json
import math
from typing import Any

from triage_env.agents.rule_based_agent import RuleBasedAgent
from triage_env.evaluation.evaluator import evaluate_agent
from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.tasks import TASK_CONFIGS


GRADER_VERSION = "v2"


def _clip_01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(*values: float) -> float:
    filtered = [float(v) for v in values]
    if not filtered:
        return 0.0
    return sum(filtered) / len(filtered)


def _safe_get(summary: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = summary.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalized_reward(avg_total_reward: float) -> float:
    # Smoothly maps arbitrary reward magnitudes to [0, 1].
    return _clip_01(0.5 + 0.5 * math.tanh(avg_total_reward / 200.0))


def _compute_components(task_name: str, summary: dict[str, Any]) -> dict[str, float]:
    task_config = TASK_CONFIGS[task_name]
    num_patients = float(task_config.num_patients)
    max_steps = float(task_config.max_steps)

    survival_rate = _clip_01(_safe_get(summary, "survival_rate"))
    critical_survival_rate = _clip_01(_safe_get(summary, "critical_survival_rate"))
    success_rate = _clip_01(_safe_get(summary, "success_rate"))
    stabilization_rate = _clip_01(_safe_get(summary, "stabilization_rate"))
    avg_health_alive = _clip_01(_safe_get(summary, "avg_health_alive") / 100.0)
    avg_total_reward = _safe_get(summary, "avg_total_reward")
    avg_episode_length = _safe_get(summary, "avg_episode_length")
    avg_deaths = _safe_get(summary, "avg_deaths")
    invalid_action_count = _safe_get(summary, "invalid_action_count")

    vent_util = 0.0
    resources = summary.get("resource_utilization", {})
    if isinstance(resources, dict):
        vent_util = _clip_01(float(resources.get("ventilators", 0.0)))

    reward_norm = _normalized_reward(avg_total_reward)
    death_penalty = _clip_01(avg_deaths / max(1.0, num_patients))
    invalid_penalty = _clip_01(invalid_action_count / 2.0)
    step_efficiency = _clip_01(1.0 - (avg_episode_length / max(1.0, max_steps)))

    rollout_achievement = _clip_01(
        _mean(survival_rate, critical_survival_rate, success_rate)
    )
    safety_errors = _clip_01(_mean(1.0 - death_penalty, 1.0 - invalid_penalty))
    efficiency = _clip_01(_mean(stabilization_rate, step_efficiency, reward_norm))

    if task_name == "task1":
        task_specific = _clip_01(_mean(avg_health_alive, success_rate, reward_norm))
    elif task_name == "task2":
        vent_balance = _clip_01(1.0 - abs(vent_util - 0.5) / 0.5)
        task_specific = _clip_01(_mean(critical_survival_rate, vent_balance, reward_norm))
    else:
        task_specific = _clip_01(
            _mean(critical_survival_rate, survival_rate, avg_health_alive)
        )

    return {
        "rollout_achievement": rollout_achievement,
        "safety_errors": safety_errors,
        "efficiency": efficiency,
        "task_specific": task_specific,
        "reward_norm": reward_norm,
        "survival_rate": survival_rate,
        "critical_survival_rate": critical_survival_rate,
        "success_rate": success_rate,
    }


def _compute_final_score(components: dict[str, float]) -> float:
    # Universal weighted formula: each component is normalized to [0, 1].
    score = (
        components["rollout_achievement"] * 0.40
        + components["safety_errors"] * 0.25
        + components["efficiency"] * 0.20
        + components["task_specific"] * 0.15
    )
    return _clip_01(score)


def grade_task(task_name: str, episodes: int = 3) -> dict[str, Any]:
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unsupported task: {task_name}")

    task_config = TASK_CONFIGS[task_name]
    summary, _ = evaluate_agent(
        env_class=TriageEnvironment,
        agent=RuleBasedAgent(),
        task=task_name,
        num_episodes=episodes,
        max_steps=task_config.max_steps,
    )

    components = _compute_components(task_name, summary)
    final_score = _compute_final_score(components)

    return {
        "grader_version": GRADER_VERSION,
        "task": task_name,
        "task_id": task_name,
        "episodes": episodes,
        "score": final_score,
        "reward": final_score,
        "score_range": [0.0, 1.0],
        "components": {
            "rollout_achievement": components["rollout_achievement"],
            "safety_errors": components["safety_errors"],
            "efficiency": components["efficiency"],
            "task_specific": components["task_specific"],
        },
        "signals": {
            "survival_rate": components["survival_rate"],
            "critical_survival_rate": components["critical_survival_rate"],
            "success_rate": components["success_rate"],
            "reward_norm": components["reward_norm"],
        },
        "summary": summary,
    }


def print_grader_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))
