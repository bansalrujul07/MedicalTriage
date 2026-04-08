from __future__ import annotations

import json
from typing import Any

from triage_env.agents.rule_based_agent import RuleBasedAgent
from triage_env.evaluation.evaluator import evaluate_agent
from triage_env.server.triage_env_environment import TriageEnvironment
from triage_env.tasks import TASK_CONFIGS


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

    return {
        "task": task_name,
        "episodes": episodes,
        "score": summary["success_rate"],
        "summary": summary,
    }


def print_grader_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))
