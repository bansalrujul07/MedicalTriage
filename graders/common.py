from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

# Make triage_env imports robust regardless of caller working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _fallback_grade(task_name: str, episodes: int, reason: str) -> dict[str, Any]:
    score = 0.0
    return {
        "grader_version": "fallback-v2.0-error",
        "status": "error",
        "task": task_name,
        "task_id": task_name,
        "episodes": episodes,
        "score": score,
        "reward": score,
        "score_range": [0.0, 1.0],
        "components": {
            "rollout_achievement": score,
            "safety_errors": score,
            "efficiency": score,
            "task_specific": score,
        },
        "signals": {
            "fallback": 1.0,
            "error": reason,
        },
        "summary": {
            "task": task_name,
            "fallback_reason": reason,
            "success_rate": score,
            "survival_rate": score,
            "critical_survival_rate": score,
            "avg_total_reward": score,
        },
    }


def grade_task(task_name: str, episodes: int = 20) -> dict[str, Any]:
    try:
        from triage_env.graders.common import grade_task as impl_grade_task

        return impl_grade_task(task_name=task_name, episodes=episodes)
    except Exception as exc:  # pragma: no cover - used for strict external validators
        err = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=10),
        }
        return _fallback_grade(task_name=task_name, episodes=episodes, reason=json.dumps(err, ensure_ascii=True))


def print_grader_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))
