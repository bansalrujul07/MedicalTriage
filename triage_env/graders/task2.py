#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
for p in (str(HERE), str(PKG_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from graders.common import grade_task as common_grade_task, print_grader_result
except ModuleNotFoundError:
    from common import grade_task as common_grade_task, print_grader_result


def _clip_score_strict(score: float) -> float:
    epsilon = 1e-6
    clipped = max(0.0, min(1.0, float(score)))
    return epsilon + clipped * (1.0 - 2.0 * epsilon)


def _normalize_result(result: dict, episodes: int, fallback_reason: str | None = None) -> dict:
    score = _clip_score_strict(float(result.get("score", 0.5)))
    result["episodes"] = int(episodes)
    result["score"] = score
    result["reward"] = score
    result.setdefault("score_range", [0.0, 1.0])
    if fallback_reason is not None:
        result.setdefault("signals", {})
        result.setdefault("summary", {})
        result["signals"].setdefault("wrapper_fallback", 1.0)
        result["signals"]["reason"] = fallback_reason
        result["summary"]["fallback_reason"] = fallback_reason
    return result


def grade_task(episodes: int = 1):
    result = common_grade_task("task2", episodes=episodes)
    if not isinstance(result, dict):
        return _normalize_result(_safe_result("non-dict-result", episodes), episodes, "non-dict-result")
    return _normalize_result(result, episodes)


def _safe_result(reason: str, episodes: int) -> dict:
    safe = _clip_score_strict(0.5)
    return {
        "grader_version": "wrapper-fallback-v1",
        "task": "task2",
        "task_id": "task2",
        "episodes": int(episodes),
        "score": safe,
        "reward": safe,
        "score_range": [0.0, 1.0],
        "components": {
            "rollout_achievement": safe,
            "safety_errors": safe,
            "efficiency": safe,
            "task_specific": safe,
        },
        "signals": {"wrapper_fallback": 1.0, "reason": reason},
        "summary": {"task": "task2", "fallback_reason": reason},
    }


def grade(episodes: int = 1) -> float:
    result = grade_task(episodes=episodes)
    return _clip_score_strict(float(result.get("score", 0.5)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validator-friendly grader wrapper for task2")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    print_grader_result(grade_task(episodes=args.episodes))


if __name__ == "__main__":
    main()
