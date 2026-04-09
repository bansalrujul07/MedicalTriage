#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for p in (str(HERE), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from graders.common import grade_task as common_grade_task, print_grader_result
except ModuleNotFoundError:
    from common import grade_task as common_grade_task, print_grader_result


def grade_task(episodes: int = 20):
    return common_grade_task("task3", episodes=episodes)


def grade(episodes: int = 20) -> float:
    result = grade_task(episodes=episodes)
    return float(result.get("score", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade task3 for MedicalTriage")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    print_grader_result(grade_task(episodes=args.episodes))


if __name__ == "__main__":
    main()
