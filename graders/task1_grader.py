#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for p in (str(REPO_ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)

from triage_env.graders.common import grade_task, print_grader_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade task1 for MedicalTriage")
    parser.add_argument("--episodes", type=int, default=3)
    args, _ = parser.parse_known_args()

    print_grader_result(grade_task("task1", episodes=args.episodes))


class Task1Grader:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return grade_task("task1", episodes=3)

    def grade(self, *args, **kwargs):
        return grade_task("task1", episodes=3)

    def evaluate(self, *args, **kwargs):
        return grade_task("task1", episodes=3)

    def run(self, *args, **kwargs):
        return grade_task("task1", episodes=3)


if __name__ == "__main__":
    main()
