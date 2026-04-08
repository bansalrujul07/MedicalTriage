#!/usr/bin/env python3
from __future__ import annotations

import argparse

try:
    from graders.common import grade_task, print_grader_result
except ModuleNotFoundError:
    from common import grade_task, print_grader_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade task3 for MedicalTriage")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    print_grader_result(grade_task("task3", episodes=args.episodes))


if __name__ == "__main__":
    main()
