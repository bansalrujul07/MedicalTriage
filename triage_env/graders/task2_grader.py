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
    from graders.common import grade_task, print_grader_result
except ModuleNotFoundError:
    from common import grade_task, print_grader_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade task2 for MedicalTriage")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    print_grader_result(grade_task("task2", episodes=args.episodes))


if __name__ == "__main__":
    main()
