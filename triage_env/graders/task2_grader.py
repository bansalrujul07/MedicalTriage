from __future__ import annotations

import argparse

from triage_env.graders.common import grade_task, print_grader_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade task2 for MedicalTriage")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    print_grader_result(grade_task("task2", episodes=args.episodes))


if __name__ == "__main__":
    main()
