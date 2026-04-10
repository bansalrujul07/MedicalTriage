import inspect
import re
from pathlib import Path

from triage_env.graders.common import grade_task as common_grade_task
from triage_env.graders.task1_grader import grade_task as task1_grade_task
from triage_env.graders.task2_grader import grade_task as task2_grade_task
from triage_env.graders.task3_grader import grade_task as task3_grade_task


def _default_episodes(func) -> int:
    return int(inspect.signature(func).parameters["episodes"].default)


def _assert_cli_episodes_default_one(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    pattern = r"add_argument\(\s*[\"']--episodes[\"']\s*,\s*type=int\s*,\s*default=1\s*\)"
    assert re.search(pattern, content), f"Expected --episodes default=1 in {path}"


def _assert_grade_task_default_one(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    pattern = r"def\s+grade_task\(\s*task_name\s*:\s*str\s*,\s*episodes\s*:\s*int\s*=\s*1\s*\)"
    assert re.search(pattern, content), f"Expected grade_task(..., episodes: int = 1) in {path}"


def test_validator_grader_defaults_aligned_to_one_episode():
    expected = _default_episodes(common_grade_task)
    assert expected == 1
    assert _default_episodes(task1_grade_task) == expected
    assert _default_episodes(task2_grade_task) == expected
    assert _default_episodes(task3_grade_task) == expected


def test_package_grader_cli_defaults_aligned_to_one_episode():
    repo_root = Path(__file__).resolve().parents[2]
    _assert_grade_task_default_one(repo_root / "triage_env" / "graders" / "common.py")
    _assert_cli_episodes_default_one(repo_root / "triage_env" / "graders" / "task1_grader.py")
    _assert_cli_episodes_default_one(repo_root / "triage_env" / "graders" / "task2_grader.py")
    _assert_cli_episodes_default_one(repo_root / "triage_env" / "graders" / "task3_grader.py")
