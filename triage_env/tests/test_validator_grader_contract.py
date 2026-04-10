from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from graders import task1, task2, task3
import graders.task1_grader as root_task1_module
import graders.task2_grader as root_task2_module
import graders.task3_grader as root_task3_module
import triage_env.graders.task1 as pkg_task1_module
import triage_env.graders.task2 as pkg_task2_module
import triage_env.graders.task3 as pkg_task3_module


def _is_strict_open_unit_interval(value: float) -> bool:
    return 0.0 < float(value) < 1.0


def _assert_epsilon_clamp(value: float, expected: float) -> None:
    assert float(value) == pytest.approx(expected, abs=1e-12)


def test_openenv_manifest_declares_three_enabled_tasks_with_graders() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "openenv.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    tasks = manifest.get("tasks", [])
    enabled = [t for t in tasks if t.get("enabled", False)]
    assert len(enabled) >= 3

    for task in enabled:
        grader_path = task.get("grader")
        assert isinstance(grader_path, str) and grader_path.endswith(".py")
        resolved = repo_root / grader_path
        assert resolved.exists(), f"Missing grader file: {grader_path}"


def test_wrapper_grade_stays_strictly_between_zero_and_one(monkeypatch) -> None:
    monkeypatch.setattr(task1, "common_grade_task", lambda *_args, **_kwargs: {"score": 0.0})
    monkeypatch.setattr(task2, "common_grade_task", lambda *_args, **_kwargs: {"score": 1.0})
    monkeypatch.setattr(task3, "common_grade_task", lambda *_args, **_kwargs: {})

    s1 = task1.grade(episodes=1)
    s2 = task2.grade(episodes=1)
    s3 = task3.grade(episodes=1)

    assert _is_strict_open_unit_interval(s1)
    assert _is_strict_open_unit_interval(s2)
    assert _is_strict_open_unit_interval(s3)


@pytest.mark.parametrize(
    ("module", "raw_score", "expected"),
    [
        (root_task1_module, 0.0, 1e-6),
        (root_task2_module, 1.0, 1.0 - 1e-6),
        (root_task3_module, 0.0, 1e-6),
        (pkg_task1_module, 1.0, 1.0 - 1e-6),
        (pkg_task2_module, 0.0, 1e-6),
        (pkg_task3_module, 1.0, 1.0 - 1e-6),
    ],
)
def test_grade_task_clamps_exact_zero_and_one(monkeypatch, module, raw_score, expected) -> None:
    monkeypatch.setattr(module, "common_grade_task", lambda *_args, **_kwargs: {"score": raw_score})

    result = module.grade_task(episodes=1)

    assert result["episodes"] == 1
    _assert_epsilon_clamp(result["score"], expected)
    _assert_epsilon_clamp(result["reward"], expected)
    assert _is_strict_open_unit_interval(result["score"])
