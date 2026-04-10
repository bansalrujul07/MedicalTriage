from __future__ import annotations

from pathlib import Path

import yaml

from graders import task1, task2, task3


def _is_strict_open_unit_interval(value: float) -> bool:
    return 0.0 < float(value) < 1.0


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
