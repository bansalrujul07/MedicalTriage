from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import triage_env.graders.common as graders_common
import triage_env.graders.task1_grader as root_task1_module
import triage_env.graders.task2_grader as root_task2_module
import triage_env.graders.task3_grader as root_task3_module


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

    top_level_graders = manifest.get("graders", {})
    assert isinstance(top_level_graders, dict)
    assert set(top_level_graders) >= {"task1", "task2", "task3"}

    for task in enabled:
        grader_path = task.get("grader")
        assert isinstance(grader_path, str) and grader_path.endswith(".py")
        resolved = repo_root / grader_path
        assert resolved.exists(), f"Missing grader file: {grader_path}"

        grader_entries = task.get("graders", [])
        assert isinstance(grader_entries, list) and grader_entries, f"Missing graders list for {task.get('id')}"
        primary = grader_entries[0]
        assert primary.get("type") == "python"
        assert isinstance(primary.get("path"), str) and (repo_root / primary["path"]).exists()
        assert isinstance(primary.get("command"), str) and primary["command"].startswith("python ")


@pytest.mark.parametrize(
    ("components", "expected"),
    [
        (
            {
                "rollout_achievement": 0.0,
                "safety_errors": 0.0,
                "efficiency": 0.0,
                "task_specific": 0.0,
            },
            1e-6,
        ),
        (
            {
                "rollout_achievement": 1.0,
                "safety_errors": 1.0,
                "efficiency": 1.0,
                "task_specific": 1.0,
            },
            1.0 - 1e-6,
        ),
    ],
)
def test_compute_final_score_clamps_exact_zero_and_one(components, expected) -> None:
    score = graders_common._compute_final_score(components)

    _assert_epsilon_clamp(score, expected)
    assert _is_strict_open_unit_interval(score)


@pytest.mark.parametrize(
    "value",
    [0.0, 1.0],
)
def test_clip_open_interval_bounds(value: float) -> None:
    clipped = graders_common._clip_open_01(value)

    if value == 0.0:
        _assert_epsilon_clamp(clipped, 1e-6)
    else:
        _assert_epsilon_clamp(clipped, 1.0 - 1e-6)
    assert _is_strict_open_unit_interval(clipped)
